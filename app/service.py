"""Pipeline service for OS Studio.

Mirrors the ComfyUI workflow `yue2_song_to_lrc_qwen3_my_voice_native_yue2`
node for node, without the server:

    idea -> Qwen3-VL lyrics -> SongLyricsClean -> YuE2 (3B + Vae)
         -> SongLyricsToLRC -> [split -> Seed-VC clone -> remix] -> MP3 + LRC

The one thing ComfyUI did for free and a standalone has to do itself is VRAM
hand-off. Qwen (8 GB), YuE2 (about 10.5 GB peak) and Seed-VC do not fit on a
16 GB card together, so each stage evicts the others before it loads — exactly
what ComfyUI's model manager does between nodes. Leave two of them resident
and the card spills into shared system memory: a song then takes 10 minutes
instead of 2.
"""
from __future__ import annotations

import gc
import os
import random
import re
import sys
import threading
import time
import traceback

import bootstrap

bootstrap.bootstrap()
APP_ROOT = bootstrap.APP_ROOT
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

import library  # noqa: E402  (needs APP_ROOT on sys.path)

# The lyrics writer's instructions are built per song from the structure the
# user picks in step 2: library.lyrics_prompt(). (The workflow's fixed prompt
# asked for "2x [verse]", which the writer copied literally and the song model
# then sang as a line.)

# Workflow values (TextGenerate node).
LYRICS_SAMPLING = dict(max_new_tokens=685, do_sample=True, temperature=0.7,
                       top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05)

# Workflow values (YuE2Loader "16 GB preset" + YuE2Sampler).
YUE2_LOAD = dict(model="YuE2-3B", vae="YuE2-Vae", device="cuda",
                 memory_budget_gib=16, offload_ar=True, offline=True,
                 attention_backend="auto", quantization="none")
YUE2_SAMPLE = dict(cfg_scale=-1, ode_steps=32, abc_max_tokens=4096,
                   semantic_max_tokens=9000, abc_temperature=0.7,
                   semantic_temperature=1.0, save_flac=False, save_abc=False,
                   vae_decode="tiled", vae_tile_frames=0, abc=None,
                   abort_after_plan=False, sampling_settings=None,
                   save_artifacts=False)

# Workflow values (SongVoiceSplit / SongVoiceClone / SongVoiceMix / Train).
VOICE_CLONE = dict(my_clip_is="talking", diffusion_steps=30, cfg_rate=0.7,
                   pitch_mode="octave to my range", pitch_shift=0,
                   reference_seconds=20, follow_original=True, fp16=True,
                   keep_loaded=False, voice_strength=1.0)
VOICE_TRAIN_STEPS = 300


def voice_clone_settings(voice: str) -> dict:
    """Use the saved reference type without leaking settings between voices."""
    return dict(VOICE_CLONE, my_clip_is=library.voice_clip_mode(voice))

# How long a song may get decides how much VRAM YuE2 reserves up front.
# Measured on an RTX 5080 with scripts_test/vram_probe.py (25 tokens = 1 s):
#   9000 tokens (6 min ceiling, the workflow's value) -> 10.55 GiB peak
#   4500 tokens (3 min ceiling)                       ->  9.64 GiB peak
# Below `needs_gb` of free VRAM the driver spills into shared system memory.
YUE2_MEMORY = {
    "standard": {"max_tokens": 9000, "needs_gb": 11.5, "max_minutes": 6},
    "low": {"max_tokens": 4500, "needs_gb": 10.3, "max_minutes": 3},
}


def resolve_memory(mode: str = "auto") -> str:
    """'auto' picks the low-memory profile on cards under 14 GB."""
    if mode in YUE2_MEMORY:
        return mode
    import torch
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        return "low" if total_gb < 14 else "standard"
    return "standard"

def resolve_song_model(model: str = "auto") -> str:
    """'auto' = the full model on big cards, the small INT8 file on small ones
    (when it is installed)."""
    if model and model != "auto":
        library.get_song_model(model)                 # raises on unknown ids
        return model
    import torch
    total_gb = 99.0
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    return library.pick_song_model(total_gb)


_run_lock = threading.Lock()      # one GPU job at a time
_cancel = threading.Event()
_lyrics_model = None
_lyrics_writer = None             # id of the writer currently on the GPU


class Cancelled(RuntimeError):
    """The user pressed Cancel."""


# ------------------------------------------------------------------ progress
class Reporter:
    """Stage list + overall percentage for one job.

    A plan entry is (key, label, weight[, typical_seconds]). Stages that report
    real progress (tokens, tiles, training steps) drive the bar directly; the
    ones that cannot (model loads) creep toward 90% over their typical time so
    the bar never looks frozen.
    """

    def __init__(self, plan):
        self.steps = [{"key": p[0], "label": p[1], "weight": float(p[2]),
                       "eta": float(p[3]) if len(p) > 3 else 0.0,
                       "state": "pending", "frac": 0.0, "since": 0.0}
                      for p in plan]
        self.current = None
        self.note = ""
        self._bar = None
        self._lock = threading.Lock()

    def _step(self, key):
        return next(s for s in self.steps if s["key"] == key)

    def start(self, key, bar=None):
        with self._lock:
            for s in self.steps:
                if s["key"] == key:
                    break
                if s["state"] != "skipped":
                    s["state"], s["frac"] = "done", 1.0
            step = self._step(key)
            step["state"], step["frac"], step["since"] = "running", 0.0, time.time()
            self.current, self._bar = key, bar
        print(f"[stage] {step['label']}", flush=True)

    def skip(self, key):
        with self._lock:
            self._step(key)["state"] = "skipped"

    def frac(self, value):
        with self._lock:
            if self.current:
                step = self._step(self.current)
                step["frac"] = max(step["frac"], min(1.0, float(value)))

    def on_bar(self, value, total):
        if self._bar is not None:
            self._bar(value, total)
        elif total:
            self.frac(value / total)

    def finish(self):
        with self._lock:
            for s in self.steps:
                if s["state"] != "skipped":
                    s["state"], s["frac"] = "done", 1.0
            self.current = None

    def snapshot(self):
        with self._lock:
            now = time.time()

            def shown(s):
                if s["state"] == "running" and s["eta"]:
                    return max(s["frac"], min(0.9, (now - s["since"]) / s["eta"]))
                return s["frac"]

            live = [s for s in self.steps if s["state"] != "skipped"]
            total = sum(s["weight"] for s in live) or 1.0
            done = sum(s["weight"] * shown(s) for s in live)
            cur = self._step(self.current) if self.current else None
            return {
                "overall": round(done / total, 4),
                "stage": cur["key"] if cur else None,
                "label": cur["label"] if cur else "",
                "stage_frac": round(shown(cur), 4) if cur else 0.0,
                "note": self.note,
                "steps": [{"key": s["key"], "label": s["label"], "state": s["state"]} for s in live],
            }


_reporter = Reporter([])


def _progress_hook(value, total, preview_image=None, node_id=None):
    try:
        _reporter.on_bar(float(value), float(total))
    except Exception:
        pass


bootstrap.set_progress_hook(_progress_hook)


# ------------------------------------------------------------------ cancel
def cancel():
    import comfy.model_management as mm
    _cancel.set()
    mm.interrupt_current_processing(True)


def _begin():
    import comfy.model_management as mm
    _cancel.clear()
    mm.interrupt_current_processing(False)


def _check():
    if _cancel.is_set():
        raise Cancelled("Cancelled")


def _release_after_cancel(exc: BaseException) -> None:
    """Cancel means "stop and give the graphics card back". A stopped run leaves its
    half-finished work (KV caches, CUDA graphs, latents) on the card, held by the
    exception's stack frames; a finished run would have cleaned that up itself.
    Clear those frames, then unload every model so nothing of ours stays on the GPU.
    The next job loads what it needs again."""
    import comfy.model_management as mm
    seen = set()
    while exc is not None and id(exc) not in seen:       # the cancel and what caused it
        seen.add(id(exc))
        traceback.clear_frames(exc.__traceback__)
        exc = exc.__context__
    for release in (_drop_lyrics_model, _drop_yue2, _drop_native, _park_voice_models,
                    mm.unload_all_models, mm.cleanup_models, mm.soft_empty_cache):
        try:
            release()
        except Exception as error:                        # keep releasing the rest
            print(f"[studio] release after cancel: {error}", flush=True)
    _flush()
    print(f"[studio] cancelled — graphics card released ({_free_gb():.1f} GiB free)", flush=True)


# ------------------------------------------------------------------ VRAM
def _free_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 99.0
    return torch.cuda.mem_get_info()[0] / 2**30


def _flush():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _drop_lyrics_model():
    global _lyrics_model, _lyrics_writer
    if _lyrics_model is not None:
        _lyrics_model = _lyrics_writer = None
        _flush()
        print(f"[studio] released the lyrics model ({_free_gb():.1f} GiB free)", flush=True)


def _yue2_loaded() -> bool:
    from yue2_repo.models import yue2 as ym
    return ym._cache.get("pipeline") is not None


def _drop_yue2():
    from yue2_repo.models import yue2 as ym
    if ym._cache.get("pipeline") is not None:
        ym.close()
        _flush()
        print(f"[studio] released YuE2 ({_free_gb():.1f} GiB free)", flush=True)


def forget_voice(name: str):
    """A deleted voice must not linger in the voice model's 'already loaded' memo
    (a new voice trained under the same name would silently reuse the old one)."""
    sv = sys.modules.get("song_voice.song_voice_node")
    model = sv._cache.get("seed_vc") if sv is not None else None
    current = getattr(model, "voice", None)
    if current and os.path.splitext(os.path.basename(str(current[0])))[0] == name:
        model.voice = ("", -1.0)                  # matches nothing: forces a reload


def _park_voice_models():
    import torch
    try:
        from song_voice import song_voice_node as sv
    except Exception:
        return
    for key in ("seed_vc", "hdemucs"):
        model = sv._cache.get(key)
        if model is not None:
            try:
                model.to(torch.device("cpu"))
            except Exception:
                pass
    _flush()


def _make_room(need_gb: float, keep: str):
    """Evict other stages' models until `need_gb` is free. `keep` is the stage
    about to run: 'lyrics', 'yue2' or 'voice'."""
    import comfy.model_management as mm
    _flush()
    evictions = []
    if keep != "lyrics":
        evictions.append(_drop_lyrics_model)
    if keep != "voice":
        evictions.append(_park_voice_models)
    evictions.append(lambda: (mm.unload_all_models(), mm.soft_empty_cache()))
    if keep != "yue2":
        evictions.append(_drop_yue2)
        evictions.append(_drop_native)
    for evict in evictions:
        if _free_gb() >= need_gb:
            break
        evict()
    print(f"[studio] VRAM free before {keep}: {_free_gb():.1f} GiB", flush=True)


def gpu_status() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"available": False}
    free, total = torch.cuda.mem_get_info()
    return {"available": True, "name": torch.cuda.get_device_name(0),
            "total_gb": round(total / 2**30, 1), "used_gb": round((total - free) / 2**30, 1)}


# ------------------------------------------------------------------ lyrics
def tidy_lyrics(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = re.sub(r"^```[a-zA-Z]*\s*$", "", text, flags=re.M)
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").split("\n")]
    text = library.strip_multipliers("\n".join(lines))        # "2x [verse]" -> "[verse]"
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def lyrics_writers() -> list[dict]:
    return library.lyrics_writers()


def _load_lyrics_model(writer: dict, rep):
    """Load one of library.LYRICS_WRITERS onto the GPU (downloading it first
    if this PC does not have it yet)."""
    global _lyrics_model, _lyrics_writer
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    path = library.writer_dir(writer)
    if not library.writer_installed(writer):
        rep.start("download")
        from huggingface_hub import snapshot_download
        print(f"[studio] downloading {writer['repo']} ({writer['download_gb']} GB) to {path}", flush=True)
        snapshot_download(writer["repo"], local_dir=path)
        _check()
    else:
        rep.skip("download")

    rep.start("load")
    _make_room(writer["vram_gb"] + 1.0, keep="lyrics")
    if writer["bits"] == 16:
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        model.to("cuda")
    else:
        from transformers import BitsAndBytesConfig
        if writer["bits"] == 8:
            quant = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    model.eval()
    model.processor = AutoProcessor.from_pretrained(path)
    _lyrics_model, _lyrics_writer = model, writer["id"]
    print(f"[studio] lyrics writer '{writer['id']}' loaded "
          f"({torch.cuda.memory_allocated() / 2**30:.1f} GiB)", flush=True)


def write_lyrics(idea: str, on_text=None, writer: str = "", structure=None, lines=None,
                 target_seconds=None, memory: str = "auto") -> str:
    """Idea -> lyrics in the requested song shape (library.clean_structure).
    `on_text(partial)` is called as the text streams in."""
    global _reporter
    if memory not in ("auto", "standard", "low"):
        raise ValueError("Unknown graphics memory setting.")
    profile = resolve_memory(memory)
    structure, lines = library.clean_structure(structure, lines)
    plan = library.lyrics_length_plan(structure, lines, target_seconds,
                                      max_seconds=YUE2_MEMORY[profile]["max_minutes"] * 60)
    counts = plan["counts"] if target_seconds is not None else None
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer

    spec = library.get_writer(writer or library.DEFAULT_WRITER)
    system_prompt = library.lyrics_prompt(structure, lines, counts=counts)
    sampling = dict(LYRICS_SAMPLING, max_new_tokens=library.lyrics_budget(structure, lines, counts=counts))
    expected_chars = max(300.0, plan["total_lines"] * 48.0)
    with _run_lock:
        _begin()
        rep = _reporter = Reporter([
            ("download", f"Downloading the lyrics writer ({spec['download_gb']} GB, once)", 200, 240),
            ("load", "Waking up the lyrics writer", 55, 60),
            ("write", "Writing lyrics", 15)])
        try:
            if _lyrics_model is not None and _lyrics_writer != spec["id"]:
                _drop_lyrics_model()                    # switching writers
            if _lyrics_model is None:
                _load_lyrics_model(spec, rep)
            else:
                rep.skip("download")
                rep.skip("load")
            _check()

            rep.start("write")
            model, proc = _lyrics_model, _lyrics_model.processor
            print(f"[studio] song shape: {' '.join(structure)} · {plan['counts']} lines "
                  f"· estimated {plan['estimated_seconds']} s (approximate)", flush=True)
            chat = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": idea}]
            prompt = proc.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[prompt], return_tensors="pt").to("cuda")
            pieces: list[str] = []

            class Live(TextStreamer):
                def on_finalized_text(self, text, stream_end=False):
                    pieces.append(text)
                    joined = "".join(pieces)
                    rep.frac(min(0.97, len(joined) / expected_chars))
                    if on_text:
                        on_text(tidy_lyrics(joined))

            class Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return _cancel.is_set()

            with torch.inference_mode():
                out = model.generate(
                    **inputs, **sampling,
                    streamer=Live(proc.tokenizer, skip_prompt=True, skip_special_tokens=True),
                    stopping_criteria=StoppingCriteriaList([Stop()]))
            _check()
            text = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            rep.finish()
            return library.fit_lyrics(tidy_lyrics(text), structure, lines, counts=counts)
        except Cancelled as exc:
            _release_after_cancel(exc)
            raise
        except Exception as exc:
            if _cancel.is_set():
                _release_after_cancel(exc)
                raise Cancelled("Cancelled") from None
            traceback.print_exc()
            raise


def write_style(idea: str, engine: str = "yue2", writer: str = "") -> dict:
    """A few words about the sound -> {"name", "style"} worded for the chosen song
    engine. Uses the same lyrics writer (Qwen) as write_lyrics."""
    global _reporter
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    spec = library.get_writer(writer or library.DEFAULT_WRITER)
    with _run_lock:
        _begin()
        rep = _reporter = Reporter([
            ("download", f"Downloading the lyrics writer ({spec['download_gb']} GB, once)", 200, 240),
            ("load", "Waking up the lyrics writer", 55, 60),
            ("write", "Writing the style", 15, 8)])
        try:
            if _lyrics_model is not None and _lyrics_writer != spec["id"]:
                _drop_lyrics_model()
            if _lyrics_model is None:
                _load_lyrics_model(spec, rep)
            else:
                rep.skip("download")
                rep.skip("load")
            _check()

            rep.start("write")
            model, proc = _lyrics_model, _lyrics_model.processor
            chat = [{"role": "system", "content": library.style_writer_prompt(engine)},
                    {"role": "user", "content": idea}]
            prompt = proc.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[prompt], return_tensors="pt").to("cuda")

            class Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return _cancel.is_set()

            with torch.inference_mode():
                out = model.generate(**inputs, **dict(LYRICS_SAMPLING, max_new_tokens=160),
                                     stopping_criteria=StoppingCriteriaList([Stop()]))
            _check()
            text = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            result = library.parse_style_reply(text)
            print(f"[studio] style for {engine}: {result['name']} — {result['style']}", flush=True)
            rep.finish()
            return result
        except Cancelled as exc:
            _release_after_cancel(exc)
            raise
        except Exception as exc:
            if _cancel.is_set():
                _release_after_cancel(exc)
                raise Cancelled("Cancelled") from None
            traceback.print_exc()
            raise


# ------------------------------------------------------------------ song (single-file checkpoints)
# models/checkpoints/yue2_3b_*.safetensors are the Comfy-Org builds of the same
# YuE2-3B. They run through the vendored comfy core's own model management
# instead of the yue2_infer runtime, and the INT8 one is what fits small cards.
# Recipe = ComfyUI-YuE2's "YuE2 Native Text to Song" workflow.
NATIVE_SAMPLER = dict(steps=32, cfg=1.0, sampler_name="dpm_2", scheduler="sgm_uniform")
_native_pipe = None               # {"name", "model", "clip", "vae"}


def _drop_native():
    global _native_pipe
    if _native_pipe is not None:
        import comfy.model_management as mm
        _native_pipe = None
        mm.unload_all_models()
        mm.cleanup_models()
        _flush()
        mm.soft_empty_cache()
        print(f"[studio] released the song model ({_free_gb():.1f} GiB free)", flush=True)


def native_load(checkpoint: str) -> dict:
    """Load a single-file YuE2 checkpoint (MODEL + text encoder + VAE)."""
    global _native_pipe
    import comfy.sd
    if _native_pipe is not None and _native_pipe["name"] == checkpoint:
        return _native_pipe
    _drop_native()
    path = library.checkpoint_path(checkpoint)
    model, clip, vae = comfy.sd.load_checkpoint_guess_config(
        path, output_vae=True, output_clip=True, embedding_directory=None)[:3]
    if model is None or clip is None or vae is None:
        raise ValueError(f"'{checkpoint}' is not a complete YuE2 checkpoint.")
    _native_pipe = {"name": checkpoint, "model": model, "clip": clip, "vae": vae}
    return _native_pipe


def native_generate(pipe: dict, style: str, lyrics: str, seed: int, max_seconds: float = 360.0,
                    cot: str = "off", on_stage=None):
    """style + lyrics -> (AUDIO dict, abc score, info string) with a single-file checkpoint."""
    import comfy.model_management as mm
    import comfy.sample
    import torch
    from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

    clip, model, vae = pipe["clip"], pipe["model"], pipe["vae"]
    abc = ""
    if cot in ("melody", "full"):                  # plan the melody first
        plan = clip.tokenize(style, lyrics=lyrics, cot=cot, seed=seed,
                             max_tokens=4096, penalty_window=100)
        abc = clip.decode(clip.generate(plan, max_length=4096, temperature=0.7, top_p=0.9,
                                        top_k=30, repetition_penalty=1.005, seed=seed))
        _check()
    mode = cot if abc.strip() else "off"
    tokens = clip.tokenize(style, lyrics=lyrics, cot=mode, seed=seed, abc=abc,
                           max_tokens=max(1, round(max_seconds * FRAMES_PER_SECOND)),
                           temperature=1.0, top_p=0.95, top_k=100, repetition_penalty=1.2)
    positive = clip.encode_from_tokens_scheduled(tokens)
    meta = positive[0][1]
    frames = int(meta["yue2_frames"])
    truncated = bool(meta.get("yue2_truncated", frames >= round(max_seconds * FRAMES_PER_SECOND)))
    _check()

    if on_stage:
        on_stage("decode")
    negative = []                                   # ConditioningZeroOut
    for cond, extra in positive:
        extra = dict(extra)
        for key in ("pooled_output", "conditioning_lyrics", "conditioning_scale", "direct_context"):
            if extra.get(key) is not None:
                extra[key] = torch.zeros_like(extra[key])
        negative.append([torch.zeros_like(cond), extra])
    latent = torch.zeros((1, 64, max(1, frames)), device=mm.intermediate_device(),
                         dtype=mm.intermediate_dtype())
    latent = comfy.sample.fix_empty_latent_channels(model, latent, None, 1920)
    noise = comfy.sample.prepare_noise(latent, seed, None)
    samples = comfy.sample.sample(model, noise, positive=positive, negative=negative,
                                  latent_image=latent, denoise=1.0, seed=seed, **NATIVE_SAMPLER)
    _check()
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[-1]
    wave = vae.decode(samples).movedim(-1, 1)
    std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    wave = wave / std
    rate = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
    seconds = wave.shape[-1] / float(rate)
    info = (f"{seconds:.1f}s @{rate}Hz | {pipe['name']} | cot={mode} seed={seed} | "
            f"sem_tokens={frames} | truncated={{'semantic': {truncated}}}")
    return {"waveform": wave.float().cpu(), "sample_rate": int(rate)}, abc, info


# ------------------------------------------------------------------ ACE-Step 1.5 (test)
# Same graph as ComfyUI's audio_ace_step1_5 workflow: UNETLoader -> ModelSamplingAuraFlow(3)
# -> KSampler(euler/simple, 50 steps, cfg 7) with TextEncodeAceStepAudio1.5 and a zeroed
# negative, then VAEDecodeAudio. Lives in the _native_pipe slot so one song model is in memory.
ACE15_ID = "ace15-xl"
ACE15_SAMPLER = dict(steps=50, cfg=7.0, sampler_name="euler", scheduler="simple")
ACE15_SHIFT = 3.0
_BPM = re.compile(r"\b(\d{2,3})\s*bpm\b", re.IGNORECASE)


def ace15_load() -> dict:
    global _native_pipe
    import comfy.model_sampling
    import comfy.sd
    import comfy.utils
    if _native_pipe is not None and _native_pipe["name"] == ACE15_ID:
        return _native_pipe
    _drop_native()
    base = comfy.sd.load_diffusion_model(library.ace15_path("unet"))
    model = base.clone()                                  # ModelSamplingAuraFlow(shift=3)

    class _Flow(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    original = model.get_model_object("model_sampling")
    sampling = _Flow(base.model.model_config)
    sampling.set_parameters(shift=ACE15_SHIFT, multiplier=1.0)
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    model.add_object_patch("model_sampling", sampling)
    clip = comfy.sd.load_clip(ckpt_paths=[library.ace15_path("te_small"), library.ace15_path("te_large")],
                              embedding_directory=None, clip_type=comfy.sd.CLIPType.ACE)
    sd, metadata = comfy.utils.load_torch_file(library.ace15_path("vae"), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    _native_pipe = {"name": ACE15_ID, "model": model, "clip": clip, "vae": vae}
    return _native_pipe


def ace15_seconds(lyrics: str, max_seconds: float) -> float:
    """ACE-Step needs the length up front: sung lines at the app's measured pace,
    plus room for an intro/outro."""
    sung = [ln for ln in lyrics.splitlines()
            if ln.strip() and not re.match(r"^\s*(\d+x\s*)?\[.*\]\s*$", ln)]
    return float(min(max_seconds, max(30.0, len(sung) * library.SECONDS_PER_LINE + 15.0)))


def ace15_generate(pipe: dict, style: str, lyrics: str, seed: int, seconds: float):
    """style + lyrics -> (AUDIO dict, "", info string)."""
    import comfy.model_management as mm
    import comfy.sample
    import torch

    clip, model, vae = pipe["clip"], pipe["model"], pipe["vae"]
    found = _BPM.search(style or "")
    style, lyrics = library.ace_style(style), library.ace_lyrics(lyrics)
    print(f"[ACE-Step] tags: {style}", flush=True)
    bpm = min(300, max(10, int(found.group(1)))) if found else 120
    settings = library.ace_song_settings(style)        # key / time signature if the style names them
    print(f"[ACE-Step] settings: bpm={bpm} {settings}", flush=True)
    tokens = clip.tokenize(style, lyrics=lyrics, bpm=bpm, duration=seconds, **settings,
                           language="en", seed=seed, generate_audio_codes=True,
                           cfg_scale=2.0, temperature=0.85, top_p=1.0, top_k=0, min_p=0.0)   # = audio_ace_step1_5_xl_sft workflow
    positive = clip.encode_from_tokens_scheduled(tokens)
    _check()
    negative = []                                   # ConditioningZeroOut
    for cond, extra in positive:
        extra = dict(extra)
        for key in ("pooled_output", "conditioning_lyrics", "conditioning_scale", "direct_context"):
            if extra.get(key) is not None:
                extra[key] = torch.zeros_like(extra[key])
        negative.append([torch.zeros_like(cond), extra])
    latent = torch.zeros((1, 64, max(1, round(seconds * 48000 / 1920))),
                         device=mm.intermediate_device(), dtype=mm.intermediate_dtype())
    latent = comfy.sample.fix_empty_latent_channels(model, latent, None, 1764)
    noise = comfy.sample.prepare_noise(latent, seed, None)
    samples = comfy.sample.sample(model, noise, positive=positive, negative=negative,
                                  latent_image=latent, denoise=1.0, seed=seed, **ACE15_SAMPLER)
    _check()
    if getattr(samples, "is_nested", False):
        samples = samples.unbind()[-1]
    wave = vae.decode(samples).movedim(-1, 1)
    std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    wave = wave / std
    rate = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 48000))
    info = f"{wave.shape[-1] / float(rate):.1f}s @{rate}Hz | ACE-Step 1.5 XL | bpm={bpm} seed={seed}"
    return {"waveform": wave.float().cpu(), "sample_rate": int(rate)}, "", info


# ------------------------------------------------------------------ song
def _ensure_yue2(rep):
    from yue2_repo.nodes.yue2 import YuE2Loader
    if _yue2_loaded():
        rep.skip("load")
    else:
        rep.start("load")
    (pipe,) = YuE2Loader().load(**YUE2_LOAD)
    return pipe


def _seconds(audio) -> float:
    return audio["waveform"].shape[-1] / float(audio["sample_rate"])


def make_song(lyrics: str, style: str, seed=None, cot: str = "off",
              voice: str = "", title: str = "", idea: str = "", memory: str = "auto",
              model: str = "auto", singer: str = "", instrumental: bool = False) -> dict:
    """Lyrics -> finished song on disk. Returns the library entry."""
    global _reporter
    import folder_paths
    import torch
    from mp3_save import next_song_path, write_mp3
    from song_lrc.song_lrc_node import SongFilename, SongLyricsClean, SongLyricsToLRC
    from yue2_repo.nodes.yue2 import YuE2Sampler

    voice = "" if instrumental else (voice or "").strip()
    singer = "" if instrumental else (singer or "").strip()
    if instrumental:
        lyrics = ""                                     # YuE2 plays music only when the lyrics are blank
    sung_style = (library.instrumental_style(style) if instrumental
                  else library.apply_singer(style, singer))   # the prompt the model actually hears
    if sung_style != style:
        print(f"[studio] singer '{singer}' -> style: {sung_style}", flush=True)
    model = resolve_song_model(model)
    spec = library.get_song_model(model)
    memory = resolve_memory(memory)
    profile = YUE2_MEMORY[memory]
    kind = spec.get("kind", "native" if spec["file"] else "yue2")
    needs_gb = profile["needs_gb"] if kind == "yue2" else spec["needs_gb"]
    sample = dict(YUE2_SAMPLE, semantic_max_tokens=profile["max_tokens"])
    seed = int(seed) if seed not in (None, "", 0, "0") else random.randint(1, 2**48)
    plan = [("prepare", "Getting ready" if instrumental else "Reading the lyrics", 1)]
    if not spec["installed"]:
        if kind == "native" and spec["file"] != library.INT8_FILE:
            raise ValueError(f"The song model '{spec['name']}' is not installed on this PC.")
        size = spec.get("download_gb") or 4
        plan += [("download", f"Downloading the song model ({size:g} GB, once)", 120, 180)]
    plan += [("load", "Loading the song model", 40, 45),
            ("compose", "Composing the song", 35),
            ("decode", "Rendering the audio", 10)]
    if instrumental:
        # YuE2 invents words and sings them even with blank lyrics (checked with Whisper),
        # so the vocals are taken out afterwards; the music-only track transcribes as silence.
        plan += [("split", "Taking out the vocals", 6, 8)]
    else:
        plan += [("align", "Timing the lyrics", 6, 8)]
    if voice:
        plan += [("split", "Separating the vocals", 6, 8),
                 ("clone", f"Singing it in {voice}", 60, 65),
                 ("mix", "Mixing", 1)]
    plan += [("save", "Saving", 2, 3)]

    with _run_lock:
        _begin()
        rep = _reporter = Reporter(plan)
        started = time.time()
        try:
            with torch.inference_mode():
                rep.start("prepare")
                if voice and voice not in [v["name"] for v in library.list_voices()]:
                    raise ValueError(f"There is no trained voice called '{voice}'.")
                lyrics = library.strip_multipliers(lyrics)       # "2x [verse]" -> "[verse]"
                if instrumental:
                    clean_lyrics, found_title = "", "Instrumental"
                else:
                    (clean_lyrics, found_title) = SongLyricsClean().clean(lyrics)
                    if not clean_lyrics.strip():
                        raise ValueError("The lyrics are empty after cleaning — add some lines.")
                (_prefix, safe_title) = SongFilename().prepare(
                    title=(title or "").strip() or found_title, folder="")
                safe_title = library.numbered_title(safe_title)   # a second take is "Title 2", not another "Title"
                _check()

                # The song model gets the whole card, like in the workflow.
                _make_room(99.0, keep="yue2")
                if kind != "yue2":
                    _drop_yue2()                         # only one song model in memory
                else:
                    _drop_native()
                if _free_gb() < needs_gb:
                    rep.note = (f"Other apps are using the graphics card ({_free_gb():.1f} GB free, "
                                f"the song model wants {needs_gb:.0f}). This song will be slow — "
                                "close ComfyUI, games or video tabs next time.")
                    print(f"[studio] WARNING: {rep.note}", flush=True)
                if kind == "ace15":
                    if not spec["installed"]:
                        rep.start("download")
                        library.download_ace15()
                        _check()
                    if _native_pipe is not None and _native_pipe["name"] == ACE15_ID:
                        rep.skip("load")
                    else:
                        rep.start("load")
                    pipe = ace15_load()
                elif kind == "native":
                    if not spec["installed"]:
                        rep.start("download")
                        library.download_int8_song_model()
                        _check()
                    if _native_pipe is not None and _native_pipe["name"] == spec["file"]:
                        rep.skip("load")
                    else:
                        rep.start("load")
                    pipe = native_load(spec["file"])
                else:
                    if not spec["installed"]:
                        rep.start("download")
                        library.download_full_song_model()
                        _check()
                    pipe = _ensure_yue2(rep)
                _check()

                sung_lines = [ln for ln in clean_lyrics.splitlines()
                              if ln.strip() and not re.match(r"^\s*(\d+x\s*)?\[.*\]\s*$", ln)]
                expected_tokens = 6000.0 if instrumental else max(800.0, len(sung_lines) * 105.0)

                def bar(value, total):
                    if total > 512:                      # token bar (abc + semantic)
                        rep.frac(min(0.97, value / expected_tokens))
                    else:                                # tiled VAE decode bar
                        if rep.current != "decode":
                            rep.start("decode", bar=bar)
                        rep.frac(value / max(total, 1.0))

                rep.start("compose", bar=None if kind == "ace15" else bar)   # ACE's own bars aren't YuE2 token counts
                if kind == "ace15":
                    (audio, abc_score, info) = ace15_generate(
                        pipe, sung_style, clean_lyrics or "[Instrumental]", seed=seed,
                        seconds=120.0 if instrumental else ace15_seconds(clean_lyrics, profile["max_minutes"] * 60.0))
                    print(f"[ACE-Step] {info}", flush=True)
                elif kind == "native":
                    (audio, abc_score, info) = native_generate(
                        pipe, sung_style, clean_lyrics, seed=seed, cot=cot,
                        max_seconds=profile["max_minutes"] * 60.0)
                    print(f"[YuE2] {info}", flush=True)
                else:
                    (audio, abc_score, info) = YuE2Sampler().generate(
                        pipeline=pipe, style=sung_style, lyrics=clean_lyrics, cot=cot,
                        seed=seed, **sample)
                _check()
                duration = _seconds(audio)
                cut_short = "'semantic': True" in str(info)   # ran into the length ceiling
                if cut_short:
                    print(f"[studio] the song hit the {profile['max_minutes']}-minute ceiling "
                          f"of the '{memory}' memory setting and may end abruptly", flush=True)

                lrc = None
                if not instrumental:
                    rep.start("align")
                    (lrc,) = SongLyricsToLRC().make_lrc(
                        lyrics=clean_lyrics, seconds=duration, audio=audio,
                        structure=abc_score or "", title=safe_title, artist="",
                        language="en", offset_seconds=0.0, timing_scale=1.0,
                        prefer_alignment=True, word_timing=True)
                    _check()

                original, final = audio, audio
                parts = None                             # (vocals, music) when they get split apart
                if instrumental:
                    from song_voice.song_voice_node import SongVoiceSplit
                    rep.start("split")
                    _make_room(3.0, keep="voice")
                    (_invented, final) = SongVoiceSplit().split(audio=audio, segment_seconds=10)
                    _park_voice_models()
                    _check()
                if voice:
                    from song_voice.song_voice_node import (
                        SongVoiceClone, SongVoiceLoad, SongVoiceMix, SongVoiceSplit)
                    rep.start("split")
                    _make_room(6.0, keep="voice")
                    (vocals, music) = SongVoiceSplit().split(audio=audio, segment_seconds=10)
                    _check()

                    rep.start("clone")
                    _make_room(6.0, keep="voice")
                    (trained,) = SongVoiceLoad().load(voice_name=voice)
                    (cloned, _semitones) = SongVoiceClone().clone(
                        vocals=vocals, trained_voice=trained, my_voice=None, **voice_clone_settings(voice))
                    _check()

                    parts = (cloned, music)
                    rep.start("mix")
                    (final,) = SongVoiceMix().mix(
                        vocals=cloned, instrumental=music, vocal_gain_db=0,
                        instrumental_gain_db=0, instrumental_semitones=0)
                    _park_voice_models()
                    _check()

                rep.start("save")
                mp3 = next_song_path(safe_title, folder_paths.get_output_directory())
                write_mp3(final, mp3)
                if voice:
                    write_mp3(original, library.sidecar(mp3, ".ai.mp3"))
                if parts:                                # the split was free: keep both halves
                    write_mp3(parts[0], library.sidecar(mp3, ".vocals.mp3"))
                    write_mp3(parts[1], library.sidecar(mp3, ".instrumental.mp3"))
                if lrc is not None:
                    with open(library.sidecar(mp3, ".lrc"), "w", encoding="utf-8", newline="\n") as fh:
                        fh.write(str(lrc))
                if (abc_score or "").strip():            # the melody plan, for remixes and covers later
                    with open(library.sidecar(mp3, ".abc"), "w", encoding="utf-8", newline="\n") as fh:
                        fh.write(abc_score.strip() + "\n")
                library.write_meta(mp3, {
                    "title": safe_title, "idea": idea, "lyrics": lyrics, "style": style,
                    "singer": singer, "seed": seed, "cot": cot, "voice": voice, "memory": memory, "model": model,
                    "cut_short": cut_short, "instrumental": bool(instrumental),
                    "duration": round(_seconds(final), 2),
                    "took": round(time.time() - started, 1), "engine": info,
                })
                print(f"[studio] saved {mp3} in {time.time() - started:.0f}s", flush=True)
            rep.finish()
            return library.describe(mp3)
        except Cancelled as exc:
            _release_after_cancel(exc)
            raise
        except Exception as exc:
            if _cancel.is_set():
                _release_after_cancel(exc)
                raise Cancelled("Cancelled") from None
            traceback.print_exc()
            raise
        finally:
            _flush()


# ------------------------------------------------------------------ split
def split_song(sid: str) -> dict:
    """Finished song -> its music (.instrumental.mp3) and its vocals (.vocals.mp3)."""
    global _reporter
    import torch
    from mp3_save import write_mp3
    from song_voice.song_voice_node import SongVoiceSplit

    mp3 = library.song_path(sid)
    with _run_lock:
        _begin()
        rep = _reporter = Reporter([("prepare", "Reading the song", 1),
                                    ("split", "Separating the music and the vocals", 30, 40),
                                    ("save", "Saving", 2, 3)])
        started = time.time()
        try:
            with torch.inference_mode():
                rep.start("prepare")
                audio = _read_clip(mp3)
                _check()
                rep.start("split")
                _make_room(3.0, keep="voice")
                (vocals, music) = SongVoiceSplit().split(audio=audio, segment_seconds=10)
                _park_voice_models()
                _check()
                rep.start("save")
                write_mp3(music, library.sidecar(mp3, ".instrumental.mp3"))
                write_mp3(vocals, library.sidecar(mp3, ".vocals.mp3"))
                print(f"[studio] split {os.path.basename(mp3)} in {time.time() - started:.0f}s", flush=True)
            rep.finish()
            return library.describe(mp3)
        except Cancelled as exc:
            _release_after_cancel(exc)
            for ext in (".instrumental.mp3", ".vocals.mp3"):    # never leave half a pair
                try:
                    os.remove(library.sidecar(mp3, ext))
                except OSError:
                    pass
            raise
        except Exception as exc:
            if _cancel.is_set():
                _release_after_cancel(exc)
                raise Cancelled("Cancelled") from None
            traceback.print_exc()
            raise
        finally:
            _flush()


# ------------------------------------------------------------------ voice
def _read_clip(path: str):
    """Audio file -> ComfyUI AUDIO dict [1, channels, samples]."""
    import numpy as np
    import torch
    try:
        import soundfile as sf
        wave, sr = sf.read(path, dtype="float32", always_2d=True)
        wave = wave.T
    except Exception:
        import librosa
        wave, sr = librosa.load(path, sr=None, mono=False)
        wave = np.atleast_2d(wave)
    return {"waveform": torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))[None],
            "sample_rate": int(sr)}


def train_voice(clip_path: str, voice_name: str, retrain: bool = False,
                recording_type: str = "talking") -> dict:
    """Fine-tune Seed-VC and remember the reference recording's performance type."""
    global _reporter
    recording_type = library.clean_voice_recording_type(recording_type)
    import torch
    from song_voice.song_voice_node import SongVoiceTrain

    with _run_lock:
        _begin()
        rep = _reporter = Reporter([("read", "Listening to your clip", 3),
                                    ("train", f"Learning the voice '{voice_name}'", 100)])
        try:
            rep.start("read")
            audio = _read_clip(clip_path)
            seconds = _seconds(audio)
            if seconds < 10:
                raise ValueError(f"That clip is only {seconds:.0f} seconds long — use 30 seconds to 3 minutes of clean talking, singing, or rapping.")
            _check()
            rep.start("train")
            _make_room(10.0, keep="voice")
            with torch.inference_mode():
                (voice,) = SongVoiceTrain().train(
                    my_voice=audio, voice_name=voice_name,
                    training_steps=VOICE_TRAIN_STEPS, retrain=bool(retrain))
            library.set_voice_recording_type(voice["name"], recording_type)
            _park_voice_models()
            _check()
            rep.finish()
            return {"name": voice["name"], "recording_type": recording_type}
        except Cancelled as exc:
            _release_after_cancel(exc)
            raise
        except Exception as exc:
            if _cancel.is_set():
                _release_after_cancel(exc)
                raise Cancelled("Cancelled") from None
            traceback.print_exc()
            raise
        finally:
            _flush()


def progress() -> dict:
    return _reporter.snapshot()
