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
VOICE_CLONE = dict(my_clip_is="talking", diffusion_steps=40, cfg_rate=0.7,
                   pitch_mode="octave to my range", pitch_shift=0,
                   reference_seconds=15, follow_original=True, fp16=True,
                   keep_loaded=False, voice_strength=0.8)
VOICE_TRAIN_STEPS = 300


def voice_clone_settings(voice: str, overrides: dict | None = None) -> dict:
    """Use the saved reference type without leaking settings between voices."""
    out = dict(VOICE_CLONE, my_clip_is=library.voice_clip_mode(voice))
    if overrides:
        clean = library.clean_voice_settings(overrides)
        for key in ("voice_strength", "diffusion_steps", "cfg_rate", "pitch_mode", "pitch_shift"):
            out[key] = clean[key]
    return out

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
    for key in ("seed_vc", "hdemucs", "roformer_vocals", "roformer_karaoke"):
        entry = sv._cache.get(key)
        model = entry[0] if isinstance(entry, tuple) else entry
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
    if keep != "cover":
        evictions.append(_drop_sheetsage)
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
                    cot: str = "off", on_stage=None, score: str = ""):
    """style + lyrics -> (AUDIO dict, abc score, info string) with a single-file checkpoint."""
    import comfy.model_management as mm
    import comfy.sample
    import torch
    from comfy.text_encoders.yue2 import FRAMES_PER_SECOND

    clip, model, vae = pipe["clip"], pipe["model"], pipe["vae"]
    abc = (score or "").strip()                   # a cover follows a given score instead of planning one
    if not abc and cot in ("melody", "full"):      # plan the melody first
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


def _stem_cache_dir(key: str) -> str:
    base = os.path.join(os.path.abspath(library.CFG.get("temp_dir") or os.path.join(library.OUTPUT_DIR, os.pardir, "temp")), "stems")
    return os.path.join(base, key)


def _prune_stem_cache(keep: int = 8) -> None:
    base = os.path.dirname(_stem_cache_dir("_"))
    if not os.path.isdir(base):
        return
    import shutil
    entries = sorted(
        (os.path.join(base, name) for name in os.listdir(base) if os.path.isdir(os.path.join(base, name))),
        key=os.path.getmtime,
    )
    for stale in entries[:-keep]:
        shutil.rmtree(stale, ignore_errors=True)


def _write_audio_wav(audio: dict, path: str) -> None:
    import soundfile as sf
    wave = audio["waveform"][0].detach().float().cpu().numpy().T
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.wav"
    sf.write(tmp, wave, int(audio["sample_rate"]), subtype="PCM_16")
    os.replace(tmp, path)


def _read_stem_cache_meta(cache_dir: str) -> dict:
    import json
    if not cache_dir:
        return {}
    try:
        with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_stem_cache_meta(cache_dir: str, meta: dict) -> None:
    import json
    os.makedirs(cache_dir, exist_ok=True)
    tmp = os.path.join(cache_dir, "meta.json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(tmp, os.path.join(cache_dir, "meta.json"))


def _clone_and_mix(audio: dict, voice: str, settings: dict, rep, stem_cache_key: str = ""):
    """Separate vocals (BS-RoFormer) -> optional lead/backing split (Karaoke RoFormer)
    -> Seed-VC clone on lead -> studio mix polish.

    Fallback stems (Hybrid Demucs or skipped Karaoke) are never written to the RoFormer
    stem cache, and any fallback is surfaced via rep.note and the returned separator info.
    Returns (final_mix, vocals_sidecar, music_sidecar, all_vocals, semitones, sep_info).
    """
    from song_voice.song_voice_node import (
        SongVoiceClone, SongVoiceLoad, SongVoiceMix, SongVoiceSplit)

    keep_backing = bool(settings.get("keep_backing", True))
    cache_dir = _stem_cache_dir(stem_cache_key) if stem_cache_key else ""
    inst_wav = os.path.join(cache_dir, "inst.wav") if cache_dir else ""
    voc_wav = os.path.join(cache_dir, "vocals.wav") if cache_dir else ""
    lead_wav = os.path.join(cache_dir, "lead.wav") if cache_dir else ""
    back_wav = os.path.join(cache_dir, "backing.wav") if cache_dir else ""
    cache_meta = _read_stem_cache_meta(cache_dir)

    splitter = SongVoiceSplit()
    warnings: list[str] = []

    if (cache_dir and cache_meta.get("separator") == "roformer"
            and os.path.isfile(inst_wav) and os.path.isfile(voc_wav)):
        rep.skip("split")
        all_vocals = _read_clip(voc_wav)
        music = _read_clip(inst_wav)
        used_separator = "roformer"
        try:
            os.utime(cache_dir, None)
        except OSError:
            pass
    else:
        rep.start("split")
        _make_room(5.0, keep="voice")
        (all_vocals, music) = splitter.split(audio=audio, segment_seconds=10)
        _check()
        used_separator = getattr(splitter, "last_separator", "roformer")
        if used_separator == "roformer":
            if cache_dir:
                try:
                    _write_audio_wav(all_vocals, voc_wav)
                    _write_audio_wav(music, inst_wav)
                    cache_meta["separator"] = "roformer"
                    _write_stem_cache_meta(cache_dir, cache_meta)
                    _prune_stem_cache()
                except OSError:
                    pass
        else:
            warnings.append("BS-RoFormer was unavailable; fell back to Hybrid Demucs (see Activity log).")
            rep.note = " ".join(warnings)

    backing_vocals = None
    used_lead = "none"
    if keep_backing:
        if (cache_dir and used_separator == "roformer" and cache_meta.get("lead") == "karaoke"
                and os.path.isfile(lead_wav) and os.path.isfile(back_wav)):
            rep.skip("lead")
            lead_vocals = _read_clip(lead_wav)
            backing_vocals = _read_clip(back_wav)
            used_lead = "karaoke"
        else:
            rep.start("lead")
            _make_room(5.0, keep="voice")
            (lead_vocals, backing_vocals) = splitter.split_lead(vocals=all_vocals)
            _check()
            used_lead = getattr(splitter, "last_lead_separator", "karaoke" if backing_vocals is not None else "none")
            if used_lead == "karaoke" and backing_vocals is not None:
                if cache_dir and used_separator == "roformer":
                    try:
                        _write_audio_wav(lead_vocals, lead_wav)
                        _write_audio_wav(backing_vocals, back_wav)
                        cache_meta["lead"] = "karaoke"
                        _write_stem_cache_meta(cache_dir, cache_meta)
                        _prune_stem_cache()
                    except OSError:
                        pass
            else:
                warnings.append("Karaoke harmony split was unavailable; cloned the full vocal stem instead (see Activity log).")
                rep.note = " ".join(warnings)
    else:
        if any(s["key"] == "lead" for s in rep.steps):
            rep.skip("lead")
        lead_vocals = all_vocals

    rep.start("clone")
    _make_room(6.0, keep="voice")
    (trained,) = SongVoiceLoad().load(voice_name=voice)
    (cloned, semitones) = SongVoiceClone().clone(
        vocals=lead_vocals, trained_voice=trained, my_voice=None,
        **voice_clone_settings(voice, settings))
    _check()

    rep.start("mix")
    mixer = SongVoiceMix()
    polish = bool(settings.get("polish", True))
    reverb = float(settings.get("reverb", 0.18))
    vocal_gain_db = float(settings.get("vocal_gain_db", 0.0))
    backing_gain_db = float(settings.get("backing_gain_db", -1.5))
    (final,) = mixer.mix(
        vocals=cloned, instrumental=music, vocal_gain_db=vocal_gain_db,
        instrumental_gain_db=0.0, instrumental_semitones=semitones,
        backing_vocals=backing_vocals, backing_gain_db=backing_gain_db,
        polish=polish, reverb_amount=reverb)
    (vocals_sidecar,) = mixer.combine_vocals(
        vocals=cloned, backing_vocals=backing_vocals, vocal_gain_db=vocal_gain_db,
        backing_gain_db=backing_gain_db, polish=polish, reverb_amount=reverb,
        backing_semitones=semitones)
    _park_voice_models()
    _check()
    sep_info = {"separator": used_separator, "lead": used_lead, "warnings": warnings}
    return final, vocals_sidecar, music, all_vocals, semitones, sep_info


def make_song(lyrics: str, style: str, seed=None, cot: str = "off",
              voice: str = "", title: str = "", idea: str = "", memory: str = "auto",
              model: str = "auto", singer: str = "", instrumental: bool = False,
              score: str = "", cover: dict | None = None,
              voice_settings: dict | None = None) -> dict:
    """Lyrics -> finished song on disk. Returns the library entry."""
    global _reporter
    import folder_paths
    import torch
    from mp3_save import next_song_path, write_mp3
    from song_lrc.song_lrc_node import SongFilename, SongLyricsClean, SongLyricsToLRC
    from yue2_repo.nodes.yue2 import YuE2Sampler

    voice = "" if instrumental else (voice or "").strip()
    singer = "" if instrumental else (singer or "").strip()
    vsettings = library.clean_voice_settings(voice_settings) if voice else None
    if instrumental:
        lyrics = ""                                     # YuE2 plays music only when the lyrics are blank
    sung_style = (library.instrumental_style(style) if instrumental
                  else library.apply_singer(style, singer))   # the prompt the model actually hears
    if sung_style != style:
        print(f"[studio] singer '{singer}' -> style: {sung_style}", flush=True)
    score = (score or "").strip()
    if score:
        cot = "full" if library.score_has_chords(score) else "melody"   # what the score carries
    model = resolve_song_model(model)
    spec = library.get_song_model(model)
    if score and spec.get("kind") == "ace15":
        raise ValueError("Covers use YuE2: pick YuE2 (or Auto) as the song model in step 3.")
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
        plan += [("split", "Taking out the vocals", 12, 16)]
    elif not voice:
        plan += [("align", "Timing the lyrics", 6, 8)]
    if voice:
        plan += [("split", "Separating the vocals", 12, 16)]
        if vsettings["keep_backing"]:
            plan += [("lead", "Isolating lead vocals from harmonies", 10, 14)]
        plan += [("clone", f"Singing it in {voice}", 60, 65),
                 ("mix", "Polishing & mixing", 2),
                 ("align", "Timing the lyrics", 6, 8)]
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
                        pipe, sung_style, clean_lyrics, seed=seed, cot=cot, score=score,
                        max_seconds=profile["max_minutes"] * 60.0)
                    print(f"[YuE2] {info}", flush=True)
                else:
                    (audio, abc_score, info) = YuE2Sampler().generate(
                        pipeline=pipe, style=sung_style, lyrics=clean_lyrics, cot=cot,
                        seed=seed, **dict(sample, abc=score or None))
                _check()
                duration = _seconds(audio)
                cut_short = "'semantic': True" in str(info)   # ran into the length ceiling
                if cut_short:
                    print(f"[studio] the song hit the {profile['max_minutes']}-minute ceiling "
                          f"of the '{memory}' memory setting and may end abruptly", flush=True)

                lrc = None
                original, final = audio, audio
                parts = None                             # (vocals, music) when they get split apart
                if instrumental:
                    from song_voice.song_voice_node import SongVoiceSplit
                    rep.start("split")
                    _make_room(3.0, keep="voice")
                    (_invented, final) = SongVoiceSplit().split(audio=audio, segment_seconds=10)
                    _park_voice_models()
                    _check()
                sep_info = None
                align_audio = audio
                if voice:
                    final, vocals_out, music, all_vocals, _semitones, sep_info = _clone_and_mix(
                        audio=audio, voice=voice, settings=vsettings, rep=rep)
                    parts = (vocals_out, music)
                    # Align WhisperX on clean isolated vocals (no guitars/drums interference)
                    align_audio = all_vocals
                if not instrumental:
                    rep.start("align")
                    (lrc,) = SongLyricsToLRC().make_lrc(
                        lyrics=clean_lyrics, seconds=duration, audio=align_audio,
                        structure=abc_score or "", title=safe_title, artist="",
                        language="en", offset_seconds=0.0, timing_scale=1.0,
                        prefer_alignment=True, word_timing=True)
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
                    "singer": singer, "seed": seed, "cot": cot, "voice": voice,
                    **({"voice_settings": vsettings} if vsettings else {}),
                    **({"separator": sep_info["separator"]} if sep_info else {}),
                    "memory": memory, "model": model,
                    "cut_short": cut_short, "instrumental": bool(instrumental),
                    **({"cover": cover} if cover else {}),
                    "duration": round(_seconds(final), 2),
                    "took": round(time.time() - started, 1), "engine": info,
                })
                print(f"[studio] saved {mp3} in {time.time() - started:.0f}s", flush=True)
            rep.finish()
            entry = library.describe(mp3)
            if sep_info and sep_info["warnings"]:
                entry["warnings"] = sep_info["warnings"]
            return entry
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


# ------------------------------------------------------------------ covers
# SheetSage2 listens to a song and writes its melody, chords, key, tempo and sections as an
# ABC score; make_song(score=...) then has YuE2 follow that melody. Checked on a real song:
# same key, tempo and length, and 0.69 melody match vs 0.26 without the score.
_ss2 = None
WHISPER_ID = "openai/whisper-large-v3-turbo"
_HALLUCINATIONS = re.compile(r"^(?:thanks? (?:you )?for watching.*|thank you\.?|you|bye\.?|music|\W*)$", re.IGNORECASE)


def _drop_sheetsage():
    global _ss2
    if _ss2 is not None:
        import comfy.model_management as mm
        _ss2 = None
        mm.unload_all_models()
        _flush()


def _load_sheetsage(rep):
    global _ss2
    import comfy.audio_encoders.audio_encoders as ae
    import comfy.model_patcher  # noqa: F401  (the encoder builds a ModelPatcher)
    import comfy.utils
    if _ss2 is None:
        if not os.path.isfile(library.sheetsage_path()):
            rep.start("download")
            library.download_sheetsage()
            _check()
        rep.start("load")
        _ss2 = ae.load_audio_encoder_from_sd(
            comfy.utils.load_torch_file(library.sheetsage_path(), safe_load=True))
    else:
        rep.skip("load")
    return _ss2


def _write_down_lyrics(vocals) -> list:
    """Isolated vocals -> [(seconds, line)] with Whisper; quiet stretches are skipped
    because Whisper invents words over silence ("Thanks for watching!")."""
    import numpy as np
    import torch
    import torchaudio
    from transformers import pipeline
    wave = vocals["waveform"][0].float().mean(dim=0)
    wave = torchaudio.functional.resample(wave, vocals["sample_rate"], 16000).numpy()
    asr = pipeline("automatic-speech-recognition", model=WHISPER_ID, device=0, torch_dtype=torch.float16)
    try:
        out = asr(wave.copy(), chunk_length_s=30, batch_size=4, return_timestamps="word",
                  generate_kwargs={"task": "transcribe"})
    finally:
        del asr
        _flush()
    words = [(*(chunk.get("timestamp") or (None, None)), chunk.get("text") or "") for chunk in out.get("chunks") or []]
    loud = float(np.sqrt(np.mean(wave ** 2))) or 1e-6
    lines = []
    for end, text in library.group_words(words):
        if _HALLUCINATIONS.match(text) or [t for _, t in lines[-2:]] == [text, text]:
            continue                                      # a third identical line in a row is Whisper looping
        part = wave[max(0, int((end - 2.5) * 16000)):int(end * 16000) + 1]
        if part.size and float(np.sqrt(np.mean(part ** 2))) < 0.25 * loud:
            continue                                      # nobody is singing here
        lines.append((end, text))
    return lines


def analyze_cover(audio_path: str, name: str, lyrics: str = "transcribe", song_id: str = "") -> dict:
    """A song file -> a saved cover plan (scores, key, tempo, sections, lyrics).
    lyrics: "transcribe" writes the sung words down, "none" leaves section tags only,
    anything else is used as the lyrics (a song made here already knows its words)."""
    global _reporter
    import torch
    from comfy.audio_encoders.sheetsage2_abc import events_to_abc, interval_rows
    import comfy.audio
    import comfy.model_management as mm

    write_down = lyrics == "transcribe"
    plan = [("prepare", "Reading the song", 1)]
    if not os.path.isfile(library.sheetsage_path()):
        plan += [("download", "Downloading the song-listening model (1.4 GB, once)", 60, 90)]
    plan += [("load", "Loading the song-listening model", 8, 10),
             ("listen", "Listening: melody, chords, key and sections", 15, 20)]
    if write_down:
        plan += [("split", "Separating the vocals", 8, 10),
                 ("words", "Writing down the lyrics", 30, 40)]
    plan += [("save", "Saving", 1)]
    with _run_lock:
        _begin()
        rep = _reporter = Reporter(plan)
        started = time.time()
        try:
            with torch.inference_mode():
                rep.start("prepare")
                audio = _read_clip(audio_path)
                seconds = _seconds(audio)
                if seconds < 15:
                    raise ValueError("That song is too short to cover; use at least 15 seconds of music.")
                _make_room(4.0, keep="cover")
                encoder = _load_sheetsage(rep)
                _check()
                rep.start("listen")
                wave = comfy.audio.resample(audio["waveform"].float().mean(dim=1), audio["sample_rate"], 24000)
                mm.load_model_gpu(encoder.patcher)
                events = encoder.model.transcribe(wave[0][None].to(encoder.load_device))
                melody = events_to_abc(events, seconds, melody_only=True)
                full = events_to_abc(events, seconds, melody_only=False)
                _drop_sheetsage()
                _check()
                summary = library.score_summary(full)
                spans = interval_rows(events, "structure", seconds)
                merged = []
                for start, end, label in spans:
                    tag = library.cover_tag(label)
                    if merged and merged[-1]["name"] == label:
                        merged[-1]["end"] = end
                    else:
                        merged.append({"name": label, "tag": tag, "start": round(start, 2), "end": round(end, 2)})
                sections = summary["sections"]
                if len(merged) == len(sections):          # the score's sections, with their times
                    for section, span in zip(sections, merged):
                        section.update(start=span["start"], end=span["end"])
                else:
                    sections = [dict(span, sung=True, lines=4) for span in merged] or sections
                if write_down:
                    from song_voice.song_voice_node import SongVoiceSplit
                    rep.start("split")
                    _make_room(3.0, keep="voice")
                    (vocals, _music) = SongVoiceSplit().split(audio=audio, segment_seconds=10)
                    _park_voice_models()
                    _check()
                    rep.start("words")
                    timed = _write_down_lyrics(vocals)
                    _check()
                    words = library.cover_lyrics(sections, timed) if timed else library.cover_lyrics(sections)
                    source = "transcribed" if timed else "none"
                elif lyrics and lyrics != "none":
                    words, source = lyrics.strip() + "\n", "song"
                else:
                    words, source = library.cover_lyrics(sections), "none"
                rep.start("save")
                info = {"id": library.cover_id(name), "name": (name or "Song").strip()[:80],
                        "duration": round(seconds, 2), "key": summary["key"], "bpm": summary["bpm"],
                        "meter": summary["meter"], "sections": sections, "lyrics": words,
                        "lyrics_source": source, "melody_abc": melody, "full_abc": full,
                        "audio": audio_path, "song_id": song_id, "created": time.time()}
                brief = library.save_cover(info)
                print(f"[studio] cover of '{info['name']}' ready in {time.time() - started:.0f}s: "
                      f"{summary['key']} {summary['bpm']} BPM, {len(sections)} sections, lyrics={source}", flush=True)
            rep.finish()
            return brief
        except Cancelled as exc:
            _drop_sheetsage()
            _release_after_cancel(exc)
            raise
        except Exception as exc:
            _drop_sheetsage()
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


def _lrc_needs_order_repair(lrc_path: str, lyrics: str, duration: float = 0.0) -> bool:
    """Detect legacy LRC files whose lyric lines were scrambled or truncated by old cover alignment."""
    if not lyrics or not os.path.isfile(lrc_path):
        return False
    try:
        from song_lrc.song_lrc_node import SongLyricsToLRC, strip_lrc_markup
        expected = SongLyricsToLRC()._get_display_lines(lyrics)
        if not expected:
            return False
        actual = []
        last_sec = 0.0
        fallback_byline = False
        with open(lrc_path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln.lower().startswith("[by:") and "section-timing" in ln.lower():
                    fallback_byline = True
                if ln.startswith("[") and "]" in ln and not ln.startswith(("[ti:", "[ar:", "[length:", "[by:")):
                    m = re.match(r"^\[(\d+):(\d+(?:\.\d+)?)\]", ln)
                    if m:
                        last_sec = max(last_sec, int(m.group(1)) * 60 + float(m.group(2)))
                    body = strip_lrc_markup(ln)
                    if body:
                        actual.append(body)
        if len(actual) == len(expected) and actual != expected:
            return True
        if duration > 60.0 and last_sec > 0.0 and last_sec < duration * 0.45 and (fallback_byline or len(actual) >= 10):
            return True
        return False
    except Exception:
        return False


# ------------------------------------------------------------------ reclone
def reclone_song(sid: str, voice: str, voice_settings: dict | None = None) -> dict:
    """Re-sing an existing song in a trained voice (reusing its AI take and cached stems)."""
    global _reporter
    import hashlib
    import json
    import shutil
    import torch
    from mp3_save import write_mp3

    mp3 = library.song_path(sid)
    song = library.describe(mp3)
    if song["instrumental"]:
        raise ValueError("Instrumental songs have no vocals to re-clone.")
    voice = (voice or "").strip()
    if not voice or voice not in [v["name"] for v in library.list_voices()]:
        raise ValueError(f"There is no trained voice called '{voice}'.")
    vsettings = library.clean_voice_settings(voice_settings)

    plan = [("prepare", "Reading the AI take", 1),
            ("split", "Separating vocals (BS-RoFormer)", 15, 16)]
    if vsettings["keep_backing"]:
        plan += [("lead", "Isolating lead vocals from harmonies", 12, 14)]
    plan += [("clone", f"Singing it in {voice}", 60, 65),
             ("mix", "Polishing & mixing", 2),
             ("save", "Saving", 2, 3)]

    with _run_lock:
        _begin()
        rep = _reporter = Reporter(plan)
        started = time.time()
        try:
            with torch.inference_mode():
                rep.start("prepare")
                ai_mp3 = library.sidecar(mp3, ".ai.mp3")
                if not os.path.isfile(ai_mp3):
                    shutil.copy2(mp3, ai_mp3)       # preserve the original AI-singer mix
                st = os.stat(ai_mp3)
                cache_key = hashlib.sha1(
                    f"roformer_v1|{os.path.abspath(ai_mp3)}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")
                ).hexdigest()[:20]
                audio = _read_clip(ai_mp3)
                _check()

                final, vocals_out, music, all_vocals, _semitones, sep_info = _clone_and_mix(
                    audio=audio, voice=voice, settings=vsettings, rep=rep,
                    stem_cache_key=cache_key)

                # Auto-repair legacy LRC files whose lyric lines were scrambled or truncated by old cover alignment
                lrc_path = library.sidecar(mp3, ".lrc")
                repaired_lrc = None
                if song.get("lyrics") and _lrc_needs_order_repair(lrc_path, song["lyrics"], _seconds(all_vocals)):
                    try:
                        from song_lrc.song_lrc_node import SongLyricsClean, SongLyricsToLRC
                        (clean_lrc_lyrics, _t) = SongLyricsClean().clean(library.strip_multipliers(song["lyrics"]))
                        abc_path = library.sidecar(mp3, ".abc")
                        abc_text = ""
                        if os.path.isfile(abc_path):
                            with open(abc_path, encoding="utf-8") as fh:
                                abc_text = fh.read()
                        (repaired_lrc,) = SongLyricsToLRC().make_lrc(
                            lyrics=clean_lrc_lyrics, seconds=_seconds(all_vocals), audio=all_vocals,
                            structure=abc_text, title=song["title"], artist="",
                            language="en", offset_seconds=0.0, timing_scale=1.0,
                            prefer_alignment=True, word_timing=True)
                    except Exception as exc:
                        print(f"[studio] skipped LRC repair: {exc}", flush=True)

                rep.start("save")
                # Write via temp files so a failure never leaves a half-written song
                tmp_main = mp3 + ".reclone.tmp.mp3"
                tmp_voc = library.sidecar(mp3, ".vocals.mp3") + ".reclone.tmp.mp3"
                tmp_inst = library.sidecar(mp3, ".instrumental.mp3") + ".reclone.tmp.mp3"
                try:
                    write_mp3(final, tmp_main)
                    write_mp3(vocals_out, tmp_voc)
                    write_mp3(music, tmp_inst)
                    os.replace(tmp_main, mp3)
                    os.replace(tmp_voc, library.sidecar(mp3, ".vocals.mp3"))
                    os.replace(tmp_inst, library.sidecar(mp3, ".instrumental.mp3"))
                    if repaired_lrc:
                        with open(lrc_path, "w", encoding="utf-8", newline="\n") as fh:
                            fh.write(str(repaired_lrc))
                        print(f"[studio] repaired scrambled LRC for {os.path.basename(mp3)}", flush=True)
                finally:
                    for p in (tmp_main, tmp_voc, tmp_inst):
                        if os.path.isfile(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass

                meta_path = library.sidecar(mp3, ".json")
                try:
                    with open(meta_path, encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    meta = {"title": song["title"]}
                meta["voice"] = voice
                meta["voice_settings"] = vsettings
                meta["separator"] = sep_info["separator"]
                meta["updated"] = time.time()
                meta["duration"] = round(_seconds(final), 2)
                library.write_meta(mp3, meta)
                print(f"[studio] re-cloned {os.path.basename(mp3)} in '{voice}' via {sep_info['separator']} ({time.time() - started:.0f}s)", flush=True)
            rep.finish()
            entry = library.describe(mp3)
            if sep_info["warnings"]:
                entry["warnings"] = sep_info["warnings"]
            return entry
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
