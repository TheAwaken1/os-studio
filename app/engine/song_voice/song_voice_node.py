"""Sing a generated song in your own voice.

Three nodes, meant to sit right after any song generator (YuE2, ACE-Step, ...):

    Split Vocals   song -> vocals + instrumental   (BS-RoFormer, with Hybrid Demucs fallback)
    Voice Clone    vocals + a clip of you -> vocals in your voice   (Seed-VC singing model)
    Remix          your vocals + instrumental -> finished song

Voice Clone is zero-shot: no training, one 10-25 s clip of you is enough.
The song keeps its exact length, so lyric timing (LRC) still lines up.
"""
import contextlib
import math
import os
import subprocess
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import comfy.model_management as mm
import comfy.utils
import folder_paths

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED_VC_DIR = os.path.join(NODE_DIR, "seed-vc")
SEED_VC_REPO = "https://github.com/Plachtaa/seed-vc"
SEED_VC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"
MODEL_DIR = os.path.join(folder_paths.models_dir, "seed_vc")

# (repo, files, local folder) -- the same layout download/seedvc.js produces.
SEED_VC_FILES = [
    ("Plachta/Seed-VC", ["DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                         "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"], "Seed-VC"),
    ("lj1995/VoiceConversionWebUI", ["rmvpe.pt"], "rmvpe"),
    ("funasr/campplus", ["campplus_cn_common.bin"], "campplus"),
    ("nvidia/bigvgan_v2_44khz_128band_512x", ["config.json", "bigvgan_generator.pt"], "bigvgan_v2_44khz_128band_512x"),
    ("openai/whisper-small", ["config.json", "model.safetensors", "preprocessor_config.json"], "whisper-small"),
]
HDEMUCS_URL = "https://download.pytorch.org/torchaudio/models/hdemucs_high_trained.pt"

# Train My Voice only: OpenVoice re-voices your clip as strangers to train against.
TRAIN_FILES = [
    ("myshell-ai/OpenVoiceV2", ["converter/config.json", "converter/checkpoint.pth"], "openvoice"),
    ("Plachta/Seed-VC", ["se_db.pt"], "Seed-VC"),
]
OPENVOICE_DIR = os.path.join(MODEL_DIR, "openvoice", "converter")
SE_DB_PATH = os.path.join(MODEL_DIR, "Seed-VC", "se_db.pt")
VOICE_DIR = getattr(folder_paths, "voices_directory", os.path.join(MODEL_DIR, "voices"))
DEFAULT_TRAIN_STEPS = 300
# People sing roughly this many semitones above their talking pitch.
TALKING_PITCH_OFFSET = 5.0

PITCH_MODES = ["keep song key", "octave to my range", "exact to my range"]

_cache = {}


# ---------------------------------------------------------------- audio utils

def _audio_in(audio):
    """ComfyUI AUDIO -> (float32 tensor [C, T] on cpu, sample_rate). Batch item 0."""
    wave = audio["waveform"]
    if wave.dim() == 3:
        wave = wave[0]
    elif wave.dim() == 1:
        wave = wave[None]
    return wave.float().cpu(), int(audio["sample_rate"])


def _audio_out(wave, sr):
    return {"waveform": wave.float().cpu()[None], "sample_rate": int(sr)}


def _resample(wave, sr, target):
    return wave if sr == target else torchaudio.functional.resample(wave, sr, target)


def _stereo(wave):
    if wave.shape[0] == 1:
        return wave.repeat(2, 1)
    return wave[:2]


def _fit(wave, length):
    if wave.shape[-1] >= length:
        return wave[..., :length]
    return F.pad(wave, (0, length - wave.shape[-1]))


def _db(gain_db):
    return 10.0 ** (gain_db / 20.0)


def _envelope_db(mono, sr, frame_ms=10):
    """RMS envelope in dB, one value per sample (frame-wise, linearly upsampled)."""
    frame = max(1, int(sr * frame_ms / 1000))
    padded = F.pad(mono, (0, (-mono.numel()) % frame))
    rms = padded.view(-1, frame).pow(2).mean(1).sqrt()
    env = 20.0 * torch.log10(rms + 1e-9)
    return F.interpolate(env[None, None], size=mono.numel(), mode="linear", align_corners=False)[0, 0]


def _follow_original(converted, original, sr, range_db=44.0):
    """Shape converted vocals by the original stem: matched phrasing dynamics, silent where the song has no singing.

    Seed-VC renders at your clip's level and can hum faintly over the stem's
    silent gaps (intros, solos). Using a smooth section-level loudness follower
    keeps quiet verses intimate and lets choruses lift naturally.
    """
    env = _envelope_db(original, sr)
    # Use 99th-percentile level (subsampled for torch.quantile size safety) so a single transient
    # spike never pushes quiet verses below the gate floor.
    stride = max(1, env.numel() // 100_000)
    peak_db = torch.quantile(env[::stride], 0.99).item()
    floor = peak_db - range_db
    active = env > floor
    if active.sum() < sr // 10:
        return converted
    global_gain = original[active].pow(2).mean().sqrt() / (converted[active].pow(2).mean().sqrt() + 1e-9)

    # Smooth phrasing-level (~1.2 s) loudness follower so verses & choruses keep original dynamics.
    hop = max(1, int(sr * 0.02))
    pad = (-original.numel()) % hop
    orig_rms = F.pad(original, (0, pad)).view(-1, hop).pow(2).mean(1).sqrt()
    conv_rms = F.pad(converted, (0, pad)).view(-1, hop).pow(2).mean(1).sqrt()
    win = max(3, int(1.2 / 0.02)) | 1
    orig_sm = F.avg_pool1d(orig_rms[None, None], win, stride=1, padding=win // 2)[0, 0]
    conv_sm = F.avg_pool1d(conv_rms[None, None], win, stride=1, padding=win // 2)[0, 0]
    local_gain = (orig_sm + 1e-5) / (conv_sm + 1e-5)
    local_gain = local_gain.clamp(global_gain * 0.45, global_gain * 2.2)
    # Blend 65% local dynamics + 35% global level for natural, pump-free phrasing.
    blended = 0.65 * local_gain + 0.35 * global_gain
    gain_curve = F.interpolate(blended[None, None], size=converted.numel(), mode="linear", align_corners=False)[0, 0]

    # Soft gate: open over 12 dB above the floor, smoothed ~35 ms so syllable edges are kept.
    gate = ((env - floor) / 12.0).clamp(0.0, 1.0)
    k = max(1, int(sr * 0.035)) | 1
    gate = F.max_pool1d(gate[None, None], k, stride=1, padding=k // 2)[0, 0]
    gate = F.avg_pool1d(gate[None, None], k, stride=1, padding=k // 2)[0, 0][:converted.numel()]
    return converted * gain_curve * gate


_REVERB_IR_CACHE = {}


def _biquad_peaking(freq, sr, gain_db, q=1.0):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    alpha = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * c
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * c
    a2 = 1.0 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _biquad_high_shelf(freq, sr, gain_db, slope=0.75):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq / sr
    c = math.cos(w0)
    s = math.sin(w0)
    alpha = s / 2.0 * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    sq = 2.0 * math.sqrt(A) * alpha
    b0 = A * ((A + 1.0) + (A - 1.0) * c + sq)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * c)
    b2 = A * ((A + 1.0) + (A - 1.0) * c - sq)
    a0 = (A + 1.0) - (A - 1.0) * c + sq
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * c)
    a2 = (A + 1.0) - (A - 1.0) * c - sq
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _stereo_vocal_ir(sr, decay_s=0.85, predelay_ms=22.0):
    """Synthesize a clean, damped stereo plate/room impulse response for vocals."""
    key = (int(sr), round(decay_s, 3), round(predelay_ms, 1))
    cached = _REVERB_IR_CACHE.get(key)
    if cached is not None:
        return cached
    from scipy import signal
    length = max(256, int(sr * (decay_s + predelay_ms / 1000.0 + 0.1)))
    rng = np.random.default_rng(20260417)
    ir = np.zeros((2, length), dtype=np.float32)

    # Decorrelated stereo early reflections (20-58 ms) for subtle width
    early = [
        (0.022, 0.32, 0.16), (0.029, 0.14, 0.28),
        (0.037, 0.22, 0.11), (0.046, 0.10, 0.20),
        (0.055, 0.14, 0.13),
    ]
    for sec, gl, gr in early:
        idx = int(sec * sr)
        if idx < length:
            ir[0, idx] += gl
            ir[1, idx] += gr

    # Smooth exponentially damped diffuse stereo tail starting after pre-delay
    pre = int(sr * predelay_ms / 1000.0)
    tail_len = max(1, length - pre)
    t = np.arange(tail_len, dtype=np.float32) / float(sr)
    # RT60 ~ decay_s -> amplitude decays by 60 dB (-6.91 in ln) over decay_s
    env = np.exp(-6.91 * t / max(0.15, decay_s))
    # Smooth 20 ms fade-in so tail blooms gently behind the dry vocal
    attack = min(tail_len, max(1, int(sr * 0.02)))
    env[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
    noise = rng.standard_normal((2, tail_len)).astype(np.float32) * env[None, :]
    ir[:, pre:] += noise * 0.22

    # Band-limit reverb (260 Hz - 5.5 kHz) so bass never muddies and sibilants never hiss
    lo = min(0.45, 260.0 / (0.5 * sr))
    hi = min(0.95, 5500.0 / (0.5 * sr))
    if hi > lo > 0.001:
        sos = signal.butter(2, [lo, hi], btype="bandpass", output="sos")
        ir = signal.sosfilt(sos, ir, axis=-1).astype(np.float32)

    # Normalize total IR energy so reverb_amount is predictable (0.18 = subtle studio glue)
    energy = float(np.sqrt(np.mean(np.sum(ir ** 2, axis=-1)))) + 1e-8
    ir = ir / energy
    _REVERB_IR_CACHE[key] = ir
    return ir


def _polish_vocals(voc_stereo, sr, reverb_amount=0.18, eq_and_comp=True):
    """Studio vocal chain so dry mono Seed-VC vocals sit naturally inside a stereo mix:
    optional 85 Hz high-pass -> subtle clarity EQ -> gentle 2.5:1 compression -> optional stereo plate reverb.
    """
    from scipy import signal
    x = voc_stereo.detach().float().cpu().numpy()
    if x.shape[-1] < 64 or float(np.max(np.abs(x))) < 1e-6:
        return voc_stereo

    if eq_and_comp:
        nyq = 0.5 * sr
        # 1. High-pass at 85 Hz (zero-phase) to clean sub-bass rumble / plosives
        if 85.0 < nyq * 0.8:
            sos_hp = signal.butter(2, 85.0 / nyq, btype="highpass", output="sos")
            x = signal.sosfiltfilt(sos_hp, x, axis=-1)

        # 2. Gentle clarity EQ: slight low-mid cleanup (-1.2 dB @ 280 Hz) + air/presence (+1.8 dB @ 4.5 kHz)
        if 4500.0 < nyq * 0.85:
            sos_eq = np.vstack([
                _biquad_peaking(280.0, sr, -1.2, q=1.0),
                _biquad_high_shelf(4500.0, sr, 1.8, slope=0.75),
            ])
            x = signal.sosfilt(sos_eq, x, axis=-1)

        # 3. Gentle soft-knee vocal compressor (2.5:1 above -18 dB RMS, matched active RMS makeup)
        mono = x.mean(axis=0)
        frame = max(1, int(sr * 0.01))
        pad = (-mono.size) % frame
        rms = np.sqrt(np.mean(np.pad(mono, (0, pad)).reshape(-1, frame) ** 2, axis=1) + 1e-10)
        db = 20.0 * np.log10(rms + 1e-9)
        # Smooth envelope (~20 ms attack, ~140 ms release)
        sm = np.empty_like(db)
        sm[0] = db[0]
        a_att = math.exp(-0.01 / 0.020)
        a_rel = math.exp(-0.01 / 0.140)
        for i in range(1, db.size):
            coeff = a_att if db[i] > sm[i - 1] else a_rel
            sm[i] = coeff * sm[i - 1] + (1.0 - coeff) * db[i]
        thresh, ratio = -18.0, 2.5
        over = np.maximum(0.0, sm - thresh)
        gain_db = -over * (1.0 - 1.0 / ratio)
        gain_lin = 10.0 ** (gain_db / 20.0)
        gain_full = np.interp(np.arange(mono.size), np.arange(gain_lin.size) * frame, gain_lin).astype(np.float32)
        comp = (x * gain_full[None, :]).astype(np.float32)
        # Preserve pre-compression active RMS so vocal level stays true to _follow_original
        active = db > (np.percentile(db, 95) - 30.0)
        if active.any():
            active_full = np.repeat(active, frame)[:mono.size]
            pre_rms = float(np.sqrt(np.mean(x[:, active_full] ** 2))) + 1e-8
            post_rms = float(np.sqrt(np.mean(comp[:, active_full] ** 2))) + 1e-8
            comp *= min(1.8, pre_rms / post_rms)
        x = comp

    # 4. Subtle stereo room/plate reverb so vocals blend into the stereo track instead of sounding pasted on
    rev = max(0.0, min(0.7, float(reverb_amount or 0.0)))
    if rev > 0.005:
        ir = _stereo_vocal_ir(sr)
        wet_l = signal.fftconvolve(x[0], ir[0], mode="full")[: x.shape[-1]]
        wet_r = signal.fftconvolve(x[1], ir[1], mode="full")[: x.shape[-1]]
        wet = np.stack([wet_l, wet_r], axis=0).astype(np.float32)
        x = x + wet * rev

    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))


# ---------------------------------------------------------------- downloads

def _hf_file(repo, filename, folder):
    path = os.path.join(MODEL_DIR, folder, filename)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        print(f"[SongVoice] downloading {repo}/{filename}")
        hf_hub_download(repo, filename, local_dir=os.path.join(MODEL_DIR, folder))
    return path


def _ensure_seed_vc_source():
    if os.path.exists(os.path.join(SEED_VC_DIR, "modules", "commons.py")):
        return
    print(f"[SongVoice] fetching Seed-VC source into {SEED_VC_DIR}")
    subprocess.run(["git", "clone", SEED_VC_REPO, SEED_VC_DIR], check=True)
    subprocess.run(["git", "-C", SEED_VC_DIR, "checkout", "-q", SEED_VC_COMMIT], check=True)


@contextlib.contextmanager
def _trusted_torch_load():
    """Seed-VC / RMVPE / BigVGAN checkpoints are pickles from known repos; torch>=2.6 refuses them by default."""
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def _import_seed_vc():
    """Put Seed-VC's top-level `modules` package on sys.path.

    Its length regulator imports descript-audio-codec only for the vector
    quantizer, which the singing model never builds (vector_quantize: false).
    That package drags in a large, pinned dependency tree, so stub it instead.
    """
    _ensure_seed_vc_source()
    existing = sys.modules.get("modules")
    if existing is not None and not os.path.abspath(getattr(existing, "__file__", "") or "").startswith(SEED_VC_DIR):
        raise RuntimeError("[SongVoice] another custom node already owns the top-level 'modules' package; "
                           "Seed-VC cannot be imported alongside it.")
    if SEED_VC_DIR not in sys.path:
        sys.path.insert(0, SEED_VC_DIR)
    try:
        import dac.nn.quantize  # noqa: F401
    except Exception:
        class VectorQuantize(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("descript-audio-codec is not installed; this Seed-VC model needs it.")
        for name in ("dac", "dac.nn", "dac.nn.quantize"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["dac.nn.quantize"].VectorQuantize = VectorQuantize


def _resolve_pitch_shift(voiced_ref, voiced_src, pitch_mode, pitch_shift=0.0, ref_pitch_offset=0.0):
    """Choose semitone shift from source singing F0 to target voice F0.

    In 'octave to my range' mode, only jump by +/-12 semitones when the gap is
    genuinely near a full octave (e.g. male <-> female, |gap| >= 9.0 st) AND
    shifting down keeps the vocal median above a clear singable floor (~115 Hz).
    This prevents deep baritone talking references (~90 Hz) from dropping normal
    male singing (~190-220 Hz) a full octave into slurred sub-bass rumble.
    """
    semitones = float(pitch_shift)
    if pitch_mode == "keep song key" or not voiced_ref.numel() or not voiced_src.numel():
        return semitones
    ref_med = float(torch.median(voiced_ref).item())
    src_med = float(torch.median(voiced_src).item())
    if ref_med <= 0.0 or src_med <= 0.0:
        return semitones
    gap = 12.0 * (math.log2(ref_med) - math.log2(src_med)) + float(ref_pitch_offset)
    if pitch_mode == "octave to my range":
        octaves = 0
        if gap <= -9.0:
            octaves = int(round(gap / 12.0))
            while octaves < 0 and (src_med * (2.0 ** octaves)) < 115.0:
                octaves += 1
        elif gap >= 9.0:
            octaves = int(round(gap / 12.0))
        semitones += 12.0 * octaves
    else:
        semitones += gap
    return semitones


# ---------------------------------------------------------------- Seed-VC

class SeedVC:
    """Seed-VC v1 singing voice conversion (44.1 kHz, f0-conditioned DiT + BigVGAN)."""

    def __init__(self, device):
        _import_seed_vc()
        import yaml
        from modules.commons import build_model, load_checkpoint, recursive_munch
        from modules.audio import mel_spectrogram
        from modules.rmvpe import RMVPE
        from modules.campplus.DTDNN import CAMPPlus
        from modules.bigvgan import bigvgan
        from transformers import AutoFeatureExtractor, WhisperModel

        self.device = device
        paths = {folder: [_hf_file(repo, f, folder) for f in files] for repo, files, folder in SEED_VC_FILES}
        dit_path, config_path = paths["Seed-VC"]

        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        params = recursive_munch(config["model_params"])
        params.dit_type = "DiT"
        spect = config["preprocess_params"]["spect_params"]
        self.sr = config["preprocess_params"]["sr"]
        self.hop = spect["hop_length"]

        self.params = params
        self.base_ckpt = dit_path
        self.voice = None
        self._base_state = None
        with _trusted_torch_load():
            model = build_model(params, stage="DiT")
            model, _, _, _ = load_checkpoint(model, None, dit_path, load_only_params=True,
                                             ignore_modules=[], is_distributed=False)
            for key in model:
                model[key].eval().to(device)
            model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
            self.model = model

            self.rmvpe = RMVPE(paths["rmvpe"][0], is_half=False, device=device)

            campplus = CAMPPlus(feat_dim=80, embedding_size=192)
            campplus.load_state_dict(torch.load(paths["campplus"][0], map_location="cpu"))
            self.campplus = campplus.eval().to(device)

            vocoder = bigvgan.BigVGAN.from_pretrained(os.path.join(MODEL_DIR, "bigvgan_v2_44khz_128band_512x"),
                                                      use_cuda_kernel=False)
            vocoder.remove_weight_norm()
            self.vocoder = vocoder.eval().to(device)

        whisper_dir = os.path.join(MODEL_DIR, "whisper-small")
        whisper = WhisperModel.from_pretrained(whisper_dir, torch_dtype=torch.float16).to(device)
        del whisper.decoder
        self.whisper = whisper.eval()
        self.whisper_features = AutoFeatureExtractor.from_pretrained(whisper_dir)

        mel_args = {
            "n_fft": spect["n_fft"], "win_size": spect["win_length"], "hop_size": spect["hop_length"],
            "num_mels": spect["n_mels"], "sampling_rate": self.sr, "fmin": spect.get("fmin", 0),
            "fmax": None if spect.get("fmax", "None") == "None" else 8000, "center": False,
        }
        self.to_mel = lambda x: mel_spectrogram(x, **mel_args)

    def to(self, device):
        for key in self.model:
            self.model[key].to(device)
        for module in (self.campplus, self.vocoder, self.whisper, self.rmvpe.model, self.rmvpe.mel_extractor):
            module.to(device)
        self.rmvpe.device = device
        self.device = device
        return self

    def set_voice(self, path, strength=1.0):
        """Swap the DiT weights: stock model blended toward a trained voice checkpoint.

        strength 0 = stock zero-shot model, 1 = fully fine-tuned. In between keeps
        most of the likeness while undoing the over-fitting that blurs lyrics.
        """
        key_ = (path, round(strength, 3)) if path else None
        if key_ == self.voice:
            return
        if self._base_state is None:
            self._base_state = {k: {n: t.detach().cpu().clone() for n, t in self.model[k].state_dict().items()}
                                for k in self.model}
        if path is None or strength <= 0:
            target = self._base_state
        else:
            with _trusted_torch_load():
                tuned = torch.load(path, map_location="cpu")["net"]
            target = {}
            for k, base in self._base_state.items():
                target[k] = {}
                for n, t in base.items():
                    ft = tuned.get(k, {}).get(n)
                    if ft is None or ft.shape != t.shape or not t.is_floating_point():
                        target[k][n] = t
                    else:
                        target[k][n] = t + (ft.float() - t) * strength
        for k in self.model:
            self.model[k].load_state_dict(target[k], strict=False)
            self.model[k].eval()
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.voice = key_

    def style_of(self, wave):
        """CAM++ speaker embedding of a whole clip (1-D at self.sr) -- uses all of it, not just the prompt."""
        w16 = torchaudio.functional.resample(wave[None].to(self.device), self.sr, 16000)
        fbank = torchaudio.compliance.kaldi.fbank(w16, num_mel_bins=80, dither=0, sample_frequency=16000)
        return self.campplus((fbank - fbank.mean(dim=0, keepdim=True)).unsqueeze(0))

    def _semantic(self, waves_16k):
        inputs = self.whisper_features([waves_16k.squeeze(0).cpu().numpy()], return_tensors="pt",
                                       return_attention_mask=True, sampling_rate=16000)
        features = self.whisper._mask_input_features(inputs.input_features,
                                                     attention_mask=inputs.attention_mask).to(self.device)
        out = self.whisper.encoder(features.to(self.whisper.encoder.dtype), return_dict=True)
        return out.last_hidden_state.float()[:, :waves_16k.size(-1) // 320 + 1]

    def _semantic_long(self, waves_16k):
        # Whisper sees 30 s at a time; stitch with 5 s of overlap, exactly as upstream.
        window, overlap = 16000 * 30, 5
        if waves_16k.size(-1) <= window:
            return self._semantic(waves_16k)
        parts, buffer, done = [], None, 0
        while done < waves_16k.size(-1):
            if buffer is None:
                chunk = waves_16k[:, done:done + window]
            else:
                chunk = torch.cat([buffer, waves_16k[:, done:done + 16000 * (30 - overlap)]], dim=-1)
            s = self._semantic(chunk)
            parts.append(s if done == 0 else s[:, 50 * overlap:])
            buffer = chunk[:, -16000 * overlap:]
            done += window if done == 0 else chunk.size(-1) - 16000 * overlap
        return torch.cat(parts, dim=1)

    @torch.no_grad()
    def convert(self, source, reference, steps, cfg_rate, pitch_mode, pitch_shift, fp16, pbar=None,
                style=None, ref_pitch_offset=0.0):
        """source / reference: 1-D float tensors at self.sr. Returns (1-D tensor at self.sr, semitones applied).

        style: precomputed speaker embedding (defaults to the reference's).
        ref_pitch_offset: semitones added to the reference's median pitch before
        comparing ranges -- people sing higher than they talk.
        """
        device = self.device
        source = source[None].to(device)
        reference = reference[None].to(device)

        src_16k = torchaudio.functional.resample(source, self.sr, 16000)
        ref_16k = torchaudio.functional.resample(reference, self.sr, 16000)
        s_alt = self._semantic_long(src_16k)
        s_ori = self._semantic(ref_16k)

        mel = self.to_mel(source.float())
        mel2 = self.to_mel(reference.float())
        target_len = torch.LongTensor([mel.size(2)]).to(device)
        prompt_len = torch.LongTensor([mel2.size(2)]).to(device)

        if style is None:
            style = self.style_of(reference[0])

        f0_ref = torch.from_numpy(self.rmvpe.infer_from_audio(ref_16k[0], thred=0.03)).to(device)[None]
        f0_src = torch.from_numpy(self.rmvpe.infer_from_audio(src_16k[0], thred=0.03)).to(device)[None]
        voiced_ref, voiced_src = f0_ref[f0_ref > 1], f0_src[f0_src > 1]
        semitones = _resolve_pitch_shift(voiced_ref, voiced_src, pitch_mode, pitch_shift, ref_pitch_offset)
        f0_alt = f0_src.clone()
        f0_alt[f0_src > 1] = f0_src[f0_src > 1] * (2.0 ** (semitones / 12.0))

        cond = self.model.length_regulator(s_alt, ylens=target_len, n_quantizers=3, f0=f0_alt)[0]
        prompt = self.model.length_regulator(s_ori, ylens=prompt_len, n_quantizers=3, f0=f0_ref)[0]

        # The DiT attends over [your clip + source chunk] within a 30 s window,
        # so each source chunk is (30 s - clip length). Chunks crossfade over 16 frames.
        overlap_frames = 16
        overlap_wave = overlap_frames * self.hop
        window = self.sr // self.hop * 30 - mel2.size(2)
        total = cond.size(1)
        fade_out = torch.cos(torch.linspace(0, math.pi / 2, overlap_wave)) ** 2
        fade_in = torch.cos(torch.linspace(math.pi / 2, 0, overlap_wave)) ** 2
        pieces, previous, done = [], None, 0
        while done < total:
            mm.throw_exception_if_processing_interrupted()
            chunk = cond[:, done:done + window]
            last = done + window >= total
            joined = torch.cat([prompt, chunk], dim=1)
            with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
                out = self.model.cfm.inference(joined, torch.LongTensor([joined.size(1)]).to(device),
                                               mel2, style, None, steps, inference_cfg_rate=cfg_rate)
                out = out[:, :, mel2.size(-1):]
            wave = self.vocoder(out.float()).squeeze().float().cpu()
            if previous is not None:
                head = wave[:overlap_wave]
                wave[:head.numel()] = head * fade_in[:head.numel()] + previous[:head.numel()] * fade_out[:head.numel()]
            if last:
                pieces.append(wave)
            else:
                pieces.append(wave[:-overlap_wave])
                previous = wave[-overlap_wave:]
            done += out.size(2) - overlap_frames
            if pbar is not None:
                pbar.update_absolute(min(done, total), total)
            if last:
                break
        return torch.cat(pieces), semitones


def _free_vram_for(need_gb):
    """Make room on the GPU. ComfyUI-managed models go first; then song generators
    that keep their own cache outside ComfyUI (ComfyUI-YuE2's legacy pipeline)
    are closed, but only if that is still not enough -- they reload next run."""
    if not torch.cuda.is_available():
        return
    free = lambda: torch.cuda.mem_get_info()[0] / 2**30
    if free() >= need_gb:
        return
    mm.unload_all_models()
    mm.soft_empty_cache()
    if free() >= need_gb:
        return
    for name, module in list(sys.modules.items()):
        if name.endswith("ComfyUI-YuE2.models.yue2") and getattr(module, "_cache", {}).get("pipeline") is not None:
            print("[SongVoice] releasing the cached YuE2 pipeline to make room")
            module.close()
    import gc
    gc.collect()
    mm.soft_empty_cache()


def _seed_vc():
    device = mm.get_torch_device()
    model = _cache.get("seed_vc")
    if model is None:
        model = _cache["seed_vc"] = SeedVC(device)
    elif model.device != device:
        model.to(device)
    return model


def _roformer(kind):
    """Cached BS-RoFormer ('vocals') or Mel-Band RoFormer ('karaoke') separator."""
    from . import roformer
    key = f"roformer_{kind}"
    entry = _cache.get(key)
    device = mm.get_torch_device()
    if entry is None:
        model, config = roformer.load_roformer_model(kind, MODEL_DIR, device)
        entry = _cache[key] = (model, config)
    else:
        entry[0].to(device)
    return entry


def _hdemucs():
    model = _cache.get("hdemucs")
    if model is None:
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        path = os.path.join(MODEL_DIR, "hdemucs_high_trained.pt")
        if not os.path.exists(path):
            os.makedirs(MODEL_DIR, exist_ok=True)
            print("[SongVoice] downloading Hybrid Demucs weights")
            torch.hub.download_url_to_file(HDEMUCS_URL, path)
        model = bundle._model_factory_func()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model = _cache["hdemucs"] = model.eval()
    return model


# ---------------------------------------------------------------- nodes

class SongVoiceSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO", {"tooltip": "The finished song."}),
            "segment_seconds": ("FLOAT", {"default": 10.0, "min": 4.0, "max": 30.0, "step": 1.0,
                                          "tooltip": "Window length when falling back to Hybrid Demucs."}),
        }}

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("vocals", "instrumental")
    FUNCTION = "split"
    CATEGORY = "audio/voice"
    DESCRIPTION = "Separate a song into vocals and instrumental (BS-RoFormer, with Hybrid Demucs fallback)."

    @torch.no_grad()
    def _split_demucs(self, wave, sr, segment_seconds):
        _free_vram_for(2.0)
        device = mm.get_torch_device()
        model = _hdemucs().to(device)
        mix = _stereo(_resample(wave, sr, 44100))

        # Standardise, then overlap-add windows with linear fades (torchaudio's HDemucs recipe).
        ref = mix.mean(0)
        mean, std = ref.mean(), ref.std() + 1e-8
        mix_n = (mix - mean) / std
        length = mix_n.shape[-1]
        overlap = int(44100 * 0.1)
        seg = int(44100 * segment_seconds) + overlap
        fade = torchaudio.transforms.Fade(fade_in_len=0, fade_out_len=overlap, fade_shape="linear")
        out = torch.zeros(len(model.sources), 2, length)
        start, end = 0, seg
        pbar = comfy.utils.ProgressBar(length)
        while start < length - overlap:
            mm.throw_exception_if_processing_interrupted()
            chunk = mix_n[:, start:end][None].to(device)
            est = fade(model(chunk))[0].cpu()
            out[:, :, start:start + est.shape[-1]] += est
            if start == 0:
                fade.fade_in_len = overlap
                start += seg - overlap
            else:
                start += seg
            end += seg
            if end >= length:
                fade.fade_out_len = 0
            pbar.update_absolute(min(start, length), length)
        out = out * std + mean
        model.to("cpu")
        mm.soft_empty_cache()

        vocals_idx = model.sources.index("vocals")
        vocals = out[vocals_idx]
        instrumental = out.sum(0) - vocals
        return (_audio_out(_resample(vocals, 44100, sr), sr),
                _audio_out(_resample(instrumental, 44100, sr), sr))

    @torch.no_grad()
    def split(self, audio, segment_seconds=10.0, separator="roformer"):
        wave, sr = _audio_in(audio)
        self.last_separator = "demucs"
        self.last_fallback_reason = ""
        if separator == "roformer":
            try:
                from . import roformer
                _free_vram_for(3.0)
                device = mm.get_torch_device()
                model, config = _roformer("vocals")
                mix = _stereo(_resample(wave, sr, 44100))
                pbar = comfy.utils.ProgressBar(100)
                try:
                    vocals, instrumental = roformer.separate_roformer(
                        model, config, mix, device=device, num_overlap=2,
                        pbar=pbar, check_cancel=mm.throw_exception_if_processing_interrupted)
                finally:
                    model.to("cpu")
                    mm.soft_empty_cache()
                self.last_separator = "roformer"
                return (_audio_out(_resample(vocals, 44100, sr), sr),
                        _audio_out(_resample(instrumental, 44100, sr), sr))
            except Exception as exc:
                # Propagate explicit user cancellations; only fall back on load/download failures.
                if "Interrupt" in type(exc).__name__ or "Cancelled" in type(exc).__name__:
                    raise
                self.last_fallback_reason = str(exc)
                print(f"[SongVoice] BS-RoFormer unavailable ({exc}); falling back to Hybrid Demucs", flush=True)
        return self._split_demucs(wave, sr, segment_seconds)

    @torch.no_grad()
    def split_lead(self, vocals):
        """Split a vocal stem into (lead_vocals, backing_vocals) using Mel-Band RoFormer Karaoke.

        Keeping chorus harmonies and backing layers out of Seed-VC prevents the
        monophonic pitch tracker (RMVPE) from collapsing during dense choruses.
        """
        wave, sr = _audio_in(vocals)
        self.last_lead_separator = "none"
        self.last_lead_fallback_reason = ""
        try:
            from . import roformer
            _free_vram_for(3.0)
            device = mm.get_torch_device()
            model, config = _roformer("karaoke")
            mix = _stereo(_resample(wave, sr, 44100))
            pbar = comfy.utils.ProgressBar(100)
            try:
                lead, backing = roformer.separate_roformer(
                    model, config, mix, device=device, num_overlap=2,
                    pbar=pbar, check_cancel=mm.throw_exception_if_processing_interrupted)
            finally:
                model.to("cpu")
                mm.soft_empty_cache()
            self.last_lead_separator = "karaoke"
            return (_audio_out(_resample(lead, 44100, sr), sr),
                    _audio_out(_resample(backing, 44100, sr), sr))
        except Exception as exc:
            if "Interrupt" in type(exc).__name__ or "Cancelled" in type(exc).__name__:
                raise
            self.last_lead_fallback_reason = str(exc)
            print(f"[SongVoice] Karaoke lead/backing split unavailable ({exc}); using full vocals", flush=True)
            return vocals, None


def _voice_names():
    if not os.path.isdir(VOICE_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(VOICE_DIR) if f.endswith(".pth"))


def _safe_name(name):
    name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or "").strip()).strip()
    return name or "my_voice"


def _voice(name):
    """SONG_VOICE: a trained checkpoint plus the cleaned clip it was trained on."""
    return {"name": name, "ckpt": os.path.join(VOICE_DIR, name + ".pth"), "clip": os.path.join(VOICE_DIR, name + ".wav")}


# Seed-VC's example clips, fetched from the pinned upstream commit on first voice
# training instead of being redistributed with this app.
SEED_VC_EXAMPLES = {
    "reference": ["azuma_0.wav", "dingzhen_0.wav", "s1p1.wav", "s1p2.wav", "s2p1.wav", "s2p2.wav",
                  "s3p1.wav", "s3p2.wav", "s4p1.wav", "s4p2.wav", "teio_0.wav", "trump_0.wav"],
    "source": ["TECHNOPOLIS - 2085 [vocals]_[cut_14sec].wav",
               "Wiz Khalifa,Charlie Puth - See You Again [vocals]_[cut_28sec].wav",
               "glados_0.wav", "jay_0.wav", "source_s1.wav", "source_s2.wav", "source_s3.wav",
               "source_s4.wav", "yae_0.wav"],
}


def _ensure_seed_vc_examples():
    import urllib.parse
    import urllib.request
    base = SEED_VC_REPO.replace("https://github.com/", "https://raw.githubusercontent.com/")
    for folder, names in SEED_VC_EXAMPLES.items():
        for name in names:
            path = os.path.join(SEED_VC_DIR, "examples", folder, name)
            if os.path.isfile(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = f"{base}/{SEED_VC_COMMIT}/examples/{folder}/{urllib.parse.quote(name)}"
            print(f"[SongVoice] downloading Seed-VC example {folder}/{name}", flush=True)
            try:
                urllib.request.urlretrieve(url, path + ".part")
                os.replace(path + ".part", path)
            except OSError as exc:            # offline: train with whatever is already here
                print(f"[SongVoice] could not download {name}: {exc}", flush=True)
                break


def _replay_pieces(sr):
    """Other people's speech and singing (bundled with Seed-VC) mixed into training batches."""
    import glob
    import librosa
    from . import voice_trainer
    _ensure_seed_vc_examples()
    pieces = []
    for path in sorted(glob.glob(os.path.join(SEED_VC_DIR, "examples", "*", "*.wav"))):
        wave = torch.from_numpy(librosa.load(path, sr=sr)[0])
        if wave.abs().max() > 0:
            pieces += voice_trainer.chop_clip(wave / wave.abs().max() * 0.95, sr)
    return pieces


def _clip_fingerprint(wave, sr):
    import hashlib
    data = _resample(wave, sr, 16000).mean(0).numpy().astype(np.float16).tobytes()
    return hashlib.sha1(data).hexdigest()


class SongVoiceTrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "my_voice": ("AUDIO", {"tooltip": "You talking (or singing). 30 s to a few minutes, one voice, no music. "
                                              "Pauses are trimmed automatically."}),
            "voice_name": ("STRING", {"default": "my_voice", "tooltip": "Saved as voices/<name>.pth"}),
            "training_steps": ("INT", {"default": DEFAULT_TRAIN_STEPS, "min": 50, "max": 5000, "step": 50,
                                       "tooltip": "More steps = closer likeness, but too many can blur the words."}),
            "retrain": ("BOOLEAN", {"default": False,
                                    "tooltip": "Train again even if this voice already exists for the same clip."}),
        }}

    RETURN_TYPES = ("SONG_VOICE",)
    RETURN_NAMES = ("trained_voice",)
    FUNCTION = "train"
    CATEGORY = "audio/voice"
    DESCRIPTION = ("Fine-tune the singing model on your voice (one-time, a few minutes). "
                   "Later runs reuse the saved voice instantly unless the clip changes.")

    def train(self, my_voice, voice_name, training_steps, retrain):
        import json
        from . import voice_trainer
        name = _safe_name(voice_name)
        voice = _voice(name)
        meta_path = os.path.join(VOICE_DIR, name + ".json")
        wave, sr = _audio_in(my_voice)
        fingerprint = _clip_fingerprint(wave, sr)
        if not retrain and os.path.exists(voice["ckpt"]) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("clip") == fingerprint and meta.get("steps") == training_steps:
                print(f"[SongVoice] using trained voice '{name}'")
                return (voice,)

        model = _seed_vc()
        mono = _resample(wave.mean(0, keepdim=True), sr, model.sr)[0]
        if mono.abs().max() > 0:
            mono = mono / mono.abs().max() * 0.95
        pieces = voice_trainer.chop_clip(mono, model.sr)
        total = sum(p.numel() for p in pieces) / model.sr
        if total < 5:
            raise ValueError(f"[SongVoice] only {total:.1f}s of voice found in the clip; give it at least 10 s.")
        print(f"[SongVoice] training '{name}' on {total:.0f}s of voice ({len(pieces)} pieces), {training_steps} steps")

        _free_vram_for(10.0)
        pbar = comfy.utils.ProgressBar(training_steps)

        def on_step(i, loss):
            mm.throw_exception_if_processing_interrupted()
            pbar.update_absolute(i, training_steps)
            if i % 50 == 0:
                print(f"[SongVoice] step {i}/{training_steps}  loss {loss:.3f}")

        for repo, files, folder in TRAIN_FILES:
            for f in files:
                _hf_file(repo, f, folder)
        # ComfyUI runs every node under torch.inference_mode(), which forbids
        # autograd. Leave it for the training block, and rebuild the audio here:
        # tensors created under inference mode can never take part in backward.
        trainer = None
        with torch.inference_mode(False), torch.enable_grad():
            train_pieces = [p.clone() for p in pieces]
            replay = [p.clone() for p in _replay_pieces(model.sr)]
            try:
                with _trusted_torch_load():
                    trainer = voice_trainer.VoiceTrainer(model, model.base_ckpt, OPENVOICE_DIR, SE_DB_PATH)
                # Measured on a 13 s talking clip: freezing the content regulator and
                # mixing other voices into every batch kept lyrics far clearer than the
                # upstream recipe at the same likeness (see README).
                trainer.train(train_pieces, training_steps, batch_size=2, on_step=on_step,
                              replay=replay, freeze_regulator=True)
                trainer.save(voice["ckpt"])
            finally:
                del trainer
                mm.soft_empty_cache()
        import soundfile as sf
        sf.write(voice["clip"], torch.cat(pieces).numpy(), model.sr)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({"clip": fingerprint, "steps": training_steps, "seconds": round(total, 1)}, fh)
        model.voice = "stale"   # force a reload: the file on disk may have been replaced
        return (voice,)


class SongVoiceLoad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"voice_name": (_voice_names() or ["(train a voice first)"],)}}

    RETURN_TYPES = ("SONG_VOICE",)
    RETURN_NAMES = ("trained_voice",)
    FUNCTION = "load"
    CATEGORY = "audio/voice"
    DESCRIPTION = "Pick a voice you already trained with Train My Voice."

    def load(self, voice_name):
        voice = _voice(voice_name)
        if not os.path.exists(voice["ckpt"]):
            raise ValueError(f"[SongVoice] no trained voice named '{voice_name}' in {VOICE_DIR}")
        return (voice,)


class SongVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vocals": ("AUDIO", {"tooltip": "Isolated vocals to re-sing (from Split Vocals)."}),
            "my_clip_is": (["talking", "singing"], {"default": "talking",
                                                    "tooltip": "Talking clips sit lower than a singing voice; this "
                                                               "corrects for it when choosing the octave."}),
            "diffusion_steps": ("INT", {"default": 30, "min": 4, "max": 100,
                                        "tooltip": "30-50 for singing. More = cleaner, slower."}),
            "cfg_rate": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "How hard to push toward your timbre."}),
            "pitch_mode": (PITCH_MODES, {"default": "octave to my range",
                                         "tooltip": "keep song key: melody untouched. octave to my range: move the "
                                                    "melody whole octaves toward your voice (safe for male<->female). "
                                                    "exact to my range: transpose to your median pitch (may change key)."}),
            "pitch_shift": ("INT", {"default": 0, "min": -24, "max": 24,
                                    "tooltip": "Extra semitones on top of pitch_mode."}),
            "reference_seconds": ("FLOAT", {"default": 20.0, "min": 3.0, "max": 25.0, "step": 1.0,
                                            "tooltip": "Longer = better likeness, but shorter conversion windows."}),
            "follow_original": ("BOOLEAN", {"default": True,
                                            "tooltip": "Match the original vocal loudness and stay silent where the "
                                                       "song has no singing (removes humming in intros/solos)."}),
            "fp16": ("BOOLEAN", {"default": True}),
            "keep_loaded": ("BOOLEAN", {"default": False,
                                        "tooltip": "Keep Seed-VC (~2 GB) on the GPU between runs."}),
        }, "optional": {
            "trained_voice": ("SONG_VOICE", {"tooltip": "From Train My Voice / Load My Voice. Much closer likeness "
                                                        "than the clip alone."}),
            "my_voice": ("AUDIO", {"tooltip": "A clip of you. Optional when trained_voice is connected (its "
                                              "training clip is used)."}),
            "voice_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                         "tooltip": "How much of the trained voice to use. Lower it (0.5-0.7) if "
                                                    "words come out mumbled; 1.0 = most like you."}),
        }}

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("vocals", "semitones")
    FUNCTION = "clone"
    CATEGORY = "audio/voice"
    DESCRIPTION = "Re-sing vocals in your voice (Seed-VC singing conversion, optionally fine-tuned on you)."

    def clone(self, vocals, my_clip_is, diffusion_steps, cfg_rate, pitch_mode, pitch_shift,
              reference_seconds, follow_original, fp16, keep_loaded, trained_voice=None, my_voice=None,
              voice_strength=1.0):
        from . import voice_trainer
        src, src_sr = _audio_in(vocals)
        _free_vram_for(4.0)
        model = _seed_vc()
        sr = model.sr
        if my_voice is not None:
            ref, ref_sr = _audio_in(my_voice)
            ref_mono = _resample(ref.mean(0, keepdim=True), ref_sr, sr)[0]
        elif trained_voice is not None and os.path.exists(trained_voice["clip"]):
            import soundfile as sf
            data, ref_sr = sf.read(trained_voice["clip"], dtype="float32")
            ref_mono = _resample(torch.from_numpy(data)[None], ref_sr, sr)[0]
        else:
            raise ValueError("[SongVoice] connect my_voice, trained_voice, or both.")
        if ref_mono.abs().max() > 0:
            ref_mono = ref_mono / ref_mono.abs().max() * 0.95
        # Pauses waste the 30 s attention window; keep only voice, and take the
        # speaker embedding from all of it rather than just the prompt slice.
        voiced = torch.cat(voice_trainer.chop_clip(ref_mono, sr) or [ref_mono])
        if voiced.numel() < sr * 2:
            raise ValueError("[SongVoice] less than 2 s of voice in the clip; give it 10-25 s.")
        prompt = voiced[:int(sr * reference_seconds)]

        model.set_voice(trained_voice["ckpt"] if trained_voice is not None else None, voice_strength)
        with torch.no_grad():
            style = model.style_of(voiced[:sr * 120])
        src_mono = _resample(src.mean(0, keepdim=True), src_sr, sr)[0]
        # Peak-normalize source vocal before Whisper/RMVPE/mel extraction so quiet
        # stems aren't starved and loud transients never clip > 1.0; _follow_original
        # restores the original phrasing dynamics right after conversion.
        src_peak = float(src_mono.abs().max().item()) if src_mono.numel() else 0.0
        src_in = (src_mono / src_peak * 0.95) if src_peak > 1e-5 else src_mono
        pbar = comfy.utils.ProgressBar(100)
        out, semitones = model.convert(src_in, prompt, diffusion_steps, cfg_rate, pitch_mode, pitch_shift, fp16,
                                       pbar=pbar, style=style,
                                       ref_pitch_offset=TALKING_PITCH_OFFSET if my_clip_is == "talking" else 0.0)
        if not keep_loaded:
            model.to(torch.device("cpu"))
            mm.soft_empty_cache()
        print(f"[SongVoice] converted {out.numel() / sr:.1f}s of vocals, pitch {semitones:+.1f} semitones")

        out = _fit(_resample(out[None], sr, src_sr), src.shape[-1])
        if follow_original:
            out = _follow_original(out[0], src.mean(0), src_sr)[None]
        out = out.repeat(2, 1)
        return (_audio_out(out, src_sr), semitones)


class SongVoiceMix:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vocals": ("AUDIO",),
            "instrumental": ("AUDIO",),
            "vocal_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5}),
            "instrumental_gain_db": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5}),
            "instrumental_semitones": ("FLOAT", {"default": 0.0, "min": -24.0, "max": 24.0, "step": 1.0,
                                                 "tooltip": "Wire Voice Clone's 'semitones' here when using "
                                                            "'exact to my range' so the band follows your key. "
                                                            "Whole octaves are ignored (the key is unchanged)."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "mix"
    CATEGORY = "audio/voice"
    DESCRIPTION = "Put the new vocals back on the instrumental, peak-safe."

    def mix(self, vocals, instrumental, vocal_gain_db=0.0, instrumental_gain_db=0.0,
            instrumental_semitones=0.0, backing_vocals=None, backing_gain_db=-1.5,
            polish=True, reverb_amount=0.18):
        inst, sr = _audio_in(instrumental)
        voc, voc_sr = _audio_in(vocals)
        inst = _stereo(inst)
        shift = instrumental_semitones - 12.0 * round(instrumental_semitones / 12.0)
        if abs(shift) >= 0.01:
            shifted = torchaudio.functional.pitch_shift(inst, sr, n_steps=int(round(shift)))
            inst = _fit(shifted, inst.shape[-1])
        voc = _fit(_stereo(_resample(voc, voc_sr, sr)), inst.shape[-1])
        if polish or float(reverb_amount or 0.0) > 0.005:
            voc = _polish_vocals(voc, sr, reverb_amount=reverb_amount, eq_and_comp=bool(polish))
        out = voc * _db(vocal_gain_db) + inst * _db(instrumental_gain_db)
        if backing_vocals is not None:
            back, back_sr = _audio_in(backing_vocals)
            back = _fit(_stereo(_resample(back, back_sr, sr)), inst.shape[-1])
            if abs(shift) >= 0.01:
                back = _fit(torchaudio.functional.pitch_shift(back, sr, n_steps=int(round(shift))), inst.shape[-1])
            out = out + back * _db(backing_gain_db)
        # Gentle soft-clipping bus limiter above 0.92 so single drum transients don't duck the whole mix
        knee = 0.92
        headroom = 0.07
        mag = out.abs()
        over = mag > knee
        if over.any():
            out = torch.where(over, out.sign() * (knee + headroom * torch.tanh((mag - knee) / headroom)), out)
        peak = out.abs().max().item()
        if peak > 0.99:
            out = out * (0.99 / peak)
        return (_audio_out(out, sr),)

    def combine_vocals(self, vocals, backing_vocals=None, vocal_gain_db=0.0,
                       backing_gain_db=-1.5, polish=True, reverb_amount=0.18,
                       backing_semitones=0.0):
        """Build the polished vocal stem (cloned lead + kept backing harmonies) for the Vocals-only sidecar."""
        voc, sr = _audio_in(vocals)
        voc = _stereo(voc)
        if polish or float(reverb_amount or 0.0) > 0.005:
            voc = _polish_vocals(voc, sr, reverb_amount=reverb_amount, eq_and_comp=bool(polish))
        out = voc * _db(vocal_gain_db)
        if backing_vocals is not None:
            back, back_sr = _audio_in(backing_vocals)
            back = _fit(_stereo(_resample(back, back_sr, sr)), voc.shape[-1])
            shift = backing_semitones - 12.0 * round(backing_semitones / 12.0)
            if abs(shift) >= 0.01:
                back = _fit(torchaudio.functional.pitch_shift(back, sr, n_steps=int(round(shift))), voc.shape[-1])
            out = out + back * _db(backing_gain_db)
        peak = out.abs().max().item()
        if peak > 0.99:
            out = out * (0.99 / peak)
        return (_audio_out(out, sr),)


NODE_CLASS_MAPPINGS = {
    "SongVoiceSplit": SongVoiceSplit,
    "SongVoiceTrain": SongVoiceTrain,
    "SongVoiceLoad": SongVoiceLoad,
    "SongVoiceClone": SongVoiceClone,
    "SongVoiceMix": SongVoiceMix,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SongVoiceSplit": "Song Voice · Split Vocals",
    "SongVoiceTrain": "Song Voice · Train My Voice",
    "SongVoiceLoad": "Song Voice · Load My Voice",
    "SongVoiceClone": "Song Voice · Clone My Voice",
    "SongVoiceMix": "Song Voice · Remix",
}
