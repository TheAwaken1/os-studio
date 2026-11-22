"""MP3 writer — replaces ComfyUI's SaveAudioAdvanced (mp3 / V0) for the app.

ComfyUI AUDIO is {'waveform': [batch, channels, samples], 'sample_rate': int}.
Batch item 0 is written; mono and stereo are both supported.
"""
from __future__ import annotations

import os
import re

_CBR_KBPS = {"V0": 320, "V2": 256, "V4": 192}


def _pcm16(audio):
    """AUDIO dict -> (int16 array [samples, channels], sample_rate, channels)."""
    import numpy as np

    wave = audio["waveform"]
    if hasattr(wave, "detach"):
        wave = wave.detach().float().cpu().numpy()
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 3:          # [batch, channels, samples] -> first item
        wave = wave[0]
    elif wave.ndim == 1:        # [samples]
        wave = wave[None, :]
    if wave.shape[0] > 2:       # LAME takes mono or stereo
        wave = wave[:2]
    pcm = (np.clip(wave.T, -1.0, 1.0) * 32767.0).astype(np.int16)
    return np.ascontiguousarray(pcm), int(audio["sample_rate"]), int(wave.shape[0])


def write_mp3(audio, path: str, quality: str = "V0") -> str:
    """Encode `audio` to exactly `path`.

    Constant bitrate on purpose: lameenc cannot write the Xing/LAME header, and
    a VBR file without one reports the wrong length and seeks to the wrong
    place in browsers — which breaks the synced-lyrics player. 320 kbps stereo
    (192 mono) is at or above V0 quality, so nothing is lost.
    """
    import lameenc

    pcm, sr, channels = _pcm16(audio)
    enc = lameenc.Encoder()
    enc.set_in_sample_rate(sr)
    enc.set_channels(channels)
    enc.set_quality(2)
    enc.set_bit_rate(_CBR_KBPS.get(quality, 320) if channels == 2 else 192)
    enc.silence()
    data = bytes(enc.encode(pcm.tobytes())) + bytes(enc.flush())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def next_song_path(filename_prefix: str, out_root: str, ext: str = "mp3") -> str:
    """<out_root>/<prefix>_<NNNNN>.<ext> with the next free counter."""
    normalized = str(filename_prefix or "").replace("\\", "/").strip("/")
    subfolder, filename = os.path.split(normalized)
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename).strip(". ") or "song"
    out_dir = os.path.abspath(os.path.join(out_root, subfolder))
    os.makedirs(out_dir, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(filename)}_(\d{{5}})\.{ext}$", re.I)
    nums = [int(m.group(1)) for name in os.listdir(out_dir)
            if (m := pattern.match(name))]
    return os.path.join(out_dir, f"{filename}_{(max(nums) + 1) if nums else 1:05d}.{ext}")


def save_mp3(audio, filename_prefix: str, out_root: str,
             format: str = "mp3", quality: str = "V0") -> str:
    """Save with an auto-incremented counter. Returns the absolute path."""
    path = next_song_path(filename_prefix, out_root, format)
    if format == "mp3":
        write_mp3(audio, path, quality)
    else:
        import soundfile as sf
        pcm, sr, _ch = _pcm16(audio)
        sf.write(path, pcm, sr, format=format.upper())
    print(f"[studio] saved {path}", flush=True)
    return path
