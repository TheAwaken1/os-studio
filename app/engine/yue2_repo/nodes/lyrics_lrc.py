# -*- coding: utf-8 -*-
"""Build an LRC file from YuE2 sectioned lyrics and the generated song."""
from __future__ import annotations

import os
import re

from .sheetsage2 import _NON_SINGING, _SECTION_ALIASES, parse_structure_spans
from .utils import mono_waveform, safe_stem, timestamp_dir

_TAG = re.compile(r"^\[([^\]]+)\]$")


def _stamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"[{int(seconds // 60):02d}:{seconds % 60:05.2f}]"


def parse_lyric_blocks(lyrics: str) -> list[tuple[str, list[str]]]:
    """Split YuE2 lyrics into ``[(section, lines)]`` in order."""
    blocks: list[tuple[str, list[str]]] = []
    name = "verse"
    bucket: list[str] = []
    for raw in (lyrics or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _TAG.match(line)
        if match:
            if bucket:
                blocks.append((name, bucket))
                bucket = []
            name = match.group(1).strip().lower() or "verse"
            continue
        bucket.append(line)
    if bucket:
        blocks.append((name, bucket))
    return blocks


def _audio_seconds(audio) -> float:
    waveform, rate = mono_waveform(audio)
    return float(waveform.numel()) / max(int(rate), 1)


def build_lrc(lyrics: str, seconds: float, title: str = "YuE2",
              artist: str = "YuE2", structure: str = "", offset: float = 0.0) -> str:
    """Time the same lyrics used to generate the song.

    Default: even spacing across the audio duration.
    If SheetSage2 ``structure`` is provided, lyric blocks are spread inside
    matching singing sections instead of across the whole track.
    """
    seconds = max(float(seconds), 1.0)
    offset = float(offset)
    blocks = parse_lyric_blocks(lyrics)
    singing = [(name, lines) for name, lines in blocks
               if lines and name not in _NON_SINGING]
    if not singing:
        singing = [(name, lines) for name, lines in blocks if lines]
    if not singing:
        raise ValueError("No lyric lines to time; add verse/chorus text, not only [section] tags")

    spans = parse_structure_spans(structure or "", seconds) if structure else []
    singing_spans = []
    for start, end, raw_name in spans:
        mapped = _SECTION_ALIASES.get(raw_name.lower(), raw_name.lower())
        if mapped not in _NON_SINGING:
            singing_spans.append((start, end, mapped))

    timed: list[tuple[float | None, str]] = []
    if singing_spans:
        count = min(len(singing), len(singing_spans))
        used_lines = 0
        for i in range(count):
            name, lines = singing[i]
            start, end, _span_name = singing_spans[i]
            start = max(start, offset)
            end = max(end, start + 0.25)
            step = (end - start) / max(len(lines), 1)
            timed.append((None, f"# [{name}]"))
            for j, line in enumerate(lines):
                timed.append((start + step * j, line))
                used_lines += 1
        leftover = singing[count:]
        if leftover:
            rest_start = max(singing_spans[count - 1][1] if count else offset, offset)
            rest_lines = [line for _, lines in leftover for line in lines]
            step = max(seconds - rest_start, 1.0) / max(len(rest_lines), 1)
            cursor = rest_start
            for name, lines in leftover:
                timed.append((None, f"# [{name}]"))
                for line in lines:
                    timed.append((cursor, line))
                    cursor += step
    else:
        total = sum(len(lines) for _, lines in singing)
        # YuE2 songs usually have an instrumental intro; starting at 0 makes
        # karaoke run ahead of the vocal. Skip ~6% lead-in and ~9% outro.
        start = max(offset, seconds * 0.06)
        usable = max(seconds * 0.91 - start, 1.0)
        step = usable / max(total, 1)
        cursor = start
        for name, lines in singing:
            timed.append((None, f"# [{name}]"))
            for line in lines:
                timed.append((cursor, line))
                cursor += step

    header = [
        f"[ti:{title}]",
        f"[ar:{artist}]",
        f"[length:{int(seconds // 60):02d}:{seconds % 60:05.2f}]",
        "[by:YuE2 lyrics-to-lrc]",
        "",
    ]
    body = []
    for stamp, text in timed:
        if stamp is None:
            body.append(text)
        else:
            body.append(f"{_stamp(stamp)}{text}")
    return "\n".join(header + body) + "\n"


class YuE2LyricsToLRC:
    """Turns the lyrics you already typed into YuE2 into a timed LRC file."""

    DESCRIPTION = (
        "Builds an .lrc from YuE2 sectioned lyrics and the generated AUDIO. "
        "Times lines across the song; optional SheetSage2 structure improves section placement."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "forceInput": True,
                    "tooltip": "The same [verse]/[chorus] lyrics fed into YuE2 Song Generator."}),
                "audio": ("AUDIO", {
                    "tooltip": "Connect YuE2 Song Generator audio so timestamps match the song length."}),
            },
            "optional": {
                "structure": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional SheetSage2 structure output. Leave empty for even timing."}),
                "title": ("STRING", {"default": "YuE2"}),
                "artist": ("STRING", {"default": "YuE2"}),
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 30.0, "step": 0.1}),
                "save": ("BOOLEAN", {"default": True,
                    "tooltip": "Save the .lrc under output/YuE2_LRC/."}),
                "filename": ("STRING", {"default": "lyrics"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lrc", "path")
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "YuE2/Lyrics"

    def convert(self, lyrics, audio, structure="", title="YuE2", artist="YuE2",
                offset_seconds=0.0, save=True, filename="lyrics"):
        seconds = _audio_seconds(audio)
        lrc = build_lrc(lyrics, seconds, title=title or "YuE2",
                        artist=artist or "YuE2", structure=structure,
                        offset=offset_seconds)
        path = ""
        if save:
            out = timestamp_dir("YuE2_LRC")
            path = os.path.join(out, safe_stem(filename, "lyrics") + ".lrc")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(lrc)
            print(f"[YuE2 LRC] {seconds:.1f}s -> {path}")
        return (lrc, path)


NODE_CLASS_MAPPINGS = {
    "YuE2LyricsToLRC": YuE2LyricsToLRC,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YuE2LyricsToLRC": "YuE2 Lyrics to LRC",
}
