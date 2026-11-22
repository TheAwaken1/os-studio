"""VRAM / speed of the single-file (ComfyUI-native) YuE2 checkpoints.

    python scripts_test/native_probe.py --cap 7 --max-seconds 180

Same idea as vram_probe.py, but for models/checkpoints/yue2_3b_int8_convrot
.safetensors and friends, driven through the vendored comfy core the way the
"YuE2 Native Text to Song" workflow does (KSampler dpm_2 / sgm_uniform, 32
steps). `--cap N` makes PyTorch refuse to go past N GiB so a 16 GB card can
stand in for a smaller one. Stop the OS Studio server first.
"""
import argparse
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "shims"), os.path.join(ROOT, "engine"), ROOT):
    sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="yue2_3b_int8_convrot.safetensors")
ap.add_argument("--cap", type=float, default=0.0)
ap.add_argument("--max-seconds", type=float, default=360.0)
ap.add_argument("--seed", type=int, default=4242)
ap.add_argument("--tag", default="")
ap.add_argument("--save", default="", help="write the result here as .mp3")
ap.add_argument("--out", default=os.path.join(ROOT, "logs", "native_probe.jsonl"))
args = ap.parse_args()

import bootstrap  # noqa: E402

bootstrap.bootstrap()
import torch  # noqa: E402

if args.cap:
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.cap * 2**30 / total), 0)

import service  # noqa: E402
from song_lrc.song_lrc_node import SongLyricsClean  # noqa: E402

STYLE = ("English, male voiceover rap, spoken word over a beat, zero melody, "
         "conversational, like reading lines to a friend, no chorus singing, "
         "minimal drums, 84 BPM")
LYRICS = SongLyricsClean().clean(open(os.path.join(ROOT, "test_lyrics.txt"), encoding="utf-8").read())[0]

row = {"tag": args.tag, "checkpoint": args.checkpoint, "cap": args.cap, "max_seconds": args.max_seconds}
series, stop = [], threading.Event()


def watch():
    while not stop.is_set():
        series.append(torch.cuda.memory_reserved() / 2**30)
        time.sleep(0.25)


threading.Thread(target=watch, daemon=True).start()
try:
    t0 = time.time()
    with torch.inference_mode():
        pipe = service.native_load(args.checkpoint)
        row["load_s"] = round(time.time() - t0, 1)
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        audio, _abc, info = service.native_generate(pipe, STYLE, LYRICS, seed=args.seed,
                                                    max_seconds=args.max_seconds)
    row.update(ok=True, gen_s=round(time.time() - t1, 1),
               song_s=round(audio["waveform"].shape[-1] / audio["sample_rate"], 1),
               shape=list(audio["waveform"].shape), sample_rate=audio["sample_rate"],
               peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               peak_reserved_gib=round(max(series or [0]), 2), info=info)
    if args.save:
        from mp3_save import write_mp3
        write_mp3(audio, args.save)
        row["saved"] = args.save
except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
    import traceback
    traceback.print_exc()
    row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}",
               peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               peak_reserved_gib=round(max(series or [0]), 2))
stop.set()
with open(args.out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("RESULT", json.dumps(row))
