"""How much VRAM does one song really need at a given setting?

    python scripts_test/vram_probe.py --budget 12 --quant fp8 --cap 11 --max-tokens 4500

Runs YuE2 once, in-process, with the song-model settings given, and reports
peak VRAM and speed. `--cap N` makes PyTorch refuse to go past N GiB (an
out-of-memory error instead of Windows silently spilling into shared RAM), so
a 16 GB card can stand in for a smaller one. Stop the OS Studio server first.
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
ap.add_argument("--budget", type=int, default=16)
ap.add_argument("--quant", default="none", choices=["none", "fp8"])
ap.add_argument("--offload", type=int, default=1)
ap.add_argument("--cap", type=float, default=0.0, help="hard VRAM cap in GiB (0 = none)")
ap.add_argument("--max-tokens", type=int, default=9000)
ap.add_argument("--tile-frames", type=int, default=0)
ap.add_argument("--seed", type=int, default=4242)
ap.add_argument("--lyrics", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=os.path.join(ROOT, "logs", "vram_probe.jsonl"))
args = ap.parse_args()

import bootstrap  # noqa: E402

bootstrap.bootstrap()
import torch  # noqa: E402
from yue2_repo.nodes.yue2 import YuE2Loader, YuE2Sampler  # noqa: E402

STYLE = ("English, male voiceover rap, spoken word over a beat, zero melody, "
         "conversational, like reading lines to a friend, no chorus singing, "
         "minimal drums, 84 BPM")
LYRICS = args.lyrics and open(args.lyrics, encoding="utf-8").read() or open(
    os.path.join(ROOT, "test_lyrics.txt"), encoding="utf-8").read()
from song_lrc.song_lrc_node import SongLyricsClean  # noqa: E402

LYRICS = SongLyricsClean().clean(LYRICS)[0]          # same cleaning the app applies

row = {"tag": args.tag, "budget": args.budget, "quant": args.quant, "offload": bool(args.offload),
       "cap": args.cap, "max_tokens": args.max_tokens, "tile_frames": args.tile_frames}
total = torch.cuda.get_device_properties(0).total_memory
series = []
stop = threading.Event()


def watch():
    while not stop.is_set():
        series.append(torch.cuda.memory_reserved() / 2**30)
        time.sleep(0.25)


try:
    t0 = time.time()
    with torch.inference_mode():
        (pipe,) = YuE2Loader().load(model="YuE2-3B", vae="YuE2-Vae", device="cuda",
                                   memory_budget_gib=args.budget, offload_ar=bool(args.offload),
                                   offline=True, attention_backend="auto", quantization=args.quant)
        row["load_s"] = round(time.time() - t0, 1)
        if args.cap:
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.cap * 2**30 / total), 0)
        torch.cuda.reset_peak_memory_stats()
        threading.Thread(target=watch, daemon=True).start()
        t1 = time.time()
        (audio, _abc, info) = YuE2Sampler().generate(
            pipeline=pipe, style=STYLE, lyrics=LYRICS, cot="off", seed=args.seed,
            cfg_scale=-1, ode_steps=32, abc_max_tokens=4096,
            semantic_max_tokens=args.max_tokens, abc_temperature=0.7,
            semantic_temperature=1.0, save_flac=False, save_abc=False,
            vae_decode="tiled", vae_tile_frames=args.tile_frames, abc=None,
            abort_after_plan=False, sampling_settings=None, save_artifacts=False)
    stop.set()
    row.update(ok=True, gen_s=round(time.time() - t1, 1),
               song_s=round(audio["waveform"].shape[-1] / audio["sample_rate"], 1),
               peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               peak_reserved_gib=round(max(series or [0]), 2), info=info)
except Exception as exc:  # noqa: BLE001 - the failure IS the measurement
    stop.set()
    row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}",
               peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               peak_reserved_gib=round(max(series or [0]), 2))

with open(args.out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row) + "\n")
print("RESULT", json.dumps(row))
