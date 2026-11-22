"""Measure each lyrics-writer option: load time, VRAM, speed, and a sample.

    python scripts_test/lyrics_probe.py            # every option
    python scripts_test/lyrics_probe.py qwen2b     # just one

Stop the OS Studio server first. Results go to logs/lyrics_probe.jsonl.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "shims"), os.path.join(ROOT, "engine"), ROOT):
    sys.path.insert(0, p)

import service  # noqa: E402
import torch  # noqa: E402

wanted = sys.argv[1:] or [w["id"] for w in service.lyrics_writers()]
out = os.path.join(ROOT, "logs", "lyrics_probe.jsonl")
IDEA = "a song about a stubborn Roomba that thinks it's a horse"

for wid in wanted:
    row = {"writer": wid}
    try:
        service._drop_lyrics_model()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        service.write_lyrics(IDEA, writer=wid)                 # cold: includes the load
        row["cold_s"] = round(time.time() - t0, 1)
        row["loaded_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        t1 = time.time()
        text = service.write_lyrics(IDEA, writer=wid)          # warm: writing only
        row["warm_s"] = round(time.time() - t1, 1)
        row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        row["chars"] = len(text)
        row["chars_per_s"] = round(len(text) / max(row["warm_s"], 0.1))
        row["sections"] = sum(1 for ln in text.splitlines() if ln.strip().endswith("]"))
        row["sample"] = text[:420]
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001
        row.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("RESULT", json.dumps(row, ensure_ascii=False))
