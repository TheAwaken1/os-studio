"""Benchmark one song through the running server and watch VRAM while it renders.

    python scripts_test/bench_song.py [--song "Title fragment"] [--seed 4242]

Re-renders an existing library song (same lyrics/style) so runs are comparable.
"""
import argparse
import json
import subprocess
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7861)
ap.add_argument("--song", default="Wakes Me Up")
ap.add_argument("--seed", type=int, default=4242)
args = ap.parse_args()
BASE = f"http://127.0.0.1:{args.port}"


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])


src = next(s for s in call("/api/songs")["songs"] if args.song.lower() in s["title"].lower() and s["lyrics"])
t0 = time.time()
jid = call("/api/song", {"lyrics": src["lyrics"], "style": src["style"], "seed": args.seed, "idea": "benchmark"})["job"]
peak, by_stage = 0, {}
while True:
    job = call("/api/job")["job"]
    used = vram_mib()
    peak = max(peak, used)
    stage = (job.get("progress") or {}).get("label") or "-"
    by_stage[stage] = max(by_stage.get(stage, 0), used)
    if job["state"] != "running":
        break
    time.sleep(1.5)
assert job["id"] == jid and job["state"] == "done", job
song = job["result"]["song"]
print(f"total {time.time() - t0:.1f}s | card peak {peak} MiB | per stage {by_stage}")
print("file:", song["file"], "| duration", song["duration"])
meta = next(s for s in call("/api/songs")["songs"] if s["id"] == song["id"])
print("took:", meta["took"])
