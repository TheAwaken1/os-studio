"""End-to-end test against the running OS Studio server (HTTP API only).

    python scripts_test/api_e2e.py [--voice my_voice] [--port 7861]

Covers every VRAM hand-off: lyrics -> song -> lyrics again -> song in a
cloned voice. Prints each stage with timings; exits non-zero on any failure.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=7861)
ap.add_argument("--voice", default="my_voice")
ap.add_argument("--skip-voice", action="store_true")
args = ap.parse_args()
BASE = f"http://127.0.0.1:{args.port}"

STYLE = ("English, male voiceover rap, spoken word over a beat, zero melody, "
         "conversational, like reading lines to a friend, no chorus singing, "
         "minimal drums, 84 BPM")


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        raise SystemExit(f"HTTP {err.code} on {path}: {err.read().decode()[:300]}")


def run(label, path, body):
    print(f"\n=== {label}", flush=True)
    t0 = time.time()
    jid = call(path, body)["job"]
    last, seq = None, 0
    while True:
        snap = call(f"/api/job?since={seq}")
        seq = snap["log_seq"]
        job = snap["job"]
        assert job and job["id"] == jid, job
        prog = job.get("progress") or {}
        if prog.get("label") and prog["label"] != last:
            last = prog["label"]
            print(f"  {time.time() - t0:6.1f}s  {last}  ({prog['overall'] * 100:.0f}% overall)", flush=True)
        if job["state"] != "running":
            break
        time.sleep(1.0)
    took = time.time() - t0
    print(f"  -> {job['state']} in {took:.1f}s", flush=True)
    if job["state"] != "done":
        raise SystemExit(f"{label} FAILED: {job['error']}")
    return job["result"], took


# wait for the engine
t0 = time.time()
while True:
    st = call("/api/state")
    if st["engine"]["state"] != "starting":
        break
    time.sleep(1)
print(f"engine: {st['engine']} after {time.time() - t0:.0f}s | gpu: {st['gpu']}")
print("voices:", [v["name"] for v in st["voices"]], "| models:", [(m["name"], m["ok"]) for m in st["models"]])
if st["engine"]["state"] != "ready":
    sys.exit(1)

res, _ = run("lyrics #1 (cold)", "/api/lyrics", {"idea": "a song about a stubborn Roomba that thinks it's a horse"})
lyrics1 = res["lyrics"]
print(f"  {len(lyrics1)} chars, first lines: {lyrics1.splitlines()[:3]}")
assert len(lyrics1) > 200 and "[" in lyrics1

res, took = run("song #1 (AI singer) — right after lyrics", "/api/song",
                {"lyrics": lyrics1, "style": STYLE, "seed": 777, "idea": "roomba horse"})
song1 = res["song"]
print("  ", {k: song1[k] for k in ("title", "file", "duration", "has_lrc", "has_ai", "took")})
assert song1["has_lrc"] and song1["duration"] > 30

res, _ = run("lyrics #2 (YuE2 still loaded)", "/api/lyrics", {"idea": "a song about Monday morning coffee"})
lyrics2 = res["lyrics"]
assert len(lyrics2) > 200

if not args.skip_voice:
    res, took = run(f"song #2 in voice '{args.voice}'", "/api/song",
                    {"lyrics": lyrics2, "style": STYLE, "seed": 4242, "voice": args.voice,
                     "idea": "monday coffee"})
    song2 = res["song"]
    print("  ", {k: song2[k] for k in ("title", "file", "duration", "has_lrc", "has_ai", "voice", "took")})
    assert song2["has_ai"] and song2["voice"] == args.voice

print("\ngpu after:", call("/api/state")["gpu"])
print("ALL OK")
