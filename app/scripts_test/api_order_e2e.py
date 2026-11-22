"""The order that broke: a fresh app, a SONG first, then "Write lyrics".

    python scripts_test/api_order_e2e.py            # against a freshly started server

Also covers the single-file INT8 song model and deleting a voice (a throwaway
dummy voice — real voices are never touched). Test songs are deleted again.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import library  # noqa: E402

BASE = "http://127.0.0.1:7861"
STYLE = "English, upbeat pop, bright female vocals, punchy drums, 110 BPM"
LYRICS = """[verse]
Coffee's cold and the screen is bright
Typing songs in the middle of the night

[chorus]
Push the button, hear it play
A little tune to save the day
"""
failures = []


def call(path, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, {"detail": err.read().decode()[:300]}


def run(label, path, body, limit=600):
    code, started = call(path, body)
    if code != 200:
        failures.append(label)
        print(f"FAIL  {label}: HTTP {code} {started}")
        return {}
    t0 = time.time()
    while time.time() - t0 < limit:
        time.sleep(3)
        job = call("/api/job")[1]["job"]
        if job["id"] == started["job"] and job["state"] != "running":
            break
    took = round(time.time() - t0)
    if job["state"] != "done":
        failures.append(label)
        print(f"FAIL  {label}: {job['state']} after {took}s — {job.get('error')}")
        return {}
    print(f"ok    {label} ({took}s)")
    return job["result"]


for _ in range(120):                                   # wait for the engine
    if call("/api/job")[1].get("engine", {}).get("state") == "ready":
        break
    time.sleep(2)

state = call("/api/state")[1]
print("song models:", [(m["id"], m["installed"]) for m in state["song_models"]], "| auto ->", state["auto_song_model"])
made = []

song = run("1. song first, full model", "/api/song",
           {"lyrics": LYRICS, "style": STYLE, "seed": 11, "title": "Order Test Full", "model": "full"}).get("song")
if song:
    made.append(song["id"])
    print("      ", song["duration"], "s")

lyrics = run("2. lyrics AFTER a song (the reported error)", "/api/lyrics",
             {"idea": "a song about a dog who loves the mailman"}).get("lyrics", "")
print("      ", len(lyrics), "chars |", lyrics[:90].replace("\n", " / "))

song = run("3. song, INT8 single-file model", "/api/song",
           {"lyrics": LYRICS, "style": STYLE, "seed": 11, "title": "Order Test Int8",
            "model": library.INT8_FILE}).get("song")
if song:
    made.append(song["id"])
    print("      ", song["duration"], "s")

lyrics = run("4. lyrics after the INT8 song, 4-bit writer", "/api/lyrics",
             {"idea": "a song about burnt toast", "writer": "qwen4b-4bit"}).get("lyrics", "")
print("      ", len(lyrics), "chars")

# ---- deleting a voice, with a dummy that is not a real model
name = "zz_delete_test"
paths = [os.path.join(library.VOICE_DIR, name + ext) for ext in (".pth", ".wav", ".json")]
for path in paths:
    with open(path, "wb") as fh:
        fh.write(b"{}" if path.endswith(".json") else b"dummy")
before = sorted(v["name"] for v in call("/api/voices")[1]["voices"])
code, out = call("/api/voices/" + name, method="DELETE")
after = sorted(v["name"] for v in call("/api/voices")[1]["voices"])
gone = not any(os.path.exists(p) for p in paths)
ok = code == 200 and gone and name in before and after == [v for v in before if v != name]
print(("ok   " if ok else "FAIL ") + f" 5. delete a voice: HTTP {code} how={out.get('how')} files gone={gone} others untouched={after == [v for v in before if v != name]}")
if not ok:
    failures.append("delete voice")
code, out = call("/api/voices/" + name, method="DELETE")
print(("ok   " if code == 404 else "FAIL ") + f" 6. deleting it twice -> HTTP {code}")
code, out = call("/api/voices/..%5C..%5Cconfig", method="DELETE")
print(("ok   " if code in (404, 400) else "FAIL ") + f" 7. path tricks -> HTTP {code}")
print("       voices now:", after)

for sid in made:
    call("/api/songs/" + sid, method="DELETE")
print("FAILED: " + ", ".join(failures) if failures else "ALL PASSED")
sys.exit(1 if failures else 0)
