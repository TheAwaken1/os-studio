"""Do the lyrics writers actually follow the chosen song structure?

    python scripts_test/structure_e2e.py                 # default writer
    python scripts_test/structure_e2e.py qwen4b-4bit     # another writer

Needs a running OS Studio. For each shape it asks for lyrics and compares
the section tags that came back with the ones requested.
"""
import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:7861"
WRITER = sys.argv[1] if len(sys.argv) > 1 else ""
SHAPES = [
    (["verse", "chorus"], 4),
    (["verse", "chorus", "verse", "chorus", "bridge", "outro"], 4),
    (["intro", "verse", "pre-chorus", "chorus", "verse", "pre-chorus", "chorus", "bridge", "chorus", "outro"], 4),
    (["verse", "verse", "verse"], 6),
    (["chorus", "verse", "chorus", "verse", "chorus"], 2),
]


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


for _ in range(120):
    if call("/api/job")["engine"]["state"] == "ready":
        break
    time.sleep(2)

bad = 0
for structure, lines in SHAPES:
    jid = call("/api/lyrics", {"idea": "a song about a dog who loves the mailman", "writer": WRITER,
                               "structure": structure, "lines": lines})["job"]
    t0 = time.time()
    while True:
        time.sleep(2)
        job = call("/api/job")["job"]
        if job["id"] == jid and job["state"] != "running":
            break
    text = (job.get("result") or {}).get("lyrics", "")
    got, counts = [], []
    for line in text.splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if m:
            got.append(m.group(1).lower())
            counts.append(0)
        elif line.strip() and counts:
            counts[-1] += 1
    want_counts = [0 if s == "intro" else (max(2, lines // 2) if s == "pre-chorus" else lines) for s in structure]
    same = got == structure
    short = [f"{s}:{c}/{w}" for s, c, w in zip(got, counts, want_counts) if c < w] if same else []
    bad += not same
    print(("ok   " if same else "DIFF ") + f"{round(time.time() - t0)}s want {' '.join(structure)} x{lines}")
    if not same:
        print("      got  " + " ".join(got))
    print("      lines per section:", counts, "| wanted:", want_counts, "| short sections:", short or "none",
          "| leftovers:", bool(re.search(r"\d\s*x\s*\[|\*\*|^#", text, re.M)))
print(f"{len(SHAPES) - bad}/{len(SHAPES)} shapes followed exactly")
