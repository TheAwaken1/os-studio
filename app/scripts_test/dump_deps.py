"""Run the whole pipeline in-process and list the pip distributions it really
imported. This is how requirements.txt was derived — re-run it after changing
the engine.

    python scripts_test/dump_deps.py --voice my_voice --clip path/to/clip.wav

Stop the OS Studio server first (it needs the GPU).
"""
import argparse
import importlib.metadata as md
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "shims"), os.path.join(ROOT, "engine"), ROOT):
    sys.path.insert(0, p)

ap = argparse.ArgumentParser()
ap.add_argument("--voice", default="my_voice")
ap.add_argument("--clip", default="")
ap.add_argument("--train-name", default="dependency_probe")
ap.add_argument("--out", default=os.path.join(ROOT, "logs", "deps.json"))
args = ap.parse_args()

import app  # noqa: E402,F401  (server imports: fastapi, uvicorn, pydantic)
import uvicorn  # noqa: E402,F401
import service  # noqa: E402

lyrics = service.write_lyrics("a ten second jingle about testing software")
print("lyrics chars:", len(lyrics))
song = service.make_song(
    "[verse]\nTesting one two, the mic is on\nShort little song, then it is gone\n\n[chorus]\nStudio test, studio test\nShort and sweet, then give it a rest",
    "English, male voiceover rap, spoken word over a beat, minimal drums, 84 BPM",
    seed=5, voice=args.voice, title="Dependency Probe")
print("song:", song["file"], song["duration"])
if args.clip:
    print("train:", service.train_voice(args.clip, args.train_name, retrain=True))

site = [p for p in sys.path if "site-packages" in p.replace("\\", "/")]
# importlib.metadata.packages_distributions() on Python 3.10 only reads
# top_level.txt, which uv-installed wheels do not ship — so map from RECORD.
pkg_to_dist = {}
for dist in md.distributions():
    for f in dist.files or []:
        parts = f.parts
        if not parts or parts[0].endswith((".dist-info", ".data", "__pycache__")) or parts[0] == "..":
            continue
        top = parts[0][:-3] if len(parts) == 1 and parts[0].endswith(".py") else parts[0].split(".")[0]
        pkg_to_dist.setdefault(top, set()).add(dist.metadata["Name"])
used, tops_seen = {}, set()
for name, mod in list(sys.modules.items()):
    top = name.split(".")[0]
    path = getattr(mod, "__file__", None) or ""
    if not path or "site-packages" not in path.replace("\\", "/"):
        continue
    tops_seen.add(top)
    for dist in pkg_to_dist.get(top, []):
        used.setdefault(dist, set()).add(top)

rows = sorted((d.lower(), md.version(d), sorted(tops)) for d, tops in used.items())
unmapped = sorted(t for t in tops_seen if t not in pkg_to_dist)
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump({"python": sys.version, "imported": rows, "unmapped_modules": unmapped,
               "song_id": song["id"]}, fh, indent=1)
print(f"{len(rows)} distributions imported -> {args.out} (unmapped modules: {unmapped})")
for d, v, tops in rows:
    print(f"{d}=={v}")
