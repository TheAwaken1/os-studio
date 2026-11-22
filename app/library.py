"""Song library + config helpers. Deliberately light: no torch, no engine.

The web server imports this at startup so the page (and the library) is usable
while the heavy engine is still warming up.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config() -> dict:
    with open(os.path.join(APP_ROOT, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.setdefault("port", 7861)
    cfg.setdefault("output_dir", os.path.join(APP_ROOT, "output", "songs"))
    cfg.setdefault("input_dir", os.path.join(APP_ROOT, "input"))
    cfg.setdefault("models_dir", os.path.join(APP_ROOT, "models"))
    return cfg


CFG = load_config()
OUTPUT_DIR = os.path.abspath(CFG["output_dir"])
MODELS_DIR = os.path.abspath(CFG["models_dir"])
# Trained voices are per-user app data, kept apart from (possibly shared) model weights.
VOICE_DIR = os.path.abspath(CFG.get("voices_dir", os.path.join(APP_ROOT, "voices")))
os.makedirs(VOICE_DIR, exist_ok=True)

_COUNTER = re.compile(r"_(\d{5})$")
_LRC_LENGTH = re.compile(r"^\[length:\s*(\d+):(\d+(?:\.\d+)?)\]", re.M)


# ------------------------------------------------------------------ ids
def song_id(mp3_path: str) -> str:
    rel = os.path.relpath(os.path.abspath(mp3_path), OUTPUT_DIR).replace("\\", "/")
    return base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii").rstrip("=")


def song_path(sid: str) -> str:
    """id -> absolute mp3 path, refusing anything outside the songs folder."""
    try:
        rel = base64.urlsafe_b64decode(sid + "=" * (-len(sid) % 4)).decode("utf-8")
    except Exception as exc:
        raise FileNotFoundError(sid) from exc
    path = os.path.abspath(os.path.join(OUTPUT_DIR, rel))
    if os.path.commonpath([OUTPUT_DIR, path]) != OUTPUT_DIR or not path.lower().endswith(".mp3"):
        raise FileNotFoundError(sid)
    if not os.path.isfile(path):
        raise FileNotFoundError(sid)
    return path


def sidecar(mp3_path: str, ext: str) -> str:
    """Path of a file that travels with the song: .lrc, .json or .ai.mp3"""
    return mp3_path[:-4] + ext


# ------------------------------------------------------------------ songs
def _duration(mp3_path: str, meta: dict, lrc_text: str | None) -> float:
    if meta.get("duration"):
        return float(meta["duration"])
    if lrc_text:
        m = _LRC_LENGTH.search(lrc_text)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
    try:
        import soundfile as sf
        return float(sf.info(mp3_path).duration)
    except Exception:
        return 0.0


def describe(mp3_path: str) -> dict:
    stem = os.path.basename(mp3_path)[:-4]
    meta: dict = {}
    try:
        with open(sidecar(mp3_path, ".json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        pass
    lrc_path = sidecar(mp3_path, ".lrc")
    lrc_text = None
    if os.path.isfile(lrc_path):
        try:
            with open(lrc_path, encoding="utf-8") as fh:
                lrc_text = fh.read()
        except OSError:
            pass
    st = os.stat(mp3_path)
    return {
        "id": song_id(mp3_path),
        "title": meta.get("title") or _COUNTER.sub("", stem).strip() or stem,
        "file": os.path.basename(mp3_path),
        "created": meta.get("created") or st.st_mtime,
        "duration": round(_duration(mp3_path, meta, lrc_text), 2),
        "size": st.st_size,
        "has_lrc": lrc_text is not None,
        "has_ai": os.path.isfile(sidecar(mp3_path, ".ai.mp3")),
        "voice": meta.get("voice") or "",
        "style": meta.get("style") or "",
        "singer": meta.get("singer") or "",
        "seed": meta.get("seed"),
        "cot": meta.get("cot") or "",
        "idea": meta.get("idea") or "",
        "lyrics": meta.get("lyrics") or "",
        "took": meta.get("took"),
        "cut_short": bool(meta.get("cut_short")),
    }


def list_songs() -> list[dict]:
    songs = []
    for folder, _dirs, files in os.walk(OUTPUT_DIR):
        for name in files:
            low = name.lower()
            if low.endswith(".mp3") and not low.endswith(".ai.mp3"):
                try:
                    songs.append(describe(os.path.join(folder, name)))
                except OSError:
                    continue
    songs.sort(key=lambda s: s["created"], reverse=True)
    return songs


def read_lrc(sid: str) -> str:
    path = sidecar(song_path(sid), ".lrc")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def delete_song(sid: str) -> None:
    mp3 = song_path(sid)
    for ext in (".mp3", ".lrc", ".json", ".ai.mp3"):
        try:
            os.remove(sidecar(mp3, ext))
        except FileNotFoundError:
            pass


# ------------------------------------------------------------------ player visuals
# The player draws from numbers measured here, not from the browser's Web Audio:
# that path is unavailable on iPhone and unreliable inside embedded app windows.
VIZ_VERSION = 1
VIZ_FPS = 20            # spectrum frames per second of audio
VIZ_BANDS = 24          # bars in the live spectrum
VIZ_PEAKS = 240         # bars in the whole-song waveform
VIZ_DIR = os.path.join(os.path.abspath(CFG.get("temp_dir") or os.path.join(OUTPUT_DIR, os.pardir, "temp")), "viz")
_VIZ_LOCK = threading.Lock()


def analyze_audio(samples, rate: int, fps: int = VIZ_FPS, bands: int = VIZ_BANDS, peaks: int = VIZ_PEAKS) -> dict:
    """Mono float samples -> loudness outline of the whole song + a per-frame spectrum."""
    import numpy as np

    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    total = int(samples.size)
    duration = total / float(rate) if rate else 0.0
    empty = {"version": VIZ_VERSION, "fps": fps, "bands": bands, "frames": 0,
             "duration": round(duration, 3), "spectrum": "", "peaks": [0] * peaks}
    if total < 2 or rate <= 0:
        return empty

    # whole-song outline
    edges = np.linspace(0, total, peaks + 1).astype(np.int64)
    outline = np.zeros(peaks, dtype=np.float64)
    for i in range(peaks):
        chunk = samples[edges[i]:max(edges[i] + 1, edges[i + 1])]
        outline[i] = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) if chunk.size else 0.0
    top = float(outline.max())
    outline = (outline / top) ** 0.7 if top > 0 else outline

    # live spectrum
    hop = max(1, int(round(rate / float(fps))))
    win = 1024 if rate < 32000 else 2048
    count = int(np.ceil(total / float(hop)))
    padded = np.concatenate([samples, np.zeros(win + hop, dtype=np.float32)])
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    limits = np.geomspace(70.0, min(8000.0, rate * 0.45), bands + 1)   # where songs actually have energy
    slices = []
    for b in range(bands):
        lo = int(np.searchsorted(freqs, limits[b]))
        hi = max(lo + 1, int(np.searchsorted(freqs, limits[b + 1])))
        slices.append((lo, min(hi, freqs.size)))
    energy = np.zeros((count, bands), dtype=np.float32)
    for start in range(0, count, 512):                      # bounded memory on long songs
        idx = np.arange(start, min(count, start + 512))
        block = padded[idx[:, None] * hop + np.arange(win)[None, :]] * window
        mag = np.abs(np.fft.rfft(block, axis=1))
        for b, (lo, hi) in enumerate(slices):
            energy[idx, b] = mag[:, lo:hi].mean(axis=1)
    level = 20.0 * np.log10(energy + 1e-9)
    loud = float(np.percentile(level, 99.5))
    if loud < -150.0:                                       # digital silence
        return {**empty, "frames": count, "spectrum": base64.b64encode(bytes(count * bands)).decode("ascii")}
    # Each bar is scaled against its own loud moments, so quiet treble still moves.
    ref = np.maximum(np.percentile(level, 99.0, axis=0), loud - 30.0)
    scaled = np.clip((level - (ref - 48.0)) / 48.0, 0.0, 1.0)
    return {
        "version": VIZ_VERSION, "fps": fps, "bands": bands, "frames": count,
        "duration": round(duration, 3),
        "spectrum": base64.b64encode(np.round(scaled * 255.0).astype(np.uint8).tobytes()).decode("ascii"),
        "peaks": [int(v) for v in np.round(np.clip(outline, 0.0, 1.0) * 255.0)],
    }


def song_viz(mp3_path: str) -> dict:
    """Measured visuals for one audio file, cached on disk beside the temp files."""
    st = os.stat(mp3_path)
    key = hashlib.sha1(f"{VIZ_VERSION}|{os.path.abspath(mp3_path)}|{st.st_size}|{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:24]
    cache = os.path.join(VIZ_DIR, key + ".json")
    try:
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        pass
    with _VIZ_LOCK:                                         # one decode at a time
        try:
            with open(cache, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
        import numpy as np
        import soundfile as sf
        data, rate = sf.read(mp3_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        del data
        while rate >= 44100:                                # half the work, same picture
            mono = mono[: mono.size // 2 * 2].reshape(-1, 2).mean(axis=1)
            rate //= 2
        result = analyze_audio(np.ascontiguousarray(mono), rate)
        try:
            os.makedirs(VIZ_DIR, exist_ok=True)
            stale = sorted((os.path.join(VIZ_DIR, n) for n in os.listdir(VIZ_DIR) if n.endswith(".json")),
                           key=os.path.getmtime)
            for old in stale[:-400]:
                os.remove(old)
            tmp = cache + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(result, fh)
            os.replace(tmp, cache)
        except OSError:
            pass                                            # the cache is only a speed-up
        return result


def write_meta(mp3_path: str, meta: dict) -> None:
    meta = dict(meta)
    meta.setdefault("created", time.time())
    with open(sidecar(mp3_path, ".json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ style library
def style_library(quick_presets: list[dict] | None = None,
                  filename: str = "style_prompts.txt") -> list[dict]:
    """Read editable prompts, excluding quick picks and repeated names/prompts.
    style_prompts.txt is written for YuE2, style_prompts_ace.txt for ACE-Step."""
    def name_key(value):
        # Parenthetical labels describe the basic genre, not a new subgenre.
        return " ".join(re.sub(r"\([^)]*\)", "", value).casefold().split())

    def prompt_key(value):
        return " ".join(value.casefold().split())

    seen_names = {name_key(p["name"]) for p in (quick_presets or [])}
    seen_prompts = {prompt_key(p["style"]) for p in (quick_presets or [])}
    try:
        with open(os.path.join(APP_ROOT, filename), encoding="utf-8-sig") as fh:
            text = "\n".join(fh.read().splitlines()[3:])
    except FileNotFoundError:
        return []
    presets = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        name, prompt = lines[0], " ".join(lines[1:])
        key, prompt_id = name_key(name), prompt_key(prompt)
        if key in seen_names or prompt_id in seen_prompts:
            continue
        seen_names.add(key)
        seen_prompts.add(prompt_id)
        low = name.lower()
        if any(word in low for word in ("orchestra", "horror")):
            category = "Cinematic & Dark"
        elif any(word in low for word in ("r&b", "ballad", "motown")):
            category = "Soul & R&B"
        elif "country" in low:
            category = "Country & Crossover"
        elif any(word in low for word in ("rave", "edm", "dance", "electronic")):
            category = "Electronic & Dance"
        elif any(word in low for word in ("rock", "punk", "metal")):
            category = "Rock & Heavy"
        elif "pop" in low:
            category = "Pop"
        else:
            category = "Rap & Trap"
        presets.append({"name": name, "style": prompt, "category": category,
                        "instrumental": bool(re.search(r"\bno vocals\b|^instrumental\b", prompt, re.IGNORECASE))})
    return presets


# ------------------------------------------------------------------ my styles
# Styles people save themselves, and built-in styles they deleted. Kept in one small
# per-user file (git-ignored) so the shipped style_prompts*.txt files stay untouched.
MY_STYLES_FILE = os.path.join(APP_ROOT, "my_styles.json")
MY_STYLES_CATEGORY = "My styles"
STYLE_NAME_MAX = 60
STYLE_PROMPT_MAX = 600
_styles_lock = threading.Lock()


def style_key(name: str) -> str:
    return " ".join(str(name or "").casefold().split())


def _read_my_styles() -> dict:
    try:
        with open(MY_STYLES_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, ValueError):
        data = {}
    styles = [s for s in data.get("styles", []) if isinstance(s, dict)
              and str(s.get("name", "")).strip() and str(s.get("style", "")).strip()]
    hidden = [str(n) for n in data.get("hidden", []) if str(n).strip()]
    return {"styles": styles, "hidden": hidden}


def _write_my_styles(data: dict) -> None:
    tmp = MY_STYLES_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, MY_STYLES_FILE)


def engine_styles(quick_presets: list[dict], filename: str) -> dict:
    """One engine's quick picks + library, minus deleted built-ins, plus my styles.
    My styles come first among the quick picks and are marked mine=True."""
    mine_data = _read_my_styles()
    hidden = {style_key(n) for n in mine_data["hidden"]}
    mine = [{"name": s["name"].strip(), "style": s["style"].strip(), "mine": True,
             "engine": s.get("engine", ""), "category": MY_STYLES_CATEGORY,
             "instrumental": bool(re.search(r"\bno vocals\b|^instrumental\b", s["style"], re.IGNORECASE))}
            for s in mine_data["styles"]]
    mine_keys = {style_key(s["name"]) for s in mine}
    quick = [dict(p) for p in quick_presets
             if style_key(p["name"]) not in hidden and style_key(p["name"]) not in mine_keys]
    # Library dedupe still sees every quick pick, so a deleted chip doesn't resurface there.
    lib = [p for p in style_library(quick_presets, filename)
           if style_key(p["name"]) not in hidden and style_key(p["name"]) not in mine_keys]
    return {"presets": mine + quick, "style_library": mine + lib,
            "hidden_count": len(hidden)}


def save_my_style(name: str, style: str, engine: str, builtin_names: set[str]) -> dict:
    name = " ".join(str(name or "").split())
    style = " ".join(str(style or "").split())
    if not name:
        raise ValueError("Give the style a name first.")
    if not style:
        raise ValueError("There is no style text to save — describe the sound first.")
    if len(name) > STYLE_NAME_MAX:
        raise ValueError(f"Keep the name under {STYLE_NAME_MAX} characters.")
    if len(style) > STYLE_PROMPT_MAX:
        raise ValueError(f"Keep the style under {STYLE_PROMPT_MAX} characters.")
    key = style_key(name)
    with _styles_lock:
        data = _read_my_styles()
        hidden = {style_key(n) for n in data["hidden"]}
        if key in {style_key(n) for n in builtin_names} and key not in hidden:
            raise ValueError(f"“{name}” is already a built-in style — pick another name.")
        data["styles"] = [s for s in data["styles"] if style_key(s["name"]) != key]
        data["styles"].insert(0, {"name": name, "style": style, "engine": engine,
                                  "created": time.time()})
        _write_my_styles(data)
    return {"name": name, "style": style}


def delete_style(name: str) -> str:
    """Remove one of my styles, or hide a built-in one. Returns "mine" or "hidden"."""
    key = style_key(name)
    if not key:
        raise ValueError("Which style?")
    with _styles_lock:
        data = _read_my_styles()
        kept = [s for s in data["styles"] if style_key(s["name"]) != key]
        if len(kept) != len(data["styles"]):
            data["styles"] = kept
            outcome = "mine"
        else:
            if key not in {style_key(n) for n in data["hidden"]}:
                data["hidden"].append(" ".join(str(name).split()))
            outcome = "hidden"
        _write_my_styles(data)
    return outcome


def restore_hidden_styles(name: str = "") -> int:
    """Bring deleted built-in styles back: one by name, or all of them."""
    with _styles_lock:
        data = _read_my_styles()
        before = len(data["hidden"])
        data["hidden"] = [n for n in data["hidden"] if name and style_key(n) != style_key(name)]
        _write_my_styles(data)
    return before - len(data["hidden"])


def style_writer_prompt(engine: str) -> str:
    """System prompt for the lyrics writer (Qwen) when it writes a style prompt."""
    if engine == "ace15":
        rules = (
            "The music model is ACE-Step. Describe only what IS in the song — never write "
            "'no X' or 'without X'. Format: language, genre and subgenre, the lead vocal "
            "(gender, tone, delivery), 3-5 instruments or sounds, mood, production feel, "
            "tempo as 'NN BPM', and a musical key such as 'A minor' or 'C major'.\n"
            "Example: English, synthwave, breathy female lead vocal, analog synth arps, "
            "gated reverb drums, pulsing bass, nostalgic night drive, glossy 80s mix, 104 BPM, F minor")
    else:
        rules = (
            "The music model is YuE2. Write short comma-separated tags. Format: language, "
            "genre and subgenre, the vocals (for example 'male vocals', 'breathy female vocals'), "
            "3-5 instruments, the mood, and tempo as 'NN BPM'. Keep it under 30 words.\n"
            "Example: English, synthwave, breathy female vocals, analog synths, gated drums, "
            "pulsing bass, nostalgic night drive, 104 BPM")
    return (
        "You write music style prompts for an AI song generator. The user describes the "
        "sound they want. Reply with exactly two lines and nothing else:\n"
        "Name: <a short 1-4 word name for the style>\n"
        "Style: <the style prompt on one line>\n\n"
        "Start the style with the language of the vocals (English unless the user asks "
        "otherwise). If the user asks for an instrumental, start the style with "
        "'Instrumental'.\n" + rules)


def parse_style_reply(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    name = re.search(r"^\s*\**name\**\s*[:：]\s*(.+)$", text, re.I | re.M)
    style = re.search(r"^\s*\**style\**\s*[:：]\s*(.+)$", text, re.I | re.M)
    if style:
        prompt = style.group(1)
    else:                                   # the model ignored the format: take the longest line
        lines = [ln for ln in text.splitlines() if ln.strip() and not re.match(r"\s*\**name", ln, re.I)]
        prompt = max(lines, key=len) if lines else ""
    prompt = " ".join(prompt.strip(' "*`\t').split())[:STYLE_PROMPT_MAX]
    title = " ".join((name.group(1) if name else "").strip(' "*`\t').split())[:STYLE_NAME_MAX]
    if not prompt:
        raise ValueError("The AI didn't come up with a style — try describing it a bit differently.")
    first = prompt.split(",")[0].strip().casefold()
    if first not in _STYLE_LEADS:               # both engines read the language from the front
        prompt = "English, " + prompt
    return {"name": title or "My AI style", "style": prompt[:STYLE_PROMPT_MAX]}


_STYLE_LEADS = {
    "instrumental", "english", "spanish", "french", "german", "italian", "portuguese",
    "japanese", "korean", "chinese", "mandarin", "cantonese", "hindi", "arabic", "russian",
    "dutch", "swedish", "turkish", "polish", "vietnamese", "thai", "indonesian", "tagalog"}


# ------------------------------------------------------------------ voices
# ------------------------------------------------------------------ AI singer type
# YuE2 reads the singer's gender/character from the style prompt. Most presets say
# "male vocal", so a chosen singer replaces any gender words already in the prompt.
SINGER_PRESETS = [
    {"id": "female", "name": "Female", "prompt": "female vocal"},
    {"id": "male", "name": "Male", "prompt": "male vocal"},
    {"id": "duet", "name": "Duet", "prompt": "male and female duet vocals"},
    {"id": "choir", "name": "Choir", "prompt": "choir vocals"},
]
_GENDER = re.compile(r"\b(?:fe)?males?\b|\b(?:wo)?m[ae]n(?:'s)?\b|\b(?:boy|girl|guy|lady|ladies)s?\b"
                     r"|\b(?:masculine|feminine|duet|choir)\b", re.IGNORECASE)
_RANGES = {"female": re.compile(r"\b(?:baritone|bass|tenor)\b(?= (?:vocals?|voice|singer))|\bbaritone\b", re.IGNORECASE),
           "male": re.compile(r"\b(?:soprano|mezzo(?:-soprano)?|alto)\b", re.IGNORECASE)}
_BARE_VOICE = re.compile(r"^(?:and\s+)?(?:vocals?|voices?|singers?|vocalists?|and)?$", re.IGNORECASE)


def clean_singer(singer: str) -> str:
    """A preset id ('female') or free text ('raspy female alto') -> prompt phrase, '' = keep the style's own."""
    text = re.sub(r"[\r\n,]+", " ", str(singer or "")).strip()[:80]
    hit = next((p for p in SINGER_PRESETS if text.lower() in (p["id"], p["name"].lower())), None)
    if hit:
        return hit["prompt"]
    text = re.sub(r"\s+", " ", text)
    if text and not re.search(r"\b(?:vocals?|voice|singers?|vocalist|rap\w*|choir)\b", text, re.IGNORECASE):
        text += " vocal"
    return text


def apply_singer(style: str, singer: str) -> str:
    """Rewrite a style prompt so the AI singer is `singer`. Empty singer -> style unchanged."""
    phrase = clean_singer(singer)
    if not phrase:
        return style
    low = phrase.lower()
    wants_gender = bool(_GENDER.search(phrase))
    female = bool(re.search(r"\b(?:female|woman|women|girl|lady|feminine)\b", low)) and "male and" not in low
    male = bool(re.search(r"\b(?<!fe)(?:male|man|men|boy|guy|masculine)\b", low)) and not female and "female" not in low
    parts = []
    for seg in (s.strip() for s in style.split(",")):
        if not seg:
            continue
        if wants_gender:
            seg = _GENDER.sub("", seg)
            if female:
                seg = _RANGES["female"].sub("", seg)
            elif male:
                seg = _RANGES["male"].sub("", seg)
            seg = re.sub(r"\s{2,}", " ", seg).strip(" -")
            if _BARE_VOICE.match(seg):
                continue
        parts.append(seg)
    # Language (e.g. "English") stays first; the singer goes right after it.
    at = 1 if parts and len(parts[0].split()) <= 2 and not re.search(r"\d|vocal|voice", parts[0], re.I) else 0
    parts.insert(at, phrase)
    return ", ".join(parts)


def safe_voice_name(name: str) -> str:
    """Same rule the voice trainer applies to the name it saves under."""
    name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in (name or "").strip()).strip()
    return name


VOICE_RECORDING_TYPES = ("talking", "singing", "rapping")


def clean_voice_recording_type(recording_type: str) -> str:
    if recording_type not in VOICE_RECORDING_TYPES:
        raise ValueError("Choose Talking, Singing, or Rapping for the recording type.")
    return recording_type


def _voice_metadata(name: str) -> dict:
    if not name or safe_voice_name(name) != name:
        raise FileNotFoundError(name)
    try:
        with open(os.path.join(VOICE_DIR, name + ".json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        return meta if isinstance(meta, dict) else {}
    except (OSError, ValueError):
        return {}


def get_voice_recording_type(name: str) -> str:
    kind = _voice_metadata(name).get("recording_type", "talking")
    return kind if kind in VOICE_RECORDING_TYPES else "talking"


def set_voice_recording_type(name: str, recording_type: str) -> None:
    """Merge into the trainer's metadata without losing its fingerprint or steps."""
    import tempfile
    kind = clean_voice_recording_type(recording_type)
    meta = _voice_metadata(name)  # validates the exact name before resolving paths
    if not os.path.isfile(os.path.join(VOICE_DIR, name + ".pth")):
        raise FileNotFoundError(name)
    meta["recording_type"] = kind
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=VOICE_DIR,
                                         prefix=".recording-type-", suffix=".tmp", delete=False) as fh:
            temp_path = fh.name
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        os.replace(temp_path, os.path.join(VOICE_DIR, name + ".json"))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def voice_clip_mode(name: str) -> str:
    # Seed-VC has two reference pitch modes, not three different voice models.
    # Singing and rap performances both skip the extra speech-to-song offset.
    return "talking" if get_voice_recording_type(name) == "talking" else "singing"


def list_voices() -> list[dict]:
    voices = []
    if os.path.isdir(VOICE_DIR):
        for name in sorted(os.listdir(VOICE_DIR)):
            if not name.endswith(".pth"):
                continue
            stem = name[:-4]
            info = {"name": stem, "seconds": None, "trained": os.path.getmtime(os.path.join(VOICE_DIR, name)),
                    "has_preview": os.path.isfile(os.path.join(VOICE_DIR, stem + ".wav"))}
            meta = _voice_metadata(stem)
            info["seconds"] = meta.get("seconds")
            kind = meta.get("recording_type", "talking")
            info["recording_type"] = kind if kind in VOICE_RECORDING_TYPES else "talking"
            voices.append(info)
    return voices


def voice_preview_path(name: str) -> str:
    """Only serve the saved recording paired with an existing trained voice."""
    if not name or safe_voice_name(name) != name:
        raise FileNotFoundError(name)
    if name not in [v["name"] for v in list_voices()]:
        raise FileNotFoundError(name)
    path = os.path.join(VOICE_DIR, name + ".wav")
    if not os.path.isfile(path):
        raise FileNotFoundError(name)
    return path


def _recycle(paths: list[str]) -> bool:
    """Send files to the Windows Recycle Bin (so a delete can be undone)."""
    if os.name != "nt" or not paths:
        return False
    import ctypes
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
                    ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
                    ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
                    ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR)]

    FO_DELETE, FOF_SILENT, FOF_NOCONFIRMATION, FOF_ALLOWUNDO, FOF_NOERRORUI = 3, 0x4, 0x10, 0x40, 0x400
    op = SHFILEOPSTRUCTW(None, FO_DELETE, "\0".join(os.path.normpath(p) for p in paths) + "\0\0", None,
                         FOF_SILENT | FOF_NOCONFIRMATION | FOF_ALLOWUNDO | FOF_NOERRORUI, False, None, None)
    try:
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    except Exception:
        return False
    return result == 0 and not any(os.path.exists(p) for p in paths)


def delete_voice(name: str) -> str:
    """Remove a trained voice. Returns 'recycled' (in the Recycle Bin) or 'deleted'."""
    if name not in [v["name"] for v in list_voices()]:
        raise FileNotFoundError(name)
    paths = [os.path.join(VOICE_DIR, name + ext) for ext in (".pth", ".wav", ".json")]
    paths = [p for p in paths if os.path.isfile(p)]
    if _recycle(paths):
        return "recycled"
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    return "deleted"


# ------------------------------------------------------------------ song models
# Section 3's "Song model" choices. "full" is the YuE2-3B folder the app has
# always used. Anything named yue2*.safetensors in <models>/checkpoints is a
# single-file build of the same model; the INT8 one is what small cards need.
# Numbers measured on an RTX 5080 (scripts_test/vram_probe.py, native_probe.py).
FULL_SONG_MODEL = {"id": "full", "kind": "yue2", "name": "YuE2 3B · full", "note": "fastest, needs a 12 GB+ card",
                   "file": "", "vram_gb": 10.6, "needs_gb": 11.5, "download_gb": 7.8}
INT8_FILE = "yue2_3b_int8_convrot.safetensors"
INT8_REPO = "Comfy-Org/YuE2"


def checkpoint_path(name: str) -> str:
    """A single-file YuE2 checkpoint in <models>/checkpoints (name only, no folders)."""
    if not name or os.path.basename(name) != name or not name.endswith(".safetensors"):
        raise ValueError("Unknown song model.")
    path = os.path.join(MODELS_DIR, "checkpoints", name)
    if not os.path.isfile(path):
        raise ValueError(f"Song model '{name}' is not in the checkpoints folder.")
    return path


FULL_SONG_REPOS = (("m-a-p/YuE2-3B", "YuE2-3B"), ("m-a-p/YuE2-Vae", "YuE2-Vae"))


def yue2_root() -> str:
    return os.path.abspath(CFG.get("model_roots", {}).get("yue2", os.path.join(MODELS_DIR, "YuE2")))


def full_song_model_installed() -> bool:
    """Both folders hold a finished model.safetensors (Hugging Face only moves
    the file into place once it is complete, so a half download does not count)."""
    return all(os.path.isfile(os.path.join(yue2_root(), name, "model.safetensors"))
               for _repo, name in FULL_SONG_REPOS)


def download_full_song_model() -> None:
    """Fetch whatever is missing of YuE2-3B + YuE2-Vae (about 7.8 GB, once)."""
    from huggingface_hub import snapshot_download
    for repo, name in FULL_SONG_REPOS:
        target = os.path.join(yue2_root(), name)
        if os.path.isfile(os.path.join(target, "model.safetensors")):
            print(f"[studio] reusing {target}", flush=True)
            continue
        print(f"[studio] downloading {repo} to {target}", flush=True)
        snapshot_download(repo, local_dir=target)


def download_int8_song_model() -> None:
    """Fetch the single-file INT8 song model for small cards (about 4 GB, once)."""
    if os.path.isfile(os.path.join(MODELS_DIR, "checkpoints", INT8_FILE)):
        print(f"[studio] reusing {INT8_FILE}", flush=True)
        return
    from huggingface_hub import hf_hub_download
    print(f"[studio] downloading {INT8_REPO}/{INT8_FILE}", flush=True)
    hf_hub_download(INT8_REPO, f"checkpoints/{INT8_FILE}", local_dir=MODELS_DIR)


# ---- ACE-Step 1.5 (test). A different engine run through the vendored ComfyUI
# core, the same way ComfyUI's own "audio_ace_step1_5" workflow does it:
# UNETLoader + DualCLIPLoader(ace) + VAELoader. Never picked by "Auto".
ACE15_REPO = "Comfy-Org/ace_step_1.5_ComfyUI_files"
ACE15_FILES = {                       # role -> (models subfolder, file name)
    "unet": ("diffusion_models", "acestep_v1.5_xl_sft_bf16.safetensors"),
    "te_small": ("text_encoders", "qwen_0.6b_ace15.safetensors"),
    "te_large": ("text_encoders", "qwen_4b_ace15.safetensors"),
    "vae": ("vae", "ace_1.5_vae.safetensors"),
}
ACE15_MODEL = {"id": "ace15-xl", "kind": "ace15", "name": "ACE-Step 1.5 XL · experimental",
               "note": "lower quality than YuE2", "file": "",
               "vram_gb": 12.0, "needs_gb": 12.0, "download_gb": 19.9}


# ACE-Step reads the style as a tag list and its text encoder has no notion of
# negation: "no chorus singing" pushes toward chorus singing and "zero melody"
# fights the music, which comes out as harsh, garbled vocals. YuE2-tuned
# presets are full of such phrases, so ACE gets a cleaned copy.
_ACE_DROP = re.compile(r"^\s*(?:no|zero|without|not|never|avoid|minus)\b|^\s*like\b", re.IGNORECASE)


def ace_style(style: str) -> str:
    """A YuE2-style prompt -> ACE-Step tags: drop negated and 'like ...' phrases."""
    tags = [seg.strip() for seg in (style or "").split(",") if seg.strip()]
    kept = [tag for tag in tags if not _ACE_DROP.search(tag)]
    return ", ".join(kept or tags)


_KEY = re.compile(r"\b([A-G])\s*([#b♯♭]?)\s*(major|minor|maj|min)\b", re.IGNORECASE)
_METER = re.compile(r"\b([2346])/(4|8)\b")


def ace_song_settings(style: str) -> dict:
    """Key and time signature written in the style ("E minor", "6/8") -> ACE-Step settings.
    Missing ones are left out so ACE picks them itself."""
    found = {}
    key = _KEY.search(style or "")
    if key:
        accidental = {"♯": "#", "♭": "b", "B": "b"}.get(key.group(2), key.group(2))
        mode = "minor" if key.group(3).lower().startswith("min") else "major"
        found["keyscale"] = f"{key.group(1).upper()}{accidental} {mode}"
    meter = _METER.search(style or "")
    found["timesignature"] = int(meter.group(1)) if meter else 4
    return found


_SECTION_TAG = re.compile(r"^\s*\[([a-z][a-z \-]*?)(\s*\d*)\]\s*$", re.IGNORECASE | re.MULTILINE)


def ace_lyrics(lyrics: str) -> str:
    """[verse] / [pre-chorus] -> [Verse] / [Pre-Chorus], the form ACE-Step was trained on."""
    def title(match):
        return "[" + "-".join(p.capitalize() for p in match.group(1).strip().split("-")).title() + match.group(2) + "]"
    return _SECTION_TAG.sub(title, lyrics or "")


def ace15_path(role: str) -> str:
    """Where an ACE-Step 1.5 file is: ComfyUI's ace-step1.5 subfolder first, then
    the plain folder. Missing files resolve to the subfolder (download target)."""
    sub, name = ACE15_FILES[role]
    home = os.path.join(MODELS_DIR, sub, "ace-step1.5", name)
    for path in (home, os.path.join(MODELS_DIR, sub, name)):
        if os.path.isfile(path):
            return path
    return home


def ace15_installed() -> bool:
    return all(os.path.isfile(ace15_path(role)) for role in ACE15_FILES)


def download_ace15() -> None:
    """Fetch whatever is missing of ACE-Step 1.5 XL (up to about 20 GB, once)."""
    import shutil
    from huggingface_hub import hf_hub_download
    for role, (sub, name) in ACE15_FILES.items():
        target = ace15_path(role)
        if os.path.isfile(target):
            print(f"[studio] reusing {target}", flush=True)
            continue
        print(f"[studio] downloading {ACE15_REPO}/{name}", flush=True)
        got = hf_hub_download(ACE15_REPO, f"split_files/{sub}/{name}",
                              local_dir=os.path.join(MODELS_DIR, "_ace15_download"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(got, target)
    shutil.rmtree(os.path.join(MODELS_DIR, "_ace15_download"), ignore_errors=True)


def song_models() -> list[dict]:
    models = [dict(FULL_SONG_MODEL, installed=full_song_model_installed())]
    try:
        files = sorted(f for f in os.listdir(os.path.join(MODELS_DIR, "checkpoints"))
                       if f.lower().startswith("yue2") and f.lower().endswith(".safetensors"))
    except OSError:
        files = []
    if INT8_FILE not in files:
        files.insert(0, INT8_FILE)                    # offered even before it is downloaded
    for name in files:
        small = "int8" in name.lower()
        models.append({
            "id": name, "kind": "native", "file": name, "download_gb": 4.0 if small else None, "installed": os.path.isfile(os.path.join(MODELS_DIR, "checkpoints", name)),
            "name": "YuE2 3B · INT8" if name == INT8_FILE else name[:-len(".safetensors")],
            "note": "small, for 8 GB cards" if small else "single file",
            "vram_gb": 6.5 if small else 10.6, "needs_gb": 6.5 if small else 11.5,
        })
    models.append(dict(ACE15_MODEL, installed=ace15_installed()))   # last: experimental, YuE2 sounds better
    return models


def get_song_model(model_id: str) -> dict:
    for model in song_models():
        if model["id"] == model_id:
            return model
    raise ValueError("Unknown song model.")


def pick_song_model(total_gb: float) -> str:
    """What 'Auto' means on a card with this much memory."""
    models = {m["id"]: m for m in song_models()}
    small = models.get(INT8_FILE)
    if total_gb < 11.5 and small:                    # downloads on first song if missing
        return INT8_FILE
    if not models["full"]["installed"] and small and small["installed"]:
        return INT8_FILE
    return "full"


# ------------------------------------------------------------------ song structure
# Step 2's "Song structure": which sections the lyrics writer must produce, in
# order, and how many lines each gets. An [intro] is instrumental (no words).
SECTIONS = ["intro", "verse", "pre-chorus", "chorus", "bridge", "outro"]
DEFAULT_STRUCTURE = ["verse", "chorus", "verse", "chorus", "bridge", "outro"]
DEFAULT_LINES = 4
SECONDS_PER_LINE = 3.9            # measured: 24 sung lines -> 93 s songs
_MULTIPLIER = re.compile(r"^[ \t]*\d+[ \t]*[x×][ \t]*(?=\[)", re.IGNORECASE | re.MULTILINE)


def clean_structure(structure, lines) -> tuple[list[str], int]:
    """Validate what the page sent; fall back to the default song shape."""
    names = [str(s).strip().lower() for s in (structure or [])]
    names = [s for s in names if s in SECTIONS][:16]
    if not any(s != "intro" for s in names):
        names = list(DEFAULT_STRUCTURE)
    try:
        lines = int(lines)
    except (TypeError, ValueError):
        lines = DEFAULT_LINES
    return names, max(2, min(8, lines))


def section_lines(name: str, lines: int) -> int:
    """How many lyric lines one section gets."""
    if name == "intro":
        return 0                                  # instrumental
    if name == "pre-chorus":
        return max(2, lines // 2)
    return lines


def lyrics_length_plan(structure, lines, target_seconds=None, max_seconds=360) -> dict:
    """Plan lyric counts, not audio timing. Estimates are deliberately approximate.

    Repeated section names share a count. A bounded DP first minimizes timing
    error, then weighted deviation from a balanced shape (pre-chorus half size).
    Auto retains the historical per-section counts and token ceiling.
    """
    structure, lines = clean_structure(structure, lines)
    if target_seconds is not None:
        if type(target_seconds) is not int or not 30 <= target_seconds <= 360:
            raise ValueError("Target length must be an integer from 30 seconds to 6:00 (360 seconds).")
        if target_seconds > max_seconds:
            raise ValueError(f"Target length exceeds the {max_seconds // 60}:{max_seconds % 60:02d} graphics memory limit.")
    intro_seconds = 8 if "intro" in structure else 0
    counts = [section_lines(name, lines) for name in structure]
    if target_seconds is not None:
        groups = list(dict.fromkeys(name for name in structure if name != "intro"))
        occurrences = {name: structure.count(name) for name in groups}
        weights = {name: 0.5 if name == "pre-chorus" else 1.0 for name in groups}
        scale = ((target_seconds - intro_seconds) / SECONDS_PER_LINE /
                 sum(occurrences[name] * weights[name] for name in groups))
        # total sung lines -> (balance cost, chosen counts). Small bounded state:
        # at most 16 sections * 32 lines, independent of the requested duration.
        states = {0: (0.0, ())}
        for name in groups:
            next_states = {}
            multiple, weight = occurrences[name], weights[name]
            for total, (cost, chosen) in states.items():
                for count in range(1, 33):
                    new_total = total + multiple * count
                    candidate = (cost + multiple * (count - weight * scale) ** 2 / weight,
                                 chosen + (count,))
                    if new_total not in next_states or candidate < next_states[new_total]:
                        next_states[new_total] = candidate
            states = next_states
        best = min(states, key=lambda total: (
            abs(total * 39 + intro_seconds * 10 - target_seconds * 10),
            states[total]))
        by_name = dict(zip(groups, states[best][1]))
        counts = [0 if name == "intro" else by_name[name] for name in structure]
    total_lines = sum(counts)
    return {"counts": counts, "total_lines": total_lines,
            "estimated_seconds": round(total_lines * SECONDS_PER_LINE + intro_seconds, 1),
            "target_seconds": target_seconds, "limit_seconds": max_seconds}


def lyrics_prompt(structure: list[str], lines: int, counts=None) -> str:
    """The lyrics writer's instructions for one song shape."""
    outline = []
    for number, name in enumerate(structure, 1):
        count = section_lines(name, lines) if counts is None else counts[number - 1]
        outline.append(f"{number}. [{name}] - " + (f"{count} lines" if count else "no lines, instrumental"))
    rules = [
        "Write YuE2 song lyrics only. No title, no subtitle, no markdown, no commentary.",
        f"Follow this outline exactly, top to bottom. It has {len(structure)} sections. "
        "Stop after the last one and do not add any other section:",
        "\n".join(outline),
        "Format: the tag alone on its own line, exactly as shown (lowercase, square brackets), "
        "then that section's lines, then a blank line. Do not write the outline numbers, "
        "the line counts or '2x'.",
    ]
    if structure.count("chorus") > 1:
        rules.append("Every [chorus] repeats the same words.")
    return "\n".join(rules)


def fit_lyrics(text: str, structure: list[str], lines: int, counts=None) -> str:
    """Make the writer's text match the requested shape: sections in the asked
    order, extra sections dropped, over-long sections trimmed, a missing repeat
    of the chorus filled in. Text without any section tags is left alone."""
    blocks, name = [], None
    for raw in strip_multipliers(text).splitlines():
        line = raw.strip()
        tag = re.match(r"^\[\s*([^\]]+?)\s*\]$", line)
        if tag:
            name = re.sub(r"\s*\d+$", "", tag.group(1).strip().lower()).replace("prechorus", "pre-chorus").replace("pre chorus", "pre-chorus")
            blocks.append([name, []])
        elif line and blocks:
            blocks[-1][1].append(line)
    if not blocks:
        return text
    used, out, first = set(), [], {}
    for position, want in enumerate(structure):
        count = section_lines(want, lines) if counts is None else counts[position]
        got = None
        for index, (have, body) in enumerate(blocks):
            if index not in used and have == want and (body or not count):
                used.add(index)
                got = body
                break
        if got is None:
            got = first.get(want) if want in ("chorus", "pre-chorus") else None
        if got is None and count:
            continue                               # the writer skipped it; nothing to sing
        body = (got or [])[:count]
        first.setdefault(want, body)
        out.append("\n".join([f"[{want}]"] + body))
    return "\n\n".join(out) if any("\n" in block for block in out) else text


def lyrics_budget(structure: list[str], lines: int, counts=None) -> int:
    """How many tokens the writer may use for this shape."""
    sung = sum(section_lines(name, lines) for name in structure) if counts is None else sum(counts)
    return max(260, min(1500 if counts is None else 3000, 90 + sung * 24))


def strip_multipliers(text: str) -> str:
    """'2x [verse]' -> '[verse]'. The song model would sing the '2x' out loud
    and the section tag would be lost."""
    return _MULTIPLIER.sub("", text or "")


# ------------------------------------------------------------------ lyrics writers
# Section 1's "Lyrics writer" choices. `vram_gb` and `load_s` were measured on
# an RTX 5080 with scripts_test/lyrics_probe.py (model loaded and writing).
# The 8-bit / 4-bit options reuse the 4B files already on disk — nothing extra
# to download — and shrink the model as it loads (bitsandbytes).
LYRICS_WRITERS = [
    {"id": "qwen4b", "name": "Qwen 4B", "note": "full size", "bits": 16,
     "folder": "Qwen3-VL-4B-Instruct", "repo": "Qwen/Qwen3-VL-4B-Instruct",
     "vram_gb": 8.4, "download_gb": 8.3},
    {"id": "qwen4b-4bit", "name": "Qwen 4B · 4-bit", "note": "half the memory, same speed", "bits": 4,
     "folder": "Qwen3-VL-4B-Instruct", "repo": "Qwen/Qwen3-VL-4B-Instruct",
     "vram_gb": 4.1, "download_gb": 8.3},
    {"id": "qwen4b-8bit", "name": "Qwen 4B · 8-bit", "note": "half the memory, 4x slower", "bits": 8,
     "folder": "Qwen3-VL-4B-Instruct", "repo": "Qwen/Qwen3-VL-4B-Instruct",
     "vram_gb": 4.6, "download_gb": 8.3},
    {"id": "qwen2b", "name": "Qwen 2B", "note": "smaller model, fastest", "bits": 16,
     "folder": "Qwen3-VL-2B-Instruct", "repo": "Qwen/Qwen3-VL-2B-Instruct",
     "vram_gb": 4.1, "download_gb": 4.3},
]
DEFAULT_WRITER = "qwen4b"
SMALL_CARD_WRITER = "qwen4b-4bit"    # Auto on cards under 11.5 GB


def writer_dir(writer: dict) -> str:
    """Where this writer's files are (or will be downloaded to)."""
    home = os.path.join(MODELS_DIR, "LLM", "Qwen-VL", writer["folder"])
    for path in (home, os.path.join(MODELS_DIR, writer["folder"])):
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
    return home


def writer_installed(writer: dict) -> bool:
    path = writer_dir(writer)
    try:
        return os.path.isfile(os.path.join(path, "config.json")) and any(
            name.endswith(".safetensors") for name in os.listdir(path))
    except OSError:
        return False


def get_writer(writer_id: str) -> dict:
    for writer in LYRICS_WRITERS:
        if writer["id"] == writer_id:
            return writer
    raise KeyError(writer_id)


def lyrics_writers() -> list[dict]:
    return [{"id": w["id"], "name": w["name"], "note": w["note"], "vram_gb": w["vram_gb"],
             "download_gb": w["download_gb"], "installed": writer_installed(w)}
            for w in LYRICS_WRITERS]


# ------------------------------------------------------------------ models
def model_check() -> list[dict]:
    """What the pipeline needs on disk, for the first-run banner."""
    yue2 = yue2_root()
    return [
        {"name": "YuE2-3B (song model)", "path": os.path.join(yue2, "YuE2-3B"),
         "ok": os.path.isfile(os.path.join(yue2, "YuE2-3B", "model.safetensors")), "required": True},
        {"name": "YuE2-Vae (audio decoder)", "path": os.path.join(yue2, "YuE2-Vae"),
         "ok": os.path.isfile(os.path.join(yue2, "YuE2-Vae", "model.safetensors")), "required": True},
    ]
