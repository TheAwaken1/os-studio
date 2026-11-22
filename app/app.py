"""OS Studio — local web app (no ComfyUI server, no Gradio).

    start.bat            ->  python app.py
    http://127.0.0.1:7861

A small FastAPI server drives the pipeline in `service.py` and serves the
single-page UI in `web/`. One GPU job runs at a time; the page polls
`/api/job` for progress.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(ROOT, "shims"), os.path.join(ROOT, "engine"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import library  # noqa: E402
import update_check  # noqa: E402

APP_ID = "yue2-song-studio"
WEB = os.path.join(ROOT, "web")
LOG_DIR = os.path.join(ROOT, "logs")
CLIP_DIR = os.path.join(os.path.abspath(library.CFG["input_dir"]), "voice_clips")

STYLE_PRESETS = [
    {"name": "Spoken rap", "style": (
        "English, male voiceover rap, spoken word over a beat, zero melody, "
        "conversational, like reading lines to a friend, no chorus singing, "
        "minimal drums, 84 BPM")},
    {"name": "Pop", "style": "English, upbeat pop, bright vocals, catchy chorus, synth, 110 BPM"},
    {"name": "Country", "style": (
        "English, country ballad, warm male vocals, acoustic guitar, fiddle, "
        "storytelling, 90 BPM")},
    {"name": "Blues", "style": (
        "English, slow blues, gritty male vocals, slide guitar, brushed drums, "
        "smoky bar, 70 BPM")},
    {"name": "Rock", "style": "English, hard rock, powerful vocals, distorted guitars, 130 BPM"},
]

# The same quick picks written for ACE-Step 1.5: plain descriptions of what is in the
# song (ACE has no "no X"), plus a key. The page swaps to these when ACE-Step is picked;
# matching names let a chosen preset follow the switch.
ACE_STYLE_PRESETS = [
    {"name": "Spoken rap", "style": (
        "English, hip hop, male rap vocal, laid-back conversational flow, spoken-word "
        "storytelling, dry vocal upfront, minimal boom bap drums, deep bass, 84 BPM, A minor")},
    {"name": "Pop", "style": (
        "English, pop, bright lead vocal, catchy chorus, synth pop, punchy drums, "
        "polished radio mix, upbeat, 110 BPM, C major")},
    {"name": "Country", "style": (
        "English, country ballad, warm male vocal, acoustic guitar, fiddle, steel guitar, "
        "heartfelt storytelling, 90 BPM, G major")},
    {"name": "Blues", "style": (
        "English, slow blues, gritty soulful male vocal, slide guitar, brushed drums, "
        "smoky bar atmosphere, warm analog mix, 70 BPM, E minor")},
    {"name": "Rock", "style": (
        "English, hard rock, powerful male vocal, distorted guitars, driving drums, "
        "big anthemic chorus, 130 BPM, E minor")},
]


# ------------------------------------------------------------------ log capture
class LogBook:
    """Keeps the last lines printed by the engine for the Activity drawer."""

    def __init__(self, path: str, keep: int = 600):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 2_000_000:
            os.replace(path, path + ".old")
        self._file = open(path, "a", encoding="utf-8", errors="replace")
        self._lines = collections.deque(maxlen=keep)
        self._seq = 0
        self._partial = ""
        self._lock = threading.Lock()

    def feed(self, text: str) -> None:
        with self._lock:
            self._partial += text
            while True:
                cut = min((i for i in (self._partial.find("\n"), self._partial.find("\r")) if i >= 0),
                          default=-1)
                if cut < 0:
                    break
                line, self._partial = self._partial[:cut].rstrip(), self._partial[cut + 1:]
                if line:
                    self._seq += 1
                    self._lines.append((self._seq, line))
                    self._file.write(time.strftime("%H:%M:%S ") + line + "\n")
            self._file.flush()

    def since(self, seq: int):
        with self._lock:
            return [{"n": n, "text": t} for n, t in self._lines if n > seq], self._seq


class Tee:
    def __init__(self, stream, book: LogBook):
        self._stream, self._book = stream, book

    def write(self, text):
        if self._stream is not None:
            try:
                self._stream.write(text)
            except Exception:
                pass
        self._book.feed(text)
        return len(text)

    def flush(self):
        if self._stream is not None:
            try:
                self._stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


LOG = LogBook(os.path.join(LOG_DIR, "studio.log"))
sys.stdout = Tee(sys.stdout, LOG)
sys.stderr = Tee(sys.stderr, LOG)


# ------------------------------------------------------------------ engine
class Engine:
    """Imports the heavy pipeline in the background so the page opens at once."""

    def __init__(self):
        self.state = "starting"      # starting | ready | error
        self.error = ""
        self.service = None
        self._ready = threading.Event()

    def warm_up(self):
        try:
            import service
            self.service = service
            self.state = "ready"
            print("[studio] engine ready", flush=True)
        except Exception as exc:
            traceback.print_exc()
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()

    def get(self):
        self._ready.wait()
        if self.state != "ready":
            raise RuntimeError(f"The engine failed to start — {self.error}")
        return self.service


ENGINE = Engine()


# ------------------------------------------------------------------ jobs
class Busy(RuntimeError):
    pass


class Jobs:
    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self.job: dict | None = None

    def start(self, kind: str, fn) -> dict:
        with self._lock:
            if self.job and self.job["state"] == "running":
                raise Busy(self.job["kind"])
            self._seq += 1
            job = {"id": self._seq, "kind": kind, "state": "running", "started": time.time(),
                   "ended": None, "partial": None, "result": None, "error": None}
            self.job = job
        threading.Thread(target=self._run, args=(job, fn), daemon=True).start()
        return job

    def _run(self, job, fn):
        try:
            service = ENGINE.get()
            job["result"] = fn(service, job)
            job["state"] = "done"
        except Exception as exc:
            if type(exc).__name__ == "Cancelled":
                job["state"] = "cancelled"
            else:
                job["state"], job["error"] = "error", friendly_error(exc)
                print(f"[studio] {job['kind']} failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            job["ended"] = time.time()

    def snapshot(self):
        job = self.job
        if not job:
            return None
        out = {k: job[k] for k in ("id", "kind", "state", "partial", "result", "error")}
        out["elapsed"] = round((job["ended"] or time.time()) - job["started"], 1)
        out["progress"] = None
        if job["state"] == "running":
            if ENGINE.state == "ready":
                out["progress"] = ENGINE.service.progress()
            else:
                out["progress"] = {"overall": 0.0, "stage": "engine", "label": "Starting the engine",
                                   "stage_frac": 0.0, "steps": []}
        return out


JOBS = Jobs()


def friendly_error(exc: Exception) -> str:
    text = f"{exc}".strip() or type(exc).__name__
    low = text.lower()
    if "out of memory" in low:
        return ("The graphics card ran out of memory. Close other GPU apps "
                "(ComfyUI, games, browsers with video) and try again.")
    if isinstance(exc, FileNotFoundError) or "no such file" in low:
        return f"A model file is missing: {text}"
    return text.replace("[SongVoice] ", "").replace("[SongLRC] ", "")


# ------------------------------------------------------------------ API
app = FastAPI(title="OS Studio", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB), name="static")


class LyricsBody(BaseModel):
    idea: str
    writer: str = ""
    structure: list[str] = []
    lines: int = library.DEFAULT_LINES
    target_seconds: int | None = Field(default=None, strict=True)
    memory: str = "auto"


class LyricsPlanBody(BaseModel):
    structure: list[str] = []
    lines: int = library.DEFAULT_LINES
    target_seconds: int | None = Field(default=None, strict=True)
    memory: str = "auto"


class SongBody(BaseModel):
    lyrics: str
    style: str
    seed: int | None = None
    cot: str = "off"
    voice: str = ""
    singer: str = ""
    title: str = ""
    idea: str = ""
    memory: str = "auto"
    model: str = "auto"


def _gpu():
    if ENGINE.state != "ready":
        return None
    try:
        return ENGINE.service.gpu_status()
    except Exception:
        return None


_CARD_GB: list = []


def _card_total_gb() -> float | None:
    """The graphics card's total memory, known before the engine has started."""
    gpu = _gpu()
    if gpu and gpu.get("total_gb"):
        return gpu["total_gb"]
    if not _CARD_GB:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout
            _CARD_GB.append(int(out.strip().splitlines()[0]) / 1024)
        except Exception:
            _CARD_GB.append(None)
    return _CARD_GB[0]


def _default_writer() -> str:
    """Auto's lyrics writer: the full-size Qwen 4B needs 8.4 GB, so smaller cards get
    the 4-bit build (4.1 GB, same speed) — the same cut-off Auto uses for the song model."""
    total = _card_total_gb()
    if total is not None and total < 11.5:
        return library.SMALL_CARD_WRITER
    return library.DEFAULT_WRITER


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(WEB, "index.html"), encoding="utf-8") as fh:
        html = fh.read()
    stamp = str(int(max(os.path.getmtime(os.path.join(WEB, f)) for f in os.listdir(WEB))))
    return HTMLResponse(html.replace("{{v}}", stamp), headers={"Cache-Control": "no-store"})


@app.get("/api/ping")
def ping():
    return {"app": APP_ID}


@app.get("/api/update")
def api_update():
    """Is there a newer OS Studio on GitHub? Off with "update_check": false in config.json."""
    if library.CFG.get("update_check", True) is False:
        return {"available": False}
    return update_check.cached_check()


@app.get("/api/state")
def state():
    styles = _style_sets()
    return {
        "engine": {"state": ENGINE.state, "error": ENGINE.error},
        "gpu": _gpu(),
        "job": JOBS.snapshot(),
        "presets": styles["yue2"]["presets"],
        "style_library": styles["yue2"]["style_library"],
        "engine_styles": styles,
        "singers": library.SINGER_PRESETS,
        "voices": library.list_voices(),
        "writers": library.lyrics_writers(),
        "default_writer": _default_writer(),
        "song_models": library.song_models(),
        "sections": library.SECTIONS,
        "default_structure": library.DEFAULT_STRUCTURE,
        "default_lines": library.DEFAULT_LINES,
        "seconds_per_line": library.SECONDS_PER_LINE,
        "auto_song_model": library.pick_song_model((_gpu() or {}).get("total_gb") or 99.0),
        "models": library.model_check(),
        "songs_dir": library.OUTPUT_DIR,
    }


STYLE_FILES = {"yue2": (STYLE_PRESETS, "style_prompts.txt"),
               "ace15": (ACE_STYLE_PRESETS, "style_prompts_ace.txt")}


def _style_sets() -> dict:
    return {kind: library.engine_styles(quick, filename) for kind, (quick, filename) in STYLE_FILES.items()}


def _builtin_style_names() -> set[str]:
    names = set()
    for quick, filename in STYLE_FILES.values():
        names.update(p["name"] for p in quick)
        names.update(p["name"] for p in library.style_library(quick, filename))
    return names


class StyleSaveBody(BaseModel):
    name: str
    style: str
    engine: str = "yue2"


class StyleDeleteBody(BaseModel):
    name: str


class StyleAIBody(BaseModel):
    idea: str
    engine: str = "yue2"
    writer: str = ""


@app.get("/api/styles")
def api_styles():
    return {"engine_styles": _style_sets()}


@app.post("/api/styles")
def api_save_style(body: StyleSaveBody):
    if body.engine not in STYLE_FILES:
        raise HTTPException(400, "Unknown song engine.")
    try:
        saved = library.save_my_style(body.name, body.style, body.engine, _builtin_style_names())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"saved": saved, "engine_styles": _style_sets()}


@app.post("/api/styles/delete")
def api_delete_style(body: StyleDeleteBody):
    try:
        outcome = library.delete_style(body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    return {"outcome": outcome, "engine_styles": _style_sets()}


class StyleRestoreBody(BaseModel):
    name: str = ""


@app.post("/api/styles/restore")
def api_restore_styles(body: StyleRestoreBody):
    return {"restored": library.restore_hidden_styles(body.name), "engine_styles": _style_sets()}


@app.post("/api/style-ai")
def api_style_ai(body: StyleAIBody):
    idea = " ".join(body.idea.split())
    if not idea:
        raise HTTPException(400, "Describe the sound you want first, e.g. dreamy late-night R&B with a female singer.")
    if len(idea) > 400:
        raise HTTPException(400, "Keep the description under 400 characters.")
    if body.engine not in STYLE_FILES:
        raise HTTPException(400, "Unknown song engine.")
    writer = body.writer or _default_writer()
    if writer not in [w["id"] for w in library.LYRICS_WRITERS]:
        raise HTTPException(400, "Unknown lyrics writer.")

    def run(service, job):
        return {"style": service.write_style(idea, engine=body.engine, writer=writer)}

    return _start("style", run)


@app.get("/api/job")
def job(since: int = 0):
    lines, seq = LOG.since(since)
    return {"job": JOBS.snapshot(), "engine": {"state": ENGINE.state, "error": ENGINE.error},
            "gpu": _gpu(), "log": lines, "log_seq": seq}


def _start(kind, fn):
    try:
        return {"job": JOBS.start(kind, fn)["id"]}
    except Busy as busy:
        raise HTTPException(409, f"Still busy with the current {busy} job.") from None


def _lyrics_memory(memory: str) -> str:
    if memory not in ("auto", "standard", "low"):
        raise HTTPException(400, "Unknown graphics memory setting.")
    if memory != "auto":
        return memory
    total_gb = (_gpu() or {}).get("total_gb")
    return "low" if total_gb is not None and total_gb < 14 else "standard"


def _lyrics_plan(body, memory: str) -> dict:
    try:
        return library.lyrics_length_plan(body.structure, body.lines, body.target_seconds,
                                          max_seconds=180 if memory == "low" else 360)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@app.post("/api/lyrics-plan")
def api_lyrics_plan(body: LyricsPlanBody):
    """Read-only estimate: no engine warm-up, GPU job, or lyric edits."""
    return _lyrics_plan(body, _lyrics_memory(body.memory))


@app.post("/api/lyrics")
def api_lyrics(body: LyricsBody):
    idea = body.idea.strip()
    if not idea:
        raise HTTPException(400, "Type what the song should be about first.")
    writer = body.writer or _default_writer()
    if writer not in [w["id"] for w in library.LYRICS_WRITERS]:
        raise HTTPException(400, "Unknown lyrics writer.")
    memory = _lyrics_memory(body.memory)
    _lyrics_plan(body, memory)                     # reject before scheduling a GPU job
    extra = {}
    if body.target_seconds is not None:
        # Freeze Auto's preview profile, including standard when GPU is unknown.
        extra = {"target_seconds": body.target_seconds, "memory": memory}
    elif body.memory != "auto":
        extra = {"memory": memory}

    def run(service, job):
        def on_text(text):
            job["partial"] = text
        return {"lyrics": service.write_lyrics(idea, on_text=on_text, writer=writer,
                                               structure=body.structure, lines=body.lines, **extra)}

    return _start("lyrics", run)


@app.post("/api/song")
def api_song(body: SongBody):
    if not body.lyrics.strip():
        raise HTTPException(400, "There are no lyrics yet — write or paste some first.")
    if not body.style.strip():
        raise HTTPException(400, "Pick a style or describe one.")
    if body.cot not in ("off", "melody", "full"):
        raise HTTPException(400, "Unknown melody planning mode.")
    if body.memory not in ("auto", "standard", "low"):
        raise HTTPException(400, "Unknown graphics memory setting.")
    if body.model != "auto":
        try:
            picked = library.get_song_model(body.model)
        except ValueError:
            raise HTTPException(400, "Unknown song model.") from None
        if not picked["installed"] and picked["id"] not in ("full", "ace15-xl", library.INT8_FILE):
            raise HTTPException(400, f"The song model '{picked['name']}' is not installed on this PC.")

    def run(service, job):
        return {"song": service.make_song(
            lyrics=body.lyrics, style=body.style.strip(), seed=body.seed, cot=body.cot,
            voice=body.voice, title=body.title, idea=body.idea, memory=body.memory,
            model=body.model, singer=body.singer)}

    return _start("song", run)


@app.post("/api/cancel")
def api_cancel():
    snap = JOBS.snapshot()
    if snap and snap["state"] == "running" and ENGINE.state == "ready":
        ENGINE.service.cancel()
    return {"ok": True}


@app.get("/api/songs")
def api_songs():
    return {"songs": library.list_songs()}


@app.delete("/api/songs/{sid}")
def api_delete_song(sid: str):
    try:
        library.delete_song(sid)
    except FileNotFoundError:
        raise HTTPException(404, "That song is no longer on disk.") from None
    return {"ok": True}


@app.get("/api/songs/{sid}/lrc", response_class=PlainTextResponse)
def api_lrc(sid: str, dl: int = 0):
    try:
        path = library.sidecar(library.song_path(sid), ".lrc")
        if dl:
            return FileResponse(path, media_type="text/plain", filename=os.path.basename(path))
        return PlainTextResponse(library.read_lrc(sid))
    except FileNotFoundError:
        raise HTTPException(404, "No timed lyrics for this song.") from None


@app.get("/api/songs/{sid}/viz")
def api_song_viz(sid: str, v: str = ""):
    """Measured waveform + spectrum for the player. Works on every device, no Web Audio."""
    try:
        path = library.song_path(sid)
    except FileNotFoundError:
        raise HTTPException(404, "That song is no longer on disk.") from None
    if v == "ai":
        ai = library.sidecar(path, ".ai.mp3")
        if os.path.isfile(ai):
            path = ai
    try:
        return JSONResponse(library.song_viz(path), headers={"Cache-Control": "private, max-age=3600"})
    except Exception as exc:  # the player falls back to a plain progress bar
        raise HTTPException(422, f"This song could not be analysed: {exc}") from None


@app.get("/media/{sid}")
def media(sid: str, v: str = "", dl: int = 0):
    try:
        path = library.song_path(sid)
    except FileNotFoundError:
        raise HTTPException(404, "That song is no longer on disk.") from None
    if v == "ai":
        ai = library.sidecar(path, ".ai.mp3")
        if os.path.isfile(ai):
            path = ai
    return FileResponse(path, media_type="audio/mpeg",
                        filename=os.path.basename(path) if dl else None)


@app.get("/api/voices")
def api_voices():
    return {"voices": library.list_voices()}


@app.get("/api/voices/{name}/preview")
def api_voice_preview(name: str):
    try:
        path = library.voice_preview_path(name)
    except FileNotFoundError:
        raise HTTPException(404, "No saved recording is available for this voice.") from None
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.post("/api/voices")
async def api_train_voice(request: Request, name: str, replace: int = 0, recording_type: str = "talking"):
    try:
        recording_type = library.clean_voice_recording_type(recording_type)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    safe = library.safe_voice_name(name)
    if not safe:
        raise HTTPException(400, "Give the voice a name.")
    if not replace and safe in [v["name"] for v in library.list_voices()]:
        raise HTTPException(409, f"A voice called '{safe}' already exists.")
    data = await request.body()
    if len(data) < 20_000:
        raise HTTPException(400, "That clip is empty or too short.")
    ext = os.path.splitext(request.headers.get("x-filename", ""))[1].lower()
    if ext not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"):
        ext = ".wav"
    os.makedirs(CLIP_DIR, exist_ok=True)
    clip = os.path.join(CLIP_DIR, f"{safe}_{time.strftime('%Y%m%d_%H%M%S')}{ext}")
    with open(clip, "wb") as fh:
        fh.write(data)

    def run(service, job):
        return {"voice": service.train_voice(clip, safe, retrain=bool(replace), recording_type=recording_type)}

    return _start("voice", run)


@app.delete("/api/voices/{name}")
def api_delete_voice(name: str):
    if JOBS.job and JOBS.job["state"] == "running":
        raise HTTPException(409, "Wait for the current job to finish, then delete the voice.")
    try:
        how = library.delete_voice(name)
    except FileNotFoundError:
        raise HTTPException(404, "That voice is no longer on this PC.") from None
    except OSError as exc:
        raise HTTPException(500, f"Could not delete the voice: {exc}") from None
    service = sys.modules.get("service")
    if service is not None:
        service.forget_voice(name)
    print(f"[studio] voice '{name}' {how}", flush=True)
    return {"ok": True, "how": how, "voices": library.list_voices()}


@app.post("/api/open-folder")
def api_open_folder(sid: str = ""):
    try:
        if sid:
            target = os.path.normpath(library.song_path(sid))
            if not os.path.exists(target):
                raise FileNotFoundError(target)
        else:
            os.makedirs(library.OUTPUT_DIR, exist_ok=True)
            target = library.OUTPUT_DIR
        _reveal(target)
    except FileNotFoundError:
        raise HTTPException(404, "That song is no longer on disk.") from None
    except OSError:
        raise HTTPException(501, f"Couldn't open a file browser on this PC. Your songs are in {library.OUTPUT_DIR}") from None
    return {"ok": True}


def _reveal(path: str) -> None:
    """Show a file (selected) or folder in the system file browser."""
    if sys.platform == "win32":
        if os.path.isdir(path):
            os.startfile(path)  # noqa: S606 (local desktop app)
        else:
            subprocess.Popen(["explorer", "/select,", path])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", path] if os.path.isfile(path) else ["open", path])
    else:                                   # Linux: file managers can't select, so open the folder
        opener = shutil.which("xdg-open")
        if not opener:
            raise OSError("xdg-open not found")
        subprocess.Popen([opener, path if os.path.isdir(path) else os.path.dirname(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ------------------------------------------------------------------ launch
def _ping(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ping", timeout=1.5) as resp:
            return json.load(resp).get("app") == APP_ID
    except Exception:
        return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _edge() -> str | None:
    found = shutil.which("msedge")
    if found:
        return found
    for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES"),
                 os.environ.get("LOCALAPPDATA")):
        if base:
            path = os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe")
            if os.path.isfile(path):
                return path
    return None


def open_window(url: str) -> None:
    """Own window (Edge app mode) when possible, else the default browser."""
    edge = _edge() if library.CFG.get("window", "app") == "app" else None
    if edge:
        try:
            subprocess.Popen([edge, f"--app={url}", "--window-size=1500,960"])
            return
        except OSError:
            pass
    webbrowser.open(url)


def _quiet_asyncio() -> None:
    """Windows logs a scary traceback every time the browser drops an idle
    connection (WinError 10054 in _call_connection_lost). It is harmless."""
    import logging

    class DropResets(logging.Filter):
        def filter(self, record):
            return "_call_connection_lost" not in record.getMessage()

    logging.getLogger("asyncio").addFilter(DropResets())


def main() -> int:
    import uvicorn

    _quiet_asyncio()
    parser = argparse.ArgumentParser(description="OS Studio")
    parser.add_argument("--port", type=int, default=int(library.CFG.get("port", 7861)))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    port = args.port
    if _ping(port):
        print(f"OS Studio is already running at http://127.0.0.1:{port} — opening it.")
        if not args.no_browser:
            open_window(f"http://127.0.0.1:{port}")
        return 0
    while not _port_free(port):
        port += 1
        if port > args.port + 20:
            print(f"Ports {args.port}-{port} are all in use.")
            return 1
    if port != args.port:
        print(f"Port {args.port} is taken by another program; using {port}.")

    url = f"http://127.0.0.1:{port}"
    print(f"\n  OS Studio  ->  {url}\n  Keep this window open. Close it to quit.\n", flush=True)
    threading.Thread(target=ENGINE.warm_up, daemon=True).start()

    if not args.no_browser:
        def launch():
            for _ in range(60):
                if _ping(port):
                    open_window(url)
                    return
                time.sleep(0.25)
        threading.Thread(target=launch, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
