"""
justdream web UI — continuity story → video.
Run: python webui.py → http://127.0.0.1:7862
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

os.environ.setdefault("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")

from app import (
    DEFAULT_FRAMES,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_PRESET,
    DEFAULT_STYLE_LOCK,
    DEFAULT_WIDTH,
    STORIES_DIR,
    UPLOADS_DIR,
    estimate_duration_s,
    estimate_wall_clock_s,
    format_duration,
    get_preset,
    list_presets,
    split_script_to_beats,
)

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PORT = 7862

_state_lock = threading.Lock()
_queue_lock = threading.Lock()
_state = {
    "busy": False,
    "status": "idle",
    "progress": None,
    "error": None,
    "video": None,
    "clips": [],
    "is_final": False,
    "job_id": None,
    "queue_len": 0,
    "messages": [],
    "stage_started": None,
    "worker_ready": False,
    "device_mode": None,
}
_job_queue: deque = deque()
_mp_ctx = mp.get_context("spawn")
_job_q: mp.Queue | None = None
_event_q: mp.Queue | None = None
_worker_proc: mp.Process | None = None
_dispatcher_wake = threading.Event()
_started = False


def _worker_main(job_q: mp.Queue, event_q: mp.Queue) -> None:
    os.environ.setdefault("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")

    from app import DEFAULT_PRESET, DEFAULT_STYLE_LOCK, LTXDream, find_resumable_story

    def emit(**payload) -> None:
        try:
            event_q.put(payload, timeout=1.0)
        except Exception:
            pass

    def on_status(msg: str, progress: float | None = None, **extra) -> None:
        payload = {"type": "progress", "status": msg}
        if progress is not None:
            payload["progress"] = progress
        if extra.get("video"):
            payload["video"] = extra["video"]
        if extra.get("clips") is not None:
            payload["clips"] = extra["clips"]
        if "is_final" in extra:
            payload["is_final"] = bool(extra["is_final"])
        emit(**payload)

    engine = LTXDream()
    emit(type="ready", device_mode=None)

    resumable = find_resumable_story()
    if resumable is not None:
        emit(type="job_start", job_id="resume", status=f"resuming {resumable.name}…", progress=0.0)
        try:
            path = engine.generate_story(script="", story_dir=resumable, on_status=on_status)
            emit(
                type="job_done",
                job_id="resume",
                video=path.name,
                device_mode=engine.device_mode,
                text=f"Resumed story ready: {path.name}",
            )
        except Exception as exc:
            emit(type="job_error", job_id="resume", error=str(exc), device_mode=engine.device_mode)

    while True:
        job = job_q.get()
        if job is None:
            break
        job_id = job["id"]
        emit(type="job_start", job_id=job_id, status="starting continuity story…", progress=0.0)
        try:
            path = engine.generate_story(
                script=job.get("script") or "",
                story_dir=Path(job["story_dir"]) if job.get("story_dir") else None,
                seed=job.get("seed"),
                preset=job.get("preset") or DEFAULT_PRESET,
                style_lock=job.get("style_lock") or DEFAULT_STYLE_LOCK,
                ref_paths=job.get("ref_paths") or [],
                on_status=on_status,
            )
            emit(
                type="job_done",
                job_id=job_id,
                video=path.name,
                device_mode=engine.device_mode,
                text=f"Story ready: {path.name}",
            )
        except Exception as exc:
            emit(type="job_error", job_id=job_id, error=str(exc), device_mode=engine.device_mode)


def _set(**kwargs) -> None:
    with _state_lock:
        _state.update(kwargs)


def _snapshot() -> dict:
    with _state_lock:
        started = _state.get("stage_started")
        elapsed = int(time.time() - started) if started and _state["busy"] else 0
        status = _state["status"]
        if _state["busy"] and elapsed >= 2:
            status = f"{status} · {elapsed // 60}m {elapsed % 60}s"
        progress = _state["progress"]
        pct = int(round(100 * progress)) if isinstance(progress, (int, float)) else None
        return {
            "busy": _state["busy"],
            "status": status,
            "progress": progress,
            "progress_pct": pct,
            "error": _state["error"],
            "video": _state["video"],
            "clips": list(_state.get("clips") or []),
            "is_final": bool(_state.get("is_final")),
            "job_id": _state["job_id"],
            "queue_len": _state["queue_len"],
            "messages": list(_state["messages"]),
            "elapsed_s": elapsed,
            "worker_ready": _state["worker_ready"],
            "device_mode": _state["device_mode"],
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
            if torch.cuda.is_available()
            else None,
            "defaults": {
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "frames": DEFAULT_FRAMES,
                "fps": DEFAULT_FPS,
                "preset": DEFAULT_PRESET,
                "style_lock": DEFAULT_STYLE_LOCK,
            },
            "presets": list_presets(),
        }


def _append_message(msg: dict) -> None:
    with _state_lock:
        _state["messages"].append(msg)
        if len(_state["messages"]) > 60:
            _state["messages"] = _state["messages"][-60:]


def _event_listener() -> None:
    assert _event_q is not None
    while True:
        try:
            ev = _event_q.get()
        except Exception:
            time.sleep(0.2)
            continue
        if ev is None:
            break
        et = ev.get("type")
        if et == "ready":
            _set(worker_ready=True, status="idle")
        elif et == "progress":
            payload = {"status": ev.get("status", "working")}
            if "progress" in ev and ev["progress"] is not None:
                with _state_lock:
                    prev = _state.get("progress")
                new_p = float(ev["progress"])
                if isinstance(prev, (int, float)):
                    new_p = max(float(prev), new_p)
                payload["progress"] = new_p
            if ev.get("video"):
                payload["video"] = ev["video"]
            if ev.get("clips") is not None:
                payload["clips"] = ev["clips"]
            if "is_final" in ev:
                payload["is_final"] = bool(ev["is_final"])
            _set(**payload)
        elif et == "job_start":
            _set(
                busy=True,
                job_id=ev.get("job_id"),
                error=None,
                progress=0.0,
                status=ev.get("status", "starting…"),
                stage_started=time.time(),
                clips=[],
                is_final=False,
                video=None,
            )
        elif et == "job_done":
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "job_id": ev.get("job_id"),
                    "text": ev.get("text") or "Done",
                    "video": ev.get("video"),
                    "ts": time.time(),
                }
            )
            _set(
                busy=False,
                status="done",
                progress=1.0,
                video=ev.get("video"),
                is_final=True,
                device_mode=ev.get("device_mode"),
                job_id=None,
                error=None,
                stage_started=None,
            )
            _dispatcher_wake.set()
        elif et == "job_error":
            err = ev.get("error") or "unknown error"
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "job_id": ev.get("job_id"),
                    "text": f"Error: {err}",
                    "error": True,
                    "ts": time.time(),
                }
            )
            _set(
                busy=False,
                status="error",
                progress=None,
                error=err,
                device_mode=ev.get("device_mode"),
                job_id=None,
                stage_started=None,
            )
            _dispatcher_wake.set()


def _dispatcher_loop() -> None:
    assert _job_q is not None
    while True:
        with _state_lock:
            busy = _state["busy"]
            ready = _state["worker_ready"]
        if busy or not ready:
            _dispatcher_wake.wait(timeout=0.5)
            _dispatcher_wake.clear()
            continue
        job = None
        with _queue_lock:
            if _job_queue:
                job = _job_queue.popleft()
                _set(queue_len=len(_job_queue))
        if job is None:
            _dispatcher_wake.wait(timeout=0.5)
            _dispatcher_wake.clear()
            continue
        _set(busy=True, job_id=job["id"], status="handing off…", progress=0.0, error=None, stage_started=time.time())
        _job_q.put(job)


def _start_runtime() -> None:
    global _job_q, _event_q, _worker_proc, _started
    if _started:
        return
    _started = True
    _job_q = _mp_ctx.Queue()
    _event_q = _mp_ctx.Queue()
    _worker_proc = _mp_ctx.Process(target=_worker_main, args=(_job_q, _event_q), name="dream-worker", daemon=True)
    _worker_proc.start()
    threading.Thread(target=_event_listener, name="dream-events", daemon=True).start()
    threading.Thread(target=_dispatcher_loop, name="dream-dispatch", daemon=True).start()


def _stop_runtime() -> None:
    global _started, _worker_proc
    if _job_q is not None:
        try:
            _job_q.put(None)
        except Exception:
            pass
    if _event_q is not None:
        try:
            _event_q.put(None)
        except Exception:
            pass
    if _worker_proc is not None and _worker_proc.is_alive():
        _worker_proc.join(timeout=5)
        if _worker_proc.is_alive():
            _worker_proc.terminate()
    _started = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _start_runtime()
    yield
    _stop_runtime()


app = FastAPI(title="justdream", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
async def status():
    return _snapshot()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    batch = UPLOADS_DIR / _stamp_safe()
    batch.mkdir(parents=True, exist_ok=True)
    for f in files:
        name = Path(f.filename or "ref.bin").name
        dest = batch / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(str(dest))
    return {"ok": True, "paths": saved}


def _stamp_safe() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


@app.post("/api/preview")
async def preview(
    script: str = Form(""),
    preset: str = Form(DEFAULT_PRESET),
    style_lock: str = Form(DEFAULT_STYLE_LOCK),
):
    script = (script or "").strip()
    if not script:
        return JSONResponse({"ok": False, "error": "Paste a script to see estimates"}, status_code=400)
    p = get_preset(preset)
    style = (style_lock or DEFAULT_STYLE_LOCK).strip()
    beats = split_script_to_beats(script, preset=p, style_lock=style)
    video_s = estimate_duration_s(len(beats), preset=p["id"])
    wall = estimate_wall_clock_s(len(beats), frames=p["frames"], preset=p["id"])
    return {
        "ok": True,
        "preset": p["id"],
        "preset_label": p["label"],
        "beats": len(beats),
        "clip_seconds": round(p["frames"] / p["fps"], 2),
        "video_seconds": round(video_s, 1),
        "video_label": format_duration(video_s),
        "minutes": round(video_s / 60, 2),
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "gen_per_clip": format_duration(wall["per_clip_s"]),
        "clip": f"{p['width']}x{p['height']} · {p['frames']}f @ {p['fps']}fps · chained",
        "mode": "offload + last-frame memory",
        "samples": beats[:2],
        "summary": (
            f"{len(beats)} clips · video {format_duration(video_s)} · "
            f"gen ETA ~{format_duration(wall['total_mid_s'])} "
            f"(range {format_duration(wall['total_lo_s'])}–{format_duration(wall['total_hi_s'])})"
        ),
    }


@app.post("/api/generate")
async def generate(
    script: str = Form(""),
    preset: str = Form(DEFAULT_PRESET),
    style_lock: str = Form(DEFAULT_STYLE_LOCK),
    ref_paths: str = Form(""),  # newline or comma separated absolute paths from /api/upload
):
    if not torch.cuda.is_available():
        return JSONResponse({"ok": False, "error": "CUDA not available"}, status_code=500)

    script = (script or "").strip()
    if not script:
        return JSONResponse({"ok": False, "error": "Paste a script first"}, status_code=400)

    p = get_preset(preset)
    style = (style_lock or DEFAULT_STYLE_LOCK).strip() or DEFAULT_STYLE_LOCK
    refs = [x.strip() for x in ref_paths.replace(",", "\n").splitlines() if x.strip()]
    beats = split_script_to_beats(script, preset=p, style_lock=style)
    secs = estimate_duration_s(len(beats), preset=p["id"])
    wall = estimate_wall_clock_s(len(beats), frames=p["frames"], preset=p["id"])
    job_id = str(uuid.uuid4())

    _append_message(
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "job_id": job_id,
            "text": (
                f"{p['label']} · {len(beats)} chained clips · video {format_duration(secs)} · "
                f"ETA ~{format_duration(wall['total_mid_s'])}"
                f"{' · ' + str(len(refs)) + ' refs' if refs else ''}\n"
                f"{script[:500]}{'…' if len(script) > 500 else ''}"
            ),
            "ts": time.time(),
        }
    )

    with _queue_lock:
        _job_queue.append(
            {
                "id": job_id,
                "script": script,
                "seed": None,
                "preset": p["id"],
                "style_lock": style,
                "ref_paths": refs,
            }
        )
        qlen = len(_job_queue)
        _set(queue_len=qlen)
    _dispatcher_wake.set()
    return {
        "ok": True,
        "job_id": job_id,
        "queue_len": qlen,
        "beats": len(beats),
        "minutes": round(secs / 60, 2),
        "video_label": format_duration(secs),
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "preset": p["id"],
        "preset_label": p["label"],
        "summary": (
            f"{len(beats)} clips · video {format_duration(secs)} · "
            f"gen ETA ~{format_duration(wall['total_mid_s'])}"
        ),
    }


def main() -> None:
    mp.freeze_support()
    venv_py = ROOT / "venv" / "Scripts" / "python.exe"
    if venv_py.is_file():
        here = Path(sys.executable).resolve()
        expected = venv_py.resolve()
        if here != expected:
            raise SystemExit(
                "Wrong Python — justdream needs its venv (bitsandbytes lives there).\n"
                f"  running:  {here}\n"
                f"  expected: {expected}\n"
                "Stop this process, then:\n"
                "  .\\venv\\Scripts\\Activate.ps1\n"
                "  python webui.py\n"
                "Or:  .\\venv\\Scripts\\python.exe webui.py"
            )
    try:
        import importlib.metadata as md

        md.version("bitsandbytes")
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            f"bitsandbytes missing/broken ({exc}).\n"
            "Fix:  .\\venv\\Scripts\\python.exe -m pip install -U bitsandbytes\n"
            "Then run:  .\\venv\\Scripts\\python.exe webui.py"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA not available (torch={torch.__version__}, cuda={torch.version.cuda}).\n"
            "Do NOT re-run full setup.ps1 — that only reinstalls deps; models stay cached.\n"
            "Fix GPU torch only:\n"
            "  .\\venv\\Scripts\\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 "
            "--index-url https://download.pytorch.org/whl/cu128"
        )
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Python: {sys.executable}")
    print(f"Stories: {STORIES_DIR}")
    print(f"Open http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info", workers=1)


if __name__ == "__main__":
    main()
