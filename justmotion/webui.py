"""
justmotion web UI — justimagine look: prompt → live keyframes → interpolate → DLSS.
Run: python webui.py → http://127.0.0.1:7863

Prefer justimagine's venv (has torch + diffusers). If you launch with system Python,
this script re-execs into justimagine\\venv automatically when torch is missing.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JUSTIMAGINE_VENV_PY = ROOT.parent / "justimagine" / "venv" / "Scripts" / "python.exe"


def _ensure_torch_python() -> None:
    """Re-launch under justimagine's venv if this interpreter has no torch."""
    try:
        import torch  # noqa: F401

        return
    except ImportError:
        pass

    if os.environ.get("JUSTMOTION_REEXEC") == "1":
        raise SystemExit(
            "torch still missing after re-exec. Activate justimagine\\venv or install torch."
        )

    if not JUSTIMAGINE_VENV_PY.is_file():
        raise SystemExit(
            "No module named 'torch'.\n"
            f"Expected justimagine venv at:\n  {JUSTIMAGINE_VENV_PY}\n"
            "Fix:\n"
            "  cd ..\\justimagine\n"
            "  .\\venv\\Scripts\\Activate.ps1\n"
            "  cd ..\\justmotion\n"
            "  python webui.py"
        )

    env = os.environ.copy()
    env["JUSTMOTION_REEXEC"] = "1"
    print(f"[justmotion] no torch in {sys.executable}", file=sys.stderr)
    print(f"[justmotion] re-launching with {JUSTIMAGINE_VENV_PY}", file=sys.stderr)
    os.execve(str(JUSTIMAGINE_VENV_PY), [str(JUSTIMAGINE_VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]], env)


_ensure_torch_python()

import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import (
    DEFAULT_GEN,
    DEFAULT_KEYFRAMES,
    EXP_DLSS_ROOT,
    JUSTIMAGINE_ROOT,
    OUTPUT_DIR,
    WORK_DIR,
    _stamp,
    estimate_wall_s,
    format_duration,
)

STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

PORT = 7863

_state_lock = threading.Lock()
_queue_lock = threading.Lock()
_state = {
    "busy": False,
    "status": "idle",
    "progress": None,
    "stage": None,
    "error": None,
    "preview": None,
    "keyframes": [],
    "job_id": None,
    "queue_len": 0,
    "messages": [],
    "stage_started": None,
    "worker_ready": False,
    "last_took_s": None,
    "device": None,
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
    def emit(**payload) -> None:
        try:
            event_q.put(payload, timeout=1.0)
        except Exception:
            pass

    device = None
    device_mode = None
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except Exception:
        pass

    emit(type="ready", device=device, device_mode=device_mode)

    from app import MotionJob

    while True:
        job = job_q.get()
        if job is None:
            break
        job_id = job["id"]
        emit(type="job_start", job_id=job_id, status="starting…", progress=0.0, stage=None)
        t0 = time.time()
        try:
            result = MotionJob(emit).run(job)
            took = time.time() - t0
            emit(
                type="job_done",
                job_id=job_id,
                preview=result["preview"],
                keyframes=result.get("keyframes") or [],
                took_s=took,
                text=f"Video ready · {result['preview']} · {format_duration(took)}",
                progress=1.0,
                stage="upscale",
            )
        except Exception as exc:
            emit(type="job_error", job_id=job_id, error=str(exc))


def _set(**kwargs) -> None:
    with _state_lock:
        _state.update(kwargs)


def _snapshot() -> dict:
    with _state_lock:
        started = _state.get("stage_started")
        elapsed = int(time.time() - started) if started and _state["busy"] else 0
        status = _state["status"]
        if _state["busy"] and elapsed >= 2:
            status = f"{status} · {elapsed}s"
        last_took_s = _state.get("last_took_s")
        return {
            "busy": _state["busy"],
            "status": status,
            "progress": _state["progress"],
            "stage": _state.get("stage"),
            "error": _state["error"],
            "preview": _state["preview"],
            "keyframes": list(_state.get("keyframes") or []),
            "job_id": _state["job_id"],
            "queue_len": _state["queue_len"],
            "messages": list(_state["messages"]),
            "elapsed_s": elapsed,
            "last_took": format_duration(last_took_s) if isinstance(last_took_s, (int, float)) else None,
            "worker_ready": _state["worker_ready"],
            "device": _state.get("device"),
            "device_mode": _state.get("device_mode"),
            "exp_root": str(EXP_DLSS_ROOT),
            "justimagine_root": str(JUSTIMAGINE_ROOT),
        }


def _append_message(msg: dict) -> None:
    with _state_lock:
        _state["messages"].append(msg)
        if len(_state["messages"]) > 50:
            _state["messages"] = _state["messages"][-50:]


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
            _set(worker_ready=True, status="idle", device=ev.get("device"), device_mode=ev.get("device_mode"))
        elif et == "progress":
            payload = {
                "status": ev.get("status", "working"),
                "stage": ev.get("stage"),
            }
            if ev.get("progress") is not None:
                with _state_lock:
                    prev = _state.get("progress")
                new_p = float(ev["progress"])
                if isinstance(prev, (int, float)):
                    new_p = max(float(prev), new_p)
                payload["progress"] = new_p
            _set(**payload)
        elif et == "keyframe":
            name = ev.get("preview")
            with _state_lock:
                kfs = list(_state.get("keyframes") or [])
                if name and name not in kfs:
                    kfs.append(name)
                _state["keyframes"] = kfs
            payload = {
                "status": ev.get("status", "keyframe"),
                "stage": "keyframes",
                "keyframes": list(_state["keyframes"]),
            }
            if ev.get("progress") is not None:
                payload["progress"] = float(ev["progress"])
            _set(**payload)
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "text": ev.get("status") or f"Keyframe {ev.get('index')}",
                    "preview": name,
                    "preview_kind": "image",
                    "ts": time.time(),
                }
            )
        elif et == "job_start":
            _set(
                busy=True,
                job_id=ev.get("job_id"),
                error=None,
                progress=0.0,
                stage=None,
                status=ev.get("status", "starting…"),
                stage_started=time.time(),
                last_took_s=None,
                keyframes=[],
                preview=None,
            )
        elif et == "job_done":
            took_s = ev.get("took_s")
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "job_id": ev.get("job_id"),
                    "text": ev.get("text") or "Done",
                    "preview": ev.get("preview"),
                    "preview_kind": "video",
                    "ts": time.time(),
                }
            )
            _set(
                busy=False,
                status=f"done · took {format_duration(took_s)}" if took_s else "done",
                progress=1.0,
                preview=ev.get("preview"),
                keyframes=list(ev.get("keyframes") or _state.get("keyframes") or []),
                stage="upscale",
                job_id=None,
                error=None,
                stage_started=None,
                last_took_s=took_s,
            )
            _dispatcher_wake.set()
        elif et == "job_error":
            err = ev.get("error") or "unknown error"
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "text": f"Error: {err}",
                    "error": True,
                    "ts": time.time(),
                }
            )
            _set(busy=False, status="error", progress=None, error=err, job_id=None, stage_started=None)
            _dispatcher_wake.set()


def _dispatcher_loop() -> None:
    assert _job_q is not None
    while True:
        _dispatcher_wake.wait(timeout=0.5)
        _dispatcher_wake.clear()
        with _state_lock:
            if _state["busy"] or not _state["worker_ready"]:
                continue
        with _queue_lock:
            if not _job_queue:
                _set(queue_len=0)
                continue
            job = _job_queue.popleft()
            _set(queue_len=len(_job_queue))
        _job_q.put(job)


def _start_runtime() -> None:
    global _job_q, _event_q, _worker_proc, _started
    if _started:
        return
    _job_q = _mp_ctx.Queue()
    _event_q = _mp_ctx.Queue()
    _worker_proc = _mp_ctx.Process(target=_worker_main, args=(_job_q, _event_q), daemon=True)
    _worker_proc.start()
    threading.Thread(target=_event_listener, daemon=True).start()
    threading.Thread(target=_dispatcher_loop, daemon=True).start()
    _started = True


def _stop_runtime() -> None:
    global _started
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
        _worker_proc.join(timeout=3)
    _started = False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _start_runtime()
    yield
    _stop_runtime()


app = FastAPI(lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def api_status() -> JSONResponse:
    return JSONResponse(_snapshot())


@app.post("/api/estimate")
async def api_estimate(
    prompt: str = Form(""),
    keyframes: int = Form(DEFAULT_KEYFRAMES),
    gen_width: int = Form(DEFAULT_GEN),
    skip_nr: str | None = Form(None),
    has_image: str = Form("0"),
    has_video: str = Form("0"),
) -> JSONResponse:
    n = max(2, int(keyframes))
    if has_video in ("1", "true", "yes"):
        need = 0
    elif has_image in ("1", "true", "yes"):
        need = max(0, n - 1)
    else:
        need = n if (prompt or "").strip() else 0
    warm = bool(_state.get("worker_ready"))
    lo, mid, hi = estimate_wall_s(
        keyframes=n,
        need_generate=need,
        gen_w=gen_width,
        target_fps=24,
        skip_nr=bool(skip_nr),
        warm=warm,
    )
    mode = "video→kf" if has_video in ("1", "true", "yes") else ("image+gen" if need < n else "prompt→kf")
    return JSONResponse(
        {
            "ok": True,
            "summary": f"{n} keyframes · {gen_width}px · {mode} · then interpolate + {'plain' if skip_nr else 'DLSS/NR'}",
            "eta_lo": format_duration(lo),
            "eta_mid": format_duration(mid),
            "eta_hi": format_duration(hi),
        }
    )


async def _save_upload(upload: UploadFile | None, dest_dir: Path, prefix: str) -> Path | None:
    if upload is None or not upload.filename:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(upload.filename).suffix or ".bin"
    path = dest_dir / f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
    path.write_bytes(await upload.read())
    return path


@app.post("/api/generate")
async def api_generate(
    prompt: str = Form(""),
    keyframes: int = Form(DEFAULT_KEYFRAMES),
    gen_width: int = Form(DEFAULT_GEN),
    gen_height: int = Form(0),
    nr_width: int = Form(1920),
    nr_intensity: float = Form(0.6),
    target_fps: int = Form(24),
    skip_nr: str | None = Form("1"),
    interp_mode: str = Form("blend"),
    seed: str = Form(""),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
) -> JSONResponse:
    prompt = (prompt or "").strip()
    stamp = _stamp()
    uploads = WORK_DIR / "_uploads" / stamp
    image_path = await _save_upload(image, uploads, "image")
    video_path = await _save_upload(video, uploads, "video")

    if not prompt and image_path is None and video_path is None:
        return JSONResponse(
            {"ok": False, "error": "Give a prompt, and/or an image, and/or a video."},
            status_code=400,
        )

    n = max(2, int(keyframes))
    gen_h = int(gen_height) if int(gen_height or 0) > 0 else int(gen_width)
    seed_val = int(seed) if str(seed).strip().lstrip("-").isdigit() else None

    if video_path:
        need = 0
    elif image_path:
        need = max(0, n - 1)
    else:
        need = n
    lo, mid, hi = estimate_wall_s(
        keyframes=n,
        need_generate=need,
        gen_w=gen_width,
        target_fps=target_fps,
        skip_nr=bool(skip_nr),
        warm=bool(_state.get("worker_ready")),
    )

    job_id = str(uuid.uuid4())
    bits = [f"{n} kf", f"{gen_width}px", f"→{nr_width}"]
    if video_path:
        bits.append("from video")
    elif image_path:
        bits.append("from image+")
    else:
        bits.append("from prompt")
    if skip_nr:
        bits.append("skip-nr")

    _append_message(
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "job_id": job_id,
            "text": (prompt or "(no prompt)") + "\n" + " · ".join(bits) + f"\nETA ~{format_duration(mid)}",
            "ts": time.time(),
        }
    )

    job = {
        "id": job_id,
        "stamp": stamp,
        "prompt": prompt,
        "keyframes": n,
        "gen_width": gen_width,
        "gen_height": gen_h,
        "nr_width": nr_width,
        "nr_intensity": nr_intensity,
        "target_fps": target_fps,
        "skip_nr": bool(skip_nr),
        "interp_mode": (interp_mode or "blend").strip().lower() or "blend",
        "seed": seed_val,
        "image_path": str(image_path) if image_path else None,
        "video_path": str(video_path) if video_path else None,
    }
    with _queue_lock:
        _job_queue.append(job)
        qlen = len(_job_queue)
    _set(queue_len=qlen)
    _dispatcher_wake.set()
    return JSONResponse(
        {
            "ok": True,
            "job_id": job_id,
            "summary": f"queued · ETA ~{format_duration(mid)} ({format_duration(lo)}–{format_duration(hi)})",
        }
    )


def main() -> None:
    mp.freeze_support()
    print(f"justmotion → http://127.0.0.1:{PORT}")
    print(f"JUSTIMAGINE_ROOT={JUSTIMAGINE_ROOT}")
    print(f"EXP_DLSS_ROOT={EXP_DLSS_ROOT}")
    print("Tip: activate justimagine\\venv so FLUX + torch are available.")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info", workers=1)


if __name__ == "__main__":
    main()
