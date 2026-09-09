"""
justimagine web UI — contextual FLUX.2 klein chat.
Run: python webui.py → http://127.0.0.1:7861
"""

from __future__ import annotations

import multiprocessing as mp
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

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
STATIC.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PORT = 7861

_state_lock = threading.Lock()
_queue_lock = threading.Lock()
_state = {
    "busy": False,
    "status": "idle",
    "progress": None,
    "error": None,
    "preview": None,
    "context": None,
    "job_id": None,
    "queue_len": 0,
    "messages": [],
    "stage_started": None,
    "hint": None,
    "worker_ready": False,
    "device_mode": None,
    "has_context": False,
    "eta_mid_s": None,
}
_job_queue: deque = deque()
_mp_ctx = mp.get_context("spawn")
_job_q: mp.Queue | None = None
_event_q: mp.Queue | None = None
_worker_proc: mp.Process | None = None
_dispatcher_wake = threading.Event()
_started = False


def _worker_main(job_q: mp.Queue, event_q: mp.Queue) -> None:
    import io as bio

    from PIL import Image

    from app import FluxKlein, OUTPUT_DIR as OUT, _stamp, save_image

    def emit(**payload) -> None:
        try:
            event_q.put(payload, timeout=1.0)
        except Exception:
            pass

    def on_status(msg: str, progress: float | None = None) -> None:
        payload = {"type": "progress", "status": msg}
        if progress is not None:
            payload["progress"] = progress
        emit(**payload)

    engine = FluxKlein()
    last_image: Image.Image | None = None
    emit(type="ready")

    while True:
        job = job_q.get()
        if job is None:
            break

        kind = job.get("kind", "generate")
        if kind == "clear_context":
            last_image = None
            emit(type="context", context=None, has_context=False)
            continue

        job_id = job["id"]
        prompt = (job.get("prompt") or "").strip()
        clear_first = bool(job.get("clear_context"))
        image_bytes = job.get("image_bytes")
        width = int(job.get("width") or 1024)
        height = int(job.get("height") or 1024)
        seed = job.get("seed")

        emit(type="job_start", job_id=job_id, status="starting…", progress=0.0)
        try:
            if clear_first:
                last_image = None

            ref = None
            if image_bytes is not None:
                ref = Image.open(bio.BytesIO(image_bytes)).convert("RGB")
                last_image = ref
            elif last_image is not None and not clear_first:
                ref = last_image

            stem = _stamp()
            if ref is not None:
                ctx_name = f"{stem}_context.png"
                ref.save(OUT / ctx_name)
                emit(type="context", context=ctx_name, has_context=True)

            result = engine.generate(
                prompt,
                image=ref,
                width=width,
                height=height,
                seed=seed,
                on_status=on_status,
            )
            path = save_image(result, stem)
            last_image = result
            emit(
                type="job_done",
                job_id=job_id,
                preview=path.name,
                context=path.name,
                has_context=True,
                device_mode=engine.device_mode,
                text=("Edited" if ref is not None else "Generated") + f": {prompt[:120]}",
            )
        except Exception as exc:
            emit(type="job_error", job_id=job_id, error=str(exc))


def _set(**kwargs) -> None:
    with _state_lock:
        # Don't reset the job clock on every status string change.
        _state.update(kwargs)


def _snapshot() -> dict:
    from app import format_duration

    with _state_lock:
        started = _state.get("stage_started")
        elapsed = int(time.time() - started) if started and _state["busy"] else 0
        status = _state["status"]
        progress = _state["progress"]
        pct = int(round(100 * progress)) if isinstance(progress, (int, float)) else None
        eta_left = None
        eta_mid = _state.get("eta_mid_s")
        if _state["busy"] and isinstance(eta_mid, (int, float)) and isinstance(progress, (int, float)):
            remain = max(0.0, float(eta_mid) * (1.0 - float(progress)))
            eta_left = format_duration(remain)
        if _state["busy"] and elapsed >= 2:
            status = f"{status} · {elapsed}s"
            if eta_left:
                status = f"{status} · ~{eta_left} left"
        return {
            "busy": _state["busy"],
            "status": status,
            "progress": progress,
            "progress_pct": pct,
            "error": _state["error"],
            "preview": _state["preview"],
            "context": _state["context"],
            "has_context": _state["has_context"],
            "job_id": _state["job_id"],
            "queue_len": _state["queue_len"],
            "messages": list(_state["messages"]),
            "elapsed_s": elapsed,
            "eta_left": eta_left,
            "hint": _state.get("hint"),
            "worker_ready": _state["worker_ready"],
            "device_mode": _state["device_mode"],
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "defaults": {"width": 1024, "height": 1024, "steps": 4},
        }


def _append_message(msg: dict) -> None:
    with _state_lock:
        _state["messages"].append(msg)
        if len(_state["messages"]) > 40:
            _state["messages"] = _state["messages"][-40:]


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
            _set(**payload)
        elif et == "context":
            _set(context=ev.get("context"), has_context=bool(ev.get("has_context")))
        elif et == "job_start":
            _set(
                busy=True,
                job_id=ev.get("job_id"),
                error=None,
                progress=0.0,
                status=ev.get("status", "starting…"),
                stage_started=time.time(),
            )
        elif et == "job_done":
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "job_id": ev.get("job_id"),
                    "text": ev.get("text") or "Done",
                    "preview": ev.get("preview"),
                    "ts": time.time(),
                }
            )
            _set(
                busy=False,
                status="done",
                progress=1.0,
                preview=ev.get("preview"),
                context=ev.get("context"),
                has_context=bool(ev.get("has_context")),
                device_mode=ev.get("device_mode"),
                job_id=None,
                error=None,
                stage_started=None,
                eta_mid_s=None,
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
            _set(busy=False, status="error", progress=None, error=err, job_id=None, stage_started=None, eta_mid_s=None)
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
        if job.get("kind") != "clear_context":
            _set(
                busy=True,
                job_id=job.get("id"),
                status="handing off…",
                progress=0.0,
                error=None,
                stage_started=time.time(),
                eta_mid_s=job.get("eta_mid_s"),
            )
        _job_q.put(job)


def _start_runtime() -> None:
    global _job_q, _event_q, _worker_proc, _started
    if _started:
        return
    _started = True
    _job_q = _mp_ctx.Queue()
    _event_q = _mp_ctx.Queue()
    _worker_proc = _mp_ctx.Process(target=_worker_main, args=(_job_q, _event_q), name="imagine-worker", daemon=True)
    _worker_proc.start()
    threading.Thread(target=_event_listener, name="imagine-events", daemon=True).start()
    threading.Thread(target=_dispatcher_loop, name="imagine-dispatch", daemon=True).start()


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


app = FastAPI(title="justimagine", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
async def status():
    return _snapshot()


@app.post("/api/clear_context")
async def clear_context():
    with _queue_lock:
        _job_queue.append({"kind": "clear_context"})
        _set(queue_len=len(_job_queue), has_context=False, context=None)
    _dispatcher_wake.set()
    return {"ok": True}


@app.post("/api/estimate")
async def estimate(
    width: int = Form(1024),
    height: int = Form(1024),
    has_ref: str = Form("0"),
):
    from app import estimate_wall_clock_s, format_duration

    wall = estimate_wall_clock_s(
        width=width,
        height=height,
        has_ref=has_ref in ("1", "true", "yes", "on"),
    )
    return {
        "ok": True,
        "width": width,
        "height": height,
        "steps": 4,
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "summary": (
            f"{width}×{height} · 4 steps · ETA ~{format_duration(wall['total_mid_s'])} "
            f"(range {format_duration(wall['total_lo_s'])}–{format_duration(wall['total_hi_s'])})"
        ),
    }


@app.post("/api/generate")
async def generate(
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
    clear_context: str = Form("0"),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: str = Form(""),
):
    if not torch.cuda.is_available():
        return JSONResponse({"ok": False, "error": "CUDA not available"}, status_code=500)

    from app import estimate_wall_clock_s, format_duration

    prompt = (prompt or "").strip()
    image_bytes = None
    filename = None
    if image is not None and image.filename:
        image_bytes = await image.read()
        filename = image.filename

    if not prompt:
        return JSONResponse({"ok": False, "error": "prompt required"}, status_code=400)

    job_id = str(uuid.uuid4())
    do_clear = clear_context in ("1", "true", "yes", "on")
    seed_val = int(seed) if str(seed).strip().isdigit() else None
    has_ref = bool(filename) or (not do_clear and _state.get("has_context"))
    wall = estimate_wall_clock_s(width=width, height=height, has_ref=bool(has_ref))

    _append_message(
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "job_id": job_id,
            "text": (
                f"{prompt}"
                f"{f'  [image: {filename}]' if filename else ''}"
                f"{'  [new]' if do_clear else ''}\n"
                f"ETA ~{format_duration(wall['total_mid_s'])} · {width}×{height}"
            ),
            "ts": time.time(),
        }
    )

    with _queue_lock:
        _job_queue.append(
            {
                "id": job_id,
                "kind": "generate",
                "prompt": prompt,
                "image_bytes": image_bytes,
                "clear_context": do_clear,
                "width": width,
                "height": height,
                "seed": seed_val,
                "eta_mid_s": wall["total_mid_s"],
            }
        )
        qlen = len(_job_queue)
        _set(queue_len=qlen)
    _dispatcher_wake.set()
    return {
        "ok": True,
        "job_id": job_id,
        "queue_len": qlen,
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "summary": f"ETA ~{format_duration(wall['total_mid_s'])} · {width}×{height} · 4 steps",
    }


def main() -> None:
    mp.freeze_support()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required. Run setup.ps1 first.")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Open http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info", workers=1)


if __name__ == "__main__":
    main()
