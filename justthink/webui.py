"""
justthink web UI — Hunyuan3D-2.0 (image/text → 3D mesh).
Run: python webui.py → http://127.0.0.1:7860

Generation runs in a separate process so /api/status never stalls on the GIL.
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

# tqdm desc → overall progress band
_TQDM_RANGES = (
    ("diffusion sampling", 0.32, 0.55),
    ("volume decoding", 0.55, 0.72),
)


def _map_tqdm_progress(desc: str, n: int, total: int) -> tuple[str, float]:
    key = (desc or "working").strip().rstrip(":").lower() or "working"
    lo, hi = 0.20, 0.95
    for name, a, b in _TQDM_RANGES:
        if name in key:
            lo, hi = a, b
            break
    else:
        # Unlabeled paint bars: delight=50, multiview=30
        if total == 50:
            lo, hi = 0.84, 0.90
        elif total == 30:
            lo, hi = 0.90, 0.98

    frac = min(1.0, max(0.0, float(n) / float(total))) if total else 0.0
    pct = int(round(100 * frac))
    label = f"{desc.strip() or 'working'} {n}/{total} ({pct}%)"
    if n == 0 and total:
        label += " — warming up / first step…"
    return label, lo + frac * (hi - lo)


def _worker_main(job_q: mp.Queue, event_q: mp.Queue) -> None:
    """GPU process — own GIL so the API process stays responsive."""
    import io

    from PIL import Image
    from tqdm import tqdm as tqdm_cls

    from app import ImageTo3D, TextToImage, free_vram, run_from_image

    def emit(**payload) -> None:
        try:
            event_q.put(payload, timeout=1.0)
        except Exception:
            pass

    orig = {
        "update": tqdm_cls.update,
        "set_description": tqdm_cls.set_description,
        "close": tqdm_cls.close,
        "__init__": tqdm_cls.__init__,
    }

    def _publish(bar, n: int | None = None) -> None:
        total = getattr(bar, "total", None) or 0
        cur = getattr(bar, "n", 0) if n is None else n
        desc = (getattr(bar, "desc", None) or "working").strip() or "working"
        if not total:
            emit(type="progress", status=desc, progress=None)
            return
        label, progress = _map_tqdm_progress(desc, int(cur), int(total))
        emit(type="progress", status=label, progress=progress)

    def __init__(self, *args, **kwargs):
        orig["__init__"](self, *args, **kwargs)
        _publish(self)

    def update(self, n=1):
        out = orig["update"](self, n)
        _publish(self)
        return out

    def set_description(self, desc=None, refresh=True):
        out = orig["set_description"](self, desc, refresh=refresh)
        _publish(self)
        return out

    def close(self):
        total = getattr(self, "total", None) or 0
        cur = getattr(self, "n", 0) or 0
        if total:
            _publish(self, n=max(cur, total))
        return orig["close"](self)

    tqdm_cls.__init__ = __init__
    tqdm_cls.update = update
    tqdm_cls.set_description = set_description
    tqdm_cls.close = close

    def on_status(msg: str, progress: float | None = None) -> None:
        payload: dict = {"type": "progress", "status": msg}
        if progress is not None:
            payload["progress"] = progress
        emit(**payload)

    t2i = TextToImage()
    i2_3d = ImageTo3D()
    last_mesh = None
    emit(type="ready")

    while True:
        job = job_q.get()
        if job is None:
            break

        job_id = job["id"]
        mode = job["mode"]
        stem = job["stem"]
        prompt = job.get("prompt")
        image_bytes = job.get("image_bytes")
        quality = job.get("quality") or "balanced"

        emit(type="job_start", job_id=job_id, status="starting…", progress=0.0)
        t0 = time.time()
        try:
            if image_bytes is not None:
                on_status("loading image…", 0.02)
                pil = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
                preview_name = f"{stem}_input.png"
                pil.save(OUTPUT_DIR / preview_name)
                emit(type="preview", preview=preview_name)
            else:
                on_status("text → image…", 0.05)
                pil = t2i.generate(prompt or "")
                preview_name = f"{stem}_preview.png"
                pil.save(OUTPUT_DIR / preview_name)
                t2i.unload()
                emit(type="preview", preview=preview_name)
                on_status("text → image done", 0.15)

            prior = last_mesh if mode == "texture" else None
            mesh, path = run_from_image(
                i2_3d,
                pil,
                mode,
                stem,
                prior_mesh=prior,
                on_status=on_status,
                quality=quality,
            )
            last_mesh = mesh
            took = time.time() - t0
            from app import format_duration, get_quality

            q = get_quality(quality)
            emit(
                type="job_done",
                job_id=job_id,
                preview=preview_name,
                mesh=path.name,
                mode=mode,
                took_s=took,
                text=(
                    f"Done ({mode} · {q['label']}: {q['shape_steps']} steps / octree {q['octree']})"
                    f" · took {format_duration(took)}"
                ),
            )
        except Exception as exc:
            free_vram()
            try:
                t2i.unload()
            except Exception:
                pass
            emit(type="job_error", job_id=job_id, error=str(exc))


_state_lock = threading.Lock()
_queue_lock = threading.Lock()
_state = {
    "busy": False,
    "status": "idle",
    "progress": None,
    "error": None,
    "preview": None,
    "mesh": None,
    "job_id": None,
    "queue_len": 0,
    "messages": [],
    "stage_started": None,
    "hint": None,
    "worker_ready": False,
    "eta_mid_s": None,
    "last_took_s": None,
}
_job_queue: deque = deque()
_mp_ctx = mp.get_context("spawn")
_job_q: mp.Queue | None = None
_event_q: mp.Queue | None = None
_worker_proc: mp.Process | None = None
_dispatcher_wake = threading.Event()
_started = False


def _set(**kwargs) -> None:
    with _state_lock:
        # Don't reset the job clock on every status string change.
        _state.update(kwargs)


def _snapshot() -> dict:
    from app import DEFAULT_QUALITY, format_duration, list_quality_presets

    with _state_lock:
        started = _state.get("stage_started")
        elapsed = int(time.time() - started) if started and _state["busy"] else 0
        status = _state["status"]
        progress = _state["progress"]
        hint = _state.get("hint")
        pct = int(round(100 * progress)) if isinstance(progress, (int, float)) else None
        eta_left = None
        eta_mid = _state.get("eta_mid_s")
        if _state["busy"] and isinstance(eta_mid, (int, float)) and isinstance(progress, (int, float)):
            remain = max(0.0, float(eta_mid) * (1.0 - float(progress)))
            eta_left = format_duration(remain)
        last_took_s = _state.get("last_took_s")
        last_took = format_duration(last_took_s) if isinstance(last_took_s, (int, float)) else None
        if _state["busy"] and elapsed >= 3:
            st = status or ""
            if "(0/" in st or "loading" in st.lower() or "warming" in st.lower():
                hint = hint or "still working — first CUDA step / model load can take a minute"
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
            "mesh": _state["mesh"],
            "job_id": _state["job_id"],
            "queue_len": _state["queue_len"],
            "messages": list(_state["messages"]),
            "elapsed_s": elapsed,
            "eta_left": eta_left,
            "last_took_s": last_took_s,
            "last_took": last_took,
            "hint": hint,
            "worker_ready": _state["worker_ready"],
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "defaults": {"quality": DEFAULT_QUALITY},
            "qualities": list_quality_presets(),
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
            _set(worker_ready=True, status="idle", hint=None)
        elif et == "progress":
            payload = {"status": ev.get("status", "working"), "hint": None}
            if "progress" in ev and ev["progress"] is not None:
                with _state_lock:
                    prev = _state.get("progress")
                new_p = float(ev["progress"])
                if isinstance(prev, (int, float)):
                    new_p = max(float(prev), new_p)
                payload["progress"] = new_p
            _set(**payload)
        elif et == "preview":
            _set(preview=ev.get("preview"))
        elif et == "job_start":
            _set(
                busy=True,
                job_id=ev.get("job_id"),
                error=None,
                mesh=None,
                progress=ev.get("progress", 0.0),
                status=ev.get("status", "starting…"),
                stage_started=time.time(),
                hint=None,
                last_took_s=None,
            )
        elif et == "job_done":
            from app import format_duration

            took_s = ev.get("took_s")
            if not isinstance(took_s, (int, float)):
                with _state_lock:
                    started = _state.get("stage_started")
                took_s = (time.time() - started) if started else None
            took_label = format_duration(took_s) if isinstance(took_s, (int, float)) else None
            text = ev.get("text") or f"Done ({ev.get('mode')})"
            if took_label and "took " not in text:
                text = f"{text} · took {took_label}"
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "job_id": ev.get("job_id"),
                    "text": text,
                    "preview": ev.get("preview"),
                    "mesh": ev.get("mesh"),
                    "took_s": took_s,
                    "ts": time.time(),
                }
            )
            _set(
                busy=False,
                status=f"done · took {took_label}" if took_label else "done",
                progress=1.0,
                mesh=ev.get("mesh"),
                preview=ev.get("preview"),
                job_id=None,
                error=None,
                stage_started=None,
                eta_mid_s=None,
                last_took_s=took_s,
                hint=None,
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
                job_id=None,
                stage_started=None,
                eta_mid_s=None,
                last_took_s=None,
                hint=None,
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
        _set(
            busy=True,
            job_id=job["id"],
            status="handing off to GPU worker…",
            progress=0.0,
            error=None,
            mesh=None,
            stage_started=time.time(),
            eta_mid_s=job.get("eta_mid_s"),
            hint=None,
        )
        _job_q.put(job)


def _start_runtime() -> None:
    global _job_q, _event_q, _worker_proc, _started
    if _started:
        return
    _started = True
    _job_q = _mp_ctx.Queue()
    _event_q = _mp_ctx.Queue()
    _worker_proc = _mp_ctx.Process(
        target=_worker_main,
        args=(_job_q, _event_q),
        name="hy3d-gpu-worker",
        daemon=True,
    )
    _worker_proc.start()
    threading.Thread(target=_event_listener, name="hy3d-events", daemon=True).start()
    threading.Thread(target=_dispatcher_loop, name="hy3d-dispatch", daemon=True).start()


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


app = FastAPI(title="justthink", lifespan=lifespan)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
async def status():
    return _snapshot()


@app.post("/api/estimate")
async def estimate(
    mode: str = Form("full"),
    has_image: str = Form("0"),
    quality: str = Form("balanced"),
):
    from app import estimate_wall_clock_s, format_duration, get_quality

    mode = (mode or "full").strip().lower()
    if mode not in ("full", "shape", "texture"):
        return JSONResponse({"ok": False, "error": "mode must be full|shape|texture"}, status_code=400)
    q = get_quality(quality)
    wall = estimate_wall_clock_s(
        mode=mode,
        has_image=has_image in ("1", "true", "yes", "on"),
        quality=q["id"],
    )
    return {
        "ok": True,
        "mode": mode,
        "quality": q["id"],
        "quality_label": q["label"],
        "shape_steps": q["shape_steps"],
        "octree": q["octree"],
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "summary": (
            f"{mode} · {q['label']} ({q['shape_steps']} steps / octree {q['octree']}) · "
            f"ETA ~{format_duration(wall['total_mid_s'])} "
            f"(range {format_duration(wall['total_lo_s'])}–{format_duration(wall['total_hi_s'])})"
        ),
    }


@app.post("/api/generate")
async def generate(
    mode: str = Form("full"),
    quality: str = Form("balanced"),
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
):
    if not torch.cuda.is_available():
        return JSONResponse({"ok": False, "error": "CUDA not available"}, status_code=500)

    mode = (mode or "full").strip().lower()
    if mode not in ("full", "shape", "texture"):
        return JSONResponse({"ok": False, "error": "mode must be full|shape|texture"}, status_code=400)

    from app import _stamp, estimate_wall_clock_s, format_duration, get_quality

    q = get_quality(quality)
    prompt = (prompt or "").strip()
    image_bytes = None
    filename = None
    if image is not None and image.filename:
        image_bytes = await image.read()
        filename = image.filename

    if not image_bytes and not prompt:
        return JSONResponse({"ok": False, "error": "provide a prompt or an image"}, status_code=400)

    job_id = str(uuid.uuid4())
    stem = _stamp()
    wall = estimate_wall_clock_s(mode=mode, has_image=bool(image_bytes), quality=q["id"])
    user_text = prompt if prompt else f"(image: {filename})"
    _append_message(
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "job_id": job_id,
            "text": (
                f"{user_text}\n"
                f"ETA ~{format_duration(wall['total_mid_s'])} · {mode} · "
                f"{q['label']} ({q['shape_steps']}/{q['octree']})"
            ),
            "mode": mode,
            "ts": time.time(),
        }
    )

    with _queue_lock:
        _job_queue.append(
            {
                "id": job_id,
                "mode": mode,
                "quality": q["id"],
                "stem": stem,
                "prompt": prompt or None,
                "image_bytes": image_bytes,
                "eta_mid_s": wall["total_mid_s"],
            }
        )
        qlen = len(_job_queue)
        _set(queue_len=qlen)
    _dispatcher_wake.set()
    return {
        "ok": True,
        "stem": stem,
        "job_id": job_id,
        "queue_len": qlen,
        "quality": q["id"],
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "summary": (
            f"ETA ~{format_duration(wall['total_mid_s'])} · {mode} · "
            f"{q['label']} ({q['shape_steps']}/{q['octree']})"
        ),
    }


def main() -> None:
    mp.freeze_support()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Activate the venv from setup.ps1 first.")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("Open http://127.0.0.1:7860")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info", workers=1)


if __name__ == "__main__":
    main()
