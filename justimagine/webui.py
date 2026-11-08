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
from typing import Annotated

import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db as chat_db

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
    "last_took_s": None,
    "last_timing": None,
    "avg_took_s": None,
    "compiled": False,
    "quant": "bf16",
    "load_s": None,
    "took_history": [],
    "chat_id": None,
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

    from app import FluxKlein, OUTPUT_DIR as OUT, _stamp, format_duration, save_image

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
    try:
        def _boot_status(msg: str, progress: float | None = None) -> None:
            payload = {"type": "progress", "status": f"startup: {msg}"}
            if progress is not None:
                payload["progress"] = progress
            emit(**payload)

        emit(type="progress", status="preloading FLUX.2 klein onto GPU…", progress=0.0)
        engine.load(on_status=_boot_status)
        emit(
            type="progress",
            status=(
                f"idle · model ready ({engine.device_mode}"
                f"{' · compiled' if engine._compiled else ''})"
                + (f" · load {format_duration(engine.load_s)}" if engine.load_s else "")
            ),
            progress=None,
        )
    except Exception as exc:
        emit(type="progress", status=f"preload failed: {exc}", progress=None)
        print(f"[worker] preload failed: {exc}")
    emit(
        type="ready",
        device_mode=engine.device_mode,
        compiled=bool(engine._compiled),
        quant=getattr(engine, "quant", "bf16"),
        load_s=engine.load_s,
    )

    while True:
        job = job_q.get()
        if job is None:
            break

        kind = job.get("kind", "generate")
        if kind == "clear_context":
            last_image = None
            emit(type="context", context=None, has_context=False)
            continue
        if kind == "set_context":
            name = (job.get("preview") or "").strip()
            path = OUT / name if name else None
            if path is not None and path.is_file():
                try:
                    last_image = Image.open(path).convert("RGB")
                    emit(type="context", context=name, has_context=True)
                except Exception as exc:
                    last_image = None
                    emit(type="context", context=None, has_context=False)
                    print(f"[worker] set_context failed: {exc}")
            else:
                last_image = None
                emit(type="context", context=None, has_context=False)
            continue

        job_id = job["id"]
        prompt = (job.get("prompt") or "").strip()
        clear_first = bool(job.get("clear_context"))
        image_bytes_list = job.get("image_bytes_list") or []
        # Back-compat with older single-image jobs.
        if not image_bytes_list and job.get("image_bytes") is not None:
            image_bytes_list = [job["image_bytes"]]
        width = int(job.get("width") or 1024)
        height = int(job.get("height") or 1024)
        seed = job.get("seed")
        steps = int(job.get("steps") or 4)
        guidance = float(job.get("guidance") if job.get("guidance") is not None else 1.0)

        emit(type="job_start", job_id=job_id, status="starting…", progress=0.0)
        t0 = time.time()
        try:
            if clear_first:
                last_image = None

            refs: list = []
            if image_bytes_list:
                for raw in image_bytes_list:
                    refs.append(Image.open(bio.BytesIO(raw)).convert("RGB"))
                last_image = refs[-1]
            elif last_image is not None and not clear_first:
                refs = [last_image]

            stem = _stamp()
            if refs:
                for i, ref_im in enumerate(refs):
                    ctx_name = f"{stem}_context_{i}.png" if len(refs) > 1 else f"{stem}_context.png"
                    ref_im.save(OUT / ctx_name)
                    if i == 0:
                        emit(type="context", context=ctx_name, has_context=True)

            ref_arg = None
            if len(refs) == 1:
                ref_arg = refs[0]
            elif len(refs) > 1:
                ref_arg = refs

            result = engine.generate(
                prompt,
                image=ref_arg,
                width=width,
                height=height,
                seed=seed,
                steps=steps,
                guidance=guidance,
                on_status=on_status,
            )
            path = save_image(result, stem)
            last_image = result
            took = time.time() - t0
            infer = engine.last_infer_s
            label = format_duration(infer if infer is not None else took)
            timing = engine.last_timing or {}
            emit(
                type="job_done",
                job_id=job_id,
                preview=path.name,
                context=path.name,
                has_context=True,
                device_mode=engine.device_mode,
                compiled=bool(engine._compiled),
                quant=getattr(engine, "quant", "bf16"),
                took_s=infer if infer is not None else took,
                timing=timing,
                edited=bool(refs),
                text="",
                label=label,
            )
        except Exception as exc:
            emit(type="job_error", job_id=job_id, error=str(exc))


def _set(**kwargs) -> None:
    with _state_lock:
        # Don't reset the job clock on every status string change.
        _state.update(kwargs)


def _snapshot() -> dict:
    from app import _vram_gb, format_duration

    with _state_lock:
        started = _state.get("stage_started")
        elapsed = int(time.time() - started) if started and _state["busy"] else 0
        status = _state["status"]
        progress = _state["progress"]
        pct = int(round(100 * progress)) if isinstance(progress, (int, float)) else None
        eta_left = None
        eta_left_s = None
        eta_mid = _state.get("eta_mid_s")
        if _state["busy"] and isinstance(eta_mid, (int, float)) and isinstance(progress, (int, float)):
            remain = max(0.0, float(eta_mid) * (1.0 - float(progress)))
            eta_left_s = remain
            eta_left = format_duration(remain)
        last_took_s = _state.get("last_took_s")
        last_took = format_duration(last_took_s) if isinstance(last_took_s, (int, float)) else None
        avg_took_s = _state.get("avg_took_s")
        avg_took = format_duration(avg_took_s) if isinstance(avg_took_s, (int, float)) else None
        timing = _state.get("last_timing")
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
            "eta_left_s": eta_left_s,
            "eta_mid_s": eta_mid,
            "last_took_s": last_took_s,
            "last_took": last_took,
            "avg_took_s": avg_took_s,
            "avg_took": avg_took,
            "last_timing": timing,
            "hint": _state.get("hint"),
            "worker_ready": _state["worker_ready"],
            "device_mode": _state["device_mode"],
            "compiled": bool(_state.get("compiled")),
            "quant": _state.get("quant") or "bf16",
            "load_s": _state.get("load_s"),
            "load": format_duration(_state.get("load_s")) if _state.get("load_s") else None,
            "vram": _vram_gb(),
            "cuda": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "chat_id": _state.get("chat_id"),
            "chat_count": chat_db.chat_count(),
            "base_url": f"http://127.0.0.1:{PORT}",
            "defaults": {
                "width": 1024,
                "height": 1024,
                "steps": 4,
                "guidance": 1.0,
                "clear_context": "0",
            },
        }


def _queue_set_context(preview: str | None) -> None:
    with _queue_lock:
        if preview:
            _job_queue.append({"kind": "set_context", "preview": preview})
        else:
            _job_queue.append({"kind": "clear_context"})
        _set(queue_len=len(_job_queue))
    _dispatcher_wake.set()


def _apply_chat_to_state(chat: dict, messages: list[dict] | None = None) -> None:
    msgs = messages if messages is not None else chat_db.load_messages(chat["id"])
    preview = None
    for m in reversed(msgs):
        if m.get("role") == "assistant" and m.get("preview"):
            preview = m["preview"]
            break
    context = chat.get("context") or preview
    has_context = bool(context) if chat.get("has_context") or preview else False
    with _state_lock:
        _state["chat_id"] = chat["id"]
        _state["messages"] = list(msgs)
        _state["preview"] = preview
        _state["context"] = context if has_context else None
        _state["has_context"] = has_context
        _state["error"] = None
        if not _state["busy"]:
            _state["status"] = "idle"
            _state["progress"] = None
            _state["job_id"] = None
    _queue_set_context(context if has_context else None)


def _append_message(msg: dict) -> None:
    with _state_lock:
        chat_id = _state.get("chat_id")
        _state["messages"].append(msg)
    if chat_id:
        try:
            chat_db.append_message(chat_id, msg)
        except Exception as exc:
            print(f"[db] append_message failed: {exc}")


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
            _set(
                worker_ready=True,
                status="idle",
                device_mode=ev.get("device_mode") or _state.get("device_mode"),
                compiled=bool(ev.get("compiled")),
                quant=ev.get("quant") or _state.get("quant") or "bf16",
                load_s=ev.get("load_s"),
            )
            with _state_lock:
                ctx = _state.get("context") if _state.get("has_context") else None
            if ctx:
                _queue_set_context(ctx)
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
            timing = ev.get("timing") if isinstance(ev.get("timing"), dict) else {}
            _append_message(
                {
                    "id": str(uuid.uuid4()),
                    "role": "assistant",
                    "job_id": ev.get("job_id"),
                    "text": "",
                    "preview": ev.get("preview"),
                    "took_s": took_s,
                    "took_label": took_label or ev.get("label"),
                    "edited": bool(ev.get("edited")),
                    "device_mode": ev.get("device_mode"),
                    "width": timing.get("width"),
                    "height": timing.get("height"),
                    "steps": timing.get("steps"),
                    "encode_s": timing.get("encode_s"),
                    "denoise_s": timing.get("denoise_s"),
                    "peak_gb": timing.get("peak_gb"),
                    "ts": time.time(),
                }
            )
            hist = list(_state.get("took_history") or [])
            if isinstance(took_s, (int, float)):
                hist.append(float(took_s))
                hist = hist[-8:]
            avg = (sum(hist) / len(hist)) if hist else None
            preview_name = ev.get("preview")
            has_ctx = bool(ev.get("has_context"))
            _set(
                busy=False,
                status=f"done · took {took_label}" if took_label else "done",
                progress=1.0,
                preview=preview_name,
                context=ev.get("context") or preview_name,
                has_context=has_ctx,
                device_mode=ev.get("device_mode"),
                compiled=bool(ev.get("compiled", _state.get("compiled"))),
                quant=ev.get("quant") or _state.get("quant") or "bf16",
                job_id=None,
                error=None,
                stage_started=None,
                eta_mid_s=None,
                last_took_s=took_s,
                last_timing=ev.get("timing"),
                took_history=hist,
                avg_took_s=avg,
            )
            with _state_lock:
                chat_id = _state.get("chat_id")
            if chat_id:
                try:
                    chat_db.update_chat_context(
                        chat_id,
                        (ev.get("context") or preview_name) if has_ctx else None,
                        has_ctx,
                    )
                except Exception as exc:
                    print(f"[db] update_chat_context failed: {exc}")
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
        if job.get("kind") not in ("clear_context", "set_context"):
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


def _boot_chat_state() -> None:
    chat_db.init_db()
    chat = chat_db.ensure_active_chat()
    msgs = chat_db.load_messages(chat["id"])
    preview = None
    for m in reversed(msgs):
        if m.get("role") == "assistant" and m.get("preview"):
            preview = m["preview"]
            break
    context = chat.get("context") or preview
    has_context = bool(context) if (chat.get("has_context") or preview) else False
    with _state_lock:
        _state["chat_id"] = chat["id"]
        _state["messages"] = list(msgs)
        _state["preview"] = preview
        _state["context"] = context if has_context else None
        _state["has_context"] = has_context


def _start_runtime() -> None:
    global _job_q, _event_q, _worker_proc, _started
    if _started:
        return
    _started = True
    _boot_chat_state()
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
    with _state_lock:
        chat_id = _state.get("chat_id")
    if chat_id:
        try:
            chat_db.update_chat_context(chat_id, None, False)
        except Exception as exc:
            print(f"[db] clear context failed: {exc}")
    _dispatcher_wake.set()
    return {"ok": True}


@app.get("/api/chats")
async def chats_list():
    with _state_lock:
        active = _state.get("chat_id")
    return {
        "ok": True,
        "active_chat_id": active,
        "chat_count": chat_db.chat_count(),
        "chats": chat_db.list_chats(),
    }


@app.post("/api/chats/{chat_id}/open")
async def open_chat(chat_id: str):
    if _state.get("busy"):
        return JSONResponse(
            {"ok": False, "error": "Wait for the current image to finish"},
            status_code=409,
        )
    chat = chat_db.get_chat(chat_id)
    if not chat:
        return JSONResponse({"ok": False, "error": "chat not found"}, status_code=404)
    chat_db.set_active_chat(chat_id)
    msgs = chat_db.load_messages(chat_id)
    _apply_chat_to_state(chat, msgs)
    return {
        "ok": True,
        "chat": chat,
        "messages": msgs,
        "chat_count": chat_db.chat_count(),
    }


@app.delete("/api/chats/{chat_id}")
async def remove_chat(chat_id: str):
    if _state.get("busy") and _state.get("chat_id") == chat_id:
        return JSONResponse(
            {"ok": False, "error": "Cannot delete the chat that is generating"},
            status_code=409,
        )
    was_active = _state.get("chat_id") == chat_id
    if not chat_db.delete_chat(chat_id):
        return JSONResponse({"ok": False, "error": "chat not found"}, status_code=404)
    messages: list = []
    if was_active:
        chat = chat_db.ensure_active_chat()
        messages = chat_db.load_messages(chat["id"])
        _apply_chat_to_state(chat, messages)
    return {
        "ok": True,
        "active_chat_id": _state.get("chat_id"),
        "chat_count": chat_db.chat_count(),
        "chats": chat_db.list_chats(),
        "messages": messages if was_active else None,
    }


@app.post("/api/new_chat")
async def new_chat():
    """Start a fresh chat: persist previous one, open empty thread."""
    if _state.get("busy"):
        return JSONResponse(
            {"ok": False, "error": "Wait for the current image to finish"},
            status_code=409,
        )
    with _state_lock:
        current_id = _state.get("chat_id")
        current_msgs = list(_state.get("messages") or [])
    # Reuse an unused empty chat instead of stacking blanks.
    if current_id and not current_msgs:
        chat = chat_db.get_chat(current_id) or chat_db.ensure_active_chat()
        _apply_chat_to_state(chat, [])
        return {
            "ok": True,
            "chat": chat,
            "reused": True,
            "chat_count": chat_db.chat_count(),
            "chats": chat_db.list_chats(),
        }
    chat = chat_db.create_chat()
    _apply_chat_to_state(chat, [])
    return {
        "ok": True,
        "chat": chat,
        "reused": False,
        "chat_count": chat_db.chat_count(),
        "chats": chat_db.list_chats(),
    }


@app.post("/api/estimate")
async def estimate(
    width: int = Form(1024),
    height: int = Form(1024),
    has_ref: str = Form("0"),
    steps: int = Form(4),
):
    from app import estimate_wall_clock_s, format_duration

    steps = max(1, min(50, int(steps)))
    warm = (_state.get("device_mode") in ("full", "staged")) and bool(_state.get("worker_ready"))
    wall = estimate_wall_clock_s(
        width=width,
        height=height,
        steps=steps,
        has_ref=has_ref in ("1", "true", "yes", "on"),
        warm=warm,
        compiled=bool(_state.get("compiled")),
        quant=_state.get("quant") or "bf16",
        avg_took_s=_state.get("avg_took_s"),
    )
    mode = _state.get("device_mode") or "unknown"
    note = ""
    if mode == "full":
        note = " · full can page Shared GPU mem on 16GB (slow)"
    elif mode == "staged":
        note = " · resident DiT (recommended on 16GB)"
    elif not warm:
        note = " · first load may take several minutes"
    if _state.get("compiled"):
        note += " · compiled"
    return {
        "ok": True,
        "width": width,
        "height": height,
        "steps": steps,
        "warm": warm,
        "device_mode": mode,
        "compiled": bool(_state.get("compiled")),
        "quant": _state.get("quant") or "bf16",
        "eta_mid_s": wall["total_mid_s"],
        "eta_lo_s": wall["total_lo_s"],
        "eta_hi_s": wall["total_hi_s"],
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "summary": (
            f"{width}×{height} · {steps} steps · ETA ~{format_duration(wall['total_mid_s'])} "
            f"(range {format_duration(wall['total_lo_s'])}–{format_duration(wall['total_hi_s'])})"
            f"{note}"
        ),
    }


@app.post("/api/generate")
async def generate(
    prompt: str = Form(""),
    image: UploadFile | None = File(None),
    images: Annotated[list[UploadFile], File()] = [],
    clear_context: str = Form("0"),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: str = Form(""),
    steps: int = Form(4),
    guidance: float = Form(1.0),
):
    if not torch.cuda.is_available():
        return JSONResponse({"ok": False, "error": "CUDA not available"}, status_code=500)

    from app import estimate_wall_clock_s, format_duration

    prompt = (prompt or "").strip()
    uploads: list[UploadFile] = []
    if images:
        uploads.extend([u for u in images if u is not None and u.filename])
    if image is not None and image.filename:
        # Single-file curl/back-compat; skip if already in the multi list.
        if not any(u.filename == image.filename for u in uploads):
            uploads.append(image)

    image_bytes_list: list[bytes] = []
    filenames: list[str] = []
    for up in uploads:
        image_bytes_list.append(await up.read())
        filenames.append(up.filename or "image.png")

    if not prompt:
        return JSONResponse({"ok": False, "error": "prompt required"}, status_code=400)

    if not _state.get("chat_id"):
        chat = chat_db.ensure_active_chat()
        _apply_chat_to_state(chat, chat_db.load_messages(chat["id"]))

    job_id = str(uuid.uuid4())
    do_clear = clear_context in ("1", "true", "yes", "on")
    seed_val = int(seed) if str(seed).strip().isdigit() else None
    steps = max(1, min(50, int(steps)))
    guidance = float(guidance)
    has_ref = bool(filenames) or (not do_clear and _state.get("has_context"))
    warm = (_state.get("device_mode") in ("full", "staged")) and bool(_state.get("worker_ready"))
    wall = estimate_wall_clock_s(
        width=width,
        height=height,
        steps=steps,
        has_ref=bool(has_ref),
        warm=warm,
        compiled=bool(_state.get("compiled")),
        quant=_state.get("quant") or "bf16",
        avg_took_s=_state.get("avg_took_s"),
    )

    attach_previews: list[str] = []
    for i, raw in enumerate(image_bytes_list):
        ext = Path(filenames[i]).suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            ext = ".png"
        name = f"{job_id[:8]}_ref_{i}{ext}"
        try:
            (OUTPUT_DIR / name).write_bytes(raw)
            attach_previews.append(name)
        except Exception as exc:
            print(f"[generate] save ref failed: {exc}")

    _append_message(
        {
            "id": str(uuid.uuid4()),
            "role": "user",
            "job_id": job_id,
            "text": prompt,
            "image_name": filenames[0] if len(filenames) == 1 else None,
            "image_names": filenames,
            "attach_previews": attach_previews,
            "ref_count": len(filenames),
            "fresh": do_clear,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed_val,
            "eta_mid_s": wall["total_mid_s"],
            "eta_label": format_duration(wall["total_mid_s"]),
            "ts": time.time(),
        }
    )

    with _queue_lock:
        _job_queue.append(
            {
                "id": job_id,
                "kind": "generate",
                "prompt": prompt,
                "image_bytes_list": image_bytes_list,
                "clear_context": do_clear,
                "width": width,
                "height": height,
                "seed": seed_val,
                "steps": steps,
                "guidance": guidance,
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
        "eta_mid_s": wall["total_mid_s"],
        "eta_lo_s": wall["total_lo_s"],
        "eta_hi_s": wall["total_hi_s"],
        "gen_eta_mid": format_duration(wall["total_mid_s"]),
        "gen_eta_lo": format_duration(wall["total_lo_s"]),
        "gen_eta_hi": format_duration(wall["total_hi_s"]),
        "summary": f"ETA ~{format_duration(wall['total_mid_s'])} · {width}×{height} · {steps} steps",
    }


@app.get("/api/docs")
async def api_docs():
    base = f"http://127.0.0.1:{PORT}"
    return {
        "title": "justimagine HTTP API",
        "base_url": base,
        "model": "FLUX.2 klein 4B (bf16, resident DiT)",
        "async": True,
        "persistence": "SQLite justimagine.db — chats + messages survive restarts",
        "note": (
            "POST /api/generate queues a job and returns immediately. "
            "Poll GET /api/status until busy is false, then read messages[].preview "
            "or download /outputs/<name>.png. "
            "Follow-ups reuse the last result as edit context; POST /api/new_chat for a fresh thread."
        ),
        "ui_presets": {
            "aspect": {
                "1:1": [1024, 1024],
                "9:16": [768, 1344],
                "3:4": [896, 1152],
                "4:3": [1152, 896],
                "16:9": [1344, 768],
            },
            "quality": {
                "low": {"steps": 4, "guidance": 1.0, "note": "fastest"},
                "medium": {"steps": 8, "guidance": 1.0, "note": "balanced"},
                "high": {"steps": 16, "guidance": 1.5, "note": "best · slower"},
            },
        },
        "endpoints": [
            {
                "method": "POST",
                "path": "/api/generate",
                "content_type": "multipart/form-data",
                "fields": {
                    "prompt": "string, required",
                    "image": "file, optional single reference (back-compat)",
                    "images": "file[], optional multi-reference PNGs/JPGs",
                    "clear_context": "0|1 — optional; UI uses /api/new_chat instead",
                    "width": "int, default 1024 (snapped to ×16)",
                    "height": "int, default 1024",
                    "seed": "int, optional",
                    "steps": "int 1–50 · quality Low=4 Med=8 High=16",
                    "guidance": "float · Low/Med=1.0 High=1.5",
                },
                "response": {
                    "ok": True,
                    "job_id": "uuid",
                    "queue_len": 1,
                    "eta_mid_s": 14,
                    "gen_eta_mid": "14s",
                    "summary": "ETA ~14s · 1024×1024 · 8 steps",
                },
                "curl": (
                    f'curl -s -X POST {base}/api/generate '
                    f'-F "prompt=a red bicycle leaning on a brick wall" '
                    f'-F "width=1024" -F "height=1024" '
                    f'-F "seed=42" -F "steps=8" -F "guidance=1.0"'
                ),
            },
            {
                "method": "POST",
                "path": "/api/generate",
                "title": "edit with multi-reference images",
                "curl": (
                    f'curl -s -X POST {base}/api/generate '
                    f'-F "prompt=combine these into one portrait" '
                    f'-F "images=@/path/to/face.png" '
                    f'-F "images=@/path/to/outfit.png" '
                    f'-F "width=1024" -F "height=1024" '
                    f'-F "steps=8" -F "guidance=1.0"'
                ),
            },
            {
                "method": "GET",
                "path": "/api/status",
                "description": (
                    "Live job state, ETA, timing, VRAM, active chat_id, "
                    "chat_count, and messages for the current chat."
                ),
                "curl": f"curl -s {base}/api/status",
            },
            {
                "method": "GET",
                "path": "/api/chats",
                "description": "List persisted chats (title, counts) and active_chat_id.",
                "curl": f"curl -s {base}/api/chats",
            },
            {
                "method": "POST",
                "path": "/api/new_chat",
                "description": "Start a new empty chat and clear edit context (preferred over clear_context).",
                "curl": f"curl -s -X POST {base}/api/new_chat",
            },
            {
                "method": "POST",
                "path": "/api/chats/{id}/open",
                "description": "Switch active chat and restore its message history + context image.",
                "curl": f"curl -s -X POST {base}/api/chats/<chat-id>/open",
            },
            {
                "method": "DELETE",
                "path": "/api/chats/{id}",
                "description": "Delete a chat and its messages from SQLite.",
                "curl": f"curl -s -X DELETE {base}/api/chats/<chat-id>",
            },
            {
                "method": "POST",
                "path": "/api/clear_context",
                "description": "Drop the last image used as edit context (same chat).",
                "curl": f"curl -s -X POST {base}/api/clear_context",
            },
            {
                "method": "POST",
                "path": "/api/estimate",
                "content_type": "multipart/form-data",
                "fields": {
                    "width": "int",
                    "height": "int",
                    "has_ref": "0|1",
                    "steps": "int",
                },
                "curl": (
                    f'curl -s -X POST {base}/api/estimate '
                    f'-F "width=1024" -F "height=1024" -F "has_ref=0" -F "steps=8"'
                ),
            },
            {
                "method": "GET",
                "path": "/api/docs",
                "description": "This JSON catalog.",
                "curl": f"curl -s {base}/api/docs",
            },
            {
                "method": "GET",
                "path": "/outputs/{filename}",
                "description": "Download a finished PNG (or saved reference).",
                "curl": f"curl -o out.png {base}/outputs/YYYYMMDD_HHMMSS.png",
            },
        ],
        "poll_example_powershell": (
            "$r = curl.exe -s -X POST http://127.0.0.1:7861/api/generate "
            "-F \"prompt=a blue cup\" -F \"width=1024\" "
            "-F \"height=1024\" -F \"steps=8\" | ConvertFrom-Json\n"
            "$id = $r.job_id\n"
            "do { Start-Sleep 1; $s = Invoke-RestMethod http://127.0.0.1:7861/api/status } "
            "while ($s.busy -or -not ($s.messages | ? { $_.job_id -eq $id -and $_.preview }))\n"
            "$img = ($s.messages | ? { $_.job_id -eq $id -and $_.preview } | select -Last 1).preview\n"
            "curl.exe -o $img \"http://127.0.0.1:7861/outputs/$img\""
        ),
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
