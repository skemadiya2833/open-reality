"""
justimagine engines — text-to-image + contextual image editing.
No ComfyUI.

Models:
  - FLUX.2 [klein] 4B (bf16)
  - Qwen-Image-2.1-Turbo (torchao int8) — Diffusers-native quant for 16GB

On 16GB cards the text encoder and DiT cannot all stay in VRAM at once.
Default keeps the DiT and VAE on the GPU for the life of the process. The text
encoder is moved on only for the prompt, then the DiT comes back and stays —
including between images. Diffusers' enable_model_cpu_offload() is not used:
its maybe_free_model_hooks() sends every module back to CPU at the end of each
__call__, which reloads the whole stack on the next image.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
QWEN_MODEL_ID = "addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit"
STEPS = 4
GUIDANCE = 1.0
DEFAULT_SIZE = 1024
DEFAULT_MODEL = "flux-klein"

# Reduce fragmentation when staging modules in/out of VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODELS: dict[str, dict[str, Any]] = {
    "flux-klein": {
        "id": "flux-klein",
        "label": "FLUX.2 klein",
        "short": "4B",
        "color": "#1a73e8",
        "repo": MODEL_ID,
        "marker": ROOT / ".model_ready",
        "quant": "bf16",
        "default_steps": 4,
        "default_guidance": 1.0,
        "default_size": 1024,
        "license_note": "Apache 2.0",
        "aspects": {
            "1:1": (1024, 1024),
            "9:16": (768, 1344),
            "3:4": (896, 1152),
            "4:3": (1152, 896),
            "16:9": (1344, 768),
        },
        "qualities": {
            "low": {"steps": 4, "guidance": 1.0, "short": "Low"},
            "medium": {"steps": 8, "guidance": 1.0, "short": "Med"},
            "high": {"steps": 16, "guidance": 1.5, "short": "High"},
        },
    },
    "qwen-turbo": {
        "id": "qwen-turbo",
        "label": "Qwen Image 2.1 Turbo",
        "short": "int8",
        "color": "#0d9488",
        "repo": QWEN_MODEL_ID,
        "marker": ROOT / ".model_ready_qwen",
        "quant": "int8",
        "default_steps": 8,
        "default_guidance": 1.0,
        "default_size": 1024,
        "license_note": "Qwen Research (non-commercial)",
        "aspects": {
            "1:1": (1024, 1024),
            "9:16": (768, 1344),
            "3:4": (896, 1152),
            "4:3": (1152, 896),
            "16:9": (1344, 768),
        },
        # Turbo ships an 8-step sigma schedule; steps are soft UI hints only.
        "qualities": {
            "low": {"steps": 8, "guidance": 1.0, "short": "Low"},
            "medium": {"steps": 8, "guidance": 1.0, "short": "Med"},
            "high": {"steps": 8, "guidance": 1.0, "short": "High"},
        },
    },
}


def model_ready(model_id: str) -> bool:
    cfg = MODELS.get(model_id)
    if not cfg:
        return False
    marker: Path = cfg["marker"]
    if marker.is_file():
        p = marker.read_text(encoding="utf-8").strip()
        if p and Path(p).is_dir():
            return True
    return False


def list_models(active_id: str | None = None) -> list[dict[str, Any]]:
    active = active_id if active_id in MODELS else DEFAULT_MODEL
    out: list[dict[str, Any]] = []
    for mid, cfg in MODELS.items():
        out.append(
            {
                "id": mid,
                "label": cfg["label"],
                "short": cfg["short"],
                "color": cfg["color"],
                "quant": cfg["quant"],
                "ready": model_ready(mid),
                "license_note": cfg["license_note"],
                "default_steps": cfg["default_steps"],
                "default_guidance": cfg["default_guidance"],
                "default_size": cfg["default_size"],
                "aspects": cfg["aspects"],
                "qualities": cfg["qualities"],
                "active": mid == active,
            }
        )
    return out


def get_model_cfg(model_id: str | None) -> dict[str, Any]:
    if model_id in MODELS:
        return MODELS[model_id]
    return MODELS[DEFAULT_MODEL]


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(round(float(seconds))))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def estimate_wall_clock_s(
    width: int = DEFAULT_SIZE,
    height: int = DEFAULT_SIZE,
    steps: int = STEPS,
    has_ref: bool = False,
    warm: bool = True,
    quant: str = "bf16",
    compiled: bool = False,
    avg_took_s: float | None = None,
) -> dict:
    """ETA for FLUX.2 klein 4B on 16GB with staged residency (warm)."""
    mp = max(0.25, (width * height) / (1024 * 1024))
    # Resident warm @ 1024² / 4 steps — ~12–16s class after residency fix
    per_mid = max(8.0, 14.0 * mp * (steps / 4.0) * (1.2 if has_ref else 1.0))
    if compiled:
        per_mid *= 0.85
    if quant == "fp8":
        per_mid *= 0.75
    elif quant == "int8":
        # Qwen Turbo int8: denoise is faster than bf16, but encode+VAE decode dominate.
        per_mid *= 1.15 if has_ref else 0.95
    if not warm:
        per_mid += 180.0
    if warm and isinstance(avg_took_s, (int, float)) and avg_took_s > 0:
        # Blend heuristic with recent wall-clock (same size/steps assumed by caller).
        per_mid = 0.45 * per_mid + 0.55 * float(avg_took_s)
    return {
        "total_lo_s": round(per_mid * (0.7 if warm else 0.5)),
        "total_mid_s": round(per_mid),
        "total_hi_s": round(per_mid * (1.7 if warm else 2.2)),
        "warm": warm,
        "note": "warm resident DiT" if warm else "includes first model load from disk",
    }


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _snap_multiple(n: int, multiple: int = 16) -> int:
    return max(multiple, (n // multiple) * multiple)


def _model_path(model_id: str = "flux-klein") -> str:
    cfg = get_model_cfg(model_id)
    marker: Path = cfg["marker"]
    if marker.is_file():
        p = marker.read_text(encoding="utf-8").strip()
        if p and Path(p).is_dir():
            return p
    return str(cfg["repo"])


def _vram_stats() -> str:
    if not torch.cuda.is_available():
        return "no cuda"
    alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    free, total = torch.cuda.mem_get_info()
    return (
        f"alloc {alloc:.1f}GB · reserved {reserved:.1f}GB · "
        f"free {free / (1024 ** 3):.1f}/{total / (1024 ** 3):.1f}GB"
    )


def _vram_gb() -> dict | None:
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    free, total = torch.cuda.mem_get_info()
    return {
        "alloc_gb": round(alloc, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free / (1024 ** 3), 2),
        "total_gb": round(total / (1024 ** 3), 2),
    }


def _want_full_gpu() -> bool:
    return os.environ.get("JUSTIMAGINE_FULL", "").strip().lower() in ("1", "true", "yes", "on")


def _want_compile() -> bool:
    # Opt in with JUSTIMAGINE_COMPILE=1. Default off on Windows: inductor needs
    # Triton, which is not reliably available there. Non-Windows defaults on.
    raw = os.environ.get("JUSTIMAGINE_COMPILE", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return os.name != "nt"


def _enable_fast_kernels() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def _install_resident_offload(pipe, device: torch.device) -> None:
    """Pin DiT + VAE on `device`. Shuttle the text encoder for its one forward.

    The text encoder and the DiT each need ~7GB, so they take turns. The VAE is
    ~0.16GB and stays beside the DiT. Nothing is moved per denoising step, and
    nothing is dropped back to CPU when the pipeline returns.
    """
    from accelerate.utils.operations import send_to_device

    transformer = pipe.transformer
    text_encoder = pipe.text_encoder
    vae = pipe.vae

    text_encoder.to("cpu")
    vae.to(device)
    transformer.to(device)

    original_forward = text_encoder.forward

    def forward(*args, **kwargs):
        print("[staged] text encoder → GPU (diffusion model stepped aside)", flush=True)
        transformer.to("cpu")
        free_vram()
        text_encoder.to(device)
        try:
            return original_forward(
                *send_to_device(args, device),
                **send_to_device(kwargs, device),
            )
        finally:
            text_encoder.to("cpu")
            free_vram()
            transformer.to(device)
            print("[staged] diffusion model back on GPU", flush=True)

    text_encoder.forward = forward

    def _keep_resident() -> None:
        for component in pipe.components.values():
            reset = getattr(component, "_reset_stateful_cache", None)
            if reset is not None:
                reset()

    pipe.maybe_free_model_hooks = _keep_resident


class FluxKlein:
    model_id = "flux-klein"

    def __init__(self) -> None:
        self.pipe = None
        self.device_mode = "unloaded"
        self.load_s: float | None = None
        self.last_infer_s: float | None = None
        self.last_timing: dict | None = None
        self._compiled = False
        self.quant = "bf16"

    def load(self, on_status=None) -> None:
        if self.pipe is not None:
            return

        def status(msg: str, progress: float | None = None) -> None:
            if on_status is not None:
                on_status(msg, progress)

        from diffusers import Flux2KleinPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        dtype = torch.bfloat16
        src = _model_path("flux-klein")
        print(f"[FluxKlein] loading {src}…")
        status("loading FLUX.2 klein weights from disk (one-time)…", 0.02)
        t0 = time.time()
        local_only = src != MODEL_ID

        _enable_fast_kernels()
        free_vram()

        self.pipe = Flux2KleinPipeline.from_pretrained(
            src,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        self.quant = "bf16"

        force_full = _want_full_gpu()
        if force_full:
            status("JUSTIMAGINE_FULL=1 — placing all modules on CUDA…", 0.06)
            try:
                self.pipe.to("cuda")
                self.device_mode = "full"
                print(f"[FluxKlein] device_mode=full · {_vram_stats()}")
                print(
                    "[FluxKlein] NOTE: full residency often hits Shared GPU memory on 16GB "
                    "and becomes much slower than staged."
                )
            except torch.cuda.OutOfMemoryError:
                free_vram()
                force_full = False
                print("[FluxKlein] full OOM — falling back to staged")

        if not force_full:
            status("pinning diffusion model on GPU (stays loaded between images)…", 0.06)
            _install_resident_offload(self.pipe, torch.device("cuda"))
            self.device_mode = "staged"
            print(f"[FluxKlein] device_mode=staged · {_vram_stats()}")
            print(
                "[FluxKlein] DiT + VAE stay on GPU. Text encoder visits only for the prompt "
                "(they cannot share a 16GB card)."
            )

        if _want_compile():
            try:
                status("torch.compile transformer (first run compiles, then faster)…", 0.08)
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer, mode="reduce-overhead", fullgraph=False
                )
                self._compiled = True
                print("[FluxKlein] transformer compiled")
            except Exception as exc:
                print(f"[FluxKlein] torch.compile skipped: {exc}")

        self.load_s = time.time() - t0
        print(f"[FluxKlein] load took {format_duration(self.load_s)}")
        status(
            f"model ready ({self.device_mode}"
            f"{' · compiled' if self._compiled else ''}"
            f") · load {format_duration(self.load_s)}",
            0.1,
        )

    def unload(self) -> None:
        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        self.device_mode = "unloaded"
        self._compiled = False
        free_vram()

    def generate(
        self,
        prompt: str,
        image: Image.Image | list[Image.Image] | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        seed: int | None = None,
        steps: int = STEPS,
        guidance: float = GUIDANCE,
        on_status=None,
    ) -> Image.Image:
        def status(msg: str, progress: float | None = None) -> None:
            if on_status is not None:
                on_status(msg, progress)

        was_cold = self.pipe is None
        self.load(on_status=on_status)
        assert self.pipe is not None

        if self.device_mode == "full":
            status(
                "device_mode=full — if this is multi-minute, Windows is paging VRAM; "
                "restart without JUSTIMAGINE_FULL=1",
                0.1,
            )

        width = _snap_multiple(width)
        height = _snap_multiple(height)
        steps = max(1, min(50, int(steps)))
        guidance = float(guidance)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance,
            "num_inference_steps": steps,
            "generator": generator,
        }
        if image is not None:
            if isinstance(image, (list, tuple)):
                refs = [im.convert("RGB") for im in image if im is not None]
                if not refs:
                    status("text → image…", 0.12)
                else:
                    status(f"editing with {len(refs)} reference image(s)…", 0.12)
                    kwargs["image"] = refs if len(refs) > 1 else refs[0]
            else:
                status("editing with reference image…", 0.12)
                kwargs["image"] = image.convert("RGB")
        else:
            status("text → image…", 0.12)

        step_t0 = time.time()
        first_step_t: float | None = None
        peak_alloc = 0.0

        def _cb(pipe, step, timestep, callback_kwargs):
            nonlocal peak_alloc, first_step_t
            if first_step_t is None:
                first_step_t = time.time()
            total = steps
            done = step + 1
            elapsed = time.time() - step_t0
            eta = (elapsed / done) * (total - done) if done else 0
            if torch.cuda.is_available():
                peak_alloc = max(peak_alloc, torch.cuda.memory_allocated(0) / (1024 ** 3))
            status(
                f"diffusion {done}/{total} · ~{format_duration(eta)} left · {self.device_mode}",
                0.15 + 0.80 * (done / total),
            )
            return callback_kwargs

        try:
            out = self.pipe(**kwargs, callback_on_step_end=_cb)
        except TypeError:
            status("diffusion (no step callback)…", 0.4)
            out = self.pipe(**kwargs)

        end_t = time.time()
        self.last_infer_s = end_t - step_t0
        encode_s = (first_step_t - step_t0) if first_step_t else None
        denoise_s = (end_t - first_step_t) if first_step_t else self.last_infer_s
        # Decode is folded into post-loop time inside the pipeline; report denoise
        # as step-loop and encode as everything before the first step callback.
        self.last_timing = {
            "total_s": round(self.last_infer_s, 2),
            "encode_s": round(encode_s, 2) if encode_s is not None else None,
            "denoise_s": round(denoise_s, 2) if denoise_s is not None else None,
            "peak_gb": round(peak_alloc, 2) if peak_alloc else None,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "seed": seed,
            "device_mode": self.device_mode,
            "compiled": self._compiled,
            "quant": self.quant,
        }
        parts = [f"infer {format_duration(self.last_infer_s)}"]
        if was_cold and self.load_s is not None:
            parts.insert(0, f"load {format_duration(self.load_s)}")
        if encode_s is not None:
            parts.append(f"encode {format_duration(encode_s)}")
        if denoise_s is not None:
            parts.append(f"denoise {format_duration(denoise_s)}")
        if peak_alloc:
            parts.append(f"peak {peak_alloc:.1f}GB")
        status(f"done · {' · '.join(parts)} ({self.device_mode})", 1.0)
        print(f"[FluxKlein] generate {' · '.join(parts)} · mode={self.device_mode} · {_vram_stats()}")
        return out.images[0]


class QwenImage21Turbo:
    """Qwen-Image-2.1-Turbo torchao int8 — same staged residency as FluxKlein."""

    model_id = "qwen-turbo"

    def __init__(self) -> None:
        self.pipe = None
        self.device_mode = "unloaded"
        self.load_s: float | None = None
        self.last_infer_s: float | None = None
        self.last_timing: dict | None = None
        self._compiled = False
        self.quant = "int8"

    def load(self, on_status=None) -> None:
        if self.pipe is not None:
            return

        def status(msg: str, progress: float | None = None) -> None:
            if on_status is not None:
                on_status(msg, progress)

        from diffusers import QwenImage21Pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        dtype = torch.bfloat16
        src = _model_path("qwen-turbo")
        print(f"[QwenTurbo] loading {src}…")
        status("loading Qwen Image 2.1 Turbo int8 from disk…", 0.02)
        t0 = time.time()
        local_only = src != QWEN_MODEL_ID

        _enable_fast_kernels()
        free_vram()

        # torchao int8 weights need torchao installed at import/load time.
        self.pipe = QwenImage21Pipeline.from_pretrained(
            src,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        self.quant = "int8"

        force_full = _want_full_gpu()
        if force_full:
            status("JUSTIMAGINE_FULL=1 — placing all modules on CUDA…", 0.06)
            try:
                self.pipe.to("cuda")
                self.device_mode = "full"
                print(f"[QwenTurbo] device_mode=full · {_vram_stats()}")
            except torch.cuda.OutOfMemoryError:
                free_vram()
                force_full = False
                print("[QwenTurbo] full OOM — falling back to staged")

        if not force_full:
            status("pinning diffusion model on GPU (stays loaded between images)…", 0.06)
            _install_resident_offload(self.pipe, torch.device("cuda"))
            self.device_mode = "staged"
            print(f"[QwenTurbo] device_mode=staged · {_vram_stats()}")
            print(
                "[QwenTurbo] DiT + VAE stay on GPU. Text encoder visits only for the prompt."
            )

        if _want_compile():
            try:
                status("torch.compile transformer (first run compiles, then faster)…", 0.08)
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer, mode="reduce-overhead", fullgraph=False
                )
                self._compiled = True
                print("[QwenTurbo] transformer compiled")
            except Exception as exc:
                print(f"[QwenTurbo] torch.compile skipped: {exc}")

        self.load_s = time.time() - t0
        print(f"[QwenTurbo] load took {format_duration(self.load_s)}")
        status(
            f"model ready ({self.device_mode}"
            f"{' · compiled' if self._compiled else ''}"
            f") · load {format_duration(self.load_s)}",
            0.1,
        )

    def unload(self) -> None:
        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        self.device_mode = "unloaded"
        self._compiled = False
        free_vram()

    def generate(
        self,
        prompt: str,
        image: Image.Image | list[Image.Image] | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        seed: int | None = None,
        steps: int = 8,
        guidance: float = 1.0,
        on_status=None,
    ) -> Image.Image:
        def status(msg: str, progress: float | None = None) -> None:
            if on_status is not None:
                on_status(msg, progress)

        was_cold = self.pipe is None
        self.load(on_status=on_status)
        assert self.pipe is not None

        width = _snap_multiple(width)
        height = _snap_multiple(height)
        steps = max(1, min(50, int(steps)))
        guidance = float(guidance)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # Turbo ships an 8-step sigma grid; num_inference_steps alone does not override it.
        sample_sigmas = getattr(self.pipe, "sample_sigmas", None)
        if sample_sigmas is None:
            sample_sigmas = getattr(getattr(self.pipe, "config", None), "sample_sigmas", None)
        turbo_steps = len(sample_sigmas) if sample_sigmas is not None else 8
        track_steps = max(1, int(turbo_steps))

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "generator": generator,
            "use_kv_cache": True,
        }
        # QwenImage21Pipeline uses true_cfg_scale (not guidance_scale). Turbo is CFG≈1.
        # Never enable CFG>1 here — it roughly doubles compute and feels "stuck".
        if guidance is not None and float(guidance) > 1.0:
            print(
                f"[QwenTurbo] ignoring guidance={guidance} (Turbo stays CFG=1)",
                flush=True,
            )

        if image is not None:
            if isinstance(image, (list, tuple)):
                refs = [im.convert("RGB") for im in image if im is not None]
                if not refs:
                    status("encoding prompt…", 0.08)
                else:
                    status(f"encoding prompt + {len(refs)} reference(s)…", 0.08)
                    kwargs["image"] = refs if len(refs) > 1 else refs[0]
            else:
                status("encoding prompt + reference…", 0.08)
                kwargs["image"] = image.convert("RGB")
        else:
            status("encoding prompt…", 0.08)

        step_t0 = time.time()
        first_step_t: float | None = None
        peak_alloc = 0.0

        def _cb(pipe, step, timestep, callback_kwargs):
            nonlocal peak_alloc, first_step_t
            if first_step_t is None:
                first_step_t = time.time()
            total = track_steps
            done = min(step + 1, total)
            elapsed = time.time() - step_t0
            if torch.cuda.is_available():
                peak_alloc = max(peak_alloc, torch.cuda.memory_allocated(0) / (1024 ** 3))
            if done >= total:
                # Last denoise step finished — VAE decode still runs inside the pipe.
                status(f"decoding image… · {self.device_mode}", 0.92)
            else:
                eta = (elapsed / done) * (total - done) if done else 0
                # Leave headroom so 100% is only hit after decode, not at last step.
                status(
                    f"diffusion {done}/{total} · ~{format_duration(eta)} left · {self.device_mode}",
                    0.12 + 0.75 * (done / total),
                )
            return callback_kwargs

        try:
            with torch.inference_mode():
                out = self.pipe(**kwargs, callback_on_step_end=_cb)
        except TypeError:
            status("diffusion (no step callback)…", 0.4)
            with torch.inference_mode():
                out = self.pipe(**kwargs)

        status("finishing…", 0.97)
        end_t = time.time()
        self.last_infer_s = end_t - step_t0
        encode_s = (first_step_t - step_t0) if first_step_t else None
        denoise_s = (end_t - first_step_t) if first_step_t else self.last_infer_s
        self.last_timing = {
            "total_s": round(self.last_infer_s, 2),
            "encode_s": round(encode_s, 2) if encode_s is not None else None,
            "denoise_s": round(denoise_s, 2) if denoise_s is not None else None,
            "peak_gb": round(peak_alloc, 2) if peak_alloc else None,
            "width": width,
            "height": height,
            "steps": track_steps,
            "guidance": 1.0,
            "seed": seed,
            "device_mode": self.device_mode,
            "compiled": self._compiled,
            "quant": self.quant,
            "model": self.model_id,
        }
        parts = [f"infer {format_duration(self.last_infer_s)}"]
        if was_cold and self.load_s is not None:
            parts.insert(0, f"load {format_duration(self.load_s)}")
        if encode_s is not None:
            parts.append(f"encode {format_duration(encode_s)}")
        if denoise_s is not None:
            parts.append(f"denoise {format_duration(denoise_s)}")
        if peak_alloc:
            parts.append(f"peak {peak_alloc:.1f}GB")
        status(f"done · {' · '.join(parts)} ({self.device_mode})", 1.0)
        print(f"[QwenTurbo] generate {' · '.join(parts)} · mode={self.device_mode} · {_vram_stats()}")
        return out.images[0]


def create_engine(model_id: str | None = None):
    mid = model_id if model_id in MODELS else DEFAULT_MODEL
    if mid == "qwen-turbo":
        return QwenImage21Turbo()
    return FluxKlein()


def save_image(img: Image.Image, stem: str | None = None) -> Path:
    stem = stem or _stamp()
    path = OUTPUT_DIR / f"{stem}.png"
    img.save(path)
    print(f"[save] {path}")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="justimagine CLI")
    parser.add_argument("prompt", nargs="?", default="a red bicycle leaning on a brick wall")
    parser.add_argument("--image", type=str, default=None, help="reference/edit image")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    args = parser.parse_args()

    cfg = get_model_cfg(args.model)
    ref = Image.open(args.image).convert("RGB") if args.image else None
    engine = create_engine(args.model)
    img = engine.generate(
        args.prompt,
        image=ref,
        width=args.width,
        height=args.height,
        seed=args.seed,
        steps=args.steps if args.steps is not None else cfg["default_steps"],
        guidance=args.guidance if args.guidance is not None else cfg["default_guidance"],
    )
    save_image(img)


if __name__ == "__main__":
    main()
