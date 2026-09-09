"""
FLUX.2 [klein] 4B — text-to-image + contextual image editing.
No ComfyUI.

On 16GB cards, Qwen3 text-encoder + DiT + VAE cannot all stay in VRAM at once
(~15.5GB+). Forcing `.to("cuda")` on everything looks like "full" but Windows
pages into Shared GPU memory → multi-minute gens at 100% CUDA / cool GPU.

Default = Diffusers stage offload (text_encoder → transformer → VAE), same as the
official HF recipe that targets ~13GB / ~10–20s warm. Set JUSTIMAGINE_FULL=1 only
if you have true headroom.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
STEPS = 4
GUIDANCE = 1.0
DEFAULT_SIZE = 1024

# Reduce fragmentation when staging modules in/out of VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
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
) -> dict:
    """ETA for FLUX.2 klein 4B on 16GB with staged residency (warm)."""
    mp = max(0.25, (width * height) / (1024 * 1024))
    # Staged warm @ 1024² / 4 steps — ~15s class (not the thrashing "full" path)
    per_mid = max(10.0, 16.0 * mp * (steps / 4.0) * (1.2 if has_ref else 1.0))
    if not warm:
        per_mid += 180.0
    return {
        "total_lo_s": round(per_mid * (0.7 if warm else 0.5)),
        "total_mid_s": round(per_mid),
        "total_hi_s": round(per_mid * (1.7 if warm else 2.2)),
        "warm": warm,
        "note": "warm staged (TE→DiT→VAE)" if warm else "includes first model load from disk",
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


def _model_path() -> str:
    marker = ROOT / ".model_ready"
    if marker.is_file():
        p = marker.read_text(encoding="utf-8").strip()
        if p and Path(p).is_dir():
            return p
    return MODEL_ID


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


def _want_full_gpu() -> bool:
    return os.environ.get("JUSTIMAGINE_FULL", "").strip().lower() in ("1", "true", "yes", "on")


class FluxKlein:
    def __init__(self) -> None:
        self.pipe = None
        self.device_mode = "unloaded"
        self.load_s: float | None = None
        self.last_infer_s: float | None = None
        self._compiled = False

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
        src = _model_path()
        print(f"[FluxKlein] loading {src}…")
        status("loading FLUX.2 klein weights from disk (one-time)…", 0.02)
        t0 = time.time()
        local_only = src != MODEL_ID

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        free_vram()

        self.pipe = Flux2KleinPipeline.from_pretrained(
            src,
            torch_dtype=dtype,
            local_files_only=local_only,
        )

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
            # Official HF 16GB path: only one stage in VRAM at a time.
            # This is NOT layer-by-layer sequential offload thrashing.
            status("enabling staged residency (text_encoder → transformer → VAE)…", 0.06)
            self.pipe.enable_model_cpu_offload()
            self.device_mode = "staged"
            print(f"[FluxKlein] device_mode=staged · {_vram_stats()}")
            print(
                "[FluxKlein] staged keeps peak ~13GB — avoids Windows Shared-GPU paging "
                "that made 'full' take ~4 minutes"
            )

        if os.environ.get("JUSTIMAGINE_COMPILE", "").strip().lower() in ("1", "true", "yes", "on"):
            try:
                status("torch.compile transformer (first run will be slower)…", 0.08)
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer, mode="reduce-overhead", fullgraph=False
                )
                self._compiled = True
                print("[FluxKlein] transformer compiled (JUSTIMAGINE_COMPILE=1)")
            except Exception as exc:
                print(f"[FluxKlein] torch.compile skipped: {exc}")

        self.load_s = time.time() - t0
        print(f"[FluxKlein] load took {format_duration(self.load_s)}")
        status(f"model ready ({self.device_mode}) · load {format_duration(self.load_s)}", 0.1)

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
        image: Image.Image | None = None,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
        seed: int | None = None,
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
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        kwargs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": GUIDANCE,
            "num_inference_steps": STEPS,
            "generator": generator,
        }
        if image is not None:
            status("editing with reference image…", 0.12)
            kwargs["image"] = image.convert("RGB")
        else:
            status("text → image…", 0.12)

        step_t0 = time.time()
        peak_alloc = 0.0

        def _cb(pipe, step, timestep, callback_kwargs):
            nonlocal peak_alloc
            total = STEPS
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

        self.last_infer_s = time.time() - step_t0
        parts = [f"infer {format_duration(self.last_infer_s)}"]
        if was_cold and self.load_s is not None:
            parts.insert(0, f"load {format_duration(self.load_s)}")
        if peak_alloc:
            parts.append(f"peak {peak_alloc:.1f}GB")
        status(f"done · {' · '.join(parts)} ({self.device_mode})", 1.0)
        print(f"[FluxKlein] generate {' · '.join(parts)} · mode={self.device_mode} · {_vram_stats()}")
        return out.images[0]


def save_image(img: Image.Image, stem: str | None = None) -> Path:
    stem = stem or _stamp()
    path = OUTPUT_DIR / f"{stem}.png"
    img.save(path)
    print(f"[save] {path}")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FLUX.2 klein 4B CLI")
    parser.add_argument("prompt", nargs="?", default="a red bicycle leaning on a brick wall")
    parser.add_argument("--image", type=str, default=None, help="reference/edit image")
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    ref = Image.open(args.image).convert("RGB") if args.image else None
    engine = FluxKlein()
    img = engine.generate(args.prompt, image=ref, width=args.width, height=args.height, seed=args.seed)
    save_image(img)


if __name__ == "__main__":
    main()
