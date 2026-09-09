"""
FLUX.2 [klein] 4B — text-to-image + contextual image editing.
No ComfyUI. bf16 on CUDA; CPU offload only if full load OOMs.
"""

from __future__ import annotations

import gc
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
) -> dict:
    """Rough ETA for FLUX.2 klein 4B on 16GB (full GPU after warm)."""
    mp = max(0.25, (width * height) / (1024 * 1024))
    # ~35s mid @ 1024² / 4 steps warm; edit slightly slower
    per_mid = max(12.0, 35.0 * mp * (steps / 4.0) * (1.15 if has_ref else 1.0))
    return {
        "total_lo_s": round(per_mid * 0.55),
        "total_mid_s": round(per_mid),
        "total_hi_s": round(per_mid * 1.8),
        "note": "warm GPU; first run after load is slower",
    }


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
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


class FluxKlein:
    def __init__(self) -> None:
        self.pipe = None
        self.device_mode = "unloaded"

    def load(self) -> None:
        if self.pipe is not None:
            return
        from diffusers import Flux2KleinPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        dtype = torch.bfloat16
        src = _model_path()
        print(f"[FluxKlein] loading {src}…")
        local_only = src != MODEL_ID
        self.pipe = Flux2KleinPipeline.from_pretrained(
            src,
            torch_dtype=dtype,
            local_files_only=local_only,
        )
        try:
            self.pipe.to("cuda")
            self.device_mode = "full"
            print("[FluxKlein] device_mode=full")
        except torch.cuda.OutOfMemoryError:
            free_vram()
            self.pipe.enable_model_cpu_offload()
            self.device_mode = "offload"
            print("[FluxKlein] OOM on full load → device_mode=offload")

    def unload(self) -> None:
        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        self.device_mode = "unloaded"
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

        self.load()
        assert self.pipe is not None

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
            status("editing with reference image…", 0.08)
            kwargs["image"] = image.convert("RGB")
        else:
            status("text → image…", 0.08)

        step_t0 = time.time()

        def _cb(pipe, step, timestep, callback_kwargs):
            total = STEPS
            done = step + 1
            elapsed = time.time() - step_t0
            eta = (elapsed / done) * (total - done) if done else 0
            status(
                f"diffusion {done}/{total} · ~{format_duration(eta)} left",
                0.12 + 0.82 * (done / total),
            )
            return callback_kwargs

        try:
            out = self.pipe(**kwargs, callback_on_step_end=_cb)
        except TypeError:
            status("diffusion (no step callback)…", 0.4)
            out = self.pipe(**kwargs)

        status("done", 1.0)
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
