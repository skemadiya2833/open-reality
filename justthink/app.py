"""
justthink — Hunyuan3D-2.0, fully native GPU execution.

No memory offloading anywhere, no mmgp, no low-VRAM flags / profiles.
Shape stage ~6GB; shape+texture combined ~16GB (official figures).
Both stages run standard full-precision on CUDA.

Sweet-spot defaults (16GB Blackwell comfort):
  - Sequential shape↔paint residency (unload the other stage before each run)
  - Quality presets: Fast (20/192), Balanced (30/256), Quality (50/384)
  - Standard v2-0 checkpoints (not turbo, not 2.1)
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
HY3D_DIR = ROOT / "Hunyuan3D-2"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if HY3D_DIR.is_dir():
    sys.path.insert(0, str(HY3D_DIR))
else:
    print(f"WARNING: {HY3D_DIR} missing — run setup.sh first.")

DEVICE = "cuda"
MODEL_ID = "tencent/Hunyuan3D-2"
SHAPE_SUBFOLDER = "hunyuan3d-dit-v2-0"
PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0"

# Sweet-spot defaults (16GB Blackwell comfort). Override per Quality preset.
SHAPE_STEPS = 30
OCTREE_RES = 256
SDXL_MODEL = "stabilityai/sdxl-turbo"
SDXL_STEPS = 4

# Fast / Balanced / Quality — mesh detail is dominated by octree + shape steps.
# Meshy is a different (closed) stack; we can't match it 1:1, but Quality pushes
# Hunyuan3D-2.0 toward its published high-res settings (50 steps / 384 octree).
QUALITY_PRESETS: dict[str, dict] = {
    "fast": {
        "label": "Fast",
        "blurb": "Draft mesh · 20 steps · octree 192 · quicker iterates",
        "shape_steps": 20,
        "octree": 192,
        "eta_scale": 0.65,
    },
    "balanced": {
        "label": "Balanced",
        "blurb": "Default · 30 steps · octree 256 · 16GB comfort",
        "shape_steps": 30,
        "octree": 256,
        "eta_scale": 1.0,
    },
    "quality": {
        "label": "Quality",
        "blurb": "Max local detail · 50 steps · octree 384 · slower, sharper geometry",
        "shape_steps": 50,
        "octree": 384,
        "eta_scale": 1.75,
    },
}
DEFAULT_QUALITY = "balanced"


def get_quality(name: str | None) -> dict:
    key = (name or DEFAULT_QUALITY).strip().lower()
    if key not in QUALITY_PRESETS:
        key = DEFAULT_QUALITY
    q = dict(QUALITY_PRESETS[key])
    q["id"] = key
    return q


def list_quality_presets() -> list[dict]:
    return [
        {
            "id": qid,
            "label": q["label"],
            "blurb": q["blurb"],
            "shape_steps": q["shape_steps"],
            "octree": q["octree"],
            "default": qid == DEFAULT_QUALITY,
        }
        for qid, q in QUALITY_PRESETS.items()
    ]


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
    mode: str = "full",
    has_image: bool = True,
    quality: str | None = None,
) -> dict:
    """Rough ETA for Hunyuan3D-2.0 on 16GB (sequential shape↔paint)."""
    mode = (mode or "full").strip().lower()
    q = get_quality(quality)
    # Calibrated loosely for warm GPU @ balanced; first load slower.
    if mode == "shape":
        mid = 180.0
    elif mode == "texture":
        mid = 240.0
    else:
        mid = 420.0
    mid *= float(q["eta_scale"])
    if not has_image:
        mid += 25.0  # SDXL-Turbo preview
    return {
        "total_lo_s": round(mid * 0.55),
        "total_mid_s": round(mid),
        "total_hi_s": round(mid * 1.75),
        "mode": mode,
        "quality": q["id"],
        "shape_steps": q["shape_steps"],
        "octree": q["octree"],
        "note": f"{q['label']}: {q['shape_steps']} steps · octree {q['octree']}",
    }


def free_vram() -> None:
    """Reclaim CUDA cache after unloading a stage.

    This is normal sequential stage swapping (shape ↔ paint), not weight
    offloading — we never spill a single model's weights across VRAM/RAM
    mid-inference.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Text → Image (SDXL-Turbo)
# ---------------------------------------------------------------------------


class TextToImage:
    def __init__(self) -> None:
        self.pipe = None

    def load(self) -> None:
        if self.pipe is not None:
            return
        from diffusers import AutoPipelineForText2Image

        print(f"[TextToImage] loading {SDXL_MODEL} on {DEVICE}...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            SDXL_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(DEVICE)

    def unload(self) -> None:
        if self.pipe is None:
            return
        del self.pipe
        self.pipe = None
        free_vram()
        print("[TextToImage] unloaded")

    def generate(self, prompt: str) -> Image.Image:
        self.load()
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=SDXL_STEPS,
            guidance_scale=0.0,
        )
        image = result.images[0].convert("RGBA")
        return image


# ---------------------------------------------------------------------------
# Image → Shape → Texture (native Hunyuan3D-2.0)
# ---------------------------------------------------------------------------


class ImageTo3D:
    """Native Hunyuan3D-2.0 pipelines.

    Keeping both stages resident is possible on paper (~6GB shape + paint),
    but we unload by design so peak VRAM stays comfortably under 16GB rather
    than riding the combined ~16GB ceiling. That sequencing is optional for
    fit, mandatory here for headroom — not the same as mmgp/offload.
    """

    def __init__(self) -> None:
        self.shape = None
        self.paint = None
        self._rembg = None

    def load_shape(self) -> None:
        if self.shape is not None:
            return
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        print(f"[ImageTo3D] loading shape {MODEL_ID}/{SHAPE_SUBFOLDER}...")
        self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID,
            subfolder=SHAPE_SUBFOLDER,
            device=DEVICE,
        )

    def load_paint(self) -> None:
        if self.paint is not None:
            return
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        # Paint from_pretrained has no device= kwarg; config defaults to cuda.
        print(f"[ImageTo3D] loading paint {MODEL_ID}/{PAINT_SUBFOLDER}...")
        self.paint = Hunyuan3DPaintPipeline.from_pretrained(
            MODEL_ID,
            subfolder=PAINT_SUBFOLDER,
        )

    def unload_shape(self) -> None:
        if self.shape is None:
            return
        del self.shape
        self.shape = None
        free_vram()
        print("[ImageTo3D] shape unloaded")

    def unload_paint(self) -> None:
        if self.paint is None:
            return
        del self.paint
        self.paint = None
        free_vram()
        print("[ImageTo3D] paint unloaded")

    def prepare_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGBA") if image.mode != "RGBA" else image
        # Official demo rembg path when source was effectively opaque RGB.
        # If already has transparency, skip.
        alpha = np.array(image.split()[-1])
        if alpha.min() >= 250:
            if self._rembg is None:
                from hy3dgen.rembg import BackgroundRemover

                self._rembg = BackgroundRemover()
            rgb = image.convert("RGB")
            image = self._rembg(rgb)
            if image.mode != "RGBA":
                image = image.convert("RGBA")
        return image

    def image_to_shape(
        self,
        image: Image.Image,
        on_status=None,
        shape_steps: int | None = None,
        octree: int | None = None,
    ):
        def status(msg: str, progress: float | None = None) -> None:
            if on_status is not None:
                on_status(msg, progress)

        steps = int(shape_steps if shape_steps is not None else SHAPE_STEPS)
        octree_res = int(octree if octree is not None else OCTREE_RES)

        # Unload paint first so peaks do not overlap (comfort headroom).
        status("freeing paint VRAM…", 0.18)
        self.unload_paint()
        status("loading shape model (GPU may look idle)…", 0.22)
        self.load_shape()
        status("preparing image…", 0.28)
        image = self.prepare_image(image)
        status(f"shape diffusion {steps} steps · octree {octree_res}…", 0.32)
        mesh = self.shape(
            image=image,
            num_inference_steps=steps,
            octree_resolution=octree_res,
        )[0]
        status("shape done", 0.72)
        return mesh

    def texture_mesh(self, mesh, image: Image.Image, on_status=None):
        def status(msg: str, progress: float | None = None) -> None:
            if on_status is not None:
                on_status(msg, progress)

        status("freeing shape VRAM…", 0.74)
        self.unload_shape()
        status("loading paint / delight models (GPU may look idle)…", 0.76)
        self.load_paint()
        status("preparing image for texture…", 0.82)
        image = self.prepare_image(image)
        status("texture generation (delight + multiview)…", 0.84)
        mesh = self.paint(mesh, image=image)
        status("texture done", 0.98)
        return mesh


# ---------------------------------------------------------------------------
# Live Open3D viewer
# ---------------------------------------------------------------------------


class LiveViewer:
    def __init__(self, window_name: str = "Hunyuan3D-2.0") -> None:
        self.window_name = window_name
        self.vis = None
        self._geom_name = "mesh"

    def _ensure(self) -> None:
        if self.vis is not None:
            return
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name, width=1024, height=768)

    def show(self, mesh_path: Path | str | None = None, trimesh_mesh=None) -> None:
        """Replace the displayed mesh and refresh the window."""
        self._ensure()
        self.vis.clear_geometries()

        if mesh_path is not None:
            path = str(mesh_path)
            mesh = o3d.io.read_triangle_mesh(path)
        elif trimesh_mesh is not None:
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(trimesh_mesh.vertices)),
                o3d.utility.Vector3iVector(np.asarray(trimesh_mesh.faces)),
            )
            if hasattr(trimesh_mesh, "visual") and hasattr(trimesh_mesh.visual, "vertex_colors"):
                colors = np.asarray(trimesh_mesh.visual.vertex_colors)[:, :3] / 255.0
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        else:
            return

        if mesh.is_empty():
            print("[LiveViewer] empty mesh, skip")
            return

        mesh.compute_vertex_normals()
        self.vis.add_geometry(mesh)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.reset_view_point(True)
        print(f"[LiveViewer] showing {mesh_path or 'in-memory mesh'}")

    def poll(self) -> None:
        if self.vis is None:
            return
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        if self.vis is None:
            return
        self.vis.destroy_window()
        self.vis = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

HELP = """
Commands:
  <text prompt>              text → image → 3D (per current mode)
  img <path>                 image → 3D (per current mode)
  mode full|shape|texture    set pipeline mode (default: full)
  help                       show this help
  quit / exit                leave

Modes:
  full     shape + texture
  shape    bare mesh only
  texture  texture an existing mesh from last shape (needs prior shape or img flow)
"""


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _save_mesh(mesh, stem: str) -> Path:
    out = OUTPUT_DIR / f"{stem}.glb"
    mesh.export(str(out))
    print(f"[save] {out}")
    return out


def run_from_image(
    i2_3d: ImageTo3D,
    image: Image.Image,
    mode: str,
    stem: str,
    prior_mesh=None,
    viewer: LiveViewer | None = None,
    on_status=None,
    quality: str | None = None,
):
    q = get_quality(quality)
    mesh = prior_mesh
    if mode in ("full", "shape"):
        print(f"[run] shape generation ({q['id']}: {q['shape_steps']} steps, octree {q['octree']})...")
        mesh = i2_3d.image_to_shape(
            image,
            on_status=on_status,
            shape_steps=q["shape_steps"],
            octree=q["octree"],
        )
        if mode == "shape":
            path = _save_mesh(mesh, f"{stem}_shape")
            i2_3d.unload_shape()
            if viewer is not None:
                viewer.show(path)
            return mesh, path

    if mode in ("full", "texture"):
        if mesh is None:
            raise RuntimeError("texture mode needs a mesh — run shape/full first or use img in full mode")
        print("[run] texture generation...")
        mesh = i2_3d.texture_mesh(mesh, image, on_status=on_status)
        path = _save_mesh(mesh, f"{stem}_textured" if mode == "full" else f"{stem}_texture")
        i2_3d.unload_paint()
        if viewer is not None:
            viewer.show(path)
        return mesh, path

    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is required. Install cu128 PyTorch via setup.sh.")
        sys.exit(1)

    print(HELP)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Outputs: {OUTPUT_DIR}")
    q = get_quality(DEFAULT_QUALITY)
    print(
        f"Quality presets: Fast/Balanced/Quality · default {q['label']} "
        f"({q['shape_steps']} steps / octree {q['octree']}), native v2-0\n"
    )

    t2i = TextToImage()
    i2_3d = ImageTo3D()
    viewer = LiveViewer()
    mode = "full"
    last_mesh = None
    last_image: Image.Image | None = None

    while True:
        viewer.poll()
        try:
            line = input(f"[{mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        lower = line.lower()
        if lower in ("quit", "exit", "q"):
            break
        if lower in ("help", "?"):
            print(HELP)
            continue
        if lower.startswith("mode "):
            new_mode = line.split(None, 1)[1].strip().lower()
            if new_mode not in ("full", "shape", "texture"):
                print("mode must be full | shape | texture")
                continue
            mode = new_mode
            print(f"mode = {mode}")
            continue

        try:
            stem = _stamp()
            if lower.startswith("img "):
                path = Path(line.split(None, 1)[1].strip().strip('"'))
                if not path.is_file():
                    print(f"file not found: {path}")
                    continue
                image = Image.open(path)
                last_image = image
                t2i.unload()
                last_mesh, _ = run_from_image(
                    i2_3d,
                    image,
                    mode,
                    stem,
                    prior_mesh=last_mesh if mode == "texture" else None,
                    viewer=viewer,
                )
            else:
                prompt = line
                print(f"[run] text→image: {prompt!r}")
                image = t2i.generate(prompt)
                preview = OUTPUT_DIR / f"{stem}_preview.png"
                image.save(preview)
                print(f"[save] {preview}")
                last_image = image
                t2i.unload()
                last_mesh, _ = run_from_image(
                    i2_3d,
                    image,
                    mode,
                    stem,
                    prior_mesh=last_mesh if mode == "texture" else None,
                    viewer=viewer,
                )
        except Exception as exc:
            print(f"[error] {exc}")
            free_vram()

    t2i.unload()
    i2_3d.unload_shape()
    i2_3d.unload_paint()
    viewer.close()
    print("bye")


if __name__ == "__main__":
    main()
