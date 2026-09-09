"""
LTX-2.3 Distilled — continuity-first story video.

Fixes the overnight failure mode (random 1–2s style-jumping clips) by:
  • one coherent clip per script beat (not 10 regenerations of the same line)
  • locked visual style on every prompt + strong anti-style-drift negative
  • last-frame conditioning so clip N+1 continues from clip N (LTX2ConditionPipeline)
  • optional reference image/video that seeds the chain and is remembered in plan.json

Device strategy on 16GB (same idea as justimagine):
  • default = Diffusers module CPU offload (TE→DiT→VAE) — reliable on 16GB
  • UI Fast/Balanced/Quality picks resolution+frames (frames dominate wall time)
  • `JUSTDREAM_GROUP=1` tries group offload (rebuilds pipe on OOM)
  • `JUSTDREAM_FULL=1` forces all-on-GPU (often pages Shared GPU memory)
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
STORIES_DIR = OUTPUT_DIR / "stories"
UPLOADS_DIR = OUTPUT_DIR / "uploads"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STORIES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Helps when staging large LTX modules in/out of 16GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = "OzzyGT/LTX-2.3-Distilled-bnb-nf4"

# Locked look — applied to EVERY beat. Change via UI "style lock".
DEFAULT_STYLE_LOCK = (
    "photorealistic live-action cinematic film still, natural skin texture, "
    "consistent character identity, consistent wardrobe and lighting, 35mm lens, "
    "shallow depth of field, color graded like a single continuous movie shot"
)

# Ban the style soup that ruined overnight runs.
STYLE_NEGATIVE = (
    "anime, manga, cartoon, comic, 3d render, cgi, pixar, unreal engine, "
    "claymation, watercolor, sketch, different art style, style change between frames, "
    "identity morph, face swap, flickering face, inconsistent character, "
    "worst quality, blurry, jittery, distorted, static, watermark, text overlay"
)

GUIDANCE = 1.0
MAX_SEGMENTS = 60


def _snap32(n: int) -> int:
    return max(32, (n // 32) * 32)


def _frames_8n1(n: int) -> int:
    if n < 9:
        return 9
    return ((n - 1) // 8) * 8 + 1


# Continuity-first presets. scene_seconds ≈ one clip length (NOT multi-regen spam).
PRESETS: dict[str, dict] = {
    "continuity": {
        "label": "Continuity · 640p · ~2s chained",
        "blurb": "Best default: last-frame chain + style lock · 640×384 · 49f · 24fps",
        "width": 640,
        "height": 384,
        "frames": 49,
        "fps": 24.0,
        "scene_seconds": 2.0,
        "fallbacks": (49, 33, 25),
        "chain": True,
    },
    "continuity_long": {
        "label": "Continuity · 640p · ~4s chained",
        "blurb": "Longer beats · 640×384 · 97f · 24fps (slower, better motion)",
        "width": 640,
        "height": 384,
        "frames": 97,
        "fps": 24.0,
        "scene_seconds": 4.0,
        "fallbacks": (97, 73, 49, 33),
        "chain": True,
    },
    "continuity_720p": {
        "label": "Continuity · 720p · ~1.4s chained",
        "blurb": "1280×704 · 25f · 24fps · chained (VRAM-heavy under offload)",
        "width": 1280,
        "height": 704,
        "frames": 25,
        "fps": 24.0,
        "scene_seconds": 1.0,
        "fallbacks": (25, 17, 9),
        "chain": True,
    },
    "cinema": {
        "label": "Cinema · 768p · ~2s chained",
        "blurb": "768×448 · 49f · 24fps · chained",
        "width": 768,
        "height": 448,
        "frames": 49,
        "fps": 24.0,
        "scene_seconds": 2.0,
        "fallbacks": (49, 33, 25),
        "chain": True,
    },
    "fast": {
        "label": "Fast draft · 512p · chained",
        "blurb": "512×320 · 33f · quick overnight tests with continuity",
        "width": 512,
        "height": 320,
        "frames": 33,
        "fps": 24.0,
        "scene_seconds": 1.4,
        "fallbacks": (33, 25, 17),
        "chain": True,
    },
}

DEFAULT_PRESET = "fast"
_p0 = PRESETS[DEFAULT_PRESET]
DEFAULT_WIDTH = int(_p0["width"])
DEFAULT_HEIGHT = int(_p0["height"])
DEFAULT_FRAMES = int(_p0["frames"])
DEFAULT_FPS = float(_p0["fps"])
SAFE_FRAMES_FALLBACK = tuple(_p0["fallbacks"])


def get_preset(name: str | None) -> dict:
    key = (name or DEFAULT_PRESET).strip().lower()
    if key not in PRESETS:
        key = DEFAULT_PRESET
    p = dict(PRESETS[key])
    p["id"] = key
    p["width"] = _snap32(int(p["width"]))
    p["height"] = _snap32(int(p["height"]))
    p["frames"] = _frames_8n1(int(p["frames"]))
    p["fps"] = float(p["fps"])
    p["scene_seconds"] = float(p["scene_seconds"])
    p["fallbacks"] = tuple(_frames_8n1(int(f)) for f in p["fallbacks"])
    p["chain"] = bool(p.get("chain", True))
    return p


def list_presets() -> list[dict]:
    out = []
    for pid in PRESETS:
        p = get_preset(pid)
        out.append(
            {
                "id": pid,
                "label": p["label"],
                "blurb": p["blurb"],
                "width": p["width"],
                "height": p["height"],
                "frames": p["frames"],
                "fps": p["fps"],
                "scene_seconds": p["scene_seconds"],
                "clips_per_scene": 1,
                "chain": p["chain"],
                "default": pid == DEFAULT_PRESET,
            }
        )
    return out


def max_frames_for_res(width: int, height: int) -> int:
    return get_preset(DEFAULT_PRESET)["frames"]


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _model_path() -> str:
    marker = ROOT / ".model_ready"
    if marker.is_file():
        p = marker.read_text(encoding="utf-8").strip()
        if p and Path(p).is_dir() and "nunchaku" not in p.lower():
            return p
    return MODEL_ID


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _want_full_gpu() -> bool:
    return _env_flag("JUSTDREAM_FULL")


def _want_group_offload() -> bool:
    """Opt-in only — group offload often OOMs during setup on 16GB + bnb."""
    return _env_flag("JUSTDREAM_GROUP")


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


# UI speed modes → preset ids (resolution/frames dominate wall time on 16GB)
SPEED_MODES: dict[str, dict] = {
    "fast": {
        "label": "Fast",
        "blurb": "512×320 · 33f · quick drafts (~half the wait of Balanced)",
        "preset": "fast",
    },
    "balanced": {
        "label": "Balanced",
        "blurb": "640×384 · 49f · continuity default",
        "preset": "continuity",
    },
    "quality": {
        "label": "Quality",
        "blurb": "768×448 · 49f · sharper, slower",
        "preset": "cinema",
    },
}


def list_speed_modes() -> list[dict]:
    out = []
    for sid, s in SPEED_MODES.items():
        p = get_preset(s["preset"])
        out.append(
            {
                "id": sid,
                "label": s["label"],
                "blurb": s["blurb"],
                "preset": s["preset"],
                "width": p["width"],
                "height": p["height"],
                "frames": p["frames"],
                "default": sid == "fast",
            }
        )
    return out


def resolve_speed_mode(speed: str | None, preset: str | None) -> tuple[str, str]:
    """Return (speed_id, preset_id). Explicit preset wins if speed empty."""
    speed = (speed or "").strip().lower()
    preset = (preset or "").strip().lower()
    if speed in SPEED_MODES:
        return speed, SPEED_MODES[speed]["preset"]
    if preset in PRESETS:
        # Infer speed from preset when possible
        for sid, s in SPEED_MODES.items():
            if s["preset"] == preset:
                return sid, preset
        return "custom", preset
    return "fast", SPEED_MODES["fast"]["preset"]


def _distill_guidance() -> dict:
    return dict(
        guidance_scale=GUIDANCE,
        audio_guidance_scale=GUIDANCE,
        stg_scale=0.0,
        audio_stg_scale=0.0,
        modality_scale=1.0,
        audio_modality_scale=1.0,
        guidance_rescale=0.0,
        audio_guidance_rescale=0.0,
        spatio_temporal_guidance_blocks=None,
    )


def _negative_prompt(style_lock: str) -> str:
    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

    return f"{DEFAULT_NEGATIVE_PROMPT}, {STYLE_NEGATIVE}, not {style_lock[:80]}"


class _Progress:
    def __init__(self, on_status, lo: float = 0.0, hi: float = 1.0) -> None:
        self._on_status = on_status
        self.lo = lo
        self.hi = hi
        self.value = lo

    def set(self, msg: str, local: float, **extra) -> None:
        local = max(0.0, min(1.0, float(local)))
        self.value = self.lo + (self.hi - self.lo) * local
        if self._on_status is not None:
            self._on_status(msg, self.value, **extra)


# ---------------------------------------------------------------------------
# Script → ONE beat = ONE clip (continuity chain handles time)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _raw_script_units(script: str) -> list[str]:
    text = (script or "").strip()
    if not text:
        return []
    scenes = [s.strip() for s in re.split(r"\n\s*\n+", text) if s.strip()]
    units: list[str] = []
    for scene in scenes:
        parts = [p.strip() for p in _SENTENCE_SPLIT.split(scene) if p.strip()]
        if not parts:
            continue
        if len(parts) == 1 and len(parts[0].split()) > 48:
            words = parts[0].split()
            for i in range(0, len(words), 28):
                units.append(" ".join(words[i : i + 28]))
        else:
            units.extend(parts)
    return units or [text]


def style_prompt(beat: str, style_lock: str, index: int, total: int) -> str:
    b = beat.strip()
    if not b.endswith((".", "!", "?")):
        b += "."
    lock = (style_lock or DEFAULT_STYLE_LOCK).strip().rstrip(".")
    if index == 0:
        return (
            f"{lock}. Opening shot of a continuous scene: {b} "
            "Keep this exact visual style for the entire film."
        )
    return (
        f"{lock}. Seamless continuation of the SAME shot and SAME character "
        f"(beat {index + 1}/{total}): {b} "
        "Match previous frame lighting, wardrobe, face, and camera. Do not change art style."
    )


def split_script_to_beats(
    script: str,
    max_beats: int = MAX_SEGMENTS,
    preset: str | dict | None = None,
    style_lock: str = DEFAULT_STYLE_LOCK,
) -> list[str]:
    """One narrative unit → one prompt. No multi-clip spam per sentence."""
    units = _raw_script_units(script)[:max_beats]
    n = len(units)
    return [style_prompt(u, style_lock, i, n) for i, u in enumerate(units)]


def estimate_duration_s(
    n_beats: int,
    frames: int | None = None,
    fps: float | None = None,
    preset: str | None = None,
) -> float:
    p = get_preset(preset)
    f = p["frames"] if frames is None else frames
    r = p["fps"] if fps is None else fps
    return n_beats * (f / r)


def estimate_wall_clock_s(
    n_beats: int,
    frames: int | None = None,
    preset: str | None = None,
    device_mode: str | None = None,
) -> dict:
    """
    Rough ETA on 16GB with module CPU offload (reliable default).
    Frame count dominates time more than resolution.
    """
    p = get_preset(preset)
    f = p["frames"] if frames is None else int(frames)
    mode = (device_mode or "offload").lower()
    # ~8–10 min mid @ 49f module offload after warm; scales ~linear with frames
    base = 9.0 * 60.0 if mode != "full" else 12.0 * 60.0
    if mode == "staged":
        base = 7.0 * 60.0
    per_mid = max(90.0, (f / 49.0) * base)
    per_lo = per_mid * 0.55
    per_hi = per_mid * 1.6
    return {
        "per_clip_s": round(per_mid),
        "total_lo_s": round(n_beats * per_lo),
        "total_mid_s": round(n_beats * per_mid),
        "total_hi_s": round(n_beats * per_hi),
        "device_mode": mode,
        "note": f"{mode} ETA on 16GB — use Fast speed mode for shorter waits",
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# Frame / media helpers
# ---------------------------------------------------------------------------

def numpy_video_to_last_pil(video_np: np.ndarray) -> Image.Image:
    """video_np: [F,H,W,C] float 0–1 or uint8."""
    frame = video_np[-1]
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(frame).convert("RGB")


def save_pil(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return path


def load_ref_image(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def extract_last_frame_from_video(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    """Prefer PyAV; fall back to ffmpeg still."""
    try:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            last = None
            for frame in container.decode(stream):
                last = frame
            if last is None:
                raise RuntimeError("no frames")
            img = last.to_image().convert("RGB")
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(f"cannot read video ref: {path}")
        tmp = path.with_suffix(".last.png")
        subprocess.run(
            [ffmpeg, "-y", "-sseof", "-0.1", "-i", str(path), "-frames:v", "1", str(tmp)],
            capture_output=True,
            check=False,
        )
        if not tmp.is_file():
            raise RuntimeError(f"ffmpeg failed to extract frame from {path}")
        img = Image.open(tmp).convert("RGB")
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    if size is not None:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def resolve_start_image(
    ref_paths: list[str] | None,
    width: int,
    height: int,
) -> Image.Image | None:
    if not ref_paths:
        return None
    size = (width, height)
    for raw in ref_paths:
        p = Path(raw)
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
            return extract_last_frame_from_video(p, size=size)
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return load_ref_image(p, size=size)
    return None


# ---------------------------------------------------------------------------
# Concat
# ---------------------------------------------------------------------------

def concat_videos(paths: list[Path], out: Path) -> Path:
    if not paths:
        raise RuntimeError("no segments to concat")
    if len(paths) == 1:
        shutil.copy2(paths[0], out)
        return out

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH")

    list_file = out.with_suffix(".txt")
    lines = []
    for p in paths:
        ap = p.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{ap}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", str(out),
        ]
        r2 = subprocess.run(cmd, capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed:\n{r.stderr[-600:]}\n{r2.stderr[-600:]}")
    try:
        list_file.unlink(missing_ok=True)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class LTXDream:
    def __init__(self) -> None:
        self.pipe = None
        self.device_mode = "unloaded"
        self.pipe_kind = "none"  # condition | t2v
        self.load_s: float | None = None
        self.last_infer_s: float | None = None

    def _instantiate_pipe(self, src: str, local_only: bool):
        try:
            from diffusers import LTX2ConditionPipeline

            pipe = LTX2ConditionPipeline.from_pretrained(
                src, torch_dtype=torch.bfloat16, local_files_only=local_only
            )
            kind = "condition"
            print("[LTXDream] LTX2ConditionPipeline (last-frame / ref chaining enabled)")
        except Exception as exc:
            print(f"[LTXDream] ConditionPipeline unavailable ({exc}); falling back to T2V")
            from diffusers import LTX2Pipeline

            pipe = LTX2Pipeline.from_pretrained(
                src, torch_dtype=torch.bfloat16, local_files_only=local_only
            )
            kind = "t2v"
        return pipe, kind

    def _drop_pipe(self) -> None:
        if self.pipe is not None:
            try:
                del self.pipe
            except Exception:
                pass
            self.pipe = None
        free_vram()

    def _enable_vae_tiling(self) -> None:
        assert self.pipe is not None
        for name in ("vae", "audio_vae"):
            vae = getattr(self.pipe, name, None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                try:
                    vae.enable_tiling()
                except Exception:
                    pass

    def _apply_module_offload(self) -> None:
        assert self.pipe is not None
        self.pipe.enable_model_cpu_offload()
        self.device_mode = "offload"
        self._enable_vae_tiling()
        print(f"[LTXDream] device_mode=offload · {_vram_stats()}")

    def load(self, on_status=None) -> None:
        if self.pipe is not None:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")

        prog = _Progress(on_status)
        src = _model_path()
        local_only = src != MODEL_ID and "nunchaku" not in src.lower()
        print(f"[LTXDream] loading {src}…")
        prog.set("loading LTX weights from disk (one-time)…", 0.02)
        t0 = time.time()

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        free_vram()

        self.pipe, self.pipe_kind = self._instantiate_pipe(src, local_only)

        if _want_full_gpu():
            prog.set("JUSTDREAM_FULL=1 — placing pipeline on CUDA…", 0.04)
            try:
                self.pipe.to("cuda")
                self.device_mode = "full"
                self._enable_vae_tiling()
                print(f"[LTXDream] device_mode=full · {_vram_stats()}")
            except torch.cuda.OutOfMemoryError:
                print("[LTXDream] full OOM — reloading clean pipe for module offload")
                self._drop_pipe()
                self.pipe, self.pipe_kind = self._instantiate_pipe(src, local_only)
                prog.set("module CPU offload (TE→DiT→VAE)…", 0.04)
                self._apply_module_offload()
        elif _want_group_offload():
            # Opt-in: often OOMs on 16GB during enable — must rebuild pipe on failure.
            prog.set("JUSTDREAM_GROUP=1 — trying group offload…", 0.04)
            try:
                self.pipe.enable_group_offload(
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="leaf_level",
                    use_stream=True,
                    record_stream=True,
                )
                self.device_mode = "staged"
                self._enable_vae_tiling()
                print(f"[LTXDream] device_mode=staged · {_vram_stats()}")
            except Exception as exc:
                print(f"[LTXDream] group offload failed ({exc}); rebuilding for module offload")
                self._drop_pipe()
                self.pipe, self.pipe_kind = self._instantiate_pipe(src, local_only)
                prog.set("module CPU offload (TE→DiT→VAE)…", 0.04)
                self._apply_module_offload()
        else:
            # Default: reliable Diffusers module staging (same family as justimagine staged).
            # Frame count is what dominates time — use UI Fast mode for speed.
            prog.set("module CPU offload (TE→DiT→VAE)…", 0.04)
            self._apply_module_offload()

        self.load_s = time.time() - t0
        print(f"[LTXDream] load took {format_duration(self.load_s)}")
        prog.set(f"ready ({self.device_mode}/{self.pipe_kind}) · load {format_duration(self.load_s)}", 0.05)

    def unload(self) -> None:
        self._drop_pipe()
        self.device_mode = "unloaded"
        self.pipe_kind = "none"
        self.load_s = None
        self.last_infer_s = None

    def generate(
        self,
        prompt: str,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        num_frames: int = DEFAULT_FRAMES,
        frame_rate: float = DEFAULT_FPS,
        seed: int | None = None,
        on_status=None,
        progress_lo: float = 0.0,
        progress_hi: float = 1.0,
        out_path: Path | None = None,
        fallbacks: tuple[int, ...] | None = None,
        condition_image: Image.Image | None = None,
        style_lock: str = DEFAULT_STYLE_LOCK,
        return_last_frame: bool = False,
    ):
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
        from diffusers.utils import encode_video

        prog = _Progress(on_status, progress_lo, progress_hi)
        t0 = time.time()
        self.load(on_status=lambda m, p: prog.set(m, min(0.08, (p or 0) * 0.08)))
        assert self.pipe is not None

        if self.device_mode == "full":
            prog.set(
                "device_mode=full — if multi-minute + Shared GPU mem, restart without JUSTDREAM_FULL",
                0.08,
            )

        width = _snap32(width)
        height = _snap32(height)
        num_frames = _frames_8n1(num_frames)
        steps = len(DISTILLED_SIGMA_VALUES)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))

        ladder = fallbacks or SAFE_FRAMES_FALLBACK
        frames_try = [num_frames] + [f for f in ladder if f < num_frames]
        seen: set[int] = set()
        frames_try = [f for f in frames_try if not (f in seen or seen.add(f))]

        neg = _negative_prompt(style_lock)
        cond_img = None
        if condition_image is not None:
            cond_img = condition_image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)

        last_err: Exception | None = None
        video = audio = None
        used_frames = num_frames
        peak_alloc = 0.0

        for attempt, frames in enumerate(frames_try):
            used_frames = frames
            step_t0 = time.time()

            def _cb(pipe, step, timestep, callback_kwargs, _frames=frames, _t0=step_t0):
                nonlocal peak_alloc
                done = step + 1
                frac = done / steps
                elapsed = time.time() - _t0
                eta = (elapsed / done) * (steps - done)
                if torch.cuda.is_available():
                    peak_alloc = max(peak_alloc, torch.cuda.memory_allocated(0) / (1024 ** 3))
                prog.set(
                    f"diffusion {done}/{steps} ({_frames}f) · ~{eta:.0f}s left · {self.device_mode}",
                    0.1 + 0.75 * frac,
                )
                return callback_kwargs

            kwargs = dict(
                prompt=prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_frames=frames,
                frame_rate=frame_rate,
                num_inference_steps=steps,
                sigmas=DISTILLED_SIGMA_VALUES,
                output_type="np",
                return_dict=False,
                generator=generator,
                callback_on_step_end=_cb,
                **_distill_guidance(),
            )

            try:
                if self.pipe_kind == "condition" and cond_img is not None:
                    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

                    conditions = [LTX2VideoCondition(frames=cond_img, index=0, strength=1.0)]
                    prog.set(f"chained I2V {width}x{height} × {frames}f…", 0.1)
                    video, audio = self.pipe(conditions=conditions, **kwargs)
                else:
                    tag = "T2V" if cond_img is None else "T2V(no-condition-pipe)"
                    prog.set(f"{tag} {width}x{height} × {frames}f…", 0.1)
                    video, audio = self.pipe(**kwargs)
                last_err = None
                self.last_infer_s = time.time() - step_t0
                break
            except torch.cuda.OutOfMemoryError as exc:
                last_err = exc
                free_vram()
                prog.set(f"OOM at {frames}f — retrying smaller…", 0.1)
                continue
            except Exception as exc:
                last_err = exc
                free_vram()
                if attempt + 1 < len(frames_try):
                    prog.set(f"error ({exc}) — retrying…", 0.1)
                    continue
                raise

        if last_err is not None or video is None:
            raise RuntimeError(f"segment failed after retries: {last_err}") from last_err

        prog.set("encoding mp4…", 0.92)
        out = out_path or (OUTPUT_DIR / f"{_stamp()}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        encode_video(
            video[0],
            fps=int(round(frame_rate)),
            audio=audio[0].float().cpu(),
            audio_sample_rate=self.pipe.vocoder.config.output_sampling_rate,
            output_path=str(out),
        )
        last_pil = numpy_video_to_last_pil(video[0])
        wall = time.time() - t0
        bits = [f"{used_frames}f", f"{wall:.0f}s", self.device_mode]
        if peak_alloc:
            bits.append(f"peak {peak_alloc:.1f}GB")
        prog.set(f"clip ready ({', '.join(bits)})", 1.0)
        print(f"[save] {out} · {' · '.join(bits)} · {_vram_stats()}")
        if return_last_frame:
            return out, last_pil
        return out

    def _plan_path(self, story_dir: Path) -> Path:
        return story_dir / "plan.json"

    def _write_plan(self, story_dir: Path, plan: dict) -> None:
        self._plan_path(story_dir).write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def _read_plan(self, story_dir: Path) -> dict:
        return json.loads(self._plan_path(story_dir).read_text(encoding="utf-8"))

    def generate_story(
        self,
        script: str,
        on_status=None,
        story_dir: Path | None = None,
        seed: int | None = None,
        preset: str | None = None,
        style_lock: str | None = None,
        ref_paths: list[str] | None = None,
    ) -> Path:
        preset_cfg = get_preset(preset)
        style = (style_lock or DEFAULT_STYLE_LOCK).strip() or DEFAULT_STYLE_LOCK
        refs = [str(Path(p)) for p in (ref_paths or []) if p]
        script = (script or "").strip()
        base_seed = 42 if seed is None else int(seed)

        if story_dir is None:
            if not script:
                raise ValueError("empty script")
            story_dir = STORIES_DIR / _stamp()
        story_dir.mkdir(parents=True, exist_ok=True)
        memory_dir = story_dir / "memory"
        memory_dir.mkdir(exist_ok=True)

        plan_file = self._plan_path(story_dir)
        if plan_file.is_file():
            plan = self._read_plan(story_dir)
            beats = plan.get("beats") or []
            style = plan.get("style_lock") or style
            refs = plan.get("ref_paths") or refs
            if plan.get("preset"):
                preset_cfg = get_preset(plan["preset"])
            if not beats and script:
                beats = split_script_to_beats(script, preset=preset_cfg, style_lock=style)
                plan["beats"] = beats
            if not beats:
                raise ValueError("resume plan has no beats")
        else:
            if not script:
                raise ValueError("empty script")
            beats = split_script_to_beats(script, preset=preset_cfg, style_lock=style)
            plan = {
                "status": "running",
                "script": script,
                "preset": preset_cfg["id"],
                "preset_label": preset_cfg["label"],
                "style_lock": style,
                "ref_paths": refs,
                "beats": beats,
                "width": preset_cfg["width"],
                "height": preset_cfg["height"],
                "frames": preset_cfg["frames"],
                "fps": preset_cfg["fps"],
                "fallbacks": list(preset_cfg["fallbacks"]),
                "chain": preset_cfg["chain"],
                "seed": base_seed,
                "memory": {
                    "last_frame": None,
                    "last_segment": None,
                    "audio_seed": base_seed,
                },
                "segments": [
                    {"i": i, "prompt": b, "file": f"seg_{i:03d}.mp4", "ok": False, "error": None}
                    for i, b in enumerate(beats)
                ],
                "final": None,
                "created": _stamp(),
            }
            # Copy refs into story folder for permanence
            saved_refs = []
            for i, rp in enumerate(refs):
                src = Path(rp)
                if not src.is_file():
                    continue
                dst = memory_dir / f"ref_{i:02d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
                saved_refs.append(str(dst))
            plan["ref_paths"] = saved_refs
            refs = saved_refs
            self._write_plan(story_dir, plan)

        n = len(beats)
        clip_s = float(plan["frames"]) / float(plan["fps"])
        if on_status:
            on_status(
                f"{plan.get('preset_label')}: {n} chained clips × ~{clip_s:.1f}s "
                f"≈ {n * clip_s / 60:.1f} min · style-locked · last-frame memory",
                0.01,
            )

        self.load(on_status=on_status)
        plan["device_mode"] = self.device_mode
        plan["pipe_kind"] = self.pipe_kind
        self._write_plan(story_dir, plan)

        # Opening condition: remembered last_frame > user refs
        cond: Image.Image | None = None
        mem_last = plan.get("memory", {}).get("last_frame")
        if mem_last and (story_dir / mem_last).is_file():
            cond = load_ref_image(
                story_dir / mem_last,
                size=(int(plan["width"]), int(plan["height"])),
            )
        elif refs:
            cond = resolve_start_image(refs, int(plan["width"]), int(plan["height"]))
            if cond is not None:
                save_pil(cond, memory_dir / "start_ref.png")
                plan["memory"]["start_ref"] = "memory/start_ref.png"
                self._write_plan(story_dir, plan)

        fallbacks = tuple(plan.get("fallbacks") or preset_cfg["fallbacks"])
        chain = bool(plan.get("chain", True))
        done_paths: list[Path] = []

        for i, seg in enumerate(plan["segments"]):
            seg_path = story_dir / seg["file"]
            lo = i / max(n, 1)
            hi = (i + 1) / max(n, 1)

            if seg.get("ok") and seg_path.is_file():
                done_paths.append(seg_path)
                lf = story_dir / "memory" / f"last_{i:03d}.png"
                if lf.is_file():
                    cond = load_ref_image(lf, size=(int(plan["width"]), int(plan["height"])))
                    plan["memory"]["last_frame"] = f"memory/last_{i:03d}.png"
                live_name = f"live_{story_dir.name}_seg_{i:03d}.mp4"
                if not (OUTPUT_DIR / live_name).is_file():
                    try:
                        shutil.copy2(seg_path, OUTPUT_DIR / live_name)
                    except Exception:
                        live_name = None
                if on_status:
                    clips = [
                        {
                            "i": j,
                            "label": f"Clip {j + 1}/{n}",
                            "file": f"live_{story_dir.name}_seg_{j:03d}.mp4",
                        }
                        for j in range(i + 1)
                        if (OUTPUT_DIR / f"live_{story_dir.name}_seg_{j:03d}.mp4").is_file()
                    ]
                    on_status(
                        f"segment {i + 1}/{n} already done — preview ready",
                        hi,
                        video=live_name if live_name and (OUTPUT_DIR / live_name).is_file() else None,
                        clips=clips,
                        is_final=False,
                    )
                continue

            prompt = seg["prompt"]
            if on_status:
                mode = "chained" if (chain and cond is not None) else "opening"
                on_status(f"segment {i + 1}/{n} ({mode}): {prompt[:70]}…", lo + 0.01 * (hi - lo))

            try:
                # Same base seed + index keeps audio/video RNG related across clips
                path, last_pil = self.generate(
                    prompt,
                    width=int(plan["width"]),
                    height=int(plan["height"]),
                    num_frames=int(plan["frames"]),
                    frame_rate=float(plan["fps"]),
                    seed=base_seed + i,
                    on_status=on_status,
                    progress_lo=lo,
                    progress_hi=hi,
                    out_path=seg_path,
                    fallbacks=fallbacks,
                    condition_image=cond if chain else None,
                    style_lock=style,
                    return_last_frame=True,
                )
                lf_rel = f"memory/last_{i:03d}.png"
                save_pil(last_pil, story_dir / lf_rel)
                plan["memory"]["last_frame"] = lf_rel
                plan["memory"]["last_segment"] = i
                plan["memory"]["audio_seed"] = base_seed + i
                if chain:
                    cond = last_pil  # remember for next clip
                seg["ok"] = True
                seg["error"] = None
                seg["file"] = path.name
                done_paths.append(path)

                # Live preview: copy into /outputs so the UI can play each clip as it finishes
                live_name = f"live_{story_dir.name}_seg_{i:03d}.mp4"
                try:
                    shutil.copy2(path, OUTPUT_DIR / live_name)
                except Exception:
                    live_name = None
                if on_status is not None:
                    clips = [
                        {
                            "i": j,
                            "label": f"Clip {j + 1}/{n}",
                            "file": f"live_{story_dir.name}_seg_{j:03d}.mp4",
                        }
                        for j in range(i + 1)
                        if (OUTPUT_DIR / f"live_{story_dir.name}_seg_{j:03d}.mp4").is_file()
                        or j == i
                    ]
                    if live_name:
                        clips[-1]["file"] = live_name
                    on_status(
                        f"clip {i + 1}/{n} ready — playing preview",
                        hi,
                        video=live_name,
                        clips=clips,
                        is_final=False,
                    )
            except Exception as exc:
                seg["ok"] = False
                seg["error"] = str(exc)
                print(f"[story] segment {i} failed: {exc}")
                # Keep previous cond so a failed middle clip doesn't reset the chain totally
            self._write_plan(story_dir, plan)
            free_vram()

        if not done_paths:
            plan["status"] = "failed"
            self._write_plan(story_dir, plan)
            raise RuntimeError("all segments failed — see plan.json")

        if on_status:
            on_status(f"stitching {len(done_paths)}/{n} segments…", 0.96)

        final = story_dir / "final.mp4"
        concat_videos(done_paths, final)
        root_copy = OUTPUT_DIR / f"story_{story_dir.name}.mp4"
        shutil.copy2(final, root_copy)

        plan["status"] = "done" if len(done_paths) == n else "partial"
        plan["final"] = final.name
        plan["root_copy"] = root_copy.name
        plan["completed_segments"] = len(done_paths)
        self._write_plan(story_dir, plan)

        if on_status:
            clips = [
                {
                    "i": j,
                    "label": f"Clip {j + 1}/{n}",
                    "file": f"live_{story_dir.name}_seg_{j:03d}.mp4",
                }
                for j in range(n)
                if (OUTPUT_DIR / f"live_{story_dir.name}_seg_{j:03d}.mp4").is_file()
            ]
            clips.append({"i": n, "label": "Final film", "file": root_copy.name, "final": True})
            on_status(
                f"done ({plan['status']}): {len(done_paths)}/{n} chained clips → {root_copy.name}",
                1.0,
                video=root_copy.name,
                clips=clips,
                is_final=True,
            )
        return root_copy


def find_resumable_story() -> Path | None:
    if not STORIES_DIR.is_dir():
        return None
    for d in sorted(STORIES_DIR.iterdir(), reverse=True):
        plan_p = d / "plan.json"
        if not plan_p.is_file():
            continue
        try:
            plan = json.loads(plan_p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if plan.get("status") in ("running", "partial"):
            segs = plan.get("segments") or []
            if any(not s.get("ok") for s in segs):
                return d
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="justdream continuity story")
    parser.add_argument("script", nargs="?", help="script text or path to .txt")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    parser.add_argument("--style", type=str, default=DEFAULT_STYLE_LOCK)
    parser.add_argument("--ref", action="append", default=[], help="reference image/video (repeatable)")
    args = parser.parse_args()

    text = ""
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.script and Path(args.script).is_file():
        text = Path(args.script).read_text(encoding="utf-8")
    elif args.script:
        text = args.script
    else:
        raise SystemExit("Pass script text or --file path")

    def _print(msg: str, progress: float | None = None, **_extra) -> None:
        print(f"[{100 * (progress or 0):5.1f}%] {msg}", flush=True)

    engine = LTXDream()
    print(
        engine.generate_story(
            text,
            on_status=_print,
            preset=args.preset,
            style_lock=args.style,
            ref_paths=args.ref,
        )
    )


if __name__ == "__main__":
    main()
