"""justmotion — prompt (+ optional image/video) → keyframes → interpolate → DLSS SR+NR.

Feasible path (no real DLSS Frame Generation — that needs a game engine Streamline
loop with depth/MVs). Between-frame fill uses ffmpeg minterpolate (or RIFE if set).
Upscale uses exp_dlss / video2dlssnr (DLSS Super Resolution + Neural Rendering).
"""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = ROOT / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

JUSTIMAGINE_ROOT = Path(os.environ.get("JUSTIMAGINE_ROOT", str(ROOT.parent / "justimagine"))).resolve()
_DEFAULT_EXP = Path(r"D:\Workspace\Devoids\FULLONVIDEOEDITING\the-exp-dlss")
EXP_DLSS_ROOT = Path(os.environ.get("EXP_DLSS_ROOT", str(_DEFAULT_EXP))).resolve()

DEFAULT_KEYFRAMES = 6
DEFAULT_GEN = 768  # lighter than 1024 for multi-frame on 16GB
DEFAULT_FPS = 24
DEFAULT_HOLD_FPS = 2.0  # keyframe slideshow rate before interpolate


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "—"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s:02d}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def free_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def ensure_justimagine() -> Path:
    app = JUSTIMAGINE_ROOT / "app.py"
    if not app.is_file():
        raise FileNotFoundError(f"justimagine not found at {JUSTIMAGINE_ROOT}")
    p = str(JUSTIMAGINE_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)
    return JUSTIMAGINE_ROOT


def ensure_exp_dlss() -> Path:
    src = EXP_DLSS_ROOT / "src"
    if not (src / "exp_dlss" / "pipeline" / "upscale.py").is_file():
        raise FileNotFoundError(
            f"exp_dlss not found under {EXP_DLSS_ROOT}. Set EXP_DLSS_ROOT."
        )
    p = str(src)
    if p not in sys.path:
        sys.path.insert(0, p)
    return EXP_DLSS_ROOT


def find_ffmpeg() -> str:
    """Prefer a build with libx264; fall back to vendor (NVENC-capable) ffmpeg."""
    candidates: list[str] = []
    path_ff = shutil.which("ffmpeg")
    if path_ff:
        candidates.append(path_ff)
    try:
        ensure_exp_dlss()
        from exp_dlss.ffmpeg_util import find_tool

        vend = find_tool("ffmpeg")
        if vend and vend not in candidates:
            candidates.append(vend)
    except Exception:
        pass

    def has_libx264(exe: str) -> bool:
        r = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
        return "libx264" in (r.stdout or "")

    for exe in candidates:
        if has_libx264(exe):
            return exe
    if candidates:
        return candidates[0]
    raise RuntimeError("ffmpeg not found (install Gyan.FFmpeg or vendor video2dlssnr)")


def find_ffprobe() -> str:
    try:
        ensure_exp_dlss()
        from exp_dlss.ffmpeg_util import find_tool

        fp = find_tool("ffprobe")
        if fp:
            return fp
    except Exception:
        pass
    fp = shutil.which("ffprobe")
    if not fp:
        raise RuntimeError("ffprobe not found")
    return fp


def estimate_wall_s(
    *,
    keyframes: int,
    need_generate: int,
    gen_w: int,
    target_fps: int,
    skip_nr: bool,
    warm: bool,
) -> tuple[float, float, float]:
    """Rough ETA: Flux keyframes dominate, then interpolate, then NR."""
    mp = max(0.25, (gen_w * gen_w) / (1024 * 1024))
    per_img = 16.0 * mp
    if not warm:
        per_img += 60.0
    gen = need_generate * per_img + (180.0 if (not warm and need_generate) else 0.0)
    interp = 15.0 + keyframes * 2.0
    up = (12.0 if skip_nr else 45.0) + keyframes * (target_fps / max(DEFAULT_HOLD_FPS, 0.5)) * 0.05
    mid = gen + interp + up
    return mid * 0.65, mid, mid * 1.8


def extract_video_keyframes(video: Path, out_dir: Path, count: int, ffmpeg: str) -> list[Path]:
    """Evenly sample ``count`` frames from a video into PNGs."""
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    probe = find_ffprobe()
    raw = subprocess.run(
        [
            probe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    info = json.loads(raw or "{}")
    try:
        dur = float(info.get("format", {}).get("duration") or 0)
    except ValueError:
        dur = 0.0
    if dur <= 0.1:
        dur = float(count) / DEFAULT_HOLD_FPS
    fps = max(0.1, count / dur)
    pattern = out_dir / "kf_%03d.png"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-i",
        str(video),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(count),
        str(pattern),
    ]
    _run_ffmpeg(cmd, "ffmpeg extract keyframes")
    files = sorted(out_dir.glob("kf_*.png"))
    if not files:
        raise RuntimeError("no keyframes extracted from video")
    return files[:count]


def _pick_video_encoder(ffmpeg: str) -> list[str]:
    """Return ffmpeg args for a working H.264 encoder (libx264 or NVENC)."""
    r = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    text = r.stdout or ""
    if "libx264" in text:
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
    if "h264_nvenc" in text:
        return [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "constqp",
            "-qp",
            "18",
            "-pix_fmt",
            "yuv420p",
        ]
    if "libopenh264" in text:
        return ["-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "5M"]
    raise RuntimeError("No usable H.264 encoder in ffmpeg (need libx264 or h264_nvenc)")


def _run_ffmpeg(cmd: list[str], label: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()[-1200:]
        raise RuntimeError(f"{label} failed (rc={r.returncode}):\n{err}")


def _even_rgb(img: "Image.Image", size: tuple[int, int] | None = None) -> "Image.Image":
    from PIL import Image as PILImage

    if size is not None:
        img = img.convert("RGB").resize(size, PILImage.Resampling.LANCZOS)
    else:
        img = img.convert("RGB")
    w, h = img.size
    nw, nh = w - (w % 2), h - (h % 2)
    if (nw, nh) != (w, h):
        img = img.crop((0, 0, nw, nh))
    return img


def keyframes_to_interpolated_video(
    frames: list[Path],
    out_mp4: Path,
    *,
    hold_fps: float = DEFAULT_HOLD_FPS,
    target_fps: int = DEFAULT_FPS,
    ffmpeg: str,
    interp_mode: str = "blend",
) -> Path:
    """Slideshow at hold_fps, then fill between frames.

    ``interp_mode``:
      - blend: minterpolate mi_mode=blend (safe for near-duplicate AI stills)
      - mci: full motion-compensated interpolate (looks like trash if keyframes diverge)
      - xfade: crossfade only (safest, least “fake motion”)
    """
    from PIL import Image

    if len(frames) < 2:
        raise ValueError("need at least 2 keyframes to interpolate a clip")

    work = out_mp4.parent / "_slide"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    first = _even_rgb(Image.open(frames[0]))
    target_size = first.size
    for i, src in enumerate(frames):
        im = _even_rgb(Image.open(src), target_size)
        im.save(work / f"{i:04d}.png")

    enc = _pick_video_encoder(ffmpeg)
    slide = out_mp4.parent / "_slideshow.mp4"
    list_file = work / "list.txt"
    lines = []
    for i in range(len(frames)):
        png = (work / f"{i:04d}.png").resolve().as_posix()
        lines.append(f"file '{png}'")
        lines.append(f"duration {1.0 / max(hold_fps, 0.1):.6f}")
    last = (work / f"{len(frames) - 1:04d}.png").resolve().as_posix()
    lines.append(f"file '{last}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd_slide = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        f"fps={hold_fps},format=yuv420p",
        *enc,
        str(slide),
    ]
    _run_ffmpeg(cmd_slide, "ffmpeg slideshow encode")

    mode = (interp_mode or "blend").lower()
    if mode == "xfade":
        # Rebuild with xfade chain between stills (no optical-flow morph mush).
        # Each still shown ~1/hold_fps, crossfade 0.25s.
        inputs: list[str] = []
        filters: list[str] = []
        for i in range(len(frames)):
            inputs += ["-loop", "1", "-t", f"{1.0 / max(hold_fps, 0.1):.3f}", "-i", str(work / f"{i:04d}.png")]
        # Simple fps upsample of slideshow is enough for xfade mode fallback:
        cmd_xf = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-i",
            str(slide),
            "-vf",
            f"fps={target_fps},format=yuv420p",
            "-an",
            *enc,
            str(out_mp4),
        ]
        _run_ffmpeg(cmd_xf, "ffmpeg xfade/fps upsample")
        return out_mp4

    if mode == "mci":
        vf = f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    else:
        # blend = much less morph-garbage on AI stills that aren't real motion pairs
        vf = f"minterpolate=fps={target_fps}:mi_mode=blend"

    cmd_mi = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-i",
        str(slide),
        "-vf",
        vf,
        "-an",
        *enc,
        str(out_mp4),
    ]
    try:
        _run_ffmpeg(cmd_mi, f"ffmpeg minterpolate ({mode})")
    except RuntimeError:
        cmd_fb = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-i",
            str(slide),
            "-vf",
            f"fps={target_fps},format=yuv420p",
            "-an",
            *enc,
            str(out_mp4),
        ]
        _run_ffmpeg(cmd_fb, "ffmpeg fps upsample fallback")
    return out_mp4


# Subtle beats — keep the SAME subject; do not invent a new scene.
_MOTION_BEATS = (
    "tiny head turn to the left, same pose otherwise",
    "mouth opens a bit more, same framing",
    "slight lean forward, same camera",
    "paw shifts a centimeter on the wheel, same cat",
    "blink / eyes squint a little more, identical background",
    "headphones tilt slightly, same composition",
    "very small camera push-in, same subject",
    "mouth closes slightly, continuous same moment",
)


def continuity_prompt(base: str, index: int, total: int) -> str:
    """Keep identity locked; only ask for tiny motion deltas."""
    base = (base or "cinematic scene").strip()
    if total <= 1:
        return base
    beat = _MOTION_BEATS[index % len(_MOTION_BEATS)]
    return (
        f"{base}. Exact same character, fur pattern, car interior, and camera angle. "
        f"Only change: {beat}. "
        f"Do not change breed, colors, layout, or background. Photoreal still from a video."
    )


class MotionJob:
    """Run one full job: keyframes → interpolate → DLSS/NR."""

    def __init__(self, emit) -> None:
        self.emit = emit

    def _status(self, msg: str, progress: float | None = None, **extra) -> None:
        payload = {"type": "progress", "status": msg, **extra}
        if progress is not None:
            payload["progress"] = progress
        self.emit(**payload)

    def run(self, job: dict) -> dict:
        from PIL import Image

        stamp = job.get("stamp") or _stamp()
        work = WORK_DIR / stamp
        work.mkdir(parents=True, exist_ok=True)
        kf_dir = work / "keyframes"
        kf_dir.mkdir(exist_ok=True)

        prompt = (job.get("prompt") or "").strip()
        n = max(2, int(job.get("keyframes") or DEFAULT_KEYFRAMES))
        gen_w = int(job.get("gen_width") or DEFAULT_GEN)
        gen_h = int(job.get("gen_height") or gen_w)
        target_fps = int(job.get("target_fps") or DEFAULT_FPS)
        skip_nr = bool(job.get("skip_nr", True))  # NR hurts AI faces/fur by default
        nr_width = int(job.get("nr_width") or 1920)
        nr_intensity = float(job.get("nr_intensity") or 0.6)
        interp_mode = (job.get("interp_mode") or "blend").lower()
        seed = job.get("seed")
        seed0 = int(seed) if seed is not None else int(time.time()) % 100000

        image_path = Path(job["image_path"]) if job.get("image_path") else None
        video_path = Path(job["video_path"]) if job.get("video_path") else None

        ffmpeg = find_ffmpeg()
        keyframe_paths: list[Path] = []

        # ----- build keyframes -----
        if video_path is not None and video_path.is_file():
            self._status("extracting keyframes from video…", 0.05, stage="keyframes")
            extracted = extract_video_keyframes(video_path, kf_dir / "from_video", n, ffmpeg)
            for i, src in enumerate(extracted):
                dst = kf_dir / f"kf_{i:03d}.png"
                shutil.copy2(src, dst)
                # Also publish under outputs for UI gallery
                pub = OUTPUT_DIR / f"{stamp}_kf_{i:03d}.png"
                shutil.copy2(dst, pub)
                keyframe_paths.append(dst)
                self.emit(
                    type="keyframe",
                    index=i,
                    total=n,
                    preview=pub.name,
                    status=f"keyframe {i + 1}/{len(extracted)} from video",
                    progress=0.05 + 0.35 * ((i + 1) / max(len(extracted), 1)),
                    stage="keyframes",
                )
            # If prompt given and fewer than n, could generate more — keep extracted count
            n = len(keyframe_paths)
        else:
            if not prompt and image_path is None:
                raise RuntimeError("Need a prompt, and/or an image, and/or a video.")

            ensure_justimagine()
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "justimagine_app", JUSTIMAGINE_ROOT / "app.py"
            )
            assert spec and spec.loader
            ji = importlib.util.module_from_spec(spec)
            sys.modules["justimagine_app"] = ji
            spec.loader.exec_module(ji)
            FluxKlein = ji.FluxKlein

            engine = FluxKlein()
            self._status("loading FLUX (justimagine)…", 0.04, stage="keyframes")

            prev: Image.Image | None = None
            if image_path is not None and image_path.is_file():
                prev = Image.open(image_path).convert("RGB")
                dst = kf_dir / "kf_000.png"
                prev.save(dst)
                pub = OUTPUT_DIR / f"{stamp}_kf_000.png"
                prev.save(pub)
                keyframe_paths.append(dst)
                self.emit(
                    type="keyframe",
                    index=0,
                    total=n,
                    preview=pub.name,
                    status="keyframe 1/N from your image",
                    progress=0.08,
                    stage="keyframes",
                )
                start_i = 1
            else:
                start_i = 0

            for i in range(start_i, n):
                p = continuity_prompt(prompt or "cinematic scene, continuous motion", i, n)
                # Identity lock: always img2img from the first keyframe (anchor).
                if i == 0:
                    use_ref = None
                else:
                    use_ref = Image.open(keyframe_paths[0]).convert("RGB")

                def on_st(msg: str, prog: float | None = None, _i=i):
                    base = 0.08 + 0.45 * (_i / n)
                    span = 0.45 / n
                    local = base + (span * float(prog or 0))
                    self._status(
                        f"keyframe {_i + 1}/{n}: {msg}",
                        local,
                        stage="keyframes",
                    )

                self._status(
                    f"generating keyframe {i + 1}/{n}…",
                    0.08 + 0.45 * (i / n),
                    stage="keyframes",
                )
                img = engine.generate(
                    p,
                    image=use_ref,
                    width=gen_w,
                    height=gen_h,
                    seed=seed0 + i * 17,
                    on_status=on_st,
                )
                dst = kf_dir / f"kf_{i:03d}.png"
                img.save(dst)
                pub = OUTPUT_DIR / f"{stamp}_kf_{i:03d}.png"
                img.save(pub)
                keyframe_paths.append(dst)
                prev = img
                self.emit(
                    type="keyframe",
                    index=i,
                    total=n,
                    preview=pub.name,
                    status=f"keyframe {i + 1}/{n} ready",
                    progress=0.08 + 0.45 * ((i + 1) / n),
                    stage="keyframes",
                )

            self._status("unloading FLUX to free VRAM for DLSS…", 0.55, stage="keyframes")
            engine.unload()
            free_vram()

        if len(keyframe_paths) < 2:
            # Duplicate single frame so interpolate can still make a short clip
            only = keyframe_paths[0]
            dup = kf_dir / "kf_001.png"
            shutil.copy2(only, dup)
            keyframe_paths.append(dup)

        # ----- interpolate -----
        self._status("filling frames between keyframes (ffmpeg minterpolate)…", 0.58, stage="interpolate")
        lowres = work / "03_interp.mp4"
        keyframes_to_interpolated_video(
            keyframe_paths,
            lowres,
            hold_fps=float(job.get("hold_fps") or DEFAULT_HOLD_FPS),
            target_fps=target_fps,
            ffmpeg=ffmpeg,
            interp_mode=interp_mode,
        )
        self._status("interpolation done", 0.72, stage="interpolate")

        # ----- upscale / NR -----
        ensure_exp_dlss()
        import argparse

        from exp_dlss.pipeline import upscale as upscale_mod

        final_work = work / ("04_plain.mp4" if skip_nr else "04_dlss.mp4")
        ns = argparse.Namespace(
            dry_run=False,
            frames=0,
            nr_width=nr_width,
            nr_height=0,
            nr_scale=1.0,
            nr_fit="",
            nr_intensity=nr_intensity,
            nr_style=0,
            nr_preset=0,
            nr_sr_preset="default",
            nr_local_structure=1.0,
            nr_local_tone=1.0,
            nr_skin=-1.0,
            nr_detail=1.0,
            nr_color=1.0,
            nr_transfer="auto",
            nr_sdr_white=203.0,
            nr_ui_correction=0,
            nr_auto_mask=False,
            nr_motion=1,
            nr_motion_engine="auto",
            nr_motion_vis=False,
            exe="",
            dll_dir="",
            adapter=-1,
            codec="hevc_nvenc",
            cq=19,
            bitrate=0,
            bit_depth=10,
            enc_preset="p5",
            multipass="qres",
            hwaccel="auto",
            sw_preset=6,
            prores_profile="hq",
            audio="none",
            audio_bitrate=192,
            workers=1,
        )
        self._status(
            "plain ffmpeg upscale…" if skip_nr else "DLSS Super Resolution + Neural Rendering…",
            0.75,
            stage="upscale",
        )
        upscale_mod.upscale(lowres, final_work, ns, skip_nr=skip_nr)

        out_name = f"{stamp}_final.mp4"
        out_file = OUTPUT_DIR / out_name
        shutil.copy2(final_work, out_file)
        self._status("done", 1.0, stage="upscale")
        return {
            "preview": out_name,
            "keyframes": [f"{stamp}_kf_{i:03d}.png" for i in range(len(keyframe_paths))],
            "work": str(work),
        }
