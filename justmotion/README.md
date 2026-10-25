# justmotion

**Prompt → keyframes (shown live) → fill between → DLSS SR + Neural Rendering → video.**

Same chat UI language as [`justimagine`](../justimagine/). Uses justimagine’s FLUX.2 klein for images, ffmpeg minterpolate to fill frames between keyframes, then [`exp_dlss`](../../FULLONVIDEOEDITING/the-exp-dlss) for DLSS Super Resolution + Neural Rendering.

**Not** NVIDIA DLSS Frame Generation (that needs a game-engine Streamline loop with depth + motion vectors). Between-frame fill is **ffmpeg minterpolate**.

UI: http://127.0.0.1:7863

## What you can give

| You provide | What happens |
|-------------|--------------|
| **Prompt only** | Generate N keyframes with FLUX (chained for continuity), show each as it finishes, interpolate, DLSS upscale |
| **Prompt + image** | Your image is keyframe 1; generate the rest; then same |
| **Video** (± prompt) | Sample N keyframes from the video, show them, interpolate denser, DLSS upscale |
| **Image only** | Treated as first keyframe; remaining keyframes need a prompt (continuity edit) |

## Run

Use **justimagine’s venv** (torch + diffusers already there):

```powershell
cd D:\Workspace\Devoids\OpenReality\justimagine
.\venv\Scripts\Activate.ps1
cd ..\justmotion
$env:EXP_DLSS_ROOT = "D:\Workspace\Devoids\FULLONVIDEOEDITING\the-exp-dlss"
python -m pip install fastapi uvicorn python-multipart   # if missing
python webui.py
```

Requires:

- justimagine set up + FLUX weights already downloaded
- `the-exp-dlss` with `vendor/video2dlssnr` (DaniilSokolyuk release via its `fetch_vendor.ps1`)

## Stages in the UI

1. **keyframes** — gallery fills as each PNG is ready  
2. **interpolate** — slideshow → minterpolate to target fps  
3. **DLSS / NR** — `video2dlssnr` SR + Neural Rendering (or plain scale with “skip NR”)

## Notes

- Keep gen size ~768 and keyframes ~4–8 on 16GB.
- NR is tuned for game footage; on AI faces it may look odd — try skip NR to compare.
- Set `JUSTIMAGINE_ROOT` / `EXP_DLSS_ROOT` if your paths differ.
