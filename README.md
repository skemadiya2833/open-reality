# OpenReality

Local, open-source **image → 3D**, **image generation / editing**, and **video generation** — without ComfyUI, without Ollama, without cloud APIs.

Three small apps share the same idea: a Windows-friendly setup script, a Python pipeline, and a lightweight FastAPI chat UI. Models download from Hugging Face with **strict allow/ignore lists** so you do not pull duplicate multi-gigabyte junk.

| Folder | What it does | Default UI | Tuned for |
|--------|--------------|------------|-----------|
| [`justthink/`](justthink/) | Text or image → textured **3D mesh (GLB)** | http://127.0.0.1:7860 | RTX 5060 Ti **16GB** Blackwell |
| [`justimagine/`](justimagine/) | Text → image, and **contextual image editing** | http://127.0.0.1:7861 | Same 16GB card, full GPU residency |
| [`justdream/`](justdream/) | Text → **video + audio** (short clips) | http://127.0.0.1:7862 | Same card via **CPU offload**; **bnb-nf4** (Nunchaku is Linux-only) |

```
OpenReality/
├── README.md          ← you are here
├── .gitignore
├── justthink/         ← 3D (Hunyuan3D-2.0)
├── justimagine/       ← images (FLUX.2 klein 4B)
└── justdream/         ← video (LTX-2.3 Distilled bnb-nf4, Windows)
```

## Why this exists

Most “run Flux / LTX / Hunyuan locally” guides point you at **ComfyUI** graphs or **Ollama**-style wrappers. That works, but it is heavy, opaque, and easy to download the wrong 40GB of duplicates.

OpenReality is the opposite:

- **No ComfyUI** — plain Diffusers / official Tencent code + a tiny HTML UI.
- **No Ollama** — direct CUDA PyTorch inference.
- **Hardware-honest defaults** for a **16GB RTX 5060 Ti (Blackwell, sm_120, cu128)**.
- **Minimal downloads** — each app’s `download_models.py` (or Hunyuan smart-load patches) skips known duplicate weight files.

You can still swap models (see each folder README). The defaults are a **sweet spot**, not a claim that they are the only good choice.

## Hardware assumptions

Primary target:

- **GPU:** NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell)
- **Driver / PyTorch:** CUDA **12.8** wheels (`cu128`) — older `cu124` builds often fail to see Blackwell.
- **Python:** **3.10–3.12** (3.14 breaks many native wheels / Open3D).

If you have more VRAM (24GB+ / 32GB+), raise resolution, steps, or frame counts and prefer full GPU residency where documented.

## Quick start

Each project is self-contained (own `venv`). From a PowerShell prompt:

```powershell
# 3D
cd justthink
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\venv\Scripts\Activate.ps1
python webui.py
# → http://127.0.0.1:7860

# Images
cd ..\justimagine
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\venv\Scripts\Activate.ps1
python webui.py
# → http://127.0.0.1:7861

# Video
cd ..\justdream
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\venv\Scripts\Activate.ps1
. .\env.ps1   # DIFFUSERS_TRUST_REMOTE_KERNELS=true
python webui.py
# → http://127.0.0.1:7862
```

Model weights land in the Hugging Face hub cache (`~/.cache/huggingface/...`), not in this git repo. **Do not commit `venv/`, `outputs/`, or weight files** — see `.gitignore`.

## Design notes shared by all three

1. **Separate process for generation** so `/api/status` stays responsive while CUDA work runs (GIL / long loads do not freeze the UI).
2. **Chat-style UI** — keep typing / queue prompts; see live status and results.
3. **Ports differ** so you can run all three at once: `7860` / `7861` / `7862`.
4. **Licenses follow upstream models** (Apache 2.0 for FLUX.2 klein 4B; LTX and Hunyuan have their own terms — read each model card before commercial use).

## Choosing other models

| If you want… | Consider… | Trade-off |
|--------------|-----------|-----------|
| Faster / smaller 3D | Hunyuan turbo or lower octree | Quality drop |
| Higher quality images | FLUX.2 larger variants | More VRAM / slower |
| Longer / sharper video | Full LTX bf16 or higher frames | Needs more VRAM or heavy offload |
| ComfyUI workflows | Official Comfy packs | Not this repo’s path |

Change the model ID / subfolder in each app’s `app.py` (and download patterns) if you know the VRAM math.

## Status

Built and validated on Windows 11 + RTX 5060 Ti 16GB. Linux users can mirror the same pip/torch steps; native Windows bits (VS Build Tools for Hunyuan extensions) are documented under `justthink/`.

## License

Application glue in this repository is provided for local experimentation. **Model weights and upstream SDKs keep their original licenses** — check Hugging Face / Tencent / Lightricks / Black Forest Labs pages before redistributing weights or shipping a product.
