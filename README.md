# OpenReality

Note: Optimizations are on the way.....

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

Each project keeps its own `venv`, but **justimagine** and **justthink** share one cu128 PyTorch copy via junctions under `.shared/torch-cu128/` (~4.1 GB once, not twice). `setup.ps1` / `link_shared_torch.ps1` wire that automatically. justdream is separate. From a PowerShell prompt:

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


#Results:

Just Think (3d): (Takes ~2 mins)
Input the image from Output of Image gen 
Output:
<img width="433" height="289" alt="image" src="https://github.com/user-attachments/assets/f8c8fa8b-cb69-40d5-b2b0-8acac0aac76a" />
The details are quite good after 3d printing (other example):
<img width="816" height="640" alt="image" src="https://github.com/user-attachments/assets/18d7cd54-45ee-446a-856c-c3f44c48704d" />
<img width="530" height="533" alt="image" src="https://github.com/user-attachments/assets/79a06fe1-87b3-4b5d-8f69-a17afc63faf9" />

Just Imagine ( Image gen): (literally fastest takes only 10-25 seconds)

Prompt: 
```
A young woman wearing Large round transparent-framed glasses with gentle, pleasant appearance. she has long, straight dark hair, parted near the middle, dark eyes and defined eyebrows, a subtle, closed-mouth smile, a soft, relaxed facial expression, white t-shirt and jeans, The overall impression should be calm and friendly.
(We could attach image as well)
```
Resource usage:
<img width="1280" height="768" alt="image" src="https://github.com/user-attachments/assets/b4c0a7aa-1583-4fd1-8852-7294f2195c0f" />
Output: 
<img width="1024" height="1024" alt="20260909_113102" src="https://github.com/user-attachments/assets/5f8d02c9-5572-4dbf-9de5-dca5b58735b0" />

Just Dream (Video Gen): (Needs optimization) 

Prompt:
```
A young woman with gentle, pleasant appearance. she has long, straight dark hair, parted near the middle, Large round, transparent-framed glasses, dark eyes and defined eyebrows, a subtle, closed-mouth smile, a soft, relaxed facial expression, white t-shirt and jeans, The overall impression from video should be calm and friendly. 
```
Output: (Offloading required)

Preview: https://github.com/user-attachments/assets/6ab0c6bb-d77d-42b0-9d2a-c77ecda6691d


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
