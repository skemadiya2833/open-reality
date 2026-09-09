# justthink

**Text or image → 3D mesh (GLB)** using official **Tencent Hunyuan3D-2.0**, fully on GPU.

No ComfyUI. No mmgp / deepbeepmeep `-2GP` fork. No `profile=` / `low_vram_mode=` CPU offload stacks.

UI: http://127.0.0.1:7860 after `python webui.py`.

## What it does

1. Optional **text → image** (SDXL-Turbo) if you only give a prompt.
2. **Shape** generation (`hunyuan3d-dit-v2-0`).
3. Optional **texture / paint** (`hunyuan3d-paint-v2-0` + delight).
4. Export **GLB** under `outputs/`, preview in the browser (`model-viewer`).

Modes in the UI: `full` (shape+texture), `shape`, `texture` (reuses last mesh).

## Why these defaults (16GB RTX 5060 Ti)

Official figures put shape around ~6GB and shape+paint near ~16GB combined. On a **16GB** card that is tight if both stages stay resident.

So we intentionally:

| Choice | Setting | Why |
|--------|---------|-----|
| Checkpoint | `tencent/Hunyuan3D-2` **v2-0** (not turbo, not 2.1) | Quality sweet spot |
| Shape steps | **30** (not 50) | Faster, still solid |
| Octree | **256** (not 384) | Comfortable VRAM / time |
| Stage memory | **Unload** paint before shape, unload shape before paint | Peak headroom without mmgp |
| Torch | **cu128** | Blackwell needs it |
| Python | **3.10–3.12** | Open3D / native wheels |

You *can* raise steps/octree on a 24GB+ card. You *can* switch to turbo subfolders for speed. Edit constants in [`app.py`](app.py).

## What we fixed along the way (so you do not repeat it)

- **Do not download every DiT weight file.** Upstream folders ship multiple ~5GB duplicates; we patched `smart_load_model` to prefer `model.fp16.safetensors` + `config.yaml` only.
- Paint prefers **safetensors** over duplicate `.bin` where possible; use [`trim_model_cache.py`](trim_model_cache.py) if an earlier full download wasted disk.
- Windows extension builds need **VS 2022 C++ tools** + a **CUDA 12.8-compatible `nvcc`** (this tree may use a staged `.cuda_home` for toolkit mismatch).
- Generation runs in a **worker process** so the status API does not freeze during volume decoding / model load.

## Setup (Windows)

```powershell
cd justthink
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\venv\Scripts\Activate.ps1
python webui.py
```

`setup.ps1` creates the venv, installs cu128 PyTorch, clones `Hunyuan3D-2`, builds `custom_rasterizer` + mesh processor, and installs deps.

CLI alternative: `python app.py` (see help text in the script).

## Disk / cache

Weights live under Hugging Face hub cache, e.g. `~/.cache/huggingface/hub/models--tencent--Hunyuan3D-2`. Useful size after trim is on the order of **~14GB** (shape + delight + paint), not the 40–50GB first-download trap.

## Other models you might choose instead

- Hunyuan **turbo** subfolders — faster, softer quality.
- Higher octree / more steps — better mesh, hungrier VRAM.
- Other open image→3D stacks (TripoSR, InstantMesh, etc.) — different quality/speed trade-offs; not wired here.

## Layout

```
justthink/
├── setup.ps1          # Windows bootstrap
├── app.py             # Pipeline + CLI
├── webui.py           # FastAPI UI (:7860)
├── static/index.html
├── trim_model_cache.py
├── outputs/           # GLBs / previews (gitignored)
└── Hunyuan3D-2/       # cloned by setup (gitignored)
```

## License

App glue is for local use. **Hunyuan3D weights and code follow Tencent’s license** — read the upstream repo / model card before commercial use.
