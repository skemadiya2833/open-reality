# justimagine

**Text → image** and **contextual image editing** with **FLUX.2 [klein] 4B** (bf16) via Diffusers.

No ComfyUI. No Ollama. One unified model for generate + multi-reference-style edit (`image=` conditioning).

UI: http://127.0.0.1:7861 after `python webui.py`, or double-click `justimagine.bat` (also on the Desktop).

## What it does

- Prompt only → new image.
- Prompt + one or more attached images → edit with multi-reference conditioning.
- Follow-up prompts **reuse the last result as context** (chat continuity). Use **New image** / `POST /api/new_chat` for a fresh thread.
- Chats and messages persist in SQLite (`justimagine.db`).
- Aspect + quality presets in the UI map to width/height and steps/guidance.
- Results land in `outputs/` as PNG.

## Why FLUX.2 klein 4B on this hardware

| Choice | Why |
|--------|-----|
| [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache 2.0, ~4B, designed for consumer GPUs (~13GB class) |
| **bf16 resident DiT on 16GB** | DiT + VAE stay on GPU; text encoder visits only for the prompt (they cannot share 16GB) |
| **4 steps**, `guidance_scale=1.0` | Distilled klein defaults — interactive latency |
| **torch.compile** | Opt-in via `JUSTIMAGINE_COMPILE=1` (default on non-Windows). Windows needs Triton for inductor, so default is off there |
| Diffusers from **git** | Needs current `Flux2KleinPipeline` |

**Speed:** warm 1024² is ~10–20s on a 5060 Ti after residency + compile. First `webui.py` start **preloads** weights. Status should show `device_mode=staged` (and `compiled`). `JUSTIMAGINE_FULL=1` forces all-on-GPU (often slower on 16GB due to paging).

UI shows live ETA, encode/denoise timing, and an **API** tab with curl examples. Machine JSON: `GET /api/docs`.

Target box: **RTX 5060 Ti 16GB Blackwell** + PyTorch **cu128**.

## Minimal download (important)

The Hub repo also contains a single-file `flux-2-klein-4b.safetensors` (~7.7GB) that **duplicates** the Diffusers shards, plus teaser JPGs.

[`download_models.py`](download_models.py) only pulls:

- `model_index.json`
- `transformer/`, `text_encoder/`, `tokenizer/`, `vae/`, `scheduler/`

Typical fetch: **~15GB**, not ~23GB. Setup prints the plan before downloading (`--yes` in automated setup).

## Setup (Windows)

```powershell
cd justimagine
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\venv\Scripts\Activate.ps1
python webui.py
```

Or: `justimagine.bat` (opens the UI and starts the server).

CLI: `python app.py "a red bicycle" --image optional.png`

## UI notes

- Composer stays enabled while generating (queue).
- Status poll stays live — generation runs in a **separate process**.
- `device_mode` in `/api/status` shows `full` or `staged`.
- API tab + `GET /api/docs` cover generate, multi-ref, chats, and presets.

## Other models you might choose instead

| Model | When |
|-------|------|
| FLUX.2 klein **9B** | Higher quality; needs more VRAM / offload |
| FLUX.1 / SDXL / SD3 | Different ecosystems; swap pipeline class in `app.py` |
| FP8 / NVFP4 klein builds | If you want smaller residency on Blackwell |

## Layout

```
justimagine/
├── justimagine.bat
├── setup.ps1
├── download_models.py
├── app.py
├── webui.py
├── db.py
├── justimagine.db   (local, gitignored)
├── static/index.html
└── outputs/
```

## License

App glue for local use. **FLUX.2 [klein] 4B is Apache 2.0** — still verify the model card for your use case.
