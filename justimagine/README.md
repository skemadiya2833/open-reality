# justimagine

**Text → image** and **contextual image editing** with **FLUX.2 [klein] 4B** (bf16) and **Qwen-Image-2.1-Turbo** (torchao int8) via Diffusers.

No ComfyUI. No Ollama. Pick the default model in the sidebar. Generate + multi-reference-style edit (`image=` conditioning).

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

[`download_models.py`](download_models.py) supports `--model flux-klein|qwen-turbo|all`.

**FLUX.2 klein** — skips the duplicate single-file weight (~7.7GB) and teasers. Typical fetch **~15GB**.

**Qwen Image 2.1 Turbo** — Diffusers-native **torchao int8** (`addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit`), ~**16.6GB**, stored under `models/qwen-turbo/`. Same staged DiT residency as FLUX (text encoder visits GPU only for the prompt). Qwen Research License = non-commercial.

```powershell
python download_models.py --model qwen-turbo --yes
```

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

## Models in the sidebar

| Model | Color | Notes |
|-------|-------|-------|
| FLUX.2 klein 4B | Blue | Default · Apache 2.0 · ~4 steps distilled |
| Qwen Image 2.1 Turbo int8 | Teal | 8-step schedule · non-commercial license · swap unloads the other model |

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
