# justimagine

**Text → image** and **contextual image editing** with **FLUX.2 [klein] 4B** (bf16) via Diffusers.

No ComfyUI. No Ollama. One unified model for generate + multi-reference-style edit (`image=` conditioning).

UI: http://127.0.0.1:7861 after `python webui.py`.

## What it does

- Prompt only → new image.
- Prompt + attached image → edit using that image as reference.
- Follow-up prompts **reuse the last result as context** (chat continuity), unless you check “start fresh” or clear context.
- Results land in `outputs/` as PNG.

## Why FLUX.2 klein 4B on this hardware

| Choice | Why |
|--------|-----|
| [`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | Apache 2.0, ~4B, designed for consumer GPUs (~13GB class) |
| **bf16** full load on 16GB | Fits; we try `.to("cuda")` first, offload only if OOM |
| **4 steps**, `guidance_scale=1.0` | Distilled klein defaults — interactive latency |
| Diffusers from **git** | Needs current `Flux2KleinPipeline` |

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

CLI: `python app.py "a red bicycle" --image optional.png`

## UI notes

- Composer stays enabled while generating (queue).
- Status poll stays live — generation runs in a **separate process**.
- `device_mode` in `/api/status` shows `full` or `offload`.

## Other models you might choose instead

| Model | When |
|-------|------|
| FLUX.2 klein **9B** | Higher quality; needs more VRAM / offload |
| FLUX.1 / SDXL / SD3 | Different ecosystems; swap pipeline class in `app.py` |
| FP8 / NVFP4 klein builds | If you want smaller residency on Blackwell |

## Layout

```
justimagine/
├── setup.ps1
├── download_models.py
├── app.py
├── webui.py
├── static/index.html
└── outputs/
```

## License

App glue for local use. **FLUX.2 [klein] 4B is Apache 2.0** — still verify the model card for your use case.
