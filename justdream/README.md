# justdream — continuity story video

The old overnight mode generated **unrelated 1–2s clips** (10 regenerations per sentence) with **no style lock** — that wasted a night. This build is continuity-first.

## What changed

| Before (broken) | Now |
|-----------------|-----|
| ~10 random clips per sentence | **1 clip per beat** |
| No visual memory | **Last-frame conditioning** (clip N+1 starts from clip N) |
| Style soup (anime + Hollywood + CGI) | **Style lock** on every prompt + anti-drift negative |
| Text only | Attach **reference images/videos** as opening keyframe |
| Forget on crash | `plan.json` + `memory/last_XXX.png` **resume** |

## How to run

```powershell
cd justdream
.\venv\Scripts\Activate.ps1
python webui.py
```

1. Pick **Continuity · 640p** (default).
2. Paste script (one sentence / paragraph = one beat).
3. Optional: edit **style lock**, attach character/scene refs.
4. **Dream** — keep PC awake.

## Presets

- **continuity** (default) — 640×384 · 49f · ~2s · chained  
- **continuity_long** — ~4s beats  
- **continuity_720p** — 1280×704 · short chained clips  
- **cinema** / **fast** — variants  

Model: `OzzyGT/LTX-2.3-Distilled-bnb-nf4` via `LTX2ConditionPipeline` when available.
