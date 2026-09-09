"""
Trim Hunyuan3D-2 HuggingFace cache to only the weight files we load.

Shape:   hunyuan3d-dit-v2-0/model.fp16.safetensors + config.yaml  (~5GB)
Paint:   hunyuan3d-paint-v2-0 (safetensors + text_encoder .bin)  (~5GB)
Delight: hunyuan3d-delight-v2-0 safetensors                      (~4GB)

Stop webui.py before running this.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path.home() / ".cache/huggingface/hub/models--tencent--Hunyuan3D-2"
KEEP_DIT = {"config.yaml", "model.fp16.safetensors"}


def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def _remove(path: Path, freed: list[int]) -> None:
    if not path.exists():
        return
    size = path.stat().st_size if path.is_file() else 0
    if path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    label = path.name if path.parent.name == "blobs" else str(path)
    print(f"remove {label} ({_gb(size)})")
    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        freed[0] += size
    except PermissionError:
        print("  LOCKED — stop webui.py / python first, then re-run.")


def main() -> None:
    if not REPO.exists():
        print("No Hunyuan3D-2 HF cache found.")
        return

    freed = [0]

    blobs = REPO / "blobs"
    if blobs.exists():
        for p in blobs.iterdir():
            if p.is_file() and ".incomplete" in p.name:
                _remove(p, freed)

    snaps = REPO / "snapshots"
    if snaps.exists():
        for snap in snaps.iterdir():
            dit = snap / "hunyuan3d-dit-v2-0"
            if dit.exists():
                for p in dit.iterdir():
                    if p.name not in KEEP_DIT:
                        _remove(p, freed)

            paint = snap / "hunyuan3d-paint-v2-0"
            if paint.exists():
                for p in paint.rglob("*.bin"):
                    if "text_encoder" in p.parts:
                        continue
                    _remove(p, freed)

    total = sum(f.stat().st_size for f in REPO.rglob("*") if f.is_file())
    print(f"done. removed ~{_gb(freed[0])}. cache now {_gb(total)}.")


if __name__ == "__main__":
    main()
