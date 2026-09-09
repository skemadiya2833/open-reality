"""
Download OzzyGT LTX-2.3 Distilled bitsandbytes NF4 pack (Windows-compatible).

Nunchaku NVFP4 lite-infer kernels are Linux-only — do not use those on Windows.
No GGUF, no full BF16, no WAN, no upscalers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

# bitsandbytes NF4 Diffusers pack — works on Windows (no Nunchaku CUDA kernels)
REPO_ID = "OzzyGT/LTX-2.3-Distilled-bnb-nf4"
ALLOW_ROOTS = {
    "model_index.json",
    "transformer",
    "text_encoder",
    "connectors",
    "vae",
    "audio_vae",
    "vocoder",
    "tokenizer",
    "processor",
    "scheduler",
}
IGNORE = ("*.md", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.mp4", "*.gguf")
FORBIDDEN = (".gguf", "wan", "upscaler", "lora", "nunchaku")


def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def plan_files() -> list[tuple[str, int]]:
    api = HfApi()
    info = api.repo_info(REPO_ID, files_metadata=True)
    chosen: list[tuple[str, int]] = []
    for sibling in info.siblings or []:
        path = sibling.rfilename
        low = path.lower()
        if any(bad in low for bad in FORBIDDEN):
            continue
        if path.endswith((".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4")):
            continue
        root = path.split("/", 1)[0]
        if path != "model_index.json" and root not in ALLOW_ROOTS:
            continue
        size = int(getattr(sibling, "size", 0) or 0)
        chosen.append((path, size))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    print("NOTE: Nunchaku NVFP4 kernels are Linux-only.")
    print(f"Windows model: {REPO_ID}")
    files = plan_files()
    total = sum(s for _, s in files)
    print(f"Will download {len(files)} files ~ {_gb(total)}")
    for path, size in sorted(files, key=lambda x: -x[1])[:15]:
        print(f"  {_gb(size):>10}  {path}")
    if len(files) > 15:
        print(f"  ... {len(files) - 15} more")

    if total <= 0:
        print("REFUSING: empty download plan")
        sys.exit(2)

    if not args.yes:
        ans = input("Proceed with download? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    allow = ["model_index.json"] + [f"{r}/*" for r in ALLOW_ROOTS if r != "model_index.json"]
    path = snapshot_download(
        REPO_ID,
        allow_patterns=allow,
        ignore_patterns=list(IGNORE),
    )
    print(f"Cached at: {path}")
    marker = Path(__file__).resolve().parent / ".model_ready"
    marker.write_text(path, encoding="utf-8")
    print("Done. Marker written to .model_ready")
    print("Optional: delete the unused Linux-only Nunchaku cache to free ~22GB:")
    print("  %USERPROFILE%\\.cache\\huggingface\\hub\\models--lite-infer--LTX-2.3-Distilled-v1.1-Diffusers-nunchaku-lite-nvfp4-bnb4-text-encoder")


if __name__ == "__main__":
    main()
