"""
Download only Diffusers components for FLUX.2-klein-4B.
Skips the duplicate single-file weight (~7.7GB) and teaser images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO_ID = "black-forest-labs/FLUX.2-klein-4B"
ALLOW = (
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
)
IGNORE = (
    "flux-2-klein-4b.safetensors",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.md",
    "*.gif",
    "*.webp",
)
FORBIDDEN_SUBSTRINGS = (
    "flux-2-klein-4b.safetensors",
)


def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def plan_files() -> list[tuple[str, int]]:
    api = HfApi()
    info = api.repo_info(REPO_ID, files_metadata=True)
    chosen: list[tuple[str, int]] = []
    for sibling in info.siblings or []:
        path = sibling.rfilename
        if any(bad in path for bad in FORBIDDEN_SUBSTRINGS):
            continue
        if path.endswith((".jpg", ".jpeg", ".png", ".md", ".gif", ".webp")):
            continue
        # Keep only allow roots
        ok = path == "model_index.json" or path.split("/", 1)[0] in {
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "vae",
        }
        if not ok:
            continue
        size = int(getattr(sibling, "size", 0) or 0)
        chosen.append((path, size))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    print(f"Repo: {REPO_ID}")
    files = plan_files()
    total = sum(s for _, s in files)
    print(f"Will download {len(files)} files ~ {_gb(total)}")
    for path, size in sorted(files, key=lambda x: -x[1])[:12]:
        print(f"  {_gb(size):>10}  {path}")
    if len(files) > 12:
        print(f"  … {len(files) - 12} more")

    junk = [p for p, _ in files if any(b in p for b in FORBIDDEN_SUBSTRINGS)]
    if junk:
        print("REFUSING: planned set still contains forbidden files:", junk)
        sys.exit(2)

    if not args.yes:
        ans = input("Proceed with download? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            sys.exit(0)

    path = snapshot_download(
        REPO_ID,
        allow_patterns=list(ALLOW),
        ignore_patterns=list(IGNORE),
    )
    print(f"Cached at: {path}")
    # Touch marker so app can prefer local_files_only
    marker = Path(__file__).resolve().parent / ".model_ready"
    marker.write_text(path, encoding="utf-8")
    print("Done. Marker written to .model_ready")


if __name__ == "__main__":
    main()
