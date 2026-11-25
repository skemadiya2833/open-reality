"""
Download Diffusers components for justimagine models.

  flux-klein  — FLUX.2-klein-4B (skip duplicate single-file + teasers)
  qwen-turbo  — Qwen-Image-2.1-Turbo torchao int8 (16GB-friendly)
  all         — both

Large Qwen shards use curl on Windows when huggingface_hub stalls on LFS/XET.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent

MODELS = {
    "flux-klein": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "marker": ROOT / ".model_ready",
        "allow": (
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "tokenizer/*",
            "transformer/*",
            "vae/*",
        ),
        "ignore": (
            "flux-2-klein-4b.safetensors",
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.md",
            "*.gif",
            "*.webp",
        ),
        "forbidden": ("flux-2-klein-4b.safetensors",),
        "allow_roots": {
            "scheduler",
            "text_encoder",
            "tokenizer",
            "transformer",
            "vae",
        },
        "method": "snapshot",
    },
    "qwen-turbo": {
        # Diffusers-native torchao int8 (~16.6 GB). Same staged residency as FLUX.
        # torchao-int8 redirects here; use diffuser-8bit directly for stable CDN URLs.
        "repo_id": "addlabsviral/Qwen-Image-2.1-Turbo-diffuser-8bit",
        "marker": ROOT / ".model_ready_qwen",
        "local_dir": ROOT / "models" / "qwen-turbo",
        "allow": (
            "model_index.json",
            "scheduler/*",
            "text_encoder/*",
            "processor/*",
            "transformer/*",
            "vae/*",
        ),
        "ignore": (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.md",
            "*.gif",
            "*.webp",
            "assets/*",
        ),
        "forbidden": (),
        "allow_roots": {
            "scheduler",
            "text_encoder",
            "processor",
            "transformer",
            "vae",
        },
        "method": "local_assemble",
        "large_files": (
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
        ),
    },
}


def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def plan_files(key: str) -> list[tuple[str, int]]:
    cfg = MODELS[key]
    api = HfApi()
    info = api.repo_info(cfg["repo_id"], files_metadata=True)
    chosen: list[tuple[str, int]] = []
    for sibling in info.siblings or []:
        path = sibling.rfilename
        if any(bad in path for bad in cfg["forbidden"]):
            continue
        if path.endswith((".jpg", ".jpeg", ".png", ".md", ".gif", ".webp")):
            continue
        if path.startswith("assets/"):
            continue
        ok = path == "model_index.json" or path.split("/", 1)[0] in cfg["allow_roots"]
        if not ok:
            continue
        size = int(getattr(sibling, "size", 0) or 0)
        chosen.append((path, size))
    return chosen


def _curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl.exe",
        "-L",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--continue-at",
        "-",
        "--fail",
        "--output",
        str(dest),
        url,
    ]
    print(f"  curl → {dest.name}", flush=True)
    subprocess.run(cmd, check=True)


def _download_qwen_local(cfg: dict, files: list[tuple[str, int]]) -> Path:
    local_dir: Path = cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg["repo_id"]
    large = set(cfg.get("large_files") or ())

    # Small files via hub (fast). Large LFS shards via curl (reliable on this box).
    for path, size in sorted(files, key=lambda x: x[1]):
        dest = local_dir / path
        if dest.is_file() and dest.stat().st_size == size and size > 0:
            print(f"  skip (exists) {path}")
            continue
        if path in large or size > 100 * 1024 * 1024:
            url = f"https://huggingface.co/{repo}/resolve/main/{path}"
            _curl_download(url, dest)
            if size and dest.stat().st_size != size:
                raise RuntimeError(
                    f"size mismatch for {path}: got {dest.stat().st_size}, expected {size}"
                )
        else:
            cached = hf_hub_download(repo_id=repo, filename=path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, dest)
            print(f"  copied {path}")

    # Sanity: Diffusers needs model_index.json
    idx = local_dir / "model_index.json"
    if not idx.is_file():
        raise RuntimeError("model_index.json missing after download")
    json.loads(idx.read_text(encoding="utf-8"))
    return local_dir


def download_one(key: str, yes: bool) -> None:
    cfg = MODELS[key]
    repo = cfg["repo_id"]
    print(f"\n=== {key} ===")
    print(f"Repo: {repo}")
    files = plan_files(key)
    total = sum(s for _, s in files)
    print(f"Will download {len(files)} files ~ {_gb(total)}")
    for path, size in sorted(files, key=lambda x: -x[1])[:12]:
        print(f"  {_gb(size):>10}  {path}")
    if len(files) > 12:
        print(f"  … {len(files) - 12} more")

    junk = [p for p, _ in files if any(b in p for b in cfg["forbidden"])]
    if junk:
        print("REFUSING: planned set still contains forbidden files:", junk)
        sys.exit(2)

    if not yes:
        ans = input(f"Proceed with {key} download? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Skipped.")
            return

    if cfg.get("method") == "local_assemble":
        path = str(_download_qwen_local(cfg, files))
    else:
        path = snapshot_download(
            repo,
            allow_patterns=list(cfg["allow"]),
            ignore_patterns=list(cfg["ignore"]),
        )
    print(f"Cached at: {path}")
    cfg["marker"].write_text(path, encoding="utf-8")
    print(f"Done. Marker written to {cfg['marker'].name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["flux-klein", "qwen-turbo", "all"],
        default="all",
        help="Which model(s) to download (default: all)",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    keys = list(MODELS) if args.model == "all" else [args.model]
    for key in keys:
        download_one(key, yes=args.yes)


if __name__ == "__main__":
    main()
