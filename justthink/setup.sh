#!/usr/bin/env bash
# Hunyuan3D-2.0 native setup (no mmgp / no offload).
# On Windows prefer: powershell -ExecutionPolicy Bypass -File .\setup.ps1
# This script works in Git Bash / WSL. Requires CUDA 12.8-capable driver (Blackwell / RTX 5060 Ti).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer python on Windows, python3 elsewhere
if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "ERROR: python not found on PATH"
  exit 1
fi

echo "==> Creating venv..."
"$PYTHON" -m venv venv

# Windows (Git Bash / MSYS) uses Scripts/; Unix uses bin/
if [[ -f "venv/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/Scripts/activate
  VENV_PY="venv/Scripts/python.exe"
elif [[ -f "venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
  VENV_PY="venv/bin/python"
else
  echo "ERROR: could not find venv activate script"
  exit 1
fi

echo "==> Upgrading pip / wheel / setuptools..."
"$VENV_PY" -m pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch >=2.7.0 (cu128)..."
"$VENV_PY" -m pip install "torch>=2.7.0" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

echo "==> Verifying CUDA..."
"$VENV_PY" - <<'PY'
import torch
import sys

ok = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {ok}")
if not ok:
    print("ERROR: CUDA not available. cu128 wheels are required for Blackwell (RTX 5060 Ti).")
    sys.exit(1)

idx = torch.cuda.current_device()
props = torch.cuda.get_device_properties(idx)
vram_gb = props.total_memory / (1024 ** 3)
print(f"device: {torch.cuda.get_device_name(idx)}")
print(f"compute capability: {props.major}.{props.minor}")
print(f"total VRAM: {vram_gb:.2f} GB")
PY

echo "==> Installing Python dependencies..."
"$VENV_PY" -m pip install \
  diffusers transformers accelerate safetensors sentencepiece protobuf \
  open3d trimesh pygltflib pillow numpy einops omegaconf

HY3D_DIR="$SCRIPT_DIR/Hunyuan3D-2"
if [[ ! -d "$HY3D_DIR/.git" ]]; then
  echo "==> Cloning official Tencent Hunyuan3D-2 (not -2GP / deepbeepmeep)..."
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "$HY3D_DIR"
else
  echo "==> Hunyuan3D-2 already present, skipping clone."
fi

echo "==> Installing Hunyuan3D-2 package..."
cd "$HY3D_DIR"
"$VENV_PY" -m pip install -r requirements.txt
"$VENV_PY" -m pip install -e .

echo "==> Building custom_rasterizer..."
cd "$HY3D_DIR/hy3dgen/texgen/custom_rasterizer"
"$VENV_PY" setup.py install

echo "==> Building differentiable_renderer..."
cd "$HY3D_DIR/hy3dgen/texgen/differentiable_renderer"
"$VENV_PY" setup.py install

cd "$SCRIPT_DIR"
echo ""
echo "Done. On Windows activate with: .\\venv\\Scripts\\Activate.ps1"
echo "Then run: python app.py"
