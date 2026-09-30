# justimagine setup - FLUX.2 klein 4B (no ComfyUI)
# Requires: Python 3.10-3.12, CUDA 12.8-capable driver (Blackwell / RTX 5060 Ti)
# Usage (from justimagine/):
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

function Assert-LastExitCode {
    param([string]$Step)
    if ($null -ne $LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Find-Python312 {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in @("3.12", "3.11", "3.10")) {
            try {
                $path = & py "-$ver" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $path) { $candidates += $path.Trim() }
            } catch {}
        }
    }
    foreach ($p in @(
            "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
        )) {
        if (Test-Path $p) { $candidates += $p }
    }
    return $candidates | Select-Object -Unique
}

Write-Host "==> Locating Python 3.10-3.12..."
$pyList = @(Find-Python312)
if ($pyList.Count -eq 0) {
    Write-Host "ERROR: Need Python 3.10-3.12. Install: winget install -e --id Python.Python.3.12"
    exit 1
}
$BasePython = $pyList[0]
Write-Host "    Using: $BasePython"
& $BasePython -c "import sys; v=sys.version_info; assert v.major==3 and 10<=v.minor<=12; print(sys.version)"
Assert-LastExitCode "Python version check"

Write-Host "==> Creating venv..."
if (-not (Test-Path "venv")) {
    & $BasePython -m venv venv
    Assert-LastExitCode "venv create"
}
$VenvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"

Write-Host "==> Upgrading pip..."
& $VenvPython -m pip install --upgrade pip wheel setuptools
Assert-LastExitCode "pip bootstrap"

Write-Host "==> Linking / installing shared PyTorch cu128 (repo .shared/torch-cu128)..."
$LinkTorch = Join-Path (Split-Path $ScriptDir -Parent) "link_shared_torch.ps1"
& $LinkTorch -Project justimagine -InstallIfMissing
Assert-LastExitCode "shared pytorch"

Write-Host "==> Verifying CUDA..."
& $VenvPython -c "import torch; assert torch.cuda.is_available(), 'CUDA missing'; p=torch.cuda.get_device_properties(0); print(torch.cuda.get_device_name(0), 'VRAM %.1f GB' % (p.total_memory/1024**3))"
Assert-LastExitCode "CUDA verify"

Write-Host "==> Installing Diffusers (git) + app deps..."
& $VenvPython -m pip install -U "git+https://github.com/huggingface/diffusers.git" transformers accelerate safetensors sentencepiece protobuf pillow numpy fastapi uvicorn python-multipart huggingface_hub
Assert-LastExitCode "deps"

Write-Host "==> Downloading FLUX.2 klein 4B (minimal files)..."
& $VenvPython download_models.py --yes
Assert-LastExitCode "model download"

New-Item -ItemType Directory -Force -Path "outputs","static" | Out-Null

Write-Host ""
Write-Host "Done. Activate and run:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "  python webui.py"
Write-Host "Open http://127.0.0.1:7861"
