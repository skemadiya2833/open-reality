# Hunyuan3D-2.0 native setup (no mmgp / no offload) - Windows PowerShell
# Requires: Python 3.10-3.12, git, VS 2022 Build Tools (C++), CUDA Toolkit (nvcc),
#           CUDA 12.8-capable driver (Blackwell / RTX 5060 Ti)
# Usage (from justthink/):
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
    # Prefer 3.10-3.12. Open3D / many CUDA extension wheels do not support 3.14.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($ver in @("3.12", "3.11", "3.10")) {
            try {
                $path = & py "-$ver" -c "import sys; print(sys.executable)" 2>$null
                if ($LASTEXITCODE -eq 0 -and $path) {
                    $candidates += $path.Trim()
                }
            } catch {
            }
        }
    }
    foreach ($p in @(
            "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
            "C:\Python312\python.exe",
            "C:\Python311\python.exe",
            "C:\Python310\python.exe"
        )) {
        if (Test-Path $p) {
            $candidates += $p
        }
    }
    return $candidates | Select-Object -Unique
}

function Resolve-CudaHome {
    # Prefer local staged CUDA home (12.8 nvcc + merged headers) for PyTorch cu128 builds.
    $staged = Join-Path $ScriptDir ".cuda_home"
    if (Test-Path (Join-Path $staged "bin\nvcc.exe")) {
        return $staged
    }
    if ($env:CUDA_HOME -and (Test-Path (Join-Path $env:CUDA_HOME "bin\nvcc.exe"))) {
        return $env:CUDA_HOME.Trim()
    }
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        return $env:CUDA_PATH.Trim()
    }
    $root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $root) {
        # Prefer 12.x to match torch cu128; avoid 13.x major mismatch.
        $dirs = @(Get-ChildItem $root -Directory | Sort-Object Name -Descending)
        foreach ($d in $dirs) {
            if ($d.Name -like "v12.*" -and (Test-Path (Join-Path $d.FullName "bin\nvcc.exe"))) {
                return $d.FullName
            }
        }
        foreach ($d in $dirs) {
            if (Test-Path (Join-Path $d.FullName "bin\nvcc.exe")) {
                return $d.FullName
            }
        }
    }
    return $null
}

function Test-Msvc {
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        return $true
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($installPath) {
            return $true
        }
    }
    return $false
}

Write-Host "==> Locating Python 3.10-3.12..."
$pyList = @(Find-Python312)
if ($pyList.Count -eq 0) {
    Write-Host ""
    Write-Host "ERROR: Need Python 3.10, 3.11, or 3.12 (you currently appear to have only 3.14)."
    Write-Host "Open3D and several Hunyuan deps do not ship Windows wheels for 3.14."
    Write-Host "Install with:"
    Write-Host "  winget install -e --id Python.Python.3.12"
    Write-Host "Then re-run this script."
    exit 1
}
$BasePython = $pyList[0]
Write-Host "    Using: $BasePython"
& $BasePython -c "import sys; v=sys.version_info; assert v.major==3 and 10<=v.minor<=12, 'Need 3.10-3.12, got ' + sys.version; print(sys.version)"
Assert-LastExitCode "Python version check"

Write-Host "==> Checking MSVC (needed for xatlas + native extensions)..."
if (-not (Test-Msvc)) {
    Write-Host ""
    Write-Host "ERROR: Visual Studio C++ build tools not found (cl.exe / VC Tools)."
    Write-Host "Install Build Tools, then re-open this terminal:"
    Write-Host '  winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
    exit 1
}
Write-Host "    MSVC OK"

Write-Host "==> Resolving CUDA_HOME (needed for custom_rasterizer)..."
$cudaHome = Resolve-CudaHome
if (-not $cudaHome) {
    Write-Host ""
    Write-Host "ERROR: CUDA Toolkit not found (nvcc missing). The GPU driver alone is not enough."
    Write-Host "Install the CUDA Toolkit, then re-run:"
    Write-Host "  winget install -e --id Nvidia.CUDA"
    Write-Host "Or install CUDA 12.8 from https://developer.nvidia.com/cuda-downloads"
    exit 1
}
$cudaHome = $cudaHome.Trim()
$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$cudaBin = Join-Path $cudaHome "bin"
$env:Path = $cudaBin + ";" + $env:Path
Write-Host ("    CUDA_HOME=" + $env:CUDA_HOME)

# Prefer building native extensions with the VS developer environment when possible
$vsDevCmd = $null
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $candidate = Join-Path $vsPath "Common7\Tools\VsDevCmd.bat"
    if (Test-Path $candidate) {
        $vsDevCmd = $candidate
    }
}

function Invoke-NativeBuild {
    param(
        [string]$WorkDir,
        [string]$PythonExe,
        [string]$SetupArgs
    )
    Push-Location $WorkDir
    try {
        # Required when VsDevCmd already activated the VC env (torch cpp_extension check).
        $env:DISTUTILS_USE_SDK = "1"
        if ($vsDevCmd) {
            $cmd = 'call "' + $vsDevCmd + '" -arch=amd64 >nul && set DISTUTILS_USE_SDK=1 && set "CUDA_HOME=' + $env:CUDA_HOME + '" && set "CUDA_PATH=' + $env:CUDA_PATH + '" && "' + $PythonExe + '" setup.py ' + $SetupArgs
            cmd.exe /c $cmd
        } else {
            & $PythonExe setup.py $SetupArgs
        }
        Assert-LastExitCode ("native build in " + $WorkDir)
    } finally {
        Pop-Location
    }
}

Write-Host "==> Recreating venv with compatible Python..."
if (Test-Path "venv") {
    Write-Host "    Removing old venv (was likely 3.14)..."
    Remove-Item -Recurse -Force "venv"
}
& $BasePython -m venv venv
Assert-LastExitCode "venv create"

$VenvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "venv python missing at $VenvPython"
}

Write-Host "==> Upgrading pip / wheel / setuptools / cmake / ninja / pybind11..."
& $VenvPython -m pip install --upgrade pip wheel setuptools cmake ninja pybind11
Assert-LastExitCode "pip bootstrap"

Write-Host "==> Installing PyTorch >=2.7.0 (cu128)..."
& $VenvPython -m pip install "torch>=2.7.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
Assert-LastExitCode "pytorch install"

Write-Host "==> Verifying CUDA..."
$verifyPath = Join-Path $env:TEMP "hy3d_cuda_verify.py"
@'
import torch, sys
ok = torch.cuda.is_available()
print("torch.cuda.is_available():", ok)
if not ok:
    print("ERROR: CUDA not available. cu128 wheels are required for Blackwell (RTX 5060 Ti).")
    sys.exit(1)
idx = torch.cuda.current_device()
props = torch.cuda.get_device_properties(idx)
vram_gb = props.total_memory / (1024 ** 3)
print("device:", torch.cuda.get_device_name(idx))
print("compute capability: %d.%d" % (props.major, props.minor))
print("total VRAM: %.2f GB" % vram_gb)
'@ | Set-Content -Path $verifyPath -Encoding ASCII
& $VenvPython $verifyPath
Assert-LastExitCode "CUDA verify"
Remove-Item $verifyPath -ErrorAction SilentlyContinue

Write-Host "==> Installing app dependencies..."
& $VenvPython -m pip install diffusers transformers accelerate safetensors sentencepiece protobuf open3d trimesh pygltflib pillow numpy einops omegaconf
Assert-LastExitCode "app deps"
& $VenvPython -c "import open3d, diffusers; print('open3d+diffusers OK')"
Assert-LastExitCode "import check"

$Hy3dDir = Join-Path $ScriptDir "Hunyuan3D-2"
if (-not (Test-Path (Join-Path $Hy3dDir ".git"))) {
    Write-Host "==> Cloning official Tencent Hunyuan3D-2 (not -2GP / deepbeepmeep)..."
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git $Hy3dDir
    Assert-LastExitCode "git clone"
} else {
    Write-Host "==> Hunyuan3D-2 already present, skipping clone."
}

Write-Host "==> Installing Hunyuan3D-2 package + requirements..."
Push-Location $Hy3dDir
try {
    & $VenvPython -m pip install -r requirements.txt
    Assert-LastExitCode "hy3d requirements"
    & $VenvPython -m pip install -e .
    Assert-LastExitCode "hy3d editable install"
} finally {
    Pop-Location
}

Write-Host "==> Building custom_rasterizer..."
# Torch 2.9+ Windows: NVCC needs -DUSE_CUDA or compiled_autograd.h hits C2872 'std' ambiguous.
$rasterSetup = Join-Path $Hy3dDir "hy3dgen\texgen\custom_rasterizer\setup.py"
if (Test-Path $rasterSetup) {
    $setupText = Get-Content -Raw $rasterSetup
    if ($setupText -notmatch "DUSE_CUDA") {
        Write-Host "    Patching custom_rasterizer setup.py for Windows USE_CUDA..."
        @'
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {"cxx": [], "nvcc": []}
if os.name == "nt":
    extra_compile_args["cxx"] += ["/DUSE_CUDA"]
    extra_compile_args["nvcc"] += ["-DUSE_CUDA", "-D_WIN32"]

custom_rasterizer_module = CUDAExtension(
    "custom_rasterizer_kernel",
    [
        "lib/custom_rasterizer_kernel/rasterizer.cpp",
        "lib/custom_rasterizer_kernel/grid_neighbor.cpp",
        "lib/custom_rasterizer_kernel/rasterizer_gpu.cu",
    ],
    extra_compile_args=extra_compile_args,
)

setup(
    packages=find_packages(),
    version="0.1",
    name="custom_rasterizer",
    include_package_data=True,
    package_dir={"": "."},
    ext_modules=[custom_rasterizer_module],
    cmdclass={"build_ext": BuildExtension},
)
'@ | Set-Content -Path $rasterSetup -Encoding ASCII
    }
}
Invoke-NativeBuild -WorkDir (Join-Path $Hy3dDir "hy3dgen\texgen\custom_rasterizer") -PythonExe $VenvPython -SetupArgs "install"

Write-Host "==> Building differentiable_renderer..."
Invoke-NativeBuild -WorkDir (Join-Path $Hy3dDir "hy3dgen\texgen\differentiable_renderer") -PythonExe $VenvPython -SetupArgs "install"

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host "Then run:"
Write-Host "  python app.py"
