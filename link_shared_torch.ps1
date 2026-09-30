# Link shared cu128 PyTorch into a project's venv (no re-download).
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\link_shared_torch.ps1 -Project justimagine
#   powershell -ExecutionPolicy Bypass -File .\link_shared_torch.ps1 -Project justthink
# Optional: -InstallIfMissing  (pip-installs torch once into .shared, then links)
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("justimagine", "justthink")]
    [string]$Project,

    [switch]$InstallIfMissing
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Shared = Join-Path $Root ".shared\torch-cu128"
$SitePackages = Join-Path $Root "$Project\venv\Lib\site-packages"
$VenvPython = Join-Path $Root "$Project\venv\Scripts\python.exe"

if (-not (Test-Path $SitePackages)) {
    throw "Missing site-packages: $SitePackages (create the venv first)"
}

$PkgNames = @(
    "torch",
    "torchvision",
    "torchaudio",
    "torchgen",
    "functorch"
)

function Get-TorchPackageDirs([string]$Dir) {
    if (-not (Test-Path $Dir)) { return @() }
    Get-ChildItem $Dir -Directory -Force | Where-Object {
        $_.Name -in $PkgNames -or $_.Name -match '^(torch|torchvision|torchaudio)-.+\.dist-info$'
    }
}

function Ensure-SharedTorch {
    $torchDir = Join-Path $Shared "torch"
    if (Test-Path $torchDir) { return }

    if (-not $InstallIfMissing) {
        throw "Shared torch missing at $Shared. Re-run with -InstallIfMissing, or install once into a venv then move it here."
    }
    if (-not (Test-Path $VenvPython)) {
        throw "Need $VenvPython to bootstrap shared torch"
    }

    New-Item -ItemType Directory -Force -Path $Shared | Out-Null
    Write-Host "==> Installing PyTorch cu128 once into shared store..."
    & $VenvPython -m pip install "torch>=2.7.0" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw "pytorch install failed" }

    foreach ($item in (Get-TorchPackageDirs $SitePackages)) {
        $dest = Join-Path $Shared $item.Name
        if (Test-Path $dest) { continue }
        # Skip if already a junction into shared
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
        Write-Host "    MOVE $($item.Name) -> .shared"
        Move-Item -LiteralPath $item.FullName -Destination $dest
    }

    if (-not (Test-Path $torchDir)) {
        throw "Shared torch still missing after install"
    }
}

function Link-IntoVenv {
    New-Item -ItemType Directory -Force -Path $Shared | Out-Null
    $sharedItems = @(Get-TorchPackageDirs $Shared)
    if ($sharedItems.Count -eq 0) {
        throw "No torch packages in $Shared"
    }

    foreach ($item in $sharedItems) {
        $link = Join-Path $SitePackages $item.Name
        $target = $item.FullName

        if (Test-Path $link) {
            $existing = Get-Item $link -Force
            $isReparse = ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0
            if ($isReparse) {
                # Already a junction/symlink — recreate only if target differs
                $currentTarget = $null
                try { $currentTarget = (Get-Item $link -Force).Target } catch {}
                if ($currentTarget -and (@($currentTarget) -contains $target)) {
                    Write-Host "    OK  $($item.Name)"
                    continue
                }
                Write-Host "    RELINK $($item.Name)"
                cmd /c rmdir "$link" | Out-Null
            } else {
                Write-Host "    REPLACE local $($item.Name) with junction"
                Remove-Item -LiteralPath $link -Recurse -Force
            }
        }

        Write-Host "    JUNCTION $($item.Name)"
        cmd /c mklink /J "$link" "$target" | Out-Null
        if (-not (Test-Path $link)) { throw "Failed to create junction for $($item.Name)" }
    }
}

Ensure-SharedTorch
Write-Host "==> Linking shared torch into $Project..."
Link-IntoVenv

Write-Host "==> Verifying..."
& $VenvPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "CUDA verify failed" }
Write-Host "Done. Shared store: $Shared"
