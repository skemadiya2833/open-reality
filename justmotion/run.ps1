# Launch justmotion with justimagine's venv (torch + FLUX deps).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root "..\justimagine\venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "ERROR: justimagine venv not found at $Py"
    Write-Host "Run justimagine\setup.ps1 first."
    exit 1
}
if (-not $env:EXP_DLSS_ROOT) {
    $env:EXP_DLSS_ROOT = "D:\Workspace\Devoids\FULLONVIDEOEDITING\the-exp-dlss"
}
Set-Location $Root
Write-Host "Using $Py"
Write-Host "EXP_DLSS_ROOT=$env:EXP_DLSS_ROOT"
& $Py webui.py @args
