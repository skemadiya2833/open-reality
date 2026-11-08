@echo off
title justimagine
setlocal EnableExtensions

rem Prefer the folder next to this script; fall back to the repo install path
set "ROOT=%~dp0"
if not exist "%ROOT%webui.py" set "ROOT=d:\Workspace\Devoids\OpenReality\justimagine\"

cd /d "%ROOT%" || (
  echo Could not open justimagine folder:
  echo   %ROOT%
  pause
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo Missing venv. From this folder run:
  echo   powershell -ExecutionPolicy Bypass -File .\setup.ps1
  pause
  exit /b 1
)

echo Starting justimagine...
echo UI: http://127.0.0.1:7861
echo.
start "" "http://127.0.0.1:7861"
"venv\Scripts\python.exe" webui.py
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo.
  echo justimagine exited with code %ERR%.
  pause
)
exit /b %ERR%
