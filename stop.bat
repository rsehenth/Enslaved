@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem  Enslaved - stop (Windows)
rem  The logic lives in stop.ps1. Stops only what provably belongs to
rem  this project: the backend from this folder's .venv (which serves
rem  both this PC and the local network), its start.bat window, and the
rem  Ollama that start.bat started and recorded. A pre-existing Ollama or
rem  one used by other programs is preserved and listed.
rem  "stop.bat dryrun" only lists, without stopping or changing anything.
rem ============================================================

echo [Enslaved] Stopping Enslaved (this PC and local network)...
if /i "%~1"=="dryrun" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" -DryRun
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1"
)
echo [Enslaved] Done. To start again, run start.bat.
timeout /t 4 >nul
