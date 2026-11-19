@echo off
setlocal
cd /d "%~dp0"

rem ============================================================
rem  Enslaved - start (Windows)
rem  - Always serves this PC (127.0.0.1) AND the local network (0.0.0.0),
rem    so phones on the same Wi-Fi can open it. NO authentication, NO HTTPS.
rem  - Arguments are rejected.
rem  - Detects Ollama on 127.0.0.1:11434; if absent and the executable exists,
rem    starts "ollama serve" and records it in .run\ollama.json so stop.bat
rem    closes only that instance. A pre-existing Ollama is never touched.
rem ============================================================

if not "%~1"=="" (
    echo [Enslaved] Unknown argument: "%~1"
    echo            Usage: start.bat   ^(no arguments; opens Enslaved on this PC and your local network^)
    exit /b 2
)

if not exist ".venv\Scripts\python.exe" (
    echo [Enslaved] Virtual environment not found in .venv
    echo            Create it with: python -m venv .venv
    echo            Then run:       .venv\Scripts\pip install -e ".[dev]"
    pause
    exit /b 1
)

set "OLLAMA_EXE="
where ollama >nul 2>nul && set "OLLAMA_EXE=ollama"
if not defined OLLAMA_EXE if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
)

curl -s -o NUL --max-time 3 http://127.0.0.1:11434/api/tags
if errorlevel 1 (
    if not defined OLLAMA_EXE (
        echo [Enslaved] Ollama was not found on this computer.
        echo            Install it from https://ollama.com and pull the model set in config\settings.yaml.
        echo            Chat will stay unavailable until Ollama responds.
    ) else (
        echo [Enslaved] Ollama is installed but not responding. Starting "ollama serve"...
        powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-ollama.ps1" -Exe "%OLLAMA_EXE%"
        for /L %%i in (1,1,6) do (
            timeout /t 1 /nobreak >NUL
            curl -s -o NUL --max-time 2 http://127.0.0.1:11434/api/tags && goto :ollama_ready
        )
        echo [Enslaved] Warning: Ollama is still not responding on 127.0.0.1:11434. Starting the backend anyway.
    )
) else (
    echo [Enslaved] Ollama detected on 127.0.0.1:11434 ^(already running; stop.bat will not close it^)
)

:ollama_ready
curl -sf -o NUL --max-time 2 http://127.0.0.1:8000/api/health
if not errorlevel 1 (
    echo [Enslaved] A backend is already running at http://127.0.0.1:8000
    echo            Open that address in your browser. A second instance will not be started.
    echo            To restart it with network access, run stop.bat and then start.bat.
    pause
    exit /b 0
)

netstat -ano -p TCP 2>nul | findstr /R /C:":8000 .*LISTENING" >nul
if not errorlevel 1 (
    echo [Enslaved] Port 8000 is already in use by another process that did not answer /api/health.
    echo            Close the process using port 8000, or set another port in config\settings.yaml.
    pause
    exit /b 1
)

rem Process variables win over .env: the bind is always this PC + local network.
set "ENSLAVED_LAN=1"
set "ENSLAVED_HOST=0.0.0.0"
set "LAN_IPS="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq 'Up' } | ForEach-Object { $_.IPv4Address.IPAddress }) -join ' '"`) do set "LAN_IPS=%%i"

echo.
echo ============================================================
echo   Enslaved is starting...
echo.
echo   On this PC:       http://127.0.0.1:8000
if defined LAN_IPS (
    for %%a in (%LAN_IPS%) do echo   On your network:  http://%%a:8000   ^(phone on the same Wi-Fi^)
) else (
    echo   On your network:  address not detected, but port 8000 is open to your local network.
)
echo.
echo   WARNING: no authentication and no HTTPS. Any device on this
echo   network can read conversations, approve tools and delete data.
echo   Use only on a network you trust. If Windows Firewall asks,
echo   allow access on "Private networks" only.
echo.
echo   To stop everything, run stop.bat.
echo   Port 11434 belongs to Ollama and only shows "Ollama is running".
echo ============================================================
echo.

".venv\Scripts\python.exe" -m backend.main
if errorlevel 1 (
    echo [Enslaved] The backend exited with an error.
    pause
    exit /b 1
)
