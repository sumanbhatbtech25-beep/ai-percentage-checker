@echo off
title Report Program (close this window to stop)
cd /d "%~dp0"
set PORT=8000
set HF_HUB_DISABLE_XET=1
rem The launch key lives in YOUR app-data folder, which other Windows accounts can't read.
rem It is what lets only you open Report Program on a shared computer.
set KEYDIR=%LOCALAPPDATA%\ReportProgram
set KEYFILE=%KEYDIR%\launch.key

rem Already running? Open it with the saved key.
curl -s -o nul http://127.0.0.1:%PORT%/api/health
if not errorlevel 1 goto open_running

if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. See README.md, "Manual setup".
  pause
  exit /b 1
)

if not exist "%KEYDIR%" mkdir "%KEYDIR%"
for /f "usebackq delims=" %%k in (`.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(24))"`) do set AIC_LOCAL_KEY=%%k
> "%KEYFILE%" echo %AIC_LOCAL_KEY%

echo Starting Report Program on http://localhost:%PORT%
echo The detection models take about 30 seconds to load. Keep this window open; close it to stop.
rem open the browser (with the key, once) a few seconds after the server starts listening
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start "" http://localhost:%PORT%/?key=%AIC_LOCAL_KEY%"
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
exit /b

:open_running
set /p AIC_LOCAL_KEY=<"%KEYFILE%"
start "" http://localhost:%PORT%/?key=%AIC_LOCAL_KEY%
exit /b
