@echo off
title Report Program (close this window to stop)
cd /d "%~dp0"
set PORT=8000
set HF_HUB_DISABLE_XET=1

rem Already running? Just open the site.
curl -s -o nul http://127.0.0.1:%PORT%/api/health && (
  start "" http://localhost:%PORT%
  exit /b
)

if not exist ".venv\Scripts\python.exe" (
  echo The Python environment is missing. See README.md, "Manual setup".
  pause
  exit /b 1
)

echo Starting Report Program on http://localhost:%PORT%
echo The detection models take about 30 seconds to load. Keep this window open; close it to stop.
rem open the browser a few seconds after the server starts listening
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start "" http://localhost:%PORT%"
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
