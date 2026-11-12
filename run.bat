@echo off
cd /d "%~dp0"
echo Starting AI Percentage Checker on http://localhost:8000
start "" http://localhost:8000
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
