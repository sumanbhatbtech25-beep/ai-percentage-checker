@echo off
title Report Program - phone access (close this window to stop)
cd /d "%~dp0"
set PORT=8000
set HF_HUB_DISABLE_XET=1

curl -s -o nul http://127.0.0.1:%PORT%/api/health && (
  echo Report Program is already running in computer-only mode.
  echo Close that window first, then start this one again.
  pause
  exit /b 1
)

echo.
echo  Use Report Program from your phone on the SAME Wi-Fi.
echo  Choose a password. Your phone will ask for it once.
echo  (Use this only on your home or another trusted Wi-Fi, not public Wi-Fi.)
echo.
set /p AIC_PASSWORD=  Password:
if "%AIC_PASSWORD%"=="" (
  echo A password is required for phone access.
  pause
  exit /b 1
)

rem find this computer's Wi-Fi / Ethernet address
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -in 'Dhcp','Manual' -and $_.IPAddress -notlike '169.254*' -and $_.InterfaceAlias -notmatch 'vEthernet|VirtualBox|VMware|Loopback|VPN|Tun' } | Select-Object -First 1).IPAddress"`) do set LANIP=%%i

rem launch key for this computer (see run.bat); the phone uses the password instead
if not exist "%LOCALAPPDATA%\ReportProgram" mkdir "%LOCALAPPDATA%\ReportProgram"
for /f "usebackq delims=" %%k in (`.venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(24))"`) do set AIC_LOCAL_KEY=%%k
> "%LOCALAPPDATA%\ReportProgram\launch.key" echo %AIC_LOCAL_KEY%

cls
echo.
echo  ==============================================================
echo    On your phone, open:   http://%LANIP%:%PORT%
echo    Password:              (the one you just chose)
echo  ==============================================================
echo.
echo  If Windows asks whether to allow Python on your network,
echo  choose "Private networks" and click Allow.
echo  This computer opens the site without a password.
echo  Keep this window open; close it to stop.
echo.
start "" /b cmd /c "timeout /t 4 /nobreak >nul & start "" http://localhost:%PORT%/?key=%AIC_LOCAL_KEY%"
.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port %PORT%
