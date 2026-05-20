@echo off
echo.
echo  [1/3] Checking dependencies...
pip install -q flask akshare pandas
echo.
echo  [2/3] Stopping any existing server on port 5002...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5002 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 1 /nobreak >nul
echo.
echo  [3/3] Launching...  http://localhost:5002
start "" http://localhost:5002
python astock_server.py
pause
