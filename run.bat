@echo off
echo.
echo  [1/3] Checking dependencies...
pip install -q flask yfinance pandas requests lxml html5lib openpyxl xlrd
echo.
echo  [2/3] Stopping any existing server on port 5001...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5001 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | Sort-Object -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }" >nul 2>&1
timeout /t 1 /nobreak >nul
echo.
echo  [3/3] Launching...  http://localhost:5001
start "" http://localhost:5001
python momentum_server.py
pause
