#!/bin/bash
cd "$(dirname "$0")"

echo "[1/3] Checking dependencies..."
pip3 install -q flask yfinance pandas requests lxml html5lib openpyxl xlrd 2>/dev/null \
  || pip install -q flask yfinance pandas requests lxml html5lib openpyxl xlrd

echo "[2/3] Stopping any existing server on port 5001..."
lsof -ti tcp:5001 | xargs kill -9 2>/dev/null
sleep 1

echo "[3/3] Launching...  http://localhost:5001"
# Open browser after a short delay to let Flask bind
(sleep 2 && open http://localhost:5001) &
python3 momentum_server.py || python momentum_server.py
