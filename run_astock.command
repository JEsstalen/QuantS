#!/bin/bash
cd "$(dirname "$0")"

echo "[1/3] Checking dependencies..."
pip3 install -q flask akshare pandas 2>/dev/null \
  || pip install -q flask akshare pandas

echo "[2/3] Stopping any existing server on port 5002..."
lsof -ti tcp:5002 | xargs kill -9 2>/dev/null
sleep 1

echo "[3/3] Launching...  http://localhost:5002"
(sleep 2 && open http://localhost:5002) &
python3 astock_server.py || python astock_server.py
