@echo off
chcp 65001 >nul
echo.
echo  正在检查依赖...
pip install -q flask yfinance pandas requests lxml html5lib openpyxl xlrd
echo  依赖安装完成
echo.
echo  启动服务中...
start "" http://localhost:5001
python momentum_server.py
pause
