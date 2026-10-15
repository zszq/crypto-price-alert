@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 未找到虚拟环境，请先按照 README.md 安装依赖。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m price_alert.cli run --config config/launch-trade.yaml
if errorlevel 1 pause
