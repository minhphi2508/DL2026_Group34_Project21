@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run SETUP_CPU.cmd or SETUP_RTX.cmd first.
  pause
  exit /b 1
)
if "%~1"=="" (
  ".venv\Scripts\python.exe" restore.py --input inputs --device auto
) else (
  ".venv\Scripts\python.exe" restore.py --input "%~1" --device auto
)
if errorlevel 1 echo Restoration reported an error. See the message and output logs.
pause
