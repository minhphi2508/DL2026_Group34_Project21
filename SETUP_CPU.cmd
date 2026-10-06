@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python setup_windows.py --device cpu
) else (
  py -3.12 setup_windows.py --device cpu
)
if errorlevel 1 echo Setup failed. Keep the error text for troubleshooting.
pause
