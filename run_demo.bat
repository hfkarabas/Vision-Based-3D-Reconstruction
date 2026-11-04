@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Follow README setup instructions first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m vision3d.demo
if errorlevel 1 pause
