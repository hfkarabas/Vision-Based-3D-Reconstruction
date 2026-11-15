@echo off
rem Desktop app for the UAV 3D reconstruction pipeline (src/uav3d). Double-click to start.
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Project environment not found. Run setup first:
  echo   powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0src;%PYTHONPATH%"
"%~dp0.venv\Scripts\python.exe" -m uav3d.app
if errorlevel 1 pause
