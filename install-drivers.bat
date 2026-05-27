@echo off
rem SimSteer - install the ViGEm (gamepad) + vJoy (wheel) drivers.
rem Double-click to run. Approve the UAC prompt(s).
setlocal
set "PS=%~dp0install_drivers.ps1"
if not exist "%PS%" set "PS=%~dp0tools\install_drivers.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS%"
