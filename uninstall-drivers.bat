@echo off
rem SimSteer - remove the ViGEm + vJoy drivers (e.g. when a real wheel
rem like a Fanatec conflicts with the virtual devices).
rem Double-click to run. Approve the UAC prompt.
setlocal
set "PS=%~dp0uninstall_drivers.ps1"
if not exist "%PS%" set "PS=%~dp0tools\uninstall_drivers.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS%"
