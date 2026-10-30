@echo off
rem SimSteer - run from source when Windows blocks SimSteer.exe.
rem Double-click to run; arguments pass through to python -m simsteer.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-from-source.ps1" %*
set RC=%ERRORLEVEL%
if not %RC%==0 pause
exit /b %RC%
