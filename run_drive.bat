@echo off
rem Double-clickable launcher for the closed-loop driving script.
rem Starts DISENGAGED — press SPACE in the overlay window to engage.
pushd "%~dp0"
".venv\Scripts\python.exe" -m pilot.main %*
set EXITCODE=%ERRORLEVEL%
popd
if %EXITCODE% neq 0 (
    echo.
    echo Exited with code %EXITCODE%.
    pause
)
