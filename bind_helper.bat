@echo off
rem Double-clickable binding helper for ETS2 (or any game with a "press the
rem input you want to bind" prompt). Picks an axis/button and wiggles it
rem so the game can detect it.
pushd "%~dp0"
".venv\Scripts\python.exe" -m tools.bind_helper
set EXITCODE=%ERRORLEVEL%
popd
if %EXITCODE% neq 0 (
    echo.
    echo Exited with code %EXITCODE%.
    pause
)
