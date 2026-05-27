@echo off
rem Double-clickable launcher for the debug overlay.
rem %~dp0 = the directory this .bat file lives in, with trailing backslash.
pushd "%~dp0"
".venv\Scripts\python.exe" -m debug.overlay %*
set EXITCODE=%ERRORLEVEL%
popd
if %EXITCODE% neq 0 (
    echo.
    echo Exited with code %EXITCODE%.
    pause
)
