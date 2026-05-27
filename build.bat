@echo off
rem SimSteer PyInstaller build. Produces dist\SimSteer\ (onedir).
rem Run from anywhere — pushd handles the space in the repo path.

pushd "%~dp0"
if not exist ".venv\Scripts\pyinstaller.exe" (
    echo .venv\Scripts\pyinstaller.exe not found.
    echo Install with: ".venv\Scripts\pip install pyinstaller"
    popd
    exit /b 1
)

echo Cleaning previous build...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Running PyInstaller...
".venv\Scripts\pyinstaller.exe" --noconfirm simsteer.spec
set BUILD_RC=%ERRORLEVEL%

rem PyInstaller 6 puts bundled data files under _internal\. The driver
rem installers need to be next to SimSteer.exe so users can find/run
rem them, so surface them to the bundle root.
if not %BUILD_RC%==0 goto after_surface
for %%F in (install-drivers.bat uninstall-drivers.bat install_drivers.ps1 uninstall_drivers.ps1) do if exist "dist\SimSteer\_internal\%%F" move /y "dist\SimSteer\_internal\%%F" "dist\SimSteer\%%F" >nul
:after_surface
popd

if %BUILD_RC% neq 0 (
    echo.
    echo Build failed with code %BUILD_RC%.
    pause
    exit /b %BUILD_RC%
)

echo.
echo Build OK. Output: dist\SimSteer\
echo   - SimSteer.exe         (release, windowed)
echo   - SimSteer-debug.exe   (debug, console)
echo.
echo Next: zip the dist\SimSteer\ folder and upload to GitHub Releases.
