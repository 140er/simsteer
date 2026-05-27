<#
.SYNOPSIS
  Install the virtual output drivers SimSteer needs: ViGEm (gamepad) and
  vJoy (wheel). Self-elevates.

.DESCRIPTION
  SimSteer steers the game through a virtual controller. This installs
  both options so you can use either --device gamepad (ViGEm) or
  --device wheel (vJoy):

    - vJoy  : via winget (ShaulEizikovich.vJoyDeviceDriver).
    - ViGEm : downloaded from the latest nefarius/ViGEmBus GitHub release
              and run (ViGEm isn't on winget). A short GUI installer
              appears - click through it.
    - pyvjoy: pip-installed into .venv if a venv is present (only needed
              to run --device wheel from source).

  A REBOOT is recommended afterward so the kernel drivers load cleanly.

  Pair with tools\uninstall_drivers.ps1 to remove them again (e.g. if a
  real wheel like a Fanatec conflicts with the virtual devices).

.NOTES
  Run from an ordinary PowerShell; it re-launches itself elevated (one
  UAC prompt). Needs internet for the downloads.
#>
[CmdletBinding()]
param([switch]$SkipViGEm, [switch]$SkipVJoy)

$ErrorActionPreference = "Stop"

# --- self-elevate ---
$admin = ([Security.Principal.WindowsPrincipal]`
    [Security.Principal.WindowsIdentity]::GetCurrent()`
    ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $admin) {
    Write-Host "Re-launching elevated (approve the UAC prompt)..."
    $argList = @("-NoProfile","-ExecutionPolicy","Bypass","-File","`"$PSCommandPath`"")
    if ($SkipViGEm) { $argList += "-SkipViGEm" }
    if ($SkipVJoy)  { $argList += "-SkipVJoy" }
    Start-Process powershell -Verb RunAs -ArgumentList $argList
    return
}

Write-Host "=== SimSteer driver install ==="

# --- vJoy via winget ---
if (-not $SkipVJoy) {
    Write-Host ""
    Write-Host "[vJoy] installing via winget (ShaulEizikovich.vJoyDeviceDriver)..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install --id ShaulEizikovich.vJoyDeviceDriver --silent `
            --accept-package-agreements --accept-source-agreements
        Write-Host "[vJoy] winget exit code: $LASTEXITCODE"
    } else {
        Write-Warning "[vJoy] winget not available. Install manually from https://github.com/njz3/vJoy/releases"
    }
}

# --- ViGEm via GitHub latest release ---
if (-not $SkipViGEm) {
    Write-Host ""
    Write-Host "[ViGEm] fetching latest release from GitHub..."
    try {
        $rel = Invoke-RestMethod `
            -Uri "https://api.github.com/repos/nefarius/ViGEmBus/releases/latest" `
            -Headers @{ "User-Agent" = "SimSteer" }
        $asset = $rel.assets | Where-Object { $_.name -like "*.exe" } | Select-Object -First 1
        if (-not $asset) { throw "no .exe asset found in release $($rel.tag_name)" }
        $dst = Join-Path $env:TEMP $asset.name
        Write-Host "[ViGEm] downloading $($asset.name) ($([math]::Round($asset.size/1MB,1)) MB)..."
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $dst -UseBasicParsing
        Write-Host "[ViGEm] launching installer - CLICK THROUGH the Install button in the window."
        Start-Process $dst -Wait
        Write-Host "[ViGEm] installer closed."
    } catch {
        Write-Warning "[ViGEm] failed: $($_.Exception.Message)"
        Write-Host "[ViGEm] install manually from https://github.com/nefarius/ViGEmBus/releases"
    }
}

# --- pyvjoy into the project venv (run-from-source wheel mode) ---
$venvPip = Join-Path (Split-Path -Parent $PSScriptRoot) ".venv\Scripts\pip.exe"
if (Test-Path $venvPip) {
    Write-Host ""
    Write-Host "[pyvjoy] pip install into .venv (for --device wheel from source)..."
    & $venvPip install pyvjoy
} else {
    Write-Host ""
    Write-Host "[pyvjoy] .venv not found next to repo - skipping (only needed for run-from-source wheel mode)."
}

Write-Host ""
Write-Host "Done. REBOOT recommended so the drivers load cleanly."
Write-Host "Press Enter to close."
[void](Read-Host)
