<#
.SYNOPSIS
  Remove the ViGEm + vJoy virtual drivers. Self-elevates.

.DESCRIPTION
  Finds every installed product whose name matches vJoy / ViGEm /
  Nefarius in the Windows uninstall registry and runs its silent
  uninstaller. Works regardless of how they were installed (winget,
  MSI, or the GitHub installer).

  Use this when a real wheel (e.g. a Fanatec) misbehaves because the
  always-present virtual devices confuse the game's controller
  enumeration. Re-add them later with tools\install_drivers.ps1.

  A REBOOT clears the leftover driver service stubs (they linger as
  "Stopped" until then - harmless).

.NOTES
  Run from an ordinary PowerShell; it re-launches itself elevated (one
  UAC prompt).
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

# --- self-elevate ---
$admin = ([Security.Principal.WindowsPrincipal]`
    [Security.Principal.WindowsIdentity]::GetCurrent()`
    ).IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)
if (-not $admin) {
    Write-Host "Re-launching elevated (approve the UAC prompt)..."
    Start-Process powershell -Verb RunAs -ArgumentList `
        "-NoProfile","-ExecutionPolicy","Bypass","-File","`"$PSCommandPath`""
    return
}

Write-Host "=== SimSteer driver uninstall ==="

$paths = @(
    "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
)
$targets = Get-ItemProperty $paths -ErrorAction SilentlyContinue |
    Where-Object { $_.DisplayName -match "vJoy|ViGEm|Nefarius" }

if (-not $targets) {
    Write-Host "Nothing to do - no vJoy / ViGEm entries found in the registry."
    Write-Host "Press Enter to close."
    [void](Read-Host)
    return
}

foreach ($t in $targets) {
    Write-Host ""
    Write-Host "Uninstalling: $($t.DisplayName) $($t.DisplayVersion)"
    try {
        if ($t.UninstallString -match "MsiExec\.exe.*(\{[0-9A-Fa-f-]+\})") {
            # MSI product (e.g. ViGEm): /X{GUID} quiet.
            $guid = $Matches[1]
            $p = Start-Process "msiexec.exe" `
                -ArgumentList "/X$guid","/quiet","/norestart" -Wait -PassThru
            Write-Host "  msiexec exit: $($p.ExitCode)"
        } elseif ($t.QuietUninstallString) {
            # Has a documented silent uninstaller (e.g. vJoy Inno setup).
            $p = Start-Process "cmd.exe" `
                -ArgumentList "/c", $t.QuietUninstallString -Wait -PassThru
            Write-Host "  exit: $($p.ExitCode)"
        } elseif ($t.UninstallString -match "unins\d+\.exe") {
            # Inno Setup uninstaller without a quiet string - add flags.
            $exe = ($t.UninstallString -replace '"', '')
            $p = Start-Process $exe `
                -ArgumentList "/VERYSILENT","/SUPPRESSMSGBOXES","/NORESTART" -Wait -PassThru
            Write-Host "  exit: $($p.ExitCode)"
        } else {
            Write-Warning "  no silent uninstall known - running interactively: $($t.UninstallString)"
            Start-Process "cmd.exe" -ArgumentList "/c", $t.UninstallString -Wait
        }
    } catch {
        Write-Warning "  failed: $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "Done. REBOOT to clear the leftover driver service stubs."
Write-Host "Press Enter to close."
[void](Read-Host)
