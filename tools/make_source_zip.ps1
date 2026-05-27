<#
.SYNOPSIS
  Build a clean, shareable source zip of SimSteer.

.DESCRIPTION
  Uses `git archive HEAD`, so ONLY git-tracked files are included. That
  automatically excludes everything that should not be shared - .venv\,
  dist\, build\, models\*.onnx, .claude\, and per-user state files
  (liveparams_*.json, livecalib_state_*.json, first_drive_done_*.flag) -
  because those are all gitignored / untracked.

  The recipient extracts the zip and follows docs\SETUP.md: create a
  venv, `pip install -r requirements.txt`, then `python
  tools\fetch_model.py` to download the ONNX models (not bundled - they
  come from comma.ai's repo).

  Commit your work first; the archive reflects HEAD, not the working
  tree.

.PARAMETER IncludeModels
  Also bundle models\ (~59 MB) so the recipient can skip fetch_model.py.
  Off by default to keep the zip small.

.EXAMPLE
  .\tools\make_source_zip.ps1
  .\tools\make_source_zip.ps1 -IncludeModels
#>
param([switch]$IncludeModels)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $verLine = Select-String -Path "pilot\version.py" `
        -Pattern '__version__\s*=\s*"([^"]+)"'
    $ver = if ($verLine) { $verLine.Matches[0].Groups[1].Value } else { "dev" }
    $out = Join-Path $root "SimSteer-src-$ver.zip"
    if (Test-Path $out) { Remove-Item $out -Force }

    $name = Split-Path $out -Leaf
    Write-Host "Archiving git-tracked source at HEAD -> $name"
    git archive --format=zip --output "$out" HEAD
    if ($LASTEXITCODE -ne 0) { throw "git archive failed (exit $LASTEXITCODE)" }

    if ($IncludeModels) {
        if (Test-Path "models") {
            Write-Host "Adding models\ (bundled so recipient can skip fetch_model.py)"
            Compress-Archive -Path "models" -DestinationPath "$out" -Update
        } else {
            Write-Warning "models\ not found - run tools\fetch_model.py first, or omit -IncludeModels."
        }
    }

    $mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
    Write-Host ""
    Write-Host "Done: $name ($mb MB)"
    Write-Host "Recipient: extract, then follow docs\SETUP.md."
    if (-not $IncludeModels) {
        Write-Host "Models are NOT in this zip - recipient runs: python tools\fetch_model.py"
    }
}
finally {
    Pop-Location
}
