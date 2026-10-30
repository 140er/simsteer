# Local check: lint, tests, CLI smoke. Creates .venv with uv on first run.
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:PYTHONDONTWRITEBYTECODE = '1'

function Invoke-Step([string]$Name, [scriptblock]$Body) {
    Write-Host "== $Name"
    & $Body
    if ($LASTEXITCODE -ne 0) { Write-Host "check.ps1: $Name failed"; exit $LASTEXITCODE }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'check.ps1: uv not found. Install it from https://docs.astral.sh/uv/'
    exit 2
}
$py = '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Invoke-Step 'venv' { uv venv -q -p 3.11 .venv } }
Invoke-Step 'install' { uv pip install -q -p $py -e '.[dev]' }

Invoke-Step 'ruff' { & $py -m ruff check --no-cache . }
Invoke-Step 'pytest' { & $py -m pytest tests -q -p no:cacheprovider }
Invoke-Step 'simsteer --help' { & $py -m simsteer --help }
Invoke-Step 'windows imports' { & $py -c "import dxcam, vgamepad; print('imported: dxcam, vgamepad')" }
