# Run SimSteer from source, for machines where Windows blocks SimSteer.exe
# (Smart App Control). First run: creates .venv with uv and Python 3.11,
# installs SimSteer and fetches the models. Later runs reuse all of it.
# Arguments pass through: .\run-from-source.ps1 --list-games
#
# Uses cmdlets only, no .NET calls, so it also runs when Windows starts
# an untrusted script in Constrained Language Mode.
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$py = Join-Path $venv 'Scripts\python.exe'
$stamp = Join-Path $venv 'simsteer-install.txt'
$uvInstall = 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'

function Fail([string]$Message) {
    Write-Host "run-from-source: $Message"
    exit 1
}

# Runs a native command. A program Windows refuses to start throws
# instead of setting $LASTEXITCODE, so turn that into a clear message.
function Invoke-Native([string]$What, [scriptblock]$Body) {
    try {
        & $Body
    } catch {
        Fail ("$What could not start: $($_.Exception.Message)`n" +
            'If Windows blocked it, the Microsoft-Windows-CodeIntegrity/Operational ' +
            'event log in Event Viewer names the file.')
    }
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit code $LASTEXITCODE)." }
}

function Find-Uv {
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    # Where the official installer puts uv; not on PATH until a new terminal.
    $default = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path $default) { return $default }
    return $null
}

$uv = Find-Uv
if (-not $uv) {
    Write-Host 'run-from-source: uv not found. It sets up Python and the packages SimSteer needs.'
    Write-Host "Official install command (https://docs.astral.sh/uv/):`n  $uvInstall"
    $answer = Read-Host 'Install uv now? [y/N]'
    if ($answer -notmatch '^(y|yes)$') { Fail 'install uv with the command above, then run this again.' }
    Invoke-Native 'The uv installer' { powershell -NoProfile -ExecutionPolicy ByPass -c 'irm https://astral.sh/uv/install.ps1 | iex' }
    $uv = Find-Uv
    if (-not $uv) { Fail 'uv was installed but not found. Open a new terminal and run this again.' }
}

if (Test-Path $py) {
    $ok = $false
    try { & $py -c 'import sys; sys.exit(sys.version_info < (3, 11))'; $ok = $LASTEXITCODE -eq 0 } catch { }
    if (-not $ok) {
        Fail "the existing .venv does not run Python 3.11 or newer. Delete the .venv folder and run this again."
    }
} else {
    Write-Host 'Creating .venv with Python 3.11...'
    Invoke-Native 'uv venv' { & $uv venv -q -p 3.11 $venv }
}

# Reinstall only when pyproject.toml changed, so later runs start fast
# and work offline.
$want = (Get-FileHash (Join-Path $root 'pyproject.toml')).Hash
$have = if (Test-Path $stamp) { (Get-Content $stamp -Raw).Trim() } else { '' }
if ($have -ne $want) {
    Write-Host 'Installing SimSteer and its packages (the first time takes a few minutes)...'
    Invoke-Native 'uv pip install' { & $uv pip install -p $py -e $root }
    Set-Content -Path $stamp -Value $want
}

$models = Join-Path $root 'models'
if (-not ((Test-Path (Join-Path $models 'driving_vision.onnx')) -and
          (Test-Path (Join-Path $models 'driving_policy.onnx')))) {
    Write-Host 'Fetching the driving models (about 60 MB, once)...'
    Invoke-Native 'tools\fetch_model.py' { & $py (Join-Path $root 'tools\fetch_model.py') }
}

try {
    & $py -m simsteer @args
} catch {
    Fail "SimSteer could not start: $($_.Exception.Message)"
}
exit $LASTEXITCODE
