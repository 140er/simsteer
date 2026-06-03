# Structural parity check (companion to tools/test_v2_parity.py).
#
# test_v2_parity.py proves the *pipeline* (warp->model->decode->overlay)
# is bit-identical on a real frame. This proves the rest: for each v1->v2
# ported module it shows every differing line, so the control + learner
# math (control.py / liveparams.py / livecalib.py) — which the numerical
# test doesn't exercise — can be confirmed to differ ONLY in import lines
# (and, for constants.py, an additive module-docstring note). Any logic
# line appearing here is a port regression.
#
#     pwsh tools/parity_structural.ps1
$ErrorActionPreference = "Stop"
$root = "F:\Comma ETS"

$pairs = @(
    @("pilot\constants.py",   "simsteer\core\constants.py"),
    @("pilot\warp.py",        "simsteer\core\warp.py"),
    @("pilot\preprocess.py",  "simsteer\core\preprocess.py"),
    @("pilot\postprocess.py", "simsteer\core\postprocess.py"),
    @("pilot\calibration.py", "simsteer\core\calibration.py"),
    @("pilot\model.py",       "simsteer\core\model.py"),
    @("pilot\controller.py",  "simsteer\core\control.py"),
    @("pilot\liveparams.py",  "simsteer\core\learners\liveparams.py"),
    @("pilot\livecalib.py",   "simsteer\core\learners\livecalib.py"),
    @("pilot\capture.py",     "simsteer\runtime\capture.py"),
    @("pilot\gamepad.py",     "simsteer\runtime\output\gamepad.py"),
    @("pilot\wheel.py",       "simsteer\runtime\output\wheel.py")
)

foreach ($p in $pairs) {
    $v1 = Join-Path $root $p[0]
    $v2 = Join-Path $root $p[1]
    if (-not (Test-Path $v1)) { Write-Output "MISSING v1: $($p[0])"; continue }
    if (-not (Test-Path $v2)) { Write-Output "MISSING v2: $($p[1])"; continue }
    $a = Get-Content $v1
    $b = Get-Content $v2
    $diff = Compare-Object $a $b
    $onlyV1 = @($diff | Where-Object { $_.SideIndicator -eq "<=" })
    $onlyV2 = @($diff | Where-Object { $_.SideIndicator -eq "=>" })
    Write-Output "==== $($p[0])  ->  $($p[1]) ===="
    Write-Output ("   lines: v1={0} v2={1}   differing: only-v1={2} only-v2={3}" -f $a.Count, $b.Count, $onlyV1.Count, $onlyV2.Count)
    foreach ($l in $onlyV1) { Write-Output ("   - v1: {0}" -f $l.InputObject.Trim()) }
    foreach ($l in $onlyV2) { Write-Output ("   + v2: {0}" -f $l.InputObject.Trim()) }
    Write-Output ""
}
