param(
    [ValidateSet('inventory', 'device-state', 'device-enable', 'device-disable')]
    [string]$Action = 'inventory',
    [string]$DeviceId
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
if ($Action -ne 'inventory') {
    if (-not $DeviceId) { throw 'A virtual display instance ID is required' }
    if ($DeviceId.IndexOfAny([char[]]'*?[]') -ge 0) { throw 'Virtual display instance IDs must not contain wildcards' }
    $device = Get-PnpDevice -InstanceId $DeviceId
    if (@($device).Count -ne 1 -or $device.InstanceId -ine $DeviceId) { throw 'Select exactly one virtual display instance ID' }
    if ($device.Class -ne 'Display' -or $DeviceId -notmatch '^(ROOT|SWD)\\') {
        throw 'Only an explicitly selected virtual display adapter is supported'
    }
    if ($Action -eq 'device-enable') {
        Enable-PnpDevice -InstanceId $DeviceId -Confirm:$false
    }
    if ($Action -eq 'device-disable') {
        Disable-PnpDevice -InstanceId $DeviceId -Confirm:$false
    }
    $problem = Get-PnpDeviceProperty -InstanceId $DeviceId -KeyName 'DEVPKEY_Device_ProblemCode'
    @{ problem = [int]$problem.Data; instance = $DeviceId } | ConvertTo-Json -Compress
    exit 0
}
$errors = [System.Collections.Generic.List[string]]::new()
$signals = [System.Collections.Generic.List[string]]::new()
$warnings = [System.Collections.Generic.List[string]]::new()
$tasks = @()
try {
    foreach ($item in @(
        @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired', 'RebootRequired'),
        @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending', 'RebootPending')
    )) {
        if (Test-Path -LiteralPath $item[0]) { $signals.Add($item[1]) }
    }
    $manager = Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager'
    if (@($manager.PendingFileRenameOperations | Where-Object { $_ }).Count) {
        $warnings.Add('PendingFileRenameOperations is set; restart may be needed after installation')
    }
    foreach ($path in @('HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UpdatePolicy\Settings',
                        'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings',
                        'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\UX\StateVariables')) {
        if (Test-Path -LiteralPath $path) {
            $key = Get-ItemProperty -LiteralPath $path
            if ($key.RebootRequired -and $key.RebootRequired -ne '0') { $signals.Add('UpdateOrchestrator RebootRequired') }
            foreach ($name in @('ScheduledRebootTime', 'ScheduledRebootTimeUTC', 'RebootDeadline', 'RebootDeadlineUTC')) {
                if ($key.$name) { $signals.Add('UpdateOrchestrator ' + $name) }
            }
        }
    }
} catch { $errors.Add('restart registry: ' + $_.Exception.Message) }
try {
    $tasks = @(Get-ScheduledTask -TaskPath '\Microsoft\Windows\UpdateOrchestrator\' | Where-Object {
        $_.TaskName -match 'Reboot'
    } | ForEach-Object {
        $info = $_ | Get-ScheduledTaskInfo
        [pscustomobject]@{
            name = $_.TaskName; state = [string]$_.State
            next_run = $(if ($info.NextRunTime -and $info.NextRunTime.Year -gt 2000) {
                $info.NextRunTime.ToUniversalTime().ToString('o')
            } else { $null })
        }
    })
} catch { $errors.Add('UpdateOrchestrator: ' + $_.Exception.Message) }
$allProcesses = @(Get-CimInstance Win32_Process)
$captureIds = @($allProcesses | Where-Object {
    $_.Name -match '^(obs(32|64)?|streamlabs (obs|desktop)|nvidia (share|overlay)|nvsphelper64|gamebar|gamebarftserver|bcastdvr|discord|teams|ms-teams)\.exe$'
} | ForEach-Object { [int]$_.ProcessId })
$captureServices = @{}
try {
    Get-CimInstance Win32_Service -Filter "Name LIKE 'BcastDVRUserService%'" | Where-Object {
        $_.State -eq 'Running' -and $_.ProcessId -gt 0
    } | ForEach-Object {
        $captureServices[[int]$_.ProcessId] = 'Game Bar capture service'
        $captureIds += [int]$_.ProcessId
    }
} catch { $warnings.Add('Game Bar capture service counters are not observable') }
$cpu = @{}
$encoder = @{}
if ($captureIds.Count) {
    try {
        Get-CimInstance Win32_PerfFormattedData_PerfProc_Process | Where-Object { $captureIds -contains [int]$_.IDProcess } | ForEach-Object {
            $cpu[[int]$_.IDProcess] = [double]$_.PercentProcessorTime
        }
    } catch { $warnings.Add('Capture CPU counters are not observable') }
    try {
        Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine | Where-Object {
            $_.Name -match 'pid_(\d+)_.*engtype_VideoEncode' -and [double]$_.UtilizationPercentage -gt 0
        } | ForEach-Object {
            if ($_.Name -match 'pid_(\d+)_') { $encoder[[int]$Matches[1]] = $true }
        }
    } catch { $warnings.Add('Capture encoder counters are not observable') }
}
$bootId = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToFileTimeUtc().ToString()
$processes = @($allProcesses | ForEach-Object {
    [pscustomobject]@{
        pid = [int]$_.ProcessId; parent = [int]$_.ParentProcessId; name = $_.Name
        path = $_.ExecutablePath; command = $_.CommandLine
        created = $(if ($_.CreationDate) { $_.CreationDate.ToFileTimeUtc().ToString() } else { $null })
        cpu_percent = $cpu[[int]$_.ProcessId]; encoder_active = [bool]$encoder[[int]$_.ProcessId]
        capture_tool = $captureServices[[int]$_.ProcessId]
    }
})
@{
    processes = $processes
    boot_id = $bootId
    warnings = @($warnings)
    restart = @{
        checked_at = [DateTime]::UtcNow.ToString('o')
        signals = @($signals); warnings = @($warnings); tasks = $tasks; errors = @($errors)
    }
    errors = @()
} | ConvertTo-Json -Depth 6 -Compress
