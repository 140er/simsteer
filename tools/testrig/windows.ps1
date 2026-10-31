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
        $signals.Add('PendingFileRenameOperations')
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
$processes = @(Get-CimInstance Win32_Process | ForEach-Object {
    [pscustomobject]@{
        pid = [int]$_.ProcessId; parent = [int]$_.ParentProcessId; name = $_.Name
        path = $_.ExecutablePath; command = $_.CommandLine
        created = $(if ($_.CreationDate) { $_.CreationDate.ToFileTimeUtc().ToString() } else { $null })
    }
})
@{
    processes = $processes
    restart = @{
        checked_at = [DateTime]::UtcNow.ToString('o')
        signals = @($signals); tasks = $tasks; errors = @($errors)
    }
    errors = @()
} | ConvertTo-Json -Depth 6 -Compress
