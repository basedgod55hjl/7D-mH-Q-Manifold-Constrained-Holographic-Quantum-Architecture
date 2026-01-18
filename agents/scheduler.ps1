# 7D Crystal System - Windows Task Scheduler Integration
# Creates scheduled tasks for autonomous operation
# Created for Sir Charles Spikes - January 2026

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Status,
    [string]$Interval = "15min"  # 15min, 1hour, 4hour, daily
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TaskName = "7D_Crystal_AutoBuild"

function Write-Status($msg, $color = "Cyan") {
    Write-Host "[SCHEDULER] $msg" -ForegroundColor $color
}

function Install-ScheduledTask {
    Write-Status "Installing scheduled task: $TaskName" "Yellow"
    
    # Remove existing task if present
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    
    # Determine interval
    $trigger = switch ($Interval) {
        "15min" { New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) }
        "1hour" { New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) }
        "4hour" { New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 4) }
        "daily" { New-ScheduledTaskTrigger -Daily -At "3:00AM" }
        default { New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15) }
    }
    
    $agentPath = Join-Path $PSScriptRoot "autonomous_agent.ps1"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -File `"$agentPath`" -Once"
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
    
    Register-ScheduledTask -TaskName $TaskName -Trigger $trigger -Action $action -Settings $settings -Principal $principal -Force
    
    Write-Status "Task installed successfully!" "Green"
    Write-Status "Interval: $Interval" "Gray"
    Write-Status "Agent: $agentPath" "Gray"
}

function Uninstall-ScheduledTask {
    Write-Status "Removing scheduled task: $TaskName" "Yellow"
    
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Status "Task removed successfully!" "Green"
    } else {
        Write-Status "Task not found" "Yellow"
    }
}

function Show-TaskStatus {
    Write-Host @"

    ╔═══════════════════════════════════════════════════════════╗
    ║      7D CRYSTAL SCHEDULED TASK STATUS                     ║
    ╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    
    if ($task) {
        $info = Get-ScheduledTaskInfo -TaskName $TaskName
        
        Write-Host "Task Name:    $TaskName" -ForegroundColor White
        Write-Host "State:        $($task.State)" -ForegroundColor $(if ($task.State -eq "Ready") { "Green" } else { "Yellow" })
        Write-Host "Last Run:     $($info.LastRunTime)" -ForegroundColor Gray
        Write-Host "Last Result:  $($info.LastTaskResult)" -ForegroundColor $(if ($info.LastTaskResult -eq 0) { "Green" } else { "Red" })
        Write-Host "Next Run:     $($info.NextRunTime)" -ForegroundColor Cyan
        
        # Show trigger info
        $triggers = $task.Triggers
        foreach ($t in $triggers) {
            Write-Host "Trigger:      $($t.Repetition.Interval)" -ForegroundColor Gray
        }
    } else {
        Write-Host "Status: NOT INSTALLED" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Use -Install to create the scheduled task" -ForegroundColor Gray
    }
}

# Main execution
Write-Host @"

    ╔═══════════════════════════════════════════════════════════╗
    ║     7D CRYSTAL SYSTEM - TASK SCHEDULER                    ║
    ║                                                           ║
    ║   Windows Task Scheduler Integration                      ║
    ║   Discovered by Sir Charles Spikes                        ║
    ╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

if ($Install) {
    Install-ScheduledTask
} elseif ($Uninstall) {
    Uninstall-ScheduledTask
} elseif ($Status) {
    Show-TaskStatus
} else {
    Show-TaskStatus
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  -Install            Create scheduled task" -ForegroundColor Gray
    Write-Host "  -Uninstall          Remove scheduled task" -ForegroundColor Gray
    Write-Host "  -Status             Show task status" -ForegroundColor Gray
    Write-Host "  -Interval <time>    Set interval (15min, 1hour, 4hour, daily)" -ForegroundColor Gray
}
