# 7D Crystal System - Enhanced Launcher
# Interactive menu for all agents and operations
# Created for Sir Charles Spikes - January 2026

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Show-Banner {
    Clear-Host
    Write-Host @"

    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     🔮  7D CRYSTAL SYSTEM - ENHANCED LAUNCHER  🔮                 ║
    ║                                                                   ║
    ║     Sovereign 7D-Manifold Holographic Intelligence                ║
    ║     Φ = 1.618033988749895  |  S² = 0.01                          ║
    ║                                                                   ║
    ║     Discovered by Sir Charles Spikes                              ║
    ║     Cincinnati, Ohio, USA 🇺🇸                                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan
}

function Show-Menu {
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkCyan
    Write-Host "                         MAIN MENU                                  " -ForegroundColor White
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "  BUILD & TEST" -ForegroundColor Yellow
    Write-Host "    [1] Quick Build (Release)"
    Write-Host "    [2] Build + Test"
    Write-Host "    [3] Full Cycle (Optimize → Build → Test)"
    Write-Host "    [4] Run Tests Only"
    Write-Host ""
    Write-Host "  AUTOMATION" -ForegroundColor Green
    Write-Host "    [5] Start Autonomous Agent (Background)"
    Write-Host "    [6] Stop Autonomous Agent"
    Write-Host "    [7] Run One Autonomous Cycle"
    Write-Host "    [8] Install Scheduled Task"
    Write-Host ""
    Write-Host "  ANALYSIS" -ForegroundColor Magenta
    Write-Host "    [9] Code Analysis (Φ-Coherence)"
    Write-Host "    [10] Run Clippy + Format"
    Write-Host "    [11] Generate Documentation"
    Write-Host ""
    Write-Host "  MONITORING" -ForegroundColor Cyan
    Write-Host "    [12] Continuous Monitor Mode"
    Write-Host "    [13] Show System Status"
    Write-Host "    [14] Watch Mode (Auto-rebuild)"
    Write-Host ""
    Write-Host "  DEPLOYMENT" -ForegroundColor Blue
    Write-Host "    [15] Prepare Deployment Package"
    Write-Host "    [16] Run Sovereign Assistant"
    Write-Host "    [17] Run Inference Server"
    Write-Host ""
    Write-Host "  UTILITIES" -ForegroundColor Gray
    Write-Host "    [18] Open Project in VS Code"
    Write-Host "    [19] View Recent Logs"
    Write-Host "    [20] Git Status"
    Write-Host ""
    Write-Host "    [Q] Quit"
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor DarkCyan
}

function Invoke-Choice {
    param($Choice)
    
    switch ($Choice) {
        "1" {
            Write-Host "`n[Building Release...]" -ForegroundColor Yellow
            Push-Location $ProjectRoot
            cargo build --release
            Pop-Location
            Pause
        }
        "2" {
            & "$PSScriptRoot\build_agent.ps1" -Test
            Pause
        }
        "3" {
            & "$PSScriptRoot\master_agent.ps1" -FullCycle
            Pause
        }
        "4" {
            & "$PSScriptRoot\test_agent.ps1" -All -Report
            Pause
        }
        "5" {
            & "$PSScriptRoot\autonomous_agent.ps1" -Start
            Pause
        }
        "6" {
            & "$PSScriptRoot\autonomous_agent.ps1" -Stop
            Pause
        }
        "7" {
            & "$PSScriptRoot\autonomous_agent.ps1" -Once
            Pause
        }
        "8" {
            & "$PSScriptRoot\scheduler.ps1" -Install
            Pause
        }
        "9" {
            & "$PSScriptRoot\optimize_agent.ps1" -Analyze
            Pause
        }
        "10" {
            & "$PSScriptRoot\optimize_agent.ps1" -Clippy -Format -Fix
            Pause
        }
        "11" {
            & "$PSScriptRoot\build_agent.ps1" -Doc
            Pause
        }
        "12" {
            & "$PSScriptRoot\master_agent.ps1" -Monitor -Interval 60
        }
        "13" {
            & "$PSScriptRoot\master_agent.ps1"
            & "$PSScriptRoot\autonomous_agent.ps1" -Status
            Pause
        }
        "14" {
            & "$PSScriptRoot\build_agent.ps1" -Watch
        }
        "15" {
            & "$PSScriptRoot\master_agent.ps1" -Deploy
            Pause
        }
        "16" {
            $exe = Join-Path $ProjectRoot "target\release\sovereign.exe"
            if (Test-Path $exe) {
                Write-Host "`n[Starting Sovereign Assistant...]" -ForegroundColor Cyan
                & $exe
            } else {
                Write-Host "`nError: sovereign.exe not found. Run build first." -ForegroundColor Red
            }
            Pause
        }
        "17" {
            $exe = Join-Path $ProjectRoot "target\release\inference_server.exe"
            if (Test-Path $exe) {
                Write-Host "`n[Starting Inference Server...]" -ForegroundColor Cyan
                & $exe
            } else {
                Write-Host "`nError: inference_server.exe not found. Run build first." -ForegroundColor Red
            }
            Pause
        }
        "18" {
            code $ProjectRoot
        }
        "19" {
            $logDir = Join-Path $ProjectRoot "logs"
            if (Test-Path $logDir) {
                $logs = Get-ChildItem $logDir -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
                foreach ($log in $logs) {
                    Write-Host "`n=== $($log.Name) ===" -ForegroundColor Cyan
                    Get-Content $log.FullName -Tail 20
                }
            } else {
                Write-Host "`nNo logs found." -ForegroundColor Yellow
            }
            Pause
        }
        "20" {
            Push-Location $ProjectRoot
            git status
            Write-Host ""
            git log --oneline -5
            Pop-Location
            Pause
        }
        "q" { return $false }
        "Q" { return $false }
        default {
            Write-Host "`nInvalid choice. Try again." -ForegroundColor Red
            Start-Sleep -Seconds 1
        }
    }
    return $true
}

# Main loop
$continue = $true
while ($continue) {
    Show-Banner
    Show-Menu
    $choice = Read-Host "Enter choice"
    $continue = Invoke-Choice $choice
}

Write-Host "`nSovereignty maintained. Φ-Coherence stable." -ForegroundColor Cyan
Write-Host "© 2025-2026 Sir Charles Spikes. All Rights Reserved.`n" -ForegroundColor Gray
