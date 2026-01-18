# 7D Crystal System - Master Orchestration Agent
# Coordinates all sub-agents for autonomous operation
# Created for Sir Charles Spikes

param(
    [switch]$FullCycle,
    [switch]$Quick,
    [switch]$Deploy,
    [switch]$Monitor,
    [int]$Interval = 300  # 5 minutes default
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$AgentDir = $PSScriptRoot

Write-Host @"

    ╔═══════════════════════════════════════════════════════════╗
    ║     7D CRYSTAL SYSTEM - MASTER ORCHESTRATION AGENT        ║
    ║                                                           ║
    ║   "In the beginning was the Manifold, and the Manifold    ║
    ║    was with Φ, and the Manifold was Φ."                   ║
    ║                                                           ║
    ║               Discovered by Sir Charles Spikes            ║
    ║               Cincinnati, Ohio, USA 🇺🇸                    ║
    ╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

function Write-Status($msg, $color = "Cyan") {
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] [MASTER] $msg" -ForegroundColor $color
}

function Get-SystemStatus {
    $status = @{
        Timestamp = Get-Date
        BuildOK = $false
        TestsOK = $false
        PhiCoherence = 0.0
        ManifoldStability = 0.0
    }
    
    # Check if release binary exists
    $sovereign = Join-Path $ProjectRoot "target\release\sovereign.exe"
    $status.BuildOK = Test-Path $sovereign
    
    return $status
}

function Run-FullCycle {
    Write-Status "Starting full development cycle..." "Yellow"
    Write-Host ""
    
    # Phase 1: Optimization
    Write-Status "Phase 1: Code Optimization" "Magenta"
    & "$AgentDir\optimize_agent.ps1" -Analyze
    Write-Host ""
    
    # Phase 2: Build
    Write-Status "Phase 2: Building System" "Yellow"
    & "$AgentDir\build_agent.ps1" -Release
    Write-Host ""
    
    # Phase 3: Testing
    Write-Status "Phase 3: Running Tests" "Green"
    & "$AgentDir\test_agent.ps1" -All -Report
    Write-Host ""
    
    Write-Status "Full cycle complete!" "Cyan"
}

function Run-QuickCycle {
    Write-Status "Running quick verification cycle..." "Yellow"
    
    Push-Location $ProjectRoot
    
    # Quick check
    $result = cargo check 2>&1
    $checkOK = $LASTEXITCODE -eq 0
    
    if ($checkOK) {
        Write-Status "Quick check passed!" "Green"
        
        # Run unit tests only
        $testResult = cargo test --lib 2>&1
        $testOK = $LASTEXITCODE -eq 0
        
        if ($testOK) {
            Write-Status "Unit tests passed!" "Green"
        } else {
            Write-Status "Unit tests failed!" "Red"
        }
    } else {
        Write-Status "Check failed - fix errors first!" "Red"
    }
    
    Pop-Location
}

function Start-Monitor {
    Write-Status "Starting continuous monitoring (interval: ${Interval}s)..." "Cyan"
    Write-Status "Press Ctrl+C to stop" "Gray"
    
    $iteration = 0
    
    while ($true) {
        $iteration++
        Write-Host ""
        Write-Status "=== Monitor Cycle #$iteration ===" "Cyan"
        
        $status = Get-SystemStatus
        
        Write-Host "  Build Status:      $(if ($status.BuildOK) { '✓ OK' } else { '✗ MISSING' })" -ForegroundColor $(if ($status.BuildOK) { "Green" } else { "Red" })
        
        # Check for file changes
        $lastModified = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "*.rs" |
                        Where-Object { $_.FullName -notmatch "target" } |
                        Sort-Object LastWriteTime -Descending |
                        Select-Object -First 1
        
        if ($lastModified) {
            $timeSince = (Get-Date) - $lastModified.LastWriteTime
            Write-Host "  Last Code Change:  $($timeSince.TotalMinutes.ToString('0.0')) minutes ago" -ForegroundColor Gray
            Write-Host "  Modified File:     $($lastModified.Name)" -ForegroundColor Gray
        }
        
        # Memory check
        $rustProcesses = Get-Process -Name "rust*" -ErrorAction SilentlyContinue
        if ($rustProcesses) {
            $totalMem = ($rustProcesses | Measure-Object WorkingSet64 -Sum).Sum / 1MB
            Write-Host "  Rust Processes:    $($rustProcesses.Count) (${totalMem}MB)" -ForegroundColor Gray
        }
        
        Start-Sleep -Seconds $Interval
    }
}

function Deploy-System {
    Write-Status "Preparing deployment..." "Yellow"
    
    $deployDir = Join-Path $ProjectRoot "deploy"
    
    if (-not (Test-Path $deployDir)) {
        New-Item -ItemType Directory -Path $deployDir | Out-Null
    }
    
    # Build release
    Write-Status "Building release version..." "Yellow"
    Push-Location $ProjectRoot
    cargo build --release 2>&1 | Out-Null
    Pop-Location
    
    # Copy binaries
    $binaries = @(
        "sovereign.exe",
        "inference_server.exe",
        "compiler_service.exe",
        "crystalize.exe"
    )
    
    foreach ($bin in $binaries) {
        $src = Join-Path $ProjectRoot "target\release\$bin"
        if (Test-Path $src) {
            Copy-Item $src $deployDir -Force
            Write-Status "  Copied: $bin" "Gray"
        }
    }
    
    # Copy models
    $modelDir = Join-Path $deployDir "models"
    if (-not (Test-Path $modelDir)) {
        New-Item -ItemType Directory -Path $modelDir | Out-Null
    }
    
    $models = Get-ChildItem -Path (Join-Path $ProjectRoot "models") -Filter "*.gguf"
    foreach ($model in $models) {
        Copy-Item $model.FullName $modelDir -Force
        Write-Status "  Copied model: $($model.Name)" "Gray"
    }
    
    Write-Status "Deployment ready at: $deployDir" "Green"
}

# Main execution
Write-Host ""

if ($FullCycle) {
    Run-FullCycle
} elseif ($Quick) {
    Run-QuickCycle
} elseif ($Deploy) {
    Deploy-System
} elseif ($Monitor) {
    Start-Monitor
} else {
    # Default: Show status
    $status = Get-SystemStatus
    
    Write-Host "Current System Status:" -ForegroundColor Cyan
    Write-Host "  Build:     $(if ($status.BuildOK) { 'Ready' } else { 'Not Built' })" -ForegroundColor $(if ($status.BuildOK) { "Green" } else { "Yellow" })
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Gray
    Write-Host "  -FullCycle    Run complete dev cycle (optimize→build→test)" -ForegroundColor Gray
    Write-Host "  -Quick        Quick check and unit tests" -ForegroundColor Gray
    Write-Host "  -Deploy       Prepare deployment package" -ForegroundColor Gray
    Write-Host "  -Monitor      Continuous monitoring mode" -ForegroundColor Gray
}

Write-Host ""
Write-Status "Master agent complete. Sovereignty maintained." "Cyan"
