# 7D Crystal System - Autonomous Agent
# Auto-builds, tests, and optimizes in the background
# Created by Sir Charles Spikes - January 2026

param(
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Once,
    [int]$IntervalMinutes = 15
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectRoot "logs"
$StateFile = Join-Path $LogDir "agent_state.json"
$PidFile = Join-Path $LogDir "agent.pid"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Write-Log {
    param($Message, $Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logLine = "[$timestamp] [$Level] $Message"
    $logFile = Join-Path $LogDir "autonomous_$(Get-Date -Format 'yyyy-MM-dd').log"
    Add-Content -Path $logFile -Value $logLine
    
    $color = switch ($Level) {
        "INFO" { "Cyan" }
        "SUCCESS" { "Green" }
        "WARN" { "Yellow" }
        "ERROR" { "Red" }
        default { "White" }
    }
    Write-Host $logLine -ForegroundColor $color
}

function Get-GitStatus {
    Push-Location $ProjectRoot
    $status = @{
        Modified = @()
        Untracked = @()
        LastCommit = ""
    }
    
    try {
        $gitStatus = git status --porcelain 2>$null
        if ($gitStatus) {
            $status.Modified = $gitStatus | Where-Object { $_ -match "^\s*M" } | ForEach-Object { $_.Substring(3) }
            $status.Untracked = $gitStatus | Where-Object { $_ -match "^\?\?" } | ForEach-Object { $_.Substring(3) }
        }
        $status.LastCommit = git log -1 --format="%H %s" 2>$null
    } catch {}
    
    Pop-Location
    return $status
}

function Test-BuildNeeded {
    # Check if source files are newer than binaries
    $srcFiles = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "*.rs" |
                Where-Object { $_.FullName -notmatch "target" } |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
    
    $binFile = Join-Path $ProjectRoot "target\release\sovereign.exe"
    
    if (-not (Test-Path $binFile)) { return $true }
    
    $binTime = (Get-Item $binFile).LastWriteTime
    $srcTime = $srcFiles.LastWriteTime
    
    return $srcTime -gt $binTime
}

function Invoke-Build {
    Write-Log "Starting release build..." "INFO"
    Push-Location $ProjectRoot
    
    $buildOutput = cargo build --release 2>&1
    $success = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    if ($success) {
        Write-Log "Build successful!" "SUCCESS"
    } else {
        Write-Log "Build failed!" "ERROR"
        $errors = $buildOutput | Where-Object { $_ -match "error" }
        foreach ($err in $errors) {
            Write-Log "  $err" "ERROR"
        }
    }
    
    return $success
}

function Invoke-Tests {
    Write-Log "Running test suite..." "INFO"
    Push-Location $ProjectRoot
    
    $testOutput = cargo test --lib 2>&1
    $success = $LASTEXITCODE -eq 0
    
    Pop-Location
    
    if ($success) {
        Write-Log "All tests passed!" "SUCCESS"
    } else {
        Write-Log "Some tests failed!" "WARN"
    }
    
    return $success
}

function Invoke-Analysis {
    Write-Log "Running code analysis..." "INFO"
    
    $stats = @{
        RustFiles = 0
        TotalLines = 0
        PhiReferences = 0
        ManifoldOps = 0
        Warnings = 0
    }
    
    $files = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "*.rs" |
             Where-Object { $_.FullName -notmatch "target" }
    
    foreach ($file in $files) {
        $content = Get-Content $file.FullName -Raw -ErrorAction SilentlyContinue
        if ($content) {
            $stats.RustFiles++
            $stats.TotalLines += ($content -split "`n").Count
            $stats.PhiReferences += ([regex]::Matches($content, "PHI|Φ|phi|golden")).Count
            $stats.ManifoldOps += ([regex]::Matches($content, "manifold|Manifold|poincare|hyperbolic")).Count
        }
    }
    
    # Φ-Coherence Score
    $phiScore = [math]::Min(100, ($stats.PhiReferences / [math]::Max(1, $stats.RustFiles)) * 15)
    
    Write-Log "Analysis complete: $($stats.RustFiles) files, $($stats.TotalLines) lines" "INFO"
    Write-Log "Φ-Coherence Score: $([math]::Round($phiScore, 1))%" "INFO"
    
    return $stats
}

function Save-State {
    param($State)
    $State | ConvertTo-Json -Depth 5 | Out-File $StateFile -Encoding UTF8
}

function Get-State {
    if (Test-Path $StateFile) {
        return Get-Content $StateFile -Raw | ConvertFrom-Json
    }
    return @{
        LastRun = $null
        LastBuild = $null
        LastTest = $null
        BuildCount = 0
        TestCount = 0
        SuccessCount = 0
    }
}

function Invoke-Cycle {
    Write-Log "=== Autonomous Cycle Starting ===" "INFO"
    Write-Log "Φ-Ratio: 1.618033988749895 | S²: 0.01" "INFO"
    
    $state = Get-State
    $state.LastRun = Get-Date -Format "o"
    
    # Check git status
    $git = Get-GitStatus
    if ($git.Modified.Count -gt 0) {
        Write-Log "Modified files: $($git.Modified -join ', ')" "INFO"
    }
    
    # Check if build needed
    $needsBuild = Test-BuildNeeded
    if ($needsBuild) {
        Write-Log "Source changes detected - rebuilding..." "INFO"
        $buildOK = Invoke-Build
        $state.BuildCount++
        $state.LastBuild = Get-Date -Format "o"
        
        if ($buildOK) {
            $state.SuccessCount++
            
            # Run tests after successful build
            $testOK = Invoke-Tests
            $state.TestCount++
            $state.LastTest = Get-Date -Format "o"
        }
    } else {
        Write-Log "No source changes - skipping build" "INFO"
    }
    
    # Always run analysis
    $analysis = Invoke-Analysis
    
    Save-State $state
    
    Write-Log "=== Cycle Complete ===" "SUCCESS"
    Write-Log "Total builds: $($state.BuildCount) | Success rate: $([math]::Round(($state.SuccessCount / [math]::Max(1, $state.BuildCount)) * 100, 1))%" "INFO"
}

function Start-Background {
    Write-Log "Starting autonomous agent in background..." "INFO"
    
    # Create background job script
    $jobScript = @"
while (`$true) {
    try {
        Push-Location "$ProjectRoot"
        
        # Check if build needed
        `$srcFiles = Get-ChildItem -Path . -Recurse -Filter "*.rs" | Where-Object { `$_.FullName -notmatch "target" } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        `$binFile = ".\target\release\sovereign.exe"
        
        if (-not (Test-Path `$binFile) -or (`$srcFiles.LastWriteTime -gt (Get-Item `$binFile).LastWriteTime)) {
            cargo build --release 2>&1 | Out-File "$LogDir\build_`$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
            cargo test --lib 2>&1 | Out-File "$LogDir\test_`$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
        }
        
        Pop-Location
    } catch {
        Add-Content -Path "$LogDir\error.log" -Value "[`$(Get-Date)] Error: `$_"
    }
    
    Start-Sleep -Seconds ($IntervalMinutes * 60)
}
"@
    
    $job = Start-Job -ScriptBlock ([scriptblock]::Create($jobScript))
    $job.Id | Out-File $PidFile
    
    Write-Log "Background agent started with Job ID: $($job.Id)" "SUCCESS"
    Write-Log "Check logs in: $LogDir" "INFO"
}

function Stop-Background {
    if (Test-Path $PidFile) {
        $jobId = Get-Content $PidFile
        Stop-Job -Id $jobId -ErrorAction SilentlyContinue
        Remove-Job -Id $jobId -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -Force
        Write-Log "Background agent stopped" "SUCCESS"
    } else {
        Write-Log "No background agent running" "WARN"
    }
}

function Show-Status {
    Write-Host @"

    ╔═══════════════════════════════════════════════════════════╗
    ║      7D CRYSTAL AUTONOMOUS AGENT STATUS                   ║
    ╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

    $state = Get-State
    
    Write-Host "Last Run:     $(if ($state.LastRun) { $state.LastRun } else { 'Never' })" -ForegroundColor Gray
    Write-Host "Last Build:   $(if ($state.LastBuild) { $state.LastBuild } else { 'Never' })" -ForegroundColor Gray
    Write-Host "Last Test:    $(if ($state.LastTest) { $state.LastTest } else { 'Never' })" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Build Count:  $($state.BuildCount)" -ForegroundColor White
    Write-Host "Test Count:   $($state.TestCount)" -ForegroundColor White
    Write-Host "Success Rate: $([math]::Round(($state.SuccessCount / [math]::Max(1, $state.BuildCount)) * 100, 1))%" -ForegroundColor Green
    
    # Check if background running
    if (Test-Path $PidFile) {
        $jobId = Get-Content $PidFile
        $job = Get-Job -Id $jobId -ErrorAction SilentlyContinue
        if ($job -and $job.State -eq "Running") {
            Write-Host ""
            Write-Host "Background Agent: RUNNING (Job $jobId)" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "Background Agent: STOPPED" -ForegroundColor Yellow
        }
    } else {
        Write-Host ""
        Write-Host "Background Agent: NOT STARTED" -ForegroundColor Gray
    }
    
    # Show recent logs
    $recentLog = Join-Path $LogDir "autonomous_$(Get-Date -Format 'yyyy-MM-dd').log"
    if (Test-Path $recentLog) {
        Write-Host ""
        Write-Host "Recent Activity:" -ForegroundColor Cyan
        Get-Content $recentLog -Tail 5 | ForEach-Object { Write-Host "  $_" -ForegroundColor Gray }
    }
}

# Main execution
Write-Host @"

    ╔═══════════════════════════════════════════════════════════╗
    ║     7D CRYSTAL SYSTEM - AUTONOMOUS AGENT                  ║
    ║                                                           ║
    ║   Token-Saving Background Build System                    ║
    ║   Discovered by Sir Charles Spikes                        ║
    ╚═══════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

if ($Start) {
    Start-Background
} elseif ($Stop) {
    Stop-Background
} elseif ($Status) {
    Show-Status
} elseif ($Once) {
    Invoke-Cycle
} else {
    Show-Status
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  -Once              Run one cycle now" -ForegroundColor Gray
    Write-Host "  -Start             Start background agent" -ForegroundColor Gray
    Write-Host "  -Stop              Stop background agent" -ForegroundColor Gray
    Write-Host "  -Status            Show current status" -ForegroundColor Gray
    Write-Host "  -IntervalMinutes N Set check interval (default: 15)" -ForegroundColor Gray
}
