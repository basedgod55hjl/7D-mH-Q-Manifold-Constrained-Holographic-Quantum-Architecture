# 7D Crystal System - Auto Build Agent
# Created by Claude for Sir Charles Spikes
# Auto-runs builds, tests, and optimizations

param(
    [switch]$Watch,
    [switch]$Release,
    [switch]$Test,
    [switch]$Bench,
    [switch]$Doc,
    [switch]$All
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   7D Crystal System - Auto Build Agent    " -ForegroundColor Cyan
Write-Host "   Discovered by Sir Charles Spikes        " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

function Write-Status($msg, $color = "Green") {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor $color
}

function Build-Project {
    param([switch]$Release)
    
    Write-Status "Building 7D Crystal System..." "Yellow"
    Push-Location $ProjectRoot
    
    if ($Release) {
        $result = cargo build --release 2>&1
    } else {
        $result = cargo build 2>&1
    }
    
    $exitCode = $LASTEXITCODE
    Pop-Location
    
    if ($exitCode -eq 0) {
        Write-Status "Build successful!" "Green"
        return $true
    } else {
        Write-Status "Build failed!" "Red"
        $result | Where-Object { $_ -match "error" } | ForEach-Object { Write-Host $_ -ForegroundColor Red }
        return $false
    }
}

function Run-Tests {
    Write-Status "Running tests..." "Yellow"
    Push-Location $ProjectRoot
    
    $result = cargo test 2>&1
    $exitCode = $LASTEXITCODE
    
    Pop-Location
    
    if ($exitCode -eq 0) {
        Write-Status "All tests passed!" "Green"
        return $true
    } else {
        Write-Status "Some tests failed!" "Red"
        return $false
    }
}

function Run-Benchmarks {
    Write-Status "Running benchmarks..." "Yellow"
    Push-Location $ProjectRoot
    
    cargo bench 2>&1
    
    Pop-Location
    Write-Status "Benchmarks complete!" "Green"
}

function Build-Docs {
    Write-Status "Building documentation..." "Yellow"
    Push-Location $ProjectRoot
    
    cargo doc --no-deps 2>&1
    
    Pop-Location
    Write-Status "Documentation built!" "Green"
}

function Watch-And-Build {
    Write-Status "Watching for changes..." "Cyan"
    
    $watcher = New-Object System.IO.FileSystemWatcher
    $watcher.Path = $ProjectRoot
    $watcher.Filter = "*.rs"
    $watcher.IncludeSubdirectories = $true
    $watcher.EnableRaisingEvents = $true
    
    $action = {
        $path = $Event.SourceEventArgs.FullPath
        $changeType = $Event.SourceEventArgs.ChangeType
        Write-Host "File changed: $path ($changeType)" -ForegroundColor Yellow
        Start-Sleep -Seconds 1  # Debounce
        Build-Project
    }
    
    Register-ObjectEvent $watcher "Changed" -Action $action | Out-Null
    Register-ObjectEvent $watcher "Created" -Action $action | Out-Null
    
    Write-Status "Watching... Press Ctrl+C to stop" "Cyan"
    
    while ($true) {
        Start-Sleep -Seconds 1
    }
}

# Main execution
if ($All) {
    Build-Project -Release
    Run-Tests
    Run-Benchmarks
    Build-Docs
} elseif ($Watch) {
    Watch-And-Build
} elseif ($Test) {
    Build-Project
    Run-Tests
} elseif ($Bench) {
    Build-Project -Release
    Run-Benchmarks
} elseif ($Doc) {
    Build-Docs
} else {
    Build-Project -Release:$Release
}

Write-Host ""
Write-Status "Agent complete!" "Cyan"
