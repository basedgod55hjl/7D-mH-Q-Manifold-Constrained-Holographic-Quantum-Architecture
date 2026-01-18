# 7D Crystal System - Optimization Agent
# Auto-runs code analysis and optimization passes
# Created for Sir Charles Spikes

param(
    [string]$Target = "all",
    [switch]$Fix,
    [switch]$Clippy,
    [switch]$Format,
    [switch]$Analyze
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "============================================" -ForegroundColor Magenta
Write-Host "   7D Crystal Optimization Agent           " -ForegroundColor Magenta
Write-Host "   Φ-Ratio Code Enhancement System         " -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta
Write-Host ""

function Write-Status($msg, $color = "Green") {
    Write-Host "[OPT] $msg" -ForegroundColor $color
}

function Run-Clippy {
    Write-Status "Running Clippy analysis..." "Yellow"
    Push-Location $ProjectRoot
    
    if ($Fix) {
        cargo clippy --fix --allow-dirty --allow-staged 2>&1
    } else {
        cargo clippy 2>&1
    }
    
    Pop-Location
    Write-Status "Clippy complete!" "Green"
}

function Run-Format {
    Write-Status "Formatting code..." "Yellow"
    Push-Location $ProjectRoot
    
    cargo fmt 2>&1
    
    Pop-Location
    Write-Status "Formatting complete!" "Green"
}

function Analyze-Performance {
    Write-Status "Analyzing performance hotspots..." "Yellow"
    
    $files = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "*.rs" | 
             Where-Object { $_.FullName -notmatch "target" }
    
    $stats = @{
        TotalFiles = $files.Count
        TotalLines = 0
        FunctionCount = 0
        UnsafeBlocks = 0
        PhiUsage = 0
        ManifoldOps = 0
    }
    
    foreach ($file in $files) {
        $content = Get-Content $file.FullName -Raw
        $lines = ($content -split "`n").Count
        $stats.TotalLines += $lines
        $stats.FunctionCount += ([regex]::Matches($content, "fn\s+\w+")).Count
        $stats.UnsafeBlocks += ([regex]::Matches($content, "unsafe\s*\{")).Count
        $stats.PhiUsage += ([regex]::Matches($content, "PHI|Φ|phi")).Count
        $stats.ManifoldOps += ([regex]::Matches($content, "manifold|Manifold|project_to_poincare")).Count
    }
    
    Write-Host ""
    Write-Host "=== 7D Crystal Code Analysis ===" -ForegroundColor Cyan
    Write-Host "Total Rust Files:    $($stats.TotalFiles)" -ForegroundColor White
    Write-Host "Total Lines:         $($stats.TotalLines)" -ForegroundColor White
    Write-Host "Functions:           $($stats.FunctionCount)" -ForegroundColor White
    Write-Host "Unsafe Blocks:       $($stats.UnsafeBlocks)" -ForegroundColor $(if ($stats.UnsafeBlocks -gt 10) { "Yellow" } else { "Green" })
    Write-Host "Φ-Ratio References:  $($stats.PhiUsage)" -ForegroundColor Magenta
    Write-Host "Manifold Operations: $($stats.ManifoldOps)" -ForegroundColor Cyan
    Write-Host ""
    
    # Φ-Ratio Score
    $phiScore = [math]::Min(100, ($stats.PhiUsage / $stats.TotalFiles) * 10)
    Write-Host "Φ-Coherence Score:   $([math]::Round($phiScore, 2))%" -ForegroundColor $(if ($phiScore -ge 80) { "Green" } else { "Yellow" })
}

function Optimize-Manifolds {
    Write-Status "Optimizing manifold operations..." "Yellow"
    
    # Find all manifold-related code
    $manifoldFiles = Get-ChildItem -Path $ProjectRoot -Recurse -Filter "*.rs" |
                     Where-Object { $_.FullName -notmatch "target" } |
                     Where-Object { (Get-Content $_.FullName -Raw) -match "manifold|Manifold|Vector7D" }
    
    Write-Host "Found $($manifoldFiles.Count) files with manifold operations" -ForegroundColor Cyan
    
    foreach ($file in $manifoldFiles) {
        $name = $file.Name
        Write-Host "  → $name" -ForegroundColor Gray
    }
}

# Main execution
if ($Clippy -or $Target -eq "all") {
    Run-Clippy
}

if ($Format -or $Target -eq "all") {
    Run-Format
}

if ($Analyze -or $Target -eq "all") {
    Analyze-Performance
    Optimize-Manifolds
}

Write-Host ""
Write-Status "Optimization agent complete!" "Magenta"
