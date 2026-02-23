# ABRASAX GOD OS - Start screen-aware self-enhance loop

param(
    [string]$Model = "qwen/qwen3-vl-4b",
    [string]$BaseUrl = "http://localhost:1234",
    [int]$Cycles = 3,
    [int]$IntervalSec = 60,
    [switch]$Apply
)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

Write-Host "ABRASAX Self-Enhance Loop" -ForegroundColor Cyan
Write-Host "Repo:    $RepoRoot" -ForegroundColor Gray
Write-Host "Model:   $Model" -ForegroundColor Gray
Write-Host "API:     $BaseUrl/v1/responses" -ForegroundColor Gray
Write-Host "Mode:    $(if ($Apply) { 'APPLY' } else { 'DRY-RUN' })" -ForegroundColor Gray

$argsList = @(
    "scripts/abrasax_self_enhance_loop.py",
    "--repo-root", "$RepoRoot",
    "--model", $Model,
    "--base-url", $BaseUrl,
    "--cycles", "$Cycles",
    "--interval-sec", "$IntervalSec"
)

if ($Apply) {
    $argsList += "--apply"
}

python @argsList
