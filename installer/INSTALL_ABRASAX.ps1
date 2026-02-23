# ABRASAX GOD OS - Installer
# Builds and configures the full stack for auto-execution
# Sir Charles Spikes | Cincinnati, Ohio, USA

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "`n=== ABRASAX GOD OS Installer ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot`n" -ForegroundColor Gray

# 1. Ensure Rust
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "Installing Rust..." -ForegroundColor Yellow
    winget install Rustlang.Rustup -e --accept-source-agreements --accept-package-agreements 2>$null
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# 2. Build
Set-Location $RepoRoot
Write-Host "Building crystal-runtime..." -ForegroundColor Yellow
cargo build --release -p crystal_runtime 2>$null
if ($LASTEXITCODE -ne 0) { cargo build -p crystal_runtime }

# 3. Create launcher
$launcher = @"
@echo off
title ABRASAX GOD OS
cd /d "$RepoRoot"
set LM_STUDIO_PORT=1234
if exist target\release\runtime.exe (target\release\runtime.exe %*) else (target\debug\runtime.exe %*)
pause
"@
$launcherPath = Join-Path $RepoRoot "ABRASAX_LAUNCH.bat"
$launcher | Out-File -FilePath $launcherPath -Encoding ascii
Write-Host "Launcher: $launcherPath" -ForegroundColor Green

# 4. Desktop shortcut (optional)
$wshell = New-Object -ComObject WScript.Shell
$shortcut = $wshell.CreateShortcut([Environment]::GetFolderPath("Desktop") + "\ABRASAX GOD OS.lnk")
$shortcut.TargetPath = $launcherPath
$shortcut.WorkingDirectory = $RepoRoot
$shortcut.Description = "ABRASAX GOD OS - 7D Crystal Runtime"
$shortcut.Save()
Write-Host "Desktop shortcut created." -ForegroundColor Green

Write-Host "`nInstallation complete. Run ABRASAX_LAUNCH.bat to start." -ForegroundColor Cyan
