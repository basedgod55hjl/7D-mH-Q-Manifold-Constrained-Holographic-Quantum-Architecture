# File: scripts/run_omnipresence_autonomous.ps1
# Autonomous non-stop launcher - build, test, run until stable

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$RUNTIME = Join-Path $ROOT "runtime"
$SCRIPTS = Join-Path $ROOT "scripts"

Write-Host "`n=== OMNIPRESENCE AUTONOMOUS LAUNCHER ===" -ForegroundColor Cyan
Write-Host "6GB VRAM | No Sleep | Multi-Agent Parallel | Full Tool Access`n" -ForegroundColor Gray

# 1. Install Python deps if needed
Write-Host "[1/5] Checking Python dependencies..." -ForegroundColor Yellow
$deps = @("mss", "psutil", "requests")
foreach ($d in $deps) {
    python -c "import $d" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Installing $d..." -ForegroundColor Gray
        pip install $d --quiet
    }
}
Write-Host "  OK`n" -ForegroundColor Green

# 2. Build runtime (Rust) - close runtime.exe first if running
Write-Host "[2/5] Building 7D Crystal Runtime..." -ForegroundColor Yellow
Get-Process -Name "runtime" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
Push-Location $RUNTIME
cargo build 2>&1 | Out-Host
$buildOk = ($LASTEXITCODE -eq 0)
Pop-Location
if ($buildOk) { Write-Host "  OK`n" -ForegroundColor Green } else { Write-Host "  (continuing anyway)`n" -ForegroundColor Gray }

# 3. Quick test
Write-Host "[3/5] Running runtime test..." -ForegroundColor Yellow
Push-Location $RUNTIME
cargo test 2>&1 | Select-Object -First 30 | Out-Host
Pop-Location
Write-Host "  Done`n" -ForegroundColor Gray

# 4. Start omnipresence (background, no sleep)
Write-Host "[4/5] Starting Omnipresence Realtime (no-sleep multi-agent)..." -ForegroundColor Yellow
$omnip = Start-Process -FilePath "python" -ArgumentList (Join-Path $SCRIPTS "omnipresence_realtime.py") -WorkingDirectory $ROOT -PassThru -NoNewWindow
Write-Host "  PID: $($omnip.Id)`n" -ForegroundColor Green

# 5. Optionally start overlay (ImGui) in parallel
Write-Host "[5/5] To run ImGui overlay in parallel: python abrasax_overlay.py" -ForegroundColor Gray
Write-Host "`

OMNIPRESENCE RUNNING - Ctrl+C to stop. Processes will continue until terminated.`n" -ForegroundColor Cyan

# Keep script alive, forward signals
try {
    Wait-Process -Id $omnip.Id -ErrorAction SilentlyContinue
} catch {}
Write-Host "Omnipresence process ended." -ForegroundColor Gray
