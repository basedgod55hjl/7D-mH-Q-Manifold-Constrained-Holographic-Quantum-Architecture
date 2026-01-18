# launch_ecosystem.ps1
$Root = 'C:\Users\BASEDGOD\Desktop\7D_Crystal_System'
$Bin = 'C:\Users\BASEDGOD\.cargo\bin\cargo.exe'

Write-Host '--- 7D Crystal System: Scale-Up ---' -ForegroundColor Cyan

# 1. Recursive Optimizer
Write-Host '[>] Launching Recursive Optimizer...'
Start-Job -ScriptBlock { param($b, $c); cd $c; & $b run --release --bin optimizer } -ArgumentList $Bin, $Root

# 2. Compiler Service
Write-Host '[>] Launching Compiler Service...'
Start-Job -ScriptBlock { param($b, $c); cd $c; & $b run --release --bin compiler_service } -ArgumentList $Bin, $Root

# 3. Crystal AGI
Write-Host '[>] Launching Crystal AGI...'
Start-Job -ScriptBlock { param($b, $c); cd $c; & $b run --release --bin crystal_agi } -ArgumentList $Bin, $Root

# 4. Quantum Hybrid
Write-Host '[>] Launching Quantum Hybrid...'
Start-Job -ScriptBlock { param($b, $c); cd $c; & $b run --release --bin quantum_hybrid } -ArgumentList $Bin, $Root

Write-Host 'All jobs started in background.' -ForegroundColor Green
