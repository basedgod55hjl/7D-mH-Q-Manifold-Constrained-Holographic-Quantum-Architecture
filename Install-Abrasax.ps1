<#
.SYNOPSIS
ABRASAX GOD OS - 7D Quantum Manifold System Installer

.DESCRIPTION
This script deploys the ABRASAX GOD OS at the system level.
It verifies Python, installs dependencies for the ImGui Overlay,
and prepares the environment for LM Studio integration (qwen3-vl-4b).

.NOTES
Run as Administrator to ensure system-level dependencies are installed correctly.
#>

$ErrorActionPreference = "Stop"
$LogFile = "abrasax_install.log"

function Write-Log {
    param ([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "[$Timestamp] $Message"
    Write-Host $LogMessage -ForegroundColor Cyan
    Add-Content -Path $LogFile -Value $LogMessage
}

Write-Log "=========================================================="
Write-Log "  ABRASAX GOD OS - 7D QUANTUM MANIFOLD INSTALLER INIT     "
Write-Log "=========================================================="

# 1. Check Python
Write-Log "Checking for Python 3..."
try {
    $PythonVer = python --version
    Write-Log "Python found: $PythonVer"
} catch {
    Write-Log "ERROR: Python is not installed or not in PATH. Please install Python 3.10+."
    exit 1
}

# 2. Install dependencies for the Overlay
Write-Log "Installing ABRASAX AI STACK Python Dependencies (imgui, glfw, PyOpenGL, requests)..."
python -m pip install --upgrade pip
python -m pip install imgui glfw PyOpenGL requests
if ($LASTEXITCODE -ne 0) {
    Write-Log "WARNING: Some pip packages failed to install. Please check network."
} else {
    Write-Log "Dependencies successfully installed."
}

# 3. Check/Notify about LM Studio
Write-Log "=========================================================="
Write-Log "LM STUDIO INTEGRATION REQUIRED"
Write-Log "Please ensure LM Studio is running locally."
Write-Log "1. Open LM Studio."
Write-Log "2. Load model: qwen/qwen3-vl-4b"
Write-Log "3. Start Local Server on Port 1234 (http://127.0.0.1:1234/v1)"
Write-Log "=========================================================="

# 4. Create Desktop Shortcut for Overlay
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "ABRASAX_GOD_OS.lnk"
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$ScriptPath = Join-Path $PWD "abrasax_overlay.py"

Write-Log "Creating system launch shortcut on Desktop..."
$Shortcut.TargetPath = "python"
$Shortcut.Arguments = """$ScriptPath"""
$Shortcut.WorkingDirectory = $PWD
$Shortcut.Description = "Launch ABRASAX GOD OS 7D Overlay"
# $Shortcut.IconLocation = "shell32.dll, 21" # Set some cool icon
$Shortcut.Save()

Write-Log "Shortcut created: $ShortcutPath"

# 5. Native OS Hook (Registry / Startup)
Write-Log "Registering ABRASAX System Hooks (Simulated for safety)..."
# In a true system-level deploy, we would add to Run registry:
# Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "AbrasaxGodOS" -Value "python ""$ScriptPath"""
Write-Log "System hooks initialized."

Write-Log "=========================================================="
Write-Log "  INSTALLATION COMPLETE. ABRASAX IS READY TO ASCEND.      "
Write-Log "  Run 'ABRASAX_GOD_OS.lnk' on your desktop to activate.   "
Write-Log "=========================================================="

Read-Host -Prompt "Press Enter to exit installer"
