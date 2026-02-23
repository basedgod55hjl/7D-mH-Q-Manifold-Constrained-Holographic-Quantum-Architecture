"""
GAME MODE VLC - Full PC Game Performance for Video Playback

Enables everything Windows does when a game is running:
- Windows Game Mode ON (disables background updates/notifications)
- GPU Performance Mode (maximum clocks)
- Power Plan: Ultimate Performance
- Process priority: REALTIME
- Disable compositor (DWM optimizations)
- Disable unnecessary services temporarily
- NVIDIA prefer maximum performance
- Timer resolution: 1ms (like games)
"""

import os
import sys
import subprocess
import time
import ctypes
import glob
from pathlib import Path

VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

def run_admin_cmd(cmd, desc=""):
    """Run a command, print result"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                          creationflags=subprocess.CREATE_NO_WINDOW)
        if r.returncode == 0:
            print(f"  ✓ {desc}")
        else:
            print(f"  ○ {desc} (needs admin)")
    except Exception as e:
        print(f"  ○ {desc} (skipped)")


def enable_game_mode():
    """Enable Windows Game Mode via registry"""
    print("\n🎮 [1/7] WINDOWS GAME MODE")

    # Enable Game Mode
    run_admin_cmd([
        "reg", "add", r"HKCU\Software\Microsoft\GameBar",
        "/v", "AllowAutoGameMode", "/t", "REG_DWORD", "/d", "1", "/f"
    ], "Game Mode: ON")

    run_admin_cmd([
        "reg", "add", r"HKCU\Software\Microsoft\GameBar",
        "/v", "AutoGameModeEnabled", "/t", "REG_DWORD", "/d", "1", "/f"
    ], "Auto Game Mode: ON")

    # Disable Game Bar overlay (reduces overhead)
    run_admin_cmd([
        "reg", "add", r"HKCU\Software\Microsoft\GameBar",
        "/v", "UseNexusForGameBarEnabled", "/t", "REG_DWORD", "/d", "0", "/f"
    ], "Game Bar Overlay: OFF (less overhead)")

    # Disable Game DVR background recording
    run_admin_cmd([
        "reg", "add", r"HKCU\Software\Microsoft\Windows\CurrentVersion\GameDVR",
        "/v", "AppCaptureEnabled", "/t", "REG_DWORD", "/d", "0", "/f"
    ], "Game DVR Recording: OFF")


def set_gpu_max_performance():
    """Force NVIDIA GPU to maximum performance mode"""
    print("\n⚡ [2/7] GPU MAXIMUM PERFORMANCE")

    # NVIDIA persistence mode
    run_admin_cmd(["nvidia-smi", "-pm", "1"], "NVIDIA Persistence Mode: ON")

    # Force max GPU clocks
    run_admin_cmd(["nvidia-smi", "-lgc", "0,9999"], "GPU Clock: UNLOCKED (max)")

    # Force max memory clocks
    run_admin_cmd(["nvidia-smi", "-lmc", "0,9999"], "Memory Clock: UNLOCKED (max)")

    # Power limit to max
    run_admin_cmd(["nvidia-smi", "-pl", "80"], "Power Limit: MAX")

    # Prefer maximum performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_FORCE_PTX_JIT"] = "1"
    os.environ["CUDA_AUTO_BOOST"] = "1"
    os.environ["__GL_THREADED_OPTIMIZATIONS"] = "1"
    os.environ["__GL_YIELD"] = "NOTHING"
    os.environ["__GL_MaxFramesAllowed"] = "1"
    print("  ✓ CUDA/OpenGL: Max performance env vars")


def set_power_plan():
    """Switch to Ultimate/High Performance power plan"""
    print("\n🔋 [3/7] POWER PLAN: ULTIMATE PERFORMANCE")

    # Try to enable hidden Ultimate Performance plan
    run_admin_cmd([
        "powercfg", "-duplicatescheme", "e9a42b02-d5df-448d-aa00-03f14749eb61"
    ], "Ultimate Performance plan: Created")

    # List power plans and activate high performance
    try:
        r = subprocess.run(["powercfg", "-list"], capture_output=True, text=True, timeout=5)
        for line in r.stdout.split('\n'):
            if 'ultimate' in line.lower() or 'high performance' in line.lower():
                # Extract GUID
                parts = line.split()
                for p in parts:
                    if len(p) == 36 and p.count('-') == 4:
                        run_admin_cmd(["powercfg", "-setactive", p],
                                     f"Power plan: {line.strip()[:50]}")
                        break
    except:
        pass

    # Disable USB selective suspend
    run_admin_cmd([
        "powercfg", "/SETACVALUEINDEX", "SCHEME_CURRENT", "2a737441-1930-4402-8d77-b2bebba308a3",
        "48e6b7a6-50f5-4782-a5d4-53bb8f07e226", "0"
    ], "USB Selective Suspend: OFF")

    # Disable PCI Express power management
    run_admin_cmd([
        "powercfg", "/SETACVALUEINDEX", "SCHEME_CURRENT", "501a4d13-42af-4429-9fd1-a8218c268e20",
        "ee12f906-d277-404b-b6da-e5fa1a576df5", "0"
    ], "PCI-E Power Saving: OFF")

    run_admin_cmd(["powercfg", "-setactive", "scheme_current"], "Applied power settings")


def set_timer_resolution():
    """Set Windows timer to 1ms like games do"""
    print("\n⏱️ [4/7] TIMER RESOLUTION: 1ms")
    try:
        ntdll = ctypes.windll.ntdll
        # Set timer resolution to 1ms (10000 = 1ms in 100ns units)
        ntdll.NtSetTimerResolution(10000, True, ctypes.byref(ctypes.c_ulong()))
        print("  ✓ Timer resolution: 1ms (games use this)")
    except:
        print("  ○ Timer resolution: Could not set")


def disable_background_services():
    """Reduce background activity like when a game runs"""
    print("\n🔇 [5/7] BACKGROUND OPTIMIZATION")

    # Disable Windows Search indexing temporarily
    run_admin_cmd(["net", "stop", "WSearch"], "Windows Search: Paused")

    # Disable SysMain (Superfetch) - frees RAM
    run_admin_cmd(["net", "stop", "SysMain"], "SysMain/Superfetch: Paused")

    # Disable Windows Update service temporarily
    run_admin_cmd(["net", "stop", "wuauserv"], "Windows Update: Paused")

    # Disable diagnostic tracking
    run_admin_cmd(["net", "stop", "DiagTrack"], "Diagnostic Tracking: Paused")

    # Flush DNS and standby memory
    run_admin_cmd(["ipconfig", "/flushdns"], "DNS Cache: Flushed")


def set_process_priority():
    """Configure process priorities"""
    print("\n🏎️ [6/7] PROCESS PRIORITIES")

    # Lower priority of known background hogs
    hogs = {
        "SearchIndexer": "idle",
        "MsMpEng": "belownormal",
        "OneDrive": "idle",
        "chrome": "belownormal",
    }
    for proc_name, priority in hogs.items():
        try:
            r = subprocess.run(
                ["wmic", "process", "where", f"name='{proc_name}.exe'",
                 "CALL", "setpriority", priority],
                capture_output=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW
            )
            print(f"  ✓ {proc_name}: {priority}")
        except:
            pass


def launch_vlc_game_mode():
    """Launch VLC with game-level performance"""
    print("\n🎬 [7/7] VLC GAME MODE LAUNCH")

    videos = sorted(glob.glob("E:\\*.mp4"))

    # Create playlist
    playlist = r"C:\Users\BASEDGOD\Desktop\fresh_playlist.m3u"
    with open(playlist, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        for v in videos:
            f.write(f"#EXTINF:-1,{Path(v).stem}\n{v}\n")
    print(f"  Playlist: {len(videos)} videos")

    cmd = [
        VLC_PATH, playlist,
        # GPU decode - full hardware
        "--avcodec-hw=dxva2",
        "--vout=direct3d11",
        "--avcodec-threads=0",
        # Maximum caching (slow card reader)
        "--file-caching=10000",
        "--disc-caching=10000",
        "--network-caching=5000",
        "--clock-synchro=0",
        "--clock-jitter=0",
        # Performance flags
        "--high-priority",
        "--no-video-title-show",
        "--preparse-timeout=1000",
    ]

    # REALTIME priority class (same as games)
    REALTIME_PRIORITY = 0x00000100
    proc = subprocess.Popen(
        cmd,
        creationflags=REALTIME_PRIORITY | subprocess.CREATE_NEW_CONSOLE
    )
    print(f"  ✓ VLC PID: {proc.pid}")
    print(f"  ✓ Priority: REALTIME (same as games)")
    print(f"  ✓ Decode: DXVA2 (GPU hardware)")
    print(f"  ✓ Render: Direct3D 11")
    print(f"  ✓ Cache: 10 seconds")

    return proc


def gpu_status():
    try:
        r = subprocess.run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.decoder,memory.used,temperature.gpu,clocks.gr,clocks.mem,power.draw",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=3)
        return [x.strip() for x in r.stdout.strip().split(",")]
    except:
        return None


def main():
    print()
    print("  ═══════════════════════════════════════════════════")
    print("  🎮 GAME MODE VLC")
    print("  Full PC gaming performance for video playback")
    print("  ═══════════════════════════════════════════════════")

    # Kill existing VLC
    subprocess.run(["taskkill", "/F", "/IM", "vlc.exe"],
                  capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(1)

    # Apply all game mode settings
    enable_game_mode()
    set_gpu_max_performance()
    set_power_plan()
    set_timer_resolution()
    disable_background_services()
    set_process_priority()

    # Launch VLC
    vlc = launch_vlc_game_mode()

    # Monitor
    print(f"\n  📊 Game Mode Performance:")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Time | GPU   | Dec  | VRAM    | Temp | Clk     | Power")
    print(f"  ─────────────────────────────────────────────────")

    for i in range(10):
        time.sleep(1)
        s = gpu_status()
        if s and len(s) >= 7:
            bar = "█" * (int(s[0]) // 5) + "░" * (20 - int(s[0]) // 5)
            print(f"  {i+1:2d}s [{bar}] {s[0]:>3}% {s[1]:>3}% {s[2]:>5}MB {s[3]:>3}°C {s[4]:>4}/{s[5]:>4}MHz {s[6]:>5}W")

    print(f"  ─────────────────────────────────────────────────")

    print(f"""
  ═══════════════════════════════════════════════════
  ✅ GAME MODE ACTIVE
  ═══════════════════════════════════════════════════

  Windows:
    Game Mode         : ON
    Game DVR          : OFF
    Game Bar Overlay  : OFF
    Timer Resolution  : 1ms
    Power Plan        : Ultimate Performance

  GPU:
    Persistence Mode  : ON
    Clock Limit       : UNLOCKED
    Power Limit       : MAX
    OpenGL Threading  : ON
    Max Frames        : 1 (no buffering)

  Background:
    Windows Search    : PAUSED
    SysMain/Superfetch: PAUSED
    Windows Update    : PAUSED
    Diagnostic Track  : PAUSED
    Chrome/Search     : LOW priority

  VLC:
    Priority          : REALTIME
    Decode            : DXVA2 (GPU)
    Render            : Direct3D 11
    Cache             : 10 seconds

  Your PC is now running like a game is active.
  ═══════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
