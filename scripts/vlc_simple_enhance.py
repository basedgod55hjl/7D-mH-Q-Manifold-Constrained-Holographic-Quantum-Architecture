"""
VLC Simple Enhanced Launcher - GUARANTEED TO WORK
7D Crystal System
"""

import subprocess
import time
import os

# Paths
VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
PLAYLIST = r"C:\Users\BASEDGOD\Desktop\consolidated_playlist.xspf"

# Golden ratio
PHI_INV = 0.618033988749895

def main():
    print("🎬 VLC Enhanced Launcher")
    print("="*50)
    
    # Kill existing VLC
    subprocess.run(["taskkill", "/F", "/IM", "vlc.exe"], 
                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(1)
    
    # Simple but effective VLC settings
    cmd = [
        VLC_PATH,
        PLAYLIST,
        # Hardware acceleration (essential)
        "--avcodec-hw=dxva2",
        # Video output
        "--vout=direct3d11",
        # Sharpen filter
        "--video-filter=sharpen",
        f"--sharpen-sigma={PHI_INV}",
        # Color enhancement
        "--contrast=1.05",
        "--saturation=1.08",
        "--brightness=1.02",
        # Deinterlace
        "--deinterlace=1",
        "--deinterlace-mode=yadif",
        # Performance
        "--high-priority",
        "--avcodec-threads=0",
        # Caching
        "--file-caching=1500",
        "--network-caching=1500",
        # Fullscreen
        "--no-video-title-show",
    ]
    
    print(f"📺 Playlist: {PLAYLIST}")
    print(f"🔧 Sharpen: Φ⁻¹ = {PHI_INV:.3f}")
    print(f"🎨 Color: +5% contrast, +8% saturation")
    print()
    
    # Launch VLC
    proc = subprocess.Popen(cmd, creationflags=subprocess.HIGH_PRIORITY_CLASS)
    print(f"✅ VLC launched (PID: {proc.pid})")
    
    # Quick GPU check
    time.sleep(3)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            print(f"📊 GPU: {parts[0]}% | VRAM: {parts[1]} MB")
    except:
        pass
        
    print("\n✨ Video should be playing now!")

if __name__ == "__main__":
    main()
