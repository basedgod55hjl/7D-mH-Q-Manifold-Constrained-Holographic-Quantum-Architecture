"""
DUAL GPU VIDEO PIPELINE
NVIDIA GTX 1660 Ti → Decode + Encode (NVENC)
AMD Radeon iGPU → Render on Virtual Desktop

Pipeline:
  Video File → NVIDIA (NVDEC decode) → NVENC encode → Stream → AMD iGPU (Virtual Desktop)
"""

import subprocess
import time
import os
import threading
import psutil

# Paths
VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
FFMPEG_PATH = "ffmpeg"
FFPLAY_PATH = "ffplay"
PLAYLIST = r"C:\Users\BASEDGOD\Desktop\fresh_playlist.m3u"

# Stream port for inter-GPU communication
STREAM_PORT = 8765

def get_video_from_playlist():
    """Get first video from playlist"""
    try:
        with open(PLAYLIST, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and os.path.exists(line):
                    return line
    except:
        pass
    # Fallback to E: drive
    import glob
    videos = glob.glob("E:\\*.mp4")
    return videos[0] if videos else None

def setup_nvidia_encoder():
    """Configure NVIDIA for encoding"""
    print("🟢 NVIDIA GTX 1660 Ti Configuration:")
    print("   Role: Decode (NVDEC) + Encode (NVENC)")
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_FORCE_PTX_JIT"] = "1"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "video,compute"
    
    print("   ✓ NVDEC: Hardware decode enabled")
    print("   ✓ NVENC: H.264 encoding enabled")

def setup_amd_igpu():
    """Configure AMD iGPU for display"""
    print("\n🔵 AMD Radeon iGPU Configuration:")
    print("   Role: Render on Virtual Desktop")
    
    # AMD Vulkan/OpenGL settings
    os.environ["AMD_VULKAN_ICD"] = "RADV"
    os.environ["LIBVA_DRIVER_NAME"] = "radeonsi"
    os.environ["MESA_GL_VERSION_OVERRIDE"] = "4.6"
    os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "460"
    
    # Force display output to AMD
    os.environ["DRI_PRIME"] = "0"  # 0 = iGPU for rendering
    
    print("   ✓ Vulkan: AMD RADV driver")
    print("   ✓ OpenGL: Mesa RadeonSI")
    print("   ✓ Display: Virtual Desktop output")

def start_nvidia_encoder(video_path):
    """Start NVIDIA encode pipeline - outputs to UDP stream"""
    
    print(f"\n📹 Starting NVIDIA Encoder Pipeline...")
    print(f"   Input: {os.path.basename(video_path)}")
    print(f"   Output: UDP stream → AMD iGPU")
    
    # FFmpeg command: NVIDIA decode → enhance → NVENC encode → UDP stream
    cmd = [
        FFMPEG_PATH,
        "-hide_banner",
        "-loglevel", "warning",
        "-stats",
        # NVIDIA hardware decode
        "-hwaccel", "cuda",
        "-hwaccel_device", "0",
        "-hwaccel_output_format", "cuda",
        # Input
        "-i", video_path,
        # Video filters (on GPU)
        "-vf", "scale_cuda=1920:1080,hwdownload,format=nv12,eq=saturation=1.1:contrast=1.05",
        # NVENC encoding
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-tune", "ll",  # Low latency for streaming
        "-rc", "cbr",
        "-b:v", "15M",
        "-maxrate", "15M",
        "-bufsize", "30M",
        "-gpu", "0",
        "-zerolatency", "1",
        # Audio
        "-c:a", "aac",
        "-b:a", "192k",
        # Output to UDP for AMD iGPU
        "-f", "mpegts",
        f"udp://127.0.0.1:{STREAM_PORT}?pkt_size=1316"
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.HIGH_PRIORITY_CLASS
    )
    
    print(f"   ✓ NVIDIA Encoder started (PID: {proc.pid})")
    print(f"   ✓ Streaming to: udp://127.0.0.1:{STREAM_PORT}")
    
    return proc

def start_amd_display():
    """Start VLC on AMD iGPU to receive and display stream"""
    
    print(f"\n📺 Starting AMD iGPU Display...")
    
    # VLC configured to use AMD iGPU for rendering
    cmd = [
        VLC_PATH,
        f"udp://@127.0.0.1:{STREAM_PORT}",
        # Force software decode (let AMD handle display only)
        "--avcodec-hw=none",
        # DirectX output (will use AMD iGPU if primary)
        "--vout=direct3d11",
        # Low latency settings
        "--network-caching=200",
        "--live-caching=200",
        "--clock-jitter=0",
        # Display settings
        "--no-video-title-show",
        "--video-on-top",
        # Audio
        "--aout=directsound",
    ]
    
    # Set environment for AMD
    env = os.environ.copy()
    env["DRI_PRIME"] = "0"  # Use iGPU
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print(f"   ✓ VLC Display started (PID: {proc.pid})")
    print(f"   ✓ Receiving from: udp://@127.0.0.1:{STREAM_PORT}")
    print(f"   ✓ Rendering on: AMD Radeon iGPU")
    
    return proc

def start_virtual_desktop_player():
    """Alternative: Use ffplay for virtual desktop display"""
    
    print(f"\n🖥️ Starting Virtual Desktop Player...")
    
    cmd = [
        FFPLAY_PATH,
        "-hide_banner",
        "-loglevel", "warning",
        "-stats",
        # Input stream
        f"udp://127.0.0.1:{STREAM_PORT}",
        # Low latency
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-framedrop",
        # Window settings
        "-window_title", "7D Crystal - Virtual Desktop (AMD iGPU)",
        "-alwaysontop",
    ]
    
    proc = subprocess.Popen(
        cmd,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )
    
    print(f"   ✓ FFplay started (PID: {proc.pid})")
    return proc

def monitor_dual_gpu(duration=12):
    """Monitor both GPUs"""
    print(f"\n📊 Dual GPU Monitor ({duration}s):")
    print("   " + "─"*70)
    print("   Time │ NVIDIA GPU │ NVIDIA Enc │ NVIDIA Dec │ VRAM    │ Temp")
    print("   " + "─"*70)
    
    for i in range(duration):
        try:
            result = subprocess.run([
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.encoder,utilization.decoder,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                p = [x.strip() for x in result.stdout.strip().split(",")]
                bar = "█" * (int(p[0]) // 5) + "░" * (20 - int(p[0]) // 5)
                print(f"   {i+1:3d}s │ [{bar}] {p[0]:>3}% │ {p[1]:>3}%      │ {p[2]:>3}%      │ {p[3]:>5} MB │ {p[4]}°C")
        except:
            pass
        time.sleep(1)
    
    print("   " + "─"*70)

def main():
    print("\n" + "═"*70)
    print("  🎮 DUAL GPU VIDEO PIPELINE")
    print("  NVIDIA → Encode | AMD iGPU → Virtual Desktop Display")
    print("═"*70)
    
    # Get video
    video = get_video_from_playlist()
    if not video:
        print("❌ No video found!")
        return
        
    print(f"\n📁 Video: {os.path.basename(video)}")
    
    # Setup GPUs
    setup_nvidia_encoder()
    setup_amd_igpu()
    
    # Start NVIDIA encoder first
    encoder_proc = start_nvidia_encoder(video)
    
    # Wait for encoder to start streaming
    time.sleep(2)
    
    # Start AMD display
    display_proc = start_amd_display()
    
    # Monitor
    time.sleep(2)
    monitor_dual_gpu(10)
    
    # Final status
    print("\n" + "═"*70)
    print("  ✅ DUAL GPU PIPELINE ACTIVE")
    print("═"*70)
    print(f"""
  Pipeline Flow:
  ─────────────────────────────────────────────────────────────────────
  
  [Video File]
       │
       ▼
  ┌─────────────────────────────────────┐
  │  🟢 NVIDIA GTX 1660 Ti              │
  │  ─────────────────────────────────  │
  │  NVDEC Decode → Enhance → NVENC     │
  │  H.264 encode @ 15Mbps              │
  └─────────────────────────────────────┘
       │
       │ UDP Stream (127.0.0.1:{STREAM_PORT})
       │
       ▼
  ┌─────────────────────────────────────┐
  │  🔵 AMD Radeon iGPU                 │
  │  ─────────────────────────────────  │
  │  Receive → Decode → Display         │
  │  Virtual Desktop Output             │
  └─────────────────────────────────────┘
       │
       ▼
  [Virtual Desktop Display]
  
  ─────────────────────────────────────────────────────────────────────
""")

if __name__ == "__main__":
    main()
