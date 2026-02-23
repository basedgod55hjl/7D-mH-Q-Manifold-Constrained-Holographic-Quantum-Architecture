import os
import sys
import subprocess
import time
import argparse
import psutil

# Application Paths
VLC_PATHS = [
    r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"
]

LM_STUDIO_PATHS = [
    r"C:\Program Files\LM Studio\LM Studio.exe",
    os.path.expandvars(r"%LOCALAPPDATA%\LM-Studio\LM Studio.exe")
]

def find_executable(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def kill_process(name):
    print(f"[*] Terminating existing {name} instances...")
    for proc in psutil.process_iter(['name']):
        if name.lower() in proc.info['name'].lower():
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    time.sleep(1)

def launch_python_quantum_proxy():
    print("[*] Launching Pure Python Quantum Memory Proxy on Port 17778...")
    proxy_script = os.path.join(os.path.dirname(__file__), "python_quantum_proxy.py")
    subprocess.Popen([sys.executable, proxy_script], creationflags=subprocess.CREATE_NEW_CONSOLE)
    time.sleep(2)

def launch_system_overwatch():
    print("[*] Activating 7D Crystal System Overwatch (Auto LLM Loop)...")
    overwatch_script = os.path.join(os.path.dirname(__file__), "system_overwatch.py")
    subprocess.Popen([sys.executable, overwatch_script, "--loop"], creationflags=subprocess.CREATE_NEW_CONSOLE)

def launch_omnipresence():
    print("\n" + "🔥"*25)
    print("🔥 LAUNCHING ABRASAX OMNIPRESENT AGI GOVERNOR 🔥")
    print("🔥 CPU interception and 0ms continuous execution Active 🔥")
    print("🔥"*25 + "\n")
    omni_script = os.path.join(os.path.dirname(__file__), "agi_omnipresence.py")
    subprocess.Popen([sys.executable, omni_script], creationflags=subprocess.CREATE_NEW_CONSOLE)

def launch_vlc(video_path=None, enable_manifold=False):
    vlc_exe = find_executable(VLC_PATHS)
    if not vlc_exe:
        print("[!] VLC Player not found.")
        return

    kill_process("vlc.exe")
    print(f"[*] Launching VLC Player with hardware acceleration flags...")
    
    cmd = [
        vlc_exe,
        "--avcodec-hw=any",
        "--ffmpeg-hw",
        "--high-priority"
    ]
    
    if enable_manifold:
        print("    -> Routing frame output to 7D Crystal Manifold (17777)")
        cmd.extend(["--sout", "#transcode{vcodec=mjpg,scale=1.0}:http{mux=mpjpeg,dst=:17777/vlc/process_frame}"])
        
    if video_path:
        cmd.append(video_path)
        
    subprocess.Popen(cmd)

def launch_lm_studio(enable_manifold=False):
    lm_exe = find_executable(LM_STUDIO_PATHS)
    if not lm_exe:
        print("[!] LM Studio not found.")
        return

    kill_process("LM Studio.exe")
    print(f"[*] Launching LM Studio...")
    
    cmd = [lm_exe]
    
    if enable_manifold:
        print("    -> ⚠️ NOTE: You must configure LM Studio proxy settings internally to point to 127.0.0.1:17778")
        
    env = os.environ.copy()
    if enable_manifold:
        env["HTTP_PROXY"] = "http://127.0.0.1:17778"
        
    subprocess.Popen(cmd, env=env)

def launch_holovid(holovid_path):
    print("[*] Launching HoloVid with FFmpeg GPU encoders...")
    
    if not os.path.exists(holovid_path):
         print(f"[!] HoloVid app not found at {holovid_path}")
         return
         
    env = os.environ.copy()
    # Force GPU usage
    env["HOLOVID_GPU_ACCEL"] = "1"
    
    subprocess.Popen([sys.executable, holovid_path], cwd=os.path.dirname(holovid_path), env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7D Crystal Bridge System Orchestrator (Python Pure Mode)")
    parser.add_argument("--vlc", action="store_true", help="Launch VLC with GPU flags")
    parser.add_argument("--lm-studio", action="store_true", help="Launch LM Studio")
    parser.add_argument("--holovid", type=str, help="Launch HoloVid with the provided executable path")
    parser.add_argument("--video", type=str, help="Optional video file for VLC")
    parser.add_argument("--pure-python", action="store_true", help="Launch Python Quantum Proxy and Overwatch Loop (Replaces Rust)")
    parser.add_argument("--omnipresence", action="store_true", help="Launch Absolute AGI System Governor (0ms Loop, Native Code Execution)")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("💎   7D CRYSTAL BRIDGE INITIALIZATION (PYTHON)   💎")
    print("="*50 + "\n")
    
    if args.omnipresence:
        launch_omnipresence()
    elif args.pure_python:
        launch_python_quantum_proxy()
        launch_system_overwatch()
    
    if args.lm_studio:
        launch_lm_studio(enable_manifold=args.pure_python or args.omnipresence)
        
    if args.vlc:
        launch_vlc(args.video, enable_manifold=args.pure_python or args.omnipresence)
        
    if args.holovid:
        launch_holovid(args.holovid)
        
    print("\n[*] Bridge initialization sequence complete.")
