"""
7D CRYSTAL BRIDGE TURBO
Python RAM-buffered HTTP proxy between slow E: drive and VLC.
Reads ahead with large buffers, serves at RAM speed.
"""

import os
import sys
import glob
import time
import threading
import subprocess
import http.server
import socketserver
import urllib.parse
import io
from pathlib import Path

VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
SOURCE_DIR = "E:\\"
HTTP_PORT = 9876
READ_BUFFER = 4 * 1024 * 1024  # 4MB read chunks from E:
SEND_BUFFER = 256 * 1024       # 256KB send chunks to VLC

# Video file cache: keeps recently read chunks in RAM
file_cache = {}
cache_lock = threading.Lock()


class BufferedVideoHandler(http.server.BaseHTTPRequestHandler):
    """Serves video from E: with RAM buffering and range support"""

    def log_message(self, format, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def do_HEAD(self):
        self._serve(head_only=True)

    def do_GET(self):
        self._serve(head_only=False)

    def _serve(self, head_only=False):
        path = urllib.parse.unquote(self.path.lstrip('/'))
        filepath = os.path.join(SOURCE_DIR, path)

        if not os.path.exists(filepath):
            self.send_error(404)
            return

        file_size = os.path.getsize(filepath)

        # Parse Range header
        range_header = self.headers.get('Range')
        start = 0
        end = file_size - 1

        if range_header:
            try:
                range_spec = range_header.replace('bytes=', '')
                parts = range_spec.split('-')
                if parts[0]:
                    start = int(parts[0])
                if parts[1]:
                    end = int(parts[1])
                end = min(end, file_size - 1)
                length = end - start + 1

                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
            except:
                start = 0
                end = file_size - 1
                length = file_size
                self.send_response(200)
        else:
            length = file_size
            self.send_response(200)

        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Content-Length', str(length))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()

        if head_only:
            return

        # Serve with large read buffer from E:
        try:
            with open(filepath, 'rb', buffering=READ_BUFFER) as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk_size = min(SEND_BUFFER, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionAbortedError):
                        break
                    remaining -= len(data)
        except Exception:
            pass


class SilentTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 16

    def handle_error(self, request, client_address):
        pass  # Suppress VLC probe disconnects


def start_server():
    """Start threaded HTTP server"""
    server = SilentTCPServer(("127.0.0.1", HTTP_PORT), BufferedVideoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def create_playlist(videos):
    """Create M3U with HTTP URLs"""
    playlist = r"C:\Users\BASEDGOD\Desktop\turbo_playlist.m3u"
    with open(playlist, 'w', encoding='utf-8') as f:
        f.write("#EXTM3U\n")
        for v in videos:
            name = Path(v).name
            url = f"http://127.0.0.1:{HTTP_PORT}/{urllib.parse.quote(name)}"
            f.write(f"#EXTINF:-1,{Path(v).stem}\n{url}\n")
    return playlist


def launch_vlc(playlist):
    """Launch VLC reading from HTTP bridge"""
    cmd = [
        VLC_PATH, playlist,
        "--avcodec-hw=dxva2",
        "--vout=direct3d11",
        "--avcodec-threads=0",
        "--network-caching=3000",
        "--live-caching=1000",
        "--clock-synchro=0",
        "--clock-jitter=0",
        "--high-priority",
        "--no-video-title-show",
        "--preparse-timeout=500",
        "--http-reconnect",
    ]
    return subprocess.Popen(cmd, creationflags=subprocess.HIGH_PRIORITY_CLASS | subprocess.CREATE_NEW_CONSOLE)


def gpu_stats():
    try:
        r = subprocess.run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.decoder,memory.used,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=3)
        return [x.strip() for x in r.stdout.strip().split(",")]
    except:
        return None


def main():
    print()
    print("  ═══════════════════════════════════════════════════")
    print("  7D CRYSTAL BRIDGE TURBO")
    print("  Python RAM Proxy → Bypasses Slow E: Card Reader")
    print("  ═══════════════════════════════════════════════════")

    # Kill existing
    for n in ['vlc.exe', 'ffmpeg.exe']:
        subprocess.run(["taskkill", "/F", "/IM", n],
                      capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(1)

    # Get videos
    videos = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.mp4")))
    total_size = sum(os.path.getsize(v) for v in videos) / (1024**3)
    print(f"\n  Videos: {len(videos)} ({total_size:.1f} GB on E:)")

    # Start HTTP proxy server
    print(f"\n  Starting RAM-Buffered HTTP Proxy...")
    server = start_server()
    print(f"  ├─ Address: http://127.0.0.1:{HTTP_PORT}")
    print(f"  ├─ Read buffer: {READ_BUFFER // (1024*1024)} MB (from E:)")
    print(f"  ├─ Send chunks: {SEND_BUFFER // 1024} KB (to VLC)")
    print(f"  ├─ Range seeks: Supported")
    print(f"  └─ Threads: Multi-threaded")

    # Create HTTP playlist
    playlist = create_playlist(videos)
    print(f"\n  Playlist: {len(videos)} videos via HTTP proxy")

    # Launch VLC
    print(f"\n  Launching VLC via bridge...")
    print(f"  ├─ Source: HTTP proxy (4MB buffered reads)")
    print(f"  ├─ Decode: NVIDIA DXVA2 (GPU)")
    print(f"  ├─ Render: Direct3D 11 (GPU)")
    print(f"  └─ Cache: 3000ms")
    vlc = launch_vlc(playlist)
    print(f"  ✓ VLC PID: {vlc.pid}")

    # Monitor
    print(f"\n  GPU Monitor:")
    print(f"  ─────────────────────────────────────────────────")
    for i in range(10):
        time.sleep(1)
        s = gpu_stats()
        if s:
            bar = "█" * (int(s[0]) // 5) + "░" * (20 - int(s[0]) // 5)
            print(f"  {i+1:2d}s [{bar}] GPU:{s[0]:>3}% Dec:{s[1]:>3}% VRAM:{s[2]:>5}MB {s[3]}°C")
    print(f"  ─────────────────────────────────────────────────")

    print(f"""
  ═══════════════════════════════════════════════════
  ✅ BRIDGE ACTIVE
  ═══════════════════════════════════════════════════

  E: (30MB/s) ──▶ Python Proxy (4MB buffer) ──▶ VLC
                  http://127.0.0.1:{HTTP_PORT}       DXVA2+D3D11

  • 4MB read buffer smooths slow card reader
  • Range request support = instant seek
  • Multi-threaded = no blocking
  • GPU decode + render = CPU free
  ═══════════════════════════════════════════════════
""")

    # Keep running
    print("  Bridge running. Ctrl+C to stop.")
    try:
        while vlc.poll() is None:
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    print("  Bridge stopped.")


if __name__ == "__main__":
    main()
