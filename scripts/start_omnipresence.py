#!/usr/bin/env python3
"""Start Omnipresence Realtime via Python - unbuffered output."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

def main():
    os.chdir(ROOT)
    os.environ["PYTHONUNBUFFERED"] = "1"
    print("[OMNI] Starting Omnipresence Realtime...", flush=True)

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mss", "psutil", "requests"], check=False)

    # -u = unbuffered stdout/stderr so output streams live
    subprocess.run([sys.executable, "-u", str(SCRIPTS / "omnipresence_realtime.py")], cwd=ROOT)

if __name__ == "__main__":
    main()
