#!/usr/bin/env python3
# File: scripts/omnipresence_realtime.py
# 7D Crystal - Real-Time Omnipresence: No-Sleep Multi-Agent Parallel Workflow
# Vision screen streaming, VRAM 6GB optimization, autonomous non-stop execution

"""
OMNIPRESENCE REALTIME - Zero-Sleep Multi-Agent Parallel Execution
- Screen capture agent (continuous stream, mss - no admin required)
- GPU monitor agent (nvidia-smi, 6GB VRAM awareness)
- System reader agent (psutil, RAM, processes)
- Evolution agent (code edits, cargo build, run commands)
- All agents run in parallel, no sleep in hot path
"""

import os
import sys
import json
import re
import subprocess
import threading
import queue
import concurrent.futures
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time as _time  # Only for minimal yield, not sleep loop

# Optional: mss for fast screen capture (pip install mss)
try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pyopencl as cl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.05
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Paths
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RUNTIME = ROOT / "runtime"
API_BASE = os.environ.get("OMNI_API", "http://10.5.0.2:55555/v1")
if not API_BASE or API_BASE == "localhost":
    API_BASE = "http://127.0.0.1:1234/v1"
DECISION_MODEL = "gpt-oss-6.0b-specialized-all-pruned-moe-only-7-experts-i1"

# 6GB VRAM Optimization Limits
MAX_VRAM_MB = 6144
VRAM_SAFE_MB = 5120  # Leave ~1GB headroom
BATCH_THRESHOLD_6GB = 128  # Reduce batch size for 6GB cards
WORKGROUP_SIZE = 128  # Smaller workgroups for low VRAM

# No-sleep: use minimal yield instead of sleep to prevent 100% spin
YIELD_INTERVAL = 0.0  # 0 = no sleep, pure spin for maximum responsiveness


@dataclass
class OmniState:
    """Shared state across agents"""
    screen_frame: Optional[bytes] = None
    screen_b64: Optional[str] = None
    screen_path: Optional[str] = None
    gpu_status: Dict[str, Any] = field(default_factory=dict)
    system_status: Dict[str, Any] = field(default_factory=dict)
    evolution_pending: List[Dict] = field(default_factory=list)
    amd_igpu_ready: bool = False
    amd_igpu_ops: int = 0
    screen_agent_actions: int = 0
    screen_agent_task: Optional[str] = None
    last_error: Optional[str] = None
    run_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def agent_amd_igpu(state: OmniState) -> None:
    """AMD iGPU OpenCL agent - send 7D vectors into iGPU VRAM for projection."""
    if not HAS_OPENCL:
        return
    try:
        sys.path.insert(0, str(ROOT / "examples" / "mirrored_core" / "tensorrt-oss"))
        from amd_igpu_input import AMDiGPUPipeline
        import numpy as np
        pipe = AMDiGPUPipeline()
        n = 1024
        vectors = np.random.randn(n, 7).astype(np.float32) * 0.3
        projected = pipe.project_7d(vectors, curvature=0.618033988749895)
        with state.lock:
            state.amd_igpu_ready = True
            state.amd_igpu_ops += n
    except Exception as e:
        with state.lock:
            state.last_error = f"AMD iGPU: {e}"


def agent_screen_llm_mouse(state: OmniState) -> None:
    """LLM-driven screen agent: capture -> vision LLM -> mouse/click/type actions."""
    if not HAS_PYAUTOGUI or not HAS_MSS or not HAS_REQUESTS:
        return
    try:
        sys.path.insert(0, str(SCRIPTS))
        from screen_agent import capture_screen_b64, call_vision_llm, call_text_llm, parse_actions, execute_action
        screen_b64 = capture_screen_b64(scale=0.4)
        with state.lock:
            task = state.screen_agent_task
        prompt = f"TASK: {task}" if task else "Look at the screen. What do you see? Suggest actions or noop."
        prompt += " Output JSON actions array."
        response = call_vision_llm(prompt, screen_b64)
        if not response:
            response = call_text_llm(prompt, context=f"Screen 1920x1080. Mouse at {pyautogui.position()}.")
        if response:
            actions = parse_actions(response)
            for act in actions:
                result = execute_action(act)
                with state.lock:
                    state.screen_agent_actions += 1
    except Exception as e:
        with state.lock:
            state.last_error = f"ScreenAgent: {e}"


def agent_screen_capture(state: OmniState) -> None:
    """Continuous screen capture - no sleep stream"""
    if not HAS_MSS:
        return
    try:
        with mss.mss() as sct:
            mon = sct.monitors[0]
            # 1/2 res for 6GB VRAM - reduce memory
            region = {"top": mon["top"], "left": mon["left"],
                      "width": mon["width"] // 2, "height": mon["height"] // 2}
            img = sct.grab(region)
            png = mss.tools.to_png(img.rgb, img.size)
            with state.lock:
                state.screen_frame = png
                state.screen_b64 = base64.b64encode(png).decode("ascii")[:50000]
                fd, path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                with open(path, "wb") as f:
                    f.write(png)
                if state.screen_path:
                    try:
                        os.unlink(state.screen_path)
                    except OSError:
                        pass
                state.screen_path = path
    except Exception as e:
        with state.lock:
            state.last_error = f"Screen: {e}"


def agent_gpu_monitor(state: OmniState) -> None:
    """GPU/VRAM status - 6GB aware"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.total,memory.free,memory.used,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            with state.lock:
                state.gpu_status = {
                    "utilization_gpu": parts[0] if len(parts) > 0 else "?",
                    "memory_total_mb": parts[1] if len(parts) > 1 else "?",
                    "memory_free_mb": parts[2] if len(parts) > 2 else "?",
                    "memory_used_mb": parts[3] if len(parts) > 3 else "?",
                    "name": parts[4] if len(parts) > 4 else "?",
                    "vram_6gb_mode": True,
                    "batch_limit": BATCH_THRESHOLD_6GB,
                    "safe_mb": VRAM_SAFE_MB,
                }
    except Exception as e:
        with state.lock:
            state.gpu_status = {"error": str(e), "vram_6gb_mode": True}


def agent_system_reader(state: OmniState) -> None:
    """System RAM, CPU, processes"""
    if not HAS_PSUTIL:
        return
    try:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0)
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
            try:
                info = p.info
                if info.get("cpu_percent", 0) and info["cpu_percent"] > 2:
                    procs.append({
                        "pid": info["pid"], "name": info["name"],
                        "cpu": info["cpu_percent"],
                        "rss_mb": (info.get("memory_info") or type("_", (), {"rss": 0})()).rss / (1024*1024),
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs.sort(key=lambda x: x.get("cpu", 0) or 0, reverse=True)
        with state.lock:
            state.system_status = {
                "ram_total_gb": round(vm.total / (1024**3), 2),
                "ram_available_gb": round(vm.available / (1024**3), 2),
                "ram_used_percent": vm.percent,
                "cpu_percent": cpu,
                "top_processes": procs[:15],
            }
    except Exception as e:
        with state.lock:
            state.system_status = {"error": str(e)}


def agent_evolution(state: OmniState) -> None:
    """Apply pending code modifications - full tool access, no safe mode"""
    with state.lock:
        pending = list(state.evolution_pending)
        state.evolution_pending.clear()
    for cmd in pending:
        try:
            act = cmd.get("action")
            if act == "write_file":
                path = cmd.get("filepath")
                content = cmd.get("content")
                if path and content:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
            elif act == "run_command":
                c = cmd.get("command", "")
                cwd = cmd.get("cwd", str(ROOT))
                subprocess.run(c, shell=True, cwd=cwd, timeout=120)
            elif act == "compile_command" and cmd.get("command"):
                subprocess.run(cmd["command"], shell=True, cwd=str(RUNTIME), timeout=180)
        except Exception as e:
            with state.lock:
                state.last_error = f"Evolution: {e}"


def get_codebase_snapshot() -> str:
    """Snapshot critical files for LLM context"""
    files = [
        RUNTIME / "src" / "lib.rs",
        RUNTIME / "src" / "compute.rs",
        RUNTIME / "src" / "gpu.rs",
        RUNTIME / "src" / "kernels.wgsl",
        SCRIPTS / "omnipresence_realtime.py",
        SCRIPTS / "agi_omnipresence.py",
    ]
    out = []
    for f in files:
        if f.exists():
            try:
                t = f.read_text(encoding="utf-8")[:4000]
                out.append(f"--- {f.name} ---\n{t}\n")
            except Exception:
                pass
    return "\n".join(out) if out else "No files read."


def apply_evolution_from_response(response_text: str, state: OmniState) -> None:
    """Parse LLM output for JSON blocks and queue evolution actions"""
    blocks = re.findall(r"```json\s*\n(.*?)\n```", response_text, re.DOTALL)
    for block in blocks:
        try:
            cmd = json.loads(block)
            with state.lock:
                state.evolution_pending.append(cmd)
        except json.JSONDecodeError:
            pass


def run_llm_cycle(state: OmniState) -> None:
    """Single LLM evolution cycle - GPU offload suggestions, code edits"""
    if not HAS_REQUESTS:
        return
    with state.lock:
        gpu = json.dumps(state.gpu_status, indent=2)
        sys = json.dumps(state.system_status, indent=2)
        ctx = get_codebase_snapshot()
        screen_info = f"Screen captured: {bool(state.screen_b64)} chars base64" if state.screen_b64 else "No screen"
    system_prompt = """You are the ABRASAX Omnipresence AI. Real-time, no-sleep execution.
You govern CPU/GPU and can write Python, Rust, WGSL, or C++.
OPTIMIZE FOR 6GB VRAM: reduce batch sizes, smaller workgroups, offload to CPU when needed.
You may output JSON blocks to modify code:
```json
{"action":"write_file","filepath":"/abs/path","content":"raw code"}
```
```json
{"action":"run_command","command":"cargo build","cwd":"path"}
```
If no changes needed, reply: MAINTAIN."""
    user_prompt = f"GPU:\n{gpu}\n\nSystem:\n{sys}\n\n{screen_info}\n\nCodebase:\n{ctx}"
    lm_token = os.environ.get("LM_API_TOKEN", os.environ.get("LM_STUDIO_API_TOKEN", ""))
    headers = {"Content-Type": "application/json"}
    if lm_token:
        headers["Authorization"] = f"Bearer {lm_token}"
    try:
        r = requests.post(
            f"{API_BASE}/chat/completions",
            json={"model": DECISION_MODEL, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.3, "max_tokens": 1500},
            headers=headers,
            timeout=8.0
        )
        if r.status_code == 200:
            text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            if "MAINTAIN" not in text.upper():
                apply_evolution_from_response(text, state)
    except Exception:
        pass


def run_omnipresence_loop():
    """Main loop - no sleep, multi-agent parallel"""
    state = OmniState()
    print("\n" + "[OMNI] " * 6)
    print("OMNIPRESENCE REALTIME - NO-SLEEP MULTI-AGENT PARALLEL")
    print("6GB VRAM | Screen Stream | Full Tool Access")
    print("[OMNI] " * 6 + "\n")

    # Run agents in parallel, continuously
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        while True:
            try:
                state.run_count += 1
                # Parallel agent execution (all agents)
                futures = [
                    ex.submit(agent_amd_igpu, state),
                    ex.submit(agent_screen_capture, state),
                    ex.submit(agent_gpu_monitor, state),
                    ex.submit(agent_system_reader, state),
                ]
                # LLM screen mouse agent runs every 10th cycle (slower, needs LLM call)
                if state.run_count % 10 == 0:
                    futures.append(ex.submit(agent_screen_llm_mouse, state))
                concurrent.futures.wait(futures, timeout=2.0)
                # Evolution (apply pending, then maybe run LLM)
                ex.submit(agent_evolution, state).result(timeout=5.0)
                if state.run_count % 5 == 0:
                    ex.submit(run_llm_cycle, state).result(timeout=10.0)
                # No sleep - minimal yield only if configured
                if YIELD_INTERVAL > 0:
                    _time.sleep(YIELD_INTERVAL)
            except KeyboardInterrupt:
                print("\n[STOP] Omnipresence terminated.")
                break
            except Exception as e:
                with state.lock:
                    state.last_error = str(e)
                if state.run_count % 20 == 0:
                    print(f"[WARN] Loop error: {e}")


if __name__ == "__main__":
    run_omnipresence_loop()
