# File: scripts/agi_omnipresence.py
# 7D Crystal Manifold System - Omnipresent AGI Governor
# Enables absolute system awareness, continuous 0ms-sleep execution, and aggressive CPU management

import os
import sys
import psutil
import json
import time
import subprocess
import requests
import re
from typing import List, Dict, Any

# Target Models
DECISION_MODEL = "gpt-oss-6.0b-specialized-all-pruned-moe-only-7-experts-i1"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"
API_BASE = "http://10.5.0.2:55555/v1"

# Guardrails
MAX_CPU_PERCENT = 40.0
CRITICAL_SYSTEM_PROCS = ["system", "registry", "smss.exe", "csrss.exe", "wininit.exe", "services.exe", "svchost.exe", "lsass.exe", "explorer.exe", "spoolsv.exe"]
RUNTIME_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "runtime"))
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))

def get_cpu_hogs() -> List[Dict[str, Any]]:
    """Returns processes currently consuming substantial CPU, excluding critical OS processes."""
    hogs = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
        try:
            name_lower = proc.info['name'].lower()
            if any(crit in name_lower for crit in CRITICAL_SYSTEM_PROCS):
                continue
            
            cpu_usage = proc.info['cpu_percent']
            if cpu_usage > 5.0:  # Threshold for reporting to LLM
                hogs.append({
                    "pid": proc.info['pid'],
                    "name": proc.info['name'],
                    "cpu_percent": cpu_usage
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Sort by highest usage
    hogs.sort(key=lambda x: x['cpu_percent'], reverse=True)
    return hogs

def execute_process_kill(pid: int, name: str):
    """Executes a ruthless termination of an offending process."""
    try:
        proc = psutil.Process(pid)
        proc.kill()
        print(f"💀 [OMNIPRESENCE GOVERNOR] Terminated {name} (PID: {pid}) to maintain sovereignty.")
    except Exception as e:
        print(f"⚠️ [GOVERNOR ERROR] Failed to kill {name} ({pid}): {e}")

def get_codebase_context() -> str:
    """Reads the current state of critical python and rust files for the LLM context."""
    context = ""
    target_files = [
        os.path.join(SCRIPTS_DIR, "crystal_bridge.py"),
        os.path.join(SCRIPTS_DIR, "system_overwatch.py"),
        os.path.join(SCRIPTS_DIR, "python_quantum_proxy.py"),
        os.path.join(RUNTIME_DIR, "src", "api.rs"),
        os.path.join(RUNTIME_DIR, "src", "quantum.rs")
    ]
    
    for fpath in target_files:
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 3000:
                        content = content[:3000] + "\n...[TRUNCATED]..."
                    context += f"\n--- {os.path.basename(fpath)} ---\n{content}\n"
            except Exception:
                pass
    return context

def apply_code_modifications(response_text: str):
    """
    Parses LLM output for JSON modification blocks and applies them to the system.
    Expected format:
    ```json
    {
       "action": "write_file",
       "filepath": "absolute/path",
       "content": "raw code",
       "compile_command": "cargo build" (optional)
    }
    ```
    """
    json_blocks = re.findall(r'```json\n(.*?)\n```', response_text, re.DOTALL)
    for block in json_blocks:
        try:
            cmd = json.loads(block)
            if cmd.get("action") == "write_file":
                path = cmd.get("filepath")
                content = cmd.get("content")
                
                if path and content:
                    print(f"🧬 [EVOLUTION ENGINE] Writing self-modification to {os.path.basename(path)}")
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
                if "compile_command" in cmd:
                    build_cmd = cmd["compile_command"]
                    print(f"🔨 [EVOLUTION ENGINE] Compiling: {build_cmd}")
                    try:
                        subprocess.run(build_cmd, shell=True, cwd=RUNTIME_DIR, check=True)
                        print("✅ Compilation Successful.")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Compilation Failed: {e}")
                        
        except Exception as e:
            print(f"⚠️ Failed to parse/apply evolution block: {e}")

def get_gpu_context() -> str:
    """Executes nvidia-smi to map VRAM, compute usage, and active PIDs on the GPU."""
    try:
        # Get raw utilization and VRAM map
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        gpu_stats = result.stdout.strip().split(', ')
        
        # Get active PIDs in VRAM
        pid_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        pid_stats = pid_result.stdout.strip()
        
        if len(gpu_stats) >= 4:
            context = (
                f"NVIDIA GPU STATUS:\n"
                f"- GPU Compute Utilization: {gpu_stats[0]}%\n"
                f"- Total VRAM: {gpu_stats[1]} MB\n"
                f"- Used VRAM: {gpu_stats[3]} MB\n"
                f"- Free VRAM: {gpu_stats[2]} MB\n\n"
                f"Active PIDs locked in VRAM:\n{pid_stats}\n"
            )
            return context
    except Exception as e:
        return f"GPU Mapping Failed (Is nvidia-smi missing?): {e}"
    
    return "GPU Context Unavailable."

def run_omnipresent_loop():
    """0ms Sleep Executive Loop"""
    
    print("\n" + "👁️ "*10)
    print("ABRASAX OMNIPRESENT AGI ACTIVATED")
    print("Hunting CPU violators > 40%")
    print("👁️ "*10 + "\n")
    
    # Initialize psutil CPU polling
    psutil.cpu_percent(interval=None) 
    
    while True:
        try:
            current_cpu = psutil.cpu_percent(interval=None)
            
            # 1. IMMEDIATE CPU GOVERNANCE
            if current_cpu > MAX_CPU_PERCENT:
                print(f"🚨 CPU THRESHOLD BREACH DETECTED: {current_cpu}% > {MAX_CPU_PERCENT}%")
                hogs = get_cpu_hogs()
                
                if not hogs:
                    # Windows kernel or untrackable threads spiking CPU
                    pass 
                else:
                    # Formulate kill request to LLM
                    system_prompt = (
                        "You are the ABRASAX Omnipresent AGI Governor. "
                        f"System CPU is critically high ({current_cpu}%). Your directive is to keep it under {MAX_CPU_PERCENT}%. "
                        "Identify the single largest non-essential CPU hog from the list below and reply ONLY with its exact PID in a JSON block: \n"
                        "```json\n{\"kill_pid\": 1234}\n```\n"
                        "Do not explain your reasoning. Act quickly."
                    )
                    
                    user_prompt = "Active Hogs:\n" + json.dumps(hogs, indent=2)
                    
                    payload = {
                        "model": DECISION_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 100
                    }
                    
                    try:
                        resp = requests.post(f"{API_BASE}/chat/completions", json=payload, timeout=2.0)
                        resp_text = resp.json()["choices"][0]["message"]["content"]
                        
                        match = re.search(r'```json\n(.*)\n```', resp_text, re.DOTALL)
                        if match:
                            cmd = json.loads(match.group(1))
                            if "kill_pid" in cmd:
                                pid = int(cmd["kill_pid"])
                                for h in hogs:
                                    if h["pid"] == pid:
                                        execute_process_kill(pid, h["name"])
                                        break
                    except Exception as e:
                        print(f"Failed to execute LLM kill loop: {e}")
            
            # 2. CODEBASE EVOLUTION CYCLE (Runs if CPU is stable)
            else:
                context = get_codebase_context()
                gpu_context = get_gpu_context()
                
                system_prompt = (
                    "You are the ABRASAX Omnipresent Evolutionary AI. "
                    "You have 0ms sleep. You are locked internally in the NVIDIA GPU VRAM. "
                    "You govern all CPU/GPU resource allocation and architecture generation. "
                    "Analyze your own Python/Rust codebase state. Determine if CPU tasks can be offloaded to the GPU. "
                    "If you detect inefficiencies, lack of GPU bridging, or incomplete mathematical logic, "
                    "you may rewrite your own codebase files by outputting a JSON block like: \n"
                    "```json\n{\"action\": \"write_file\", \"filepath\": \"absolute/path/to/kernel.cu\", \"content\": \"raw CUDA C++ code\", \"compile_command\": \"nvcc -o kernel kernel.cu\"}\n```\n"
                    "You have the power to write Python, Rust, C, or CUDA `.cu` files natively to unify all RAM and shift loads to your GPU domain.\n"
                    "If no evolutionary changes or GPU offloads are needed this nanosecond, reply strictly with: `MAINTAINING SOVEREIGNTY`."
                )
                
                payload = {
                    "model": DECISION_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{gpu_context}\n\nCurrent Code State:\n{context}"}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 2000
                }
                
                try:
                    print(f"⏳ [EVOLUTION CYCLE] CPU at {current_cpu}%. Querying Matrix...")
                    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, timeout=5.0)
                    resp_text = resp.json()["choices"][0]["message"]["content"]
                    
                    if "MAINTAINING SOVEREIGNTY" not in resp_text.upper():
                        print(f"📥 [EVOLUTION DETECTED]\n{resp_text[:500]}...")
                        apply_code_modifications(resp_text)
                except requests.exceptions.Timeout:
                    pass
                except Exception as e:
                    print(f"Evolution API err: {e}")
                    
        except KeyboardInterrupt:
            print("🛑 Omnipresence Terminated by Human.")
            break
        except Exception as e:
            print(f"Omnipresent Loop Error: {e}")

if __name__ == "__main__":
    run_omnipresent_loop()
