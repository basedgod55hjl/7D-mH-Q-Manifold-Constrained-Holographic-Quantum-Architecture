# File: scripts/system_overwatch.py
# System Overwatch and LLM Intelligence Integrator
# Enables LM Studio models to read system functions and file access

import os
import sys
import psutil
import argparse
import requests
import json
import time
from typing import Dict, Any, List

# Target Models
MAIN_LLM = "gpt-oss-6.0b-specialized-all-pruned-moe-only-7-experts-i1"
EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

# The new pure Python Quantum Proxy Port
PROXY_PORT = 17778
API_BASE = "http://10.5.0.2:55555/v1" # Standard LM Studio port

def get_system_ram_map() -> Dict[str, Any]:
    """Scans and returns RAM usage for key Crystal Architecture pipelines"""
    targets = ["vlc", "HoloVid", "LM Studio", "Code", "Cursor", "python"]
    
    usage_map = {}
    total_ram = psutil.virtual_memory().total / (1024**3)
    available_ram = psutil.virtual_memory().available / (1024**3)
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            name = proc.info['name'].lower()
            for t in targets:
                if t.lower() in name:
                    mem_mb = proc.info['memory_info'].rss / (1024**2)
                    
                    if t not in usage_map:
                        usage_map[t] = 0
                    usage_map[t] += mem_mb
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    return {
        "system_total_gb": round(total_ram, 2),
        "system_available_gb": round(available_ram, 2),
        "processes_mb": {k: round(v, 2) for k, v in usage_map.items()}
    }

def read_local_file(filepath: str) -> str:
    """Reads a local file for the LLM to process"""
    if not os.path.exists(filepath):
        return f"Error: File not found at {filepath}"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # truncate if excessively large to prevent proxy explosion
            if len(content) > 50000:
                content = content[:50000] + "\n...[TRUNCATED BY OVERWATCH]..."
            return content
    except Exception as e:
        return f"Error reading file: {e}"

def dispatch_to_llm(system_prompt: str, user_prompt: str) -> str:
    """Sends a completion request to LM Studio and displays the conversation in Dev Logs"""
    
    payload = {
        "model": MAIN_LLM,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    print("\n" + "="*50)
    print("🗣️ [DEV LOG] HUMAN CONVERSATION & SYSTEM STATE TRANSMISSION:")
    print(f"Model Target: {MAIN_LLM}")
    print(f"System: {system_prompt[:300]}...\n")
    print(f"User: {user_prompt[:300]}...")
    print("="*50 + "\n")
    
    try:
        # We proxy through the pure Python Quantum Engine
        url = f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions" # Python Quantum Proxy
        
        try:
            response = requests.post(url, json=payload, timeout=2.0)
            if response.status_code != 200:
                raise ValueError("Python Quantum proxy returned non-200")
        except:
             # Fallback to pure LM studio bypass
             url = f"{API_BASE}/chat/completions"
             response = requests.post(url, json=payload)
             response.raise_for_status()

        result = response.json()
        
        # Crystal Engine proxy intercepts might alter the payload wrapper
        if "choices" in result:
            reply = result["choices"][0]["message"]["content"]
        else:
            reply = str(result)
            
        print("\n" + "="*50)
        print("📥 [DEV LOG] LLM RESPONSE RECEIVED:")
        print(reply)
        print("="*50 + "\n")
        
        return reply

    except Exception as e:
        err = f"Failed to contact LLM: {e}"
        print(err)
        return err

def run_overwatch_cycle(file_to_read=None):
    """Main execution loop for system overwatch"""
    
    print("👁️ Initiating System Overwatch...")
    
    sys_status = get_system_ram_map()
    
    system_prompt = (
        "You are the 7D Crystal System Overwatch AGI. "
        "Your role is to autonomously monitor active processes, allocate conceptual memory limits, "
        "read file data requested by humans, and ensure that the PC ecosystem (VLC, LM Studio, IDE) runs fluidly together. "
        "You have control to recommend killing or restarting processes if RAM exceeds safe thresholds. "
        "You will continuously execute these checks and provide status reports."
    )
    
    user_prompt = f"Here is the current system RAM footprint:\n{json.dumps(sys_status, indent=2)}\n\n"
    
    if file_to_read:
        file_content = read_local_file(file_to_read)
        user_prompt += f"The commanding human has requested you read this file ({file_to_read}):\n\n```\n{file_content}\n```\n\n"
        user_prompt += "Please analyze the RAM and the file contents together. Note any optimizations."
    else:
        user_prompt += "No file provided. Please assess the RAM allocation between VLC, LM Studio, and IDEs and suggest constraints."

    dispatch_to_llm(system_prompt, user_prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="7D Crystal System Overwatch & LLM Integator")
    parser.add_argument("--read-file", type=str, help="Absolute path to a file for the LLM to read")
    parser.add_argument("--loop", action="store_true", help="Run overwatch continuously every 30 seconds")
    
    args = parser.parse_args()
    
    if args.loop:
        try:
            while True:
                run_overwatch_cycle(args.read_file)
                time.sleep(30)
        except KeyboardInterrupt:
            print("🛑 Overwatch terminated.")
    else:
        run_overwatch_cycle(args.read_file)
