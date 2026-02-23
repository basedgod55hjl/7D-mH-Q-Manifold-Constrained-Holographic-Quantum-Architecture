#!/usr/bin/env python3
"""
AUTONOMOUS AGENT - Self-Read, Learn, Reason, Execute
4 parallel agents: Reader, Reasoner (DeepSeek-R1), Embedder (Nomic), Executor
Reward loop for correct embeddings and self-upgrade metrics
Nomic embeddings live into AMD iGPU VRAM tensor via OpenCL
"""

import os
import sys
import json
import re
import time
import threading
import concurrent.futures
import subprocess
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import requests
import psutil
import yaml

# AMD iGPU pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "mirrored_core" / "tensorrt-oss"))
try:
    from amd_igpu_input import AMDiGPUPipeline
    HAS_AMD = True
except ImportError:
    HAS_AMD = False

try:
    import pyopencl as cl
except ImportError:
    cl = None

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SYSCONFIG = Path(r"C:\Users\BASEDGOD\Desktop\SYSTEM CONFIG")
LOG_DIR = SCRIPTS / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Load config
CONFIG_PATH = SCRIPTS / "action_config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# LM Studio auth
LM_TOKEN = os.environ.get("LM_API_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
# Auth disabled on LM Studio restart - clear token to avoid 401
if not LM_TOKEN or LM_TOKEN.startswith("sk-"):
    LM_TOKEN = ""
LM_BASE = CONFIG["endpoints"]["primary"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "autonomous.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AGENT")


def lm_headers():
    h = {"Content-Type": "application/json"}
    if LM_TOKEN:
        h["Authorization"] = f"Bearer {LM_TOKEN}"
    return h


def lm_chat(messages, model=None, max_tokens=2048, temperature=0.3):
    model = model or CONFIG["models"]["reasoning"]
    try:
        r = requests.post(
            f"{LM_BASE}/chat/completions",
            json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            headers=lm_headers(), timeout=60,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"LM chat error: {e}")
    return None


def lm_embed(texts, model=None):
    model = model or CONFIG["models"]["embedding"]
    if isinstance(texts, str):
        texts = [texts]
    try:
        r = requests.post(
            f"{LM_BASE}/embeddings",
            json={"model": model, "input": texts},
            headers=lm_headers(), timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            return [d["embedding"] for d in data["data"]]
    except Exception as e:
        log.warning(f"Embed error: {e}")
    return None


def log_jsonl(path, entry):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ====================================================================
# STATE
# ====================================================================
@dataclass
class AgentState:
    files_read: Dict[str, str] = field(default_factory=dict)  # path -> hash
    file_contents: Dict[str, str] = field(default_factory=dict)
    embeddings_cache: Dict[str, List[float]] = field(default_factory=dict)
    rewards: float = 0.0
    metrics: Dict[str, float] = field(default_factory=lambda: {"embed_quality": 0, "compile_ok": 0, "actions": 0, "cycles": 0})
    pending_actions: List[Dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


# ====================================================================
# AGENT 1: READER - Scans SYSTEM CONFIG, reads files, hashes for change detection
# ====================================================================
def agent_reader(state: AgentState):
    scan_dirs = [Path(d) for d in CONFIG["agents"][0]["scan_dirs"]]
    exts = set(CONFIG["agents"][0]["extensions"])
    new_files = 0
    for d in scan_dirs:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.suffix in exts and f.is_file() and f.stat().st_size < 500_000:
                fpath = str(f)
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    h = hashlib.md5(content.encode()).hexdigest()
                    with state.lock:
                        if state.files_read.get(fpath) != h:
                            state.files_read[fpath] = h
                            state.file_contents[fpath] = content[:8000]
                            new_files += 1
                except Exception:
                    pass
    log.info(f"[READER] Scanned {len(state.files_read)} files, {new_files} new/changed")


# ====================================================================
# AGENT 2: REASONER - DeepSeek-R1 reads code, reasons, suggests actions
# ====================================================================
def agent_reasoner(state: AgentState):
    with state.lock:
        recent = dict(list(state.file_contents.items())[-5:])
        metrics = dict(state.metrics)

    if not recent:
        return

    context = ""
    for path, content in recent.items():
        context += f"\n--- {Path(path).name} ---\n{content[:2000]}\n"

    prompt = f"""You are the ABRASAX autonomous reasoning agent (DeepSeek-R1).
You have read these files from the system:
{context}

Current metrics: {json.dumps(metrics)}

Your task:
1. Analyze the code for improvements, GPU offloads, or missing connections
2. Suggest concrete actions as JSON blocks
3. If code compiles and embeddings improve, you get rewards

Output actions as:
```json
{{"action":"write_file","filepath":"abs/path","content":"code"}}
```
```json
{{"action":"run_command","command":"cargo build","cwd":"path"}}
```
If nothing to do: MAINTAIN"""

    response = lm_chat([
        {"role": "system", "content": "You are an autonomous code reasoning agent. Output JSON action blocks."},
        {"role": "user", "content": prompt},
    ])

    if response and "MAINTAIN" not in response.upper():
        blocks = re.findall(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
        for block in blocks:
            try:
                cmd = json.loads(block)
                with state.lock:
                    state.pending_actions.append(cmd)
            except json.JSONDecodeError:
                pass
        log.info(f"[REASONER] Generated {len(blocks)} actions")
        log_jsonl(LOG_DIR / "thought_stream.jsonl", {
            "ts": datetime.now().isoformat(), "agent": "reasoner",
            "response": response[:2000], "actions": len(blocks),
        })
    else:
        log.info("[REASONER] MAINTAIN")


# ====================================================================
# AGENT 3: EMBEDDER - Nomic embeds -> AMD iGPU VRAM tensor
# ====================================================================
def agent_embedder(state: AgentState):
    with state.lock:
        texts_to_embed = []
        paths_to_embed = []
        for path, content in state.file_contents.items():
            if path not in state.embeddings_cache:
                texts_to_embed.append(content[:1000])
                paths_to_embed.append(path)
        texts_to_embed = texts_to_embed[:64]  # batch limit
        paths_to_embed = paths_to_embed[:64]

    if not texts_to_embed:
        return

    embeddings = lm_embed(texts_to_embed)
    if not embeddings:
        log.warning("[EMBEDDER] No embeddings returned")
        return

    # Send embeddings into AMD iGPU VRAM
    vram_loaded = False
    if HAS_AMD:
        try:
            pipe = AMDiGPUPipeline()
            emb_array = np.array(embeddings, dtype=np.float32)
            # Project embedding dimensions through 7D manifold
            # Take first 7 dims or reshape
            n = emb_array.shape[0]
            if emb_array.shape[1] >= 7:
                vram_input = emb_array[:, :7]
            else:
                vram_input = np.zeros((n, 7), dtype=np.float32)
                vram_input[:, :emb_array.shape[1]] = emb_array
            projected = pipe.project_7d(vram_input, curvature=0.618033988749895)
            vram_loaded = True
            log.info(f"[EMBEDDER] {n} embeddings -> AMD iGPU VRAM (7D projected)")
        except Exception as e:
            log.warning(f"[EMBEDDER] AMD iGPU error: {e}")

    # Cache and compute reward
    reward = 0.0
    with state.lock:
        for i, path in enumerate(paths_to_embed):
            if i < len(embeddings):
                state.embeddings_cache[path] = embeddings[i]
                norm = np.linalg.norm(embeddings[i])
                if norm > 0.5:
                    reward += CONFIG["rewards"]["embedding_quality"]["reward"]
        state.rewards += reward
        state.metrics["embed_quality"] = len(state.embeddings_cache)

    log_jsonl(LOG_DIR / "embeddings.jsonl", {
        "ts": datetime.now().isoformat(), "count": len(texts_to_embed),
        "vram_loaded": vram_loaded, "reward": reward,
    })
    log.info(f"[EMBEDDER] Embedded {len(texts_to_embed)} texts, reward +{reward:.1f}, total {state.rewards:.1f}")


# ====================================================================
# AGENT 4: EXECUTOR - Applies actions, compiles, runs commands
# ====================================================================
def agent_executor(state: AgentState):
    with state.lock:
        actions = list(state.pending_actions)
        state.pending_actions.clear()

    if not actions:
        return

    for cmd in actions:
        act = cmd.get("action")
        try:
            if act == "write_file":
                path = cmd.get("filepath")
                content = cmd.get("content")
                if path and content:
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    log.info(f"[EXECUTOR] Wrote {Path(path).name}")
                    with state.lock:
                        state.metrics["actions"] += 1

            elif act == "run_command":
                c = cmd.get("command", "")
                cwd = cmd.get("cwd", str(ROOT))
                r = subprocess.run(c, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120)
                ok = r.returncode == 0
                log.info(f"[EXECUTOR] Ran '{c[:50]}' -> {'OK' if ok else 'FAIL'}")
                if ok:
                    with state.lock:
                        state.rewards += CONFIG["rewards"]["self_upgrade"]["reward"]
                        state.metrics["compile_ok"] += 1
                with state.lock:
                    state.metrics["actions"] += 1

            elif act == "compile":
                c = cmd.get("command", "cargo build")
                r = subprocess.run(c, shell=True, cwd=str(ROOT / "runtime"), capture_output=True, text=True, timeout=180)
                ok = r.returncode == 0
                log.info(f"[EXECUTOR] Compile '{c[:50]}' -> {'OK' if ok else 'FAIL'}")
                if ok:
                    with state.lock:
                        state.rewards += CONFIG["rewards"]["self_upgrade"]["reward"]
                        state.metrics["compile_ok"] += 1

        except Exception as e:
            log.warning(f"[EXECUTOR] Error: {e}")

    log_jsonl(LOG_DIR / "metrics.jsonl", {
        "ts": datetime.now().isoformat(), "actions": len(actions),
        "metrics": state.metrics, "rewards": state.rewards,
    })


# ====================================================================
# MAIN LOOP
# ====================================================================
def run_autonomous():
    state = AgentState()
    log.info("=" * 60)
    log.info("AUTONOMOUS AGENT - Self-Read / Learn / Reason / Execute")
    log.info(f"Reasoning: {CONFIG['models']['reasoning']}")
    log.info(f"Embedding: {CONFIG['models']['embedding']}")
    log.info(f"AMD iGPU: {'YES' if HAS_AMD else 'NO'}")
    log.info(f"LM Token: {'SET' if LM_TOKEN else 'MISSING'}")
    log.info("=" * 60)

    cycle = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        while True:
            cycle += 1
            state.metrics["cycles"] = cycle
            try:
                # All 4 agents in parallel
                futures = [
                    ex.submit(agent_reader, state),
                    ex.submit(agent_embedder, state),
                ]
                # Reasoner every 3rd cycle, executor every cycle
                if cycle % 3 == 0:
                    futures.append(ex.submit(agent_reasoner, state))
                futures.append(ex.submit(agent_executor, state))

                concurrent.futures.wait(futures, timeout=30)

                if cycle % 10 == 0:
                    log.info(f"[LOOP] Cycle {cycle} | Files: {len(state.files_read)} | "
                             f"Embeds: {len(state.embeddings_cache)} | "
                             f"Rewards: {state.rewards:.1f} | "
                             f"Actions: {state.metrics['actions']}")

            except KeyboardInterrupt:
                log.info("[STOP] Autonomous agent terminated.")
                break
            except Exception as e:
                log.warning(f"[LOOP] Error: {e}")


if __name__ == "__main__":
    run_autonomous()
