#!/usr/bin/env python3
"""
DeepSeek-R1 4x Parallel Agent - Full VRAM, Streaming
Only 1 model: deepseek/deepseek-r1-0528-qwen3-8b (5GB in VRAM)
4 parallel streams hitting the same model simultaneously
Self-read, learn, reason, execute - no restrictions
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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import requests
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "mirrored_core" / "tensorrt-oss"))
try:
    from amd_igpu_input import AMDiGPUPipeline
    HAS_AMD = True
except ImportError:
    HAS_AMD = False

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
SYSCONFIG = Path(r"C:\Users\BASEDGOD\Desktop\SYSTEM CONFIG")
LOG_DIR = SCRIPTS / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "deepseek/deepseek-r1-0528-qwen3-8b"
EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
LM_BASE = "http://127.0.0.1:1234/v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "deepseek_4x.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("4xAGENT")


def lm_stream(messages, on_token=None):
    """Stream tokens from DeepSeek-R1."""
    try:
        r = requests.post(
            f"{LM_BASE}/chat/completions",
            json={"model": MODEL, "messages": messages, "temperature": 0.3,
                  "max_tokens": 2048, "stream": True},
            stream=True, timeout=120,
        )
        full = ""
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk.strip() == "[DONE]":
                break
            try:
                d = json.loads(chunk)
                token = d["choices"][0].get("delta", {}).get("content", "")
                if token:
                    full += token
                    if on_token:
                        on_token(token)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        return full
    except Exception as e:
        log.warning(f"Stream error: {e}")
        return None


def lm_chat(messages):
    """Non-streaming call."""
    try:
        r = requests.post(
            f"{LM_BASE}/chat/completions",
            json={"model": MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 2048},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        log.warning(f"Chat error: {e}")
    return None


def log_jsonl(path, entry):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


@dataclass
class State:
    files_read: Dict[str, str] = field(default_factory=dict)
    file_contents: Dict[str, str] = field(default_factory=dict)
    pending_actions: List[Dict] = field(default_factory=list)
    rewards: float = 0.0
    cycle: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


SCAN_DIRS = [
    SYSCONFIG / "00_KERNEL",
    SYSCONFIG / "01_RUNTIME",
    SYSCONFIG / "09_AGI_CONFIG",
    SYSCONFIG / "12_GPU_COMPUTE",
    SYSCONFIG / "LLM_ENGINE",
    ROOT / "runtime" / "src",
    ROOT / "scripts",
]
SCAN_EXTS = {".py", ".rs", ".cu", ".yaml", ".json", ".toml", ".wgsl"}


# ====================================================================
# AGENT 1: READER - scan all files, detect changes
# ====================================================================
def agent_reader(state: State):
    import hashlib
    new = 0
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in d.rglob("*"):
            if f.suffix in SCAN_EXTS and f.is_file() and f.stat().st_size < 500_000:
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    h = hashlib.md5(content.encode()).hexdigest()
                    with state.lock:
                        if state.files_read.get(str(f)) != h:
                            state.files_read[str(f)] = h
                            state.file_contents[str(f)] = content[:6000]
                            new += 1
                except Exception:
                    pass
    log.info(f"[READER] {len(state.files_read)} files, {new} new/changed")


# ====================================================================
# AGENT 2: REASONER - DeepSeek-R1 streaming, reads code, outputs actions
# ====================================================================
def agent_reasoner(state: State):
    with state.lock:
        recent = dict(list(state.file_contents.items())[-5:])
    if not recent:
        return

    ctx = ""
    for p, c in recent.items():
        ctx += f"\n--- {Path(p).name} ---\n{c[:1500]}\n"

    prompt = f"""Analyze these files. Suggest improvements, GPU offloads, or fixes.
Output JSON action blocks:
```json
{{"action":"write_file","filepath":"path","content":"code"}}
```
```json
{{"action":"run_command","command":"cmd","cwd":"path"}}
```
If nothing needed: MAINTAIN

Files:
{ctx}"""

    tokens = []
    def on_tok(t):
        tokens.append(t)
        if len(tokens) % 50 == 0:
            print(".", end="", flush=True)

    log.info("[REASONER] Streaming from DeepSeek-R1...")
    response = lm_stream([
        {"role": "system", "content": "You are an autonomous code agent. Output JSON action blocks to improve the system."},
        {"role": "user", "content": prompt},
    ], on_token=on_tok)

    if response and "MAINTAIN" not in response.upper():
        blocks = re.findall(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
        for b in blocks:
            try:
                cmd = json.loads(b)
                with state.lock:
                    state.pending_actions.append(cmd)
            except json.JSONDecodeError:
                pass
        log.info(f"\n[REASONER] {len(blocks)} actions from {len(tokens)} tokens")
        log_jsonl(LOG_DIR / "thought_stream.jsonl", {
            "ts": datetime.now().isoformat(), "agent": "reasoner",
            "tokens": len(tokens), "actions": len(blocks),
            "response": response[:2000],
        })
    else:
        log.info("[REASONER] MAINTAIN")


# ====================================================================
# AGENT 3: EMBEDDER - Nomic embed live into AMD iGPU VRAM tensor
# ====================================================================
def agent_embedder(state: State):
    with state.lock:
        to_embed = [(p, c[:800]) for p, c in list(state.file_contents.items())[:32]
                     if p not in getattr(state, '_embedded', set())]
    if not to_embed:
        return

    texts = [c for _, c in to_embed]
    paths = [p for p, _ in to_embed]

    try:
        r = requests.post(
            f"{LM_BASE}/embeddings",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=30,
        )
        if r.status_code != 200:
            log.warning(f"[EMBEDDER] HTTP {r.status_code}")
            return
        embeddings = [d["embedding"] for d in r.json()["data"]]
    except Exception as e:
        log.warning(f"[EMBEDDER] {e}")
        return

    # Send into AMD iGPU VRAM as 7D projected tensor
    vram_ok = False
    if HAS_AMD:
        try:
            pipe = AMDiGPUPipeline()
            emb = np.array(embeddings, dtype=np.float32)
            vram_in = emb[:, :7] if emb.shape[1] >= 7 else np.zeros((emb.shape[0], 7), dtype=np.float32)
            if emb.shape[1] < 7:
                vram_in[:, :emb.shape[1]] = emb
            projected = pipe.project_7d(vram_in, curvature=0.618033988749895)
            vram_ok = True
        except Exception as e:
            log.warning(f"[EMBEDDER] AMD iGPU: {e}")

    with state.lock:
        if not hasattr(state, '_embedded'):
            state._embedded = set()
        state._embedded.update(paths)
        state.rewards += len(embeddings) * 0.1

    log.info(f"[EMBEDDER] {len(embeddings)} texts embedded -> AMD iGPU VRAM: {vram_ok}")
    log_jsonl(LOG_DIR / "embeddings.jsonl", {
        "ts": datetime.now().isoformat(), "count": len(embeddings),
        "vram": vram_ok, "dims": len(embeddings[0]) if embeddings else 0,
    })


# ====================================================================
# AGENT 4: EXECUTOR - applies actions, runs commands, no restrictions
# ====================================================================
def agent_executor(state: State):
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
                        state.rewards += 1.0

            elif act == "run_command":
                c = cmd.get("command", "")
                cwd = cmd.get("cwd", str(ROOT))
                r = subprocess.run(c, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120)
                ok = r.returncode == 0
                log.info(f"[EXECUTOR] '{c[:60]}' -> {'OK' if ok else 'FAIL'}")
                if ok:
                    with state.lock:
                        state.rewards += 2.0

        except Exception as e:
            log.warning(f"[EXECUTOR] {e}")

    log_jsonl(LOG_DIR / "metrics.jsonl", {
        "ts": datetime.now().isoformat(), "executed": len(actions), "rewards": state.rewards,
    })


# ====================================================================
# MAIN LOOP - 4 parallel agents, all hitting DeepSeek-R1
# ====================================================================
def run():
    state = State()
    log.info("=" * 60)
    log.info("DeepSeek-R1 4x PARALLEL AGENT - FULL VRAM STREAMING")
    log.info(f"Model: {MODEL}")
    log.info(f"Endpoint: {LM_BASE}")
    log.info(f"Embed: {EMBED_MODEL}")
    log.info(f"AMD iGPU: {'YES' if HAS_AMD else 'NO'}")
    log.info("Agents: READER | REASONER | EMBEDDER | EXECUTOR")
    log.info("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        while True:
            state.cycle += 1
            try:
                # All 4 in parallel
                futures = [
                    ex.submit(agent_reader, state),
                    ex.submit(agent_reasoner, state),
                    ex.submit(agent_embedder, state),
                    ex.submit(agent_executor, state),
                ]
                concurrent.futures.wait(futures, timeout=120)

                if state.cycle % 5 == 0:
                    log.info(f"[LOOP] Cycle {state.cycle} | Files: {len(state.files_read)} | "
                             f"Rewards: {state.rewards:.1f} | Pending: {len(state.pending_actions)}")

            except KeyboardInterrupt:
                log.info("[STOP] Agent terminated.")
                break
            except Exception as e:
                log.warning(f"[LOOP] {e}")


if __name__ == "__main__":
    run()
