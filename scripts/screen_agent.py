#!/usr/bin/env python3
"""
Screen Agent - LLM-driven mouse/click/type/screen control
Captures screen -> base64 -> sends to vision LLM -> parses actions -> executes

Actions the LLM can output:
  {"action":"click","x":500,"y":300}
  {"action":"doubleclick","x":500,"y":300}
  {"action":"rightclick","x":500,"y":300}
  {"action":"type","text":"hello world"}
  {"action":"hotkey","keys":["ctrl","s"]}
  {"action":"scroll","amount":-3}
  {"action":"move","x":500,"y":300}
  {"action":"screenshot"}
  {"action":"wait","seconds":1}
  {"action":"noop"}
"""

import os
import sys
import json
import re
import base64
import io
import time
import threading
import traceback
from pathlib import Path

import pyautogui
import mss
import mss.tools
import requests
from PIL import Image

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

# LLM endpoints (try local first, then remote)
LLM_ENDPOINTS = [
    os.environ.get("OMNI_API", "http://127.0.0.1:1234/v1"),
    "http://10.5.0.2:55555/v1",
]
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen/qwen3-vl-4b")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "gpt-oss-6.0b-specialized-all-pruned-moe-only-7-experts-i1")

ROOT = Path(__file__).resolve().parent.parent


def capture_screen_b64(scale: float = 0.5) -> str:
    """Capture screen, resize for VRAM efficiency, return base64 PNG."""
    with mss.mss() as sct:
        mon = sct.monitors[0]
        img = sct.grab(mon)
        pil = Image.frombytes("RGB", (img.width, img.height), img.rgb)

    w, h = int(pil.width * scale), int(pil.height * scale)
    pil = pil.resize((w, h), Image.LANCZOS)

    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def capture_screen_pil(scale: float = 0.5) -> Image.Image:
    """Capture screen as PIL Image."""
    with mss.mss() as sct:
        mon = sct.monitors[0]
        img = sct.grab(mon)
        pil = Image.frombytes("RGB", (img.width, img.height), img.rgb)
    w, h = int(pil.width * scale), int(pil.height * scale)
    return pil.resize((w, h), Image.LANCZOS)


LM_API_TOKEN = os.environ.get("LM_API_TOKEN", os.environ.get("LM_STUDIO_API_TOKEN", ""))


def call_llm(messages, model=None, endpoint=None, timeout=30):
    """Call LLM with fallback across endpoints. Handles auth."""
    model = model or TEXT_MODEL
    endpoints = [endpoint] if endpoint else LLM_ENDPOINTS

    headers = {"Content-Type": "application/json"}
    if LM_API_TOKEN:
        headers["Authorization"] = f"Bearer {LM_API_TOKEN}"

    for ep in endpoints:
        try:
            url = f"{ep}/chat/completions"
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1024,
            }
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            continue
    return None


def call_vision_llm(prompt: str, image_b64: str, model=None):
    """Call vision LLM with image."""
    model = model or VISION_MODEL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]}
    ]
    return call_llm(messages, model=model, timeout=60)


def call_text_llm(prompt: str, context: str = ""):
    """Call text LLM for action planning."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\n{prompt}" if context else prompt}
    ]
    return call_llm(messages)


SYSTEM_PROMPT = """You are the ABRASAX Screen Agent. You see the user's screen and control mouse/keyboard.
You output JSON action blocks to interact with the PC. Output one or more actions per turn.

Available actions (output as JSON array):
```json
[
  {"action":"click","x":500,"y":300},
  {"action":"doubleclick","x":500,"y":300},
  {"action":"rightclick","x":500,"y":300},
  {"action":"type","text":"hello"},
  {"action":"hotkey","keys":["ctrl","s"]},
  {"action":"scroll","amount":-3},
  {"action":"move","x":500,"y":300},
  {"action":"wait","seconds":1},
  {"action":"noop"}
]
```

Screen coordinates are at FULL resolution (1920x1080). The screenshot you see may be scaled down.
When you see the screenshot, multiply coordinates by 2 if the image appears half-size.

If you don't need to do anything, output: [{"action":"noop"}]
Always wrap actions in a JSON array. Be precise with coordinates."""


def parse_actions(response: str) -> list:
    """Parse LLM response for JSON action blocks."""
    if not response:
        return []

    # Try to find JSON array
    arrays = re.findall(r'\[[\s\S]*?\]', response)
    for arr_str in arrays:
        try:
            actions = json.loads(arr_str)
            if isinstance(actions, list) and len(actions) > 0:
                if any(isinstance(a, dict) and "action" in a for a in actions):
                    return actions
        except json.JSONDecodeError:
            continue

    # Try individual JSON objects
    objects = re.findall(r'\{[^{}]+\}', response)
    actions = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
            if isinstance(obj, dict) and "action" in obj:
                actions.append(obj)
        except json.JSONDecodeError:
            continue
    return actions


def execute_action(action: dict) -> str:
    """Execute a single mouse/keyboard action."""
    act = action.get("action", "noop")

    if act == "click":
        x, y = int(action["x"]), int(action["y"])
        pyautogui.click(x, y)
        return f"clicked ({x},{y})"

    elif act == "doubleclick":
        x, y = int(action["x"]), int(action["y"])
        pyautogui.doubleClick(x, y)
        return f"doubleclicked ({x},{y})"

    elif act == "rightclick":
        x, y = int(action["x"]), int(action["y"])
        pyautogui.rightClick(x, y)
        return f"rightclicked ({x},{y})"

    elif act == "type":
        text = action.get("text", "")
        pyautogui.typewrite(text, interval=0.02) if text.isascii() else pyautogui.write(text)
        return f"typed '{text[:30]}'"

    elif act == "hotkey":
        keys = action.get("keys", [])
        if keys:
            pyautogui.hotkey(*keys)
        return f"hotkey {'+'.join(keys)}"

    elif act == "scroll":
        amount = int(action.get("amount", -3))
        pyautogui.scroll(amount)
        return f"scrolled {amount}"

    elif act == "move":
        x, y = int(action["x"]), int(action["y"])
        pyautogui.moveTo(x, y)
        return f"moved to ({x},{y})"

    elif act == "wait":
        secs = float(action.get("seconds", 1))
        time.sleep(secs)
        return f"waited {secs}s"

    elif act == "noop":
        return "noop"

    return f"unknown action: {act}"


def run_screen_agent(task: str = None, max_cycles: int = 0):
    """
    Main loop: capture screen -> LLM decides actions -> execute -> repeat.
    max_cycles=0 means run forever.
    """
    print("=" * 60)
    print("SCREEN AGENT - LLM Mouse/Click/Type Control")
    print(f"Screen: {pyautogui.size().width}x{pyautogui.size().height}")
    print(f"Vision Model: {VISION_MODEL}")
    print(f"Text Model: {TEXT_MODEL}")
    print("=" * 60)

    cycle = 0
    history = []

    while max_cycles == 0 or cycle < max_cycles:
        cycle += 1
        try:
            # 1. Capture screen
            screen_b64 = capture_screen_b64(scale=0.5)
            print(f"\n[Cycle {cycle}] Screen captured ({len(screen_b64)//1024} KB b64)")

            # 2. Build prompt
            if task:
                prompt = f"TASK: {task}\n\nLook at the screenshot. What actions should I take next? Output JSON actions."
            else:
                prompt = "Look at the screenshot. Describe what you see and suggest any useful actions. Output JSON actions or noop."

            if history:
                last_actions = "; ".join(history[-5:])
                prompt += f"\n\nPrevious actions: {last_actions}"

            # 3. Call vision LLM
            print(f"[Cycle {cycle}] Sending to LLM...")
            response = call_vision_llm(prompt, screen_b64)

            if not response:
                # Fallback to text-only LLM
                response = call_text_llm(
                    prompt,
                    context=f"Screen is 1920x1080. Mouse at {pyautogui.position()}."
                )

            if response:
                print(f"[Cycle {cycle}] LLM response: {response[:200]}...")

                # 4. Parse and execute actions
                actions = parse_actions(response)
                if not actions:
                    actions = [{"action": "noop"}]

                for act in actions:
                    result = execute_action(act)
                    print(f"  -> {result}")
                    history.append(result)
            else:
                print(f"[Cycle {cycle}] No LLM response, waiting...")
                time.sleep(2)

        except KeyboardInterrupt:
            print("\n[STOP] Screen agent terminated.")
            break
        except Exception as e:
            print(f"[ERROR] Cycle {cycle}: {e}")
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Screen Agent - LLM mouse/click control")
    parser.add_argument("--task", type=str, default=None, help="Task for the agent to accomplish")
    parser.add_argument("--cycles", type=int, default=0, help="Max cycles (0=forever)")
    args = parser.parse_args()

    run_screen_agent(task=args.task, max_cycles=args.cycles)
