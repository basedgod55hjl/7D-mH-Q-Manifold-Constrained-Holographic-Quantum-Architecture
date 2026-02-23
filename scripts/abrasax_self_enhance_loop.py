#!/usr/bin/env python3
"""
ABRASAX GOD OS - Screen-aware self-enhance loop for local LM Studio.

Flow:
1) Capture desktop screenshot
2) Ask local LLM for build/test enhancement commands
3) Execute allowlisted commands (only when --apply is set)

This is intentionally constrained for safety.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'requests'. Install with: pip install requests pillow"
    ) from exc

try:
    from PIL import ImageGrab
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'Pillow'. Install with: pip install pillow"
    ) from exc


ALLOWED_PREFIXES = (
    "cargo build",
    "cargo check",
    "cargo test",
    "python ",
    "py ",
    "pip install ",
    "cmake ",
    "ctest",
    "ruff ",
    "maturin ",
)

BLOCKED_FRAGMENTS = (
    "rm -rf",
    "del /f",
    "format ",
    "shutdown",
    "reboot",
    "reg delete",
    "takeown",
    "icacls",
    "diskpart",
    "bcdedit",
)


def capture_screen_b64(max_side: int, jpeg_quality: int) -> tuple[str, str]:
    image = ImageGrab.grab(all_screens=True).convert("RGB")
    width, height = image.size
    max_dim = max(width, height)
    if max_dim > max_side:
        scale = max_side / float(max_dim)
        image = image.resize((int(width * scale), int(height * scale)))

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return "image/jpeg", base64.b64encode(buffer.getvalue()).decode("ascii")


def build_prompt(repo_root: Path) -> str:
    return (
        "You are ABRASAX self-enhance build planner. Analyze the screenshot and repository context. "
        f"Repository root: {repo_root}. "
        "Return STRICT JSON only with keys: summary (string), commands (array of shell commands), "
        "reasoning (string), risk (low|medium|high). "
        "Focus only on build, test, lint, and local automation improvements."
    )


def call_lm_studio(
    *,
    base_url: str,
    token: str | None,
    model: str,
    prompt: str,
    screenshot_mime: str,
    screenshot_b64: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:{screenshot_mime};base64,{screenshot_b64}",
                    },
                ],
            }
        ],
        "temperature": 0.2,
    }

    url = f"{base_url.rstrip('/')}/v1/responses"
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    response.raise_for_status()
    return response.json()


def extract_text(response_json: Dict[str, Any]) -> str:
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]

    chunks = response_json.get("output", [])
    texts: List[str] = []
    if isinstance(chunks, list):
        for chunk in chunks:
            content = chunk.get("content", [])
            if isinstance(content, list):
                for item in content:
                    txt = item.get("text")
                    if isinstance(txt, str):
                        texts.append(txt)
    if texts:
        return "\n".join(texts)

    # Fallback for chat-like shape
    choices = response_json.get("choices", [])
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content
    return ""


def parse_plan(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {"summary": "No model text returned.", "commands": [], "reasoning": "", "risk": "high"}

    # Best effort extraction of first JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                parsed.setdefault("commands", [])
                return parsed
        except json.JSONDecodeError:
            pass

    return {
        "summary": "Could not parse strict JSON plan from model response.",
        "commands": [],
        "reasoning": text[:2000],
        "risk": "high",
    }


def is_command_allowed(cmd: str) -> bool:
    normalized = cmd.strip().lower()
    if not normalized:
        return False
    if any(fragment in normalized for fragment in BLOCKED_FRAGMENTS):
        return False
    return any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def run_command(cmd: str, cwd: Path, timeout_sec: int) -> Dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return {
        "command": cmd,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "elapsed_sec": round(time.time() - started, 2),
    }


def ensure_log_dir(repo_root: Path) -> Path:
    log_dir = repo_root / "scripts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="ABRASAX screen-aware self-enhance loop")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--base-url", default=os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234"))
    parser.add_argument("--model", default=os.environ.get("LM_STUDIO_MODEL", "qwen/qwen3-vl-4b"))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--command-timeout-sec", type=int, default=900)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=55)
    parser.add_argument("--apply", action="store_true", help="Execute allowlisted commands")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    token = os.environ.get("LM_STUDIO_API_TOKEN") or os.environ.get("LM_API_TOKEN")

    log_dir = ensure_log_dir(repo_root)
    log_path = log_dir / f"self_enhance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

    print(f"[ABRASAX] Repo root: {repo_root}")
    print(f"[ABRASAX] Model: {args.model}")
    print(f"[ABRASAX] Endpoint: {args.base_url.rstrip('/')}/v1/responses")
    print(f"[ABRASAX] Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"[ABRASAX] Log: {log_path}")

    for i in range(args.cycles):
        cycle_no = i + 1
        timestamp = datetime.now().isoformat()
        print(f"\n[ABRASAX] Cycle {cycle_no}/{args.cycles} @ {timestamp}")

        try:
            screenshot_mime, screenshot_b64 = capture_screen_b64(
                max_side=args.max_side,
                jpeg_quality=args.jpeg_quality,
            )
            prompt = build_prompt(repo_root)
            response_json = call_lm_studio(
                base_url=args.base_url,
                token=token,
                model=args.model,
                prompt=prompt,
                screenshot_mime=screenshot_mime,
                screenshot_b64=screenshot_b64,
                timeout_sec=args.timeout_sec,
            )
            model_text = extract_text(response_json)
            plan = parse_plan(model_text)
        except Exception as exc:  # pylint: disable=broad-except
            error_row = {
                "ts": timestamp,
                "cycle": cycle_no,
                "error": str(exc),
            }
            append_jsonl(log_path, error_row)
            print(f"[ABRASAX] LM call failed: {exc}")
            if cycle_no < args.cycles:
                time.sleep(args.interval_sec)
            continue

        summary = str(plan.get("summary", ""))
        commands = plan.get("commands", [])
        if not isinstance(commands, list):
            commands = []

        print(f"[ABRASAX] Summary: {summary}")
        print(f"[ABRASAX] Proposed commands: {len(commands)}")

        results: List[Dict[str, Any]] = []
        for raw_cmd in commands:
            cmd = str(raw_cmd).strip()
            allowed = is_command_allowed(cmd)
            if not allowed:
                results.append(
                    {
                        "command": cmd,
                        "executed": False,
                        "reason": "Blocked by allowlist",
                    }
                )
                print(f"  - BLOCKED: {cmd}")
                continue

            if not args.apply:
                results.append(
                    {
                        "command": cmd,
                        "executed": False,
                        "reason": "Dry-run mode",
                    }
                )
                print(f"  - DRY-RUN: {cmd}")
                continue

            print(f"  - RUN: {cmd}")
            result = run_command(cmd, cwd=repo_root, timeout_sec=args.command_timeout_sec)
            result["executed"] = True
            results.append(result)

        row = {
            "ts": timestamp,
            "cycle": cycle_no,
            "summary": summary,
            "risk": plan.get("risk", ""),
            "reasoning": plan.get("reasoning", ""),
            "commands": commands,
            "results": results,
        }
        append_jsonl(log_path, row)

        if cycle_no < args.cycles:
            time.sleep(args.interval_sec)

    print("\n[ABRASAX] Self-enhance loop complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
