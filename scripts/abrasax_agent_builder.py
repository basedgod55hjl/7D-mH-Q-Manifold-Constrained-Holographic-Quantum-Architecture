#!/usr/bin/env python3
"""
ABRASAX GOD OS - Auto Agent Builder
Loads LM Studio model and builds agents via /v1/responses API.
Can inject into PC live, build UI, memory, etc.

Sir Charles Spikes | Cincinnati, Ohio, USA
"""

import json
import os
import urllib.request
import urllib.error

LM_PORT = int(os.environ.get("LM_STUDIO_PORT", "1234"))
BRIDGE_PORT = 17777
USE_BRIDGE = os.environ.get("ABRASAX_USE_BRIDGE", "").lower() in ("1", "true", "yes")


def call_lm_responses(
    model: str = "qwen/qwen3-vl-4b",
    input_text: str = "What is the weather like in Boston today?",
    tools: list | None = None,
    tool_choice: str = "auto",
) -> dict:
    url = f"http://localhost:{BRIDGE_PORT}/v1/responses" if USE_BRIDGE else f"http://localhost:{LM_PORT}/v1/responses"
    token = os.environ.get("LM_STUDIO_API_TOKEN") or os.environ.get("LM_API_TOKEN")

    tools = tools or [
        {
            "type": "function",
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        }
    ]

    data = {
        "model": model,
        "input": input_text,
        "tools": tools,
        "tool_choice": tool_choice,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"status": "error", "code": e.code, "body": e.read().decode()}
    except urllib.error.URLError as e:
        return {"status": "error", "message": str(e), "hint": "Ensure LM Studio is running"}


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "What is the weather like in Boston today?"
    out = call_lm_responses(input_text=inp)
    print(json.dumps(out, indent=2))
