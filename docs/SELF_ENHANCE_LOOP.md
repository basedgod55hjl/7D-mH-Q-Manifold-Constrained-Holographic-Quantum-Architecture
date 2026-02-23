# ABRASAX Screen-Aware Self-Enhance Loop

This loop lets your local LM Studio model:

1. view current screen state (desktop screenshot)
2. propose build/test/lint improvements
3. execute only allowlisted commands (when enabled)

## Requirements

```powershell
pip install requests pillow
```

## Run (Dry-Run First)

```powershell
.\scripts\start_self_enhance_loop.ps1 -Cycles 3 -IntervalSec 60
```

Dry-run mode logs what the model wants to run but does not execute commands.

## Run (Apply Mode)

```powershell
.\scripts\start_self_enhance_loop.ps1 -Cycles 5 -IntervalSec 45 -Apply
```

## LM Studio Auth

If your LM Studio local server requires a token:

```powershell
$env:LM_STUDIO_API_TOKEN = "your_token_here"
```

## Logs

Logs are written to:

`scripts/logs/self_enhance_*.jsonl`

## Safety Model

- Command allowlist only (`cargo build/check/test`, `python`, `cmake`, etc.)
- Explicit blocklist for destructive operations
- `--apply` required for command execution
- Default is dry-run

