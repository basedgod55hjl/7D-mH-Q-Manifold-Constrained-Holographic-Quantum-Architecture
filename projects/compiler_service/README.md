# Sovereign Compiler as a Service

## Overview

REST API for compiling 7D-MHQL programs to native binaries.

## Endpoints

- `POST /compile` - Compile .7d source to binary
- `POST /execute` - Compile and execute in sandbox
- `GET /version` - Compiler version info

## Security

- Sandboxed execution environment
- Resource limits (CPU, memory, time)
- No filesystem access from executed code
