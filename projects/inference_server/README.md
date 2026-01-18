# Multi-GPU Inference Server

## Overview

High-performance REST/gRPC inference server with CUDA kernel integration.

## Features

- Batched inference with dynamic batching
- Multi-GPU load balancing
- KV cache management
- Streaming responses

## Endpoints

- `POST /v1/completions` - Text completion
- `POST /v1/chat` - Chat completion
- `GET /v1/models` - List loaded models
- `GET /health` - Health check
