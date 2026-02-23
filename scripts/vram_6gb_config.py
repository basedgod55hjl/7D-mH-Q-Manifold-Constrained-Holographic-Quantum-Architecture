#!/usr/bin/env python3
# File: scripts/vram_6gb_config.py
# 6GB VRAM optimization constants - ImGui, OpenCL, time kernels, higher math in VRAM

"""
6GB VRAM Budget (6144 MB)
- ImGui textures: max 512MB
- Compute buffers: max 4GB working set
- Time kernels / 7D manifold: batch 128, workgroup 128
- Headroom: 1GB for driver/overhead
"""

# Limits for 6GB cards (GTX 1660 Ti, RTX 3060 6GB, etc.)
MAX_VRAM_MB = 6144
SAFE_VRAM_MB = 5120  # Leave headroom
BATCH_SIZE_7D = 128
WORKGROUP_SIZE = 128
MAX_TEXTURE_DIM = 2048  # ImGui/overlay
MAX_CONCURRENT_KERNELS = 4

# WGPU limits when creating device (pass to Limits)
WGPU_LIMITS_6GB = {
    "max_storage_buffer_size": 256 * 1024 * 1024,  # 256MB per buffer
    "max_buffer_size": 512 * 1024 * 1024,
    "max_storage_bindings": 8,
    "max_compute_workgroups_per_dimension": 65535,
    "max_compute_workgroup_size_x": 256,
    "max_compute_workgroup_size_y": 256,
    "max_compute_workgroup_size_z": 64,
}

# Batch thresholds for ComputeDispatcher (Rust)
BATCH_THRESHOLD_GPU = 64  # Use GPU only for batches >= 64 on 6GB
