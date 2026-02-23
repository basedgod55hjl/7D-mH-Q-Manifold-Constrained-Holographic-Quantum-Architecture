#!/usr/bin/env python3
"""
AMD iGPU VRAM Input Pipeline - 7D Crystal Manifold
Sends input data into AMD Radeon iGPU VRAM via OpenCL, runs 7D projection kernel,
then optionally transfers results to NVIDIA CUDA for TensorRT inference.

Hardware: AMD Radeon gfx90c (512MB dedicated / 6GB shared), 7 CUs
"""

import numpy as np
import pyopencl as cl
import sys
import os
from pathlib import Path

# 7D Constants
PHI = 1.618033988749895
PHI_INV = 0.618033988749895
S2_STABILITY = 0.01
DIMS = 7

# AMD iGPU VRAM budget (shared memory - be conservative)
AMD_MAX_ALLOC_MB = 512
AMD_BATCH_LIMIT = 16384  # max vectors per batch for iGPU

# OpenCL kernel: 7D Poincare projection on AMD iGPU
CRYSTAL_7D_CL_KERNEL = r"""
#define PHI 1.618033988749895f
#define PHI_INV 0.618033988749895f
#define S2_STABILITY 0.01f
#define DIMS 7

__kernel void project_to_poincare_7d(
    __global const float* input,
    __global float* output,
    const int n,
    const float curvature)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    float in_vec[7];
    float norm_sq = 0.0f;

    for (int i = 0; i < DIMS; i++) {
        float val = input[idx * DIMS + i];
        in_vec[i] = val;
        norm_sq += val * val;
    }

    float max_norm = 1.0f - S2_STABILITY;
    float scale = 1.0f;
    if (norm_sq >= max_norm * max_norm) {
        scale = max_norm / sqrt(norm_sq);
    }

    float phi_powers[7] = {1.0f, PHI, PHI*PHI, PHI*PHI*PHI,
                           PHI*PHI*PHI*PHI, PHI*PHI*PHI*PHI*PHI,
                           PHI*PHI*PHI*PHI*PHI*PHI};

    float denom = 1.0f + norm_sq * fabs(curvature);
    float final_norm_sq = 0.0f;

    for (int i = 0; i < DIMS; i++) {
        float val = in_vec[i] * scale * phi_powers[i] / denom;
        output[idx * DIMS + i] = val;
        final_norm_sq += val * val;
    }

    if (final_norm_sq > max_norm * max_norm) {
        float final_scale = max_norm / sqrt(final_norm_sq);
        for (int i = 0; i < DIMS; i++) {
            output[idx * DIMS + i] *= final_scale;
        }
    }
}

__kernel void mobius_add_7d(
    __global const float* u,
    __global const float* v,
    __global float* result,
    const int n,
    const float curvature)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    float c = fabs(curvature);
    float u_sq = 0.0f, v_sq = 0.0f, uv = 0.0f;

    for (int i = 0; i < DIMS; i++) {
        float ui = u[idx * DIMS + i];
        float vi = v[idx * DIMS + i];
        u_sq += ui * ui;
        v_sq += vi * vi;
        uv += ui * vi;
    }

    float den = 1.0f + 2.0f * c * uv + c * c * u_sq * v_sq;
    if (fabs(den) < 1e-10f) {
        for (int i = 0; i < DIMS; i++) result[idx * DIMS + i] = 0.0f;
        return;
    }

    float cu = 1.0f + 2.0f * c * uv + c * v_sq;
    float cv = 1.0f - c * u_sq;

    float max_norm = 1.0f - S2_STABILITY;
    float r_sq = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        float val = (cu * u[idx * DIMS + i] + cv * v[idx * DIMS + i]) / den;
        result[idx * DIMS + i] = val;
        r_sq += val * val;
    }

    if (r_sq > max_norm * max_norm) {
        float s = max_norm / sqrt(r_sq);
        for (int i = 0; i < DIMS; i++) result[idx * DIMS + i] *= s;
    }
}

__kernel void hyperbolic_distance_7d(
    __global const float* u,
    __global const float* v,
    __global float* distances,
    const int n,
    const float curvature)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    float c = fabs(curvature);
    float u_sq = 0.0f, v_sq = 0.0f, diff_sq = 0.0f;

    for (int i = 0; i < DIMS; i++) {
        float d = u[idx * DIMS + i] - v[idx * DIMS + i];
        diff_sq += d * d;
        u_sq += u[idx * DIMS + i] * u[idx * DIMS + i];
        v_sq += v[idx * DIMS + i] * v[idx * DIMS + i];
    }

    float den = (1.0f - c * u_sq) * (1.0f - c * v_sq);
    if (den <= 0.0f) { distances[idx] = 1e10f; return; }

    float arg = 1.0f + 2.0f * c * diff_sq / den;
    distances[idx] = (1.0f / sqrt(c)) * log(arg + sqrt(arg * arg - 1.0f));
}

__kernel void holographic_fold_7d(
    __global const float* patterns,
    __global const float* phases,
    __global float* output,
    const int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;

    float phase = phases[idx];
    float cos_p = cos(phase * PHI);
    float sin_p = sin(phase * PHI);

    float energy = 0.0f;
    for (int i = 0; i < DIMS; i++) {
        float base = patterns[idx * DIMS + i];
        float rotated = base * cos_p;
        if (i > 0) rotated += patterns[idx * DIMS + i - 1] * sin_p * 0.5f;
        if (i < DIMS - 1) rotated += patterns[idx * DIMS + i + 1] * sin_p * 0.5f;
        output[idx * DIMS + i] = rotated;
        energy += rotated * rotated;
    }

    if (energy > 1e-10f) {
        float s = 1.0f / sqrt(energy);
        for (int i = 0; i < DIMS; i++) output[idx * DIMS + i] *= s;
    }
}
"""


class AMDiGPUPipeline:
    """AMD iGPU OpenCL pipeline for 7D Crystal manifold operations."""

    def __init__(self, platform_index=None):
        platforms = cl.get_platforms()
        amd_idx = platform_index
        if amd_idx is None:
            for i, p in enumerate(platforms):
                if "AMD" in p.name.upper():
                    amd_idx = i
                    break
        if amd_idx is None:
            raise RuntimeError("No AMD OpenCL platform found. Available: " +
                               ", ".join(p.name for p in platforms))

        self.platform = platforms[amd_idx]
        devices = self.platform.get_devices(cl.device_type.GPU)
        if not devices:
            devices = self.platform.get_devices(cl.device_type.ALL)
        if not devices:
            raise RuntimeError(f"No GPU devices on AMD platform: {self.platform.name}")

        self.device = devices[0]
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        self.program = cl.Program(self.ctx, CRYSTAL_7D_CL_KERNEL).build()

        mem_mb = self.device.global_mem_size // (1024 * 1024)
        print(f"[AMD iGPU] Device: {self.device.name}")
        print(f"[AMD iGPU] Global Memory: {mem_mb} MB, Compute Units: {self.device.max_compute_units}")
        print(f"[AMD iGPU] Max Alloc: {self.device.max_mem_alloc_size // (1024*1024)} MB")
        print(f"[AMD iGPU] 7D Crystal kernels compiled OK")

    def send_to_vram(self, data: np.ndarray) -> cl.Buffer:
        """Send numpy array into AMD iGPU VRAM as OpenCL buffer."""
        buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
        return buf

    def read_from_vram(self, buf: cl.Buffer, shape, dtype=np.float32) -> np.ndarray:
        """Read OpenCL buffer back to host numpy array."""
        out = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(self.queue, out, buf).wait()
        return out

    def project_7d(self, vectors: np.ndarray, curvature: float = PHI_INV) -> np.ndarray:
        """Project vectors to Poincare ball on AMD iGPU VRAM."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = vectors.shape[0]
        assert vectors.shape[1] == DIMS

        d_in = self.send_to_vram(vectors)
        d_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, vectors.nbytes)

        evt = self.program.project_to_poincare_7d(
            self.queue, (n,), None, d_in, d_out,
            np.int32(n), np.float32(curvature))
        evt.wait()

        result = self.read_from_vram(d_out, vectors.shape)
        t_ns = evt.profile.end - evt.profile.start
        print(f"[AMD iGPU] project_7d: {n} vectors in {t_ns/1e6:.2f} ms")
        return result

    def mobius_add(self, u: np.ndarray, v: np.ndarray, curvature: float = PHI_INV) -> np.ndarray:
        """Mobius addition on AMD iGPU."""
        u = np.ascontiguousarray(u, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        n = min(u.shape[0], v.shape[0])

        d_u = self.send_to_vram(u[:n])
        d_v = self.send_to_vram(v[:n])
        d_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, n * DIMS * 4)

        evt = self.program.mobius_add_7d(
            self.queue, (n,), None, d_u, d_v, d_out,
            np.int32(n), np.float32(curvature))
        evt.wait()

        result = self.read_from_vram(d_out, (n, DIMS))
        t_ns = evt.profile.end - evt.profile.start
        print(f"[AMD iGPU] mobius_add: {n} pairs in {t_ns/1e6:.2f} ms")
        return result

    def hyperbolic_distance(self, u: np.ndarray, v: np.ndarray, curvature: float = PHI_INV) -> np.ndarray:
        """Hyperbolic distance on AMD iGPU."""
        u = np.ascontiguousarray(u, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        n = min(u.shape[0], v.shape[0])

        d_u = self.send_to_vram(u[:n])
        d_v = self.send_to_vram(v[:n])
        d_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, n * 4)

        evt = self.program.hyperbolic_distance_7d(
            self.queue, (n,), None, d_u, d_v, d_out,
            np.int32(n), np.float32(curvature))
        evt.wait()

        result = self.read_from_vram(d_out, (n,))
        t_ns = evt.profile.end - evt.profile.start
        print(f"[AMD iGPU] hyperbolic_distance: {n} pairs in {t_ns/1e6:.2f} ms")
        return result

    def holographic_fold(self, patterns: np.ndarray, phases: np.ndarray) -> np.ndarray:
        """Holographic fold on AMD iGPU."""
        patterns = np.ascontiguousarray(patterns, dtype=np.float32)
        phases = np.ascontiguousarray(phases, dtype=np.float32)
        n = min(patterns.shape[0], phases.shape[0])

        d_pat = self.send_to_vram(patterns[:n])
        d_ph = self.send_to_vram(phases[:n])
        d_out = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, n * DIMS * 4)

        evt = self.program.holographic_fold_7d(
            self.queue, (n,), None, d_pat, d_ph, d_out, np.int32(n))
        evt.wait()

        result = self.read_from_vram(d_out, (n, DIMS))
        t_ns = evt.profile.end - evt.profile.start
        print(f"[AMD iGPU] holographic_fold: {n} patterns in {t_ns/1e6:.2f} ms")
        return result

    def prepare_for_tensorrt(self, projected: np.ndarray) -> np.ndarray:
        """
        Format AMD iGPU output for NVIDIA TensorRT input.
        Returns contiguous float32 ready for cudaMemcpy.
        """
        return np.ascontiguousarray(projected, dtype=np.float32)


def run_pipeline():
    """Full pipeline: AMD iGPU VRAM input -> 7D projection -> ready for TensorRT."""
    print("=" * 60)
    print("AMD iGPU -> VRAM Input Pipeline -> 7D Crystal Manifold")
    print("=" * 60)

    pipe = AMDiGPUPipeline()

    # Generate test vectors
    n = 4096
    vectors = np.random.randn(n, DIMS).astype(np.float32) * 0.5
    print(f"\n[INPUT] {n} vectors x {DIMS}D ({vectors.nbytes / 1024:.1f} KB)")

    # Project on AMD iGPU
    projected = pipe.project_7d(vectors)

    # Verify all inside unit ball
    norms = np.linalg.norm(projected, axis=1)
    print(f"[VERIFY] max norm: {norms.max():.6f} (must be < 1.0)")
    assert norms.max() < 1.0, "Projection failed: vectors outside unit ball"

    # Mobius addition
    u = projected[:n//2]
    v = projected[n//2:]
    added = pipe.mobius_add(u, v)
    add_norms = np.linalg.norm(added, axis=1)
    print(f"[VERIFY] mobius max norm: {add_norms.max():.6f}")

    # Hyperbolic distance
    dists = pipe.hyperbolic_distance(u, v)
    print(f"[VERIFY] distances: min={dists.min():.4f}, max={dists.max():.4f}, mean={dists.mean():.4f}")

    # Holographic fold
    phases = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
    folded = pipe.holographic_fold(projected, phases)
    fold_norms = np.linalg.norm(folded, axis=1)
    print(f"[VERIFY] fold norms: min={fold_norms.min():.4f}, max={fold_norms.max():.4f}")

    # Prepare for TensorRT transfer
    trt_input = pipe.prepare_for_tensorrt(projected)
    print(f"\n[TRT READY] {trt_input.shape} float32, {trt_input.nbytes / 1024:.1f} KB -> cudaMemcpy to NVIDIA")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE - AMD iGPU VRAM -> 7D -> TensorRT ready")
    print("=" * 60)
    return trt_input


if __name__ == "__main__":
    run_pipeline()
