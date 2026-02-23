// File: runtime/src/wgpu_gpu.rs
// Minimal WGPU backend shim for ABRASAX build continuity.

use crate::Vector7D;

#[derive(Debug, Default, Clone, Copy)]
pub struct WgpuStats {
    pub ops: u64,
}

pub struct WgpuExecutor {
    pub stats: WgpuStats,
}

impl WgpuExecutor {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            stats: WgpuStats::default(),
        })
    }

    pub fn device_info(&self) -> &'static str {
        "WGPU backend (shim)"
    }

    pub fn project_batch(
        &mut self,
        vectors: &mut [Vector7D],
        curvature: f64,
    ) -> anyhow::Result<()> {
        for v in vectors.iter_mut() {
            *v = v.project(curvature);
            self.stats.ops += 1;
        }
        Ok(())
    }
}
