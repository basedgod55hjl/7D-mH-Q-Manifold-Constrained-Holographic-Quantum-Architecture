// File: runtime/src/compute.rs
// Unified Compute Dispatcher for 7D Crystal System
// Routes operations to GPU (CUDA) or CPU automatically

use crate::gpu::{
    self, GpuError, GpuExecutor, GpuStats, Vector7D,
    project_to_poincare_cpu, mobius_add_cpu, hyperbolic_distance_cpu,
    PHI, PHI_INV, S2_STABILITY,
};
use crate::kernels;

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeBackend {
    Auto,
    Gpu,
    Cpu,
}

/// Unified compute statistics
#[derive(Debug, Default, Clone)]
pub struct ComputeStats {
    pub gpu_ops: u64,
    pub cpu_ops: u64,
    pub gpu_fallbacks: u64,
    pub total_projections: u64,
    pub total_mobius_ops: u64,
    pub total_distance_ops: u64,
    pub total_fold_ops: u64,
}

/// Unified compute dispatcher
pub struct ComputeDispatcher {
    #[cfg(feature = "cuda")]
    gpu: Option<GpuExecutor>,
    backend: ComputeBackend,
    pub stats: ComputeStats,
    batch_threshold: usize,  // Minimum batch size to use GPU
}

impl ComputeDispatcher {
    /// Create new dispatcher with automatic backend selection
    pub fn new() -> Self {
        Self::with_backend(ComputeBackend::Auto)
    }
    
    /// Create dispatcher with specific backend
    pub fn with_backend(backend: ComputeBackend) -> Self {
        #[cfg(feature = "cuda")]
        let gpu = if backend != ComputeBackend::Cpu {
            match GpuExecutor::new() {
                Ok(mut exec) => {
                    // Load kernels
                    if let Err(e) = exec.load_kernels(kernels::get_kernel_source()) {
                        eprintln!("[7D Crystal] Failed to load GPU kernels: {}", e);
                        None
                    } else {
                        println!("[7D Crystal] GPU initialized: {}", exec.device_info());
                        Some(exec)
                    }
                }
                Err(e) => {
                    if backend == ComputeBackend::Gpu {
                        eprintln!("[7D Crystal] GPU requested but not available: {}", e);
                    }
                    None
                }
            }
        } else {
            None
        };
        
        #[cfg(not(feature = "cuda"))]
        let gpu: Option<()> = None;
        
        Self {
            #[cfg(feature = "cuda")]
            gpu,
            backend,
            stats: ComputeStats::default(),
            batch_threshold: 64,  // Use GPU for batches >= 64
        }
    }
    
    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
    
    /// Get current backend info
    pub fn backend_info(&self) -> String {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref gpu) = self.gpu {
                format!("GPU (CUDA) - {}", gpu.device_info())
            } else {
                "CPU (SIMD when available)".to_string()
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            "CPU only (CUDA feature not enabled)".to_string()
        }
    }
    
    /// Set minimum batch size for GPU usage
    pub fn set_batch_threshold(&mut self, threshold: usize) {
        self.batch_threshold = threshold;
    }
    
    /// Project vectors to Poincaré ball
    pub fn project_to_poincare(
        &mut self,
        vectors: &mut [Vector7D],
        curvature: f32,
    ) {
        let n = vectors.len();
        self.stats.total_projections += n as u64;
        
        #[cfg(feature = "cuda")]
        {
            // Use GPU for large batches
            if n >= self.batch_threshold && self.gpu.is_some() {
                if let Some(ref mut gpu) = self.gpu {
                    match gpu.project_to_poincare_batch(vectors, curvature) {
                        Ok(results) => {
                            for (i, r) in results.into_iter().enumerate() {
                                vectors[i] = r;
                            }
                            self.stats.gpu_ops += n as u64;
                            return;
                        }
                        Err(e) => {
                            eprintln!("[7D Crystal] GPU projection failed, falling back to CPU: {}", e);
                            self.stats.gpu_fallbacks += 1;
                        }
                    }
                }
            }
        }
        
        // CPU fallback
        for v in vectors.iter_mut() {
            project_to_poincare_cpu(v, curvature);
        }
        self.stats.cpu_ops += n as u64;
    }
    
    /// Möbius addition
    pub fn mobius_add(
        &mut self,
        u: &[Vector7D],
        v: &[Vector7D],
        curvature: f32,
    ) -> Vec<Vector7D> {
        let n = u.len().min(v.len());
        self.stats.total_mobius_ops += n as u64;
        
        #[cfg(feature = "cuda")]
        {
            if n >= self.batch_threshold && self.gpu.is_some() {
                if let Some(ref mut gpu) = self.gpu {
                    match gpu.mobius_add_batch(u, v, curvature) {
                        Ok(results) => {
                            self.stats.gpu_ops += n as u64;
                            return results;
                        }
                        Err(e) => {
                            eprintln!("[7D Crystal] GPU Möbius add failed, falling back to CPU: {}", e);
                            self.stats.gpu_fallbacks += 1;
                        }
                    }
                }
            }
        }
        
        // CPU fallback
        let results: Vec<Vector7D> = (0..n)
            .map(|i| mobius_add_cpu(&u[i], &v[i], curvature))
            .collect();
        self.stats.cpu_ops += n as u64;
        results
    }
    
    /// Hyperbolic distances
    pub fn hyperbolic_distance(
        &mut self,
        u: &[Vector7D],
        v: &[Vector7D],
        curvature: f32,
    ) -> Vec<f32> {
        let n = u.len().min(v.len());
        self.stats.total_distance_ops += n as u64;
        
        #[cfg(feature = "cuda")]
        {
            if n >= self.batch_threshold && self.gpu.is_some() {
                if let Some(ref mut gpu) = self.gpu {
                    match gpu.hyperbolic_distance_batch(u, v, curvature) {
                        Ok(results) => {
                            self.stats.gpu_ops += n as u64;
                            return results;
                        }
                        Err(e) => {
                            eprintln!("[7D Crystal] GPU distance failed, falling back to CPU: {}", e);
                            self.stats.gpu_fallbacks += 1;
                        }
                    }
                }
            }
        }
        
        // CPU fallback
        let results: Vec<f32> = (0..n)
            .map(|i| hyperbolic_distance_cpu(&u[i], &v[i], curvature))
            .collect();
        self.stats.cpu_ops += n as u64;
        results
    }
    
    /// Holographic fold
    pub fn holographic_fold(
        &mut self,
        patterns: &[Vector7D],
        phases: &[f32],
    ) -> Vec<Vector7D> {
        let n = patterns.len().min(phases.len());
        self.stats.total_fold_ops += n as u64;
        
        #[cfg(feature = "cuda")]
        {
            if n >= self.batch_threshold && self.gpu.is_some() {
                if let Some(ref mut gpu) = self.gpu {
                    match gpu.holographic_fold_batch(patterns, phases) {
                        Ok(results) => {
                            self.stats.gpu_ops += n as u64;
                            return results;
                        }
                        Err(e) => {
                            eprintln!("[7D Crystal] GPU fold failed, falling back to CPU: {}", e);
                            self.stats.gpu_fallbacks += 1;
                        }
                    }
                }
            }
        }
        
        // CPU fallback - simplified fold
        let results: Vec<Vector7D> = (0..n)
            .map(|i| {
                let mut out = Vector7D::zeros();
                let phase = phases[i];
                let cos_p = (phase * PHI).cos();
                let sin_p = (phase * PHI).sin();
                
                for j in 0..7 {
                    let base = patterns[i].data[j];
                    let mut rotated = base * cos_p;
                    if j > 0 {
                        rotated += patterns[i].data[j-1] * sin_p * 0.5;
                    }
                    if j < 6 {
                        rotated += patterns[i].data[j+1] * sin_p * 0.5;
                    }
                    out.data[j] = rotated;
                }
                
                // Normalize
                let norm = out.norm();
                if norm > 1e-10 {
                    for j in 0..7 {
                        out.data[j] /= norm;
                    }
                }
                out
            })
            .collect();
        self.stats.cpu_ops += n as u64;
        results
    }
    
    /// Get compute statistics
    pub fn get_stats(&self) -> &ComputeStats {
        &self.stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ComputeStats::default();
    }
    
    /// Print performance summary
    pub fn print_summary(&self) {
        println!("\n╔═══════════════════════════════════════════════════════════════╗");
        println!("║       7D CRYSTAL COMPUTE SUMMARY                              ║");
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!("║ Backend: {:52} ║", self.backend_info());
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!("║ GPU Operations:     {:>10}                               ║", self.stats.gpu_ops);
        println!("║ CPU Operations:     {:>10}                               ║", self.stats.cpu_ops);
        println!("║ GPU Fallbacks:      {:>10}                               ║", self.stats.gpu_fallbacks);
        println!("╠═══════════════════════════════════════════════════════════════╣");
        println!("║ Total Projections:  {:>10}                               ║", self.stats.total_projections);
        println!("║ Total Möbius Ops:   {:>10}                               ║", self.stats.total_mobius_ops);
        println!("║ Total Distance Ops: {:>10}                               ║", self.stats.total_distance_ops);
        println!("║ Total Fold Ops:     {:>10}                               ║", self.stats.total_fold_ops);
        println!("╚═══════════════════════════════════════════════════════════════╝");
    }
}

impl Default for ComputeDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dispatcher_creation() {
        let disp = ComputeDispatcher::new();
        println!("Backend: {}", disp.backend_info());
    }
    
    #[test]
    fn test_projection() {
        let mut disp = ComputeDispatcher::new();
        let mut vectors = vec![
            Vector7D::new([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            Vector7D::new([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        ];
        
        disp.project_to_poincare(&mut vectors, PHI_INV);
        
        for v in &vectors {
            assert!(v.norm() < 1.0, "Vector should be inside unit ball");
        }
    }
    
    #[test]
    fn test_mobius_add() {
        let mut disp = ComputeDispatcher::new();
        let u = vec![Vector7D::new([0.1; 7])];
        let v = vec![Vector7D::new([0.1; 7])];
        
        let result = disp.mobius_add(&u, &v, PHI_INV);
        
        assert_eq!(result.len(), 1);
        assert!(result[0].norm() < 1.0);
    }
    
    #[test]
    fn test_distance() {
        let mut disp = ComputeDispatcher::new();
        let u = vec![Vector7D::new([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])];
        let v = vec![Vector7D::new([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])];
        
        let dist = disp.hyperbolic_distance(&u, &v, PHI_INV);
        
        assert_eq!(dist.len(), 1);
        assert!(dist[0] > 0.0);
    }
}
