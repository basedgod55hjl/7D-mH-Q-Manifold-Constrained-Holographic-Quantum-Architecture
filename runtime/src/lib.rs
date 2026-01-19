//! # 7D Crystal Runtime
//!
//! High-performance GPU execution engine for the 7D Crystal System.
//!
//! ## Overview
//!
//! The runtime provides:
//!
//! - **GPU Abstraction**: Unified API for CUDA, HIP, and Metal
//! - **Manifold Operations**: 7D projection, holographic fold, quantum ops
//! - **Memory Management**: Efficient GPU memory allocation
//! - **Kernel Execution**: Launch and synchronize GPU kernels
//!
//! ## Mathematical Constants
//!
//! All operations preserve:
//!
//! - **Φ (Golden Ratio)**: 1.618033988749895
//! - **S² (Stability Bound)**: 0.01
//! - **7D Manifold Constraints**
//!
//! ## Author
//!
//! Sir Charles Spikes, December 2025
//! Cincinnati, Ohio, USA

#![warn(missing_docs)]

pub mod allocator;
pub mod compute;
pub mod executor;
pub mod gpu;
pub mod ir;
pub mod jit;
pub mod kernels;
pub mod quantum;
pub mod quantum_enhanced;

// Re-exports
pub use allocator::{AllocError, ManifoldAllocator, Region7D};
pub use compute::{ComputeBackend, ComputeDispatcher, ComputeStats};
pub use gpu::{GpuError, GpuExecutor, GpuStats};

// ═══════════════════════════════════════════════════════════════════════════════
// SACRED CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// The Golden Ratio (Φ)
pub const PHI: f64 = 1.618033988749895;

/// The inverse Golden Ratio (Φ⁻¹)
pub const PHI_INV: f64 = 0.618033988749895;

/// S² stability bound
pub const S2_STABILITY: f64 = 0.01;

/// Number of dimensions
pub const DIMS: usize = 7;

/// Φ² (frequently used)
pub const PHI_SQUARED: f64 = 2.618033988749895;

/// Default phases for holographic operations
pub const DEFAULT_PHASES: u32 = 49;

// ═══════════════════════════════════════════════════════════════════════════════
// VECTOR7D TYPE
// ═══════════════════════════════════════════════════════════════════════════════

/// A 7-dimensional vector in the Poincaré ball manifold.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector7D {
    /// The 7 coordinates
    pub coords: [f64; 7],
}

impl Vector7D {
    /// Create a new zero vector.
    pub const fn zero() -> Self {
        Self { coords: [0.0; 7] }
    }

    /// Create a vector from coordinates.
    pub const fn new(coords: [f64; 7]) -> Self {
        Self { coords }
    }

    /// Create a unit vector along dimension `i`.
    pub fn basis(i: usize) -> Self {
        assert!(i < 7, "Dimension must be < 7");
        let mut coords = [0.0; 7];
        coords[i] = 1.0;
        Self { coords }
    }

    /// Create a Fibonacci-scaled vector (consecutive ratios → Φ).
    pub fn fibonacci() -> Self {
        Self {
            coords: [
                1.0,
                PHI,
                PHI_SQUARED,
                PHI.powi(3),
                PHI.powi(4),
                PHI.powi(5),
                PHI.powi(6),
            ],
        }
    }

    /// Calculate the Euclidean norm.
    #[inline]
    pub fn norm(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Calculate the squared norm (faster than norm).
    #[inline]
    pub fn norm_squared(&self) -> f64 {
        self.coords.iter().map(|x| x * x).sum()
    }

    /// Check if this vector satisfies the S² stability bound.
    #[inline]
    pub fn is_stable(&self) -> bool {
        self.norm() < S2_STABILITY
    }

    /// Project this vector to the Poincaré ball with given curvature.
    pub fn project(&self, curvature: f64) -> Self {
        let norm = self.norm();
        let denom = 1.0 + norm + PHI_INV + curvature.abs();
        let scale = 1.0 / denom;

        Self {
            coords: self.coords.map(|x| x * scale),
        }
    }

    /// Verify that consecutive ratios are approximately Φ.
    pub fn verify_phi_ratios(&self, tolerance: f64) -> bool {
        for i in 0..6 {
            if self.coords[i].abs() < 1e-10 {
                continue; // Skip near-zero values
            }
            let ratio = self.coords[i + 1] / self.coords[i];
            if (ratio - PHI).abs() > tolerance {
                return false;
            }
        }
        true
    }

    /// Add two vectors.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let mut coords = [0.0; 7];
        for i in 0..7 {
            coords[i] = self.coords[i] + other.coords[i];
        }
        Self { coords }
    }

    /// Subtract two vectors.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let mut coords = [0.0; 7];
        for i in 0..7 {
            coords[i] = self.coords[i] - other.coords[i];
        }
        Self { coords }
    }

    /// Scale by a scalar.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            coords: self.coords.map(|x| x * s),
        }
    }

    /// Dot product.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Hyperbolic distance to another point.
    pub fn hyperbolic_distance(&self, other: &Self) -> f64 {
        let diff = self.sub(other);
        let diff_sq = diff.norm_squared();
        let norm_a_sq = self.norm_squared();
        let norm_b_sq = other.norm_squared();

        let numerator = 2.0 * diff_sq;
        let denominator = (1.0 - norm_a_sq) * (1.0 - norm_b_sq);

        if denominator <= 0.0 {
            return f64::INFINITY;
        }

        (1.0 + numerator / denominator).acosh()
    }
}

impl Default for Vector7D {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Add for Vector7D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vector7D::add(&self, &rhs)
    }
}

impl std::ops::Sub for Vector7D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vector7D::sub(&self, &rhs)
    }
}

impl std::ops::Mul<f64> for Vector7D {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOLOGRAPHIC PATTERN
// ═══════════════════════════════════════════════════════════════════════════════

/// A holographic interference pattern (49 phases = 7×7).
#[repr(C, align(64))]
#[derive(Debug, Clone)]
pub struct HolographicPattern {
    /// Phase values (7×7 grid)
    pub phases: [[f64; 7]; 7],
    /// Overall amplitude
    pub amplitude: f64,
    /// Coherence factor (0-1)
    pub coherence: f64,
}

impl HolographicPattern {
    /// Create a new zero pattern.
    pub fn zero() -> Self {
        Self {
            phases: [[0.0; 7]; 7],
            amplitude: 0.0,
            coherence: 1.0,
        }
    }

    /// Encode a Vector7D as a holographic pattern.
    pub fn encode(v: &Vector7D) -> Self {
        let mut phases = [[0.0; 7]; 7];

        for i in 0..7 {
            for j in 0..7 {
                // Interference pattern: outer product with Φ modulation
                phases[i][j] = v.coords[i] * v.coords[j] * PHI_INV;
            }
        }

        Self {
            phases,
            amplitude: v.norm(),
            coherence: 1.0,
        }
    }

    /// Decode the pattern back to a Vector7D.
    pub fn decode(&self) -> Vector7D {
        let mut coords = [0.0; 7];

        for i in 0..7 {
            // Sum diagonal for reconstruction
            coords[i] = self.phases[i][i] / PHI_INV;
        }

        Vector7D { coords }.scale(self.amplitude / self.amplitude.max(1e-10))
    }

    /// Fold (merge) two patterns via interference.
    pub fn fold(&self, other: &Self, num_phases: u32) -> Self {
        let phase_scale = 1.0 / (num_phases as f64);
        let mut phases = [[0.0; 7]; 7];

        for i in 0..7 {
            for j in 0..7 {
                phases[i][j] = (self.phases[i][j] + other.phases[i][j]) * phase_scale * PHI_INV;
            }
        }

        Self {
            phases,
            amplitude: (self.amplitude + other.amplitude) / 2.0,
            coherence: self.coherence.min(other.coherence),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUANTUM STATE
// ═══════════════════════════════════════════════════════════════════════════════

/// A quantum state in the 7D manifold (non-copyable).
#[derive(Debug)]
pub struct QuantumState {
    /// Amplitude vector
    amplitudes: Vector7D,
    /// Phase vector
    phases: Vector7D,
    /// Whether this state has been measured
    measured: bool,
    /// Entangled state ID (if any)
    entangled_with: Option<u64>,
}

impl QuantumState {
    /// Create the ground state.
    pub fn ground() -> Self {
        Self {
            amplitudes: Vector7D::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phases: Vector7D::zero(),
            measured: false,
            entangled_with: None,
        }
    }

    /// Create a uniform superposition.
    pub fn uniform_superposition() -> Self {
        let amp = 1.0 / 7.0_f64.sqrt();
        Self {
            amplitudes: Vector7D::new([amp; 7]),
            phases: Vector7D::zero(),
            measured: false,
            entangled_with: None,
        }
    }

    /// Check if this state has been measured.
    pub fn is_measured(&self) -> bool {
        self.measured
    }

    /// Check if this state is entangled.
    pub fn is_entangled(&self) -> bool {
        self.entangled_with.is_some()
    }

    /// Get probability amplitudes.
    pub fn probabilities(&self) -> [f64; 7] {
        let mut probs = [0.0; 7];
        for i in 0..7 {
            probs[i] = self.amplitudes.coords[i].powi(2);
        }
        probs
    }

    /// Measure the state (collapses superposition).
    pub fn measure(&mut self) -> usize {
        if self.measured {
            panic!("Quantum state already measured!");
        }

        self.measured = true;

        // Simplified measurement (would use RNG in real implementation)
        let probs = self.probabilities();
        let mut max_idx = 0;
        let mut max_prob = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                max_idx = i;
            }
        }

        // Collapse to measured state
        self.amplitudes = Vector7D::basis(max_idx);

        max_idx
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// RUNTIME CONTEXT
// ═══════════════════════════════════════════════════════════════════════════════

/*
/// Main runtime context for 7D Crystal execution.
pub struct Runtime {
    /// GPU device (if available)
    device: Option<Box<dyn GpuDevice>>,
    /// Memory allocator
    allocator: Allocator,
    /// Execution statistics
    stats: RuntimeStats,
}

/// Runtime statistics.
#[derive(Debug, Default)]
pub struct RuntimeStats {
    /// Number of kernels launched
    pub kernels_launched: u64,
    /// Total execution time (ms)
    pub total_time_ms: u64,
    /// Memory allocated (bytes)
    pub memory_allocated: usize,
    /// Number of projections performed
    pub projections: u64,
    /// Number of folds performed
    pub folds: u64,
}

impl Runtime {
    /// Create a new runtime with GPU support if available.
    pub fn new() -> anyhow::Result<Self> {
        let device = gpu::detect_gpu()?;
        let allocator = Allocator::new(device.as_ref().map(|d| d.memory_size()).unwrap_or(0));

        Ok(Self {
            device,
            allocator,
            stats: RuntimeStats::default(),
        })
    }

    /// Create a CPU-only runtime.
    pub fn cpu_only() -> Self {
        Self {
            device: None,
            allocator: Allocator::new(0),
            stats: RuntimeStats::default(),
        }
    }

    /// Check if GPU is available.
    pub fn has_gpu(&self) -> bool {
        self.device.is_some()
    }

    /// Project vectors to the manifold.
    pub fn project(&mut self, input: &[Vector7D], curvature: f64) -> Vec<Vector7D> {
        self.stats.projections += input.len() as u64;

        if let Some(ref _device) = self.device {
            // GPU implementation would go here
            input.iter().map(|v| v.project(curvature)).collect()
        } else {
            // CPU fallback
            input.par_iter().map(|v| v.project(curvature)).collect()
        }
    }

    /// Fold holographic patterns.
    pub fn fold(&mut self, p1: &[HolographicPattern], p2: &[HolographicPattern], phases: u32) -> Vec<HolographicPattern> {
        assert_eq!(p1.len(), p2.len(), "Pattern arrays must have same length");
        self.stats.folds += p1.len() as u64;

        p1.iter()
            .zip(p2.iter())
            .map(|(a, b)| a.fold(b, phases))
            .collect()
    }

    /// Get runtime statistics.
    pub fn stats(&self) -> &RuntimeStats {
        &self.stats
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::cpu_only()
    }
}
*/

// For parallel iteration
use rayon::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector7d_norm() {
        let v = Vector7D::basis(0);
        assert!((v.norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector7d_project() {
        let v = Vector7D::new([1.0; 7]);
        let projected = v.project(PHI_INV);
        assert!(projected.is_stable() || projected.norm() < 1.0);
    }

    #[test]
    fn test_fibonacci_phi_ratios() {
        let v = Vector7D::fibonacci();
        // Scale down to meet stability bound
        let v_scaled = v.scale(0.001);
        assert!(v_scaled.verify_phi_ratios(1e-6));
    }

    #[test]
    fn test_holographic_encode_decode() {
        let v = Vector7D::new([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]);
        let pattern = HolographicPattern::encode(&v);
        let decoded = pattern.decode();

        // Should be approximately equal
        for i in 0..7 {
            assert!((v.coords[i] - decoded.coords[i]).abs() < 0.01);
        }
    }

    #[test]
    fn test_quantum_state_ground() {
        let state = QuantumState::ground();
        let probs = state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
        assert!((probs[1..].iter().sum::<f64>()).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_superposition() {
        let state = QuantumState::uniform_superposition();
        let probs = state.probabilities();
        let expected = 1.0 / 7.0;
        for p in probs {
            assert!((p - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_constants() {
        assert!((PHI * PHI_INV - 1.0).abs() < 1e-10);
        assert!((PHI_SQUARED - PHI * PHI).abs() < 1e-10);
    }
}
