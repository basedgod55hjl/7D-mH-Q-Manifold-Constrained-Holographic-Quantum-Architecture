// File: transformer/src/lib.rs
// 7D Crystal Transformer - Novel Architecture with Manifold Attention
// Revolutionary transformer with 7D hyperbolic geometry
// Discovered by Sir Charles Spikes, December 24, 2025

pub mod attention;
pub mod layers;

use std::f64::consts::PI;

// 7D CRYSTAL CONSTANTS
pub const PHI: f64 = 1.618033988749894848204586834365638;
pub const PHI_INV: f64 = 0.618033988749894848204586834365638;
pub const PHI_SQUARED: f64 = 2.618033988749894848204586834365638;
pub const S2_STABILITY: f64 = 0.01;
pub const MANIFOLD_DIMS: usize = 7;
pub const CURVATURE: f64 = -1.0;

pub const PHI_BASIS: [f64; 7] = [
    1.0, 1.618033988749895, 2.618033988749895, 4.23606797749979,
    6.854101966249685, 11.090169943749475, 17.94427190999916,
];

/// Project vector to 7D Poincaré ball
pub fn project_to_poincare(v: &[f64], curvature: f64) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = 1.0 + norm + PHI_INV + curvature.abs();
    let scale = if norm > S2_STABILITY { 1.0 / (denom * (norm / S2_STABILITY)) } else { 1.0 / denom };
    v.iter().enumerate().map(|(i, &x)| {
        let phi_weight = if i < 7 { PHI_BASIS[i] / PHI_BASIS[6] } else { 1.0 };
        x * scale * phi_weight
    }).collect()
}

/// Hyperbolic distance in Poincaré ball
pub fn hyperbolic_distance(u: &[f64], v: &[f64]) -> f64 {
    let u_norm_sq: f64 = u.iter().map(|x| x * x).sum();
    let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
    let diff_norm_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let num = 2.0 * diff_norm_sq;
    let den = (1.0 - u_norm_sq) * (1.0 - v_norm_sq);
    (1.0 + num / den.max(1e-10)).acosh()
}

/// Möbius addition
pub fn mobius_add(u: &[f64], v: &[f64], curvature: f64) -> Vec<f64> {
    let c = -curvature;
    let u_norm_sq: f64 = u.iter().map(|x| x * x).sum();
    let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
    let uv_dot: f64 = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    let num_u = 1.0 + 2.0 * c * uv_dot + c * v_norm_sq;
    let num_v = 1.0 - c * u_norm_sq;
    let den = 1.0 + 2.0 * c * uv_dot + c * c * u_norm_sq * v_norm_sq;
    u.iter().zip(v.iter()).map(|(&ui, &vi)| (num_u * ui + num_v * vi) / den.max(1e-10)).collect()
}

#[derive(Debug, Clone)]
pub struct Crystal7DConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub manifold_enabled: bool,
    pub curvature: f64,
    pub phi_attention: bool,
    pub use_swiglu: bool,
    pub rope_theta: f64,
}

impl Default for Crystal7DConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096, intermediate_size: 14336, num_layers: 32,
            num_attention_heads: 32, num_kv_heads: 8, vocab_size: 128256,
            max_seq_len: 131072, manifold_enabled: true, curvature: CURVATURE,
            phi_attention: true, use_swiglu: true, rope_theta: 500000.0,
        }
    }
}

impl Crystal7DConfig {
    pub fn head_dim(&self) -> usize { self.hidden_size / self.num_attention_heads }
    pub fn kv_dim(&self) -> usize { self.num_kv_heads * self.head_dim() }
    
    pub fn crystal_1_5b() -> Self {
        Self { hidden_size: 1536, intermediate_size: 8960, num_layers: 28,
               num_attention_heads: 12, num_kv_heads: 2, ..Default::default() }
    }
    pub fn crystal_8b() -> Self { Self::default() }
    pub fn crystal_32b() -> Self {
        Self { hidden_size: 6144, intermediate_size: 16384, num_layers: 60,
               num_attention_heads: 48, num_kv_heads: 8, ..Default::default() }
    }
    pub fn crystal_70b() -> Self {
        Self { hidden_size: 8192, intermediate_size: 28672, num_layers: 80,
               num_attention_heads: 64, num_kv_heads: 8, ..Default::default() }
    }
}

pub struct Crystal7DAttention {
    config: Crystal7DConfig,
    phi_weights: Vec<f64>,
}

impl Crystal7DAttention {
    pub fn new(config: Crystal7DConfig) -> Self {
        let head_dim = config.head_dim();
        let phi_weights: Vec<f64> = (0..head_dim).map(|i| {
            if i < 7 { PHI_BASIS[i] / PHI_BASIS[6] } else { 1.0 }
        }).collect();
        Self { config, phi_weights }
    }
    
    pub fn compute_scores(&self, q: &[f64], k: &[f64], seq_len_q: usize, seq_len_k: usize) -> Vec<f64> {
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let mut scores = vec![0.0; seq_len_q * seq_len_k * num_heads];
        
        for h in 0..num_heads {
            for qi in 0..seq_len_q {
                for ki in 0..seq_len_k {
                    let q_off = (qi * num_heads + h) * head_dim;
                    let k_off = (ki * num_heads + h) * head_dim;
                    let mut score = 0.0;
                    for d in 0..head_dim {
                        let w = if self.config.phi_attention { self.phi_weights[d] } else { 1.0 };
                        score += q[q_off + d] * k[k_off + d] * w;
                    }
                    scores[(h * seq_len_q + qi) * seq_len_k + ki] = score * scale;
                }
            }
        }
        scores
    }
}

pub struct Crystal7DRoPE {
    dim: usize, max_seq_len: usize, theta: f64, phi_modulated: bool,
    cos_cache: Vec<f64>, sin_cache: Vec<f64>,
}

impl Crystal7DRoPE {
    pub fn new(dim: usize, max_seq_len: usize, theta: f64, phi_modulated: bool) -> Self {
        let mut rope = Self { dim, max_seq_len, theta, phi_modulated,
            cos_cache: vec![0.0; max_seq_len * dim / 2],
            sin_cache: vec![0.0; max_seq_len * dim / 2] };
        for pos in 0..max_seq_len {
            for i in 0..(dim / 2) {
                let freq = 1.0 / theta.powf((2 * i) as f64 / dim as f64);
                let mod_freq = if phi_modulated && i < 7 { freq * (PHI_BASIS[i] / PHI_BASIS[6]) } else { freq };
                let angle = pos as f64 * mod_freq;
                let idx = pos * (dim / 2) + i;
                rope.cos_cache[idx] = angle.cos();
                rope.sin_cache[idx] = angle.sin();
            }
        }
        rope
    }
    
    pub fn apply(&self, q: &mut [f64], k: &mut [f64], positions: &[usize]) {
        for (token_idx, &pos) in positions.iter().enumerate() {
            if pos >= self.max_seq_len { continue; }
            let cache_off = pos * (self.dim / 2);
            for i in 0..(self.dim / 2) {
                let idx0 = token_idx * self.dim + i * 2;
                let (cos, sin) = (self.cos_cache[cache_off + i], self.sin_cache[cache_off + i]);
                let (q0, q1) = (q[idx0], q[idx0 + 1]);
                q[idx0] = q0 * cos - q1 * sin;
                q[idx0 + 1] = q0 * sin + q1 * cos;
                let (k0, k1) = (k[idx0], k[idx0 + 1]);
                k[idx0] = k0 * cos - k1 * sin;
                k[idx0 + 1] = k0 * sin + k1 * cos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_phi_basis() {
        for i in 0..6 {
            let ratio = PHI_BASIS[i + 1] / PHI_BASIS[i];
            assert!((ratio - PHI).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_poincare_projection() {
        let v = vec![0.5, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2];
        let projected = project_to_poincare(&v, CURVATURE);
        let norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0);
    }
}
