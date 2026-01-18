// File: transformer/src/attention.rs
// 7D Crystal Attention Mechanisms
use super::*;

/// Flash Attention with 7D manifold constraints
pub struct Crystal7DFlashAttention {
    pub config: Crystal7DConfig,
    pub chunk_size: usize,
}

impl Crystal7DFlashAttention {
    pub fn new(config: Crystal7DConfig) -> Self {
        Self { config, chunk_size: 256 }
    }
    
    /// Online softmax with tiling for memory efficiency
    pub fn forward(&self, q: &[f64], k: &[f64], v: &[f64], seq_len: usize) -> Vec<f64> {
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        
        let mut output = vec![0.0; seq_len * num_heads * head_dim];
        
        for h in 0..num_heads {
            for q_chunk_start in (0..seq_len).step_by(self.chunk_size) {
                let q_chunk_end = (q_chunk_start + self.chunk_size).min(seq_len);
                
                for qi in q_chunk_start..q_chunk_end {
                    let mut max_score = f64::NEG_INFINITY;
                    let mut scores = vec![0.0; seq_len];
                    
                    // Compute scores
                    for ki in 0..seq_len {
                        let q_off = (qi * num_heads + h) * head_dim;
                        let k_off = (ki * num_heads + h) * head_dim;
                        
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            let phi_w = if d < 7 && self.config.phi_attention {
                                PHI_BASIS[d] / PHI_BASIS[6]
                            } else { 1.0 };
                            score += q[q_off + d] * k[k_off + d] * phi_w;
                        }
                        scores[ki] = score * scale;
                        if scores[ki] > max_score { max_score = scores[ki]; }
                    }
                    
                    // Softmax
                    let mut exp_sum = 0.0;
                    for s in &mut scores {
                        *s = (*s - max_score).exp();
                        exp_sum += *s;
                    }
                    for s in &mut scores { *s /= exp_sum; }
                    
                    // Weighted sum
                    let out_off = (qi * num_heads + h) * head_dim;
                    for d in 0..head_dim {
                        let mut val = 0.0;
                        for ki in 0..seq_len {
                            let v_off = (ki * num_heads + h) * head_dim;
                            val += scores[ki] * v[v_off + d];
                        }
                        output[out_off + d] = val;
                    }
                }
            }
        }
        output
    }
}

/// Grouped Query Attention with 7D manifold
pub struct Crystal7DGQA {
    pub config: Crystal7DConfig,
}

impl Crystal7DGQA {
    pub fn new(config: Crystal7DConfig) -> Self {
        Self { config }
    }
    
    pub fn expand_kv(&self, kv: &[f64], seq_len: usize) -> Vec<f64> {
        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let repeat = num_heads / num_kv_heads;
        
        let mut expanded = vec![0.0; seq_len * num_heads * head_dim];
        
        for pos in 0..seq_len {
            for kv_h in 0..num_kv_heads {
                let src_off = (pos * num_kv_heads + kv_h) * head_dim;
                for r in 0..repeat {
                    let h = kv_h * repeat + r;
                    let dst_off = (pos * num_heads + h) * head_dim;
                    for d in 0..head_dim {
                        expanded[dst_off + d] = kv[src_off + d];
                    }
                }
            }
        }
        expanded
    }
}

/// Multi-Query Attention variant
pub struct Crystal7DMQA {
    pub config: Crystal7DConfig,
}

impl Crystal7DMQA {
    pub fn new(config: Crystal7DConfig) -> Self {
        Self { config }
    }
}

/// Sliding Window Attention for long sequences
pub struct Crystal7DSlidingWindow {
    pub config: Crystal7DConfig,
    pub window_size: usize,
}

impl Crystal7DSlidingWindow {
    pub fn new(config: Crystal7DConfig, window_size: usize) -> Self {
        Self { config, window_size }
    }
    
    pub fn create_mask(&self, seq_len: usize) -> Vec<bool> {
        let mut mask = vec![false; seq_len * seq_len];
        for i in 0..seq_len {
            let start = if i >= self.window_size { i - self.window_size } else { 0 };
            for j in start..=i {
                mask[i * seq_len + j] = true;
            }
        }
        mask
    }
}
