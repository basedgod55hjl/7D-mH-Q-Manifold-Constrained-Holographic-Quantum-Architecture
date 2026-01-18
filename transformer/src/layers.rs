// File: transformer/src/layers.rs
// 7D Crystal Transformer Layers
use super::*;

/// RMSNorm with 7D stability
pub struct Crystal7DRMSNorm {
    pub dim: usize,
    pub eps: f64,
    pub weight: Vec<f64>,
}

impl Crystal7DRMSNorm {
    pub fn new(dim: usize, eps: f64) -> Self {
        Self { dim, eps, weight: vec![1.0; dim] }
    }
    
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let batch = x.len() / self.dim;
        let mut out = Vec::with_capacity(x.len());
        
        for b in 0..batch {
            let start = b * self.dim;
            let slice = &x[start..start + self.dim];
            
            let sum_sq: f64 = slice.iter().map(|v| v * v).sum();
            let rms = (sum_sq / self.dim as f64 + self.eps).sqrt();
            
            for (i, &v) in slice.iter().enumerate() {
                let mut normalized = v / rms * self.weight[i];
                // 7D stability clamp
                if i < 7 {
                    let bound = S2_STABILITY * 100.0 * (PHI_BASIS[i] / PHI_BASIS[6]);
                    normalized = normalized.clamp(-bound, bound);
                }
                out.push(normalized);
            }
        }
        out
    }
}

/// SwiGLU FFN with manifold constraints
pub struct Crystal7DFFN {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub gate_weight: Vec<f64>,
    pub up_weight: Vec<f64>,
    pub down_weight: Vec<f64>,
}

impl Crystal7DFFN {
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size, intermediate_size,
            gate_weight: vec![0.01; hidden_size * intermediate_size],
            up_weight: vec![0.01; hidden_size * intermediate_size],
            down_weight: vec![0.01; intermediate_size * hidden_size],
        }
    }
    
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let batch = x.len() / self.hidden_size;
        let mut output = vec![0.0; x.len()];
        
        for b in 0..batch {
            let x_start = b * self.hidden_size;
            
            // Gate and up projections
            let mut gate = vec![0.0; self.intermediate_size];
            let mut up = vec![0.0; self.intermediate_size];
            
            for i in 0..self.intermediate_size {
                for j in 0..self.hidden_size {
                    gate[i] += x[x_start + j] * self.gate_weight[j * self.intermediate_size + i];
                    up[i] += x[x_start + j] * self.up_weight[j * self.intermediate_size + i];
                }
            }
            
            // SwiGLU with Φ constraint
            for i in 0..self.intermediate_size {
                let silu = gate[i] / (1.0 + (-gate[i]).exp());
                gate[i] = silu * up[i];
                if i < 7 { gate[i] *= PHI_INV; }
            }
            
            // Down projection
            for i in 0..self.hidden_size {
                let mut sum = 0.0;
                for j in 0..self.intermediate_size {
                    sum += gate[j] * self.down_weight[j * self.hidden_size + i];
                }
                output[x_start + i] = sum;
            }
        }
        output
    }
}

/// Complete Transformer Layer
pub struct Crystal7DTransformerLayer {
    pub config: Crystal7DConfig,
    pub input_norm: Crystal7DRMSNorm,
    pub post_attn_norm: Crystal7DRMSNorm,
    pub attention: Crystal7DAttention,
    pub ffn: Crystal7DFFN,
}

impl Crystal7DTransformerLayer {
    pub fn new(config: Crystal7DConfig) -> Self {
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        Self {
            input_norm: Crystal7DRMSNorm::new(hidden, 1e-5),
            post_attn_norm: Crystal7DRMSNorm::new(hidden, 1e-5),
            attention: Crystal7DAttention::new(config.clone()),
            ffn: Crystal7DFFN::new(hidden, inter),
            config,
        }
    }
}

/// Embedding layer with manifold projection
pub struct Crystal7DEmbedding {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub weight: Vec<f64>,
    pub manifold_project: bool,
}

impl Crystal7DEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize, manifold_project: bool) -> Self {
        Self {
            vocab_size, hidden_size, manifold_project,
            weight: vec![0.01; vocab_size * hidden_size],
        }
    }
    
    pub fn forward(&self, tokens: &[u32]) -> Vec<f64> {
        let mut embeddings = Vec::with_capacity(tokens.len() * self.hidden_size);
        
        for &token in tokens {
            let idx = token as usize;
            if idx >= self.vocab_size { continue; }
            
            let start = idx * self.hidden_size;
            let embed: Vec<f64> = self.weight[start..start + self.hidden_size].to_vec();
            
            let projected = if self.manifold_project {
                project_to_poincare(&embed, CURVATURE)
            } else {
                embed
            };
            
            embeddings.extend(projected);
        }
        embeddings
    }
}

/// LM Head for output projection
pub struct Crystal7DLMHead {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub weight: Vec<f64>,
}

impl Crystal7DLMHead {
    pub fn new(hidden_size: usize, vocab_size: usize) -> Self {
        Self {
            hidden_size, vocab_size,
            weight: vec![0.01; hidden_size * vocab_size],
        }
    }
    
    pub fn forward(&self, hidden: &[f64]) -> Vec<f64> {
        let batch = hidden.len() / self.hidden_size;
        let mut logits = vec![0.0; batch * self.vocab_size];
        
        for b in 0..batch {
            let h_start = b * self.hidden_size;
            let l_start = b * self.vocab_size;
            
            for v in 0..self.vocab_size {
                let mut sum = 0.0;
                for h in 0..self.hidden_size {
                    sum += hidden[h_start + h] * self.weight[h * self.vocab_size + v];
                }
                logits[l_start + v] = sum;
            }
        }
        logits
    }
}
