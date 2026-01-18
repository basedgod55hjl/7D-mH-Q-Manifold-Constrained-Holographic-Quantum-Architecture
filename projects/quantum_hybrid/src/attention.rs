//! Quantum Attention Module

use super::simulator::QuantumCircuit;

pub struct QuantumAttention {
    dim: usize,
    num_heads: usize,
}

impl QuantumAttention {
    pub fn new(dim: usize, num_heads: usize) -> Self {
        Self { dim, num_heads }
    }

    pub fn forward(&self, query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
        // Compute simple dot-product attention as baseline
        let mut scores: Vec<f64> = keys
            .iter()
            .map(|k| query.iter().zip(k.iter()).map(|(q, k)| q * k).sum())
            .collect();

        // Softmax
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();
        let weights: Vec<f64> = exp_scores.iter().map(|e| e / sum).collect();

        // Weighted sum
        let mut output = vec![0.0; self.dim];
        for (w, k) in weights.iter().zip(keys.iter()) {
            for (o, &kv) in output.iter_mut().zip(k.iter()) {
                *o += w * kv;
            }
        }

        output
    }
}
