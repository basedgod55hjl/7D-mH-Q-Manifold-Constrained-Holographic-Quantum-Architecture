// File: model_runner/src/sampler.rs
// Token Sampling with Temperature and Top-P/Top-K
// 7D Crystal System

use anyhow::Result;
use rand::Rng;

/// Sampling parameters
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.0,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
        }
    }
}

/// Token sampler
#[derive(Debug, Clone)]
pub struct Sampler;

impl Default for Sampler {
    fn default() -> Self {
        Self
    }
}

impl Sampler {
    /// Sample next token from logits
    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams) -> Result<u32> {
        let vocab_size = logits.len();

        // Greedy decoding
        if params.temperature <= 0.0 || params.top_k == 1 {
            let (max_idx, _) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            return Ok(max_idx as u32);
        }

        // Apply temperature
        let scaled: Vec<f32> = logits.iter().map(|&x| x / params.temperature).collect();

        // Top-K filtering
        let mut indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let k = params.top_k.min(vocab_size);
        let top_k_items: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

        // Softmax on top-k
        let max_val = top_k_items[0].1;
        let mut probs: Vec<(usize, f32)> = top_k_items
            .iter()
            .map(|&(idx, val)| (idx, (val - max_val).exp()))
            .collect();

        let sum: f32 = probs.iter().map(|x| x.1).sum();
        for p in &mut probs {
            p.1 /= sum;
        }

        // Top-P (nucleus) filtering
        if params.top_p < 1.0 {
            let mut cumsum = 0.0;
            let mut cutoff_idx = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= params.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff_idx);

            // Renormalize
            let new_sum: f32 = probs.iter().map(|x| x.1).sum();
            for p in &mut probs {
                p.1 /= new_sum;
            }
        }

        // Sample from distribution
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        let fallback = probs[0].0;

        for &(idx, prob) in probs.iter() {
            cumsum += prob;
            if r <= cumsum {
                return Ok(idx as u32);
            }
        }

        // Fallback to most likely
        Ok(fallback as u32)
    }

    /// Sample with repetition penalty
    pub fn sample_with_penalty(
        &mut self,
        logits: &[f32],
        params: &SamplingParams,
        previous_tokens: &[u32],
    ) -> Result<u32> {
        if params.repetition_penalty == 1.0 || previous_tokens.is_empty() {
            return self.sample(logits, params);
        }

        let mut modified_logits = logits.to_vec();

        // Apply repetition penalty
        for &token in previous_tokens {
            let idx = token as usize;
            if idx < modified_logits.len() {
                if modified_logits[idx] > 0.0 {
                    modified_logits[idx] /= params.repetition_penalty;
                } else {
                    modified_logits[idx] *= params.repetition_penalty;
                }
            }
        }

        self.sample(&modified_logits, params)
    }
}
