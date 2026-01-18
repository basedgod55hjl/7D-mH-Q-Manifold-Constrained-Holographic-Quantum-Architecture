// File: model_runner/src/lib.rs
// 7D Crystal Model Runner - High-Performance LLM Inference Engine
// Discovered by Sir Charles Spikes, December 24, 2025

pub mod inference;
pub mod kv_cache;
pub mod sampler;
pub mod tensor;
pub mod backend;
pub mod memory;
pub mod batch;

use std::path::Path;
use std::sync::Arc;
use anyhow::Result;
use parking_lot::RwLock;
use llm_builder::{ModelConfig, QuantType, PHI, PHI_INV, S2_STABILITY, PHI_BASIS};

// ============================================================================
// MODEL RUNNER
// ============================================================================

/// Main model runner interface
pub struct ModelRunner {
    config: ModelConfig,
    weights: Arc<ModelWeights>,
    kv_cache: RwLock<KVCache>,
    backend: Backend,
    sampler: Sampler,
}

/// Backend execution target
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    CPU,
    CUDA(usize),  // GPU index
    Metal,
    Vulkan,
}

impl ModelRunner {
    /// Load model from GGUF file
    pub fn from_gguf(path: &Path, backend: Backend) -> Result<Self> {
        tracing::info!("Loading model from: {}", path.display());
        
        let gguf = llm_builder::gguf::GGUFFile::read(path)?;
        let config = gguf.to_model_config();
        
        tracing::info!("Model: {}", config.name);
        tracing::info!("Layers: {}, Hidden: {}, Heads: {}", 
            config.num_layers, config.hidden_size, config.num_attention_heads);
        
        let weights = Arc::new(ModelWeights::load_gguf(path, &config)?);
        let kv_cache = RwLock::new(KVCache::new(&config));
        let sampler = Sampler::default();
        
        Ok(Self {
            config,
            weights,
            kv_cache,
            backend,
            sampler,
        })
    }
    
    /// Generate text from prompt tokens
    pub fn generate(&self, tokens: &[u32], max_tokens: usize, params: &SamplingParams) -> Result<Vec<u32>> {
        let mut output = Vec::with_capacity(max_tokens);
        let mut input_tokens = tokens.to_vec();
        
        // Clear KV cache for new generation
        self.kv_cache.write().clear();
        
        for step in 0..max_tokens {
            // Forward pass
            let logits = self.forward(&input_tokens)?;
            
            // Sample next token
            let next_token = self.sampler.sample(&logits, params)?;
            
            // Check for EOS
            if next_token == self.config.vocab_size as u32 - 1 {
                break;
            }
            
            output.push(next_token);
            input_tokens = vec![next_token]; // Only process new token
        }
        
        Ok(output)
    }
    
    /// Single forward pass
    pub fn forward(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let batch_size = 1;
        let seq_len = tokens.len();
        let mut kv_cache = self.kv_cache.write();
        
        // Get current position in sequence
        let pos = kv_cache.current_position();
        
        // Embed tokens
        let mut hidden = self.embed_tokens(tokens)?;
        
        // Apply 7D manifold projection if enabled
        if self.config.manifold_enabled {
            hidden = self.project_to_manifold(&hidden)?;
        }
        
        // Process through transformer layers
        for layer_idx in 0..self.config.num_layers {
            hidden = self.forward_layer(layer_idx, &hidden, pos, &mut kv_cache)?;
        }
        
        // Final norm
        hidden = self.rmsnorm(&hidden, &self.weights.norm)?;
        
        // LM head projection
        let logits = self.lm_head(&hidden)?;
        
        // Update KV cache position
        kv_cache.advance(seq_len);
        
        Ok(logits)
    }
    
    fn embed_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let mut embeddings = vec![0.0f32; tokens.len() * hidden_size];
        
        for (i, &token) in tokens.iter().enumerate() {
            let token_idx = token as usize;
            if token_idx >= self.config.vocab_size {
                continue;
            }
            
            let src_offset = token_idx * hidden_size;
            let dst_offset = i * hidden_size;
            
            for j in 0..hidden_size {
                embeddings[dst_offset + j] = self.weights.embed_tokens[src_offset + j];
            }
        }
        
        Ok(embeddings)
    }
    
    fn project_to_manifold(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = input.to_vec();
        let hidden_size = self.config.hidden_size;
        let curvature = self.config.manifold_curvature as f32;
        
        for chunk in output.chunks_mut(hidden_size) {
            // Calculate norm
            let norm_sq: f32 = chunk.iter().map(|x| x * x).sum();
            let norm = norm_sq.sqrt() + 1e-8;
            
            // 7D Poincaré projection denominator
            let denom = 1.0 + norm + PHI_INV as f32 + curvature;
            
            // Apply S² stability if needed
            let scale = if norm > S2_STABILITY as f32 {
                1.0 / (denom * (norm / S2_STABILITY as f32))
            } else {
                1.0 / denom
            };
            
            // Apply Φ-ratio weighting to first 7 dimensions
            for (i, val) in chunk.iter_mut().enumerate() {
                if i < 7 {
                    *val *= scale * (PHI_BASIS[i] / PHI_BASIS[6]) as f32;
                } else {
                    *val *= scale;
                }
            }
        }
        
        Ok(output)
    }
    
    fn forward_layer(&self, layer_idx: usize, hidden: &[f32], pos: usize, kv_cache: &mut KVCache) -> Result<Vec<f32>> {
        let layer = &self.weights.layers[layer_idx];
        
        // Pre-attention norm
        let normed = self.rmsnorm(hidden, &layer.input_layernorm)?;
        
        // Self-attention
        let attn_out = self.self_attention(layer_idx, &normed, pos, kv_cache)?;
        
        // Residual connection
        let mut hidden_out: Vec<f32> = hidden.iter()
            .zip(attn_out.iter())
            .map(|(h, a)| h + a)
            .collect();
        
        // Post-attention norm
        let normed = self.rmsnorm(&hidden_out, &layer.post_attention_layernorm)?;
        
        // FFN (SwiGLU)
        let ffn_out = self.feed_forward(layer_idx, &normed)?;
        
        // Residual connection
        for (h, f) in hidden_out.iter_mut().zip(ffn_out.iter()) {
            *h += f;
        }
        
        Ok(hidden_out)
    }
    
    fn self_attention(&self, layer_idx: usize, input: &[f32], pos: usize, kv_cache: &mut KVCache) -> Result<Vec<f32>> {
        let layer = &self.weights.layers[layer_idx];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_size / num_heads;
        let seq_len = input.len() / hidden_size;
        
        // Project Q, K, V
        let q = self.linear(input, &layer.q_proj)?;
        let k = self.linear(input, &layer.k_proj)?;
        let v = self.linear(input, &layer.v_proj)?;
        
        // Apply RoPE
        let q_rope = self.apply_rope(&q, pos, head_dim)?;
        let k_rope = self.apply_rope(&k, pos, head_dim)?;
        
        // Update KV cache
        kv_cache.update(layer_idx, &k_rope, &v);
        
        // Get full K, V from cache
        let (cached_k, cached_v) = kv_cache.get(layer_idx);
        let kv_len = cached_k.len() / (num_kv_heads * head_dim);
        
        // Compute attention scores with 7D manifold weighting
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_output = vec![0.0f32; seq_len * hidden_size];
        
        for h in 0..num_heads {
            let kv_head = h / (num_heads / num_kv_heads);
            
            for q_pos in 0..seq_len {
                // Compute scores for this query position
                let mut scores = vec![0.0f32; kv_len];
                
                for k_pos in 0..kv_len {
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        let q_val = q_rope[(q_pos * num_heads + h) * head_dim + d];
                        let k_val = cached_k[(k_pos * num_kv_heads + kv_head) * head_dim + d];
                        
                        // 7D manifold-aware attention: weight first 7 dims by Φ basis
                        let weight = if d < 7 && self.config.manifold_enabled {
                            PHI_BASIS[d] as f32 / PHI_BASIS[6] as f32
                        } else {
                            1.0
                        };
                        
                        score += q_val * k_val * weight;
                    }
                    scores[k_pos] = score * scale;
                }
                
                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = scores.iter().map(|s| (s - max_score).exp()).sum();
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp() / exp_sum;
                }
                
                // Weighted sum of values
                for d in 0..head_dim {
                    let mut val = 0.0f32;
                    for k_pos in 0..kv_len {
                        val += scores[k_pos] * cached_v[(k_pos * num_kv_heads + kv_head) * head_dim + d];
                    }
                    attn_output[(q_pos * num_heads + h) * head_dim + d] = val;
                }
            }
        }
        
        // Output projection
        self.linear(&attn_output, &layer.o_proj)
    }
    
    fn feed_forward(&self, layer_idx: usize, input: &[f32]) -> Result<Vec<f32>> {
        let layer = &self.weights.layers[layer_idx];
        
        // Gate and up projections
        let gate = self.linear(input, &layer.gate_proj)?;
        let up = self.linear(input, &layer.up_proj)?;
        
        // SwiGLU activation
        let mut hidden: Vec<f32> = gate.iter()
            .zip(up.iter())
            .map(|(g, u)| {
                let silu = g / (1.0 + (-g).exp());
                silu * u
            })
            .collect();
        
        // Apply 7D constraint to intermediate
        if self.config.manifold_enabled {
            for (i, val) in hidden.iter_mut().enumerate() {
                if i % self.config.intermediate_size < 7 {
                    *val *= PHI_INV as f32;
                }
            }
        }
        
        // Down projection
        self.linear(&hidden, &layer.down_proj)
    }
    
    fn rmsnorm(&self, input: &[f32], weight: &[f32]) -> Result<Vec<f32>> {
        let hidden_size = self.config.hidden_size;
        let eps = self.config.rms_norm_eps as f32;
        
        let mut output = Vec::with_capacity(input.len());
        
        for chunk in input.chunks(hidden_size) {
            let sum_sq: f32 = chunk.iter().map(|x| x * x).sum();
            let rms = (sum_sq / hidden_size as f32 + eps).sqrt();
            let inv_rms = 1.0 / rms;
            
            for (i, &val) in chunk.iter().enumerate() {
                output.push(val * inv_rms * weight[i]);
            }
        }
        
        Ok(output)
    }
    
    fn linear(&self, input: &[f32], weight: &LinearWeight) -> Result<Vec<f32>> {
        let in_features = weight.in_features;
        let out_features = weight.out_features;
        let batch = input.len() / in_features;
        
        let mut output = vec![0.0f32; batch * out_features];
        
        // Simple matmul (would use optimized BLAS/CUDA in production)
        for b in 0..batch {
            for o in 0..out_features {
                let mut sum = 0.0f32;
                for i in 0..in_features {
                    sum += input[b * in_features + i] * weight.data[o * in_features + i];
                }
                output[b * out_features + o] = sum;
            }
        }
        
        Ok(output)
    }
    
    fn apply_rope(&self, input: &[f32], pos: usize, head_dim: usize) -> Result<Vec<f32>> {
        let mut output = input.to_vec();
        let theta = self.config.rope_theta as f32;
        
        // Apply rotary embedding to pairs
        for (i, pair) in output.chunks_mut(2).enumerate() {
            if pair.len() < 2 { break; }
            
            let dim_idx = (i % (head_dim / 2)) as f32;
            let freq = 1.0 / theta.powf(dim_idx / (head_dim as f32 / 2.0));
            let angle = pos as f32 * freq;
            
            // 7D modulation
            let phi_mod = if i < 7 && self.config.manifold_enabled {
                (PHI_BASIS[i] / PHI_BASIS[6]) as f32
            } else {
                1.0
            };
            
            let cos_val = angle.cos() * phi_mod;
            let sin_val = angle.sin() * phi_mod;
            
            let x0 = pair[0];
            let x1 = pair[1];
            pair[0] = x0 * cos_val - x1 * sin_val;
            pair[1] = x0 * sin_val + x1 * cos_val;
        }
        
        Ok(output)
    }
    
    fn lm_head(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        self.linear(hidden, &self.weights.lm_head)
    }
    
    /// Get model config
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
}

// ============================================================================
// WEIGHTS
// ============================================================================

pub struct ModelWeights {
    pub embed_tokens: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub norm: Vec<f32>,
    pub lm_head: LinearWeight,
}

pub struct LayerWeights {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_proj: LinearWeight,
    pub k_proj: LinearWeight,
    pub v_proj: LinearWeight,
    pub o_proj: LinearWeight,
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

pub struct LinearWeight {
    pub data: Vec<f32>,
    pub in_features: usize,
    pub out_features: usize,
}

impl ModelWeights {
    fn load_gguf(path: &Path, config: &ModelConfig) -> Result<Self> {
        // In production, this would memory-map the GGUF and load tensors
        // For now, create placeholder weights
        
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let kv_dim = config.kv_dim();
        
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LayerWeights {
                input_layernorm: vec![1.0; hidden_size],
                post_attention_layernorm: vec![1.0; hidden_size],
                q_proj: LinearWeight {
                    data: vec![0.01; hidden_size * hidden_size],
                    in_features: hidden_size,
                    out_features: hidden_size,
                },
                k_proj: LinearWeight {
                    data: vec![0.01; hidden_size * kv_dim],
                    in_features: hidden_size,
                    out_features: kv_dim,
                },
                v_proj: LinearWeight {
                    data: vec![0.01; hidden_size * kv_dim],
                    in_features: hidden_size,
                    out_features: kv_dim,
                },
                o_proj: LinearWeight {
                    data: vec![0.01; hidden_size * hidden_size],
                    in_features: hidden_size,
                    out_features: hidden_size,
                },
                gate_proj: LinearWeight {
                    data: vec![0.01; hidden_size * intermediate_size],
                    in_features: hidden_size,
                    out_features: intermediate_size,
                },
                up_proj: LinearWeight {
                    data: vec![0.01; hidden_size * intermediate_size],
                    in_features: hidden_size,
                    out_features: intermediate_size,
                },
                down_proj: LinearWeight {
                    data: vec![0.01; intermediate_size * hidden_size],
                    in_features: intermediate_size,
                    out_features: hidden_size,
                },
            });
        }
        
        Ok(Self {
            embed_tokens: vec![0.01; vocab_size * hidden_size],
            layers,
            norm: vec![1.0; hidden_size],
            lm_head: LinearWeight {
                data: vec![0.01; hidden_size * vocab_size],
                in_features: hidden_size,
                out_features: vocab_size,
            },
        })
    }
}

// ============================================================================
// KV CACHE
// ============================================================================

pub struct KVCache {
    k_cache: Vec<Vec<f32>>,  // [layer][pos * kv_heads * head_dim]
    v_cache: Vec<Vec<f32>>,
    position: usize,
    max_seq_len: usize,
    num_layers: usize,
    kv_dim: usize,
}

impl KVCache {
    pub fn new(config: &ModelConfig) -> Self {
        let kv_dim = config.kv_dim();
        Self {
            k_cache: vec![Vec::new(); config.num_layers],
            v_cache: vec![Vec::new(); config.num_layers],
            position: 0,
            max_seq_len: config.max_position_embeddings,
            num_layers: config.num_layers,
            kv_dim,
        }
    }
    
    pub fn current_position(&self) -> usize {
        self.position
    }
    
    pub fn update(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        self.k_cache[layer].extend_from_slice(k);
        self.v_cache[layer].extend_from_slice(v);
    }
    
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
        (&self.k_cache[layer], &self.v_cache[layer])
    }
    
    pub fn advance(&mut self, tokens: usize) {
        self.position += tokens;
    }
    
    pub fn clear(&mut self) {
        for layer in &mut self.k_cache {
            layer.clear();
        }
        for layer in &mut self.v_cache {
            layer.clear();
        }
        self.position = 0;
    }
}

// ============================================================================
// SAMPLER
// ============================================================================

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

pub struct Sampler {
    rng: rand::rngs::ThreadRng,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }
}

impl Sampler {
    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams) -> Result<u32> {
        use rand::Rng;
        
        let vocab_size = logits.len();
        
        // Apply temperature
        let mut probs: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .map(|(i, &l)| (i, l / params.temperature))
            .collect();
        
        // Sort for top-k/top-p
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Top-k filtering
        if params.top_k > 0 && params.top_k < vocab_size {
            probs.truncate(params.top_k);
        }
        
        // Softmax
        let max_logit = probs.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = probs.iter().map(|(_, l)| (l - max_logit).exp()).sum();
        
        for (_, l) in probs.iter_mut() {
            *l = (*l - max_logit).exp() / exp_sum;
        }
        
        // Top-p (nucleus) filtering
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
            let sum: f32 = probs.iter().map(|(_, p)| p).sum();
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
        
        // Sample
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;
        for (idx, prob) in probs {
            cumsum += prob;
            if r < cumsum {
                return Ok(idx as u32);
            }
        }
        
        Ok(probs.last().map(|(i, _)| *i as u32).unwrap_or(0))
    }
}
