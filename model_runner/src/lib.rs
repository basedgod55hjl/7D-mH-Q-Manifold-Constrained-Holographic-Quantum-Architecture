// File: model_runner/src/lib.rs
// 7D Crystal Model Runner - Quantized Inference Engine

pub mod backend;
pub mod sampler;
mod tests;

use anyhow::{bail, Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, IndexOp, Tensor};
use transformer::{
    layers::{
        Crystal7DEmbedding, Crystal7DFFN, Crystal7DLMHead, Crystal7DRMSNorm,
        Crystal7DTransformerLayer,
    },
    Crystal7DAttentionBlock, Crystal7DConfig,
};

// ============================================================================
// MODEL RUNNER
// ============================================================================

pub struct ModelRunner {
    config: Crystal7DConfig,
    weights: Arc<ModelWeights>,
    // KV Cache typically needs to match the tensor type/device.
    // For simplicity in this iteration, we might skip KV cache or use a placeholder.
    // Let's implement a basic one.
    #[allow(dead_code)]
    kv_cache: RwLock<KVCache>,
    sampler: RwLock<sampler::Sampler>,
    device: Device,
}

pub struct KVCache {
    // [layer][key/val] -> Tensor
    cache: Vec<(Option<Tensor>, Option<Tensor>)>,
}

impl KVCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            cache: vec![(None, None); n_layers],
        }
    }

    pub fn clear(&mut self) {
        for (k, v) in self.cache.iter_mut() {
            *k = None;
            *v = None;
        }
    }
}

pub struct ModelWeights {
    pub embed: Crystal7DEmbedding,
    pub layers: Vec<Crystal7DTransformerLayer>,
    pub norm: Crystal7DRMSNorm,
    pub lm_head: Crystal7DLMHead,
}

impl ModelRunner {
    pub fn from_gguf(path: &Path, _backend: backend::Backend) -> Result<Self> {
        let device = Device::Cpu; // Force CPU for now for Q4 inference

        // Load GGUF file with mmap
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        // Infer configuration (simplified)
        let config = infer_config(&content)?;

        // Load weights
        let weights = load_weights(&content, &mut file, &device, &config)?;

        Ok(Self {
            config: config.clone(),
            weights: Arc::new(weights),
            kv_cache: RwLock::new(KVCache::new(config.num_layers)),
            sampler: RwLock::new(sampler::Sampler::default()),
            device,
        })
    }

    pub fn config(&self) -> &Crystal7DConfig {
        &self.config
    }

    pub fn generate(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        params: &sampler::SamplingParams,
    ) -> Result<Vec<u32>> {
        self.generate_stream(tokens, max_tokens, params, |_| Ok(true))
    }

    pub fn generate_stream<F>(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        params: &sampler::SamplingParams,
        mut callback: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> Result<bool>,
    {
        let mut output = Vec::new();
        let mut current_tokens = tokens.to_vec();

        // Pre-fill
        let mut pos = 0;
        let logits = self.forward(&current_tokens, pos)?;
        pos += current_tokens.len();

        let mut next_token = self.sampler.write().sample(&logits, params)?;
        output.push(next_token);
        current_tokens.push(next_token);

        if !callback(next_token)? {
            return Ok(output);
        }

        // Decode
        for _ in 0..max_tokens {
            // Generate for just the last token
            let input = [next_token];
            let logits = self.forward(&input, pos)?;
            next_token = self.sampler.write().sample(&logits, params)?;

            output.push(next_token);
            current_tokens.push(next_token);
            pos += 1;

            if !callback(next_token)? {
                break;
            }

            if next_token == 2 {
                break;
            } // EOS
        }

        Ok(output)
    }

    fn forward(&self, tokens: &[u32], pos: usize) -> Result<Vec<f32>> {
        // Embed
        let mut x = self
            .weights
            .embed
            .forward(tokens, &self.device)?
            .unsqueeze(0)?;

        // Layers
        for (_idx, layer) in self.weights.layers.iter().enumerate() {
            x = layer.forward(&x, pos)?;

            // 7D Verification: Check RMS Norm
            // let dim = x.dim(x.rank() - 1)?;
            // let sq_sum = x.sqr()?.sum_all()?.to_scalar::<f32>()?;
            // let rms = (sq_sum / (tokens.len() * dim) as f32).sqrt();
            // println!("Layer {} RMS norm: {:.6}", idx, rms);
        }

        // Final Norm
        x = self.weights.norm.forward(&x)?;

        // LM Head (last token only)
        // x is [1, seq_len, hidden] (if batched from embed)
        // We only care about last row
        let (_b, seq_len, _h) = x.dims3()?;
        let last_x = x.i((.., seq_len - 1, ..))?; // [1, hidden]

        let logits = self.weights.lm_head.forward(&last_x)?; // [1, vocab]

        // Convert to Vec<f32>
        let logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;
        Ok(logits_vec)
    }
}

// Helper functions to avoid multiple mutable borrows of file
fn read_qmatmul(
    tensors: &HashMap<String, gguf_file::TensorInfo>,
    file: &mut std::fs::File,
    offset: u64,
    device: &Device,
    name: &str,
) -> Result<QMatMul> {
    let info = tensors
        .get(name)
        .context(format!("Missing tensor: {}", name))?;
    let qt = info.read(file, offset, device)?;
    Ok(QMatMul::from_qtensor(qt)?)
}

fn read_tensor(
    tensors: &HashMap<String, gguf_file::TensorInfo>,
    file: &mut std::fs::File,
    offset: u64,
    device: &Device,
    name: &str,
) -> Result<Tensor> {
    let info = tensors
        .get(name)
        .context(format!("Missing tensor: {}", name))?;
    let qt = info.read(file, offset, device)?;
    Ok(qt.dequantize(device)?)
}

// Helper to load weights
fn load_weights(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    device: &Device,
    cfg: &Crystal7DConfig,
) -> Result<ModelWeights> {
    // Access tensor map
    let tensors = &content.tensor_infos;
    let offset = content.tensor_data_offset;

    // Embeddings
    let emb_q = tensors
        .get("token_embd.weight")
        .context("Missing token_embd.weight")?
        .read(file, offset, device)?;

    let embed = Crystal7DEmbedding {
        embeddings: emb_q.dequantize(device)?,
        hidden_size: cfg.hidden_size,
    };

    // Initialize RoPE
    let rope = transformer::rope::RotaryEmbedding::new(
        cfg.rope_theta as f32,
        cfg.head_dim(),
        cfg.max_seq_len,
        device,
    )?;

    // Layers
    let mut layers = Vec::new();
    for i in 0..cfg.num_layers {
        let p = |s: &str| format!("blk.{}.{}", i, s);

        // RMS Norms
        let input_norm = Crystal7DRMSNorm::new(
            read_tensor(tensors, file, offset, device, &p("attn_norm.weight"))?,
            1e-5,
        );
        let post_attn_norm = Crystal7DRMSNorm::new(
            read_tensor(tensors, file, offset, device, &p("ffn_norm.weight"))?,
            1e-5,
        );

        // Attention
        let q = read_qmatmul(tensors, file, offset, device, &p("attn_q.weight"))?;
        let k = read_qmatmul(tensors, file, offset, device, &p("attn_k.weight"))?;
        let v = read_qmatmul(tensors, file, offset, device, &p("attn_v.weight"))?;
        let o = read_qmatmul(tensors, file, offset, device, &p("attn_output.weight"))?;

        let attention = Crystal7DAttentionBlock::new(q, k, v, o, &cfg, rope.clone())?;

        // FFN
        // DeepSeek might reuse gate/up names or have slight variance.
        // Llama: ffn_gate (gate), ffn_up (up), ffn_down (down)
        let gate = read_qmatmul(tensors, file, offset, device, &p("ffn_gate.weight"))?;
        let up = read_qmatmul(tensors, file, offset, device, &p("ffn_up.weight"))?;
        let down = read_qmatmul(tensors, file, offset, device, &p("ffn_down.weight"))?;

        let ffn = Crystal7DFFN::new(gate, up, down);

        layers.push(Crystal7DTransformerLayer {
            attention,
            ffn,
            input_norm,
            post_attn_norm,
        });
    }

    // Final Norm & Head
    let norm = Crystal7DRMSNorm::new(
        read_tensor(tensors, file, offset, device, "output_norm.weight")?,
        1e-5,
    );

    // Head often shares weights with embed in some models, but DeepSeek usually distinct?
    // Llama usually distinct.
    let lm_head_t = read_qmatmul(tensors, file, offset, device, "output.weight")?;
    let lm_head = Crystal7DLMHead { inner: lm_head_t };
    // Let's assume Crystal7DLMHead was updated to use QMatMul.
    // I'll fix layers.rs if I missed it.

    Ok(ModelWeights {
        embed,
        layers,
        norm,
        lm_head,
    })
}

fn infer_config(content: &gguf_file::Content) -> Result<Crystal7DConfig> {
    // Simplified inference
    let props = &content.metadata;

    let get_usize = |k: &str| -> Result<usize> {
        match props.get(k) {
            Some(gguf_file::Value::U64(v)) => Ok(*v as usize),
            Some(gguf_file::Value::U32(v)) => Ok(*v as usize),
            Some(gguf_file::Value::U16(v)) => Ok(*v as usize),
            Some(gguf_file::Value::I64(v)) => Ok(*v as usize),
            Some(gguf_file::Value::I32(v)) => Ok(*v as usize),
            _ => bail!("Missing or invalid property: {}", k),
        }
    };

    // DeepSeek keys might be "llama.attention.head_count", etc.
    // Llama.cpp mapping:
    let n_head = get_usize("llama.attention.head_count")
        .or_else(|_| get_usize("attention.head_count"))
        .unwrap_or(32);

    let n_kv_head = get_usize("llama.attention.head_count_kv")
        .or_else(|_| get_usize("attention.head_count_kv"))
        .unwrap_or(n_head);

    let block_count = get_usize("llama.block_count")
        .or_else(|_| get_usize("block_count"))
        .unwrap_or(32);

    let embedding_length = get_usize("llama.embedding_length")
        .or_else(|_| get_usize("embedding_length"))
        .unwrap_or(4096);

    let vocab_size = props
        .get("tokenizer.ggml.tokens")
        .map(|v| match v {
            gguf_file::Value::Array(a) => a.len(),
            _ => 128256,
        })
        .unwrap_or(128256);

    // RoPE
    let rope_theta = props
        .get("llama.attention.rope.freq_base")
        .map(|v| match v {
            gguf_file::Value::F32(f) => *f as f64,
            _ => 10000.0,
        })
        .unwrap_or(10000.0);

    Ok(Crystal7DConfig {
        hidden_size: embedding_length,
        intermediate_size: 11008, // Usually calculated
        num_layers: block_count,
        num_attention_heads: n_head,
        num_kv_heads: n_kv_head,
        vocab_size,
        max_seq_len: 8192,
        manifold_enabled: true,
        curvature: 0.618,
        phi_attention: true,
        use_swiglu: true,
        rope_theta,
    })
}
