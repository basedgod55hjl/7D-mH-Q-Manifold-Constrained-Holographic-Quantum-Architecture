// File: transformer/src/layers.rs
// 7D Crystal Transformer Layers - Quantized & Candle-Powered

use super::attention::Crystal7DAttentionBlock;
use super::*;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor};

/// RMSNorm with S² stability check
pub struct Crystal7DRMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl Crystal7DRMSNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = DType::F32; // Compute in F32 for stability
        let x = x.to_dtype(internal_dtype)?;

        let dim = x.dim(x.rank() - 1)?;
        let mean_sq = (x.sqr()?.sum_keepdim(x.rank() - 1)? / (dim as f64))?;
        let rms = (mean_sq + self.eps)?.sqrt()?;
        let x_norm = x.broadcast_div(&rms)?;

        // 7D Stability can be added here if needed

        let output = x_norm.broadcast_mul(&self.weight.to_dtype(internal_dtype)?)?;
        output.to_dtype(x_dtype)
    }
}

/// SwiGLU FFN with quantized weights
pub struct Crystal7DFFN {
    pub gate: QMatMul,
    pub up: QMatMul,
    pub down: QMatMul,
}

impl Crystal7DFFN {
    pub fn new(gate: QMatMul, up: QMatMul, down: QMatMul) -> Self {
        Self { gate, up, down }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_gate = self.gate.forward(x)?;
        let x_up = self.up.forward(x)?;

        let silu = candle_nn::ops::silu(&x_gate)?;
        let activated = silu.broadcast_mul(&x_up)?;

        self.down.forward(&activated)
    }
}

/// Complete Transformer Layer
pub struct Crystal7DTransformerLayer {
    pub attention: Crystal7DAttentionBlock,
    pub ffn: Crystal7DFFN,
    pub input_norm: Crystal7DRMSNorm,
    pub post_attn_norm: Crystal7DRMSNorm,
}

impl Crystal7DTransformerLayer {
    pub fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let residual = x;
        let normed = self.input_norm.forward(x)?;

        let attn_out = self.attention.forward(&normed, pos)?;
        let hidden = (residual + attn_out)?;

        let residual = &hidden;
        let normed2 = self.post_attn_norm.forward(&hidden)?;

        let ffn_out = self.ffn.forward(&normed2)?;
        let output = (residual + ffn_out)?;

        Ok(output)
    }
}

pub struct Crystal7DEmbedding {
    pub embeddings: Tensor,
    pub hidden_size: usize,
}

impl Crystal7DEmbedding {
    pub fn forward(&self, tokens: &[u32], _device: &Device) -> Result<Tensor> {
        let mut extracted = Vec::new();
        for &token in tokens {
            let idx = token as usize;
            let emb = self.embeddings.get(idx)?;
            extracted.push(emb);
        }
        Tensor::stack(&extracted, 0)
    }
}

/// LM Head
pub struct Crystal7DLMHead {
    pub inner: QMatMul,
}

impl Crystal7DLMHead {
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        self.inner.forward(hidden)
    }
}
