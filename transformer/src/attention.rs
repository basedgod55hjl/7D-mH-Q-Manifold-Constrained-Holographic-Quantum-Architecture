use crate::rope::RotaryEmbedding;
use crate::Crystal7DConfig;
use crate::PHI_BASIS;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Result, Tensor};

pub struct Crystal7DAttentionBlock {
    q: QMatMul,
    k: QMatMul,
    v: QMatMul,
    o: QMatMul,
    rope: RotaryEmbedding,
    phi_bias: Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    scale: f64,
}

impl Crystal7DAttentionBlock {
    pub fn new(
        q: QMatMul,
        k: QMatMul,
        v: QMatMul,
        o: QMatMul,
        config: &Crystal7DConfig,
        rope: RotaryEmbedding,
    ) -> Result<Self> {
        let n_head = config.num_attention_heads;
        let n_kv_head = config.num_kv_heads;
        let head_dim = config.head_dim();
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Pre-compute Φ-scaling bias (1.0 for now, placeholder for 7D logic)
        // In a real 7D implementation, this would be a per-feature bias.
        // We use a dummy scalar for compatibility
        let phi_bias = Tensor::new(&[1.0f32], &Device::Cpu)?;

        Ok(Self {
            q,
            k,
            v,
            o,
            rope,
            phi_bias,
            n_head,
            n_kv_head,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        // x: [1, 1, hidden_size] (usually 1 token inference)
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        let q = self.q.forward(x)?; // [b, seq, n_head * h_dim]
        let k = self.k.forward(x)?; // [b, seq, n_kv * h_dim]
        let v = self.v.forward(x)?; // [b, seq, n_kv * h_dim]

        // Reshape
        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        // We need positions. For inference, we usually pass a single position index.
        // If we process a chunk, we need a list.
        // Here we assume simple inference (1 token) or contiguous chunk starting at index_pos.
        // Apply RoPE
        // We assume contiguous sequence starting at index_pos
        let q = self.rope.forward(&q, index_pos)?;
        let k = self.rope.forward(&k, index_pos)?;

        // Repeat KV for GQA
        let k = utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        // Attention
        // (b, n_head, seq, h_dim) @ (b, n_head, h_dim, seq) -> (b, n_head, seq, seq)
        let att = (q.matmul(&k.t()?)? * self.scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;
        let y = att.matmul(&v)?; // (b, n_head, seq, h_dim)

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, hidden_size))?;
        let output = self.o.forward(&y)?;

        Ok(output)
    }
}

// Helper module for utils if needed, or inline
mod utils {
    use candle_core::{Result, Tensor, D};
    pub fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b, n_kv_head, seq, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b, n_kv_head, n_rep, seq, head_dim))?;
            x.reshape((b, n_kv_head * n_rep, seq, head_dim))
        }
    }
}
