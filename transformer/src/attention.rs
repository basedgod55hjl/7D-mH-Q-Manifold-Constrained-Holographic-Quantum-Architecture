use crate::rope::RotaryEmbedding;
use crate::Crystal7DConfig;
use candle_core::quantized::QMatMul;
use candle_core::{Device, Module, Result, Tensor};

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
        eprintln!("=== ATTENTION FORWARD START ===");
        eprintln!("Input x shape: {:?}", x.shape());
        eprintln!(
            "Config: n_head={}, n_kv_head={}, head_dim={}",
            self.n_head, self.n_kv_head, self.head_dim
        );

        // x: [batch, seq_len, hidden_size]
        let (b_sz, seq_len, _hidden_size) = x.dims3()?;

        let q = self.q.forward(x)?; // [b, seq, n_head * head_dim]
        eprintln!("After Q matmul: {:?}", q.shape());

        let k = self.k.forward(x)?; // [b, seq, n_kv_head * head_dim]
        eprintln!("After K matmul: {:?}", k.shape());

        let v = self.v.forward(x)?; // [b, seq, n_kv_head * head_dim]
        eprintln!("After V matmul: {:?}", v.shape());

        // Reshape to [batch, seq, n_heads, head_dim]
        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;

        eprintln!(
            "DEBUG: After reshape - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Transpose to [batch, n_heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        eprintln!(
            "DEBUG: After transpose - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Apply RoPE to Q and K
        let q = self.rope.forward(&q, index_pos)?;
        let k = self.rope.forward(&k, index_pos)?;

        // Repeat KV for GQA: [b, n_kv_head, seq, head_dim] -> [b, n_head, seq, head_dim]
        let n_rep = self.n_head / self.n_kv_head;
        let k = utils::repeat_kv(k, n_rep)?;
        let v = utils::repeat_kv(v, n_rep)?;

        eprintln!(
            "DEBUG: After repeat_kv - K: {:?}, V: {:?}",
            k.shape(),
            v.shape()
        );

        // Attention computation
        // Q: [b, n_head, seq, head_dim]
        // K^T: [b, n_head, head_dim, seq]
        // Result: [b, n_head, seq, seq]
        let k_t = k.transpose(2, 3)?; // [b, n_head, head_dim, seq]
        let att = q.matmul(&k_t)?;
        let att = (att * self.scale)?;
        let att = candle_nn::ops::softmax(&att, candle_core::D::Minus1)?;

        // att @ V: [b, n_head, seq, seq] @ [b, n_head, seq, head_dim] -> [b, n_head, seq, head_dim]
        let y = att.matmul(&v)?;

        // Transpose back and reshape: [b, n_head, seq, head_dim] -> [b, seq, n_head, head_dim] -> [b, seq, hidden]
        let y = y.transpose(1, 2)?.contiguous()?;
        let y = y.reshape((b_sz, seq_len, self.n_head * self.head_dim))?;

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
