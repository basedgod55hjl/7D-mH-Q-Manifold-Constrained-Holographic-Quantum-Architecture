use candle_core::{DType, Device, Result, Tensor, D};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        base: f32,
        head_dim: usize,
        max_position_embeddings: usize,
        device: &Device,
    ) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?; // [head_dim / 2]
        let idx_theta = Tensor::arange(0, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.dim(0)?))?)?; // [seq_len, head_dim / 2]

        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;

        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?; // [seq_len, head_dim]
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?; // [seq_len, head_dim]

        Ok(Self { cos, sin, head_dim })
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, _n_head, seq_len, _head_dim) = x.dims4()?;

        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;

        // Unsqueeze to broadcast over batch and heads: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        let x1 = x.narrow(D::Minus1, 0, self.head_dim / 2)?;
        let x2 = x.narrow(D::Minus1, self.head_dim / 2, self.head_dim / 2)?;

        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;

        // x * cos + rotate_x * sin
        let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
        Ok(rope)
    }
}
