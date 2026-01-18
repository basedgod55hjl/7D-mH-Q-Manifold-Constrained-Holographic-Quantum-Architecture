// 7D Crystal LLM Builder - Quantization Module
// Implements phi-aware scaling and manifold-preserving quantization

use crate::QuantConfig;

pub struct PhiScaler {
    pub phi: f64,
}

impl PhiScaler {
    pub fn new() -> Self {
        Self { phi: crate::PHI }
    }

    pub fn scale(&self, val: f32) -> f32 {
        (val as f64 * self.phi) as f32
    }

    pub fn scale_tensor(&self, data: &[f32]) -> Vec<f32> {
        data.iter().map(|&x| self.scale(x)).collect()
    }
}

pub struct BlockQuantizer {
    pub config: QuantConfig,
}

impl BlockQuantizer {
    pub fn new(config: QuantConfig) -> Self {
        Self { config }
    }

    pub fn quantize_f32_to_q4_phi(&self, data: &[f32]) -> Vec<u8> {
        let scaler = PhiScaler::new();
        let mut output = Vec::with_capacity(data.len() / 2);

        for chunk in data.chunks_exact(2) {
            let s1 = scaler.scale(chunk[0]);
            let s2 = scaler.scale(chunk[1]);

            let q1 = ((s1.clamp(-1.0, 1.0) + 1.0) * 7.5) as u8;
            let q2 = ((s2.clamp(-1.0, 1.0) + 1.0) * 7.5) as u8;
            output.push((q1 << 4) | (q2 & 0x0F));
        }
        output
    }

    /// Quantizes f32 to 2-bit using Φ-manifold projection (4 levels)
    pub fn quantize_f32_to_q2_phi(&self, data: &[f32]) -> Vec<u8> {
        let scaler = PhiScaler::new();
        let mut output = Vec::with_capacity(data.len() / 4);

        for chunk in data.chunks_exact(4) {
            let mut byte = 0u8;
            for (i, &val) in chunk.iter().enumerate() {
                let s = scaler.scale(val);
                let q = if s < -0.5 {
                    0
                } else if s < 0.0 {
                    1
                } else if s < 0.5 {
                    2
                } else {
                    3
                };
                byte |= q << (i * 2);
            }
            output.push(byte);
        }
        output
    }
}
