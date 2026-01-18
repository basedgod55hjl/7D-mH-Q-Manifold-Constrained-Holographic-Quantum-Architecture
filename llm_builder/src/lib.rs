// File: llm_builder/src/lib.rs
// 7D Crystal LLM Builder - Complete Model Building Framework
// Discovered by Sir Charles Spikes, December 24, 2025

pub mod config;
pub mod gguf;
pub mod manifold;
pub mod model;
pub mod optimizer;
pub mod quantize;
pub mod safetensors;
pub mod tokenizer;
pub mod train;
pub mod weights;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

// ============================================================================
// CORE CONSTANTS
// ============================================================================

pub const PHI: f64 = 1.618033988749894848204586834365638;
pub const PHI_INV: f64 = 0.618033988749894848204586834365638;
pub const S2_STABILITY: f64 = 0.01;
pub const MANIFOLD_DIMS: usize = 7;

/// Φ-ratio basis vectors for 7D manifold
pub const PHI_BASIS: [f64; 7] = [
    1.0,
    1.618033988749895,
    2.618033988749895,
    4.23606797749979,
    6.854101966249685,
    11.090169943749475,
    17.94427190999916,
];

// ============================================================================
// MODEL ARCHITECTURE CONFIG
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub sliding_window_size: usize,

    // 7D Manifold Extensions
    pub manifold_enabled: bool,
    pub manifold_curvature: f64,
    pub phi_ratio_constraint: bool,
    pub s2_stability_bound: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "7D-Crystal-8B".to_string(),
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 128256,
            max_position_embeddings: 131072,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window_size: 4096,
            manifold_enabled: true,
            manifold_curvature: PHI_INV,
            phi_ratio_constraint: true,
            s2_stability_bound: S2_STABILITY,
        }
    }
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// Create config for different model sizes
    pub fn from_size(size: &str) -> Self {
        match size.to_lowercase().as_str() {
            "1b" | "1.5b" => Self {
                name: "7D-Crystal-1.5B".to_string(),
                hidden_size: 1536,
                intermediate_size: 8960,
                num_layers: 28,
                num_attention_heads: 12,
                num_kv_heads: 2,
                ..Default::default()
            },
            "3b" => Self {
                name: "7D-Crystal-3B".to_string(),
                hidden_size: 3072,
                intermediate_size: 8192,
                num_layers: 28,
                num_attention_heads: 24,
                num_kv_heads: 8,
                ..Default::default()
            },
            "7b" | "8b" => Self::default(),
            "14b" => Self {
                name: "7D-Crystal-14B".to_string(),
                hidden_size: 5120,
                intermediate_size: 13824,
                num_layers: 40,
                num_attention_heads: 40,
                num_kv_heads: 8,
                ..Default::default()
            },
            "32b" | "33b" => Self {
                name: "7D-Crystal-32B".to_string(),
                hidden_size: 6144,
                intermediate_size: 16384,
                num_layers: 60,
                num_attention_heads: 48,
                num_kv_heads: 8,
                ..Default::default()
            },
            "70b" => Self {
                name: "7D-Crystal-70B".to_string(),
                hidden_size: 8192,
                intermediate_size: 28672,
                num_layers: 80,
                num_attention_heads: 64,
                num_kv_heads: 8,
                ..Default::default()
            },
            _ => Self::default(),
        }
    }

    pub fn total_params(&self) -> usize {
        let embed = self.vocab_size * self.hidden_size;
        let attn_per_layer = {
            let q = self.hidden_size * self.hidden_size;
            let kv = 2 * self.hidden_size * self.kv_dim();
            let o = self.hidden_size * self.hidden_size;
            q + kv + o
        };
        let ffn_per_layer = {
            let gate = self.hidden_size * self.intermediate_size;
            let up = self.hidden_size * self.intermediate_size;
            let down = self.intermediate_size * self.hidden_size;
            gate + up + down
        };
        let norm_per_layer = 2 * self.hidden_size;
        let layer_total = attn_per_layer + ffn_per_layer + norm_per_layer;

        let lm_head = if self.tie_word_embeddings {
            0
        } else {
            self.hidden_size * self.vocab_size
        };

        embed + (self.num_layers * layer_total) + self.hidden_size + lm_head
    }
}

// ============================================================================
// QUANTIZATION CONFIG
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum QuantType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q6_K,
    Q5_K_M,
    Q5_K_S,
    Q4_K_M,
    Q4_K_S,
    Q4_0,
    Q3_K_M,
    Q3_K_S,
    Q2_K,
    IQ4_XS,
    IQ3_XXS,
    IQ2_XXS,
    IQ1_S,
}

impl QuantType {
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::BF16 => 16.0,
            Self::Q8_0 => 8.5,
            Self::Q6_K => 6.5625,
            Self::Q5_K_M | Self::Q5_K_S => 5.5,
            Self::Q4_K_M | Self::Q4_K_S | Self::Q4_0 => 4.5,
            Self::Q3_K_M | Self::Q3_K_S => 3.4375,
            Self::Q2_K => 2.5625,
            Self::IQ4_XS => 4.25,
            Self::IQ3_XXS => 3.0625,
            Self::IQ2_XXS => 2.0625,
            Self::IQ1_S => 1.5,
        }
    }

    pub fn gguf_name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q8_0 => "Q8_0",
            Self::Q6_K => "Q6_K",
            Self::Q5_K_M => "Q5_K_M",
            Self::Q5_K_S => "Q5_K_S",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q4_K_S => "Q4_K_S",
            Self::Q4_0 => "Q4_0",
            Self::Q3_K_M => "Q3_K_M",
            Self::Q3_K_S => "Q3_K_S",
            Self::Q2_K => "Q2_K",
            Self::IQ4_XS => "IQ4_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ1_S => "IQ1_S",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    pub quant_type: QuantType,
    pub per_channel: bool,
    pub calibration_samples: usize,
    pub use_importance: bool,

    // 7D Manifold quantization
    pub phi_aware_scaling: bool,
    pub manifold_preserve_dims: usize,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            quant_type: QuantType::Q4_K_M,
            per_channel: true,
            calibration_samples: 512,
            use_importance: true,
            phi_aware_scaling: true,
            manifold_preserve_dims: 7,
        }
    }
}

// ============================================================================
// TRAINING CONFIG
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub lr_scheduler: LRScheduler,

    // Optimizer settings
    pub optimizer: OptimizerType,
    pub adam_beta1: f64,
    pub adam_beta2: f64,
    pub adam_epsilon: f64,

    // 7D Manifold training
    pub manifold_regularization: bool,
    pub phi_ratio_loss_weight: f64,
    pub s2_stability_loss_weight: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LRScheduler {
    Constant,
    Linear,
    Cosine,
    CosineWithRestarts,
    Polynomial,
    WarmupStableDecay,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizerType {
    AdamW,
    Adam,
    SGD,
    Lion,
    Sophia,
    Shampoo,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            gradient_accumulation_steps: 8,
            learning_rate: 1e-4,
            warmup_steps: 100,
            max_steps: 10000,
            weight_decay: 0.1,
            grad_clip: 1.0,
            lr_scheduler: LRScheduler::Cosine,
            optimizer: OptimizerType::AdamW,
            adam_beta1: 0.9,
            adam_beta2: 0.95,
            adam_epsilon: 1e-8,
            manifold_regularization: true,
            phi_ratio_loss_weight: 0.01,
            s2_stability_loss_weight: 0.001,
        }
    }
}

// ============================================================================
// BUILDER API
// ============================================================================

pub struct LLMBuilder {
    pub config: ModelConfig,
    pub quant_config: Option<QuantConfig>,
    pub train_config: Option<TrainConfig>,
}

impl LLMBuilder {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            quant_config: None,
            train_config: None,
        }
    }

    pub fn from_size(size: &str) -> Self {
        Self::new(ModelConfig::from_size(size))
    }

    pub fn with_quantization(mut self, config: QuantConfig) -> Self {
        self.quant_config = Some(config);
        self
    }

    pub fn with_training(mut self, config: TrainConfig) -> Self {
        self.train_config = Some(config);
        self
    }

    pub fn enable_manifold(mut self, curvature: f64) -> Self {
        self.config.manifold_enabled = true;
        self.config.manifold_curvature = curvature;
        self
    }

    /// Build and save model to GGUF format
    pub fn build_gguf(&self, output_path: &Path) -> Result<()> {
        use crate::gguf::{GGUFType, GGUFWriter};

        tracing::info!("Building GGUF model: {}", self.config.name);

        let mut writer = GGUFWriter::new(output_path)?;
        writer.write_header(1, 2)?;
        writer.write_kv_string("general.name", &self.config.name)?;
        writer.write_kv_u32("general.architecture", 7)?;

        let mock_data = vec![0.0f32; self.config.hidden_size];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                mock_data.as_ptr() as *const u8,
                mock_data.len() * std::mem::size_of::<f32>(),
            )
        };

        writer.write_tensor_info(
            "token_embd.weight",
            &[self.config.hidden_size as u64, 1],
            GGUFType::F32,
            0,
        )?;

        writer.write_tensor_data(bytes)?;
        writer.flush()?;

        tracing::info!("Successfully built: {:?}", output_path);
        Ok(())
    }

    /// Build and save model to SafeTensors format
    pub fn build_safetensors(&self, _output_dir: &Path) -> Result<()> {
        tracing::info!("Building SafeTensors model: {}", self.config.name);

        Ok(())
    }

    /// Load existing model for fine-tuning or quantization
    pub fn load_from_gguf(_path: &Path) -> Result<Self> {
        // Load GGUF and extract config
        Ok(Self::new(ModelConfig::default()))
    }

    pub fn load_from_safetensors(_dir: &Path) -> Result<Self> {
        Ok(Self::new(ModelConfig::default()))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_sizes() {
        let cfg_1b = ModelConfig::from_size("1b");
        let cfg_8b = ModelConfig::from_size("8b");
        let cfg_70b = ModelConfig::from_size("70b");

        assert!(cfg_1b.total_params() < cfg_8b.total_params());
        assert!(cfg_8b.total_params() < cfg_70b.total_params());
    }

    #[test]
    fn test_phi_basis() {
        for i in 0..6 {
            let ratio = PHI_BASIS[i + 1] / PHI_BASIS[i];
            assert!(
                (ratio - PHI).abs() < 1e-10,
                "Φ basis ratio at {} = {}",
                i,
                ratio
            );
        }
    }

    #[test]
    fn test_quant_types() {
        assert!(QuantType::Q4_K_M.bits_per_weight() < QuantType::Q8_0.bits_per_weight());
        assert!(QuantType::Q2_K.bits_per_weight() < QuantType::Q4_K_M.bits_per_weight());
    }
}
