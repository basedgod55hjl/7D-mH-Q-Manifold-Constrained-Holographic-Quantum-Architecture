# 7D Crystal API Reference

## Core Constants

### Rust

```rust
pub const PHI: f64 = 1.618033988749894848204586834365638;
pub const PHI_INV: f64 = 0.618033988749894848204586834365638;
pub const S2_STABILITY: f64 = 0.01;
pub const MANIFOLD_DIMS: usize = 7;
pub const CURVATURE: f64 = -1.0;

pub const PHI_BASIS: [f64; 7] = [
    1.0, 1.618033988749895, 2.618033988749895, 4.23606797749979,
    6.854101966249685, 11.090169943749475, 17.94427190999916,
];
```

### Python

```python
PHI = 1.618033988749894848204586834365638
PHI_INV = 0.618033988749894848204586834365638
S2_STABILITY = 0.01
MANIFOLD_DIMS = 7

PHI_BASIS = np.array([
    1.0, 1.618033988749895, 2.618033988749895, 4.23606797749979,
    6.854101966249685, 11.090169943749475, 17.94427190999916
])
```

### CUDA

```cuda
#define PHI 1.6180339887498949f
#define PHI_INV 0.6180339887498949f
#define S2_STABILITY_BOUND 0.01f
#define MANIFOLD_DIMS 7

__constant__ float PHI_BASIS[7] = {
    1.0f, 1.618f, 2.618f, 4.236f, 6.854f, 11.090f, 17.944f
};
```

---

## ModelConfig

### Struct Definition

```rust
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
    
    // 7D Manifold
    pub manifold_enabled: bool,
    pub manifold_curvature: f64,
    pub phi_ratio_constraint: bool,
    pub s2_stability_bound: f64,
}
```

### Methods

```rust
impl ModelConfig {
    /// Create default 8B config
    fn default() -> Self;
    
    /// Create config by size: "1b", "3b", "7b", "8b", "14b", "32b", "70b"
    fn from_size(size: &str) -> Self;
    
    /// Head dimension (hidden_size / num_attention_heads)
    fn head_dim(&self) -> usize;
    
    /// KV dimension (num_kv_heads * head_dim)
    fn kv_dim(&self) -> usize;
    
    /// Estimate total parameters
    fn total_params(&self) -> usize;
}
```

### Preset Configurations

| Size | hidden | intermediate | layers | heads | kv_heads |
|------|--------|--------------|--------|-------|----------|
| 1.5B | 1536 | 8960 | 28 | 12 | 2 |
| 3B | 3072 | 8192 | 28 | 24 | 8 |
| 8B | 4096 | 14336 | 32 | 32 | 8 |
| 14B | 5120 | 13824 | 40 | 40 | 8 |
| 32B | 6144 | 16384 | 60 | 48 | 8 |
| 70B | 8192 | 28672 | 80 | 64 | 8 |

---

## QuantType

### Enum Definition

```rust
pub enum QuantType {
    F32, F16, BF16,
    Q8_0, Q6_K,
    Q5_K_M, Q5_K_S,
    Q4_K_M, Q4_K_S, Q4_0,
    Q3_K_M, Q3_K_S,
    Q2_K,
    IQ4_XS, IQ3_XXS, IQ2_XXS, IQ1_S,
}
```

### Methods

```rust
impl QuantType {
    /// Bits per weight (including overhead)
    fn bits_per_weight(&self) -> f32;
    
    /// GGUF type name
    fn gguf_name(&self) -> &'static str;
}
```

### Bits Per Weight

| Type | BPW | Description |
|------|-----|-------------|
| F32 | 32.0 | Full precision |
| F16 | 16.0 | Half precision |
| BF16 | 16.0 | Brain float |
| Q8_0 | 8.5 | 8-bit basic |
| Q6_K | 6.56 | 6-bit K-quant |
| Q5_K_M | 5.5 | 5-bit K-quant |
| Q4_K_M | 4.5 | 4-bit K-quant (recommended) |
| Q4_0 | 4.5 | 4-bit basic |
| Q3_K_M | 3.44 | 3-bit K-quant |
| Q2_K | 2.56 | 2-bit K-quant |

---

## QuantConfig

```rust
pub struct QuantConfig {
    pub quant_type: QuantType,
    pub per_channel: bool,
    pub calibration_samples: usize,
    pub use_importance: bool,
    
    // 7D Manifold
    pub phi_aware_scaling: bool,
    pub manifold_preserve_dims: usize,
}
```

---

## TrainConfig

```rust
pub struct TrainConfig {
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub learning_rate: f64,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub weight_decay: f64,
    pub grad_clip: f64,
    pub lr_scheduler: LRScheduler,
    pub optimizer: OptimizerType,
    pub adam_beta1: f64,
    pub adam_beta2: f64,
    pub adam_epsilon: f64,
    
    // 7D Manifold
    pub manifold_regularization: bool,
    pub phi_ratio_loss_weight: f64,
    pub s2_stability_loss_weight: f64,
}

pub enum LRScheduler {
    Constant, Linear, Cosine, CosineWithRestarts, Polynomial, WarmupStableDecay,
}

pub enum OptimizerType {
    AdamW, Adam, SGD, Lion, Sophia, Shampoo,
}
```

---

## LLMBuilder

### Constructor

```rust
impl LLMBuilder {
    /// Create builder with config
    pub fn new(config: ModelConfig) -> Self;
    
    /// Create builder from size string
    pub fn from_size(size: &str) -> Self;
}
```

### Configuration

```rust
impl LLMBuilder {
    /// Add quantization config
    pub fn with_quantization(self, config: QuantConfig) -> Self;
    
    /// Add training config
    pub fn with_training(self, config: TrainConfig) -> Self;
    
    /// Enable 7D manifold with curvature
    pub fn enable_manifold(self, curvature: f64) -> Self;
}
```

### Building

```rust
impl LLMBuilder {
    /// Build and save to GGUF
    pub fn build_gguf(&self, path: &Path) -> Result<()>;
    
    /// Build and save to SafeTensors
    pub fn build_safetensors(&self, dir: &Path) -> Result<()>;
}
```

### Loading

```rust
impl LLMBuilder {
    /// Load from GGUF file
    pub fn load_from_gguf(path: &Path) -> Result<Self>;
    
    /// Load from SafeTensors directory
    pub fn load_from_safetensors(dir: &Path) -> Result<Self>;
}
```

### Example

```rust
use llm_builder::{LLMBuilder, QuantConfig, QuantType};

let builder = LLMBuilder::from_size("8b")
    .enable_manifold(0.618)
    .with_quantization(QuantConfig {
        quant_type: QuantType::Q4_K_M,
        phi_aware_scaling: true,
        ..Default::default()
    });

builder.build_gguf(Path::new("model.gguf"))?;
```

---

## GGUFWriter

```rust
impl GGUFWriter {
    /// Create new writer
    pub fn new(path: &Path) -> Result<Self>;
    
    /// Write header with tensor and KV counts
    pub fn write_header(&mut self, tensor_count: u64, kv_count: u64) -> Result<()>;
    
    /// Write string metadata
    pub fn write_kv_string(&mut self, key: &str, value: &str) -> Result<()>;
    
    /// Write u32 metadata
    pub fn write_kv_u32(&mut self, key: &str, value: u32) -> Result<()>;
    
    /// Write tensor info
    pub fn write_tensor_info(
        &mut self, name: &str, dims: &[u64], 
        ggml_type: GGUFType, offset: u64
    ) -> Result<()>;
    
    /// Write tensor data (auto-pads to 32 bytes)
    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()>;
    
    /// Flush to disk
    pub fn flush(&mut self) -> Result<()>;
}
```

---

## GGUFReader

```rust
impl GGUFReader {
    /// Open GGUF file
    pub fn open(path: &Path) -> Result<Self>;
    
    /// Read all metadata to HashMap
    pub fn read_metadata(&mut self) -> Result<HashMap<String, String>>;
}
```

---

## Manifold Operations

### Rust

```rust
/// Project to 7D Poincaré ball
pub fn project_to_poincare(v: &[f64], curvature: f64) -> Vec<f64>;

/// Hyperbolic distance
pub fn hyperbolic_distance(u: &[f64], v: &[f64]) -> f64;

/// Möbius addition
pub fn mobius_add(u: &[f64], v: &[f64], curvature: f64) -> Vec<f64>;

/// 7D projection operator
pub fn project_7d(x: &[f32]) -> [f32; 7];

/// Manifold arrow (directional alignment)
pub fn manifold_arrow(source: &[f32; 7], target: &[f32; 7]) -> f32;
```

### Python

```python
def project_to_poincare(v: np.ndarray, curvature: float = CURVATURE) -> np.ndarray:
    """Project vector onto 7D Poincaré ball."""
    
def hyperbolic_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Compute hyperbolic distance."""
    
def mobius_add(u: np.ndarray, v: np.ndarray, curvature: float = CURVATURE) -> np.ndarray:
    """Möbius addition in hyperbolic space."""
```

### CUDA

```cuda
__global__ void project_to_7d_poincare(
    float* input, float* output, 
    int batch_size, int dim, float curvature
);

__global__ void holographic_fold_7d(
    float* patterns, float* output,
    int batch_size, int num_patterns, int pattern_dim
);

__global__ void manifold_attention_kernel(
    float* Q, float* K, float* V, float* output, float* mask,
    int batch_size, int num_heads, int seq_len, int head_dim
);
```

---

## CUDA Kernels

### Projection

```cuda
__global__ void project_to_7d_poincare(
    float* input, float* output, 
    int batch_size, int dim, float curvature
);

__global__ void project_7d_to_3d(
    float* input_7d, float* output_3d, int batch_size
);
```

### Attention

```cuda
__global__ void manifold_attention_kernel(
    float* Q, float* K, float* V, float* output, float* mask,
    int batch_size, int num_heads, int seq_len, int head_dim
);

__global__ void rope_7d_kernel(
    float* q, float* k, float* cos_cache, float* sin_cache,
    int batch_size, int seq_len, int num_heads, int head_dim
);
```

### Feed-Forward

```cuda
__global__ void swiglu_ffn_7d_kernel(
    float* input, float* gate_weight, float* up_weight, float* down_weight,
    float* output, int batch_size, int hidden_size, int intermediate_size
);

__global__ void rmsnorm_7d_kernel(
    float* input, float* weight, float* output,
    int batch_size, int dim, float eps
);
```

### Quantization

```cuda
__global__ void quantize_int4_phi_kernel(
    float* input, uint8_t* output, float* scales,
    int n_elements, int group_size
);

__global__ void dequantize_int4_phi_kernel(
    uint8_t* input, float* scales, float* output,
    int n_elements, int group_size
);
```

---

## CLI Interface

### Commands

```bash
# Run inference
crystal7d run --model <path> --prompt <text> [options]

# Quantize model
crystal7d quantize --input <path> --output <path> --quant-type <type>

# Inspect model
crystal7d inspect --model <path> [--verbose]

# Run benchmarks
crystal7d benchmark [--seq-len <n>] [--iterations <n>]

# Run tests
crystal7d test [--verbose] [--filter <name>]

# Show info
crystal7d info
```

### Options

```
--model, -m         Model path (GGUF file)
--prompt, -p        Input prompt
--max-tokens        Maximum tokens to generate (default: 100)
--temperature       Sampling temperature (default: 0.7)
--top-p             Top-p sampling (default: 0.9)
--top-k             Top-k sampling (default: 40)
--backend           Compute backend: cpu, cuda, metal (default: cpu)
--quant-type        Quantization type: Q4_K_M, Q8_0, etc.
--phi-aware         Enable Φ-aware scaling
--verbose           Verbose output
```

---

**Discoverer**: Sir Charles Spikes
**Date**: December 24, 2025
**Location**: Cincinnati, Ohio, USA 🇺🇸
