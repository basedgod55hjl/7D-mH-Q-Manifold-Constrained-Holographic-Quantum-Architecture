use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

pub struct GGUFLoader;

impl GGUFLoader {
    /// Load all tensors from a GGUF file into a HashMap of F32 Tensors (dequantized)
    pub fn load_deepseek_weights(path: &Path) -> Result<HashMap<String, Tensor>> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        let mut weights = HashMap::new();

        for (tensor_name, tensor_info) in content.tensor_infos.iter() {
            let tensor = tensor_info.read(&mut file, &content.tensor_data_offset, &Device::Cpu)?;

            // Dequantize to F32 immediately for our custom engine
            // In a real optimized scenario, we'd keep them quantized or use Candle's QTensor
            // But to fit the existing "Vec<f32>" architecture of ModelRunner, we dequantize.
            let tensor_f32 = tensor.dequantize(&Device::Cpu)?;

            weights.insert(tensor_name.clone(), tensor_f32);
        }

        Ok(weights)
    }
}
