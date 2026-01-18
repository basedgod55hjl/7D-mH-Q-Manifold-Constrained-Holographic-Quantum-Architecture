// 7D Crystal LLM Builder - "Build" Entry Point
// This script demonstrates building a full model GGUF from the framework

use llm_builder::{LLMBuilder, ModelConfig, QuantConfig, QuantType};
use std::path::Path;

fn main() -> anyhow::Result<()> {
    // 1. Configure the build
    println!("--- Initializing 7D Crystal LLM Build ---");
    let model_size = "1.5b";
    let mut builder = LLMBuilder::from_size(model_size);

    // 2. Set quantization policy
    let quant_config = QuantConfig {
        quant_type: QuantType::Q4_K_M,
        ..Default::default()
    };
    builder = builder.with_quantization(quant_config);

    // 3. Define output path
    let output_dir = Path::new("models").join("7D-Crystal-1.5B");
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
    }
    let output_path = output_dir.join("7D-Crystal-1.5B-Q4_K_M.gguf");

    // 4. Execute the build
    println!("Target Path: {:?}", output_path);
    println!("Architecture: 7D Crystal Manifold");

    match builder.build_gguf(&output_path) {
        Ok(_) => {
            println!("\n--- Build Successful ---");
            println!(
                "GGUF File Size: {} bytes",
                std::fs::metadata(&output_path)?.len()
            );
        }
        Err(e) => eprintln!("Build failed: {}", e),
    }

    Ok(())
}
