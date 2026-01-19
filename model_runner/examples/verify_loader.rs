use llm_builder::{LLMBuilder, ModelConfig};
use model_runner::{ModelRunner, backend::Backend, sampler::SamplingParams};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== 7D Crystal LLM Loader Verification ===");

    // 1. Define Tiny Config
    // Must match the simple structure we support
    let config = ModelConfig {
        name: "Tiny-Verifier".to_string(),
        hidden_size: 64,
        intermediate_size: 192,
        num_layers: 2,
        num_attention_heads: 4,
        num_kv_heads: 2,
        vocab_size: 1000,
        max_position_embeddings: 128,
        rope_theta: 10000.0,
        ..Default::default()
    };

    println!("Configuration: {:?}", config);

    // 2. Build Dummy Model
    let builder = LLMBuilder::new(config.clone());
    let path = Path::new("tiny_model.gguf");
    
    println!("Building GGUF at {:?}...", path);
    match builder.build_gguf(path) {
        Ok(_) => println!("Build complete."),
        Err(e) => {
            eprintln!("Build failed: {}", e);
            return Err(e.into());
        }
    }

    // 3. Load Model
    println!("Loading model with ModelRunner...");
    let backend = Backend::CPU;
    
    match ModelRunner::from_gguf(path, backend) {
        Ok(runner) => {
            println!("Model loaded successfully!");
            println!("Model Config: {:?}", runner.config());

            // 4. Test Generation
            let tokens = vec![1, 2, 3];
            let params = SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                ..Default::default()
            };

            println!("Running inference step...");
            match runner.generate(&tokens, 5, &params) {
                Ok(output) => println!("Generated tokens: {:?}", output),
                Err(e) => eprintln!("Inference failed: {}", e),
            }
        },
        Err(e) => {
            eprintln!("Model load failed: {}", e);
            // Don't error out, we want to clean up
        }
    }

    // Cleanup
    if path.exists() {
        std::fs::remove_file(path)?;
        println!("Test artifact cleaned up.");
    }
    
    println!("=== VERIFICATION FINISHED ===");
    Ok(())
}
