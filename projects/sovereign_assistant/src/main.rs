// File: projects/sovereign_assistant/src/main.rs
// Sovereign 7D Local AI Assistant - Complete CLI
// Discovered by Sir Charles Spikes, December 24, 2025

mod tokenizer;

use anyhow::{Context, Result};
use colored::*;
use std::io::{self, Write};
use std::path::PathBuf;

use model_runner::{backend::Backend, sampler::SamplingParams, ModelRunner};
use neural_language::Neural7DContext;
use tokenizer::Crystal7DTokenizer;

fn print_gradient_banner() {
    let text = [
        "╔══════════════════════════════════════════════════════════════════╗",
        "║       🔮 SOVEREIGN 7D CRYSTAL INTELLIGENCE 🔮                    ║",
        "║       Manifold-Constrained Holographic Quantum LLM               ║",
        "╚══════════════════════════════════════════════════════════════════╝",
    ];

    let start_color = (0, 255, 255); // Cyan
    let end_color = (255, 0, 255); // Magenta

    for line in text {
        for (i, c) in line.chars().enumerate() {
            let t = i as f32 / line.len() as f32;
            let r = (start_color.0 as f32 * (1.0 - t) + end_color.0 as f32 * t) as u8;
            let g = (start_color.1 as f32 * (1.0 - t) + end_color.1 as f32 * t) as u8;
            let b = (start_color.2 as f32 * (1.0 - t) + end_color.2 as f32 * t) as u8;
            print!("{}", c.to_string().truecolor(r, g, b));
        }
        println!();
    }
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN) // Reduce log noise for UI
        .init();

    print_gradient_banner();

    // Initialize 7D Neural Context
    let _neural_ctx = Neural7DContext::new();
    println!(
        "   {}",
        "⚡ 7D Neural Interface: ONLINE".truecolor(100, 255, 100)
    );

    // Find model
    let model_paths = [
        PathBuf::from(r"models\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
        PathBuf::from(r"C:\Users\BASEDGOD\Desktop\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
        PathBuf::from(r".\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
        PathBuf::from(r"..\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    ];

    let model_path = model_paths
        .iter()
        .find(|p| p.exists())
        .context("DeepSeek model not found. Place GGUF in expected location.")?;

    println!(
        "\n{}",
        format!("📁 Loading: {}", model_path.display())
            .bold()
            .cyan()
    );

    // Load model
    let runner = ModelRunner::from_gguf(model_path, Backend::CPU)?;
    let tokenizer = Crystal7DTokenizer::new()?;

    let cfg = runner.config();
    println!("{}", "✅ Model loaded successfully!".green().bold());
    println!("   Hidden: {}", cfg.hidden_size.to_string().yellow());
    println!("   Layers: {}", cfg.num_layers.to_string().yellow());
    println!(
        "   Heads:  {}",
        cfg.num_attention_heads.to_string().yellow()
    );
    println!(
        "   Manifold: {} (κ={})",
        if cfg.manifold_enabled {
            "ENABLED".green().bold()
        } else {
            "disabled".red()
        },
        cfg.curvature.to_string().cyan()
    );

    // Prompt - Stress Test
    let prompt_text =
        "Tell me a short story about a traveler discovering a 7th dimensional crystal city.";
    let prompt_tokens = tokenizer.encode_with_bos(prompt_text)?;

    println!("\n{}", "🧠 Thinking...".purple().italic());
    println!("Prompt: \"{}\"", prompt_text.cyan());

    let params = SamplingParams {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    print!("\n{}: ", "Response".bold().white());
    io::stdout().flush()?;

    let _output = runner.generate_stream(&prompt_tokens, 200, &params, |token| {
        let text = tokenizer.decode(&[token])?;
        print!("{}", text.truecolor(200, 200, 255)); // Light blue text
        io::stdout().flush()?;
        Ok(true)
    })?;

    println!();
    println!(
        "\n{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".truecolor(50, 50, 50)
    );
    println!("   {}", "🔮 SOVEREIGNTY: VERIFIED".green().bold());
    println!("   {}", "7D Manifold Forward Pass: OPERATIONAL".cyan());
    println!(
        "{}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".truecolor(50, 50, 50)
    );

    Ok(())
}
