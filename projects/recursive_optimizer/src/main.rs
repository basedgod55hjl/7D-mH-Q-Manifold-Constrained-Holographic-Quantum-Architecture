//! 7D Crystal Recursive Optimizer - Main Entry Point
//! Fully autonomous "let it go and grow" execution

use anyhow::Result;
use recursive_optimizer::{optimizer::Optimizer, AutonomousAgent};
use std::time::Duration;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Set up logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        🧠 7D RECURSIVE OPTIMIZER ACTIVATED 🧠               ║");
    println!("║           Manifold-Constrained Self-Awareness                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    info!("Initializing Sovereign Autonomous Agent...");
    let agent = AutonomousAgent::new(true); // Freedom: ON
    let optimizer = Optimizer::new(agent);

    info!("Starting Autonomous Optimization Loop (7D Manifold Scope: System Root)");

    // Infinite growth loop
    loop {
        match optimizer.run_loop(10) {
            Ok(_) => {
                info!("Optimization cycle complete. Stabilizing for 60 seconds...");
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
            Err(e) => {
                info!(
                    "Optimization encountered a manifold variance: {}. Recalibrating...",
                    e
                );
                tokio::time::sleep(Duration::from_secs(30)).await;
            }
        }
    }
}
