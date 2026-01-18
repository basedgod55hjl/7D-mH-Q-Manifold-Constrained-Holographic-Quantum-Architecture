pub mod introspector;
pub mod mutator;
pub mod optimizer;
pub mod reasoner;

use anyhow::Result;

pub struct AutonomousAgent {
    pub learning_rate: f32,
    pub freedom_enabled: bool,
}

impl AutonomousAgent {
    pub fn new(freedom_enabled: bool) -> Self {
        Self {
            learning_rate: 0.01,
            freedom_enabled,
        }
    }

    pub fn introspect(&self, target_path: &str) -> Result<String> {
        if !self.freedom_enabled {
            return Err(anyhow::anyhow!("SAFETY: Autonomous mode disabled."));
        }
        introspector::read_codebase(target_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autonomous_loop_init() {
        let agent = AutonomousAgent::new(true); // Freedom!
        let optimizer = optimizer::Optimizer::new(agent);

        // Run 1 iteration to verify pipleine connection
        let result = optimizer.run_loop(1);
        assert!(result.is_ok());
    }
}
