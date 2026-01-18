// 7D Crystal LLM Builder - Manifold Module
// Implements 7D Manifold Projection Operators: ⑦ and ->

use crate::PHI_BASIS;

/// The ⑦ operator: Projects a vector into the 7D Crystal manifold.
/// This uses the PHI_BASIS vectors to define the 7 axes of the manifold.
pub fn project_7d(x: &[f32]) -> [f32; 7] {
    let mut result = [0.0; 7];
    let n = x.len();

    // Each of the 7 dimensions is a weighted sum using the PHI basis
    for i in 0..7 {
        let basis = PHI_BASIS[i] as f32;
        let mut sum = 0.0;
        for (j, &val) in x.iter().enumerate() {
            // Apply a periodic Φ-harmonic transformation
            sum += val * (basis * (j as f32) / (n as f32) * 2.0 * std::f32::consts::PI).cos();
        }
        result[i] = sum / (n as f32).sqrt();
    }

    result
}

/// The -> operator: Manifold Arrow / Projection operator.
/// Measures the directional alignment between two manifold states.
pub fn manifold_arrow(source: &[f32; 7], target: &[f32; 7]) -> f32 {
    let mut dot = 0.0;
    let mut mag_s = 0.0;
    let mut mag_t = 0.0;

    for i in 0..7 {
        dot += source[i] * target[i];
        mag_s += source[i] * source[i];
        mag_t += target[i] * target[i];
    }

    // Regularized cosine similarity in 7D space
    if mag_s > 0.0 && mag_t > 0.0 {
        dot / (mag_s.sqrt() * mag_t.sqrt())
    } else {
        0.0
    }
}

pub struct ManifoldState {
    pub coordinates: [f32; 7],
    pub energy: f32,
    pub stability: f32,
}

impl ManifoldState {
    pub fn from_vector(x: &[f32]) -> Self {
        let coords = project_7d(x);
        let energy = coords.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let stability = 1.0 / (1.0 + energy / crate::S2_STABILITY as f32);

        Self {
            coordinates: coords,
            energy,
            stability,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_energy() {
        let x = vec![1.0; 128];
        let state = ManifoldState::from_vector(&x);
        assert!(state.energy >= 0.0);
    }

    #[test]
    fn test_arrow_identity() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let similarity = manifold_arrow(&x, &x);
        assert!((similarity - 1.0).abs() < 1e-6);
    }
}
