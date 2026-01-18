//! Hybrid Layer Module

use super::simulator::{MeasurementBasis, QuantumCircuit};

pub struct HybridLayer {
    classical_dim: usize,
    num_qubits: usize,
}

impl HybridLayer {
    pub fn new(classical_dim: usize, num_qubits: usize) -> Self {
        Self {
            classical_dim,
            num_qubits,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut circuit = QuantumCircuit::new(self.num_qubits);

        // Encode classical input as rotation angles
        for (q, &val) in input.iter().enumerate().take(self.num_qubits) {
            if val > 0.5 {
                circuit.hadamard(q);
            }
        }

        // Entangle
        for q in 0..self.num_qubits.saturating_sub(1) {
            circuit.cnot(q, q + 1);
        }

        circuit.run();
        circuit.probabilities()
    }
}
