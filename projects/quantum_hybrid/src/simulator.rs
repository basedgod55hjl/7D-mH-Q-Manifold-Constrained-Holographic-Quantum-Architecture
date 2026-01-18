//! Quantum Hybrid - Simulator Module
//! Re-export from main for separate compilation

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::PI;

const PHI: f64 = 1.618033988749895;
const PHI_INV: f64 = 0.618033988749895;
const MANIFOLD_DIMS: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementBasis {
    Computational,
    Fourier,
    Manifold,
    Holographic,
}

pub struct QuantumCircuit {
    pub num_qubits: usize,
    state: Vec<Complex64>,
}

impl QuantumCircuit {
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[0] = Complex64::new(1.0, 0.0);
        Self { num_qubits, state }
    }

    pub fn hadamard(&mut self, qubit: usize) {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let dim = self.state.len();
        let mask = 1 << qubit;
        for i in 0..dim {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = Complex64::new(sqrt2_inv, 0.0) * (a + b);
                self.state[j] = Complex64::new(sqrt2_inv, 0.0) * (a - b);
            }
        }
    }

    pub fn cnot(&mut self, control: usize, target: usize) {
        let dim = self.state.len();
        let c_mask = 1 << control;
        let t_mask = 1 << target;
        for i in 0..dim {
            if (i & c_mask != 0) && (i & t_mask == 0) {
                let j = i | t_mask;
                self.state.swap(i, j);
            }
        }
    }

    pub fn phi_gate(&mut self, qubit: usize) {
        let dim = self.state.len();
        let mask = 1 << qubit;
        let phase = Complex64::from_polar(1.0, 2.0 * PI * PHI_INV);
        for i in 0..dim {
            if i & mask != 0 {
                self.state[i] *= phase;
            }
        }
    }

    pub fn z(&mut self, qubit: usize) {
        let dim = self.state.len();
        let mask = 1 << qubit;
        for i in 0..dim {
            if i & mask != 0 {
                self.state[i] = -self.state[i];
            }
        }
    }

    pub fn x(&mut self, qubit: usize) {
        let dim = self.state.len();
        let mask = 1 << qubit;
        for i in 0..dim {
            if i & mask == 0 {
                let j = i | mask;
                self.state.swap(i, j);
            }
        }
    }

    pub fn run(&mut self) {
        let norm_sq: f64 = self.state.iter().map(|c| c.norm_sqr()).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-10 {
            for c in &mut self.state {
                *c /= norm;
            }
        }
    }

    pub fn measure(&mut self, _basis: MeasurementBasis) -> usize {
        let probs: Vec<f64> = self.state.iter().map(|c| c.norm_sqr()).collect();
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r <= cumsum {
                self.state = vec![Complex64::new(0.0, 0.0); self.state.len()];
                self.state[i] = Complex64::new(1.0, 0.0);
                return i;
            }
        }
        self.state.len() - 1
    }

    pub fn probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|c| c.norm_sqr()).collect()
    }

    pub fn amplitudes(&self) -> &[Complex64] {
        &self.state
    }
}
