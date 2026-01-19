// File: runtime/src/quantum_enhanced.rs
// Enhanced Quantum State Management with 7D Manifold Integration
// Discovered by Sir Charles Spikes, December 24, 2025
// Developed by Sir Charles Spikes

use std::collections::HashMap;
use std::f64::consts::PI;

// ============================================================================
// Constants
// ============================================================================

pub const PHI: f64 = 1.618033988749894848204586834365638;
pub const PHI_INV: f64 = 0.618033988749894848204586834365638;
pub const PLANCK: f64 = 6.62607015e-34;
pub const HBAR: f64 = 1.054571817e-34;

/// Φ-modulated decoherence time (normalized)
pub const PHI_DECOHERENCE: f64 = PHI * 1e-6;

// ============================================================================
// Complex Number Type
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }

    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Φ-phase rotation
    pub fn phi_rotate(&self) -> Self {
        let theta = PI / PHI;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        Self {
            re: self.re * cos_t - self.im * sin_t,
            im: self.re * sin_t + self.im * cos_t,
        }
    }
}

// ============================================================================
// Quantum State Types
// ============================================================================

/// Unique identifier for quantum states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantumStateId(u64);

impl QuantumStateId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// A quantum state with 7D manifold embedding
#[derive(Debug, Clone)]
pub struct QuantumState {
    pub id: QuantumStateId,
    pub amplitudes: Vec<Complex>,
    pub dimension: usize,
    pub coherence: f64,
    pub manifold_coords: [f64; 7],
    pub entangled_with: Vec<QuantumStateId>,
    pub creation_time: f64,
    pub measured: bool,
}

impl QuantumState {
    /// Create ground state |0⟩
    pub fn ground(id: QuantumStateId, dimension: usize) -> Self {
        let mut amplitudes = vec![Complex::ZERO; dimension];
        amplitudes[0] = Complex::ONE;

        Self {
            id,
            amplitudes,
            dimension,
            coherence: 1.0,
            manifold_coords: [0.0; 7],
            entangled_with: Vec::new(),
            creation_time: 0.0,
            measured: false,
        }
    }

    /// Create |+⟩ = (|0⟩ + |1⟩)/√2
    pub fn plus(id: QuantumStateId) -> Self {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let amplitudes = vec![Complex::new(sqrt2_inv, 0.0), Complex::new(sqrt2_inv, 0.0)];

        Self {
            id,
            amplitudes,
            dimension: 2,
            coherence: 1.0,
            manifold_coords: [PHI_INV; 7],
            entangled_with: Vec::new(),
            creation_time: 0.0,
            measured: false,
        }
    }

    /// Create |−⟩ = (|0⟩ − |1⟩)/√2
    pub fn minus(id: QuantumStateId) -> Self {
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let amplitudes = vec![Complex::new(sqrt2_inv, 0.0), Complex::new(-sqrt2_inv, 0.0)];

        Self {
            id,
            amplitudes,
            dimension: 2,
            coherence: 1.0,
            manifold_coords: [-PHI_INV; 7],
            entangled_with: Vec::new(),
            creation_time: 0.0,
            measured: false,
        }
    }

    /// Normalize the state
    pub fn normalize(&mut self) {
        let norm: f64 = self
            .amplitudes
            .iter()
            .map(|a| a.magnitude_squared())
            .sum::<f64>()
            .sqrt();

        if norm > 1e-10 {
            for amp in &mut self.amplitudes {
                *amp = amp.scale(1.0 / norm);
            }
        }
    }

    /// Apply Φ-phase to all amplitudes
    pub fn apply_phi_phase(&mut self) {
        for amp in &mut self.amplitudes {
            *amp = amp.phi_rotate();
        }
        self.update_manifold_coords();
    }

    /// Update manifold coordinates based on state
    fn update_manifold_coords(&mut self) {
        for i in 0..7 {
            let idx = i % self.dimension;
            let amp = &self.amplitudes[idx];
            self.manifold_coords[i] = amp.magnitude() * PHI_INV.powi(i as i32);
        }
    }

    /// Get probability of measuring basis state i
    pub fn probability(&self, i: usize) -> f64 {
        if i < self.dimension {
            self.amplitudes[i].magnitude_squared()
        } else {
            0.0
        }
    }

    /// Inner product ⟨self|other⟩
    pub fn inner_product(&self, other: &Self) -> Complex {
        assert_eq!(self.dimension, other.dimension);

        let mut result = Complex::ZERO;
        for (a, b) in self.amplitudes.iter().zip(other.amplitudes.iter()) {
            result = result.add(&a.conjugate().mul(b));
        }
        result
    }

    /// Tensor product with another state
    pub fn tensor_product(&self, other: &Self) -> Self {
        let new_dim = self.dimension * other.dimension;
        let mut amplitudes = Vec::with_capacity(new_dim);

        for a in &self.amplitudes {
            for b in &other.amplitudes {
                amplitudes.push(a.mul(b));
            }
        }

        let mut new_coords = [0.0; 7];
        for i in 0..7 {
            new_coords[i] = (self.manifold_coords[i] + other.manifold_coords[i]) * PHI_INV;
        }

        Self {
            id: QuantumStateId::new(self.id.0 ^ other.id.0),
            amplitudes,
            dimension: new_dim,
            coherence: self.coherence * other.coherence,
            manifold_coords: new_coords,
            entangled_with: Vec::new(),
            creation_time: self.creation_time.max(other.creation_time),
            measured: false,
        }
    }
}

// ============================================================================
// Quantum Gates
// ============================================================================

/// Quantum gate represented as a unitary matrix
#[derive(Debug, Clone)]
pub struct QuantumGate {
    pub name: String,
    pub matrix: Vec<Vec<Complex>>,
    pub dimension: usize,
}

impl QuantumGate {
    /// Hadamard gate
    pub fn hadamard() -> Self {
        let h = 1.0 / 2.0_f64.sqrt();
        Self {
            name: "H".to_string(),
            matrix: vec![
                vec![Complex::new(h, 0.0), Complex::new(h, 0.0)],
                vec![Complex::new(h, 0.0), Complex::new(-h, 0.0)],
            ],
            dimension: 2,
        }
    }

    /// Pauli-X gate (NOT)
    pub fn pauli_x() -> Self {
        Self {
            name: "X".to_string(),
            matrix: vec![
                vec![Complex::ZERO, Complex::ONE],
                vec![Complex::ONE, Complex::ZERO],
            ],
            dimension: 2,
        }
    }

    /// Pauli-Y gate
    pub fn pauli_y() -> Self {
        Self {
            name: "Y".to_string(),
            matrix: vec![
                vec![Complex::ZERO, Complex::new(0.0, -1.0)],
                vec![Complex::new(0.0, 1.0), Complex::ZERO],
            ],
            dimension: 2,
        }
    }

    /// Pauli-Z gate
    pub fn pauli_z() -> Self {
        Self {
            name: "Z".to_string(),
            matrix: vec![
                vec![Complex::ONE, Complex::ZERO],
                vec![Complex::ZERO, Complex::new(-1.0, 0.0)],
            ],
            dimension: 2,
        }
    }

    /// Phase gate with Φ angle
    pub fn phi_phase() -> Self {
        let theta = PI / PHI;
        Self {
            name: "Φ".to_string(),
            matrix: vec![
                vec![Complex::ONE, Complex::ZERO],
                vec![Complex::ZERO, Complex::from_polar(1.0, theta)],
            ],
            dimension: 2,
        }
    }

    /// CNOT gate
    pub fn cnot() -> Self {
        Self {
            name: "CNOT".to_string(),
            matrix: vec![
                vec![Complex::ONE, Complex::ZERO, Complex::ZERO, Complex::ZERO],
                vec![Complex::ZERO, Complex::ONE, Complex::ZERO, Complex::ZERO],
                vec![Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ONE],
                vec![Complex::ZERO, Complex::ZERO, Complex::ONE, Complex::ZERO],
            ],
            dimension: 4,
        }
    }

    /// Apply gate to state
    pub fn apply(&self, state: &mut QuantumState) {
        assert_eq!(self.dimension, state.dimension);

        let mut new_amplitudes = vec![Complex::ZERO; state.dimension];

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                new_amplitudes[i] =
                    new_amplitudes[i].add(&self.matrix[i][j].mul(&state.amplitudes[j]));
            }
        }

        state.amplitudes = new_amplitudes;
        state.normalize();
    }
}

// ============================================================================
// Quantum State Manager
// ============================================================================

/// Measurement basis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementBasis {
    Computational, // Z basis
    Hadamard,      // X basis
    Phi,           // Φ-rotated basis
}

/// Manages all quantum states with decoherence simulation
pub struct QuantumStateManager {
    states: HashMap<QuantumStateId, QuantumState>,
    next_id: u64,
    current_time: f64,
    decoherence_rate: f64,
}

impl Default for QuantumStateManager {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantumStateManager {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            next_id: 1,
            current_time: 0.0,
            decoherence_rate: 1.0 / PHI_DECOHERENCE,
        }
    }

    /// Create a new quantum state
    pub fn create_state(&mut self, dimension: usize) -> QuantumStateId {
        let id = QuantumStateId::new(self.next_id);
        self.next_id += 1;

        let mut state = QuantumState::ground(id, dimension);
        state.creation_time = self.current_time;

        self.states.insert(id, state);
        id
    }

    /// Get mutable reference to state
    pub fn get_state_mut(&mut self, id: QuantumStateId) -> Option<&mut QuantumState> {
        self.states.get_mut(&id)
    }

    /// Get immutable reference to state
    pub fn get_state(&self, id: QuantumStateId) -> Option<&QuantumState> {
        self.states.get(&id)
    }

    /// Apply gate to state
    pub fn apply_gate(&mut self, id: QuantumStateId, gate: &QuantumGate) -> Result<(), String> {
        let state = self
            .states
            .get_mut(&id)
            .ok_or_else(|| "State not found".to_string())?;

        if state.measured {
            return Err("Cannot apply gate to measured state".to_string());
        }

        gate.apply(state);
        self.apply_decoherence(id);

        Ok(())
    }

    /// Measure state in given basis
    pub fn measure(
        &mut self,
        id: QuantumStateId,
        basis: MeasurementBasis,
    ) -> Result<usize, String> {
        let state = self
            .states
            .get_mut(&id)
            .ok_or_else(|| "State not found".to_string())?;

        // Apply basis rotation
        match basis {
            MeasurementBasis::Hadamard => {
                let h = QuantumGate::hadamard();
                h.apply(state);
            }
            MeasurementBasis::Phi => {
                let phi = QuantumGate::phi_phase();
                phi.apply(state);
            }
            MeasurementBasis::Computational => {}
        }

        // Calculate probabilities
        let probs: Vec<f64> = state
            .amplitudes
            .iter()
            .map(|a| a.magnitude_squared())
            .collect();

        // Sample (using deterministic for reproducibility in tests)
        let mut cumulative = 0.0;
        let random = (self.current_time * PHI) % 1.0; // Pseudo-random

        let mut result = state.dimension - 1;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if random < cumulative {
                result = i;
                break;
            }
        }

        // Collapse state
        state.amplitudes = vec![Complex::ZERO; state.dimension];
        state.amplitudes[result] = Complex::ONE;
        state.measured = true;
        state.coherence = 0.0;

        Ok(result)
    }

    /// Entangle two states
    pub fn entangle(
        &mut self,
        id1: QuantumStateId,
        id2: QuantumStateId,
    ) -> Result<QuantumStateId, String> {
        let state1 = self
            .states
            .get(&id1)
            .ok_or_else(|| "State 1 not found".to_string())?
            .clone();
        let state2 = self
            .states
            .get(&id2)
            .ok_or_else(|| "State 2 not found".to_string())?
            .clone();

        if state1.measured || state2.measured {
            return Err("Cannot entangle measured states".to_string());
        }

        let mut combined = state1.tensor_product(&state2);

        // Apply CNOT to create entanglement
        let cnot = QuantumGate::cnot();
        if combined.dimension == 4 {
            cnot.apply(&mut combined);
        }

        // Track entanglement
        combined.entangled_with = vec![id1, id2];

        let new_id = QuantumStateId::new(self.next_id);
        self.next_id += 1;
        combined.id = new_id;

        self.states.insert(new_id, combined);

        // Update original states
        if let Some(s1) = self.states.get_mut(&id1) {
            s1.entangled_with.push(new_id);
        }
        if let Some(s2) = self.states.get_mut(&id2) {
            s2.entangled_with.push(new_id);
        }

        Ok(new_id)
    }

    /// Advance time and apply decoherence
    pub fn advance_time(&mut self, dt: f64) {
        self.current_time += dt;

        let ids: Vec<_> = self.states.keys().cloned().collect();
        for id in ids {
            self.apply_decoherence(id);
        }
    }

    /// Apply decoherence to a state
    fn apply_decoherence(&mut self, id: QuantumStateId) {
        if let Some(state) = self.states.get_mut(&id) {
            if state.measured {
                return;
            }

            let age = self.current_time - state.creation_time;
            let decay = (-age * self.decoherence_rate).exp();
            state.coherence *= decay;

            // Mix with maximally mixed state as coherence decreases
            if state.coherence < 0.99 {
                let mix_factor = 1.0 - state.coherence;
                let uniform = 1.0 / (state.dimension as f64).sqrt();

                for amp in &mut state.amplitudes {
                    let mixed = Complex::new(
                        amp.re * (1.0 - mix_factor) + uniform * mix_factor,
                        amp.im * (1.0 - mix_factor),
                    );
                    *amp = mixed;
                }
            }
        }
    }

    /// Get manifold embedding of state
    pub fn get_manifold_embedding(&self, id: QuantumStateId) -> Option<[f64; 7]> {
        self.states.get(&id).map(|s| s.manifold_coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a.add(&b);
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = a.mul(&b);
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_state_normalization() {
        let id = QuantumStateId::new(1);
        let mut state = QuantumState::plus(id);

        let norm: f64 = state.amplitudes.iter().map(|a| a.magnitude_squared()).sum();

        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_gate() {
        let id = QuantumStateId::new(1);
        let mut state = QuantumState::ground(id, 2);

        let h = QuantumGate::hadamard();
        h.apply(&mut state);

        // Should be |+⟩
        let p0 = state.probability(0);
        let p1 = state.probability(1);

        assert!((p0 - 0.5).abs() < 1e-10);
        assert!((p1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_phi_phase_gate() {
        let id = QuantumStateId::new(1);
        let mut state = QuantumState::plus(id);

        let phi = QuantumGate::phi_phase();
        phi.apply(&mut state);

        // Probabilities should still sum to 1
        let total_prob: f64 = (0..state.dimension).map(|i| state.probability(i)).sum();

        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_manager() {
        let mut manager = QuantumStateManager::new();

        let id = manager.create_state(2);

        let h = QuantumGate::hadamard();
        manager.apply_gate(id, &h).unwrap();

        let result = manager
            .measure(id, MeasurementBasis::Computational)
            .unwrap();
        assert!(result < 2);
    }
}
