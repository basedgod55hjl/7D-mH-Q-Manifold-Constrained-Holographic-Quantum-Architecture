// 7D Crystal Quantum State Manager
// Manages quantum superposition, entanglement, and measurement
// Discovered by Sir Charles Spikes, December 24, 2025

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

const PHI: f64 = 1.618033988749895;
const PHI_INV: f64 = 0.618033988749895;
const S2_BOUND: f64 = 0.01;

/// Unique identifier for quantum states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateID(u64);

impl StateID {
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        StateID(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// Measurement basis for quantum states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Basis {
    /// Computational basis (|0⟩, |1⟩)
    Computational,
    /// Fourier basis (phase states)
    Fourier,
    /// Manifold basis (7D coordinate states)
    Manifold,
    /// Holographic basis (interference patterns)
    Holographic,
}

/// Complex amplitude for quantum states
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
        }
    }

    pub fn one() -> Self {
        Self {
            real: 1.0,
            imag: 0.0,
        }
    }

    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self {
            real: magnitude * phase.cos(),
            imag: magnitude * phase.sin(),
        }
    }

    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn magnitude_sq(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            real: self.real * factor,
            imag: self.imag * factor,
        }
    }
}

/// Wave function representation
#[derive(Debug, Clone)]
pub struct WaveFunction {
    /// Amplitude for each basis state
    amplitudes: Vec<Complex>,
    /// Dimension of the Hilbert space
    dimension: usize,
    /// Normalization factor
    normalization: f64,
    /// Whether state has been measured
    collapsed: bool,
}

impl WaveFunction {
    /// Create ground state |0⟩
    pub fn ground(dimension: usize) -> Self {
        let mut amplitudes = vec![Complex::zero(); dimension];
        if dimension > 0 {
            amplitudes[0] = Complex::one();
        }
        Self {
            amplitudes,
            dimension,
            normalization: 1.0,
            collapsed: false,
        }
    }

    /// Create superposition of all states (|+⟩ state)
    pub fn plus(dimension: usize) -> Self {
        let amplitude = Complex::new(1.0 / (dimension as f64).sqrt(), 0.0);
        let amplitudes = vec![amplitude; dimension];
        Self {
            amplitudes,
            dimension,
            normalization: 1.0,
            collapsed: false,
        }
    }

    /// Create state with given amplitudes
    pub fn from_amplitudes(amplitudes: Vec<Complex>) -> Self {
        let dimension = amplitudes.len();
        let mut wf = Self {
            amplitudes,
            dimension,
            normalization: 1.0,
            collapsed: false,
        };
        wf.normalize();
        wf
    }

    /// Get probability of finding system in state |i⟩
    pub fn probability(&self, i: usize) -> f64 {
        if i < self.dimension {
            self.amplitudes[i].magnitude_sq()
        } else {
            0.0
        }
    }

    /// Normalize the wave function
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a.magnitude_sq()).sum();

        if norm_sq > 1e-15 {
            let norm = norm_sq.sqrt();
            self.normalization = norm;
            for amp in &mut self.amplitudes {
                *amp = amp.scale(1.0 / norm);
            }
        }
    }

    /// Apply Φ-ratio phase rotation to all states
    pub fn apply_phi_rotation(&mut self) {
        let phase = std::f64::consts::PI * PHI_INV;
        for (i, amp) in self.amplitudes.iter_mut().enumerate() {
            let state_phase = phase * (i as f64);
            let rotation = Complex::from_polar(1.0, state_phase);
            *amp = amp.mul(&rotation);
        }
    }

    /// Measure in given basis, collapsing the state
    pub fn measure(&mut self, _basis: Basis) -> usize {
        if self.collapsed {
            // Already collapsed, return previously measured state
            for (i, amp) in self.amplitudes.iter().enumerate() {
                if amp.magnitude_sq() > 0.99 {
                    return i;
                }
            }
            return 0;
        }

        // Generate "random" outcome based on Φ
        let rng_state = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0) as f64)
            * PHI;

        let random = (rng_state.sin() + 1.0) / 2.0;

        // Find outcome based on cumulative probabilities
        let mut cumulative = 0.0;
        let mut outcome = 0;

        for (i, amp) in self.amplitudes.iter().enumerate() {
            cumulative += amp.magnitude_sq();
            if random <= cumulative {
                outcome = i;
                break;
            }
        }

        // Collapse the state
        self.collapsed = true;
        for (i, amp) in self.amplitudes.iter_mut().enumerate() {
            if i == outcome {
                *amp = Complex::one();
            } else {
                *amp = Complex::zero();
            }
        }

        outcome
    }
}

/// Entanglement record between two states
#[derive(Debug, Clone)]
pub struct EntanglementRecord {
    pub state_a: StateID,
    pub state_b: StateID,
    pub correlation: f64,
    pub basis: Basis,
}

/// Quantum State Manager
/// Tracks all quantum states, superpositions, and entanglements
pub struct QuantumStateManager {
    /// All managed quantum states
    states: HashMap<StateID, WaveFunction>,
    /// Entanglement graph
    entanglements: HashMap<StateID, Vec<StateID>>,
    /// Entanglement records
    entanglement_records: Vec<EntanglementRecord>,
    /// Decoherence time constant (in abstract time units)
    decoherence_time: f64,
}

impl QuantumStateManager {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            entanglements: HashMap::new(),
            entanglement_records: Vec::new(),
            decoherence_time: 1.0 / S2_BOUND, // Related to S² stability
        }
    }

    /// Create a new quantum state in ground state
    pub fn create_state(&mut self, dimension: usize) -> StateID {
        let id = StateID::new();
        self.states.insert(id, WaveFunction::ground(dimension));
        id
    }

    /// Create superposition from multiple states with weights
    pub fn superpose(&mut self, states: &[StateID], weights: &[f64]) -> StateID {
        // Normalize weights using Φ
        let weight_sum: f64 = weights.iter().sum();
        let normalized_weights: Vec<f64> =
            weights.iter().map(|w| w * PHI_INV / weight_sum).collect();

        // Find maximum dimension
        let max_dim = states
            .iter()
            .filter_map(|id| self.states.get(id))
            .map(|wf| wf.dimension)
            .max()
            .unwrap_or(2);

        // Create superposition
        let mut amplitudes = vec![Complex::zero(); max_dim];

        for (i, &state_id) in states.iter().enumerate() {
            if let Some(wf) = self.states.get(&state_id) {
                let weight = normalized_weights.get(i).copied().unwrap_or(1.0);
                for (j, amp) in wf.amplitudes.iter().enumerate() {
                    amplitudes[j] = amplitudes[j].add(&amp.scale(weight.sqrt()));
                }
            }
        }

        let new_id = StateID::new();
        self.states
            .insert(new_id, WaveFunction::from_amplitudes(amplitudes));

        new_id
    }

    /// Entangle two quantum states
    pub fn entangle(&mut self, a: StateID, b: StateID) -> Result<(), QuantumError> {
        if !self.states.contains_key(&a) || !self.states.contains_key(&b) {
            return Err(QuantumError::StateNotFound);
        }

        // Check if already entangled
        if let Some(partners) = self.entanglements.get(&a) {
            if partners.contains(&b) {
                return Err(QuantumError::AlreadyEntangled);
            }
        }

        // Create bidirectional entanglement
        self.entanglements.entry(a).or_insert_with(Vec::new).push(b);
        self.entanglements.entry(b).or_insert_with(Vec::new).push(a);

        // Record entanglement
        self.entanglement_records.push(EntanglementRecord {
            state_a: a,
            state_b: b,
            correlation: PHI_INV, // Bell correlation
            basis: Basis::Computational,
        });

        Ok(())
    }

    /// Measure a quantum state
    pub fn measure(&mut self, state_id: StateID, basis: Basis) -> Result<f64, QuantumError> {
        let outcome = {
            let state = self
                .states
                .get_mut(&state_id)
                .ok_or(QuantumError::StateNotFound)?;
            state.measure(basis) as f64
        };

        // Collapse all entangled states
        if let Some(entangled) = self.find_entangled(state_id) {
            for partner_id in entangled {
                self.collapse_entangled(partner_id, outcome)?;
            }
        }

        Ok(outcome)
    }

    /// Apply quantum evolution (time evolution operator)
    pub fn evolve(&mut self, state_id: StateID, time_steps: usize) -> Result<(), QuantumError> {
        let state = self
            .states
            .get_mut(&state_id)
            .ok_or(QuantumError::StateNotFound)?;

        // Apply Φ-ratio phase rotation for each time step
        for _ in 0..time_steps {
            state.apply_phi_rotation();
        }

        Ok(())
    }

    /// Get state probability distribution
    pub fn get_probabilities(&self, state_id: StateID) -> Result<Vec<f64>, QuantumError> {
        let state = self
            .states
            .get(&state_id)
            .ok_or(QuantumError::StateNotFound)?;

        Ok((0..state.dimension).map(|i| state.probability(i)).collect())
    }

    /// Remove a quantum state
    pub fn destroy_state(&mut self, state_id: StateID) {
        self.states.remove(&state_id);
        self.entanglements.remove(&state_id);

        // Remove from other entanglement lists
        for partners in self.entanglements.values_mut() {
            partners.retain(|&id| id != state_id);
        }
    }

    /// Get number of active states
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    // Private helpers

    fn find_entangled(&self, state_id: StateID) -> Option<Vec<StateID>> {
        self.entanglements.get(&state_id).cloned()
    }

    fn collapse_entangled(
        &mut self,
        state_id: StateID,
        correlated_outcome: f64,
    ) -> Result<(), QuantumError> {
        let state = self
            .states
            .get_mut(&state_id)
            .ok_or(QuantumError::StateNotFound)?;

        // Collapse to correlated state (Bell state behavior)
        let outcome = (correlated_outcome * PHI_INV).round() as usize;
        let outcome = outcome % state.dimension;

        // Force collapse
        for (i, amp) in state.amplitudes.iter_mut().enumerate() {
            if i == outcome {
                *amp = Complex::one();
            } else {
                *amp = Complex::zero();
            }
        }
        state.collapsed = true;

        Ok(())
    }
}

/// Quantum operation errors
#[derive(Debug)]
pub enum QuantumError {
    StateNotFound,
    AlreadyEntangled,
    DecoherenceOccurred,
    MeasurementFailed,
    UnitarityViolation,
}

impl std::fmt::Display for QuantumError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            QuantumError::StateNotFound => write!(f, "Quantum state not found"),
            QuantumError::AlreadyEntangled => write!(f, "States are already entangled"),
            QuantumError::DecoherenceOccurred => write!(f, "Quantum decoherence occurred"),
            QuantumError::MeasurementFailed => write!(f, "Quantum measurement failed"),
            QuantumError::UnitarityViolation => write!(f, "Unitarity violation detected"),
        }
    }
}

impl std::error::Error for QuantumError {}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let mut qsm = QuantumStateManager::new();
        let state = qsm.create_state(2);
        assert_eq!(qsm.state_count(), 1);

        let probs = qsm.get_probabilities(state).unwrap();
        assert!((probs[0] - 1.0).abs() < 1e-10); // Ground state
        assert!(probs[1].abs() < 1e-10);
    }

    #[test]
    fn test_superposition() {
        let mut qsm = QuantumStateManager::new();
        let state1 = qsm.create_state(2);
        let state2 = qsm.create_state(2);

        let superposed = qsm.superpose(&[state1, state2], &[1.0, 1.0]);

        let probs = qsm.get_probabilities(superposed).unwrap();
        // Should have some probability in |0⟩
        assert!(probs[0] > 0.0);
    }

    #[test]
    fn test_entanglement() {
        let mut qsm = QuantumStateManager::new();
        let state_a = qsm.create_state(2);
        let state_b = qsm.create_state(2);

        qsm.entangle(state_a, state_b).unwrap();

        // Should not be able to entangle again
        assert!(qsm.entangle(state_a, state_b).is_err());
    }

    #[test]
    fn test_measurement() {
        let mut qsm = QuantumStateManager::new();
        let state = qsm.create_state(2);

        let result = qsm.measure(state, Basis::Computational).unwrap();
        // Ground state should measure to 0
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_evolution() {
        let mut qsm = QuantumStateManager::new();
        let state = qsm.create_state(4);

        // Evolve should not error
        qsm.evolve(state, 10).unwrap();
    }

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 0.0);
        let b = Complex::new(0.0, 1.0);

        let sum = a.add(&b);
        assert!((sum.real - 1.0).abs() < 1e-10);
        assert!((sum.imag - 1.0).abs() < 1e-10);

        let product = a.mul(&b);
        assert!(product.real.abs() < 1e-10);
        assert!((product.imag - 1.0).abs() < 1e-10);
    }
}
