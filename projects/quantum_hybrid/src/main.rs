//! Quantum Hybrid Main

mod attention;
mod hybrid;
mod simulator;

pub use attention::*;
pub use hybrid::*;
pub use simulator::*;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     🔮 QUANTUM-CLASSICAL HYBRID SIMULATOR 🔮                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut circuit = QuantumCircuit::new(4);
    circuit.hadamard(0);
    circuit.hadamard(1);
    circuit.cnot(0, 2);
    circuit.cnot(1, 3);
    circuit.phi_gate(0);
    circuit.run();

    println!("\nQuantum circuit created with 4 qubits");
    println!(
        "Measurement: {:016b}",
        circuit.measure(MeasurementBasis::Computational)
    );

    let hybrid = HybridLayer::new(8, 4);
    let input: Vec<f64> = (0..8).map(|i| (i as f64) * 0.1).collect();
    let output = hybrid.forward(&input);
    println!("\nHybrid layer output: {:?}", &output[..4]);

    println!("\n✓ Quantum-Classical Hybrid complete!");
}
