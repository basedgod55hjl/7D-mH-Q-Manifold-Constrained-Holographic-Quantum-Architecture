# 🔮 Sovereign Local AI Assistant

The **Sovereign Assistant** is the primary interface to the **7D Crystal System**. It provides a high-fidelity, manifold-constrained environment for interacting with the **DeepSeek-R1-Distill-Llama-8B** model.

## Features

- **High-Fidelity UI**: RGB Linear Gradient interaction banner with "Thinking..." states.
- **7D Streaming**: Real-time token generation streaming from the Poincare Ball manifold.
- **Neural Interface**: Native integration with the `neural_language` crate.
- **Quantized Inference**: Runs efficiently on consumer hardware (Q4_K_M).

## Usage

```powershell
# Run the assistant
cargo run --release --bin sovereign
```

## Architecture

The assistant orchestrates:

1. **`llm_builder`**: Loads GGUF weights.
2. **`model_runner`**: Executes 7D-constrained inference.
3. **`neural_language`**: Provides the 7D logic context.
