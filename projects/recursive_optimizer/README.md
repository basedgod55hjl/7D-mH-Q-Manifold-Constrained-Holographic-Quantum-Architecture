# 🔄 7D Recursive Optimizer

**Sovereign Self-Modification Engine**

This crate is responsible for the "Autonomy Loop":

1. **Introspection**: Reading the system's own source code.
2. **Hypothesis**: Generating potential optimizations via the 7D Manifold.
3. **Mutation**: Safely applying diffs to the codebase.

## ⚠️ Safety Warning

This module has the capability to modify the running system.
Sandbox constraints must be explicitly disabled (`--freedom`) to function.

## Architecture

- `introspector.rs`: AST parsing of Rust code.
- `mutator.rs`: Applying textual patches.
- `verifier.rs`: Running `cargo test` on modified code.
