# Contributing to 7D Crystal System

First off, thank you for considering contributing to the 7D Crystal System! It's people like you who help make it a sovereign, mathematically perfect computing environment.

## Code of Conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs

- **Check for existing issues**: Before opening a new issue, search the issue tracker to see if the problem has already been reported.
- **Provide detail**: Include a clear title and description, as much relevant information as possible, and a code sample or a test case that demonstrates the issue.

### Suggesting Enhancements

- **Explain the "why"**: Clearly explain why this enhancement would be useful and how it aligns with the 7D Crystal System vision.
- **Be specific**: Provide as much detail as possible about the proposed changes.

### Pull Requests

1. **Fork the repository** and create your branch from `main`.
2. **Follow the style guide**: Ensure your code follows the Rust and C++ style guides used in the project.
3. **Verify Φ-ratio preservation**: Any new algorithm or operator must maintain Φ-ratio relationships.
4. **Ensure S² stability**: All manifold operations must be bounded by S² (0.01).
5. **Add tests**: Any new feature must include comprehensive unit and/or integration tests.
6. **Update documentation**: If you're adding a new feature, update the relevant documentation.

## Development Setup

### Prerequisites

- **Rust** 1.75+
- **CUDA Toolkit** 12.0+
- **Git**

### Building

```bash
cargo build --release
```

### Testing

```bash
cargo test
python megatron/core/transformer/test_nvidia_integration.py
```

## Style Guide

- **Rust**: Use `cargo fmt` and `cargo clippy`.
- **C++**: Follow Chromium style guide for MatX/CUDA code.
- **Documentation**: Use GitHub Flavored Markdown.

## Mathematical Constraints

- **Φ** = 1.618033988749895
- **S² Stability Bound** = 0.01
- **Dimensions** = 7

All contributions must respect these constants.

---

**Sovereignty through mathematical perfection.**
