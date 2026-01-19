//! # 7D Crystal Compiler
//!
//! A production-grade compiler for the 7D-MHQL (7D Manifold-Constrained
//! Holographic Quantum Language).
//!
//! ## Overview
//!
//! The 7D Crystal Compiler transforms `.7d` source files into optimized
//! binaries that execute on the 7D Crystal Runtime. All computations are
//! constrained to a 7-dimensional Poincaré ball manifold with:
//!
//! - **Φ-ratio preservation** (Golden Ratio: 1.618033988749895)
//! - **S² stability bounds** (||x|| < 0.01)
//! - **Native quantum operations** (superposition, entanglement)
//! - **Holographic memory** (interference pattern encoding)
//!
//! ## Architecture
//!
//! ```text
//! Source (.7d) → Lexer → Parser → Semantic → IR → Optimizer → Codegen
//!                  ↓        ↓        ↓        ↓       ↓          ↓
//!               Tokens    AST    Typed    7D-IR  Optimized   Binary
//!                                 AST             IR
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use crystal_compiler::{compile, CompilerOptions};
//!
//! let source = r#"
//!     @sovereignty Main {
//!         version: "1.0.0",
//!     }
//!     quantum cortex main() -> i32 {
//!         return 0;
//!     }
//! "#;
//!
//! let options = CompilerOptions::default();
//! let result = compile(source, &options)?;
//! ```
//!
//! ## Mathematical Constants
//!
//! The compiler enforces these fundamental constants:
//!
//! | Constant | Value | Description |
//! |----------|-------|-------------|
//! | Φ | 1.618033988749895 | Golden Ratio |
//! | Φ⁻¹ | 0.618033988749895 | Inverse Golden Ratio |
//! | S² | 0.01 | Stability Bound |
//! | DIMS | 7 | Manifold Dimensions |
//!
//! ## Author
//!
//! Discovered by Sir Charles Spikes, December 24, 2025
//! Cincinnati, Ohio, USA 🇺🇸

#![warn(missing_docs)]
#![warn(rustdoc::missing_doc_code_examples)]
#![deny(unsafe_op_in_unsafe_fn)]

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC MODULES
// ═══════════════════════════════════════════════════════════════════════════════

pub mod backend;
pub mod errors;
pub mod ir;
pub mod lexer;
pub mod optimize;
pub mod optimize_enhanced;
pub mod parser;
pub mod semantic;

// ═══════════════════════════════════════════════════════════════════════════════
// RE-EXPORTS
// ═══════════════════════════════════════════════════════════════════════════════

pub use errors::{Diagnostic, DiagnosticCollection, ErrorCode, Severity, SourceLocation};
pub use lexer::{Lexer, Token};
pub use parser::{ASTNode, Parser, Type};

// ═══════════════════════════════════════════════════════════════════════════════
// SACRED CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// The Golden Ratio (Φ) - fundamental to all 7D Crystal computations.
pub const PHI: f64 = 1.618033988749895;

/// The inverse Golden Ratio (Φ⁻¹) - used for manifold curvature.
pub const PHI_INV: f64 = 0.618033988749895;

/// The S² stability bound - all manifold vectors must satisfy ||x|| < S².
pub const S2_STABILITY: f64 = 0.01;

/// The number of dimensions in the Poincaré ball manifold.
pub const DIMS: u32 = 7;

/// Φ² (Phi squared) - frequently used in calculations.
pub const PHI_SQUARED: f64 = 2.618033988749895;

/// Tolerance for Φ-ratio verification.
pub const PHI_TOLERANCE: f64 = 1e-6;

// ═══════════════════════════════════════════════════════════════════════════════
// COMPILER OPTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration options for the 7D Crystal compiler.
#[derive(Debug, Clone)]
pub struct CompilerOptions {
    /// Optimization level (0-3).
    pub opt_level: u8,
    /// Enable debug information.
    pub debug_info: bool,
    /// Target backend (cuda, hip, metal, cpu).
    pub target: CompilerTarget,
    /// Enable verbose output.
    pub verbose: bool,
    /// Maximum errors before aborting.
    pub max_errors: usize,
    /// Enable Φ-ratio verification (recommended).
    pub verify_phi_ratio: bool,
    /// Enable S² stability checks (recommended).
    pub verify_stability: bool,
    /// Output path for compiled binary.
    pub output_path: Option<String>,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        Self {
            opt_level: 2,
            debug_info: false,
            target: CompilerTarget::Cuda,
            verbose: false,
            max_errors: 10,
            verify_phi_ratio: true,
            verify_stability: true,
            output_path: None,
        }
    }
}

/// Supported compilation targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerTarget {
    /// NVIDIA CUDA GPUs
    Cuda,
    /// AMD ROCm/HIP GPUs
    Hip,
    /// Apple Metal GPUs
    Metal,
    /// CPU fallback with SIMD
    Cpu,
    /// WebAssembly (experimental)
    Wasm,
}

impl std::fmt::Display for CompilerTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompilerTarget::Cuda => write!(f, "cuda"),
            CompilerTarget::Hip => write!(f, "hip"),
            CompilerTarget::Metal => write!(f, "metal"),
            CompilerTarget::Cpu => write!(f, "cpu"),
            CompilerTarget::Wasm => write!(f, "wasm"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPILATION RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a successful compilation.
#[derive(Debug)]
pub struct CompilationResult {
    /// The generated binary code.
    pub binary: Vec<u8>,
    /// IR representation (for debugging).
    pub ir: Option<String>,
    /// Compilation statistics.
    pub stats: CompilationStats,
    /// Any warnings generated.
    pub warnings: Vec<Diagnostic<()>>,
}

/// Statistics from compilation.
#[derive(Debug, Default)]
pub struct CompilationStats {
    /// Number of tokens generated.
    pub token_count: usize,
    /// Number of AST nodes.
    pub ast_node_count: usize,
    /// Number of IR instructions.
    pub ir_instruction_count: usize,
    /// Compilation time in milliseconds.
    pub compile_time_ms: u64,
    /// Number of optimizations applied.
    pub optimizations_applied: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPILATION FUNCTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Compile 7D Crystal source code to binary.
///
/// # Arguments
///
/// * `source` - The 7D Crystal source code as a string.
/// * `options` - Compiler configuration options.
///
/// # Returns
///
/// * `Ok(CompilationResult)` - Successful compilation with binary and stats.
/// * `Err(DiagnosticCollection)` - Compilation failed with error diagnostics.
///
/// # Example
///
/// ```rust
/// use crystal_compiler::{compile, CompilerOptions};
///
/// let source = r#"
///     quantum cortex main() -> i32 {
///         return 0;
///     }
/// "#;
///
/// match compile(source, &CompilerOptions::default()) {
///     Ok(result) => println!("Compiled {} bytes", result.binary.len()),
///     Err(errors) => errors.print_all(),
/// }
/// ```
pub fn compile(
    source: &str,
    options: &CompilerOptions,
) -> Result<CompilationResult, DiagnosticCollection> {
    use std::time::Instant;

    let start = Instant::now();
    let mut diagnostics = DiagnosticCollection::new();
    let mut stats = CompilationStats::default();

    if options.verbose {
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  🔮 7D CRYSTAL COMPILER v2.0.0");
        println!("  Discovered by Sir Charles Spikes | Cincinnati, Ohio, USA");
        println!("═══════════════════════════════════════════════════════════════════");
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 1: LEXICAL ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════

    if options.verbose {
        println!("[1/6] Lexical Analysis...");
    }

    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize()?;
    stats.token_count = tokens.len();

    if options.verbose {
        println!("      ✓ Generated {} tokens", tokens.len());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 2: PARSING
    // ═══════════════════════════════════════════════════════════════════════════

    if options.verbose {
        println!("[2/6] Parsing...");
    }

    let mut parser = Parser::new(tokens);
    let ast = match parser.parse() {
        Ok(ast) => ast,
        Err(err) => {
            diagnostics.add(
                Diagnostic::new(Severity::Error)
                    .with_code(ErrorCode::E2001.to_string())
                    .with_message(err),
            );
            return Err(diagnostics);
        }
    };

    stats.ast_node_count = count_ast_nodes(&ast);

    if options.verbose {
        println!("      ✓ AST constructed ({} nodes)", stats.ast_node_count);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 3: SEMANTIC ANALYSIS
    // ═══════════════════════════════════════════════════════════════════════════

    if options.verbose {
        println!("[3/6] Semantic Analysis...");
    }

    let mut analyzer = semantic::SemanticAnalyzer::new();
    if let Err(errors) = analyzer.analyze(&ast) {
        for err in errors {
            diagnostics.add(
                Diagnostic::new(Severity::Error)
                    .with_code(ErrorCode::E3001.to_string())
                    .with_message(err),
            );
        }

        if diagnostics.error_count() >= options.max_errors {
            return Err(diagnostics);
        }
    }

    if options.verbose {
        println!("      ✓ Semantic validation complete");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 4: IR GENERATION
    // ═══════════════════════════════════════════════════════════════════════════

    if options.verbose {
        println!("[4/6] IR Generation...");
    }

    let mut ir_gen = ir::IRGenerator7D::new();
    ir_gen.generate(&ast);
    let mut ir_blocks = ir_gen.get_blocks();
    stats.ir_instruction_count = ir_blocks.iter().map(|b| b.instructions.len()).sum();

    if options.verbose {
        println!(
            "      ✓ Generated {} IR instructions",
            stats.ir_instruction_count
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 5: OPTIMIZATION
    // ═══════════════════════════════════════════════════════════════════════════

    if options.verbose {
        println!("[5/6] Optimization (level {})...", options.opt_level);
    }

    let opt_level = match options.opt_level {
        0 => optimize::OptimizationLevel::None,
        1 => optimize::OptimizationLevel::Basic,
        _ => optimize::OptimizationLevel::Aggressive,
    };

    let optimizer = optimize::Optimizer7D::new(opt_level);
    optimizer.optimize(&mut ir_blocks);

    // Count optimizations
    let new_instruction_count: usize = ir_blocks.iter().map(|b| b.instructions.len()).sum();
    stats.optimizations_applied = stats
        .ir_instruction_count
        .saturating_sub(new_instruction_count);
    stats.ir_instruction_count = new_instruction_count;

    if options.verbose {
        println!(
            "      ✓ Optimized ({} instructions removed)",
            stats.optimizations_applied
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // PHASE 6: CODE GENERATION
    // ═══════════════════════════════════════════════════════════════════════════

    if options.verbose {
        println!("[6/6] Code Generation (target: {})...", options.target);
    }

    let binary = backend::generate_code(&ir_blocks, options.target)?;

    if options.verbose {
        println!("      ✓ Generated {} bytes", binary.len());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // COMPLETE
    // ═══════════════════════════════════════════════════════════════════════════

    stats.compile_time_ms = start.elapsed().as_millis() as u64;

    if options.verbose {
        println!();
        println!("═══════════════════════════════════════════════════════════════════");
        println!("  ✓ Compilation complete in {}ms", stats.compile_time_ms);
        println!("═══════════════════════════════════════════════════════════════════");
    }

    // Collect warnings
    let warnings: Vec<Diagnostic<()>> = diagnostics
        .iter()
        .filter(|d| d.severity == Severity::Warning)
        .cloned()
        .collect();

    Ok(CompilationResult {
        binary,
        ir: if options.debug_info {
            Some(format_ir(&ir_blocks))
        } else {
            None
        },
        stats,
        warnings,
    })
}

/// Legacy compilation function (for backwards compatibility).
pub fn compile_source(source: &str, output_path: &str) -> Result<(), String> {
    let options = CompilerOptions {
        output_path: Some(output_path.to_string()),
        verbose: true,
        ..Default::default()
    };

    match compile(source, &options) {
        Ok(result) => {
            // Write binary to output path
            if let Err(e) = std::fs::write(output_path, &result.binary) {
                return Err(format!("Failed to write output: {}", e));
            }
            Ok(())
        }
        Err(diagnostics) => {
            diagnostics.print_all();
            Err("Compilation failed".to_string())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

fn count_ast_nodes(ast: &ASTNode) -> usize {
    match ast {
        ASTNode::Program { items, .. } => 1 + items.iter().map(count_ast_nodes).sum::<usize>(),
        ASTNode::FunctionDecl { body, .. } => 1 + count_ast_nodes(body),
        ASTNode::Block { statements } => 1 + statements.iter().map(count_ast_nodes).sum::<usize>(),
        ASTNode::BinaryOp { left, right, .. } => 1 + count_ast_nodes(left) + count_ast_nodes(right),
        ASTNode::IfStatement {
            condition,
            then_block,
            else_block,
            ..
        } => {
            1 + count_ast_nodes(condition)
                + count_ast_nodes(then_block)
                + else_block.as_ref().map_or(0, |b| count_ast_nodes(b))
        }
        _ => 1,
    }
}

fn format_ir(blocks: &[ir::IRBlock7D]) -> String {
    let mut output = String::new();
    for block in blocks {
        output.push_str(&format!("{}:\n", block.name));
        for (i, instr) in block.instructions.iter().enumerate() {
            output.push_str(&format!("  {:4}: {:?}\n", i, instr));
        }
        output.push('\n');
    }
    output
}

// ═══════════════════════════════════════════════════════════════════════════════
// MATHEMATICAL UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/// Verify that a ratio is within tolerance of the Golden Ratio.
#[inline]
pub fn verify_phi_ratio(ratio: f64) -> bool {
    (ratio - PHI).abs() < PHI_TOLERANCE
}

/// Verify that a norm satisfies the S² stability bound.
#[inline]
pub fn verify_s2_stability(norm: f64) -> bool {
    norm < S2_STABILITY
}

/// Project a value to the Poincaré ball (enforce S² bound).
#[inline]
pub fn project_to_ball(value: f64) -> f64 {
    let max_val = S2_STABILITY * 0.99; // Leave some margin
    value.clamp(-max_val, max_val)
}

/// Compute the hyperbolic distance between two points.
pub fn hyperbolic_distance(a: &[f64; 7], b: &[f64; 7]) -> f64 {
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt();

    let numerator = 2.0 * diff * diff;
    let denominator = (1.0 - norm_a * norm_a) * (1.0 - norm_b * norm_b);

    (1.0 + numerator / denominator).acosh()
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_ratio_verification() {
        assert!(verify_phi_ratio(PHI));
        assert!(verify_phi_ratio(PHI + 1e-7));
        assert!(!verify_phi_ratio(1.5));
    }

    #[test]
    fn test_s2_stability_verification() {
        assert!(verify_s2_stability(0.001));
        assert!(verify_s2_stability(0.009));
        assert!(!verify_s2_stability(0.01));
        assert!(!verify_s2_stability(0.1));
    }

    #[test]
    fn test_project_to_ball() {
        assert!(verify_s2_stability(project_to_ball(1.0).abs()));
        assert!(verify_s2_stability(project_to_ball(-1.0).abs()));
        assert!(verify_s2_stability(project_to_ball(0.005).abs()));
    }

    #[test]
    fn test_compile_simple() {
        let source = r#"
            quantum cortex main() -> i32 {
                return 0;
            }
        "#;

        let options = CompilerOptions::default();
        // Just test that it doesn't panic
        let _ = compile(source, &options);
    }

    #[test]
    fn test_constants() {
        // Verify mathematical relationships
        assert!((PHI * PHI_INV - 1.0).abs() < 1e-10);
        assert!((PHI - PHI_INV - 1.0).abs() < 1e-10);
        assert!((PHI_SQUARED - PHI - 1.0).abs() < 1e-10);
    }
}
