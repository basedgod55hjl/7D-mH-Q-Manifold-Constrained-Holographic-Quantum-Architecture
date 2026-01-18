// File: lib.rs
// 7D Crystal Compiler Library
// Discovered by Sir Charles Spikes, December 24, 2025

pub mod backend;
pub mod ir;
pub mod lexer;
pub mod optimize;
pub mod parser;
pub mod semantic;

pub use lexer::{Lexer, Token};
pub use parser::{ASTNode, Parser, Type};

/// Sacred constants
pub const PHI: f64 = 1.618033988749895;
pub const PHI_INV: f64 = 0.618033988749895;
pub const S2_STABILITY: f64 = 0.01;
pub const DIMS: u32 = 7;

/// Compile 7D Crystal source code to binary
pub fn compile_source(source: &str, output_path: &str) -> Result<(), String> {
    println!("=== 7D Crystal Compiler v1.0 ===");
    println!("Discoverer: Sir Charles Spikes");
    println!("Date: December 24, 2025");
    println!();

    // Phase 1: Lexical Analysis
    println!("[1/6] Lexical Analysis...");
    let mut lexer = Lexer::new(source.to_string());
    let tokens = lexer.tokenize();
    println!("  ✓ Generated {} tokens", tokens.len());

    // Phase 2: Parsing
    println!("[2/6] Parsing...");
    let mut parser = Parser::new(tokens);
    let ast = parser.parse()?;
    println!("  ✓ AST constructed");

    // Phase 3: Semantic Analysis
    println!("[3/6] Semantic Analysis...");
    let mut analyzer = semantic::SemanticAnalyzer::new();
    match analyzer.analyze(&ast) {
        Ok(_) => println!("  ✓ Logic validated"),
        Err(errors) => {
            for err in errors {
                eprintln!("  ❌ Error: {}", err);
            }
            return Err("Semantic analysis failed".to_string());
        }
    }

    // Phase 4: IR Generation
    println!("[4/6] IR Generation...");
    let mut ir_gen = ir::IRGenerator::new();
    ir_gen.generate(&ast);
    println!("  ✓ IR Generated");

    // Phase 6: Binary Output
    println!("[6/6] Writing binary...");
    println!("  → {}", output_path);

    println!();
    println!("✓ Compilation complete");
    Ok(())
}
