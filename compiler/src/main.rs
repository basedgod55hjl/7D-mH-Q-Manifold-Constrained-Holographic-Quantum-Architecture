use std::env;
use std::fs;
use std::process;

use crystal_compiler::backend::{Backend7D, CpuBackend, CudaBackend};
use crystal_compiler::ir::IRGenerator7D;
use crystal_compiler::lexer::Lexer;
use crystal_compiler::optimize::{OptimizationLevel, Optimizer7D};
use crystal_compiler::parser::Parser;
use crystal_compiler::semantic::SemanticAnalyzer;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: crystal_compiler <command> [options]");
        process::exit(1);
    }

    let command = &args[1];

    if command == "sovereignty" {
        println!("Sovereignty v1.0 complete.");
        return;
    }

    let source_path = &args[1];
    let mut backend_type = "x86";

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--backend" && i + 1 < args.len() {
            backend_type = &args[i + 1];
            i += 2;
        } else if args[i] == "-o" && i + 1 < args.len() {
            // Skip output for now as it's hardcoded later
            i += 2;
        } else {
            i += 1;
        }
    }

    let source_code = match fs::read_to_string(source_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    // 1. Lexical Analysis
    let mut lexer = Lexer::new(&source_code);
    let tokens = match lexer.tokenize() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Lexer Error: {:?}", e);
            process::exit(1);
        }
    };

    // 2. Parsing
    let mut parser = Parser::new(tokens);
    let ast = match parser.parse() {
        Ok(node) => node,
        Err(e) => {
            eprintln!("Parse Error: {}", e);
            process::exit(1);
        }
    };

    // 3. Semantic Analysis
    let mut analyzer = SemanticAnalyzer::new();
    match analyzer.analyze(&ast) {
        Ok(()) => {}
        Err(errors) => {
            for err in errors {
                eprintln!("Semantic Error: {}", err);
            }
            process::exit(1);
        }
    }

    // 4. IR Generation
    let mut ir_gen = IRGenerator7D::new();
    ir_gen.generate(&ast);
    let mut blocks = ir_gen.blocks;

    // 5. Optimization
    let optimizer = Optimizer7D::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut blocks);

    // 6. Code Generation
    let backend: Box<dyn Backend7D> = match backend_type {
        "cuda" => Box::new(CudaBackend::new()),
        _ => Box::new(CpuBackend::new()),
    };

    let output = match backend.emit(&blocks) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Codegen Error: {}", e);
            process::exit(1);
        }
    };

    let output_file = format!("{}.out", source_path);
    fs::write(&output_file, output).unwrap();
    println!("Compilation successful: {}", output_file);
}
