use std::env;
use std::fs;
use std::process;

use crystal_compiler::backend::{cuda::CudaBackend, x86_64::X86Backend, Backend};
use crystal_compiler::ir::{IRGenerator, IROptimizer};
use crystal_compiler::lexer::Lexer;
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

    let source_path = if command == "compile" {
        if args.len() < 3 {
            process::exit(1);
        }
        &args[2]
    } else {
        &args[1]
    };

    let backend_type = if args.len() > 3 && args[3] == "--backend" {
        &args[4]
    } else {
        "x86"
    };

    let source_code = match fs::read_to_string(source_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    };

    // 1. Lexical Analysis
    let mut lexer = Lexer::new(source_code);
    let tokens = lexer.tokenize();

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
    let mut ir_gen = IRGenerator::new();
    ir_gen.generate(&ast);
    let blocks = IROptimizer::optimize(ir_gen.blocks);

    // 5. Code Generation
    let backend: Box<dyn Backend> = match backend_type {
        "cuda" => Box::new(CudaBackend),
        _ => Box::new(X86Backend),
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
