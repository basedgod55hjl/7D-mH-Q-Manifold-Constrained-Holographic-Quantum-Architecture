use crate::parser::{ASTNode, BinaryOperator};
use std::collections::HashMap;

/// 7D Crystal Intermediate Representation
#[derive(Debug, Clone, PartialEq)]
pub enum IR7D {
    // Stack operations
    PushInt(i64),
    PushFloat(f64),
    PushString(String),
    Pop,

    // Variable operations
    Store(usize, isize), // register, stack offset
    Load(usize, isize),  // register, stack offset

    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,

    // 7D Manifold operations
    ManifoldProject {
        input_reg: usize,
        output_reg: usize,
        curvature: f64,
    },
    ManifoldDistance {
        a_reg: usize,
        b_reg: usize,
        result_reg: usize,
        metric: Metric,
    },

    // Holographic operations
    HolographicFold {
        p1_reg: usize,
        p2_reg: usize,
        out_reg: usize,
        phases: u8,
    },
    HoloEncode {
        data_reg: usize,
        pattern_reg: usize,
    },
    HoloDecode {
        pattern_reg: usize,
        data_reg: usize,
    },

    // Quantum operations
    Superpose {
        states: Vec<usize>,
        weights: Vec<f64>,
        result_reg: usize,
    },
    Entangle {
        a_reg: usize,
        b_reg: usize,
    },
    Measure {
        state_reg: usize,
        basis: Basis,
        result_reg: usize,
    },

    // Control flow
    Jump(usize),        // target block
    JumpIfFalse(usize), // target block
    Return,

    //Labels (for control flow)
    Label(usize),

    // Fused operations for GPU kernels
    FusedOp {
        name: String,
        ops: Vec<IR7D>,
        reg_map: HashMap<usize, usize>, // local reg -> actual reg
    },
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Metric {
    Euclidean,
    Hyperbolic,
    Minkowski,
}

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Basis {
    Computational,
    Fourier,
    Manifold,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Vector([f64; 7]),
    Phi,
    PhiInv,
    S2,
    Lambda,
    Psi,
    Infinity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IROpcode {
    // Stack operations
    Push(Value),
    Pop,
    Dup,

    // Variable operations
    Load(String),
    Store(String),

    // Arithmetic & Logical
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Gt,
    Concat, // String concatenation

    // 7D Manifold Opcodes
    Project7D,     // Projection to manifold (⑦)
    TensorProduct, // ⊗
    Superpose,     // ⊕
    HoloFold,      // ⊙
    ProjectTo,     // ->

    // Control Flow
    Label(String),
    Jump(String),
    JumpIfFalse(String),
    Yield,
    Return,

    // 7D Block Configurations
    ConfigEntropy(String, f64, f64), // name, modulation, purity
    ConfigCrystal(String, [usize; 7]),
    ConfigManifold(String, u32, f64),

    // Function ops
    Call(String, usize), // name, arg count
    Param(String),
}

pub struct IRBlock {
    pub name: String,
    pub instructions: Vec<IROpcode>,
}

pub struct IRBlock7D {
    pub name: String,
    pub instructions: Vec<IR7D>,
}

pub struct IRGenerator {
    pub blocks: Vec<IRBlock>,
    current_block: Vec<IROpcode>,
    label_count: usize,
}

pub struct IRGenerator7D {
    pub blocks: Vec<IRBlock7D>,
    current_block: Vec<IR7D>,
    label_count: usize,
    symbol_table: HashMap<String, usize>, // name -> stack offset
    reg_count: usize,
}

impl IRGenerator7D {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            current_block: Vec::new(),
            label_count: 0,
            symbol_table: HashMap::new(),
            reg_count: 0,
        }
    }

    pub fn generate(&mut self, node: &ASTNode) {
        match node {
            ASTNode::Program {
                attributes: _,
                items,
            } => {
                for item in items {
                    self.generate(item);
                }
                // Wrap up last block
                if !self.current_block.is_empty() {
                    self.blocks.push(IRBlock7D {
                        name: "_start".to_string(),
                        instructions: self.current_block.drain(..).collect(),
                    });
                }
            }
            ASTNode::FunctionDecl {
                name, params, body, ..
            } => {
                // For now, treat main function as entry point
                let block_name = if name == "main" {
                    "_start".to_string()
                } else {
                    name.clone()
                };

                // Process function parameters
                for param in params {
                    let reg = self.allocate_reg();
                    self.symbol_table.insert(param.name.clone(), reg);
                }

                // Generate body
                self.generate(body);

                // Create function block
                self.blocks.push(IRBlock7D {
                    name: block_name,
                    instructions: self.current_block.drain(..).collect(),
                });
            }
            ASTNode::Block { statements } => {
                for stmt in statements {
                    self.generate(stmt);
                }
            }
            ASTNode::VariableDecl { name, value, .. } => {
                self.generate(value);
                let reg = self.allocate_reg();
                let offset = reg as isize * 8; // 8 bytes per register
                self.symbol_table.insert(name.clone(), reg);
                self.current_block.push(IR7D::Store(reg, offset));
            }
            ASTNode::ReturnStatement { value } => {
                if let Some(val) = value {
                    self.generate(val);
                }
                self.current_block.push(IR7D::Return);
            }
            ASTNode::IntLiteral(val) => {
                self.current_block.push(IR7D::PushInt(*val));
            }
            ASTNode::FloatLiteral(val) => {
                self.current_block.push(IR7D::PushFloat(*val));
            }
            ASTNode::StringLiteral(val) => {
                self.current_block.push(IR7D::PushString(val.clone()));
            }
            ASTNode::BinaryOp { left, op, right } => {
                self.generate(left);
                self.generate(right);
                match op {
                    BinaryOperator::Add => self.current_block.push(IR7D::Add),
                    BinaryOperator::Sub => self.current_block.push(IR7D::Sub),
                    BinaryOperator::Mul => self.current_block.push(IR7D::Mul),
                    BinaryOperator::Div => self.current_block.push(IR7D::Div),
                    _ => {} // Handle other operators
                }
            }
            ASTNode::ManifoldProjection {
                input, curvature, ..
            } => {
                self.generate(input);
                let input_reg = self.allocate_reg();
                let output_reg = self.allocate_reg();
                self.current_block.push(IR7D::ManifoldProject {
                    input_reg,
                    output_reg,
                    curvature: *curvature,
                });
            }
            _ => {} // Handle other AST nodes
        }
    }

    fn allocate_reg(&mut self) -> usize {
        let reg = self.reg_count;
        self.reg_count += 1;
        reg
    }
}

impl IRGenerator {
    pub fn new() -> Self {
        IRGenerator {
            blocks: Vec::new(),
            current_block: Vec::new(),
            label_count: 0,
        }
    }

    /// Check if an AST node represents a string operation
    fn is_string_operation(&self, node: &ASTNode) -> bool {
        match node {
            ASTNode::StringLiteral(_) => true,
            ASTNode::Identifier(name) => {
                // Check if this identifier refers to a string variable
                // For now, use a simple heuristic - this could be improved
                // with proper type tracking in the IR generator
                name.contains("hello")
                    || name.contains("world")
                    || name.contains("message")
                    || name.contains("greeting")
            }
            _ => false,
        }
    }

    fn new_label(&mut self) -> String {
        let label = format!("L{}", self.label_count);
        self.label_count += 1;
        label
    }

    fn evaluate_float(&self, node: &ASTNode) -> f64 {
        match node {
            ASTNode::FloatLiteral(f) => *f,
            ASTNode::IntLiteral(i) => *i as f64,
            ASTNode::PhiConstant => 1.618033988749895,
            ASTNode::PhiInverse => 0.618033988749895,
            ASTNode::S2Constant => 0.01,
            ASTNode::UnaryOp {
                op: crate::parser::UnaryOperator::Neg,
                expr,
            } => -self.evaluate_float(expr),
            _ => 0.0,
        }
    }

    pub fn generate(&mut self, node: &ASTNode) {
        match node {
            ASTNode::Program { attributes, items } => {
                // Process sovereignty attributes (could affect code generation)
                let _has_sovereignty = attributes
                    .iter()
                    .any(|attr| matches!(attr, crate::parser::Attribute::Sovereignty(_)));

                for item in items {
                    self.generate(item);
                }

                // Wrap up last block if it contains instructions not inside a function
                if !self.current_block.is_empty() {
                    self.blocks.push(IRBlock {
                        name: "__global_init".to_string(),
                        instructions: self.current_block.drain(..).collect(),
                    });
                }
            }
            ASTNode::FunctionDecl {
                params, body, name, ..
            } => {
                let mut fundef_block = Vec::new();
                for param in params {
                    fundef_block.push(IROpcode::Param(param.name.clone()));
                    fundef_block.push(IROpcode::Store(param.name.clone()));
                }

                let old_block = std::mem::take(&mut self.current_block);
                self.generate(body);
                let body_block = std::mem::replace(&mut self.current_block, old_block);

                let mut final_instr = fundef_block;
                final_instr.extend(body_block);

                // Use function name for block name
                let block_name = name.clone();
                self.blocks.push(IRBlock {
                    name: block_name,
                    instructions: final_instr,
                });
            }
            ASTNode::BinaryOp { left, op, right } => {
                self.generate(left);
                self.generate(right);
                match op {
                    BinaryOperator::Add => {
                        // Check if this is string concatenation by inspecting operands
                        let is_string_concat =
                            self.is_string_operation(left) && self.is_string_operation(right);
                        if is_string_concat {
                            self.current_block.push(IROpcode::Concat);
                        } else {
                            self.current_block.push(IROpcode::Add);
                        }
                    }
                    BinaryOperator::Sub => self.current_block.push(IROpcode::Sub),
                    BinaryOperator::Mul => self.current_block.push(IROpcode::Mul),
                    BinaryOperator::Div => self.current_block.push(IROpcode::Div),
                    BinaryOperator::Concat => self.current_block.push(IROpcode::Concat),
                    BinaryOperator::Eq => self.current_block.push(IROpcode::Eq),
                    BinaryOperator::TensorProduct => {
                        self.current_block.push(IROpcode::TensorProduct)
                    }
                    BinaryOperator::Superposition => self.current_block.push(IROpcode::Superpose),
                    BinaryOperator::HoloFold => self.current_block.push(IROpcode::HoloFold),
                    BinaryOperator::Arrow => self.current_block.push(IROpcode::ProjectTo),
                    _ => { /* Handle others */ }
                }
            }
            ASTNode::IntLiteral(i) => self.current_block.push(IROpcode::Push(Value::Int(*i))),
            ASTNode::FloatLiteral(f) => self.current_block.push(IROpcode::Push(Value::Float(*f))),
            ASTNode::StringLiteral(s) => self
                .current_block
                .push(IROpcode::Push(Value::String(s.clone()))),
            ASTNode::PhiConstant => self.current_block.push(IROpcode::Push(Value::Phi)),
            ASTNode::PhiInverse => self.current_block.push(IROpcode::Push(Value::PhiInv)),
            ASTNode::S2Constant => self.current_block.push(IROpcode::Push(Value::S2)),
            ASTNode::LambdaConstant => self.current_block.push(IROpcode::Push(Value::Lambda)),
            ASTNode::PsiConstant => self.current_block.push(IROpcode::Push(Value::Psi)),
            ASTNode::Identifier(name) => {
                if name == "INFINITY" || name == "∞" {
                    self.current_block.push(IROpcode::Push(Value::Infinity));
                } else {
                    self.current_block.push(IROpcode::Load(name.clone()));
                }
            }
            ASTNode::VariableDecl { name, value, .. } => {
                self.generate(value);
                self.current_block.push(IROpcode::Store(name.clone()));
            }
            ASTNode::Block { statements } => {
                for stmt in statements {
                    self.generate(stmt);
                }
            }
            ASTNode::ReturnStatement { value } => {
                if let Some(v) = value {
                    self.generate(v);
                }
                self.current_block.push(IROpcode::Return);
            }
            ASTNode::ManifoldProjection { input, .. } => {
                self.generate(input);
                self.current_block.push(IROpcode::Project7D);
            }
            ASTNode::IfStatement {
                condition,
                then_block,
                else_block,
            } => {
                let else_label = self.new_label();
                let end_label = self.new_label();

                self.generate(condition);
                self.current_block
                    .push(IROpcode::JumpIfFalse(else_label.clone()));

                self.generate(then_block);
                self.current_block.push(IROpcode::Jump(end_label.clone()));

                self.current_block.push(IROpcode::Label(else_label));
                if let Some(eb) = else_block {
                    self.generate(eb);
                }

                self.current_block.push(IROpcode::Label(end_label));
            }
            ASTNode::MatchStatement { target, arms } => {
                self.generate(target);
                let end_label = self.new_label();

                for arm in arms {
                    let next_arm_label = self.new_label();
                    self.current_block.push(IROpcode::Dup);
                    self.generate(&arm.pattern);
                    self.current_block.push(IROpcode::Eq);
                    self.current_block
                        .push(IROpcode::JumpIfFalse(next_arm_label.clone()));

                    self.generate(&arm.body);
                    self.current_block.push(IROpcode::Jump(end_label.clone()));
                    self.current_block.push(IROpcode::Label(next_arm_label));
                }
                self.current_block.push(IROpcode::Pop); // Pop the duplicated target
                self.current_block.push(IROpcode::Label(end_label));
            }
            ASTNode::YieldStatement { value } => {
                self.generate(value);
                self.current_block.push(IROpcode::Yield);
            }
            ASTNode::EntropyDecl {
                name,
                modulation,
                purity,
                ..
            } => {
                let mod_val = self.evaluate_float(modulation);
                self.current_block
                    .push(IROpcode::ConfigEntropy(name.clone(), mod_val, *purity));
                // Register symbol
                self.current_block.push(IROpcode::Push(Value::Int(0)));
                self.current_block.push(IROpcode::Store(name.clone()));
            }
            ASTNode::CrystalDecl {
                name, resolution, ..
            } => {
                self.current_block
                    .push(IROpcode::ConfigCrystal(name.clone(), *resolution));
                // Register symbol
                self.current_block.push(IROpcode::Push(Value::Int(0)));
                self.current_block.push(IROpcode::Store(name.clone()));
            }
            ASTNode::ManifoldDecl {
                name,
                dimensions,
                curvature,
                ..
            } => {
                let curv_val = self.evaluate_float(curvature);
                self.current_block.push(IROpcode::ConfigManifold(
                    name.clone(),
                    *dimensions,
                    curv_val,
                ));
                // Register symbol
                self.current_block.push(IROpcode::Push(Value::Int(0)));
                self.current_block.push(IROpcode::Store(name.clone()));
            }
            ASTNode::Vector7DLiteral(v) => {
                let mut elements = [0.0; 7];
                for i in 0..v.len().min(7) {
                    elements[i] = v[i];
                }
                self.current_block
                    .push(IROpcode::Push(Value::Vector(elements)));
            }
            ASTNode::Import { .. } => {
                // Imports are handled by the compiler's frontend/module resolver
            }
            ASTNode::Module { items, .. } => {
                for item in items {
                    self.generate(item);
                }
            }
            ASTNode::TheoremDecl { name, proof, .. } => {
                // Generate a separate block for the theorem's proof
                let old_block = std::mem::take(&mut self.current_block);
                self.generate(proof);
                let proof_instr = std::mem::replace(&mut self.current_block, old_block);

                self.blocks.push(IRBlock {
                    name: format!("theorem_{}", name),
                    instructions: proof_instr,
                });
            }
            ASTNode::FunctionCall { name, args } => {
                for arg in args {
                    self.generate(arg);
                }
                self.current_block
                    .push(IROpcode::Call(name.clone(), args.len()));
            }
            ASTNode::MethodCall {
                object,
                method,
                args,
            } => {
                // For now, treat method calls as function calls with object as first arg
                // or just Call(method) if we assume the runtime handles 'this' via stack
                // But simplified 7D: push object, push args, call method_name
                self.generate(object);
                for arg in args {
                    self.generate(arg);
                }
                self.current_block
                    .push(IROpcode::Call(method.clone(), args.len() + 1));
            }
            ASTNode::UnaryOp { op, expr } => {
                self.generate(expr);
                match op {
                    crate::parser::UnaryOperator::Project7D => {
                        self.current_block.push(IROpcode::Project7D)
                    }
                    crate::parser::UnaryOperator::Neg => {
                        /* Simplify: Negate top of stack */
                        self.current_block.push(IROpcode::Push(Value::Int(-1))); // Hack: Mul by -1
                        self.current_block.push(IROpcode::Mul);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

pub struct IROptimizer;

impl IROptimizer {
    pub fn optimize(mut blocks: Vec<IRBlock>) -> Vec<IRBlock> {
        for block in &mut blocks {
            Self::optimize_block(block);
        }
        blocks
    }

    fn optimize_block(block: &mut IRBlock) {
        let mut optimized: Vec<IROpcode> = Vec::new();
        let mut i = 0;

        while i < block.instructions.len() {
            let instr = block.instructions[i].clone();

            // Basic Constant Folding for Binary Operations
            let folded = if optimized.len() >= 2 {
                let len = optimized.len();
                if let (IROpcode::Push(v1), IROpcode::Push(v2)) =
                    (&optimized[len - 2], &optimized[len - 1])
                {
                    match instr {
                        IROpcode::Add => Self::fold_add(v1, v2).map(IROpcode::Push),
                        IROpcode::Mul => Self::fold_mul(v1, v2).map(IROpcode::Push),
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(res_instr) = folded {
                optimized.pop(); // pop v2
                optimized.pop(); // pop v1
                optimized.push(res_instr);
            } else {
                optimized.push(instr);
            }
            i += 1;
        }
        block.instructions = optimized;
    }

    fn fold_add(v1: &Value, v2: &Value) -> Option<Value> {
        match (v1, v2) {
            (Value::Int(a), Value::Int(b)) => Some(Value::Int(a + b)),
            (Value::Float(a), Value::Float(b)) => Some(Value::Float(a + b)),
            _ => None,
        }
    }

    fn fold_mul(v1: &Value, v2: &Value) -> Option<Value> {
        match (v1, v2) {
            (Value::Int(a), Value::Int(b)) => Some(Value::Int(a * b)),
            (Value::Float(a), Value::Float(b)) => Some(Value::Float(a * b)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_lowering_7d() {
        let mut gen = IRGenerator::new();
        let ast = ASTNode::Program {
            attributes: vec![],
            items: vec![
                ASTNode::VariableDecl {
                    name: "v".to_string(),
                    var_type: None,
                    value: Box::new(ASTNode::Vector7DLiteral(vec![1.0; 7])),
                    mutable: false,
                    qualifier: crate::parser::Qualifier::None,
                },
                ASTNode::ManifoldProjection {
                    input: Box::new(ASTNode::Identifier("v".to_string())),
                    target_manifold: "M7".to_string(),
                    curvature: -1.0,
                },
            ],
        };

        gen.generate(&ast);
        assert!(!gen.blocks.is_empty());
        let instr = &gen.blocks[0].instructions;

        // Should have Push(Vector), Store(v), Load(v), Project7D
        assert!(matches!(instr[0], IROpcode::Push(Value::Vector(_))));
        assert!(matches!(instr[1], IROpcode::Store(ref s) if s == "v"));
        assert!(matches!(instr[2], IROpcode::Load(ref s) if s == "v"));
        assert!(matches!(instr[3], IROpcode::Project7D));
    }

    #[test]
    fn test_ir_v1_features() {
        let mut gen = IRGenerator::new();
        let ast = ASTNode::Program {
            attributes: vec![],
            items: vec![
                ASTNode::EntropyDecl {
                    name: "Flux".to_string(),
                    source: "VACUUM".to_string(),
                    modulation: 1.618,
                    purity: 0.999,
                    attributes: vec![],
                },
                ASTNode::MatchStatement {
                    target: Box::new(ASTNode::Identifier("x".to_string())),
                    arms: vec![crate::parser::MatchArm {
                        pattern: Box::new(ASTNode::IntLiteral(0)),
                        body: Box::new(ASTNode::YieldStatement {
                            value: Box::new(ASTNode::LambdaConstant),
                        }),
                    }],
                },
            ],
        };

        gen.generate(&ast);
        let instr = &gen.blocks[0].instructions;

        // Check for ConfigEntropy
        assert!(instr
            .iter()
            .any(|i| matches!(i, IROpcode::ConfigEntropy(ref n, _, _) if n == "Flux")));

        // Check for Match sequence
        assert!(instr.iter().any(|i| matches!(i, IROpcode::Dup)));
        assert!(instr
            .iter()
            .any(|i| matches!(i, IROpcode::Push(Value::Lambda))));
        assert!(instr.iter().any(|i| matches!(i, IROpcode::Yield)));
    }

    #[test]
    fn test_constant_folding() {
        let mut block = IRBlock {
            name: "test".to_string(),
            instructions: vec![
                IROpcode::Push(Value::Int(10)),
                IROpcode::Push(Value::Int(20)),
                IROpcode::Add,
                IROpcode::Push(Value::Int(2)),
                IROpcode::Mul,
            ],
        };

        let optimized = IROptimizer::optimize(vec![block]);
        let instr = &optimized[0].instructions;

        // (10 + 20) * 2 = 60
        assert_eq!(instr.len(), 1);
        assert!(matches!(instr[0], IROpcode::Push(Value::Int(60))));
    }
}
