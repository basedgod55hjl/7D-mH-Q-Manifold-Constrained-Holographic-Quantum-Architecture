use crate::parser::{ASTNode, Topology, Type};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Symbol {
    pub name: String,
    pub sym_type: Type,
    pub is_mutable: bool,
}

pub struct SymbolTable {
    scopes: Vec<HashMap<String, Symbol>>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn define(&mut self, name: String, symbol: Symbol) -> Result<(), String> {
        if let Some(scope) = self.scopes.last_mut() {
            if scope.contains_key(&name) {
                return Err(format!("Symbol '{}' already defined in this scope", name));
            }
            scope.insert(name, symbol);
            Ok(())
        } else {
            Err("No scope available".to_string())
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(symbol) = scope.get(name) {
                return Some(symbol);
            }
        }
        None
    }
}

pub struct SemanticAnalyzer {
    symbol_table: SymbolTable,
    errors: Vec<String>,
}

impl SemanticAnalyzer {
    pub fn new() -> Self {
        let mut symbol_table = SymbolTable::new();
        // Register built-in functions
        let builtins = vec![
            ("println", Type::Void),
            ("sqrt", Type::Float),
            ("dot", Type::Float),
            ("norm", Type::Float),
            ("create_poincare_7d", Type::ManifoldState),
            ("crystallize", Type::Int),
            ("flux", Type::Float),
            ("fold", Type::ManifoldState),
            ("evolve", Type::QuantumState),
            ("verify_phi_ratios", Type::Bool),
            ("project", Type::ManifoldState),
            ("measure", Type::Bool),
        ];

        for (name, ret_type) in builtins {
            let _ = symbol_table.define(
                name.to_string(),
                Symbol {
                    name: name.to_string(),
                    sym_type: ret_type,
                    is_mutable: false,
                },
            );
        }

        SemanticAnalyzer {
            symbol_table,
            errors: Vec::new(),
        }
    }

    pub fn analyze(&mut self, root: &ASTNode) -> Result<(), Vec<String>> {
        self.check_node(root);
        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(self.errors.clone())
        }
    }

    fn check_node(&mut self, node: &ASTNode) {
        match node {
            ASTNode::Program { items, .. } => {
                for item in items {
                    self.check_node(item);
                }
            }
            ASTNode::ManifoldDecl {
                name,
                dimensions,
                curvature,
                topology,
                ..
            } => {
                if *dimensions != 7 {
                    self.errors
                        .push(format!("Manifold '{}' must be 7-dimensional", name));
                }
                if matches!(topology, Topology::PoincareBall) {
                    // Check curvature if possible
                }
                let sym = Symbol {
                    name: name.clone(),
                    sym_type: Type::ManifoldState,
                    is_mutable: false,
                };
                let _ = self.symbol_table.define(name.clone(), sym);
            }
            ASTNode::CrystalDecl { name, purity, .. } => {
                if *purity < 0.0 || *purity > 1.0 {
                    self.errors
                        .push(format!("Crystal '{}' purity must be 0-1", name));
                }
                let sym = Symbol {
                    name: name.clone(),
                    sym_type: Type::Int,
                    is_mutable: false,
                };
                let _ = self.symbol_table.define(name.clone(), sym);
            }
            ASTNode::EntropyDecl { name, purity, .. } => {
                if *purity < 0.99 {
                    self.errors
                        .push(format!("Entropy '{}' purity {} below 0.99", name, purity));
                }
                let sym = Symbol {
                    name: name.clone(),
                    sym_type: Type::EntropyPattern,
                    is_mutable: false,
                };
                let _ = self.symbol_table.define(name.clone(), sym);
            }
            ASTNode::VariableDecl {
                name,
                var_type,
                value,
                mutable,
                ..
            } => {
                self.check_node(value);
                let expr_type = self.infer_type(value);
                if let Some(expected) = var_type {
                    if *expected != expr_type {
                        self.errors.push(format!(
                            "Type mismatch for '{}': expected {:?}, found {:?}",
                            name, expected, expr_type
                        ));
                    }
                }
                let sym = Symbol {
                    name: name.clone(),
                    sym_type: expr_type,
                    is_mutable: *mutable,
                };
                if let Err(e) = self.symbol_table.define(name.clone(), sym) {
                    self.errors.push(e);
                }
            }
            ASTNode::FunctionDecl {
                name, params, body, ..
            } => {
                let _ = self.symbol_table.define(
                    name.clone(),
                    Symbol {
                        name: name.clone(),
                        sym_type: Type::Void,
                        is_mutable: false,
                    },
                );
                self.symbol_table.enter_scope();
                for param in params {
                    let _ = self.symbol_table.define(
                        param.name.clone(),
                        Symbol {
                            name: param.name.clone(),
                            sym_type: param.param_type.clone(),
                            is_mutable: false,
                        },
                    );
                }
                self.check_node(body);
                self.symbol_table.exit_scope();
            }
            ASTNode::Block { statements } => {
                self.symbol_table.enter_scope();
                for stmt in statements {
                    self.check_node(stmt);
                }
                self.symbol_table.exit_scope();
            }
            ASTNode::BinaryOp { left, right, .. } => {
                self.check_node(left);
                self.check_node(right);
            }
            ASTNode::UnaryOp { expr, .. } => {
                self.check_node(expr);
            }
            ASTNode::IfStatement {
                condition,
                then_block,
                else_block,
            } => {
                self.check_node(condition);
                self.check_node(then_block);
                if let Some(eb) = else_block {
                    self.check_node(eb);
                }
            }
            ASTNode::WhileLoop { condition, body } => {
                self.check_node(condition);
                self.check_node(body);
            }
            ASTNode::ReturnStatement { value } => {
                if let Some(v) = value {
                    self.check_node(v);
                }
            }
            ASTNode::FunctionCall { name, args } => {
                if self.symbol_table.lookup(name).is_none() {
                    self.errors.push(format!("Undefined function: {}", name));
                }
                for arg in args {
                    self.check_node(arg);
                }
            }
            ASTNode::Identifier(name) => {
                if self.symbol_table.lookup(name).is_none() {
                    self.errors.push(format!("Undefined identifier: {}", name));
                }
            }
            ASTNode::MatchStatement { target, arms } => {
                self.check_node(target);
                for arm in arms {
                    self.check_node(&arm.pattern);
                    self.check_node(&arm.body);
                }
            }
            _ => {}
        }
    }

    fn infer_type(&mut self, node: &ASTNode) -> Type {
        match node {
            ASTNode::IntLiteral(_) => Type::Int,
            ASTNode::FloatLiteral(_) => Type::Float,
            ASTNode::StringLiteral(_) => Type::String,
            ASTNode::BoolLiteral(_) => Type::Bool,
            ASTNode::Vector7DLiteral(_) => Type::Vector7D,
            ASTNode::PhiConstant
            | ASTNode::PhiInverse
            | ASTNode::S2Constant
            | ASTNode::LambdaConstant
            | ASTNode::PsiConstant => Type::Float,
            ASTNode::Identifier(name) => {
                if let Some(sym) = self.symbol_table.lookup(name) {
                    sym.sym_type.clone()
                } else {
                    Type::Void
                }
            }
            ASTNode::BinaryOp { left, .. } => self.infer_type(left),
            ASTNode::UnaryOp { expr, .. } => self.infer_type(expr),
            ASTNode::FunctionCall { .. } => Type::Void,
            _ => Type::Void,
        }
    }
}
