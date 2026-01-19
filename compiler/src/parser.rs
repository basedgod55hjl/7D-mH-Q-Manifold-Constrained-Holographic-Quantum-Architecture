use crate::lexer::Token;
use std::collections::HashMap;

// --- AST Nodes ---

#[derive(Debug, Clone, PartialEq)]
pub enum ASTNode {
    Program {
        attributes: Vec<Attribute>,
        items: Vec<ASTNode>,
    },

    // Declarations
    FunctionDecl {
        name: String,
        params: Vec<Parameter>,
        return_type: Type,
        body: Box<ASTNode>,
        attributes: Vec<FunctionAttribute>,
    },

    VariableDecl {
        name: String,
        var_type: Option<Type>,
        value: Box<ASTNode>,
        mutable: bool,
        qualifier: Qualifier,
    },

    ManifoldDecl {
        name: String,
        dimensions: u32,
        curvature: Box<ASTNode>,
        topology: Topology,
        attributes: Vec<Attribute>,
    },

    CrystalDecl {
        name: String,
        resolution: [usize; 7],
        encoding: Encoding,
        purity: f64,
        attributes: Vec<Attribute>,
    },

    EntropyDecl {
        name: String,
        source: String,
        modulation: Box<ASTNode>,
        purity: f64,
        attributes: Vec<Attribute>,
    },

    SovereigntyDecl {
        name: String,
        config: HashMap<String, String>,
    },

    TheoremDecl {
        name: String,
        statement: String,
        proof: Box<ASTNode>,
        qed: bool,
    },

    // Expressions
    BinaryOp {
        left: Box<ASTNode>,
        op: BinaryOperator,
        right: Box<ASTNode>,
    },

    UnaryOp {
        op: UnaryOperator,
        expr: Box<ASTNode>,
    },

    Identifier(String),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    Vector7DLiteral(Vec<f64>),

    // Native 7D constants
    PhiConstant,
    PhiInverse,
    S2Constant,
    LambdaConstant,
    PsiConstant,

    // 7D Specific Operations
    ManifoldProjection {
        input: Box<ASTNode>,
        target_manifold: String,
        curvature: f64,
    },

    HolographicFold {
        pattern1: Box<ASTNode>,
        pattern2: Box<ASTNode>,
        phases: usize,
    },

    QuantumSuperposition {
        states: Vec<ASTNode>,
        weights: Vec<f64>,
    },

    // Literals and Structures
    ObjectLiteral {
        name: String,
        fields: Vec<(String, ASTNode)>,
    },

    NamedArgument {
        name: String,
        value: Box<ASTNode>,
    },

    MemberAccess {
        object: Box<ASTNode>,
        member: String,
    },

    MethodCall {
        object: Box<ASTNode>,
        method: String,
        args: Vec<ASTNode>,
    },

    FunctionCall {
        name: String,
        args: Vec<ASTNode>,
    },

    // Control Flow
    Block {
        statements: Vec<ASTNode>,
    },

    IfStatement {
        condition: Box<ASTNode>,
        then_block: Box<ASTNode>,
        else_block: Option<Box<ASTNode>>,
    },

    WhileLoop {
        condition: Box<ASTNode>,
        body: Box<ASTNode>,
    },

    ForLoop {
        iterator: String,
        iterable: Box<ASTNode>,
        body: Box<ASTNode>,
    },

    ReturnStatement {
        value: Option<Box<ASTNode>>,
    },

    YieldStatement {
        value: Box<ASTNode>,
    },

    MatchStatement {
        target: Box<ASTNode>,
        arms: Vec<MatchArm>,
    },

    Import {
        path: Vec<String>,
        alias: Option<String>,
    },

    Module {
        name: String,
        items: Vec<ASTNode>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Box<ASTNode>,
    pub body: Box<ASTNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Int,
    Float,
    String,
    Bool,
    Vector7D,
    QuantumState,
    ManifoldState,
    EntropyPattern,
    Void,
    Custom(String),
    Reference(Box<Type>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Topology {
    PoincareBall,
    Hyperbolic,
    Euclidean,
    CrystalLattice,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Encoding {
    InterferencePatterns,
    PhaseShift,
    AmplitudeModulation,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    Sovereignty(String),
    Cortex,
    Manifold,
    Crystal,
    Entropy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionAttribute {
    QuantumLogic,
    QuantumCortex,
    Native,
    Async,
    Pure,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Qualifier {
    None,
    Manifold,
    Crystal,
    Entropy,
    Hologram,
    Quantum,
    Arrow,
    Assign,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    TensorProduct, // ⊗
    Superposition, // ⊕
    HoloFold,      // ⊙
    Concat,        // ..
    Arrow,         // ->
    Assign,        // =
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Not,
    Neg,
    Ref,
    Deref,
    Project7D, // ⑦
}

// --- Parser Implementation ---

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
    current_token: Token,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        let current_token = tokens.get(0).cloned().unwrap_or(Token::Eof);
        Parser {
            tokens,
            position: 0,
            current_token,
        }
    }

    fn advance(&mut self) {
        self.position += 1;
        self.current_token = self
            .tokens
            .get(self.position)
            .cloned()
            .unwrap_or(Token::Eof);
    }

    fn peek(&self) -> Token {
        self.tokens
            .get(self.position + 1)
            .cloned()
            .unwrap_or(Token::Eof)
    }

    fn expect(&mut self, expected: Token) -> Result<(), String> {
        if self.current_token == expected {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "Parse Error: Expected {:?}, found {:?} at position {}. Previous token: {:?}",
                expected,
                self.current_token,
                self.position,
                self.tokens.get(self.position.saturating_sub(1))
            ))
        }
    }

    pub fn parse(&mut self) -> Result<ASTNode, String> {
        let mut items = Vec::new();
        let program_attributes = Vec::new(); // Unused for now

        while self.current_token != Token::Eof {
            items.push(self.parse_item()?);
        }

        Ok(ASTNode::Program {
            attributes: program_attributes,
            items,
        })
    }

    fn parse_item(&mut self) -> Result<ASTNode, String> {
        let mut attributes = Vec::new();

        while let Token::Attribute(attr) = &self.current_token {
            let attr_name = attr.clone();
            self.advance();
            if attr_name == "sovereignty" {
                if let Token::Identifier(name) = &self.current_token {
                    let name = name.clone();
                    self.advance();
                    return self.parse_sovereignty_decl(name);
                }
            }
            attributes.push(match attr_name.as_str() {
                "manifold" => Attribute::Manifold,
                "crystal" => Attribute::Crystal,
                "entropy" => Attribute::Entropy,
                "cortex" => Attribute::Cortex,
                _ => Attribute::Sovereignty(attr_name),
            });
        }

        match &self.current_token {
            Token::Manifold | Token::Crystal | Token::Entropy | Token::Hologram => {
                self.parse_variable_declaration(None)
            }
            Token::Theorem => self.parse_theorem_decl(),
            Token::Fn => self.parse_function_with_attributes(Vec::new()),
            Token::Quantum | Token::QuantumLogic | Token::QuantumCortex => {
                self.parse_function_with_qualifier()
            }
            Token::Import => self.parse_import(),
            Token::Module => self.parse_module(),
            Token::Identifier(_) | Token::Let | Token::Mut => {
                if let Some(attr) = attributes.last() {
                    match attr {
                        Attribute::Manifold => self.parse_manifold_decl(attributes),
                        Attribute::Crystal => self.parse_crystal_decl(attributes),
                        Attribute::Entropy => self.parse_entropy_decl(attributes),
                        _ => self.parse_variable_declaration(None),
                    }
                } else {
                    self.parse_variable_declaration(None)
                }
            }
            _ => {
                if self.is_type_token(&self.current_token) {
                    self.parse_variable_declaration(None)
                } else {
                    Err(format!(
                        "Unexpected token in item: {:?}",
                        self.current_token
                    ))
                }
            }
        }
    }

    fn is_type_token(&self, token: &Token) -> bool {
        match token {
            Token::I8
            | Token::U8
            | Token::I16
            | Token::U16
            | Token::I32
            | Token::U32
            | Token::I64
            | Token::U64
            | Token::F32
            | Token::F64
            | Token::Bool
            | Token::Vector7D
            | Token::QuantumState
            | Token::Pattern
            | Token::WaveFunction
            | Token::ManifoldType
            | Token::CrystalType
            | Token::HologramType
            | Token::QuantumLogic
            | Token::QuantumCortex => true,
            Token::Identifier(id) => [
                "Manifold",
                "Crystal",
                "Hologram",
                "QuantumState",
                "Vector7D",
                "Proof",
            ]
            .contains(&id.as_str()),
            _ => false,
        }
    }

    fn parse_sovereignty_decl(&mut self, name: String) -> Result<ASTNode, String> {
        self.expect(Token::LeftBrace)?;
        let mut config = HashMap::new();

        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            if let Token::Identifier(key) = &self.current_token {
                let key = key.clone();
                self.advance();
                self.expect(Token::Colon)?;
                let value = match &self.current_token {
                    Token::StringLiteral(s) => s.clone(),
                    Token::IntLiteral(i) => i.to_string(),
                    Token::FloatLiteral(f) => f.to_string(),
                    Token::Identifier(id) => id.clone(),
                    _ => {
                        return Err(format!(
                            "Expected config value, found {:?}",
                            self.current_token
                        ))
                    }
                };
                self.advance();
                config.insert(key, value);
                if self.current_token == Token::Comma {
                    self.advance();
                }
            } else {
                break;
            }
        }

        self.expect(Token::RightBrace)?;
        Ok(ASTNode::SovereigntyDecl { name, config })
    }

    fn parse_manifold_decl(&mut self, _attrs: Vec<Attribute>) -> Result<ASTNode, String> {
        if self.current_token == Token::Manifold {
            self.advance();
        }
        let name = if let Token::Identifier(n) = &self.current_token {
            let name = n.clone();
            self.advance();
            name
        } else {
            return Err("Expected manifold name".to_string());
        };
        self.expect(Token::LeftBrace)?;
        let mut dimensions = 7;
        let mut curvature = Box::new(ASTNode::FloatLiteral(-1.0));
        let mut topology = Topology::PoincareBall;
        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            if let Token::Identifier(prop) = &self.current_token {
                let prop_name = prop.clone();
                self.advance();
                self.expect(Token::Colon)?;
                match prop_name.as_str() {
                    "dimensions" => {
                        if let Token::IntLiteral(n) = self.current_token {
                            dimensions = n as u32;
                        }
                        self.advance();
                    }
                    "curvature" => {
                        curvature = Box::new(self.parse_expression()?);
                    }
                    "topology" => {
                        if let Token::Identifier(top) = &self.current_token {
                            topology = match top.as_str() {
                                "POINCARE_BALL" => Topology::PoincareBall,
                                "HYPERBOLIC" => Topology::Hyperbolic,
                                "EUCLIDEAN" => Topology::Euclidean,
                                _ => Topology::PoincareBall,
                            };
                            self.advance();
                        } else {
                            self.advance();
                        }
                    }
                    _ => {
                        self.advance();
                    }
                }
                if self.current_token == Token::Comma {
                    self.advance();
                }
            } else {
                break;
            }
        }
        self.expect(Token::RightBrace)?;
        Ok(ASTNode::ManifoldDecl {
            name,
            dimensions,
            curvature,
            topology,
            attributes: _attrs,
        })
    }

    fn parse_crystal_decl(&mut self, _attrs: Vec<Attribute>) -> Result<ASTNode, String> {
        if self.current_token == Token::Crystal {
            self.advance();
        }
        let name = if let Token::Identifier(n) = &self.current_token {
            let name = n.clone();
            self.advance();
            name
        } else {
            return Err("Expected crystal name".to_string());
        };
        self.expect(Token::LeftBrace)?;
        let mut resolution = [1usize; 7];
        let mut encoding = Encoding::InterferencePatterns;
        let mut purity = 1.0;
        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            if let Token::Identifier(prop) = &self.current_token {
                let prop_name = prop.clone();
                self.advance();
                self.expect(Token::Colon)?;
                match prop_name.as_str() {
                    "resolution" => {
                        if self.current_token == Token::LeftBracket {
                            self.advance();
                            for i in 0..7 {
                                if let Token::IntLiteral(n) = self.current_token {
                                    resolution[i] = n as usize;
                                    self.advance();
                                    if self.current_token == Token::Comma {
                                        self.advance();
                                    }
                                }
                            }
                            self.expect(Token::RightBracket)?;
                        }
                    }
                    "encoding" => {
                        if let Token::Identifier(e) = &self.current_token {
                            encoding = match e.as_str() {
                                "INTERFERENCE_PATTERNS" => Encoding::InterferencePatterns,
                                "PHASE_SHIFT" => Encoding::PhaseShift,
                                _ => Encoding::InterferencePatterns,
                            };
                            self.advance();
                        } else {
                            self.advance();
                        }
                    }
                    "purity" | "coherence" => {
                        if let Token::FloatLiteral(f) = self.current_token {
                            purity = f;
                        }
                        self.advance();
                    }
                    _ => {
                        self.advance();
                    }
                }
                if self.current_token == Token::Comma {
                    self.advance();
                }
            } else {
                break;
            }
        }
        self.expect(Token::RightBrace)?;
        Ok(ASTNode::CrystalDecl {
            name,
            resolution,
            encoding,
            purity,
            attributes: _attrs,
        })
    }

    fn parse_entropy_decl(&mut self, _attrs: Vec<Attribute>) -> Result<ASTNode, String> {
        if self.current_token == Token::Entropy {
            self.advance();
        }
        let name = if let Token::Identifier(n) = &self.current_token {
            let name = n.clone();
            self.advance();
            name
        } else {
            return Err("Expected entropy name".to_string());
        };
        self.expect(Token::LeftBrace)?;
        let mut source = String::new();
        let mut modulation = Box::new(ASTNode::FloatLiteral(1.0));
        let mut purity = 1.0;
        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            if let Token::Identifier(prop) = &self.current_token {
                let prop_name = prop.clone();
                self.advance();
                self.expect(Token::Colon)?;
                match prop_name.as_str() {
                    "source" => {
                        if let Token::StringLiteral(s) = &self.current_token {
                            source = s.clone();
                        }
                        self.advance();
                    }
                    "modulation" => {
                        modulation = Box::new(self.parse_expression()?);
                    }
                    "purity" => {
                        if let Token::FloatLiteral(f) = self.current_token {
                            purity = f;
                        }
                        self.advance();
                    }
                    _ => {
                        self.advance();
                    }
                }
                if self.current_token == Token::Comma {
                    self.advance();
                }
            } else {
                break;
            }
        }
        self.expect(Token::RightBrace)?;
        Ok(ASTNode::EntropyDecl {
            name,
            source,
            modulation,
            purity,
            attributes: _attrs,
        })
    }

    fn parse_theorem_decl(&mut self) -> Result<ASTNode, String> {
        self.expect(Token::Theorem)?;
        let name = if let Token::Identifier(n) = &self.current_token {
            let n = n.clone();
            self.advance();
            n
        } else {
            return Err("Expected theorem name".to_string());
        };
        self.expect(Token::LeftParen)?;
        self.expect(Token::RightParen)?;
        if self.current_token == Token::Arrow {
            self.advance();
            let _ret_type = self.parse_type()?;
        }
        self.expect(Token::LeftBrace)?;
        let mut statement = String::new();
        let mut proof = Box::new(ASTNode::Block { statements: vec![] });
        let mut qed = false;
        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            let prop_name = match &self.current_token {
                Token::Identifier(n) => n.clone(),
                Token::Proof => "proof".to_string(), // Handle keyword proof as identifier key
                _ => break,
            };
            self.advance();
            self.expect(Token::Colon)?;
            match prop_name.as_str() {
                "statement" => {
                    if let Token::StringLiteral(s) = &self.current_token {
                        statement = s.clone();
                    }
                    self.advance();
                }
                "proof" => {
                    proof = Box::new(self.parse_block()?);
                }
                "qed" => {
                    if let Token::BoolLiteral(b) = self.current_token {
                        qed = b;
                    }
                    self.advance();
                }
                _ => {
                    self.advance();
                }
            }
            if self.current_token == Token::Comma {
                self.advance();
            }
        }
        self.expect(Token::RightBrace)?;
        Ok(ASTNode::TheoremDecl {
            name,
            statement,
            proof,
            qed,
        })
    }

    fn parse_function_with_qualifier(&mut self) -> Result<ASTNode, String> {
        let mut attributes = Vec::new();
        let mut qual = Qualifier::None;

        loop {
            match &self.current_token {
                Token::Quantum => {
                    self.advance();
                    qual = Qualifier::Quantum;
                }
                Token::QuantumLogic => {
                    self.advance();
                    attributes.push(FunctionAttribute::QuantumLogic);
                    qual = Qualifier::Quantum;
                }
                Token::QuantumCortex => {
                    self.advance();
                    attributes.push(FunctionAttribute::QuantumCortex);
                    qual = Qualifier::Quantum;
                }
                _ => break,
            }
        }

        if self.current_token == Token::Let {
            return self.parse_variable_declaration(Some(qual));
        }

        if self.current_token == Token::Fn {
            self.advance();
        }

        self.parse_function_with_attributes_inner(attributes)
    }

    fn parse_function_with_attributes(
        &mut self,
        attrs: Vec<FunctionAttribute>,
    ) -> Result<ASTNode, String> {
        self.expect(Token::Fn)?;
        self.parse_function_with_attributes_inner(attrs)
    }

    fn parse_function_with_attributes_inner(
        &mut self,
        attrs: Vec<FunctionAttribute>,
    ) -> Result<ASTNode, String> {
        let name = match &self.current_token {
            Token::Identifier(n) => {
                let n = n.clone();
                self.advance();
                n
            }
            Token::Project7D => {
                self.advance();
                "⑦".to_string()
            }
            _ => {
                return Err(format!(
                    "Expected function name, found {:?}",
                    self.current_token
                ));
            }
        };
        self.expect(Token::LeftParen)?;
        let params = self.parse_parameter_list()?;
        self.expect(Token::RightParen)?;
        let mut return_type = Type::Void;
        if self.current_token == Token::Arrow {
            self.advance();
            return_type = self.parse_type()?;
        }
        let body = Box::new(self.parse_block()?);
        Ok(ASTNode::FunctionDecl {
            name,
            params,
            return_type,
            body,
            attributes: attrs,
        })
    }

    fn parse_parameter_list(&mut self) -> Result<Vec<Parameter>, String> {
        let mut params = Vec::new();
        while self.current_token != Token::RightParen {
            let name = if let Token::Identifier(n) = &self.current_token {
                let n = n.clone();
                self.advance();
                n
            } else {
                return Err("Expected parameter name".to_string());
            };
            self.expect(Token::Colon)?;
            let param_type = self.parse_type()?;
            params.push(Parameter { name, param_type });
            if self.current_token == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        Ok(params)
    }

    fn parse_type(&mut self) -> Result<Type, String> {
        let t = match &self.current_token {
            Token::I8
            | Token::U8
            | Token::I16
            | Token::U16
            | Token::I32
            | Token::U32
            | Token::I64
            | Token::U64 => Type::Int,
            Token::F32 | Token::F64 => Type::Float,
            Token::Bool => Type::Bool,
            Token::Vector7D => Type::Vector7D,
            Token::QuantumState => Type::QuantumState,
            Token::ManifoldType => Type::ManifoldState,
            Token::Pattern => Type::EntropyPattern,
            Token::Proof => Type::Bool, // Handle Token::Proof -> Type::Bool
            Token::Ampersand => {
                self.advance();
                let inner = self.parse_type()?;
                return Ok(Type::Reference(Box::new(inner)));
            }
            Token::Identifier(id) => {
                match id.as_str() {
                    "Manifold" => Type::ManifoldState,
                    "Crystal" => Type::Int, // Lattice ID?
                    "QuantumState" => Type::QuantumState,
                    "Vector7D" => Type::Vector7D,
                    "Proof" => Type::Bool,
                    _ => Type::Custom(id.clone()),
                }
            }
            _ => return Err(format!("Expected type, found {:?}", self.current_token)),
        };
        self.advance();
        Ok(t)
    }

    fn parse_variable_declaration(
        &mut self,
        initial_qualifier: Option<Qualifier>,
    ) -> Result<ASTNode, String> {
        let mut qualifier = initial_qualifier.unwrap_or(Qualifier::None);
        if qualifier == Qualifier::None {
            qualifier = match &self.current_token {
                Token::Manifold => {
                    self.advance();
                    Qualifier::Manifold
                }
                Token::Crystal => {
                    self.advance();
                    Qualifier::Crystal
                }
                Token::Entropy => {
                    self.advance();
                    Qualifier::Entropy
                }
                Token::Hologram => {
                    self.advance();
                    Qualifier::Hologram
                }
                Token::Quantum => {
                    self.advance();
                    Qualifier::Quantum
                }
                _ => Qualifier::None,
            };
        }
        if self.current_token == Token::Let {
            self.advance();
        }
        let mut mutable = false;
        if self.current_token == Token::Mut {
            self.advance();
            mutable = true;
        }

        // Complex name/type parsing logic to match identifier vs Type Identifier
        let mut var_type = None;
        let name;

        // Peek ahead to see if we have Type Name
        let is_typed_decl = if let Token::Identifier(_) = &self.current_token {
            if let Token::Identifier(_) = self.peek() {
                true
            } else {
                false
            } // Identifier Identifier
        } else if self.is_type_token(&self.current_token) {
            if let Token::Identifier(_) = self.peek() {
                true
            } else {
                false
            } // Type Identifier (e.g. Vector7D name)
        } else {
            false
        };

        if is_typed_decl {
            var_type = Some(self.parse_type()?);
            if let Token::Identifier(n) = &self.current_token {
                name = n.clone();
                self.advance();
            } else {
                return Err("Expected variable name".to_string());
            }
        } else {
            // Un-typed or type inferred later
            if let Token::Identifier(n) = &self.current_token {
                name = n.clone();
                self.advance();
            } else {
                return Err(format!(
                    "Expected variable name, found {:?}",
                    self.current_token
                ));
            }
        }

        if var_type.is_none() && self.current_token == Token::Colon {
            self.advance();
            var_type = Some(self.parse_type()?);
        }
        self.expect(Token::Assign)?;
        let value = Box::new(self.parse_expression()?);
        if self.current_token == Token::Semicolon {
            self.advance();
        }
        Ok(ASTNode::VariableDecl {
            name,
            var_type,
            value,
            mutable,
            qualifier,
        })
    }

    fn parse_block(&mut self) -> Result<ASTNode, String> {
        self.expect(Token::LeftBrace)?;
        let mut statements = Vec::new();
        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            statements.push(self.parse_statement()?);
        }
        self.expect(Token::RightBrace)?;
        Ok(ASTNode::Block { statements })
    }

    fn parse_statement(&mut self) -> Result<ASTNode, String> {
        match &self.current_token {
            Token::Let
            | Token::Mut
            | Token::Manifold
            | Token::Crystal
            | Token::Entropy
            | Token::Quantum
            | Token::Hologram => self.parse_variable_declaration(None),
            Token::If => self.parse_if_statement(),
            Token::While => self.parse_while_loop(),
            Token::Return => self.parse_return_statement(),
            Token::Yield => self.parse_yield_statement(),
            Token::Match => self.parse_match_statement(),
            _ => {
                let expr = self.parse_expression()?;
                if self.current_token == Token::Semicolon {
                    self.advance();
                }
                Ok(expr)
            }
        }
    }

    fn parse_if_statement(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let condition = Box::new(self.parse_expression()?);
        let then_block = Box::new(self.parse_block()?);
        let mut else_block = None;
        if self.current_token == Token::Else {
            self.advance();
            else_block = Some(Box::new(if self.current_token == Token::If {
                self.parse_if_statement()?
            } else {
                self.parse_block()?
            }));
        }
        Ok(ASTNode::IfStatement {
            condition,
            then_block,
            else_block,
        })
    }

    fn parse_while_loop(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let condition = Box::new(self.parse_expression()?);
        let body = Box::new(self.parse_block()?);
        Ok(ASTNode::WhileLoop { condition, body })
    }

    fn parse_return_statement(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let mut value = None;
        if self.current_token != Token::Semicolon && self.current_token != Token::RightBrace {
            value = Some(Box::new(self.parse_expression()?));
        }
        if self.current_token == Token::Semicolon {
            self.advance();
        }
        Ok(ASTNode::ReturnStatement { value })
    }

    fn parse_yield_statement(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let value = Box::new(self.parse_expression()?);
        if self.current_token == Token::Semicolon {
            self.advance();
        }
        Ok(ASTNode::YieldStatement { value })
    }

    fn parse_match_statement(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let target = Box::new(self.parse_expression()?);
        self.expect(Token::LeftBrace)?;
        let mut arms = Vec::new();
        while self.current_token != Token::RightBrace {
            let pattern = Box::new(self.parse_expression()?);
            self.expect(Token::FatArrow)?;
            let body = Box::new(if self.current_token == Token::LeftBrace {
                self.parse_block()?
            } else {
                self.parse_statement()?
            });
            arms.push(MatchArm { pattern, body });
            if self.current_token == Token::Comma {
                self.advance();
            }
        }
        self.advance();
        Ok(ASTNode::MatchStatement { target, arms })
    }

    fn parse_expression(&mut self) -> Result<ASTNode, String> {
        self.parse_binary_op(0)
    }

    fn parse_binary_op(&mut self, precedence: u8) -> Result<ASTNode, String> {
        let mut left = self.parse_unary()?;
        while let Some(op) = self.get_binary_op(&self.current_token) {
            let op_precedence = self.get_precedence(&op);
            if op_precedence < precedence {
                break;
            }
            self.advance();
            let right = self.parse_binary_op(op_precedence + 1)?;
            left = ASTNode::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn get_binary_op(&self, token: &Token) -> Option<BinaryOperator> {
        match token {
            Token::Plus => Some(BinaryOperator::Add),
            Token::Minus => Some(BinaryOperator::Sub),
            Token::Star => Some(BinaryOperator::Mul),
            Token::Slash => Some(BinaryOperator::Div),
            Token::Eq => Some(BinaryOperator::Eq),
            Token::Ne => Some(BinaryOperator::Ne),
            Token::Lt => Some(BinaryOperator::Lt),
            Token::Gt => Some(BinaryOperator::Gt),
            Token::Le => Some(BinaryOperator::Le),
            Token::Ge => Some(BinaryOperator::Ge),
            Token::And => Some(BinaryOperator::And),
            Token::Or => Some(BinaryOperator::Or),
            Token::TensorProduct => Some(BinaryOperator::TensorProduct),
            Token::Superposition => Some(BinaryOperator::Superposition),
            Token::HoloFold => Some(BinaryOperator::HoloFold),
            Token::Arrow => Some(BinaryOperator::Arrow),
            Token::Assign => Some(BinaryOperator::Assign),
            _ => None,
        }
    }

    fn get_precedence(&self, op: &BinaryOperator) -> u8 {
        match op {
            BinaryOperator::Assign => 1,
            BinaryOperator::Or => 2,
            BinaryOperator::And => 3,
            BinaryOperator::Eq | BinaryOperator::Ne => 4,
            BinaryOperator::Lt | BinaryOperator::Gt | BinaryOperator::Le | BinaryOperator::Ge => 5,
            BinaryOperator::Add | BinaryOperator::Sub => 6,
            BinaryOperator::Mul | BinaryOperator::Div => 7,
            BinaryOperator::TensorProduct
            | BinaryOperator::Superposition
            | BinaryOperator::HoloFold => 7,
            BinaryOperator::Arrow => 1,
            _ => 0,
        }
    }

    fn parse_unary(&mut self) -> Result<ASTNode, String> {
        match &self.current_token {
            Token::Not => {
                self.advance();
                Ok(ASTNode::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            Token::Minus => {
                self.advance();
                Ok(ASTNode::UnaryOp {
                    op: UnaryOperator::Neg,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            Token::Project7D => {
                self.advance();
                Ok(ASTNode::UnaryOp {
                    op: UnaryOperator::Project7D,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            Token::Ampersand => {
                self.advance();
                Ok(ASTNode::UnaryOp {
                    op: UnaryOperator::Ref,
                    expr: Box::new(self.parse_unary()?),
                })
            }
            Token::HoloFold => {
                self.advance();
                let left = self.parse_expression()?;
                self.expect(Token::Comma)?;
                let right = self.parse_expression()?;
                Ok(ASTNode::BinaryOp {
                    left: Box::new(left),
                    op: BinaryOperator::HoloFold,
                    right: Box::new(right),
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<ASTNode, String> {
        let mut expr = self.parse_primary()?;
        loop {
            match &self.current_token {
                Token::LeftParen => {
                    self.advance();
                    let args = self.parse_argument_list()?;
                    self.expect(Token::RightParen)?;
                    if let ASTNode::Identifier(name) = expr {
                        expr = ASTNode::FunctionCall { name, args };
                    } else if let ASTNode::MemberAccess { object, member } = expr {
                        expr = ASTNode::MethodCall {
                            object,
                            method: member,
                            args,
                        };
                    } else {
                        expr = ASTNode::FunctionCall {
                            name: format!("{:?}", expr),
                            args,
                        };
                    }
                }
                Token::Dot => {
                    self.advance();
                    match &self.current_token {
                        Token::Identifier(member) => {
                            let member = member.clone();
                            self.advance();
                            expr = ASTNode::MemberAccess {
                                object: Box::new(expr),
                                member,
                            };
                        }
                        Token::Star => {
                            self.advance();
                            expr = ASTNode::MemberAccess {
                                object: Box::new(expr),
                                member: "*".to_string(),
                            };
                        }
                        _ => return Err("Expected member after .".to_string()),
                    }
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_argument_list(&mut self) -> Result<Vec<ASTNode>, String> {
        let mut args = Vec::new();
        while self.current_token != Token::RightParen {
            if let Token::Identifier(name) = &self.current_token {
                if let Some(Token::Colon) = self.tokens.get(self.position + 1) {
                    let name = name.clone();
                    self.advance();
                    self.advance();
                    let value = Box::new(self.parse_expression()?);
                    args.push(ASTNode::NamedArgument { name, value });
                } else {
                    args.push(self.parse_expression()?);
                }
            } else {
                args.push(self.parse_expression()?);
            }
            if self.current_token == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        Ok(args)
    }

    fn parse_primary(&mut self) -> Result<ASTNode, String> {
        match &self.current_token {
            Token::IntLiteral(n) => {
                let v = *n;
                self.advance();
                Ok(ASTNode::IntLiteral(v))
            }
            Token::FloatLiteral(n) => {
                let v = *n;
                self.advance();
                Ok(ASTNode::FloatLiteral(v))
            }
            Token::StringLiteral(s) => {
                let v = s.clone();
                self.advance();
                Ok(ASTNode::StringLiteral(v))
            }
            Token::BoolLiteral(b) => {
                let v = *b;
                self.advance();
                Ok(ASTNode::BoolLiteral(v))
            }
            Token::PhiConstant => {
                self.advance();
                Ok(ASTNode::PhiConstant)
            }
            Token::PhiInverse => {
                self.advance();
                Ok(ASTNode::PhiInverse)
            }
            Token::S2Constant => {
                self.advance();
                Ok(ASTNode::S2Constant)
            }
            Token::LambdaConstant => {
                self.advance();
                Ok(ASTNode::LambdaConstant)
            }
            Token::PsiConstant => {
                self.advance();
                Ok(ASTNode::PsiConstant)
            }
            token @ (Token::Identifier(_)
            | Token::Proof
            | Token::Consciousness
            | Token::Infinite
            | Token::Eternity) => {
                let name = match token {
                    Token::Identifier(id) => id.clone(),
                    Token::Proof => "Proof".to_string(),
                    Token::Consciousness => "Consciousness".to_string(),
                    Token::Infinite => "INFINITE".to_string(),
                    Token::Eternity => "ETERNITY".to_string(),
                    _ => unreachable!(),
                };
                self.advance();
                // STRICT CHECK: Only trigger ObjectLiteral if the identifier is followed by { AND the identifier is Capitalized
                if self.current_token == Token::LeftBrace
                    && name.chars().next().map_or(false, |c| c.is_uppercase())
                {
                    self.parse_object_literal(name)
                } else {
                    Ok(ASTNode::Identifier(name))
                }
            }
            Token::LeftParen => {
                self.advance();
                let expr = self.parse_expression()?;
                self.expect(Token::RightParen)?;
                Ok(expr)
            }
            Token::LeftBracket => self.parse_vector_literal(),
            _ => {
                eprintln!(
                    "DEBUG PRE-ERROR: Token={:?} Pos={}",
                    self.current_token, self.position
                );
                Err(format!(
                    "Unexpected token in primary: {:?} at position {}",
                    self.current_token, self.position
                ))
            }
        }
    }

    fn parse_vector_literal(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let mut values = Vec::new();
        while self.current_token != Token::RightBracket {
            match self.parse_expression()? {
                ASTNode::IntLiteral(n) => values.push(n as f64),
                ASTNode::FloatLiteral(n) => values.push(n),
                ASTNode::PhiConstant => values.push(1.618033988749895),
                ASTNode::PhiInverse => values.push(0.618033988749895),
                ASTNode::S2Constant => values.push(0.01),
                _ => {}
            }
            if self.current_token == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        self.expect(Token::RightBracket)?;
        Ok(ASTNode::Vector7DLiteral(values))
    }

    fn parse_object_literal(&mut self, name: String) -> Result<ASTNode, String> {
        self.expect(Token::LeftBrace)?;
        let mut fields = Vec::new();
        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            // STRICT FIELD KEY CHECK: Only identifier
            let field_name = if let Token::Identifier(n) = &self.current_token {
                n.clone()
            } else {
                return Err(format!(
                    "Expected field name in object literal, found {:?}",
                    self.current_token
                ));
            };
            self.advance();
            self.expect(Token::Colon)?;
            let value = self.parse_expression()?;
            fields.push((field_name, value));
            if self.current_token == Token::Comma {
                self.advance();
            }
        }
        self.expect(Token::RightBrace)?;
        Ok(ASTNode::ObjectLiteral { name, fields })
    }

    fn parse_import(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let mut path = Vec::new();
        loop {
            match &self.current_token {
                Token::Identifier(n) => {
                    path.push(n.clone());
                    self.advance();
                }
                Token::Star => {
                    path.push("*".to_string());
                    self.advance();
                    break;
                }
                _ => {
                    let s = format!("{:?}", self.current_token).to_lowercase();
                    path.push(s);
                    self.advance();
                }
            }
            if self.current_token == Token::Dot {
                self.advance();
            } else if self.current_token == Token::ColonColon {
                self.advance();
            } else {
                break;
            }
        }
        if self.current_token == Token::Semicolon {
            self.advance();
        }
        Ok(ASTNode::Import { path, alias: None })
    }

    fn parse_module(&mut self) -> Result<ASTNode, String> {
        self.advance();
        let name = if let Token::Identifier(n) = &self.current_token {
            n.clone()
        } else {
            "Module".to_string()
        };
        self.advance();
        self.expect(Token::LeftBrace)?;
        let mut items = Vec::new();
        while self.current_token != Token::RightBrace && self.current_token != Token::Eof {
            items.push(self.parse_item()?);
        }
        self.expect(Token::RightBrace)?;
        Ok(ASTNode::Module { name, items })
    }
}
