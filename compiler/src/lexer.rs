// File: lexer.rs
// 7D Crystal Lexer - Tokenization of 7D-MHQL source code
// Discovered by Sir Charles Spikes, December 24, 2025

use std::fmt;

/// Token types in 7D Crystal language
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords - Declarations
    Manifold,
    Crystal,
    Hologram,
    Quantum,
    Entropy,
    Theorem,
    Proof,
    Module,
    Import,
    Export,

    // Keywords - Control Flow
    If,
    Else,
    While,
    For,
    Match,
    Break,
    Continue,
    Return,
    Yield,

    // Keywords - Qualifiers
    Sovereignty,   // @sovereignty
    QuantumCortex, // quantum cortex
    QuantumLogic,  // quantum logic
    ManifoldConst, // manifold constant
    Let,
    Mut,
    Fn,

    // Primitive Types
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,

    // 7D Types
    HyperbolicReal,
    Vector7D,
    Matrix7D,
    Tensor7D,
    ManifoldType,
    CrystalType,
    HologramType,
    QuantumState,
    Pattern,
    WaveFunction,
    Consciousness,

    // Operators - Mathematical
    Plus,    // +
    Minus,   // -
    Star,    // *
    Slash,   // /
    Percent, // %

    // Operators - 7D Specific
    TensorProduct, // ⊗
    Superposition, // ⊕
    HoloFold,      // ⊙
    Project7D,     // ⑦

    // Operators - Comparison
    Eq, // ==
    Ne, // !=
    Lt, // <
    Gt, // >
    Le, // <=
    Ge, // >=

    // Operators - Logical
    And, // &&
    Or,  // ||
    Not, // !

    // Assignment
    Assign, // =

    // Delimiters
    LeftParen,    // (
    RightParen,   // )
    LeftBrace,    // {
    RightBrace,   // }
    LeftBracket,  // [
    RightBracket, // ]
    Semicolon,    // ;
    Comma,        // ,
    Dot,          // .
    Colon,        // :
    ColonColon,   // ::
    Arrow,        // ->
    FatArrow,     // =>
    Ampersand,    // &

    // Constants
    PhiConstant,    // Φ
    PhiInverse,     // Φ⁻¹
    S2Constant,     // S²
    Infinity,       // ∞
    LambdaConstant, // Λ
    PsiConstant,    // Ψ
    Infinite,
    Eternity,

    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Identifiers
    Identifier(String),

    // Attributes
    Attribute(String), // @sovereignty, @manifold, etc.

    // Special
    Eof,
    Unknown(char),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Manifold => write!(f, "manifold"),
            Token::Quantum => write!(f, "quantum"),
            Token::PhiConstant => write!(f, "Φ"),
            Token::Identifier(s) => write!(f, "identifier({})", s),
            Token::IntLiteral(n) => write!(f, "int({})", n),
            Token::FloatLiteral(n) => write!(f, "float({})", n),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Lexer for 7D Crystal language
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    current_char: Option<char>,
    line: usize,
    column: usize,
}

impl Lexer {
    pub fn new(input: String) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let current_char = chars.get(0).copied();

        Lexer {
            input: chars,
            position: 0,
            current_char,
            line: 1,
            column: 1,
        }
    }

    /// Advance to next character
    fn advance(&mut self) {
        if self.current_char == Some('\n') {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }

        self.position += 1;
        self.current_char = self.input.get(self.position).copied();
    }

    /// Peek at next character without advancing
    fn peek(&self, offset: usize) -> Option<char> {
        self.input.get(self.position + offset).copied()
    }

    /// Skip whitespace
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip single-line comment (//)
    fn skip_comment(&mut self) {
        while let Some(ch) = self.current_char {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    /// Skip multi-line comment (/* ... */)
    fn skip_multiline_comment(&mut self) {
        self.advance(); // Skip '/'
        self.advance(); // Skip '*'

        while let Some(ch) = self.current_char {
            if ch == '*' && self.peek(1) == Some('/') {
                self.advance(); // Skip '*'
                self.advance(); // Skip '/'
                break;
            }
            self.advance();
        }
    }

    /// Read number (integer or float)
    fn read_number(&mut self) -> Token {
        let mut num_str = String::new();
        let mut is_float = false;

        // Handle hex (0x), binary (0b)
        if self.current_char == Some('0') {
            if self.peek(1) == Some('x') || self.peek(1) == Some('X') {
                return self.read_hex();
            } else if self.peek(1) == Some('b') || self.peek(1) == Some('B') {
                return self.read_binary();
            }
        }

        // Read digits
        while let Some(ch) = self.current_char {
            if ch.is_numeric() || ch == '.' || ch == '_' {
                if ch == '.' {
                    if is_float {
                        break; // Second dot, stop
                    }
                    is_float = true;
                }
                if ch != '_' {
                    num_str.push(ch);
                }
                self.advance();
            } else {
                break;
            }
        }

        // Handle scientific notation
        if let Some('e') | Some('E') = self.current_char {
            is_float = true;
            num_str.push('e');
            self.advance();

            if let Some('+') | Some('-') = self.current_char {
                num_str.push(self.current_char.unwrap());
                self.advance();
            }

            while let Some(ch) = self.current_char {
                if ch.is_numeric() {
                    num_str.push(ch);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        if is_float {
            Token::FloatLiteral(num_str.parse().unwrap_or(0.0))
        } else {
            Token::IntLiteral(num_str.parse().unwrap_or(0))
        }
    }

    /// Read hexadecimal number
    fn read_hex(&mut self) -> Token {
        self.advance(); // Skip '0'
        self.advance(); // Skip 'x'

        let mut hex_str = String::from("0x");
        while let Some(ch) = self.current_char {
            if ch.is_ascii_hexdigit() || ch == '_' {
                if ch != '_' {
                    hex_str.push(ch);
                }
                self.advance();
            } else {
                break;
            }
        }

        let value = i64::from_str_radix(&hex_str[2..], 16).unwrap_or(0);
        Token::IntLiteral(value)
    }

    /// Read binary number
    fn read_binary(&mut self) -> Token {
        self.advance(); // Skip '0'
        self.advance(); // Skip 'b'

        let mut bin_str = String::new();
        while let Some(ch) = self.current_char {
            if ch == '0' || ch == '1' || ch == '_' {
                if ch != '_' {
                    bin_str.push(ch);
                }
                self.advance();
            } else {
                break;
            }
        }

        let value = i64::from_str_radix(&bin_str, 2).unwrap_or(0);
        Token::IntLiteral(value)
    }

    /// Read identifier or keyword
    fn read_identifier(&mut self) -> Token {
        let mut ident = String::new();

        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        // Check for keywords
        self.keyword_or_identifier(ident)
    }

    /// Convert identifier to keyword if applicable
    fn keyword_or_identifier(&self, ident: String) -> Token {
        match ident.as_str() {
            // Declarations
            "manifold" => Token::Manifold,
            "crystal" => Token::Crystal,
            "hologram" => Token::Hologram,
            "quantum" => Token::Quantum,
            "entropy" => Token::Entropy,
            "theorem" => Token::Theorem,
            "proof" => Token::Proof,
            "module" => Token::Module,
            "import" => Token::Import,
            "export" => Token::Export,

            // Control flow
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "for" => Token::For,
            "match" => Token::Match,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "return" => Token::Return,
            "yield" => Token::Yield,

            // Sovereignty and quantum
            "sovereignty" => Token::Sovereignty,

            // Qualifiers
            "cortex" => Token::QuantumCortex,
            "logic" => Token::QuantumLogic,
            "constant" => Token::ManifoldConst,

            "let" => Token::Let,
            "mut" => Token::Mut,
            "fn" => Token::Fn,

            // Primitive types
            "i8" => Token::I8,
            "i16" => Token::I16,
            "i32" => Token::I32,
            "i64" => Token::I64,
            "u8" => Token::U8,
            "u16" => Token::U16,
            "u32" => Token::U32,
            "u64" => Token::U64,
            "f32" => Token::F32,
            "f64" => Token::F64,
            "bool" => Token::Bool,

            // 7D types
            "HyperbolicReal" => Token::HyperbolicReal,
            "Vector7D" => Token::Vector7D,
            "Matrix7D" => Token::Matrix7D,
            "Tensor7D" => Token::Tensor7D,
            "Manifold" => Token::ManifoldType,
            "Crystal" => Token::CrystalType,
            "Hologram" => Token::HologramType,
            "QuantumState" => Token::QuantumState,
            "Pattern" => Token::Pattern,
            "WaveFunction" => Token::WaveFunction,

            // Boolean literals
            "consciousness" => Token::Consciousness,
            "Consciousness" => Token::Consciousness,
            "Proof" => Token::Proof,
            "INFINITE" => Token::Infinite,
            "ETERNITY" => Token::Eternity,
            "true" => Token::BoolLiteral(true),
            "false" => Token::BoolLiteral(false),

            _ => Token::Identifier(ident),
        }
    }

    /// Read string literal
    fn read_string(&mut self) -> Token {
        self.advance(); // Skip opening quote

        let mut string = String::new();

        while let Some(ch) = self.current_char {
            if ch == '"' {
                self.advance(); // Skip closing quote
                break;
            } else if ch == '\\' {
                // Handle escape sequences
                self.advance();
                if let Some(escaped) = self.current_char {
                    let ch = match escaped {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '\\' => '\\',
                        '"' => '"',
                        _ => escaped,
                    };
                    string.push(ch);
                    self.advance();
                }
            } else {
                string.push(ch);
                self.advance();
            }
        }

        Token::StringLiteral(string)
    }

    /// Read attribute (@sovereignty, @manifold, etc.)
    fn read_attribute(&mut self) -> Token {
        self.advance(); // Skip '@'

        let mut attr = String::new();
        while let Some(ch) = self.current_char {
            if ch.is_alphanumeric() || ch == '_' {
                attr.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        Token::Attribute(attr)
    }

    /// Get next token
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        // Handle comments
        if self.current_char == Some('/') {
            if self.peek(1) == Some('/') {
                self.skip_comment();
                return self.next_token();
            } else if self.peek(1) == Some('*') {
                self.skip_multiline_comment();
                return self.next_token();
            }
        }

        match self.current_char {
            None => Token::Eof,

            // String literals
            Some('"') => self.read_string(),

            // Attributes
            Some('@') => self.read_attribute(),

            // Mathematical operators (UTF-8 symbols)
            Some('⊗') => {
                self.advance();
                Token::TensorProduct
            }
            Some('⊕') => {
                self.advance();
                Token::Superposition
            }
            Some('⊙') => {
                self.advance();
                Token::HoloFold
            }
            Some('⑦') => {
                self.advance();
                Token::Project7D
            }
            Some('Φ') => {
                self.advance();
                // Check for Φ⁻¹
                if self.current_char == Some('⁻') && self.peek(1) == Some('¹') {
                    self.advance(); // Skip ⁻
                    self.advance(); // Skip ¹
                    Token::PhiInverse
                } else {
                    Token::PhiConstant
                }
            }
            Some('S') if self.peek(1) == Some('²') => {
                self.advance(); // Skip 'S'
                self.advance(); // Skip '²'
                Token::S2Constant
            }
            Some('∞') => {
                self.advance();
                Token::Infinity
            }
            Some('Λ') => {
                self.advance();
                Token::LambdaConstant
            }
            Some('Ψ') => {
                self.advance();
                Token::PsiConstant
            }

            // Single-character tokens
            Some('+') => {
                self.advance();
                Token::Plus
            }
            Some('-') => {
                self.advance();
                if self.current_char == Some('>') {
                    self.advance();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            Some('*') => {
                self.advance();
                Token::Star
            }
            Some('/') => {
                self.advance();
                Token::Slash
            }
            Some('%') => {
                self.advance();
                Token::Percent
            }
            Some('(') => {
                self.advance();
                Token::LeftParen
            }
            Some(')') => {
                self.advance();
                Token::RightParen
            }
            Some('{') => {
                self.advance();
                Token::LeftBrace
            }
            Some('}') => {
                self.advance();
                Token::RightBrace
            }
            Some('[') => {
                self.advance();
                Token::LeftBracket
            }
            Some(']') => {
                self.advance();
                Token::RightBracket
            }
            Some(';') => {
                self.advance();
                Token::Semicolon
            }
            Some(',') => {
                self.advance();
                Token::Comma
            }
            Some('.') => {
                self.advance();
                Token::Dot
            }

            // Two-character operators
            Some('=') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::Eq
                } else if self.current_char == Some('>') {
                    self.advance();
                    Token::FatArrow
                } else {
                    Token::Assign
                }
            }
            Some('!') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::Ne
                } else {
                    Token::Not
                }
            }
            Some('<') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::Le
                } else {
                    Token::Lt
                }
            }
            Some('>') => {
                self.advance();
                if self.current_char == Some('=') {
                    self.advance();
                    Token::Ge
                } else {
                    Token::Gt
                }
            }
            Some('&') => {
                self.advance();
                if self.current_char == Some('&') {
                    self.advance();
                    Token::And
                } else {
                    Token::Ampersand
                }
            }
            Some('|') => {
                self.advance();
                if self.current_char == Some('|') {
                    self.advance();
                    Token::Or
                } else {
                    Token::Unknown('|')
                }
            }
            Some(':') => {
                self.advance();
                if self.current_char == Some(':') {
                    self.advance();
                    Token::ColonColon
                } else {
                    Token::Colon
                }
            }

            // Generic Number fallback (must be after S²)
            Some(ch) if ch.is_numeric() => {
                let mut i = 0;
                let mut is_pure_number = true;
                let mut is_hex = false;

                if ch == '0' {
                    match self.peek(1) {
                        Some('x') | Some('X') => is_hex = true,
                        _ => {}
                    }
                }

                while let Some(next) = self.peek(i) {
                    if next.is_whitespace() || "()[]{};,+-*/%&|=<>!@".contains(next) {
                        break;
                    }
                    if next.is_alphabetic() {
                        if is_hex && next.is_ascii_hexdigit() {
                            // Hex digit is fine
                        } else if i == 1
                            && (next == 'x' || next == 'X' || next == 'b' || next == 'B')
                            && self.peek(0) == Some('0')
                        {
                            // Prefix is fine
                        } else {
                            is_pure_number = false;
                            break;
                        }
                    }
                    i += 1;
                }

                if is_pure_number {
                    self.read_number()
                } else {
                    self.read_identifier()
                }
            }

            // Generic Identifier fallback (must be after Φ)
            Some(ch) if ch.is_alphabetic() || ch == '_' => self.read_identifier(),

            Some(ch) => {
                self.advance();
                Token::Unknown(ch)
            }
        }
    }

    /// Tokenize entire input
    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token();
            if token == Token::Eof {
                tokens.push(token);
                break;
            }
            tokens.push(token);
        }

        tokens
    }

    /// Get current position info
    pub fn position(&self) -> (usize, usize) {
        (self.line, self.column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let mut lexer = Lexer::new("manifold quantum cortex fn".to_string());
        assert_eq!(lexer.next_token(), Token::Manifold);
        assert_eq!(lexer.next_token(), Token::Quantum);
        assert_eq!(lexer.next_token(), Token::QuantumCortex);
        assert_eq!(lexer.next_token(), Token::Fn);
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 3.14159 0x7D 0b1111101".to_string());
        assert_eq!(lexer.next_token(), Token::IntLiteral(42));
        assert_eq!(lexer.next_token(), Token::FloatLiteral(3.14159));
        assert_eq!(lexer.next_token(), Token::IntLiteral(0x7D));
        assert_eq!(lexer.next_token(), Token::IntLiteral(0b1111101));
    }

    #[test]
    fn test_mathematical_operators() {
        let mut lexer = Lexer::new("⊗ ⊕ ⊙ ⑦ Φ Φ⁻¹ S²".to_string());
        assert_eq!(lexer.next_token(), Token::TensorProduct);
        assert_eq!(lexer.next_token(), Token::Superposition);
        assert_eq!(lexer.next_token(), Token::HoloFold);
        assert_eq!(lexer.next_token(), Token::Project7D);
        assert_eq!(lexer.next_token(), Token::PhiConstant);
        assert_eq!(lexer.next_token(), Token::PhiInverse);
        assert_eq!(lexer.next_token(), Token::S2Constant);
    }

    #[test]
    fn test_string_literal() {
        let mut lexer = Lexer::new(r#""Hello, Manifold!""#.to_string());
        assert_eq!(
            lexer.next_token(),
            Token::StringLiteral("Hello, Manifold!".to_string())
        );
    }

    #[test]
    fn test_attributes() {
        let mut lexer = Lexer::new("@sovereignty @manifold @crystal".to_string());
        assert_eq!(
            lexer.next_token(),
            Token::Attribute("sovereignty".to_string())
        );
        assert_eq!(lexer.next_token(), Token::Attribute("manifold".to_string()));
        assert_eq!(lexer.next_token(), Token::Attribute("crystal".to_string()));
    }
}
