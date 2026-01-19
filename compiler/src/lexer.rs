// File: lexer.rs
// 7D Crystal Lexer - High-performance tokenization of 7D-MHQL source code.
// Licensed under the MIT License.
// Discovered by Sir Charles Spikes | December 24, 2025 | Cincinnati, Ohio, USA 🇺🇸

use crate::errors::{CrystalError, CrystalResult, ErrorCode};
use std::fmt;

/// Token types in 7D-MHQL.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ═══════════════════════════════════════════════════════════════════════════
    // KEYWORDS
    // ═══════════════════════════════════════════════════════════════════════════

    // Declarations
    Manifold, // manifold
    Crystal,  // crystal
    Hologram, // hologram
    Quantum,  // quantum
    Entropy,  // entropy
    Theorem,  // theorem
    Proof,    // proof
    Module,   // module
    Import,   // import
    Export,   // export

    // Control Flow
    If,       // if
    Else,     // else
    While,    // while
    For,      // for
    Match,    // match
    Break,    // break
    Continue, // continue
    Return,   // return
    Yield,    // yield

    // Sovereignty & Qualifiers
    Sovereignty,   // sovereignty
    QuantumCortex, // quantum cortex
    QuantumLogic,  // quantum logic
    Constant,      // manifold constant
    Let,           // let
    Mut,           // mut
    Fn,            // fn

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
    Bool, // bool

    // 7D Specific Types
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

    // ═══════════════════════════════════════════════════════════════════════════
    // OPERATORS & SYMBOLS
    // ═══════════════════════════════════════════════════════════════════════════

    // Mathematical
    Plus,    // +
    Minus,   // -
    Star,    // *
    Slash,   // /
    Percent, // %
    PlusEq,  // +=
    MinusEq, // -=
    StarEq,  // *=
    SlashEq, // /=

    // 7D Sacred Operators
    TensorProduct, // ⊗
    Superposition, // ⊕
    HoloFold,      // ⊙
    Project7D,     // ⑦

    // Comparisons
    Eq, // ==
    Ne, // !=
    Lt, // <
    Gt, // >
    Le, // <=
    Ge, // >=

    // Logical
    And, // &&
    Or,  // ||
    Not, // !

    // Assignment & Flow
    Assign,   // =
    Arrow,    // ->
    FatArrow, // =>

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
    At,           // @
    Ampersand,    // &

    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTANTS & LITERALS
    // ═══════════════════════════════════════════════════════════════════════════

    // Sacred Constants
    PhiConstant, // Φ
    PhiInverse,  // Φ⁻¹
    S2Constant,  // S²
    Infinite,    // ∞
    // ∞
    LambdaConstant, // Λ
    PsiConstant,    // Ψ
    Eternity,       // eternity

    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Identifiers
    Identifier(String),
    Attribute(String),

    // Meta
    Eof,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Identifier(s) => write!(f, "{}", s),
            Token::IntLiteral(n) => write!(f, "{}", n),
            Token::FloatLiteral(n) => write!(f, "{}", n),
            Token::PhiConstant => write!(f, "Φ"),
            Token::PhiInverse => write!(f, "Φ⁻¹"),
            Token::Eof => write!(f, "EOF"),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Lexer state machine.
pub struct Lexer {
    input: Vec<char>,
    position: usize,
    line: usize,
    column: usize,
    file_name: String,
}

impl Lexer {
    pub fn new(source: &str) -> Self {
        Self {
            input: source.chars().collect(),
            position: 0,
            line: 1,
            column: 1,
            file_name: "source.7d".to_string(),
        }
    }

    pub fn with_filename(mut self, name: &str) -> Self {
        self.file_name = name.to_string();
        self
    }

    fn current(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    fn peek(&self, offset: usize) -> Option<char> {
        self.input.get(self.position + offset).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.current();
        if let Some(c) = ch {
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.position += 1;
        }
        ch
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '/' && self.peek(1) == Some('/') {
                // Single line comment
                while let Some(c) = self.current() {
                    if c == '\n' {
                        break;
                    }
                    self.advance();
                }
            } else if ch == '/' && self.peek(1) == Some('*') {
                // Multi-line comment
                self.advance();
                self.advance();
                while let Some(c) = self.current() {
                    if c == '*' && self.peek(1) == Some('/') {
                        self.advance();
                        self.advance();
                        break;
                    }
                    self.advance();
                }
            } else {
                break;
            }
        }
    }

    fn read_identifier(&mut self) -> Token {
        let mut ident = String::new();
        while let Some(ch) = self.current() {
            if ch.is_alphanumeric() || ch == '_' {
                ident.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match ident.as_str() {
            "manifold" => Token::Manifold,
            "crystal" => Token::Crystal,
            "hologram" => Token::Hologram,
            "entropy" => Token::Entropy,
            "theorem" => Token::Theorem,
            "proof" => Token::Proof,
            "module" => Token::Module,
            "import" => Token::Import,
            "export" => Token::Export,
            "if" => Token::If,
            "else" => Token::Else,
            "while" => Token::While,
            "for" => Token::For,
            "match" => Token::Match,
            "break" => Token::Break,
            "continue" => Token::Continue,
            "return" => Token::Return,
            "yield" => Token::Yield,
            "sovereignty" => Token::Sovereignty,
            "quantum" => {
                let saved_pos = self.position;
                let saved_line = self.line;
                let saved_col = self.column;

                self.skip_whitespace();
                let mut next_ident = String::new();
                while let Some(ch) = self.current() {
                    if ch.is_alphanumeric() || ch == '_' {
                        next_ident.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }

                match next_ident.as_str() {
                    "cortex" => Token::QuantumCortex,
                    "logic" => Token::QuantumLogic,
                    _ => {
                        // Backtrack
                        self.position = saved_pos;
                        self.line = saved_line;
                        self.column = saved_col;
                        Token::Quantum
                    }
                }
            }
            "constant" => Token::Constant,
            "let" => Token::Let,
            "mut" => Token::Mut,
            "fn" => Token::Fn,
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
            "consciousness" => Token::Consciousness,
            "eternity" => Token::Eternity,
            "true" => Token::BoolLiteral(true),
            "false" => Token::BoolLiteral(false),
            _ => Token::Identifier(ident),
        }
    }

    fn read_number(&mut self) -> CrystalResult<Token> {
        let mut num_str = String::new();
        let mut is_float = false;

        // Handle hex/binary prefixes
        if self.current() == Some('0') {
            match self.peek(1) {
                Some('x') => {
                    self.advance();
                    self.advance();
                    let mut hex = String::new();
                    while let Some(ch) = self.current() {
                        if ch.is_ascii_hexdigit() || ch == '_' {
                            if ch != '_' {
                                hex.push(ch);
                            }
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    return Ok(Token::IntLiteral(
                        i64::from_str_radix(&hex, 16).unwrap_or(0),
                    ));
                }
                Some('b') => {
                    self.advance();
                    self.advance();
                    let mut bin = String::new();
                    while let Some(ch) = self.current() {
                        if ch == '0' || ch == '1' || ch == '_' {
                            if ch != '_' {
                                bin.push(ch);
                            }
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    return Ok(Token::IntLiteral(i64::from_str_radix(&bin, 2).unwrap_or(0)));
                }
                _ => {}
            }
        }

        while let Some(ch) = self.current() {
            if ch.is_ascii_digit() || ch == '_' {
                if ch != '_' {
                    num_str.push(ch);
                }
                self.advance();
            } else if ch == '.' && !is_float && self.peek(1).map_or(false, |c| c.is_ascii_digit()) {
                is_float = true;
                num_str.push('.');
                self.advance();
            } else {
                break;
            }
        }

        if is_float {
            match num_str.parse::<f64>() {
                Ok(f) => Ok(Token::FloatLiteral(f)),
                Err(_) => Err(CrystalError::new(
                    ErrorCode::InvalidNumericLiteral,
                    &format!("Invalid float: {}", num_str),
                )
                .at(&self.file_name, self.line, self.column)),
            }
        } else {
            match num_str.parse::<i64>() {
                Ok(i) => Ok(Token::IntLiteral(i)),
                Err(_) => Err(CrystalError::new(
                    ErrorCode::InvalidNumericLiteral,
                    &format!("Invalid integer: {}", num_str),
                )
                .at(&self.file_name, self.line, self.column)),
            }
        }
    }

    fn read_string(&mut self) -> CrystalResult<Token> {
        self.advance(); // Skip "
        let mut s = String::new();
        while let Some(ch) = self.advance() {
            if ch == '"' {
                return Ok(Token::StringLiteral(s));
            }
            if ch == '\\' {
                match self.advance() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('\\') => s.push('\\'),
                    Some('"') => s.push('"'),
                    Some(c) => s.push(c),
                    None => break,
                }
            } else {
                s.push(ch);
            }
        }
        Err(
            CrystalError::new(ErrorCode::UnterminatedString, "String literal not closed").at(
                &self.file_name,
                self.line,
                self.column,
            ),
        )
    }

    pub fn next_token(&mut self) -> CrystalResult<Token> {
        self.skip_whitespace();

        let start_line = self.line;
        let start_col = self.column;

        let ch = match self.current() {
            None => return Ok(Token::Eof),
            Some(c) => c,
        };

        // ══════════════════════════════════════════════════════════════════════════
        // SACRED UTF-8 HANDLING
        // ══════════════════════════════════════════════════════════════════════════

        match ch {
            '⊗' => {
                self.advance();
                return Ok(Token::TensorProduct);
            }
            '⊕' => {
                self.advance();
                return Ok(Token::Superposition);
            }
            '⊙' => {
                self.advance();
                return Ok(Token::HoloFold);
            }
            '⑦' => {
                self.advance();
                return Ok(Token::Project7D);
            }
            'Φ' => {
                self.advance();
                if self.current() == Some('⁻') && self.peek(1) == Some('¹') {
                    self.advance();
                    self.advance();
                    return Ok(Token::PhiInverse);
                }
                return Ok(Token::PhiConstant);
            }
            '∞' => {
                self.advance();
                return Ok(Token::Infinite);
            }
            'Λ' => {
                self.advance();
                return Ok(Token::LambdaConstant);
            }
            'Ψ' => {
                self.advance();
                return Ok(Token::PsiConstant);
            }
            'S' if self.peek(1) == Some('²') => {
                self.advance();
                self.advance();
                return Ok(Token::S2Constant);
            }
            _ => {}
        }

        // Standard Operators & Symbols
        match ch {
            '(' => {
                self.advance();
                Ok(Token::LeftParen)
            }
            ')' => {
                self.advance();
                Ok(Token::RightParen)
            }
            '{' => {
                self.advance();
                Ok(Token::LeftBrace)
            }
            '}' => {
                self.advance();
                Ok(Token::RightBrace)
            }
            '[' => {
                self.advance();
                Ok(Token::LeftBracket)
            }
            ']' => {
                self.advance();
                Ok(Token::RightBracket)
            }
            ';' => {
                self.advance();
                Ok(Token::Semicolon)
            }
            ',' => {
                self.advance();
                Ok(Token::Comma)
            }
            '.' => {
                self.advance();
                Ok(Token::Dot)
            }
            ':' => {
                self.advance();
                if self.current() == Some(':') {
                    self.advance();
                    Ok(Token::ColonColon)
                } else {
                    Ok(Token::Colon)
                }
            }
            '@' => {
                self.advance();
                if let Some(ch) = self.current() {
                    if ch.is_alphabetic() || ch == '_' {
                        let ident = match self.read_identifier() {
                            Token::Identifier(s) => s,
                            t => format!("{:?}", t).to_lowercase(),
                        };
                        return Ok(Token::Attribute(ident));
                    }
                }
                Ok(Token::At)
            }
            '&' => {
                self.advance();
                if self.current() == Some('&') {
                    self.advance();
                    Ok(Token::And)
                } else {
                    Ok(Token::Ampersand)
                }
            }
            '|' => {
                self.advance();
                if self.current() == Some('|') {
                    self.advance();
                    Ok(Token::Or)
                } else {
                    Err(
                        CrystalError::new(ErrorCode::UnexpectedChar, "Invalid character '|'").at(
                            &self.file_name,
                            start_line,
                            start_col,
                        ),
                    )
                }
            }
            '=' => {
                self.advance();
                match self.current() {
                    Some('=') => {
                        self.advance();
                        Ok(Token::Eq)
                    }
                    Some('>') => {
                        self.advance();
                        Ok(Token::FatArrow)
                    }
                    _ => Ok(Token::Assign),
                }
            }
            '!' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    Ok(Token::Ne)
                } else {
                    Ok(Token::Not)
                }
            }
            '<' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    Ok(Token::Le)
                } else {
                    Ok(Token::Lt)
                }
            }
            '>' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            '+' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    Ok(Token::PlusEq)
                } else {
                    Ok(Token::Plus)
                }
            }
            '-' => {
                self.advance();
                match self.current() {
                    Some('=') => {
                        self.advance();
                        Ok(Token::MinusEq)
                    }
                    Some('>') => {
                        self.advance();
                        Ok(Token::Arrow)
                    }
                    _ => Ok(Token::Minus),
                }
            }
            '*' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    Ok(Token::StarEq)
                } else {
                    Ok(Token::Star)
                }
            }
            '/' => {
                self.advance();
                if self.current() == Some('=') {
                    self.advance();
                    Ok(Token::SlashEq)
                } else {
                    Ok(Token::Slash)
                }
            }
            '%' => {
                self.advance();
                Ok(Token::Percent)
            }
            '"' => self.read_string(),
            '0'..='9' => self.read_number(),
            'a'..='z' | 'A'..='Z' | '_' => Ok(self.read_identifier()),
            _ => {
                self.advance();
                Err(CrystalError::new(
                    ErrorCode::UnexpectedChar,
                    &format!("Unexpected character: '{}'", ch),
                )
                .at(&self.file_name, start_line, start_col))
            }
        }
    }

    pub fn tokenize(&mut self) -> CrystalResult<Vec<Token>> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok == Token::Eof;
            tokens.push(tok.clone());
            if is_eof {
                break;
            }
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sacred_set() {
        let mut lexer = Lexer::new("⊗ ⊕ ⊙ ⑦ Φ Φ⁻¹ S² ∞ Λ Ψ");
        assert_eq!(lexer.next_token().unwrap(), Token::TensorProduct);
        assert_eq!(lexer.next_token().unwrap(), Token::Superposition);
        assert_eq!(lexer.next_token().unwrap(), Token::HoloFold);
        assert_eq!(lexer.next_token().unwrap(), Token::Project7D);
        assert_eq!(lexer.next_token().unwrap(), Token::PhiConstant);
        assert_eq!(lexer.next_token().unwrap(), Token::PhiInverse);
        assert_eq!(lexer.next_token().unwrap(), Token::S2Constant);
        assert_eq!(lexer.next_token().unwrap(), Token::Infinity);
        assert_eq!(lexer.next_token().unwrap(), Token::LambdaConstant);
        assert_eq!(lexer.next_token().unwrap(), Token::PsiConstant);
    }

    #[test]
    fn test_attributes() {
        let mut lexer = Lexer::new("@crystal @manifold @cortex");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Attribute("crystal".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Attribute("manifold".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Attribute("cortex".to_string())
        );
    }

    #[test]
    fn test_keywords() {
        let mut lexer = Lexer::new("manifold crystal hologram quantum entropy theorem proof");
        assert_eq!(lexer.next_token().unwrap(), Token::Manifold);
        assert_eq!(lexer.next_token().unwrap(), Token::Crystal);
        assert_eq!(lexer.next_token().unwrap(), Token::Hologram);
        assert_eq!(lexer.next_token().unwrap(), Token::Quantum);
        assert_eq!(lexer.next_token().unwrap(), Token::Entropy);
        assert_eq!(lexer.next_token().unwrap(), Token::Theorem);
        assert_eq!(lexer.next_token().unwrap(), Token::Proof);
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("123 0xFF 0b1010 3.14159");
        assert_eq!(lexer.next_token().unwrap(), Token::IntLiteral(123));
        assert_eq!(lexer.next_token().unwrap(), Token::IntLiteral(255));
        assert_eq!(lexer.next_token().unwrap(), Token::IntLiteral(10));
        assert_eq!(lexer.next_token().unwrap(), Token::FloatLiteral(3.14159));
    }

    #[test]
    fn test_strings() {
        let mut lexer = Lexer::new("\"hello 7D\" \"escaped \\\" quote\"");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::StringLiteral("hello 7D".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::StringLiteral("escaped \" quote".to_string())
        );
    }
}
