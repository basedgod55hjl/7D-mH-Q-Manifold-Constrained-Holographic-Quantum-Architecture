pub use codespan_reporting::diagnostic::{Diagnostic, Severity};
use colored::*;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Lexer Errors (E1xxx)
    UnexpectedChar = 1001,
    UnterminatedString = 1002,
    InvalidNumericLiteral = 1003,

    // Parser Errors (E2xxx)
    UnexpectedToken = 2001,
    MismatchedBrackets = 2002,
    MissingSemicolon = 2003,
    InvalidAttribute = 2004,

    // Semantic Errors (E3xxx)
    PhiRatioViolation = 3001,
    S2StabilityViolation = 3002,
    QuantumCloningAttempt = 3003,
    UninitializedManifold = 3004,
    DimensionMismatch = 3005,

    // CodeGen/Runtime Errors (E4xxx)
    GpuAllocationFailed = 4001,
    KernelCompilationError = 4002,
    ManifoldDivergence = 4003,

    // Additional errors needed by the codebase
    E2001 = 2901,
    E3001 = 3901,
    E4001 = 4901,
    E4003 = 4903,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{:04}", *self as u16)
    }
}

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct CrystalError {
    pub code: ErrorCode,
    pub message: String,
    pub location: Option<SourceLocation>,
    pub suggestion: Option<String>,
    pub notes: Vec<String>,
}

impl CrystalError {
    pub fn new(code: ErrorCode, message: &str) -> Self {
        Self {
            code,
            message: message.to_string(),
            location: None,
            suggestion: None,
            notes: vec![],
        }
    }

    pub fn at(mut self, file: &str, line: usize, col: usize) -> Self {
        self.location = Some(SourceLocation {
            file: file.to_string(),
            line,
            col,
        });
        self
    }

    pub fn suggest(mut self, suggestion: &str) -> Self {
        self.suggestion = Some(suggestion.to_string());
        self
    }

    pub fn note(mut self, note: &str) -> Self {
        self.notes.push(note.to_string());
        self
    }

    pub fn report(&self) {
        let error_header = format!("error[{}]: {}", self.code, self.message)
            .red()
            .bold();

        println!("{}", error_header);

        if let Some(loc) = &self.location {
            println!(
                "  {} {}:{}:{}",
                "-->".blue().bold(),
                loc.file,
                loc.line,
                loc.col
            );
        }

        if let Some(sugg) = &self.suggestion {
            println!("  {} {}", "help:".cyan().bold(), sugg);
        }

        for note in &self.notes {
            println!("  {} {}", "note:".yellow().bold(), note);
        }

        println!();
    }
}

#[derive(Debug, Clone, Default)]
pub struct DiagnosticCollection {
    pub diagnostics: Vec<Diagnostic<()>>,
}

impl DiagnosticCollection {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, diagnostic: Diagnostic<()>) {
        self.diagnostics.push(diagnostic);
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    pub fn print_all(&self) {
        for d in &self.diagnostics {
            println!("{:?}: {}", d.severity, d.message);
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Diagnostic<()>> {
        self.diagnostics.iter()
    }
}

impl From<CrystalError> for DiagnosticCollection {
    fn from(err: CrystalError) -> Self {
        let mut collection = Self::new();
        let d = Diagnostic::new(Severity::Error)
            .with_message(err.message)
            .with_code(err.code.to_string());
        collection.add(d);
        collection
    }
}

pub type CrystalResult<T> = Result<T, CrystalError>;

// Macro for easy error reporting
#[macro_export]
macro_rules! throw_error {
    ($code:expr, $msg:expr) => {
        return Err($crate::errors::CrystalError::new($code, $msg))
    };
}
