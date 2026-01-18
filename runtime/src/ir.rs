// 7D Crystal IR Types for Runtime
// This mirrors the compiler's IR for runtime execution

#[derive(Debug, Clone, PartialEq)]
pub enum IR7D {
    // Stack operations
    PushInt(i64),
    PushFloat(f64),
    Pop,

    // Memory operations
    Store(usize, isize), // reg, offset
    Load(usize, isize),  // reg, offset

    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,

    // 7D Manifold Operations
    ManifoldProject {
        input_reg: usize,
        output_reg: usize,
        curvature: f64,
    },

    // Holographic Operations
    HolographicFold {
        p1_reg: usize,
        p2_reg: usize,
        out_reg: usize,
        phases: u8,
    },

    // Control Flow
    Return,
    Jump(usize),
    JumpIfFalse(usize),

    // Labels
    Label(usize),

    // Function calls
    Call(String, usize), // name, arg count
}

impl IR7D {
    pub fn to_bits(&self) -> u64 {
        // Simple discriminant for hashing
        match self {
            IR7D::PushInt(_) => 1,
            IR7D::PushFloat(_) => 2,
            IR7D::Pop => 3,
            IR7D::Store(_, _) => 4,
            IR7D::Load(_, _) => 5,
            IR7D::Add => 6,
            IR7D::Sub => 7,
            IR7D::Mul => 8,
            IR7D::Div => 9,
            IR7D::ManifoldProject { .. } => 10,
            IR7D::HolographicFold { .. } => 11,
            IR7D::Return => 12,
            IR7D::Jump(_) => 13,
            IR7D::JumpIfFalse(_) => 14,
            IR7D::Label(_) => 15,
            IR7D::Call(_, _) => 16,
        }
    }
}
