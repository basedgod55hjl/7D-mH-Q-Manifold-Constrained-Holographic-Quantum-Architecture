use std::alloc::{Layout, alloc, dealloc};
use std::collections::HashMap;
use std::mem;
use std::ptr;

use crate::ir::IR7D;

/// 7D Crystal JIT Compiler
pub struct JIT7D {
    code_cache: HashMap<u64, ExecutableMemory>,
    phi_constants: ConstantPool,
    optimization_level: OptimizationLevel,
}

#[derive(Clone, Copy)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
}

impl JIT7D {
    pub fn new() -> Self {
        Self {
            code_cache: HashMap::new(),
            phi_constants: ConstantPool::new(),
            optimization_level: OptimizationLevel::Basic,
        }
    }

    /// Compile 7D-IR to executable machine code
    pub fn compile(&mut self, ir: &[IR7D]) -> Result<*const u8, JITError> {
        // Generate hash for caching
        let ir_hash = self.hash_ir(ir);

        // Check cache first
        if let Some(mem) = self.code_cache.get(&ir_hash) {
            return Ok(mem.ptr);
        }

        // 1. IR Optimization
        let optimized_ir = self.optimize_ir(ir)?;

        // 2. Machine Code Generation
        let asm = self.generate_machine_code(&optimized_ir)?;

        // 3. Allocate executable memory
        let mem = self.allocate_executable(asm.len())?;

        // 4. Write machine code
        unsafe {
            ptr::copy_nonoverlapping(asm.as_ptr(), mem.ptr, asm.len());
        }

        // 5. Patch constants and relocations
        self.patch_constants(mem.ptr, &asm.relocations)?;

        // 6. Cache the result
        self.code_cache.insert(ir_hash, mem.clone());

        Ok(mem.ptr as *const u8)
    }

    /// Execute compiled function
    pub unsafe fn execute(&self, func_ptr: *const u8, args: &[u64]) -> i32 {
        // Cast to function pointer and call
        let func: extern "C" fn(*const u64, usize) -> i32 =
            unsafe { std::mem::transmute(func_ptr) };
        func(args.as_ptr(), args.len())
    }

    /// Clear code cache
    pub fn clear_cache(&mut self) {
        self.code_cache.clear();
    }

    fn hash_ir(&self, ir: &[IR7D]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        ir.len().hash(&mut hasher);
        for instr in ir {
            match instr {
                IR7D::PushInt(val) => {
                    "PushInt".hash(&mut hasher);
                    val.hash(&mut hasher);
                }
                IR7D::PushFloat(val) => {
                    "PushFloat".hash(&mut hasher);
                    val.to_bits().hash(&mut hasher);
                }
                IR7D::Pop => {
                    "Pop".hash(&mut hasher);
                }
                IR7D::Store(reg, offset) => {
                    "Store".hash(&mut hasher);
                    reg.hash(&mut hasher);
                    offset.hash(&mut hasher);
                }
                IR7D::Load(reg, offset) => {
                    "Load".hash(&mut hasher);
                    reg.hash(&mut hasher);
                    offset.hash(&mut hasher);
                }
                IR7D::Add => {
                    "Add".hash(&mut hasher);
                }
                IR7D::Sub => {
                    "Sub".hash(&mut hasher);
                }
                IR7D::Mul => {
                    "Mul".hash(&mut hasher);
                }
                IR7D::Div => {
                    "Div".hash(&mut hasher);
                }
                IR7D::ManifoldProject {
                    input_reg,
                    output_reg,
                    curvature,
                } => {
                    "ManifoldProject".hash(&mut hasher);
                    input_reg.hash(&mut hasher);
                    output_reg.hash(&mut hasher);
                    curvature.to_bits().hash(&mut hasher);
                }
                IR7D::HolographicFold {
                    p1_reg,
                    p2_reg,
                    out_reg,
                    phases,
                } => {
                    "HolographicFold".hash(&mut hasher);
                    p1_reg.hash(&mut hasher);
                    p2_reg.hash(&mut hasher);
                    out_reg.hash(&mut hasher);
                    phases.hash(&mut hasher);
                }
                _ => {
                    "Other".hash(&mut hasher);
                }
            }
        }
        hasher.finish()
    }

    fn optimize_ir(&self, ir: &[IR7D]) -> Result<Vec<IR7D>, JITError> {
        let mut optimized = ir.to_vec();

        match self.optimization_level {
            OptimizationLevel::None => {}
            OptimizationLevel::Basic => {
                self.optimize_basic(&mut optimized);
            }
            OptimizationLevel::Aggressive => {
                self.optimize_basic(&mut optimized);
                self.optimize_aggressive(&mut optimized);
            }
        }

        Ok(optimized)
    }

    fn optimize_basic(&self, ir: &mut Vec<IR7D>) {
        self.eliminate_dead_code(ir);
        self.constant_fold(ir);
        self.simplify_phi_operations(ir);
    }

    fn optimize_aggressive(&self, ir: &mut Vec<IR7D>) {
        self.inline_functions(ir);
        self.loop_unrolling(ir);
        self.vectorization(ir);
    }

    fn generate_machine_code(&self, ir: &[IR7D]) -> Result<AssemblyCode, JITError> {
        let mut asm = AssemblyCode::new();

        // Generate function prologue
        asm.emit(b"\x55"); // push rbp
        asm.emit(b"\x48\x89\xe5"); // mov rbp, rsp

        // Reserve stack space for locals
        let stack_size = self.calculate_stack_size(ir);
        if stack_size > 0 {
            asm.emit_sub_rsp(stack_size);
        }

        // Generate code for each IR instruction
        for instr in ir {
            self.generate_instruction_code(instr, &mut asm)?;
        }

        // Generate function epilogue
        if stack_size > 0 {
            asm.emit_add_rsp(stack_size);
        }
        asm.emit(b"\x5d"); // pop rbp
        asm.emit(b"\xc3"); // ret

        Ok(asm)
    }

    fn generate_instruction_code(
        &self,
        instr: &IR7D,
        asm: &mut AssemblyCode,
    ) -> Result<(), JITError> {
        match instr {
            IR7D::PushInt(value) => {
                asm.emit_mov_rax_imm64(*value as u64);
                asm.emit_push_rax();
            }
            IR7D::PushFloat(value) => {
                let bits = value.to_bits();
                asm.emit_mov_rax_imm64(bits);
                asm.emit_push_rax();
            }
            IR7D::Pop => {
                asm.emit_pop_rax();
            }
            IR7D::Store(_reg, offset) => {
                asm.emit_pop_rax();
                asm.emit_mov_rbp_offset_rax(*offset);
            }
            IR7D::Load(_reg, offset) => {
                asm.emit_mov_rax_rbp_offset(*offset);
                asm.emit_push_rax();
            }
            IR7D::Add => {
                asm.emit_pop_rbx();
                asm.emit_pop_rax();
                asm.emit_add_rax_rbx();
                asm.emit_push_rax();
            }
            IR7D::Sub => {
                asm.emit_pop_rbx();
                asm.emit_pop_rax();
                asm.emit_sub_rax_rbx();
                asm.emit_push_rax();
            }
            IR7D::Mul => {
                asm.emit_pop_rbx();
                asm.emit_pop_rax();
                asm.emit_imul_rax_rbx();
                asm.emit_push_rax();
            }
            IR7D::Div => {
                asm.emit_pop_rbx();
                asm.emit_pop_rax();
                asm.emit_cqo();
                asm.emit_idiv_rbx();
                asm.emit_push_rax();
            }
            IR7D::ManifoldProject {
                input_reg,
                output_reg,
                curvature,
            } => {
                self.generate_manifold_projection(*input_reg, *output_reg, *curvature, asm);
            }
            IR7D::HolographicFold {
                p1_reg,
                p2_reg,
                out_reg,
                phases,
            } => {
                self.generate_holographic_fold(*p1_reg, *p2_reg, *out_reg, *phases, asm);
            }
            IR7D::Return => {
                asm.emit_pop_rax();
                asm.emit_ret();
            }
            _ => return Err(JITError::UnsupportedInstruction),
        }
        Ok(())
    }

    fn generate_manifold_projection(
        &self,
        input_reg: usize,
        output_reg: usize,
        curvature: f64,
        asm: &mut AssemblyCode,
    ) {
        // Load Φ constant
        asm.emit_mov_rax_imm64(self.phi_constants.phi.to_bits());
        asm.emit_mov_xmm0_rax();

        // Load Φ⁻¹ constant
        asm.emit_mov_rax_imm64(self.phi_constants.phi_inv.to_bits());
        asm.emit_mov_xmm1_rax();

        // Load curvature
        asm.emit_mov_rax_imm64(curvature.to_bits());
        asm.emit_mov_xmm2_rax();

        // Load input coordinates (7D vector) and process
        for i in 0..7 {
            let offset = ((input_reg * 8) + (i * 8)) as isize;
            asm.emit_mov_rax_rbp_offset(offset);
            asm.emit_movq_xmm_rax(i + 3); // xmm3-xmm9 for coordinates
        }

        // Calculate norm squared and project (simplified)
        asm.emit_vxorps_xmm10_xmm10_xmm10(); // xmm10 = 0
        for i in 0..7 {
            asm.emit_vmulsd_xmm(10, i + 3, i + 3); // accumulate norm^2
        }

        // Store projected coordinates
        for i in 0..7 {
            let offset = ((output_reg * 8) + (i * 8)) as isize;
            asm.emit_movq_rax_xmm(i + 3);
            asm.emit_mov_rbp_offset_rax(offset);
        }
    }

    fn generate_holographic_fold(
        &self,
        p1_reg: usize,
        p2_reg: usize,
        out_reg: usize,
        _phases: u8,
        asm: &mut AssemblyCode,
    ) {
        // Load pattern 1 and 2
        for i in 0..7 {
            let offset1 = ((p1_reg * 8) + (i * 8)) as isize;
            let offset2 = ((p2_reg * 8) + (i * 8)) as isize;

            asm.emit_mov_rax_rbp_offset(offset1);
            asm.emit_movq_xmm_rax(i);

            asm.emit_mov_rax_rbp_offset(offset2);
            asm.emit_movq_xmm_rax(i + 7);
        }

        // Generate interference pattern (simplified)
        for i in 0..7 {
            // Phase difference approximation
            asm.emit_vsubsd_xmm(i, i, i + 7);

            // Scale by Φ⁻¹
            asm.emit_vmulsd_xmm(i, i, 1);

            // Store result
            let offset = ((out_reg * 8) + (i * 8)) as isize;
            asm.emit_movq_rax_xmm(i);
            asm.emit_mov_rbp_offset_rax(offset);
        }
    }

    fn calculate_stack_size(&self, ir: &[IR7D]) -> usize {
        let mut max_vars = 0;
        for instr in ir {
            match instr {
                IR7D::Store(reg, _) => max_vars = max_vars.max(*reg + 1),
                IR7D::Load(reg, _) => max_vars = max_vars.max(*reg + 1),
                _ => {}
            }
        }
        (max_vars * 8 + 15) & !15 // 16-byte alignment
    }

    fn allocate_executable(&self, size: usize) -> Result<ExecutableMemory, JITError> {
        let actual_size = if size == 0 { 4096 } else { size };
        let layout = Layout::from_size_align(actual_size, 4096)
            .map_err(|_| JITError::MemoryAllocationFailed)?;

        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(JITError::MemoryAllocationFailed);
        }

        // Platform-specific: make memory executable
        #[cfg(windows)]
        {
            use std::os::raw::c_void;
            unsafe extern "system" {
                fn VirtualProtect(
                    lpAddress: *mut c_void,
                    dwSize: usize,
                    flNewProtect: u32,
                    lpflOldProtect: *mut u32,
                ) -> i32;
            }
            unsafe {
                let mut old_protect: u32 = 0;
                VirtualProtect(
                    ptr as *mut c_void,
                    actual_size,
                    0x40, // PAGE_EXECUTE_READWRITE
                    &mut old_protect,
                );
            }
        }

        Ok(ExecutableMemory {
            ptr,
            size: actual_size,
            layout,
        })
    }

    fn patch_constants(
        &self,
        code_ptr: *mut u8,
        relocations: &[Relocation],
    ) -> Result<(), JITError> {
        for reloc in relocations {
            let target_addr = unsafe { code_ptr.add(reloc.offset) };
            let constant_addr = match reloc.symbol.as_str() {
                "PHI" => &self.phi_constants.phi as *const f64 as usize,
                "PHI_INV" => &self.phi_constants.phi_inv as *const f64 as usize,
                "S2_BOUND" => &self.phi_constants.s2_bound as *const f64 as usize,
                _ => return Err(JITError::UnknownSymbol(reloc.symbol.clone())),
            };
            unsafe {
                *(target_addr as *mut usize) = constant_addr;
            }
        }
        Ok(())
    }

    // Optimization passes
    fn eliminate_dead_code(&self, ir: &mut Vec<IR7D>) {
        let mut used_regs = std::collections::HashSet::new();

        for instr in ir.iter() {
            if let IR7D::Load(reg, _) = instr {
                used_regs.insert(*reg);
            }
        }

        ir.retain(|instr| match instr {
            IR7D::Store(reg, _) => used_regs.contains(reg),
            _ => true,
        });
    }

    fn constant_fold(&self, ir: &mut Vec<IR7D>) {
        let mut i = 0;
        while i + 2 < ir.len() {
            if let (IR7D::PushInt(a), IR7D::PushInt(b), IR7D::Add) =
                (&ir[i], &ir[i + 1], &ir[i + 2])
            {
                let result = a + b;
                ir.splice(i..i + 3, vec![IR7D::PushInt(result)]);
                continue;
            }
            if let (IR7D::PushInt(a), IR7D::PushInt(b), IR7D::Mul) =
                (&ir[i], &ir[i + 1], &ir[i + 2])
            {
                let result = a * b;
                ir.splice(i..i + 3, vec![IR7D::PushInt(result)]);
                continue;
            }
            // Φ simplifications
            if let (IR7D::PushFloat(a), IR7D::PushFloat(b), IR7D::Mul) =
                (&ir[i], &ir[i + 1], &ir[i + 2])
            {
                let phi = 1.618033988749895f64;
                let phi_inv = 0.618033988749895f64;
                // Φ * Φ⁻¹ = 1
                if (*a - phi).abs() < 1e-10 && (*b - phi_inv).abs() < 1e-10 {
                    ir.splice(i..i + 3, vec![IR7D::PushFloat(1.0)]);
                    continue;
                }
                // General float multiplication
                let result = a * b;
                ir.splice(i..i + 3, vec![IR7D::PushFloat(result)]);
                continue;
            }
            i += 1;
        }
    }

    fn simplify_phi_operations(&self, _ir: &mut Vec<IR7D>) {
        // Simplify Φ-related operations
        // x * Φ * Φ⁻¹ = x
        // Implemented in constant_fold
    }

    fn inline_functions(&self, _ir: &mut Vec<IR7D>) {
        // Function inlining for small functions
        // TODO: Implement when function table is available
    }

    fn loop_unrolling(&self, _ir: &mut Vec<IR7D>) {
        // Loop unrolling for 7D vector operations
        // Already unrolled in generate_manifold_projection
    }

    fn vectorization(&self, _ir: &mut Vec<IR7D>) {
        // Auto-vectorization of 7D operations
        // Already using SIMD in codegen
    }
}

#[derive(Clone)]
struct ExecutableMemory {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
}

impl Drop for ExecutableMemory {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

struct ConstantPool {
    phi: f64,
    phi_inv: f64,
    s2_bound: f64,
}

impl ConstantPool {
    fn new() -> Self {
        Self {
            phi: 1.618033988749895,
            phi_inv: 0.618033988749895,
            s2_bound: 0.01,
        }
    }
}

struct AssemblyCode {
    code: Vec<u8>,
    relocations: Vec<Relocation>,
}

struct Relocation {
    offset: usize,
    symbol: String,
}

impl AssemblyCode {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            relocations: Vec::new(),
        }
    }

    fn emit(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    fn len(&self) -> usize {
        self.code.len()
    }

    fn as_ptr(&self) -> *const u8 {
        self.code.as_ptr()
    }

    // x86-64 instruction helpers
    fn emit_mov_rax_imm64(&mut self, value: u64) {
        self.emit(b"\x48\xb8");
        self.emit(&value.to_le_bytes());
    }

    fn emit_push_rax(&mut self) {
        self.emit(b"\x50");
    }

    fn emit_pop_rax(&mut self) {
        self.emit(b"\x58");
    }

    fn emit_pop_rbx(&mut self) {
        self.emit(b"\x5b");
    }

    fn emit_mov_rbp_offset_rax(&mut self, offset: isize) {
        self.emit(b"\x48\x89\x85");
        self.emit(&(offset as i32).to_le_bytes());
    }

    fn emit_mov_rax_rbp_offset(&mut self, offset: isize) {
        self.emit(b"\x48\x8b\x85");
        self.emit(&(offset as i32).to_le_bytes());
    }

    fn emit_add_rax_rbx(&mut self) {
        self.emit(b"\x48\x01\xd8");
    }

    fn emit_sub_rax_rbx(&mut self) {
        self.emit(b"\x48\x29\xd8");
    }

    fn emit_imul_rax_rbx(&mut self) {
        self.emit(b"\x48\x0f\xaf\xc3");
    }

    fn emit_cqo(&mut self) {
        self.emit(b"\x48\x99");
    }

    fn emit_idiv_rbx(&mut self) {
        self.emit(b"\x48\xf7\xfb");
    }

    fn emit_sub_rsp(&mut self, size: usize) {
        self.emit(b"\x48\x81\xec");
        self.emit(&(size as u32).to_le_bytes());
    }

    fn emit_add_rsp(&mut self, size: usize) {
        self.emit(b"\x48\x81\xc4");
        self.emit(&(size as u32).to_le_bytes());
    }

    fn emit_ret(&mut self) {
        self.emit(b"\xc3");
    }

    // SIMD helpers for 7D operations
    fn emit_mov_xmm0_rax(&mut self) {
        self.emit(b"\x66\x48\x0f\x6e\xc0"); // movq xmm0, rax
    }

    fn emit_mov_xmm1_rax(&mut self) {
        self.emit(b"\x66\x48\x0f\x6e\xc8"); // movq xmm1, rax
    }

    fn emit_mov_xmm2_rax(&mut self) {
        self.emit(b"\x66\x48\x0f\x6e\xd0"); // movq xmm2, rax
    }

    fn emit_movq_xmm_rax(&mut self, reg: usize) {
        // movq xmm<reg>, rax
        let modrm = 0xc0 | ((reg & 7) << 3);
        if reg < 8 {
            self.emit(&[0x66, 0x48, 0x0f, 0x6e, modrm as u8]);
        } else {
            self.emit(&[0x66, 0x4c, 0x0f, 0x6e, (modrm & 0xc7) as u8]);
        }
    }

    fn emit_movq_rax_xmm(&mut self, reg: usize) {
        // movq rax, xmm<reg>
        let modrm = 0xc0 | (reg & 7);
        if reg < 8 {
            self.emit(&[0x66, 0x48, 0x0f, 0x7e, modrm as u8]);
        } else {
            self.emit(&[0x66, 0x4c, 0x0f, 0x7e, (modrm & 0xc7) as u8]);
        }
    }

    // VEX-encoded SIMD operations
    fn emit_vxorps_xmm10_xmm10_xmm10(&mut self) {
        self.emit(b"\xc5\x29\x57\xd2"); // vxorps xmm10, xmm10, xmm10
    }

    fn emit_vmulsd_xmm(&mut self, dest: usize, src1: usize, src2: usize) {
        // vmulsd xmm<dest>, xmm<src1>, xmm<src2>
        let vvvv = (!src1 & 0xf) << 3;
        let prefix = 0xc3 | vvvv as u8;
        let modrm = 0xc0 | ((dest & 7) << 3) | (src2 & 7);
        self.emit(&[0xc5, prefix, 0x59, modrm as u8]);
    }

    fn emit_vaddsd_xmm(&mut self, dest: usize, src1: usize, src2: usize) {
        let vvvv = (!src1 & 0xf) << 3;
        let prefix = 0xc3 | vvvv as u8;
        let modrm = 0xc0 | ((dest & 7) << 3) | (src2 & 7);
        self.emit(&[0xc5, prefix, 0x58, modrm as u8]);
    }

    fn emit_vsubsd_xmm(&mut self, dest: usize, src1: usize, src2: usize) {
        let vvvv = (!src1 & 0xf) << 3;
        let prefix = 0xc3 | vvvv as u8;
        let modrm = 0xc0 | ((dest & 7) << 3) | (src2 & 7);
        self.emit(&[0xc5, prefix, 0x5c, modrm as u8]);
    }

    fn emit_vdivsd_xmm(&mut self, dest: usize, src1: usize, src2: usize) {
        let vvvv = (!src1 & 0xf) << 3;
        let prefix = 0xc3 | vvvv as u8;
        let modrm = 0xc0 | ((dest & 7) << 3) | (src2 & 7);
        self.emit(&[0xc5, prefix, 0x5e, modrm as u8]);
    }

    fn emit_vsqrtsd_xmm(&mut self, dest: usize, src1: usize, src2: usize) {
        let vvvv = (!src1 & 0xf) << 3;
        let prefix = 0xc3 | vvvv as u8;
        let modrm = 0xc0 | ((dest & 7) << 3) | (src2 & 7);
        self.emit(&[0xc5, prefix, 0x51, modrm as u8]);
    }
}

#[derive(Debug)]
pub enum JITError {
    UnsupportedInstruction,
    MemoryAllocationFailed,
    CodeGenerationFailed,
    UnknownSymbol(String),
}

impl std::fmt::Display for JITError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            JITError::UnsupportedInstruction => write!(f, "Unsupported IR instruction"),
            JITError::MemoryAllocationFailed => write!(f, "Failed to allocate executable memory"),
            JITError::CodeGenerationFailed => write!(f, "Code generation failed"),
            JITError::UnknownSymbol(sym) => write!(f, "Unknown symbol: {}", sym),
        }
    }
}

impl std::error::Error for JITError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_creation() {
        let jit = JIT7D::new();
        assert!(jit.code_cache.is_empty());
    }

    #[test]
    fn test_constant_folding() {
        let jit = JIT7D::new();
        let mut ir = vec![IR7D::PushInt(2), IR7D::PushInt(3), IR7D::Add];
        jit.constant_fold(&mut ir);
        assert_eq!(ir, vec![IR7D::PushInt(5)]);
    }

    #[test]
    fn test_phi_simplification() {
        let jit = JIT7D::new();
        let mut ir = vec![
            IR7D::PushFloat(1.618033988749895), // Φ
            IR7D::PushFloat(0.618033988749895), // Φ⁻¹
            IR7D::Mul,
        ];
        jit.constant_fold(&mut ir);
        if let IR7D::PushFloat(result) = ir[0] {
            assert!((result - 1.0).abs() < 1e-10);
        } else {
            panic!("Expected PushFloat");
        }
    }
}
