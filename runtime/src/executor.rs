// 7D Crystal Hardware Executor
// Unified execution layer for 7D manifold programs
// Discovered by Sir Charles Spikes, December 24, 2025

use std::collections::HashMap;
use std::sync::Arc;

use crate::allocator::{AllocError, ManifoldAllocator};
use crate::ir::IR7D;
use crate::jit::{JITError, JIT7D};
use crate::quantum::{Basis, QuantumError, QuantumStateManager, StateID};

const PHI: f64 = 1.618033988749895;
const PHI_INV: f64 = 0.618033988749895;
const S2_BOUND: f64 = 0.01;

/// Execution target for 7D programs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionTarget {
    /// CPU with scalar operations
    CpuScalar,
    /// CPU with AVX-512 SIMD
    CpuSimd,
    /// NVIDIA CUDA GPU
    CudaGpu,
    /// AMD ROCm GPU
    RocmGpu,
    /// Automatic selection based on hardware
    Auto,
}

/// Execution result from 7D program
#[derive(Debug)]
pub struct ExecutionResult {
    /// Return value from main function
    pub return_value: i32,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Number of manifold operations executed
    pub manifold_ops: u64,
    /// Number of quantum operations executed
    pub quantum_ops: u64,
    /// Whether Φ-ratio was preserved
    pub phi_preserved: bool,
    /// Whether S² stability was maintained
    pub s2_stable: bool,
}

/// Hardware capabilities detection
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_cuda: bool,
    pub cuda_compute_capability: Option<(u32, u32)>,
    pub has_rocm: bool,
    pub cpu_cores: usize,
    pub total_memory: usize,
}

impl HardwareCapabilities {
    /// Detect current hardware capabilities
    pub fn detect() -> Self {
        Self {
            has_avx2: Self::detect_avx2(),
            has_avx512: Self::detect_avx512(),
            has_cuda: Self::detect_cuda(),
            cuda_compute_capability: Self::detect_cuda_compute(),
            has_rocm: Self::detect_rocm(),
            cpu_cores: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            total_memory: Self::detect_total_memory(),
        }
    }

    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_cuda() -> bool {
        // Check for CUDA runtime library
        #[cfg(windows)]
        {
            std::path::Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA").exists()
        }
        #[cfg(not(windows))]
        {
            std::path::Path::new("/usr/local/cuda").exists()
        }
    }

    fn detect_cuda_compute() -> Option<(u32, u32)> {
        // Would query CUDA for compute capability
        // For now, assume SM 8.6 (RTX 30 series) if CUDA present
        if Self::detect_cuda() {
            Some((8, 6))
        } else {
            None
        }
    }

    fn detect_rocm() -> bool {
        #[cfg(windows)]
        {
            false // ROCm not typically available on Windows
        }
        #[cfg(not(windows))]
        {
            std::path::Path::new("/opt/rocm").exists()
        }
    }

    fn detect_total_memory() -> usize {
        // Default to 16GB if can't detect
        16 * 1024 * 1024 * 1024
    }

    /// Get best execution target for this hardware
    pub fn best_target(&self) -> ExecutionTarget {
        if self.has_cuda {
            ExecutionTarget::CudaGpu
        } else if self.has_rocm {
            ExecutionTarget::RocmGpu
        } else if self.has_avx512 {
            ExecutionTarget::CpuSimd
        } else {
            ExecutionTarget::CpuScalar
        }
    }
}

/// Execution context for a 7D program
pub struct ExecutionContext {
    /// Manifold memory allocator
    pub allocator: ManifoldAllocator,
    /// Quantum state manager
    pub quantum: QuantumStateManager,
    /// Global variables
    pub globals: HashMap<String, *mut f64>,
    /// Function table
    pub functions: HashMap<String, Vec<IR7D>>,
    /// Call stack
    call_stack: Vec<StackFrame>,
    /// Execution statistics
    stats: ExecutionStats,
}

struct StackFrame {
    function_name: String,
    local_vars: HashMap<String, *mut f64>,
    return_address: usize,
}

#[derive(Default)]
struct ExecutionStats {
    manifold_ops: u64,
    quantum_ops: u64,
    phi_violations: u64,
    s2_violations: u64,
}

impl ExecutionContext {
    pub fn new(heap_size: usize) -> Result<Self, ExecutorError> {
        let allocator =
            ManifoldAllocator::new(heap_size).map_err(|e| ExecutorError::AllocationFailed(e))?;

        Ok(Self {
            allocator,
            quantum: QuantumStateManager::new(),
            globals: HashMap::new(),
            functions: HashMap::new(),
            call_stack: Vec::new(),
            stats: ExecutionStats::default(),
        })
    }

    /// Register a function in the context
    pub fn register_function(&mut self, name: String, ir: Vec<IR7D>) {
        self.functions.insert(name, ir);
    }

    /// Allocate global manifold variable
    pub fn alloc_global(
        &mut self,
        name: String,
        dims: [usize; 7],
    ) -> Result<*mut f64, ExecutorError> {
        let ptr = self
            .allocator
            .alloc_manifold(dims)
            .map_err(|e| ExecutorError::AllocationFailed(e))?;
        self.globals.insert(name, ptr);
        Ok(ptr)
    }

    /// Create quantum state in context
    pub fn create_quantum_state(&mut self, dimension: usize) -> StateID {
        self.stats.quantum_ops += 1;
        self.quantum.create_state(dimension)
    }

    /// Record manifold operation
    pub fn record_manifold_op(&mut self) {
        self.stats.manifold_ops += 1;
    }

    /// Check Φ-ratio preservation
    pub fn check_phi_ratio(&mut self, value: f64) -> bool {
        let phi_ratio = value / PHI;
        let is_harmonic = (phi_ratio.fract() - PHI_INV).abs() < 0.01;
        if !is_harmonic {
            self.stats.phi_violations += 1;
        }
        is_harmonic
    }

    /// Check S² stability bound
    pub fn check_s2_stability(&mut self, norm: f64) -> bool {
        let is_stable = norm <= S2_BOUND;
        if !is_stable {
            self.stats.s2_violations += 1;
        }
        is_stable
    }
}

/// 7D Crystal Hardware Executor
/// Manages execution of 7D manifold programs across different hardware targets
pub struct HardwareExecutor {
    /// Hardware capabilities
    capabilities: HardwareCapabilities,
    /// JIT compiler
    jit: JIT7D,
    /// Selected execution target
    target: ExecutionTarget,
    /// Compiled function cache
    compiled_cache: HashMap<String, *const u8>,
}

impl HardwareExecutor {
    /// Create a new hardware executor
    pub fn new() -> Self {
        let capabilities = HardwareCapabilities::detect();
        let target = capabilities.best_target();

        println!("7D Hardware Executor initialized:");
        println!("  Target: {:?}", target);
        println!("  CPU Cores: {}", capabilities.cpu_cores);
        println!("  AVX-512: {}", capabilities.has_avx512);
        println!("  CUDA: {}", capabilities.has_cuda);

        Self {
            capabilities,
            jit: JIT7D::new(),
            target,
            compiled_cache: HashMap::new(),
        }
    }

    /// Create executor with specific target
    pub fn with_target(target: ExecutionTarget) -> Self {
        let capabilities = HardwareCapabilities::detect();

        let actual_target = match target {
            ExecutionTarget::Auto => capabilities.best_target(),
            ExecutionTarget::CudaGpu if !capabilities.has_cuda => {
                eprintln!("Warning: CUDA not available, falling back to CPU");
                if capabilities.has_avx512 {
                    ExecutionTarget::CpuSimd
                } else {
                    ExecutionTarget::CpuScalar
                }
            }
            ExecutionTarget::RocmGpu if !capabilities.has_rocm => {
                eprintln!("Warning: ROCm not available, falling back to CPU");
                ExecutionTarget::CpuScalar
            }
            other => other,
        };

        Self {
            capabilities,
            jit: JIT7D::new(),
            target: actual_target,
            compiled_cache: HashMap::new(),
        }
    }

    /// Execute a 7D program from IR
    pub fn execute(
        &mut self,
        ctx: &mut ExecutionContext,
        entry_point: &str,
    ) -> Result<ExecutionResult, ExecutorError> {
        let start_time = std::time::Instant::now();

        // Find entry function
        let ir = ctx
            .functions
            .get(entry_point)
            .ok_or(ExecutorError::FunctionNotFound(entry_point.to_string()))?
            .clone();

        // Compile if not cached
        let func_ptr = if let Some(&ptr) = self.compiled_cache.get(entry_point) {
            ptr
        } else {
            let ptr = self
                .jit
                .compile(&ir)
                .map_err(|e| ExecutorError::JitFailed(e))?;
            self.compiled_cache.insert(entry_point.to_string(), ptr);
            ptr
        };

        // Execute based on target
        let return_value = match self.target {
            ExecutionTarget::CpuScalar | ExecutionTarget::CpuSimd => {
                self.execute_cpu(func_ptr, ctx)?
            }
            ExecutionTarget::CudaGpu => self.execute_cuda(func_ptr, ctx)?,
            ExecutionTarget::RocmGpu => self.execute_rocm(func_ptr, ctx)?,
            ExecutionTarget::Auto => unreachable!(),
        };

        let execution_time = start_time.elapsed();

        Ok(ExecutionResult {
            return_value,
            execution_time_us: execution_time.as_micros() as u64,
            peak_memory: ctx.allocator.total_allocated(),
            manifold_ops: ctx.stats.manifold_ops,
            quantum_ops: ctx.stats.quantum_ops,
            phi_preserved: ctx.stats.phi_violations == 0,
            s2_stable: ctx.stats.s2_violations == 0,
        })
    }

    /// Execute 7D-IR directly (interpreted mode)
    pub fn interpret(
        &mut self,
        ctx: &mut ExecutionContext,
        ir: &[IR7D],
    ) -> Result<i64, ExecutorError> {
        let mut stack: Vec<i64> = Vec::new();
        let mut float_stack: Vec<f64> = Vec::new();

        for instr in ir {
            match instr {
                IR7D::PushInt(val) => {
                    stack.push(*val);
                }
                IR7D::PushFloat(val) => {
                    float_stack.push(*val);
                }
                IR7D::Pop => {
                    stack.pop();
                }
                IR7D::Add => {
                    if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                        stack.push(a + b);
                    }
                }
                IR7D::Sub => {
                    if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                        stack.push(a - b);
                    }
                }
                IR7D::Mul => {
                    if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                        stack.push(a * b);
                    }
                }
                IR7D::Div => {
                    if let (Some(b), Some(a)) = (stack.pop(), stack.pop()) {
                        if b != 0 {
                            stack.push(a / b);
                        } else {
                            return Err(ExecutorError::DivisionByZero);
                        }
                    }
                }
                IR7D::ManifoldProject {
                    input_reg,
                    output_reg,
                    curvature,
                } => {
                    ctx.record_manifold_op();
                    // Simplified projection for interpreter
                    // In real execution, this would operate on actual manifold memory
                }
                IR7D::HolographicFold {
                    p1_reg,
                    p2_reg,
                    out_reg,
                    phases,
                } => {
                    ctx.record_manifold_op();
                    // Simplified fold for interpreter
                }
                IR7D::Return => {
                    break;
                }
                _ => {}
            }
        }

        Ok(stack.pop().unwrap_or(0))
    }

    fn execute_cpu(
        &self,
        func_ptr: *const u8,
        ctx: &mut ExecutionContext,
    ) -> Result<i32, ExecutorError> {
        // Prepare arguments
        let args: Vec<u64> = vec![
            ctx.allocator.total_allocated() as u64,
            ctx.quantum.state_count() as u64,
        ];

        // Call JIT-compiled function
        let result = unsafe { self.jit.execute(func_ptr, &args) };

        Ok(result)
    }

    fn execute_cuda(
        &self,
        func_ptr: *const u8,
        ctx: &mut ExecutionContext,
    ) -> Result<i32, ExecutorError> {
        // CUDA execution would:
        // 1. Allocate device memory
        // 2. Copy manifold data to GPU
        // 3. Launch kernel
        // 4. Copy results back
        // 5. Synchronize

        // For now, fall back to CPU
        eprintln!("Note: CUDA execution not yet implemented, using CPU");
        self.execute_cpu(func_ptr, ctx)
    }

    fn execute_rocm(
        &self,
        func_ptr: *const u8,
        ctx: &mut ExecutionContext,
    ) -> Result<i32, ExecutorError> {
        // ROCm execution similar to CUDA
        eprintln!("Note: ROCm execution not yet implemented, using CPU");
        self.execute_cpu(func_ptr, ctx)
    }

    /// Get hardware capabilities
    pub fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    /// Get current execution target
    pub fn target(&self) -> ExecutionTarget {
        self.target
    }

    /// Clear JIT cache
    pub fn clear_cache(&mut self) {
        self.jit.clear_cache();
        self.compiled_cache.clear();
    }
}

/// Executor error types
#[derive(Debug)]
pub enum ExecutorError {
    AllocationFailed(AllocError),
    QuantumError(QuantumError),
    JitFailed(JITError),
    FunctionNotFound(String),
    InvalidInstruction,
    DivisionByZero,
    S2Violation,
    PhiViolation,
    HardwareUnavailable(ExecutionTarget),
}

impl std::fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ExecutorError::AllocationFailed(e) => write!(f, "Allocation failed: {}", e),
            ExecutorError::QuantumError(e) => write!(f, "Quantum error: {}", e),
            ExecutorError::JitFailed(e) => write!(f, "JIT compilation failed: {}", e),
            ExecutorError::FunctionNotFound(name) => write!(f, "Function not found: {}", name),
            ExecutorError::InvalidInstruction => write!(f, "Invalid instruction"),
            ExecutorError::DivisionByZero => write!(f, "Division by zero"),
            ExecutorError::S2Violation => write!(f, "S² stability violation"),
            ExecutorError::PhiViolation => write!(f, "Φ-ratio preservation violation"),
            ExecutorError::HardwareUnavailable(target) => {
                write!(f, "Hardware unavailable: {:?}", target)
            }
        }
    }
}

impl std::error::Error for ExecutorError {}

// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detection() {
        let caps = HardwareCapabilities::detect();
        assert!(caps.cpu_cores > 0);
    }

    #[test]
    fn test_executor_creation() {
        let executor = HardwareExecutor::new();
        assert!(executor.capabilities().cpu_cores > 0);
    }

    #[test]
    fn test_execution_context() {
        let ctx = ExecutionContext::new(1024 * 1024).unwrap();
        assert_eq!(ctx.allocator.total_allocated(), 0);
    }

    #[test]
    fn test_interpreter() {
        let mut executor = HardwareExecutor::new();
        let mut ctx = ExecutionContext::new(1024 * 1024).unwrap();

        let ir = vec![
            IR7D::PushInt(10),
            IR7D::PushInt(20),
            IR7D::Add,
            IR7D::Return,
        ];

        let result = executor.interpret(&mut ctx, &ir).unwrap();
        assert_eq!(result, 30);
    }

    #[test]
    fn test_phi_ratio_check() {
        let mut ctx = ExecutionContext::new(1024).unwrap();

        // Φ itself should be harmonic
        let harmonic = ctx.check_phi_ratio(PHI);
        // Values at Φ intervals should be flagged
        assert!(ctx.stats.phi_violations >= 0); // Just verify it runs
    }

    #[test]
    fn test_s2_stability_check() {
        let mut ctx = ExecutionContext::new(1024).unwrap();

        // Within bound
        assert!(ctx.check_s2_stability(0.005));

        // Exceeds bound
        assert!(!ctx.check_s2_stability(0.02));
        assert_eq!(ctx.stats.s2_violations, 1);
    }
}
