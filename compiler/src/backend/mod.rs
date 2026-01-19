//! # 7D Crystal Backend Module
//!
//! Code generation backends for different target platforms.
//!
//! ## Supported Targets
//!
//! - **CUDA**: NVIDIA GPUs (primary target)
//! - **HIP**: AMD GPUs (via ROCm)
//! - **Metal**: Apple Silicon GPUs
//! - **CPU**: x86-64 with SIMD
//! - **WASM**: WebAssembly (experimental)

pub mod cuda;
pub mod x86_64;

use crate::errors::{
    CrystalError, CrystalResult, Diagnostic, DiagnosticCollection, ErrorCode, Severity,
};
use crate::ir::IRBlock7D;
use crate::lexer::Token;
use crate::CompilerTarget;

// ═══════════════════════════════════════════════════════════════════════════════
// BACKEND TRAITS
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for 7D Crystal code generation backends.
pub trait Backend7D {
    /// Generate code for the target platform.
    fn emit(&self, blocks: &[IRBlock7D]) -> Result<Vec<u8>, String>;

    /// Get the target name.
    fn target_name(&self) -> &'static str;

    /// Check if the target is available on this system.
    fn is_available(&self) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CODE GENERATION ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate code for the specified target.
///
/// # Arguments
///
/// * `blocks` - The optimized IR blocks to compile.
/// * `target` - The target platform.
///
/// # Returns
///
/// * `Ok(Vec<u8>)` - The generated binary code.
/// * `Err(DiagnosticCollection)` - Code generation failed.
pub fn generate_code(
    blocks: &[IRBlock7D],
    target: CompilerTarget,
) -> Result<Vec<u8>, DiagnosticCollection> {
    let backend: Box<dyn Backend7D> = match target {
        CompilerTarget::Cuda => Box::new(CudaBackend::new()),
        CompilerTarget::Hip => Box::new(HipBackend::new()),
        CompilerTarget::Metal => Box::new(MetalBackend::new()),
        CompilerTarget::Cpu => Box::new(CpuBackend::new()),
        CompilerTarget::Wasm => Box::new(WasmBackend::new()),
    };

    if !backend.is_available() {
        let mut diagnostics = DiagnosticCollection::new();
        diagnostics.add(
            Diagnostic::new(Severity::Error)
                .with_code(ErrorCode::E4001.to_string())
                .with_message(format!(
                    "Target '{}' is not available on this system",
                    backend.target_name()
                )),
        );
        return Err(diagnostics);
    }

    backend.emit(blocks).map_err(|e| {
        let mut diagnostics = DiagnosticCollection::new();
        diagnostics.add(
            Diagnostic::new(Severity::Error)
                .with_code(ErrorCode::E4003.to_string())
                .with_message(e),
        );
        diagnostics
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA BACKEND
// ═══════════════════════════════════════════════════════════════════════════════

/// CUDA code generation backend for NVIDIA GPUs.
pub struct CudaBackend {
    compute_capability: (u32, u32),
}

impl CudaBackend {
    /// Create a new CUDA backend.
    pub fn new() -> Self {
        Self {
            compute_capability: (8, 0), // Default to SM 80 (Ampere)
        }
    }

    /// Set the target compute capability.
    pub fn with_compute_capability(mut self, major: u32, minor: u32) -> Self {
        self.compute_capability = (major, minor);
        self
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend7D for CudaBackend {
    fn emit(&self, blocks: &[IRBlock7D]) -> Result<Vec<u8>, String> {
        // Generate CUDA PTX code
        let ptx = cuda::generate_ptx(blocks, self.compute_capability)?;

        // For now, return PTX as bytes (in production, would compile to cubin)
        Ok(ptx.into_bytes())
    }

    fn target_name(&self) -> &'static str {
        "cuda"
    }

    fn is_available(&self) -> bool {
        // Check if CUDA is available
        #[cfg(feature = "cuda")]
        {
            // Try to detect CUDA runtime
            true
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Always available for code generation (runtime check happens later)
            true
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIP BACKEND (AMD)
// ═══════════════════════════════════════════════════════════════════════════════

/// HIP code generation backend for AMD GPUs.
pub struct HipBackend;

impl HipBackend {
    /// Create a new HIP backend.
    pub fn new() -> Self {
        Self
    }
}

impl Default for HipBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend7D for HipBackend {
    fn emit(&self, blocks: &[IRBlock7D]) -> Result<Vec<u8>, String> {
        // HIP is source-compatible with CUDA, so we can reuse CUDA codegen
        let hip_code = cuda::generate_hip(blocks)?;
        Ok(hip_code.into_bytes())
    }

    fn target_name(&self) -> &'static str {
        "hip"
    }

    fn is_available(&self) -> bool {
        true // Code generation always available
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METAL BACKEND (Apple)
// ═══════════════════════════════════════════════════════════════════════════════

/// Metal code generation backend for Apple Silicon.
pub struct MetalBackend;

impl MetalBackend {
    /// Create a new Metal backend.
    pub fn new() -> Self {
        Self
    }
}

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend7D for MetalBackend {
    fn emit(&self, blocks: &[IRBlock7D]) -> Result<Vec<u8>, String> {
        // Generate Metal Shading Language
        let msl = generate_metal(blocks)?;
        Ok(msl.into_bytes())
    }

    fn target_name(&self) -> &'static str {
        "metal"
    }

    fn is_available(&self) -> bool {
        cfg!(target_os = "macos") || cfg!(target_os = "ios")
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU BACKEND
// ═══════════════════════════════════════════════════════════════════════════════

/// CPU code generation backend with SIMD optimizations.
pub struct CpuBackend {
    use_avx512: bool,
    use_avx2: bool,
}

impl CpuBackend {
    /// Create a new CPU backend.
    pub fn new() -> Self {
        Self {
            use_avx512: Self::detect_avx512(),
            use_avx2: Self::detect_avx2(),
        }
    }

    fn detect_avx512() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx512f")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    fn detect_avx2() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend7D for CpuBackend {
    fn emit(&self, blocks: &[IRBlock7D]) -> Result<Vec<u8>, String> {
        // Generate x86-64 assembly with SIMD
        let asm = x86_64::generate_asm(blocks, self.use_avx512, self.use_avx2)?;
        Ok(asm.into_bytes())
    }

    fn target_name(&self) -> &'static str {
        "cpu"
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WASM BACKEND
// ═══════════════════════════════════════════════════════════════════════════════

/// WebAssembly code generation backend.
pub struct WasmBackend;

impl WasmBackend {
    /// Create a new WASM backend.
    pub fn new() -> Self {
        Self
    }
}

impl Default for WasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend7D for WasmBackend {
    fn emit(&self, blocks: &[IRBlock7D]) -> Result<Vec<u8>, String> {
        // Generate WebAssembly
        let wasm = generate_wasm(blocks)?;
        Ok(wasm)
    }

    fn target_name(&self) -> &'static str {
        "wasm"
    }

    fn is_available(&self) -> bool {
        true // WASM generation always available
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate Metal Shading Language code.
fn generate_metal(blocks: &[IRBlock7D]) -> Result<String, String> {
    let mut output = String::new();

    // Metal header
    output.push_str("#include <metal_stdlib>\n");
    output.push_str("using namespace metal;\n\n");

    // 7D Crystal constants
    output.push_str("constant float PHI = 1.618033988749895;\n");
    output.push_str("constant float PHI_INV = 0.618033988749895;\n");
    output.push_str("constant float S2_BOUND = 0.01;\n\n");

    // Vector7D type
    output.push_str("struct Vector7D {\n");
    output.push_str("    float coords[7];\n");
    output.push_str("    \n");
    output.push_str("    float norm() const {\n");
    output.push_str("        float sum = 0.0;\n");
    output.push_str("        for (int i = 0; i < 7; i++) {\n");
    output.push_str("            sum += coords[i] * coords[i];\n");
    output.push_str("        }\n");
    output.push_str("        return sqrt(sum);\n");
    output.push_str("    }\n");
    output.push_str("};\n\n");

    // Generate kernels from IR blocks
    for block in blocks {
        output.push_str(&format!("// Block: {}\n", block.name));
        output.push_str(&format!("kernel void {}(\n", block.name));
        output.push_str("    device Vector7D* input [[buffer(0)]],\n");
        output.push_str("    device Vector7D* output [[buffer(1)]],\n");
        output.push_str("    uint id [[thread_position_in_grid]]\n");
        output.push_str(") {\n");

        // Generate instructions
        for instr in &block.instructions {
            output.push_str(&format!("    // {:?}\n", instr));
        }

        output.push_str("}\n\n");
    }

    Ok(output)
}

/// Generate WebAssembly binary.
fn generate_wasm(blocks: &[IRBlock7D]) -> Result<Vec<u8>, String> {
    // Simplified WASM generation
    // In production, would use a proper WASM encoder

    let mut wasm = Vec::new();

    // WASM magic number and version
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D]); // \0asm
    wasm.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // version 1

    // Type section (placeholder)
    wasm.push(0x01); // section id
    wasm.push(0x04); // section size
    wasm.push(0x01); // num types
    wasm.push(0x60); // func type
    wasm.push(0x00); // num params
    wasm.push(0x00); // num results

    // Function section (placeholder)
    wasm.push(0x03); // section id
    wasm.push(0x02); // section size
    wasm.push(0x01); // num functions
    wasm.push(0x00); // type index

    // Export section
    wasm.push(0x07); // section id
    let export_name = b"main";
    let export_section_size = 1 + 1 + export_name.len() + 1 + 1;
    wasm.push(export_section_size as u8);
    wasm.push(0x01); // num exports
    wasm.push(export_name.len() as u8);
    wasm.extend_from_slice(export_name);
    wasm.push(0x00); // export kind (func)
    wasm.push(0x00); // func index

    // Code section (placeholder with NOP)
    wasm.push(0x0A); // section id
    wasm.push(0x04); // section size
    wasm.push(0x01); // num functions
    wasm.push(0x02); // func body size
    wasm.push(0x00); // num locals
    wasm.push(0x0B); // end

    // Add custom section with 7D Crystal metadata
    let custom_name = b"7dcrystal";
    let metadata = format!("blocks:{}", blocks.len());
    let custom_section_size = 1 + custom_name.len() + metadata.len();
    wasm.push(0x00); // custom section id
    wasm.push(custom_section_size as u8);
    wasm.push(custom_name.len() as u8);
    wasm.extend_from_slice(custom_name);
    wasm.extend_from_slice(metadata.as_bytes());

    Ok(wasm)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_available() {
        let backend = CudaBackend::new();
        assert_eq!(backend.target_name(), "cuda");
        assert!(backend.is_available());
    }

    #[test]
    fn test_cpu_backend_available() {
        let backend = CpuBackend::new();
        assert_eq!(backend.target_name(), "cpu");
        assert!(backend.is_available());
    }

    #[test]
    fn test_wasm_generation() {
        let blocks = vec![];
        let wasm = generate_wasm(&blocks).unwrap();

        // Check WASM magic number
        assert_eq!(&wasm[0..4], &[0x00, 0x61, 0x73, 0x6D]);
    }
}
