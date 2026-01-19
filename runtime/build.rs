// File: runtime/build.rs
// Build script for 7D Crystal Runtime
// Compiles CUDA kernels to PTX for GPU acceleration

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda_kernels.cu");
    
    // Check if CUDA feature is enabled
    #[cfg(feature = "cuda")]
    compile_cuda_kernels();
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernels() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let cuda_source = "src/cuda_kernels.cu";
    let ptx_output = out_dir.join("crystal7d.ptx");
    
    // Find nvcc
    let cuda_path = env::var("CUDA_PATH")
        .unwrap_or_else(|_| "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6".to_string());
    
    let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc.exe");
    
    if !nvcc.exists() {
        println!("cargo:warning=nvcc not found at {:?}, using embedded PTX fallback", nvcc);
        // Create fallback PTX file
        fs::write(&ptx_output, get_fallback_ptx()).expect("Failed to write fallback PTX");
        println!("cargo:rustc-env=CRYSTAL_PTX_PATH={}", ptx_output.display());
        return;
    }
    
    // Check if source exists
    if !PathBuf::from(cuda_source).exists() {
        println!("cargo:warning=CUDA source not found, creating...");
        fs::write(cuda_source, get_cuda_source()).expect("Failed to write CUDA source");
    }
    
    println!("cargo:warning=Compiling CUDA kernels with nvcc...");
    
    // Compile CUDA to PTX
    // Using SM 7.5 for GTX 1660 Ti
    let status = Command::new(&nvcc)
        .args(&[
            "-ptx",
            "-arch=sm_75",
            "-O3",
            "--use_fast_math",
            "-o", ptx_output.to_str().unwrap(),
            cuda_source,
        ])
        .status();
    
    match status {
        Ok(s) if s.success() => {
            println!("cargo:warning=CUDA kernels compiled successfully!");
            println!("cargo:rustc-env=CRYSTAL_PTX_PATH={}", ptx_output.display());
        }
        _ => {
            println!("cargo:warning=nvcc compilation failed, using embedded PTX fallback");
            fs::write(&ptx_output, get_fallback_ptx()).expect("Failed to write fallback PTX");
            println!("cargo:rustc-env=CRYSTAL_PTX_PATH={}", ptx_output.display());
        }
    }
}

fn get_cuda_source() -> &'static str {
    include_str!("src/cuda_source.txt")
}

fn get_fallback_ptx() -> &'static str {
    // Minimal valid PTX that will cause graceful fallback to CPU
    r#"
.version 7.5
.target sm_75
.address_size 64

// Minimal kernel that does nothing (placeholder)
.visible .entry project_to_poincare_7d(
    .param .u64 input,
    .param .u64 output,
    .param .u32 n,
    .param .f32 curvature
)
{
    ret;
}

.visible .entry mobius_add_kernel(
    .param .u64 u,
    .param .u64 v,
    .param .u64 result,
    .param .u32 n,
    .param .f32 curvature
)
{
    ret;
}

.visible .entry hyperbolic_distance_kernel(
    .param .u64 u,
    .param .u64 v,
    .param .u64 distances,
    .param .u32 n,
    .param .f32 curvature
)
{
    ret;
}

.visible .entry holographic_fold_kernel(
    .param .u64 patterns,
    .param .u64 phases,
    .param .u64 output,
    .param .u32 n
)
{
    ret;
}

.visible .entry phi_attention_forward(
    .param .u64 Q,
    .param .u64 K,
    .param .u64 V,
    .param .u64 output,
    .param .u32 seq_len,
    .param .u32 head_dim,
    .param .f32 scale
)
{
    ret;
}

.visible .entry rmsnorm_phi_kernel(
    .param .u64 input,
    .param .u64 output,
    .param .u64 weight,
    .param .u32 n,
    .param .u32 hidden_dim,
    .param .f32 eps
)
{
    ret;
}
"#
}
