use anyhow::{Context, Result};
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("=== 7D Crystal CUDA Verifier ===");

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: verify_cuda <path-to-ptx-file>");
        return Ok(());
    }

    let ptx_path = PathBuf::from(&args[1]);
    println!("Loading PTX from: {:?}", ptx_path);

    let ptx_content = fs::read_to_string(&ptx_path)
        .with_context(|| format!("Failed to read PTX file: {:?}", ptx_path))?;

    // Initialize CUDA Device (Card 0)
    let dev =
        CudaDevice::new(0).context("Failed to initialize CUDA device. Is a GPU available?")?;
    println!("CUDA Device 0 initialized: {:?}", dev.name());

    // Load PTX Module
    // We use load_ptx asking for specific safe names or just load the whole thing
    let ptx = Ptx::from_src(&ptx_content);
    dev.load_ptx(
        ptx,
        "crystal_module",
        &["manifold_project_7d", "holographic_fold_7d"],
    )
    .context("Failed to load PTX module")?;

    println!("PTX Module loaded successfully.");

    // Sacred constants initialized in PTX for this test
    println!("Sacred constants (hardcoded in PTX) ready.");

    // Prepare Data for verify_main
    const COUNT: usize = 1024;
    // verify_main takes (input_ptr, output_ptr, count)
    // It does its own internal logic generation, but let's provide valid pointers.

    // Allocate buffers
    // Input won't be strictly used by verify_main's hardcoded logic but we allocate to be safe
    // Initialize input with 1.0s for consistent verification
    let input_host = vec![1.0f64; COUNT * 7];
    let input_dev = dev.htod_copy(input_host)?;

    println!("Preparing to execute kernel: manifold_project_7d");

    let func = dev
        .get_func("crystal_module", "manifold_project_7d")
        .context("Could not find kernel 'manifold_project_7d'")?;

    let curvature = 0.618033988749895f64;
    let cfg = LaunchConfig::for_num_elems(COUNT as u32);

    // Verify result logic
    // Input vector [1.0; 7], norm = sqrt(7) = 2.64575131106
    // Denom = 1 + norm + PHI_INV + curvature.abs()
    // Denom = 1 + 2.645751 + 0.618034 + 0.618034 = 4.881819
    // Scale = 1 / 4.881819 = 0.2048416
    let expected = 0.2048416;

    // Use f64 view for output
    // Re-allocate correctly as f64 with proper size (COUNT * 7)
    let mut output_dev_f64 = dev.alloc_zeros::<f64>(COUNT * 7)?;

    unsafe { func.launch(cfg, (&input_dev, &output_dev_f64, curvature, COUNT as u64)) }
        .context("Failed to launch kernel")?;

    dev.synchronize()?;
    println!("Kernel execution completed.");

    let output_host = dev.dtoh_sync_copy(&output_dev_f64)?;

    let first_val = output_host[0];
    println!("First output value: {}", first_val);
    println!("Expected value: ~{}", expected);

    // Check first vector components
    for i in 0..7 {
        let val = output_host[i];
        if (val - expected).abs() > 1e-4 {
            eprintln!(
                "Verification FAILED at index {}: got {}, expected {}",
                i, val, expected
            );
            std::process::exit(1);
        }
    }

    println!("VERIFICATION SUCCESS: Output matches 7D Manifold Projection Constraints.");
    Ok(())
}
