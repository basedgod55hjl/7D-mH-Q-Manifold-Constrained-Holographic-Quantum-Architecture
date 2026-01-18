// File: model_runner/src/backend.rs
// Compute Backend Abstraction
// 7D Crystal System

/// Available compute backends
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Backend {
    CPU,
    CUDA(usize), // GPU index
    Metal,
}

impl Default for Backend {
    fn default() -> Self {
        Backend::CPU
    }
}

impl Backend {
    pub fn cuda_if_available() -> Self {
        // Check CUDA availability at runtime
        #[cfg(feature = "cuda")]
        {
            if cudarc::driver::CudaDevice::new(0).is_ok() {
                return Backend::CUDA(0);
            }
        }
        Backend::CPU
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backend::CPU => "CPU",
            Backend::CUDA(_idx) => "CUDA",
            Backend::Metal => "Metal",
        }
    }
}
