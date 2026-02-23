// ABRASAX GOD OS - GPU Module (CPU fallback when CUDA unavailable)
// Sir Charles Spikes | Cincinnati, Ohio, USA

pub use crate::{Vector7D, PHI, PHI_INV, S2_STABILITY};

#[derive(Debug, Clone)]
pub struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for GpuError {}

#[derive(Debug, Default)]
pub struct GpuStats {
    pub ops: u64,
}

pub struct GpuExecutor;

impl GpuExecutor {
    pub fn new() -> Result<Self, GpuError> {
        Err(GpuError("GPU not available (use cpu-only build)".to_string()))
    }

    pub fn load_kernels(&mut self, _src: &str) -> Result<(), GpuError> {
        Ok(())
    }

    pub fn device_info(&self) -> &'static str {
        "CPU fallback"
    }

    pub fn project_to_poincare_batch(
        &mut self,
        _v: &mut [Vector7D],
        _curvature: f32,
    ) -> Result<Vec<Vector7D>, GpuError> {
        Err(GpuError("GPU not available".to_string()))
    }

    pub fn mobius_add_batch(
        &mut self,
        _u: &[Vector7D],
        _v: &[Vector7D],
        _curvature: f32,
    ) -> Result<Vec<Vector7D>, GpuError> {
        Err(GpuError("GPU not available".to_string()))
    }

    pub fn hyperbolic_distance_batch(
        &mut self,
        _u: &[Vector7D],
        _v: &[Vector7D],
        _curvature: f32,
    ) -> Result<Vec<f32>, GpuError> {
        Err(GpuError("GPU not available".to_string()))
    }

    pub fn holographic_fold_batch(
        &mut self,
        _patterns: &[Vector7D],
        _phases: &[f32],
    ) -> Result<Vec<Vector7D>, GpuError> {
        Err(GpuError("GPU not available".to_string()))
    }
}

pub fn project_to_poincare_cpu(v: &mut Vector7D, curvature: f32) {
    let projected = v.project(curvature as f64);
    v.coords = projected.coords;
}

pub fn mobius_add_cpu(u: &Vector7D, v: &Vector7D, curvature: f32) -> Vector7D {
    let k = curvature as f64;
    let u_norm_sq = u.norm_squared();
    let v_norm_sq = v.norm_squared();
    let uv = u.coords.iter().zip(v.coords.iter()).map(|(a, b)| a * b).sum::<f64>();
    let denom = 1.0 + k * k * u_norm_sq * v_norm_sq + 2.0 * k * uv;
    let mut coords = [0.0; 7];
    for i in 0..7 {
        coords[i] = ((1.0 + 2.0 * k * uv + k * v_norm_sq) * u.coords[i]
            + (1.0 - k * u_norm_sq) * v.coords[i])
            / denom;
    }
    Vector7D::new(coords)
}

pub fn hyperbolic_distance_cpu(u: &Vector7D, v: &Vector7D, curvature: f32) -> f32 {
    let diff_sq: f64 = u
        .coords
        .iter()
        .zip(v.coords.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let diff = diff_sq.sqrt();
    let k = curvature as f64;
    let u_norm_sq = u.norm_squared();
    let v_norm_sq = v.norm_squared();
    let num = 2.0 * k * diff * diff;
    let den = (1.0 - k * u_norm_sq).max(1e-10) * (1.0 - k * v_norm_sq).max(1e-10);
    (1.0 + num / den).acosh() as f32
}
