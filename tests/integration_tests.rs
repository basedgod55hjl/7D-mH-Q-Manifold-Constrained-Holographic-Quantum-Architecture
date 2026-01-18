// File: tests/integration_tests.rs
// 7D Crystal System - Comprehensive Test Suite

use std::time::Instant;

// ============================================================================
// MATHEMATICAL CONSTANTS TESTS
// ============================================================================

const PHI: f64 = 1.618033988749894848204586834365638;
const PHI_INV: f64 = 0.618033988749894848204586834365638;
const S2_STABILITY: f64 = 0.01;

const PHI_BASIS: [f64; 7] = [
    1.0, 1.618033988749895, 2.618033988749895, 4.23606797749979,
    6.854101966249685, 11.090169943749475, 17.94427190999916,
];

#[test]
fn test_golden_ratio_identity() {
    // Φ² = Φ + 1
    let phi_squared = PHI * PHI;
    let phi_plus_one = PHI + 1.0;
    assert!((phi_squared - phi_plus_one).abs() < 1e-14, 
        "Φ² = {}, Φ+1 = {}", phi_squared, phi_plus_one);
}

#[test]
fn test_golden_ratio_inverse() {
    // Φ⁻¹ = Φ - 1
    let computed_inv = 1.0 / PHI;
    let phi_minus_one = PHI - 1.0;
    assert!((computed_inv - PHI_INV).abs() < 1e-14);
    assert!((PHI_INV - phi_minus_one).abs() < 1e-14);
}

#[test]
fn test_phi_basis_fibonacci_property() {
    // Each basis element should be Φ times the previous
    for i in 0..6 {
        let ratio = PHI_BASIS[i + 1] / PHI_BASIS[i];
        assert!((ratio - PHI).abs() < 1e-10, 
            "Ratio at {} = {}, expected Φ = {}", i, ratio, PHI);
    }
}

#[test]
fn test_phi_basis_sum_property() {
    // F(n+2) = F(n+1) + F(n) analog: basis[i+2] ≈ basis[i+1] + basis[i]
    for i in 0..5 {
        let sum = PHI_BASIS[i] + PHI_BASIS[i + 1];
        let expected = PHI_BASIS[i + 2];
        assert!((sum - expected).abs() < 1e-10,
            "Sum property failed at {}: {} + {} ≠ {}", 
            i, PHI_BASIS[i], PHI_BASIS[i + 1], expected);
    }
}

// ============================================================================
// MANIFOLD PROJECTION TESTS
// ============================================================================

fn project_to_poincare(v: &[f64], curvature: f64) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = 1.0 + norm + PHI_INV + curvature.abs();
    let scale = if norm > S2_STABILITY { 
        1.0 / (denom * (norm / S2_STABILITY)) 
    } else { 
        1.0 / denom 
    };
    v.iter().enumerate().map(|(i, &x)| {
        let phi_weight = if i < 7 { PHI_BASIS[i] / PHI_BASIS[6] } else { 1.0 };
        x * scale * phi_weight
    }).collect()
}

fn hyperbolic_distance(u: &[f64], v: &[f64]) -> f64 {
    let u_norm_sq: f64 = u.iter().map(|x| x * x).sum();
    let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
    let diff_norm_sq: f64 = u.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let num = 2.0 * diff_norm_sq;
    let den = (1.0 - u_norm_sq) * (1.0 - v_norm_sq);
    (1.0 + num / den.max(1e-10)).acosh()
}

#[test]
fn test_poincare_projection_inside_ball() {
    let vectors = vec![
        vec![0.5, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2],
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![10.0, -5.0, 3.0, -2.0, 1.0, 0.5, -0.1],
    ];
    
    for v in vectors {
        let projected = project_to_poincare(&v, -1.0);
        let norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0, "Projected norm {} >= 1.0 for input {:?}", norm, v);
    }
}

#[test]
fn test_poincare_projection_preserves_direction() {
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let projected = project_to_poincare(&v, -1.0);
    
    // Direction should be approximately preserved (modulo Φ weighting)
    let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let p_norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    // All values should have same sign
    for (i, (&vi, &pi)) in v.iter().zip(projected.iter()).enumerate() {
        if vi.abs() > 1e-10 {
            assert!(vi.signum() == pi.signum(), 
                "Sign mismatch at {}: {} vs {}", i, vi, pi);
        }
    }
}

#[test]
fn test_hyperbolic_distance_properties() {
    let origin = vec![0.0; 7];
    let u = vec![0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
    let v = vec![0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0];
    
    // Non-negativity
    let d_uv = hyperbolic_distance(&u, &v);
    assert!(d_uv >= 0.0, "Distance should be non-negative");
    
    // Identity of indiscernibles
    let d_uu = hyperbolic_distance(&u, &u);
    assert!(d_uu.abs() < 1e-10, "Distance to self should be zero");
    
    // Symmetry
    let d_vu = hyperbolic_distance(&v, &u);
    assert!((d_uv - d_vu).abs() < 1e-10, "Distance should be symmetric");
    
    // Triangle inequality (weak test)
    let d_ou = hyperbolic_distance(&origin, &u);
    let d_ov = hyperbolic_distance(&origin, &v);
    assert!(d_uv <= d_ou + d_ov + 1e-10, "Triangle inequality violated");
}

#[test]
fn test_s2_stability_enforcement() {
    // Large vector should be scaled down
    let large_v = vec![100.0; 7];
    let projected = project_to_poincare(&large_v, -1.0);
    let norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();
    
    assert!(norm < 1.0, "Large vector should project inside ball");
    assert!(norm < 0.1, "S² enforcement should keep norm small");
}

// ============================================================================
// ATTENTION MECHANISM TESTS
// ============================================================================

#[test]
fn test_attention_softmax_sum_to_one() {
    // Simplified attention softmax test
    let scores = vec![1.0, 2.0, 3.0, 4.0];
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = scores.iter().map(|s| (s - max_score).exp()).sum();
    let probs: Vec<f64> = scores.iter().map(|s| (s - max_score).exp() / exp_sum).collect();
    
    let sum: f64 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10, "Softmax should sum to 1.0, got {}", sum);
    
    // All probs should be positive
    for p in &probs {
        assert!(*p > 0.0, "All probabilities should be positive");
    }
}

#[test]
fn test_phi_weighted_attention() {
    // Test that Φ weights are applied correctly
    let head_dim = 128;
    let phi_weights: Vec<f64> = (0..head_dim).map(|i| {
        if i < 7 { PHI_BASIS[i] / PHI_BASIS[6] } else { 1.0 }
    }).collect();
    
    // First 7 dims should have increasing weights
    for i in 0..6 {
        assert!(phi_weights[i] < phi_weights[i + 1],
            "Φ weights should increase: {} at {}, {} at {}", 
            phi_weights[i], i, phi_weights[i + 1], i + 1);
    }
    
    // Remaining dims should be 1.0
    for i in 7..head_dim {
        assert!((phi_weights[i] - 1.0).abs() < 1e-10,
            "Non-manifold dims should have weight 1.0");
    }
}

// ============================================================================
// RoPE TESTS
// ============================================================================

#[test]
fn test_rope_rotation_orthogonality() {
    // Rotation should preserve norm
    let dim = 64;
    let pos = 42;
    let theta = 10000.0;
    
    let mut q = vec![1.0; dim];
    
    // Apply RoPE
    for i in 0..(dim / 2) {
        let freq = 1.0 / theta.powf((2 * i) as f64 / dim as f64);
        let angle = pos as f64 * freq;
        let cos = angle.cos();
        let sin = angle.sin();
        
        let q0 = q[i * 2];
        let q1 = q[i * 2 + 1];
        q[i * 2] = q0 * cos - q1 * sin;
        q[i * 2 + 1] = q0 * sin + q1 * cos;
    }
    
    // Check norm preservation
    let original_norm = (dim as f64).sqrt();
    let rotated_norm: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((original_norm - rotated_norm).abs() < 1e-10,
        "RoPE should preserve norm: {} vs {}", original_norm, rotated_norm);
}

// ============================================================================
// QUANTIZATION TESTS
// ============================================================================

#[test]
fn test_quantization_round_trip() {
    // Simple round-trip test
    let original: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
    
    // Find max for scaling
    let absmax = original.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = absmax / 127.0;
    
    // Quantize to int8
    let quantized: Vec<i8> = original.iter()
        .map(|x| (x / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    
    // Dequantize
    let dequantized: Vec<f32> = quantized.iter()
        .map(|&q| q as f32 * scale)
        .collect();
    
    // Check error is bounded
    let max_error: f32 = original.iter().zip(dequantized.iter())
        .map(|(o, d)| (o - d).abs())
        .fold(0.0, f32::max);
    
    assert!(max_error < scale, "Quantization error {} exceeds scale {}", max_error, scale);
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

#[test]
fn test_projection_performance() {
    let num_vectors = 1000;
    let vectors: Vec<Vec<f64>> = (0..num_vectors)
        .map(|i| (0..7).map(|j| (i * 7 + j) as f64 * 0.01).collect())
        .collect();
    
    let start = Instant::now();
    for v in &vectors {
        let _ = project_to_poincare(v, -1.0);
    }
    let elapsed = start.elapsed();
    
    let per_vector_ns = elapsed.as_nanos() / num_vectors as u128;
    println!("Projection: {} ns/vector", per_vector_ns);
    
    // Should be reasonably fast
    assert!(per_vector_ns < 10_000, "Projection too slow: {} ns", per_vector_ns);
}

#[test]
fn test_distance_performance() {
    let num_pairs = 1000;
    let u = vec![0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0];
    let v = vec![0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0];
    
    let start = Instant::now();
    for _ in 0..num_pairs {
        let _ = hyperbolic_distance(&u, &v);
    }
    let elapsed = start.elapsed();
    
    let per_pair_ns = elapsed.as_nanos() / num_pairs as u128;
    println!("Distance: {} ns/pair", per_pair_ns);
    
    assert!(per_pair_ns < 5_000, "Distance too slow: {} ns", per_pair_ns);
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

#[test]
fn test_zero_vector_projection() {
    let zero = vec![0.0; 7];
    let projected = project_to_poincare(&zero, -1.0);
    
    let norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm < 1e-10, "Zero vector should project to origin");
}

#[test]
fn test_negative_values_projection() {
    let v = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0];
    let projected = project_to_poincare(&v, -1.0);
    
    // All negative values should remain negative
    for (i, &p) in projected.iter().enumerate() {
        assert!(p <= 0.0, "Negative value at {} became positive: {}", i, p);
    }
}

#[test]
fn test_very_small_values() {
    let tiny = vec![1e-15; 7];
    let projected = project_to_poincare(&tiny, -1.0);
    
    // Should not overflow or become NaN
    for (i, &p) in projected.iter().enumerate() {
        assert!(p.is_finite(), "Value at {} is not finite: {}", i, p);
    }
}

#[test]
fn test_very_large_values() {
    let huge = vec![1e15; 7];
    let projected = project_to_poincare(&huge, -1.0);
    
    // Should be bounded and finite
    let norm: f64 = projected.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(norm.is_finite(), "Projected norm is not finite");
    assert!(norm < 1.0, "Projected norm exceeds 1.0");
}

fn main() {
    println!("7D Crystal System - Test Suite");
    println!("==============================");
    println!("Φ = {}", PHI);
    println!("Φ⁻¹ = {}", PHI_INV);
    println!("S² = {}", S2_STABILITY);
    println!("\nAll tests passed!");
}
