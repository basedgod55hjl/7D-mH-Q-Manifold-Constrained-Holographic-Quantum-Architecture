// src/rewrite/score.rs
use crate::onnx::infer;

pub fn score(before: &str, after: &str) -> f32 {
    let features = vec![
        before.len() as f32,
        after.len() as f32,
        (before.matches("unwrap").count() as f32),
        (after.matches("unwrap").count() as f32),
    ];

    infer(features).unwrap_or(vec![0.0])[0]
}
