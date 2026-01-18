// src/rewrite/mod.rs
use crate::{task::Task, reward};
use super::*;

pub mod ast;
pub mod score;
pub mod debate;
pub mod cross_lang;
pub mod shadow;

pub fn attempt(path: &str, code: &str, task: &Task) {
    let ast = ast::parse_rust(code);

    if !ast::has_unwrap(&ast) {
        return;
    }

    let proposal = code.replace("unwrap()", "expect(\"safe\")");
    let score = score::score(code, &proposal);

    if score > task.confidence_required {
        if shadow::shadow_test("cargo test") {
            std::fs::write(path, proposal).unwrap();
            reward::reward(&task.id, score);
        }
    }
}
