// src/rewrite/debate.rs
use serde_json::json;

pub fn debate(candidates: Vec<String>) -> String {
    let winner = candidates.into_iter().max_by_key(|c| c.len()).unwrap();

    std::fs::write(
        "knowledge/debates.json",
        json!({ "winner": winner }).to_string(),
    )
    .ok();

    winner
}
