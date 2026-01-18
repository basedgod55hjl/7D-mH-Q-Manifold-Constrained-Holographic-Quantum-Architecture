// 7D Crystal LLM Builder - Tokenizer Module
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            merges: Vec::new(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = text.chars().map(|c| c.to_string()).collect::<Vec<_>>();

        // Iteratively apply merges
        for (a, b) in &self.merges {
            let mut new_tokens = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && &tokens[i] == a && &tokens[i + 1] == b {
                    new_tokens.push(format!("{}{}", a, b));
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }
            tokens = new_tokens;
        }

        tokens
            .iter()
            .map(|t| *self.vocab.get(t).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let rev_vocab: HashMap<u32, String> =
            self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();

        ids.iter()
            .map(|id| rev_vocab.get(id).cloned().unwrap_or_default())
            .collect::<Vec<_>>()
            .join("")
    }
}

pub struct UnigramTokenizer {
    pub vocab: Vec<(String, f32)>,
}

impl UnigramTokenizer {
    pub fn new() -> Self {
        Self { vocab: Vec::new() }
    }
}
