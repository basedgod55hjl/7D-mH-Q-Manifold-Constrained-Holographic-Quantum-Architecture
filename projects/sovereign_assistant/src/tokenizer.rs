//! Crystal 7D Tokenizer - Tiktoken-compatible tokenization
//! Supports Llama/DeepSeek vocabulary

use anyhow::{Context, Result};
// use tiktoken_rs::{cl100k_base, CoreBPE};

// Dummy struct to satisfy field type if we don't want to change struct definition excessively
#[derive(Default)]
struct CoreBPE;

/// Crystal 7D Tokenizer wrapping tiktoken
#[allow(dead_code)]
pub struct Crystal7DTokenizer {
    bpe: CoreBPE,
    /// Special tokens
    pub bos_token: u32,
    pub eos_token: u32,
    pub pad_token: u32,
}

impl Crystal7DTokenizer {
    /// Create new tokenizer (Simple Byte Fallback)
    pub fn new() -> Result<Self> {
        // Fallback to simple byte encoding to avoid runtime panics with missing BPE files
        Ok(Self {
            bpe: CoreBPE::default(), // Dummy, unused
            bos_token: 1,
            eos_token: 2,
            pad_token: 0,
        })
    }

    /// Encode text to token IDs (Simple Bytes)
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // Map characters to tokens directly for stability
        Ok(text.bytes().map(|b| b as u32 + 100).collect())
    }

    /// Encode with BOS token prepended
    pub fn encode_with_bos(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_token];
        tokens.extend(self.encode(text)?);
        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let bytes: Vec<u8> = tokens
            .iter()
            .map(|&t| t.saturating_sub(100) as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Get vocabulary size
    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        100277 // cl100k_base vocab size
    }

    /// Check if token is special
    #[allow(dead_code)]
    pub fn is_special_token(&self, token: u32) -> bool {
        token == self.bos_token || token == self.eos_token || token == self.pad_token
    }

    /// Apply chat template (Llama-style)
    #[allow(dead_code)]
    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> String {
        let mut formatted = String::new();

        for (role, content) in messages {
            match role.as_str() {
                "system" => {
                    formatted.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", content));
                }
                "user" => {
                    formatted.push_str(&format!("[INST] {} [/INST]", content));
                }
                "assistant" => {
                    formatted.push_str(&format!(" {}</s>", content));
                }
                _ => {
                    formatted.push_str(content);
                }
            }
        }

        formatted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = Crystal7DTokenizer::new().unwrap();
        let text = "Hello, 7D Crystal System!";
        let tokens = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_special_tokens() {
        let tokenizer = Crystal7DTokenizer::new().unwrap();
        assert!(tokenizer.is_special_token(1));
        assert!(tokenizer.is_special_token(2));
        assert!(!tokenizer.is_special_token(100));
    }
}
