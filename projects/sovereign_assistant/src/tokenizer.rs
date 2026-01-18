//! Crystal 7D Tokenizer - Tiktoken-compatible tokenization
//! Supports Llama/DeepSeek vocabulary

use anyhow::{Context, Result};
use tiktoken_rs::{cl100k_base, CoreBPE};

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
    /// Create new tokenizer with cl100k_base (GPT-4/DeepSeek compatible)
    pub fn new() -> Result<Self> {
        let bpe = cl100k_base().context("Failed to load cl100k_base tokenizer")?;

        Ok(Self {
            bpe,
            bos_token: 1, // <s>
            eos_token: 2, // </s>
            pad_token: 0, // <pad>
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let tokens = self.bpe.encode_with_special_tokens(text);
        Ok(tokens.into_iter().map(|t| t as u32).collect())
    }

    /// Encode with BOS token prepended
    pub fn encode_with_bos(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = vec![self.bos_token];
        tokens.extend(self.encode(text)?);
        Ok(tokens)
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let tokens_usize: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
        let text = self.bpe.decode(tokens_usize)?;
        Ok(text)
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
