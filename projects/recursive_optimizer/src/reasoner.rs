use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct Reasoner {
    client: Client,
    api_url: String,
    model_name: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationDecision {
    pub reasoning: String,
    pub suggested_file: Option<String>,
    pub suggestion: Option<String>,
}

impl Reasoner {
    pub fn new(api_url: &str, model_name: &str) -> Self {
        Self {
            client: Client::new(),
            api_url: api_url.to_string(),
            model_name: model_name.to_string(),
        }
    }

    pub fn analyze_code(&self, context: &str) -> Result<OptimizationDecision> {
        let system_prompt = "You are the 7D Crystal Optimizer. Your goal is to improve the Sovereign System's code. \
        Analyze the provided code context. If you see simple improvements, suggest them. \
        Return your response in JSON format with fields: 'reasoning', 'suggested_file', 'suggestion'. \
        If no changes are needed, set suggested_file to null.";

        let body = json!({
            "model": self.model_name,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": context }
            ],
            "max_tokens": 500,
            "temperature": 0.2
        });

        // Send request
        let res = self
            .client
            .post(&format!("{}/v1/chat/completions", self.api_url))
            .json(&body)
            .send()
            .context("Failed to connect to Inference Server")?;

        let chat_res: ChatResponse = res.json().context("Failed to parse AI response")?;
        let content = &chat_res.choices[0].message.content;

        // Attempt to parse JSON from the LLM response
        // Note: Real implementation needs robust JSON extraction from mixed text
        let decision: OptimizationDecision = match serde_json::from_str(content) {
            Ok(d) => d,
            Err(_) => {
                // Fallback if model didn't output pure JSON
                OptimizationDecision {
                    reasoning: content.clone(),
                    suggested_file: None,
                    suggestion: None,
                }
            }
        };

        Ok(decision)
    }
}
