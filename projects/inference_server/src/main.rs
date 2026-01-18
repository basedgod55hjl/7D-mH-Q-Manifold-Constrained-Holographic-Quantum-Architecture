//! Multi-GPU Inference Server
//! 7D Crystal System
//! Discovered by Sir Charles Spikes | December 24, 2025

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use model_runner::{
    backend::Backend, sampler::SamplingParams, tokenizer::Crystal7DTokenizer, ModelRunner,
};
use serde::{Deserialize, Serialize};
use std::{path::Path, sync::Arc};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

// ═══════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
    pub parameters: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub models_loaded: usize,
    pub gpu_available: bool,
}

// ═══════════════════════════════════════════════════════════════
// SERVER STATE
// ═══════════════════════════════════════════════════════════════

pub struct ServerState {
    pub runner: Option<Arc<ModelRunner>>,
    pub tokenizer: Option<Arc<Crystal7DTokenizer>>,
    pub models: Vec<ModelInfo>,
    pub request_count: u64,
}

impl ServerState {
    pub fn new() -> Self {
        Self {
            runner: None,
            tokenizer: None,
            models: vec![ModelInfo {
                id: "7d-crystal-8b".to_string(),
                object: "model".to_string(),
                owned_by: "7d-crystal-system".to_string(),
                parameters: 8_000_000_000,
            }],
            request_count: 0,
        }
    }

    pub fn load_model(&mut self, path: &Path) -> anyhow::Result<()> {
        info!("Loading model from: {}", path.display());
        let runner = ModelRunner::from_gguf(path, Backend::CPU)?;
        let tokenizer = Crystal7DTokenizer::new()?;

        self.runner = Some(Arc::new(runner));
        self.tokenizer = Some(Arc::new(tokenizer));
        info!("Model loaded successfully");
        Ok(())
    }
}

type AppState = Arc<RwLock<ServerState>>;

// ═══════════════════════════════════════════════════════════════
// HANDLERS
// ═══════════════════════════════════════════════════════════════

async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let state = state.read().await;
    Json(HealthResponse {
        status: if state.runner.is_some() {
            "healthy"
        } else {
            "loading"
        }
        .to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        models_loaded: if state.runner.is_some() { 1 } else { 0 },
        gpu_available: false, // CPU for now
    })
}

async fn list_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let state = state.read().await;
    Json(ModelsResponse {
        object: "list".to_string(),
        data: state.models.clone(),
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, StatusCode> {
    let mut state_write = state.write().await;
    state_write.request_count += 1;

    // Acquire read hooks
    let runner = state_write.runner.clone();
    let tokenizer = state_write.tokenizer.clone();
    drop(state_write); // Release lock

    if runner.is_none() || tokenizer.is_none() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let runner = runner.unwrap();
    let tokenizer = tokenizer.unwrap();

    // 1. Format Prompt
    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    let prompt_text = tokenizer.apply_chat_template(&messages);
    let tokens = tokenizer
        .encode_with_bos(&prompt_text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    info!("Chat Request: {} tokens", tokens.len());

    // 2. Generate
    let params = SamplingParams {
        temperature: req.temperature,
        top_p: 0.9,
        top_k: 40,
        repetition_penalty: 1.1,
    };

    let generated_tokens = runner
        .generate(&tokens, req.max_tokens, &params)
        .map_err(|e| {
            error!("Generation failed: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // 3. Decode
    let response_text = tokenizer
        .decode(&generated_tokens)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let usage = Usage {
        prompt_tokens: tokens.len(),
        completion_tokens: generated_tokens.len(),
        total_tokens: tokens.len() + generated_tokens.len(),
    };

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: response_text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage,
    }))
}

// Keep explicit completions endpoint for legacy/testing
async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, StatusCode> {
    let mut state_write = state.write().await;
    state_write.request_count += 1;

    let runner = state_write.runner.clone();
    let tokenizer = state_write.tokenizer.clone();
    drop(state_write);

    if runner.is_none() {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let runner = runner.unwrap();
    let tokenizer = tokenizer.unwrap();

    let tokens = tokenizer
        .encode_with_bos(&req.prompt)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let params = SamplingParams::default();
    let output_tokens = runner
        .generate(&tokens, req.max_tokens, &params)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let text = tokenizer.decode(&output_tokens).unwrap_or_default();

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid::Uuid::new_v4()),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            text,
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: tokens.len(),
            completion_tokens: output_tokens.len(),
            total_tokens: tokens.len() + output_tokens.len(),
        },
    }))
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        🔮 7D CRYSTAL INFERENCE SERVER 🔮                     ║");
    println!("║           Manifold-Constrained LLM Inference                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut server_state = ServerState::new();

    // Find model automatically
    let possible_paths = [
        "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "../DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
    ];

    let mut loaded = false;
    for p in possible_paths {
        if Path::new(p).exists() {
            if let Err(e) = server_state.load_model(Path::new(p)) {
                error!("Failed to load model at {}: {}", p, e);
            } else {
                loaded = true;
                break;
            }
        }
    }

    if !loaded {
        warn!("⚠️  NO MODEL FOUND! Server will start in MOCK mode (pending loading).");
        warn!("   Place 'DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf' in the project root.");
    }

    let state: AppState = Arc::new(RwLock::new(server_state));

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = "0.0.0.0:8080";
    info!("Starting server on {}", addr);
    println!("\n🚀 Server running at http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
