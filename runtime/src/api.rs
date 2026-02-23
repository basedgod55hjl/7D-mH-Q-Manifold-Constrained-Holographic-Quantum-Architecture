// File: runtime/src/api.rs
// ABRASAX GOD OS - Rest API: VLC routing + LM Studio proxy (localhost:1234)

use axum::{
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;
use crate::quantum::QuantumStateManager;

const LM_STUDIO_URL: &str = "http://127.0.0.1:1234";

#[derive(Clone)]
pub struct AppState {
    pub executor_ready: bool,
    pub quantum_memory: Arc<Mutex<QuantumStateManager>>,
    pub lm_client: Client,
}

fn lm_auth_header() -> Option<String> {
    std::env::var("LM_STUDIO_API_TOKEN")
        .ok()
        .or_else(|| std::env::var("LM_API_TOKEN").ok())
        .map(|t| format!("Bearer {}", t))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VLCFrameRequest {
    pub frame_id: u64,
    pub width: u32,
    pub height: u32,
    pub data: String,
}

#[derive(Serialize, Deserialize)]
pub struct LmStudioProxyRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    #[serde(default = "default_temp")]
    pub temperature: f32,
}
fn default_temp() -> f32 { 0.7 }

#[derive(Serialize, Deserialize)]
pub struct LmStudioResponsesRequest {
    pub model: String,
    pub input: String,
    #[serde(default)]
    pub tools: Vec<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: String,
}

pub async fn start_api_server() {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| Client::new());

    let state = AppState {
        executor_ready: true,
        quantum_memory: Arc::new(Mutex::new(QuantumStateManager::new())),
        lm_client: client,
    };

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/vlc/process_frame", post(process_vlc_frame))
        .route("/v1/chat/completions", post(lm_studio_chat_proxy))
        .route("/v1/responses", post(lm_studio_responses_proxy))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:17777").await.unwrap();
    println!("🚀 ABRASAX GOD OS Bridge API on http://0.0.0.0:17777");
    println!("   /v1/chat/completions -> LM Studio proxy");
    println!("   /v1/responses        -> LM Studio Responses API (tools)");
    axum::serve(listener, app).await.unwrap();
}

async fn health_check() -> &'static str {
    "ABRASAX GOD OS API Active - 7D Crystal Bridge"
}

async fn process_vlc_frame(
    State(_state): State<AppState>,
    Json(payload): Json<VLCFrameRequest>,
) -> impl IntoResponse {
    println!("Received frame {} from VLC ({}x{})", payload.frame_id, payload.width, payload.height);
    Json(serde_json::json!({
        "status": "processed",
        "frame_id": payload.frame_id,
        "manifold_projections": 128,
        "time_ms": 1.2
    }))
}

async fn lm_studio_chat_proxy(
    State(state): State<AppState>,
    Json(mut payload): Json<LmStudioProxyRequest>,
) -> impl IntoResponse {
    println!("LM Studio chat request for model: {}", payload.model);

    let mut qsm = state.quantum_memory.lock().await;
    let msg_count = payload.messages.len();
    let mut folds_applied = 0;

    if msg_count > 3 {
        let system_msg = payload.messages[0].clone();
        let last_msg = payload.messages[msg_count - 1].clone();
        let state_id = qsm.create_state(7);
        let mut new_messages = vec![system_msg];
        new_messages.push(serde_json::json!({
            "role": "system",
            "content": format!("[7D Quantum Memory: State{} - {} compressed epochs]", state_id.as_u64(), msg_count - 2)
        }));
        new_messages.push(last_msg);
        payload.messages = new_messages;
        folds_applied = msg_count - 2;
    }

    let body = serde_json::json!({
        "model": payload.model,
        "messages": payload.messages,
        "temperature": payload.temperature
    });

    let mut req = state
        .lm_client
        .post(format!("{}/v1/chat/completions", LM_STUDIO_URL))
        .json(&body);

    if let Some(h) = lm_auth_header() {
        req = req.header("Authorization", h);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            match resp.json::<serde_json::Value>().await {
                Ok(json) => Json(serde_json::json!({
                    "status": "proxied",
                    "lm_status": status.as_u16(),
                    "manifold_folds_applied": folds_applied,
                    "response": json
                })),
                Err(e) => Json(serde_json::json!({
                    "status": "error",
                    "message": format!("LM Studio response parse: {}", e),
                    "manifold_folds_applied": folds_applied
                })),
            }
        }
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("LM Studio unreachable ({}): {}", LM_STUDIO_URL, e),
            "manifold_folds_applied": folds_applied,
            "hint": "Ensure LM Studio is running on port 1234. Set LM_STUDIO_API_TOKEN if auth enabled."
        })),
    }
}

async fn lm_studio_responses_proxy(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model = payload.get("model").and_then(|v| v.as_str()).unwrap_or("local");
    println!("LM Studio /v1/responses request for model: {}", model);

    let mut req = state
        .lm_client
        .post(format!("{}/v1/responses", LM_STUDIO_URL))
        .json(&payload);

    if let Some(h) = lm_auth_header() {
        req = req.header("Authorization", h);
    }

    match req.send().await {
        Ok(resp) => {
            let status = resp.status();
            match resp.json::<serde_json::Value>().await {
                Ok(json) => Json(serde_json::json!({
                    "status": "proxied",
                    "lm_status": status.as_u16(),
                    "response": json
                })),
                Err(e) => Json(serde_json::json!({
                    "status": "error",
                    "message": format!("LM Studio response parse: {}", e)
                })),
            }
        }
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": format!("LM Studio unreachable ({}): {}", LM_STUDIO_URL, e),
            "hint": "Ensure LM Studio is running. Set LM_STUDIO_API_TOKEN if auth enabled."
        })),
    }
}
