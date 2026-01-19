// File: projects/sovereign_assistant/src/tools.rs
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    pub tool: String,
    pub input: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
}

pub struct ToolExecutor;

impl ToolExecutor {
    pub fn execute(call: ToolCall) -> Result<ToolResult> {
        match call.tool.as_str() {
            "sh" | "shell" => Self::execute_sh(&call.input),
            "python" | "py" => Self::execute_python(&call.input),
            "read" | "read_file" => Self::read_file(&call.input),
            "web_search" => Self::execute_web_search(&call.input),
            "code_search" | "find" => Self::execute_code_search(&call.input),
            "write" | "write_file" => {
                let parts: Vec<&str> = call.input.splitn(2, '|').collect();
                if parts.len() == 2 {
                    Self::write_file(parts[0], parts[1])
                } else {
                    Err(anyhow!("Invalid write format. Use 'path|content'"))
                }
            }
            _ => Err(anyhow!("Unknown tool: {}", call.tool)),
        }
    }

    fn execute_sh(command: &str) -> Result<ToolResult> {
        let output = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .arg("-Command")
                .arg(command)
                .output()?
        } else {
            Command::new("sh").arg("-c").arg(command).output()?
        };

        Ok(ToolResult {
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string()
                + &String::from_utf8_lossy(&output.stderr).to_string(),
        })
    }

    fn execute_python(code: &str) -> Result<ToolResult> {
        let output = Command::new("python").arg("-c").arg(code).output()?;

        Ok(ToolResult {
            success: output.status.success(),
            output: String::from_utf8_lossy(&output.stdout).to_string()
                + &String::from_utf8_lossy(&output.stderr).to_string(),
        })
    }

    fn read_file(path: &str) -> Result<ToolResult> {
        match fs::read_to_string(path) {
            Ok(content) => Ok(ToolResult {
                success: true,
                output: content,
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: e.to_string(),
            }),
        }
    }

    fn write_file(path: &str, content: &str) -> Result<ToolResult> {
        match fs::write(path, content) {
            Ok(_) => Ok(ToolResult {
                success: true,
                output: format!("Successfully wrote to {}", path),
            }),
            Err(e) => Ok(ToolResult {
                success: false,
                output: e.to_string(),
            }),
        }
    }

    fn execute_web_search(query: &str) -> Result<ToolResult> {
        let output = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .arg("-Command")
                .arg(format!("curl.exe -s -L \"https://duckduckgo.com/html/?q={}\" | Select-String -Pattern \"result__snippet\" | Select-Object -First 5", query))
                .output()?
        } else {
            Command::new("sh")
                .arg("-c")
                .arg(format!("curl -s -L \"https://duckduckgo.com/html/?q={}\" | grep \"result__snippet\" | head -n 5", query))
                .output()?
        };

        let result = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(ToolResult {
            success: true,
            output: if result.is_empty() {
                "No results found or search blocked.".to_string()
            } else {
                result
            },
        })
    }

    fn execute_code_search(query: &str) -> Result<ToolResult> {
        let output = if cfg!(target_os = "windows") {
            Command::new("powershell")
                .arg("-Command")
                .arg(format!("rg -i -C 2 \"{}\" .", query))
                .output()?
        } else {
            Command::new("grep")
                .arg("-r")
                .arg("-i")
                .arg(query)
                .arg(".")
                .output()?
        };

        let result = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(ToolResult {
            success: true,
            output: if result.is_empty() {
                "No matches found.".to_string()
            } else {
                result
            },
        })
    }
}
