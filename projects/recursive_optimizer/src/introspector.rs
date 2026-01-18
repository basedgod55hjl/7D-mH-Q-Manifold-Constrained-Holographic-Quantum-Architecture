use anyhow::Result;
use std::fs;
// use std::path::Path; // unused
use walkdir::WalkDir;

pub fn read_codebase(root_path: &str) -> Result<String> {
    let mut total_code = String::new();

    for entry in WalkDir::new(root_path).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(path) {
                total_code.push_str(&format!("\n// File: {:?}\n", path));
                total_code.push_str(&content);
            }
        }
    }

    Ok(total_code)
}
