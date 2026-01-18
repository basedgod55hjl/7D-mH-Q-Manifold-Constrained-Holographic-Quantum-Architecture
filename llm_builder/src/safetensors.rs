// 7D Crystal LLM Builder - SafeTensors Module
// Implements SafeTensors format writer

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use serde_json::json;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub struct SafeTensorsWriter;

impl SafeTensorsWriter {
    pub fn save(path: &Path, tensors: &serde_json::Value) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let json_str = serde_json::to_string(tensors)?;
        let json_bytes = json_str.as_bytes();
        let header_len = json_bytes.len() as u64;

        writer.write_u64::<LittleEndian>(header_len)?;
        writer.write_all(json_bytes)?;

        Ok(())
    }
}
