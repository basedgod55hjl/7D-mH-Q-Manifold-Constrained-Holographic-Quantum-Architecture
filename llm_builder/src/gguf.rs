// File: llm_builder/src/gguf.rs
// GGUF Tensor Loading for 7D Crystal System

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::io::{BufWriter, Write};
use std::path::Path;

// ============================================================================
// GGUF READER - For loading existing GGUF models
// ============================================================================

pub struct GGUFReader {
    pub tensor_count: usize,
    pub kv_count: usize,
    content: gguf_file::Content,
    file: std::fs::File,
}

impl GGUFReader {
    /// Open a GGUF file for reading
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = std::fs::File::open(path)
            .context(format!("Failed to open GGUF: {}", path.display()))?;

        let content =
            gguf_file::Content::read(&mut file).context("Failed to parse GGUF content")?;

        let tensor_count = content.tensor_infos.len();
        let kv_count = content.metadata.len();

        Ok(Self {
            tensor_count,
            kv_count,
            content,
            file,
        })
    }

    /// Read metadata as key-value pairs
    pub fn read_metadata(&self) -> Result<HashMap<String, String>> {
        let mut meta = HashMap::new();
        for (key, value) in self.content.metadata.iter() {
            meta.insert(key.clone(), format!("{:?}", value));
        }
        Ok(meta)
    }

    /// Load all tensors (dequantized to F32)
    pub fn load_all_tensors(&mut self) -> Result<HashMap<String, Tensor>> {
        let device = Device::Cpu;
        let mut weights = HashMap::new();

        for (tensor_name, tensor_info) in self.content.tensor_infos.iter() {
            let qtensor = tensor_info
                .read(&mut self.file, self.content.tensor_data_offset, &device)
                .context(format!("Failed to read tensor: {}", tensor_name))?;

            let tensor_f32 = qtensor
                .dequantize(&device)
                .context(format!("Failed to dequantize: {}", tensor_name))?;

            weights.insert(tensor_name.clone(), tensor_f32);
        }

        Ok(weights)
    }
}

// Alias for backward compatibility
pub type GGUFLoader = GGUFReader;

impl GGUFLoader {
    /// Load all tensors from a GGUF file (convenience method)
    pub fn load_deepseek_weights(path: &Path) -> Result<HashMap<String, Tensor>> {
        let mut reader = Self::open(path)?;
        reader.load_all_tensors()
    }

    /// Get metadata from GGUF file (convenience method)
    pub fn get_metadata(path: &Path) -> Result<HashMap<String, String>> {
        let reader = Self::open(path)?;
        reader.read_metadata()
    }
}

// ============================================================================
// GGUF WRITER - For model export
// ============================================================================

pub enum GGUFType {
    F32,
    F16,
    Q80,
    Q4KM,
}

impl GGUFType {
    fn to_u32(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q80 => 8,
            Self::Q4KM => 15,
        }
    }
}

pub struct GGUFWriter {
    writer: BufWriter<std::fs::File>,
}

impl GGUFWriter {
    pub fn new(path: &Path) -> Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    pub fn write_header(&mut self, version: u32, tensor_count: u64) -> Result<()> {
        self.writer.write_all(b"GGUF")?;
        self.writer.write_all(&version.to_le_bytes())?;
        self.writer.write_all(&tensor_count.to_le_bytes())?;
        self.writer.write_all(&0u64.to_le_bytes())?;
        Ok(())
    }

    pub fn write_kv_string(&mut self, key: &str, value: &str) -> Result<()> {
        self.writer.write_all(&(key.len() as u64).to_le_bytes())?;
        self.writer.write_all(key.as_bytes())?;
        self.writer.write_all(&8u32.to_le_bytes())?;
        self.writer.write_all(&(value.len() as u64).to_le_bytes())?;
        self.writer.write_all(value.as_bytes())?;
        Ok(())
    }

    pub fn write_kv_u32(&mut self, key: &str, value: u32) -> Result<()> {
        self.writer.write_all(&(key.len() as u64).to_le_bytes())?;
        self.writer.write_all(key.as_bytes())?;
        self.writer.write_all(&4u32.to_le_bytes())?;
        self.writer.write_all(&value.to_le_bytes())?;
        Ok(())
    }

    pub fn write_tensor_info(
        &mut self,
        name: &str,
        dims: &[u64],
        dtype: GGUFType,
        offset: u64,
    ) -> Result<()> {
        self.writer.write_all(&(name.len() as u64).to_le_bytes())?;
        self.writer.write_all(name.as_bytes())?;
        self.writer.write_all(&(dims.len() as u32).to_le_bytes())?;
        for d in dims {
            self.writer.write_all(&d.to_le_bytes())?;
        }
        self.writer.write_all(&dtype.to_u32().to_le_bytes())?;
        self.writer.write_all(&offset.to_le_bytes())?;
        Ok(())
    }

    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}
