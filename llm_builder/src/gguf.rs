use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
pub const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone, Copy)]
pub enum GGUFType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
}

pub struct GGUFWriter {
    pub writer: BufWriter<File>,
}

impl GGUFWriter {
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write Magic and Version
        writer.write_u32::<LittleEndian>(GGUF_MAGIC)?;
        writer.write_u32::<LittleEndian>(GGUF_VERSION)?;

        Ok(Self { writer })
    }

    pub fn write_header(&mut self, tensor_count: u64, kv_count: u64) -> Result<()> {
        self.writer.write_u64::<LittleEndian>(tensor_count)?;
        self.writer.write_u64::<LittleEndian>(kv_count)?;
        Ok(())
    }

    pub fn write_kv_string(&mut self, key: &str, value: &str) -> Result<()> {
        self.write_string(key)?;
        self.writer.write_u32::<LittleEndian>(1)?; // GGUF_METADATA_VALUE_TYPE_STRING
        self.write_string(value)?;
        Ok(())
    }

    pub fn write_kv_u32(&mut self, key: &str, value: u32) -> Result<()> {
        self.write_string(key)?;
        self.writer.write_u32::<LittleEndian>(4)?; // GGUF_METADATA_VALUE_TYPE_UINT32
        self.writer.write_u32::<LittleEndian>(value)?;
        Ok(())
    }

    fn write_string(&mut self, s: &str) -> Result<()> {
        let bytes = s.as_bytes();
        self.writer.write_u64::<LittleEndian>(bytes.len() as u64)?;
        self.writer.write_all(bytes)?;
        Ok(())
    }

    pub fn write_tensor_info(
        &mut self,
        name: &str,
        dims: &[u64],
        ggml_type: GGUFType,
        offset: u64,
    ) -> Result<()> {
        self.write_string(name)?;
        self.writer.write_u32::<LittleEndian>(dims.len() as u32)?;
        for &dim in dims {
            self.writer.write_u64::<LittleEndian>(dim)?;
        }
        self.writer.write_u32::<LittleEndian>(ggml_type as u32)?;
        self.writer.write_u64::<LittleEndian>(offset)?;
        Ok(())
    }

    pub fn write_tensor_data(&mut self, data: &[u8]) -> Result<()> {
        self.writer.write_all(data)?;
        // GGUF requires 32-byte alignment for tensor data
        let padding = (32 - (data.len() % 32)) % 32;
        for _ in 0..padding {
            self.writer.write_u8(0)?;
        }
        Ok(())
    }
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

pub struct GGUFReader {
    pub reader: std::io::BufReader<std::fs::File>,
    pub tensor_count: u64,
    pub kv_count: u64,
}

impl GGUFReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);

        let magic = reader.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            anyhow::bail!("Invalid GGUF magic: {:08x}", magic);
        }
        let _version = reader.read_u32::<LittleEndian>()?;

        let tensor_count = reader.read_u64::<LittleEndian>()?;
        let kv_count = reader.read_u64::<LittleEndian>()?;

        Ok(Self {
            reader,
            tensor_count,
            kv_count,
        })
    }

    pub fn read_metadata(&mut self) -> Result<std::collections::HashMap<String, String>> {
        let mut metadata = std::collections::HashMap::new();
        for _ in 0..self.kv_count {
            let key = self.read_string()?;
            let val_type = self.reader.read_u32::<LittleEndian>()?;
            match val_type {
                8 => {
                    // String
                    let val = self.read_string()?;
                    metadata.insert(key, val);
                }
                _ => {
                    // Skip other types correctly
                    self.skip_kv_value(val_type)?;
                }
            }
        }
        Ok(metadata)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.reader.read_u64::<LittleEndian>()?;
        if len > 1024 * 1024 {
            // Safety check
            anyhow::bail!("String length too large: {}", len);
        }
        let mut buf = vec![0u8; len as usize];
        self.reader.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).to_string())
    }

    fn skip_kv_value(&mut self, val_type: u32) -> Result<()> {
        match val_type {
            0 | 1 | 7 => {
                let _ = self.reader.read_u8()?;
            } // u8, i8, bool
            2 | 3 => {
                let _ = self.reader.read_u16::<LittleEndian>()?;
            } // u16, i16
            4 | 5 | 6 => {
                let _ = self.reader.read_u32::<LittleEndian>()?;
            } // u32, i32, f32
            10 | 11 | 12 => {
                let _ = self.reader.read_u64::<LittleEndian>()?;
            } // u64, i64, f64
            8 => {
                // String
                let len = self.reader.read_u64::<LittleEndian>()?;
                std::io::copy(&mut self.reader.by_ref().take(len), &mut std::io::sink())?;
            }
            9 => {
                // Array
                let item_type = self.reader.read_u32::<LittleEndian>()?;
                let len = self.reader.read_u64::<LittleEndian>()?;
                for _ in 0..len {
                    self.skip_kv_value(item_type)?;
                }
            }
            _ => anyhow::bail!("Unknown KV type: {}", val_type),
        }
        Ok(())
    }
}
