pub mod cuda;
pub mod x86_64;

use crate::ir::{IRBlock, IRBlock7D};

pub trait Backend {
    fn emit(&self, blocks: &[IRBlock]) -> Result<String, String>;
}

pub trait Backend7D {
    fn emit(&self, blocks: &[IRBlock7D]) -> Result<String, String>;
}
