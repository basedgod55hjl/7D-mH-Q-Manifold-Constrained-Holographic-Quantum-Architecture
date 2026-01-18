// src/rewrite/shadow.rs
use std::process::Command;

pub fn shadow_test(cmd: &str) -> bool {
    let prod = Command::new(cmd).output().ok();
    let shadow = Command::new(cmd).env("BLSS_SHADOW", "1").output().ok();

    prod.is_some() && shadow.is_some() && prod.unwrap().stdout == shadow.unwrap().stdout
}
