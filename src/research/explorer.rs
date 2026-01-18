// src/research/explorer.rs
use std::{thread, time::Duration};

pub fn research_loop() {
    thread::spawn(|| loop {
        println!("[RESEARCH] Scanning new techniques");
        thread::sleep(Duration::from_secs(120));
    });
}
