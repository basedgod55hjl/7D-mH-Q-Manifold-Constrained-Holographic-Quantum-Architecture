// src/rewrite/cross_lang.rs
pub enum Language {
    Rust,
    Python,
    Cpp,
}

pub fn normalize(code: &str, lang: Language) -> String {
    match lang {
        Language::Rust => code.replace("unwrap()", "expect(\"safe\")"),
        Language::Python => code.replace("print(", "logging.info("),
        Language::Cpp => code.replace("malloc", "std::unique_ptr"),
    }
}
