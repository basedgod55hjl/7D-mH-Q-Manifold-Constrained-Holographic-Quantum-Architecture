// src/rewrite/ast.rs
use syn::{visit::Visit, File};

pub fn parse_rust(code: &str) -> File {
    syn::parse_file(code).expect("AST parse failed")
}

pub fn has_unwrap(ast: &File) -> bool {
    struct Finder {
        found: bool,
    }
    impl<'ast> Visit<'ast> for Finder {
        fn visit_expr_method_call(&mut self, node: &'ast syn::ExprMethodCall) {
            if node.method == "unwrap" {
                self.found = true;
            }
        }
    }
    let mut f = Finder { found: false };
    f.visit_file(ast);
    f.found
}
