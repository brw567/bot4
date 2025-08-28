//! Fake Implementation Detector for QualityGate Agent
//! Detects placeholders, TODOs, and incomplete implementations

use anyhow::Result;
use syn::{File, Item, Expr, Stmt, Block};
use syn::visit::{self, Visit};

pub struct FakeDetector {
    violations: Vec<String>,
}

impl FakeDetector {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
        }
    }

    /// Analyze AST for fake implementations
    pub fn analyze_ast(&self, syntax_tree: &File) -> Result<Vec<String>> {
        let mut visitor = FakeVisitor {
            violations: Vec::new(),
        };
        
        visitor.visit_file(syntax_tree);
        
        Ok(visitor.violations)
    }

    /// Check for suspicious patterns in function bodies
    pub fn check_function_body(&self, body: &Block) -> Vec<String> {
        let mut violations = Vec::new();
        
        // Check for empty blocks
        if body.stmts.is_empty() {
            violations.push("Empty function body detected".to_string());
        }
        
        // Check for single return statements that might be placeholders
        if body.stmts.len() == 1 {
            if let Some(Stmt::Expr(Expr::Return(_), _)) = body.stmts.first() {
                violations.push("Function contains only return statement - possible placeholder".to_string());
            }
        }
        
        violations
    }
}

struct FakeVisitor {
    violations: Vec<String>,
}

impl<'ast> Visit<'ast> for FakeVisitor {
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        // Check function body for suspicious patterns
        let body = &node.block;
        
        // Check for empty functions
        if body.stmts.is_empty() {
            self.violations.push(format!(
                "Function '{}' has empty body",
                node.sig.ident
            ));
        }
        
        // Check for functions that only panic
        if body.stmts.len() == 1 {
            if let Some(Stmt::Expr(expr, _)) = body.stmts.first() {
                if let Expr::Macro(macro_call) = expr {
                    let macro_path = macro_call.mac.path.segments.last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default();
                    
                    if macro_path == "todo" || macro_path == "unimplemented" || macro_path == "panic" {
                        self.violations.push(format!(
                            "Function '{}' contains {}! macro",
                            node.sig.ident, macro_path
                        ));
                    }
                }
            }
        }
        
        // Continue visiting
        visit::visit_item_fn(self, node);
    }
    
    fn visit_expr(&mut self, expr: &'ast Expr) {
        match expr {
            Expr::Macro(macro_call) => {
                let macro_path = macro_call.mac.path.segments.last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                
                // Detect problematic macros
                match macro_path.as_str() {
                    "todo" => self.violations.push("todo! macro found".to_string()),
                    "unimplemented" => self.violations.push("unimplemented! macro found".to_string()),
                    "unreachable" => self.violations.push("unreachable! macro found - verify logic".to_string()),
                    _ => {}
                }
            }
            // Check for hardcoded test values
            Expr::Lit(lit) => {
                if let syn::Lit::Str(s) = &lit.lit {
                    let value = s.value();
                    if value == "test" || value == "dummy" || value == "fake" || value == "placeholder" {
                        self.violations.push(format!("Suspicious string literal: '{}'", value));
                    }
                }
            }
            _ => {}
        }
        
        // Continue visiting
        visit::visit_expr(self, expr);
    }
}