//! Rules enforcement module for Bot4
//! Enforces Alex's ZERO TOLERANCE policies

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tokio::fs;
use regex::Regex;
use syn::{parse_file, Item, ItemFn, Expr};
use tracing::{info, warn, error, debug};

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorHandling {
    pub proper: bool,
    pub message: String,
}

pub struct RulesEnforcer {
    workspace_path: PathBuf,
}

impl RulesEnforcer {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self { workspace_path }
    }
    
    /// Check for fake implementations - ZERO TOLERANCE
    pub async fn check_fake_implementations(&self, component: &str) -> Result<Vec<String>> {
        let mut fakes = Vec::new();
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        
        if !component_path.exists() {
            // Try alternate path
            let alt_path = self.workspace_path.join("crates").join(component);
            if alt_path.exists() {
                return self.scan_for_fakes(&alt_path).await;
            }
            return Ok(fakes);
        }
        
        self.scan_for_fakes(&component_path).await
    }
    
    async fn scan_for_fakes(&self, path: &Path) -> Result<Vec<String>> {
        let mut fakes = Vec::new();
        
        for entry in walkdir::WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            let relative_path = entry.path().strip_prefix(&self.workspace_path)
                .unwrap_or(entry.path())
                .display()
                .to_string();
            
            // Check for forbidden patterns
            let forbidden_patterns = [
                (r"todo!\(\)", "todo!() macro"),
                (r"unimplemented!\(\)", "unimplemented!() macro"),
                (r"panic!\(\"not implemented", "panic with 'not implemented'"),
                (r"// TODO", "TODO comment"),
                (r"// FIXME", "FIXME comment"),
                (r"// HACK", "HACK comment"),
                (r"// XXX", "XXX comment"),
                (r"return\s+0\.0;?\s*//\s*fake", "fake return value"),
                (r"return\s+vec!\[\];?\s*//\s*placeholder", "placeholder return"),
                (r"mock_", "mock implementation"),
                (r"dummy_", "dummy implementation"),
                (r"fake_", "fake implementation"),
                (r"placeholder_", "placeholder implementation"),
                (r"\.unwrap\(\)\s*//\s*temporary", "temporary unwrap"),
                (r"hardcoded", "hardcoded value comment"),
                (r"magic number", "magic number comment"),
            ];
            
            for (pattern, description) in &forbidden_patterns {
                let regex = Regex::new(pattern)?;
                for (line_num, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        fakes.push(format!("{}:{} - {}", relative_path, line_num + 1, description));
                    }
                }
            }
            
            // Parse Rust code for more sophisticated checks
            if let Ok(syntax_tree) = parse_file(&content) {
                for item in syntax_tree.items {
                    if let Item::Fn(func) = item {
                        // Check for functions that just return constants
                        if self.is_fake_function(&func) {
                            fakes.push(format!("{}:{} - fake function implementation", 
                                             relative_path, func.sig.ident));
                        }
                    }
                }
            }
        }
        
        Ok(fakes)
    }
    
    fn is_fake_function(&self, func: &ItemFn) -> bool {
        // Check if function body is suspiciously simple
        if func.block.stmts.len() == 1 {
            if let Some(stmt) = func.block.stmts.first() {
                // Check for direct return of literal
                if let syn::Stmt::Expr(Expr::Return(ret_expr)) = stmt {
                    if let Some(expr) = &ret_expr.expr {
                        match &**expr {
                            Expr::Lit(_) => return true, // Returns literal
                            Expr::Tuple(tuple) if tuple.elems.is_empty() => return true, // Returns ()
                            _ => {}
                        }
                    }
                }
            }
        }
        
        false
    }
    
    /// Check test coverage - MUST BE 100%
    pub async fn check_test_coverage(&self, component: &str) -> Result<f64> {
        // Try to get coverage from tarpaulin or grcov output
        let coverage_file = self.workspace_path
            .join("target/coverage")
            .join(format!("{}.json", component));
        
        if coverage_file.exists() {
            let content = fs::read_to_string(&coverage_file).await?;
            if let Ok(coverage_data) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(percentage) = coverage_data["coverage_percentage"].as_f64() {
                    return Ok(percentage);
                }
            }
        }
        
        // Fallback: Count test functions vs regular functions
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        if component_path.exists() {
            let (functions, tests) = self.count_functions_and_tests(&component_path).await?;
            if functions > 0 {
                // Rough estimate: assume each function needs at least one test
                let coverage = (tests as f64 / functions as f64) * 100.0;
                return Ok(coverage.min(100.0));
            }
        }
        
        Ok(0.0)
    }
    
    async fn count_functions_and_tests(&self, path: &Path) -> Result<(usize, usize)> {
        let mut functions = 0;
        let mut tests = 0;
        
        for entry in walkdir::WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            
            if let Ok(syntax_tree) = parse_file(&content) {
                for item in syntax_tree.items {
                    if let Item::Fn(func) = item {
                        let is_test = func.attrs.iter().any(|attr| {
                            attr.path.segments.iter().any(|s| s.ident == "test")
                        });
                        
                        if is_test {
                            tests += 1;
                        } else if matches!(func.vis, syn::Visibility::Public(_)) {
                            functions += 1;
                        }
                    }
                }
            }
        }
        
        Ok((functions, tests))
    }
    
    /// Check for TODOs and placeholders
    pub async fn check_todos(&self, component: &str) -> Result<Vec<String>> {
        let mut todos = Vec::new();
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        
        if !component_path.exists() {
            return Ok(todos);
        }
        
        for entry in walkdir::WalkDir::new(&component_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            let relative_path = entry.path().strip_prefix(&self.workspace_path)
                .unwrap_or(entry.path())
                .display()
                .to_string();
            
            for (line_num, line) in content.lines().enumerate() {
                if line.contains("TODO") || line.contains("FIXME") || 
                   line.contains("XXX") || line.contains("HACK") {
                    todos.push(format!("{}:{} - {}", relative_path, line_num + 1, line.trim()));
                }
            }
        }
        
        Ok(todos)
    }
    
    /// Check for hardcoded values
    pub async fn check_hardcoded_values(&self, component: &str) -> Result<Vec<String>> {
        let mut hardcoded = Vec::new();
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        
        if !component_path.exists() {
            return Ok(hardcoded);
        }
        
        for entry in walkdir::WalkDir::new(&component_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            let relative_path = entry.path().strip_prefix(&self.workspace_path)
                .unwrap_or(entry.path())
                .display()
                .to_string();
            
            // Check for suspicious patterns
            let patterns = [
                (r#""localhost""#, "hardcoded localhost"),
                (r#""127\.0\.0\.1""#, "hardcoded IP"),
                (r#""[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+""#, "hardcoded IP address"),
                (r":\d{4,5}[^0-9]", "hardcoded port"),
                (r#""[a-zA-Z0-9]{20,}""#, "possible hardcoded API key"),
                (r"password\s*=\s*\"[^\"]+\"", "hardcoded password"),
                (r"secret\s*=\s*\"[^\"]+\"", "hardcoded secret"),
                (r"api_key\s*=\s*\"[^\"]+\"", "hardcoded API key"),
                (r"0\.02[^0-9]", "hardcoded 2% limit"),
                (r"0\.15[^0-9]", "hardcoded 15% drawdown"),
            ];
            
            for (pattern, description) in &patterns {
                let regex = Regex::new(pattern)?;
                for (line_num, line) in content.lines().enumerate() {
                    // Skip comments and test code
                    if line.trim_start().starts_with("//") || 
                       line.contains("#[test]") ||
                       line.contains("#[cfg(test)]") {
                        continue;
                    }
                    
                    if regex.is_match(line) {
                        hardcoded.push(format!("{}:{} - {}", relative_path, line_num + 1, description));
                    }
                }
            }
        }
        
        Ok(hardcoded)
    }
    
    /// Check for proper error handling
    pub async fn check_error_handling(&self, component: &str) -> Result<ErrorHandling> {
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        
        if !component_path.exists() {
            return Ok(ErrorHandling {
                proper: false,
                message: "Component not found".to_string(),
            });
        }
        
        let mut unwrap_count = 0;
        let mut expect_count = 0;
        let mut result_count = 0;
        let mut proper_handling = 0;
        
        for entry in walkdir::WalkDir::new(&component_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            
            // Count different error handling patterns
            unwrap_count += content.matches(".unwrap()").count();
            expect_count += content.matches(".expect(").count();
            result_count += content.matches("Result<").count();
            proper_handling += content.matches("?").count();
            proper_handling += content.matches("match ").count();
            proper_handling += content.matches("if let Ok(").count();
            proper_handling += content.matches("if let Err(").count();
        }
        
        let total_error_points = unwrap_count + expect_count + result_count;
        
        if total_error_points == 0 {
            return Ok(ErrorHandling {
                proper: true,
                message: "No error-prone operations found".to_string(),
            });
        }
        
        let proper_ratio = proper_handling as f64 / total_error_points as f64;
        
        Ok(ErrorHandling {
            proper: proper_ratio > 0.8 && unwrap_count < 5,
            message: format!(
                "Found {} unwraps, {} expects, {} proper handlers (ratio: {:.1}%)",
                unwrap_count, expect_count, proper_handling, proper_ratio * 100.0
            ),
        })
    }
    
    /// Check for circuit breakers - MANDATORY
    pub async fn check_circuit_breakers(&self, component: &str) -> Result<bool> {
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        
        if !component_path.exists() {
            return Ok(false);
        }
        
        // Components that require circuit breakers
        let requires_circuit_breaker = [
            "trading_engine", "risk_engine", "exchange_connector",
            "ml_pipeline", "order_manager", "execution", "strategies"
        ];
        
        if !requires_circuit_breaker.iter().any(|&req| component.contains(req)) {
            // This component doesn't require a circuit breaker
            return Ok(true);
        }
        
        for entry in walkdir::WalkDir::new(&component_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            
            // Check for circuit breaker patterns
            if content.contains("CircuitBreaker") ||
               content.contains("circuit_breaker") ||
               content.contains("kill_switch") ||
               content.contains("KillSwitch") {
                return Ok(true);
            }
        }
        
        error!("CRITICAL: {} requires circuit breaker but none found!", component);
        Ok(false)
    }
    
    /// Check for performance documentation
    pub async fn check_performance_docs(&self, component: &str) -> Result<bool> {
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        
        if !component_path.exists() {
            return Ok(false);
        }
        
        for entry in walkdir::WalkDir::new(&component_path)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            let content = fs::read_to_string(entry.path()).await?;
            
            // Check for performance documentation patterns
            if content.contains("Performance:") ||
               content.contains("Latency:") ||
               content.contains("Throughput:") ||
               content.contains("Complexity:") ||
               content.contains("O(") {
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Scan entire codebase for fake implementations
    pub async fn scan_all_fake_implementations(&self) -> Result<Vec<String>> {
        let mut all_fakes = Vec::new();
        
        let paths = [
            self.workspace_path.join("rust_core/crates"),
            self.workspace_path.join("rust_core/src"),
            self.workspace_path.join("crates"),
        ];
        
        for path in &paths {
            if path.exists() {
                let fakes = self.scan_for_fakes(path).await?;
                all_fakes.extend(fakes);
            }
        }
        
        Ok(all_fakes)
    }
    
    /// Get critical violations that block deployment
    pub async fn get_critical_violations(&self) -> Result<Vec<String>> {
        let mut violations = Vec::new();
        
        // Scan for critical issues
        let fakes = self.scan_all_fake_implementations().await?;
        for fake in fakes {
            violations.push(format!("FAKE_IMPLEMENTATION: {}", fake));
        }
        
        // Check for missing circuit breakers in critical components
        let critical_components = ["trading_engine", "risk_engine", "execution"];
        for component in &critical_components {
            if !self.check_circuit_breakers(component).await? {
                violations.push(format!("MISSING_CIRCUIT_BREAKER: {}", component));
            }
        }
        
        Ok(violations)
    }
}