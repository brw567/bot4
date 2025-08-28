//! Documentation checking module for Bot4
//! Ensures all code is properly documented per Alex's requirements

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use syn::{parse_file, Item, ItemFn, ItemStruct, ItemTrait, ItemImpl, ItemEnum, ItemMod};
use tokio::fs;
use regex::Regex;
use tracing::{info, warn, debug};

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentationReport {
    pub coverage: DocCoverage,
    pub violations: Vec<DocViolation>,
    pub missing: Vec<MissingDoc>,
    pub quality_issues: Vec<QualityIssue>,
    pub readme_status: ReadmeStatus,
    pub overall_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocCoverage {
    pub total_items: usize,
    pub documented_items: usize,
    pub coverage_percent: f64,
    pub by_type: HashMap<String, TypeCoverage>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TypeCoverage {
    pub total: usize,
    pub documented: usize,
    pub percent: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocViolation {
    pub file: String,
    pub line: usize,
    pub item: String,
    pub violation_type: String,
    pub message: String,
    pub severity: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MissingDoc {
    pub file: String,
    pub line: usize,
    pub item_type: String,
    pub item_name: String,
    pub is_public: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityIssue {
    pub file: String,
    pub line: usize,
    pub issue_type: String,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReadmeStatus {
    pub exists: bool,
    pub has_description: bool,
    pub has_usage_examples: bool,
    pub has_api_docs: bool,
    pub has_architecture: bool,
    pub last_updated_days_ago: Option<u64>,
}

pub struct DocumentationChecker {
    workspace_path: PathBuf,
}

impl DocumentationChecker {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self { workspace_path }
    }
    
    pub async fn check_documentation(&self, path: &str) -> Result<DocumentationReport> {
        info!("Checking documentation for: {}", path);
        
        let target_path = self.workspace_path.join(path);
        
        let mut report = DocumentationReport {
            coverage: DocCoverage {
                total_items: 0,
                documented_items: 0,
                coverage_percent: 0.0,
                by_type: HashMap::new(),
            },
            violations: Vec::new(),
            missing: Vec::new(),
            quality_issues: Vec::new(),
            readme_status: self.check_readme(&target_path).await?,
            overall_score: 0.0,
        };
        
        // Analyze Rust files
        if target_path.is_dir() {
            self.analyze_directory(&target_path, &mut report).await?;
        } else if target_path.is_file() {
            self.analyze_file(&target_path, &mut report).await?;
        }
        
        // Calculate coverage percentages
        if report.coverage.total_items > 0 {
            report.coverage.coverage_percent = 
                (report.coverage.documented_items as f64 / report.coverage.total_items as f64) * 100.0;
        }
        
        for (_, type_cov) in report.coverage.by_type.iter_mut() {
            if type_cov.total > 0 {
                type_cov.percent = (type_cov.documented as f64 / type_cov.total as f64) * 100.0;
            }
        }
        
        // Calculate overall score
        report.overall_score = self.calculate_score(&report);
        
        Ok(report)
    }
    
    async fn analyze_directory(&self, dir: &Path, report: &mut DocumentationReport) -> Result<()> {
        for entry in walkdir::WalkDir::new(dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
        {
            self.analyze_file(entry.path(), report).await?;
        }
        
        Ok(())
    }
    
    async fn analyze_file(&self, file: &Path, report: &mut DocumentationReport) -> Result<()> {
        let content = fs::read_to_string(file).await?;
        
        // Parse the Rust file
        if let Ok(syntax_tree) = parse_file(&content) {
            // Check module-level documentation
            if !content.starts_with("//!") && !content.starts_with("/*!") {
                report.quality_issues.push(QualityIssue {
                    file: file.display().to_string(),
                    line: 1,
                    issue_type: "missing_module_doc".to_string(),
                    description: "Module lacks top-level documentation (//! or /*!).".to_string(),
                });
            }
            
            // Analyze each item
            for item in syntax_tree.items {
                self.analyze_item(&item, file, &content, report)?;
            }
        }
        
        // Check for Bot4-specific documentation requirements
        self.check_bot4_requirements(file, &content, report)?;
        
        // Check documentation quality
        self.check_doc_quality(file, &content, report)?;
        
        Ok(())
    }
    
    fn analyze_item(
        &self, 
        item: &Item, 
        file: &Path, 
        content: &str,
        report: &mut DocumentationReport
    ) -> Result<()> {
        match item {
            Item::Fn(item_fn) => {
                let is_public = matches!(item_fn.vis, syn::Visibility::Public(_));
                let item_type = "function";
                let item_name = item_fn.sig.ident.to_string();
                
                self.update_coverage(report, item_type);
                
                // Check for documentation
                let has_doc = self.has_documentation(&item_fn.attrs);
                if has_doc {
                    self.update_documented(report, item_type);
                    
                    // Check doc quality for public functions
                    if is_public {
                        self.check_function_doc_quality(item_fn, file, report)?;
                    }
                } else if is_public && !item_name.starts_with("_") {
                    report.missing.push(MissingDoc {
                        file: file.display().to_string(),
                        line: 0, // Would need span info for accurate line
                        item_type: item_type.to_string(),
                        item_name,
                        is_public,
                    });
                    
                    report.violations.push(DocViolation {
                        file: file.display().to_string(),
                        line: 0,
                        item: item_name.clone(),
                        violation_type: "missing_doc".to_string(),
                        message: format!("Public function '{}' lacks documentation", item_name),
                        severity: "high".to_string(),
                    });
                }
                
                // Check for performance documentation if function name suggests it's critical
                if item_name.contains("execute") || item_name.contains("process") || 
                   item_name.contains("calculate") || item_name.contains("submit") {
                    self.check_performance_doc(&item_fn.attrs, &item_name, file, report)?;
                }
            }
            
            Item::Struct(item_struct) => {
                let is_public = matches!(item_struct.vis, syn::Visibility::Public(_));
                let item_type = "struct";
                let item_name = item_struct.ident.to_string();
                
                self.update_coverage(report, item_type);
                
                if self.has_documentation(&item_struct.attrs) {
                    self.update_documented(report, item_type);
                } else if is_public {
                    report.missing.push(MissingDoc {
                        file: file.display().to_string(),
                        line: 0,
                        item_type: item_type.to_string(),
                        item_name: item_name.clone(),
                        is_public,
                    });
                }
                
                // Check field documentation for public structs
                if is_public {
                    for field in &item_struct.fields {
                        if let Some(ident) = &field.ident {
                            if !self.has_documentation(&field.attrs) {
                                report.quality_issues.push(QualityIssue {
                                    file: file.display().to_string(),
                                    line: 0,
                                    issue_type: "undocumented_field".to_string(),
                                    description: format!("Field '{}' in struct '{}' lacks documentation", 
                                                       ident, item_name),
                                });
                            }
                        }
                    }
                }
            }
            
            Item::Trait(item_trait) => {
                let is_public = matches!(item_trait.vis, syn::Visibility::Public(_));
                let item_type = "trait";
                let item_name = item_trait.ident.to_string();
                
                self.update_coverage(report, item_type);
                
                if self.has_documentation(&item_trait.attrs) {
                    self.update_documented(report, item_type);
                } else if is_public {
                    report.missing.push(MissingDoc {
                        file: file.display().to_string(),
                        line: 0,
                        item_type: item_type.to_string(),
                        item_name,
                        is_public,
                    });
                }
            }
            
            Item::Enum(item_enum) => {
                let is_public = matches!(item_enum.vis, syn::Visibility::Public(_));
                let item_type = "enum";
                let item_name = item_enum.ident.to_string();
                
                self.update_coverage(report, item_type);
                
                if self.has_documentation(&item_enum.attrs) {
                    self.update_documented(report, item_type);
                } else if is_public {
                    report.missing.push(MissingDoc {
                        file: file.display().to_string(),
                        line: 0,
                        item_type: item_type.to_string(),
                        item_name: item_name.clone(),
                        is_public,
                    });
                }
                
                // Check variant documentation
                for variant in &item_enum.variants {
                    if !self.has_documentation(&variant.attrs) && is_public {
                        report.quality_issues.push(QualityIssue {
                            file: file.display().to_string(),
                            line: 0,
                            issue_type: "undocumented_variant".to_string(),
                            description: format!("Variant '{}' in enum '{}' lacks documentation", 
                                               variant.ident, item_name),
                        });
                    }
                }
            }
            
            Item::Impl(item_impl) => {
                // Check impl block documentation
                if self.has_documentation(&item_impl.attrs) {
                    self.update_coverage(report, "impl");
                    self.update_documented(report, "impl");
                }
                
                // Check methods within impl
                for impl_item in &item_impl.items {
                    if let syn::ImplItem::Method(method) = impl_item {
                        let is_public = matches!(method.vis, syn::Visibility::Public(_));
                        if is_public {
                            self.update_coverage(report, "method");
                            if self.has_documentation(&method.attrs) {
                                self.update_documented(report, "method");
                            } else {
                                report.missing.push(MissingDoc {
                                    file: file.display().to_string(),
                                    line: 0,
                                    item_type: "method".to_string(),
                                    item_name: method.sig.ident.to_string(),
                                    is_public,
                                });
                            }
                        }
                    }
                }
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    fn check_bot4_requirements(&self, file: &Path, content: &str, report: &mut DocumentationReport) -> Result<()> {
        // Check for required documentation patterns per Bot4 standards
        
        // Check for performance documentation
        if content.contains("pub fn execute") || content.contains("pub fn submit_order") {
            if !content.contains("Performance:") && !content.contains("Latency:") {
                report.violations.push(DocViolation {
                    file: file.display().to_string(),
                    line: 0,
                    item: file.file_name().unwrap_or_default().to_str().unwrap_or("").to_string(),
                    violation_type: "missing_performance_doc".to_string(),
                    message: "Performance-critical functions must document latency requirements".to_string(),
                    severity: "high".to_string(),
                });
            }
        }
        
        // Check for phase documentation
        if !content.contains("Phase:") && !content.contains("Layer:") {
            report.quality_issues.push(QualityIssue {
                file: file.display().to_string(),
                line: 0,
                issue_type: "missing_phase_doc".to_string(),
                description: "File should document which phase/layer it belongs to".to_string(),
            });
        }
        
        // Check for safety documentation in risk components
        if file.to_str().unwrap_or("").contains("risk") || content.contains("circuit_breaker") {
            if !content.contains("Safety:") && !content.contains("Risk:") {
                report.violations.push(DocViolation {
                    file: file.display().to_string(),
                    line: 0,
                    item: file.file_name().unwrap_or_default().to_str().unwrap_or("").to_string(),
                    violation_type: "missing_safety_doc".to_string(),
                    message: "Risk management components must document safety constraints".to_string(),
                    severity: "critical".to_string(),
                });
            }
        }
        
        // Check for example usage in public APIs
        let example_regex = Regex::new(r"```rust")?;
        let pub_fn_regex = Regex::new(r"pub fn \w+")?;
        
        let pub_fn_count = pub_fn_regex.find_iter(content).count();
        let example_count = example_regex.find_iter(content).count();
        
        if pub_fn_count > 5 && example_count == 0 {
            report.quality_issues.push(QualityIssue {
                file: file.display().to_string(),
                line: 0,
                issue_type: "missing_examples".to_string(),
                description: format!("File has {} public functions but no usage examples", pub_fn_count),
            });
        }
        
        Ok(())
    }
    
    fn check_doc_quality(&self, file: &Path, content: &str, report: &mut DocumentationReport) -> Result<()> {
        // Check for low-quality documentation patterns
        let lines: Vec<&str> = content.lines().collect();
        
        for (i, line) in lines.iter().enumerate() {
            if line.starts_with("///") || line.starts_with("//!") {
                let doc_content = line.trim_start_matches("///").trim_start_matches("//!").trim();
                
                // Check for placeholder documentation
                if doc_content == "TODO" || doc_content == "FIXME" || 
                   doc_content == "Documentation" || doc_content == "..." {
                    report.quality_issues.push(QualityIssue {
                        file: file.display().to_string(),
                        line: i + 1,
                        issue_type: "placeholder_doc".to_string(),
                        description: format!("Placeholder documentation: '{}'", doc_content),
                    });
                }
                
                // Check for too-short documentation
                if doc_content.len() > 0 && doc_content.len() < 10 && 
                   !doc_content.starts_with('[') && !doc_content.starts_with('`') {
                    report.quality_issues.push(QualityIssue {
                        file: file.display().to_string(),
                        line: i + 1,
                        issue_type: "brief_doc".to_string(),
                        description: format!("Documentation too brief: '{}'", doc_content),
                    });
                }
                
                // Check for missing punctuation
                if doc_content.len() > 20 && 
                   !doc_content.ends_with('.') && 
                   !doc_content.ends_with('!') && 
                   !doc_content.ends_with('?') &&
                   !doc_content.ends_with(':') &&
                   !doc_content.ends_with('`') &&
                   !doc_content.contains("```") {
                    report.quality_issues.push(QualityIssue {
                        file: file.display().to_string(),
                        line: i + 1,
                        issue_type: "missing_punctuation".to_string(),
                        description: "Documentation should end with punctuation".to_string(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    fn check_function_doc_quality(&self, func: &ItemFn, file: &Path, report: &mut DocumentationReport) -> Result<()> {
        let doc_comments = self.extract_doc_comments(&func.attrs);
        let doc_text = doc_comments.join(" ");
        
        // Check for parameter documentation
        for input in &func.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let param_name = pat_ident.ident.to_string();
                    if !doc_text.contains(&param_name) && param_name != "self" {
                        report.quality_issues.push(QualityIssue {
                            file: file.display().to_string(),
                            line: 0,
                            issue_type: "undocumented_parameter".to_string(),
                            description: format!("Parameter '{}' in function '{}' is not documented", 
                                               param_name, func.sig.ident),
                        });
                    }
                }
            }
        }
        
        // Check for return value documentation
        if !matches!(func.sig.output, syn::ReturnType::Default) {
            if !doc_text.contains("Returns") && !doc_text.contains("return") && !doc_text.contains("->") {
                report.quality_issues.push(QualityIssue {
                    file: file.display().to_string(),
                    line: 0,
                    issue_type: "undocumented_return".to_string(),
                    description: format!("Function '{}' does not document its return value", func.sig.ident),
                });
            }
        }
        
        // Check for error documentation if Result is returned
        if let syn::ReturnType::Type(_, ret_type) = &func.sig.output {
            if let syn::Type::Path(type_path) = &**ret_type {
                if type_path.path.segments.last().map(|s| s.ident.to_string()) == Some("Result".to_string()) {
                    if !doc_text.contains("Error") && !doc_text.contains("Err") && !doc_text.contains("fail") {
                        report.quality_issues.push(QualityIssue {
                            file: file.display().to_string(),
                            line: 0,
                            issue_type: "undocumented_errors".to_string(),
                            description: format!("Function '{}' returns Result but doesn't document errors", func.sig.ident),
                        });
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn check_performance_doc(&self, attrs: &[syn::Attribute], name: &str, file: &Path, report: &mut DocumentationReport) -> Result<()> {
        let doc_comments = self.extract_doc_comments(attrs);
        let doc_text = doc_comments.join(" ");
        
        if !doc_text.contains("Performance:") && 
           !doc_text.contains("Latency:") && 
           !doc_text.contains("Complexity:") &&
           !doc_text.contains("O(") {
            report.quality_issues.push(QualityIssue {
                file: file.display().to_string(),
                line: 0,
                issue_type: "missing_performance_info".to_string(),
                description: format!("Performance-critical function '{}' lacks performance documentation", name),
            });
        }
        
        Ok(())
    }
    
    async fn check_readme(&self, path: &Path) -> Result<ReadmeStatus> {
        let readme_path = if path.is_dir() {
            path.join("README.md")
        } else {
            path.parent().unwrap_or(Path::new(".")).join("README.md")
        };
        
        let mut status = ReadmeStatus {
            exists: false,
            has_description: false,
            has_usage_examples: false,
            has_api_docs: false,
            has_architecture: false,
            last_updated_days_ago: None,
        };
        
        if readme_path.exists() {
            status.exists = true;
            
            let content = fs::read_to_string(&readme_path).await?;
            
            // Check for required sections
            status.has_description = content.contains("## Description") || 
                                    content.contains("# ") || 
                                    content.len() > 100;
            status.has_usage_examples = content.contains("## Usage") || 
                                       content.contains("## Example") || 
                                       content.contains("```rust");
            status.has_api_docs = content.contains("## API") || 
                                content.contains("## Documentation");
            status.has_architecture = content.contains("## Architecture") || 
                                     content.contains("## Design");
            
            // Check last modified time
            if let Ok(metadata) = fs::metadata(&readme_path).await {
                if let Ok(modified) = metadata.modified() {
                    if let Ok(elapsed) = modified.elapsed() {
                        status.last_updated_days_ago = Some(elapsed.as_secs() / 86400);
                    }
                }
            }
        }
        
        Ok(status)
    }
    
    fn has_documentation(&self, attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|attr| {
            attr.path.segments.len() == 1 && 
            attr.path.segments[0].ident == "doc"
        })
    }
    
    fn extract_doc_comments(&self, attrs: &[syn::Attribute]) -> Vec<String> {
        let mut comments = Vec::new();
        
        for attr in attrs {
            if attr.path.segments.len() == 1 && attr.path.segments[0].ident == "doc" {
                if let Ok(syn::Meta::NameValue(meta)) = attr.parse_meta() {
                    if let syn::Lit::Str(lit_str) = meta.lit {
                        comments.push(lit_str.value());
                    }
                }
            }
        }
        
        comments
    }
    
    fn update_coverage(&self, report: &mut DocumentationReport, item_type: &str) {
        report.coverage.total_items += 1;
        
        let type_cov = report.coverage.by_type
            .entry(item_type.to_string())
            .or_insert(TypeCoverage {
                total: 0,
                documented: 0,
                percent: 0.0,
            });
        type_cov.total += 1;
    }
    
    fn update_documented(&self, report: &mut DocumentationReport, item_type: &str) {
        report.coverage.documented_items += 1;
        
        if let Some(type_cov) = report.coverage.by_type.get_mut(item_type) {
            type_cov.documented += 1;
        }
    }
    
    fn calculate_score(&self, report: &DocumentationReport) -> f64 {
        let mut score = 100.0;
        
        // Base score on coverage (40% weight)
        let coverage_score = report.coverage.coverage_percent * 0.4;
        
        // Deduct for violations (30% weight)
        let violation_penalty = report.violations.iter().map(|v| {
            match v.severity.as_str() {
                "critical" => 10.0,
                "high" => 5.0,
                "medium" => 2.0,
                _ => 1.0,
            }
        }).sum::<f64>();
        
        // Deduct for quality issues (20% weight)
        let quality_penalty = (report.quality_issues.len() as f64) * 0.5;
        
        // Bonus for good README (10% weight)
        let readme_bonus = if report.readme_status.exists {
            let mut bonus = 5.0;
            if report.readme_status.has_usage_examples { bonus += 2.0; }
            if report.readme_status.has_api_docs { bonus += 2.0; }
            if report.readme_status.has_architecture { bonus += 1.0; }
            bonus
        } else {
            0.0
        };
        
        score = coverage_score + readme_bonus - violation_penalty - quality_penalty;
        
        // Alex's requirement: 100% coverage or fail
        if report.coverage.coverage_percent < 100.0 {
            score = score.min(50.0); // Cap at 50% if not 100% documented
        }
        
        score.max(0.0).min(100.0)
    }
}