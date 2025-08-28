//! Coverage Analyzer Module for QualityGate Agent
//! Analyzes test coverage and enforces 100% coverage requirement

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageData {
    pub line_coverage_percent: f64,
    pub branch_coverage_percent: f64,
    pub function_coverage_percent: f64,
    pub files: Vec<FileCoverage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileCoverage {
    pub path: String,
    pub coverage: f64,
    pub covered_lines: u32,
    pub total_lines: u32,
    pub uncovered_lines: Vec<u32>,
}

pub struct CoverageAnalyzer {
    threshold: f64,
}

impl CoverageAnalyzer {
    pub fn new() -> Self {
        Self {
            threshold: 100.0, // 100% coverage requirement
        }
    }

    /// Parse tarpaulin JSON output
    pub fn parse_tarpaulin_output(&self, output: &[u8]) -> Result<CoverageData> {
        let json: serde_json::Value = serde_json::from_slice(output)?;
        
        let mut files = Vec::new();
        let mut total_covered = 0u32;
        let mut total_lines = 0u32;
        
        if let Some(file_data) = json["files"].as_object() {
            for (path, coverage_info) in file_data {
                if let Some(covered) = coverage_info["covered"].as_u64() {
                    let total = coverage_info["coverable"].as_u64().unwrap_or(0);
                    let coverage_percent = if total > 0 {
                        (covered as f64 / total as f64) * 100.0
                    } else {
                        100.0
                    };
                    
                    let uncovered_lines = coverage_info["uncovered_lines"]
                        .as_array()
                        .map(|arr| arr.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect())
                        .unwrap_or_default();
                    
                    files.push(FileCoverage {
                        path: path.clone(),
                        coverage: coverage_percent,
                        covered_lines: covered as u32,
                        total_lines: total as u32,
                        uncovered_lines,
                    });
                    
                    total_covered += covered as u32;
                    total_lines += total as u32;
                }
            }
        }
        
        let line_coverage_percent = if total_lines > 0 {
            (total_covered as f64 / total_lines as f64) * 100.0
        } else {
            100.0
        };
        
        Ok(CoverageData {
            line_coverage_percent,
            branch_coverage_percent: line_coverage_percent, // Simplified for now
            function_coverage_percent: line_coverage_percent,
            files,
        })
    }

    /// Check if coverage meets requirements
    pub fn meets_requirements(&self, coverage: &CoverageData) -> bool {
        coverage.line_coverage_percent >= self.threshold
    }

    /// Get detailed coverage report
    pub fn generate_report(&self, coverage: &CoverageData) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("Overall Coverage: {:.2}%\n", coverage.line_coverage_percent));
        report.push_str(&format!("Required: {:.2}%\n\n", self.threshold));
        
        if coverage.line_coverage_percent < self.threshold {
            report.push_str("❌ COVERAGE INSUFFICIENT\n\n");
            report.push_str("Files needing coverage:\n");
            
            for file in &coverage.files {
                if file.coverage < self.threshold {
                    report.push_str(&format!(
                        "  {} - {:.2}% ({}/{} lines)\n",
                        file.path, file.coverage, file.covered_lines, file.total_lines
                    ));
                    
                    if !file.uncovered_lines.is_empty() && file.uncovered_lines.len() <= 10 {
                        report.push_str(&format!(
                            "    Uncovered lines: {:?}\n",
                            file.uncovered_lines
                        ));
                    }
                }
            }
        } else {
            report.push_str("✅ COVERAGE MEETS REQUIREMENTS\n");
        }
        
        report
    }
}