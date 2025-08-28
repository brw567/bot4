//! Performance checking module for Bot4
//! Enforces latency requirements and identifies bottlenecks

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use regex::Regex;
use tracing::{info, warn, debug};

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub benchmarks: Vec<BenchmarkResult>,
    pub bottlenecks: Vec<Bottleneck>,
    pub violations: Vec<PerformanceViolation>,
    pub recommendations: Vec<String>,
    pub overall_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub measured_latency: f64,
    pub target_latency: f64,
    pub unit: String,
    pub passed: bool,
    pub percentiles: LatencyPercentiles,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Bottleneck {
    pub location: String,
    pub operation: String,
    pub latency_ms: f64,
    pub percentage_of_total: f64,
    pub call_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceViolation {
    pub component: String,
    pub requirement: String,
    pub measured: String,
    pub expected: String,
    pub severity: String,
}

pub struct PerformanceChecker {
    workspace_path: PathBuf,
    performance_targets: HashMap<String, f64>,
}

impl PerformanceChecker {
    pub fn new(workspace_path: PathBuf) -> Self {
        let mut performance_targets = HashMap::new();
        
        // Bot4 critical performance targets
        performance_targets.insert("decision_latency".to_string(), 50.0); // 50ns
        performance_targets.insert("order_submission".to_string(), 100.0); // 100μs
        performance_targets.insert("risk_calculation".to_string(), 1000.0); // 1ms
        performance_targets.insert("feature_extraction".to_string(), 500.0); // 500μs
        performance_targets.insert("ml_inference".to_string(), 1000.0); // 1ms
        performance_targets.insert("websocket_processing".to_string(), 100.0); // 100μs
        performance_targets.insert("database_write".to_string(), 5000.0); // 5ms
        performance_targets.insert("cache_read".to_string(), 100.0); // 100μs
        
        Self {
            workspace_path,
            performance_targets,
        }
    }
    
    pub async fn check_performance(&self, component: &str) -> Result<PerformanceReport> {
        info!("Checking performance for component: {}", component);
        
        let mut report = PerformanceReport {
            benchmarks: Vec::new(),
            bottlenecks: Vec::new(),
            violations: Vec::new(),
            recommendations: Vec::new(),
            overall_score: 0.0,
        };
        
        // Run benchmarks
        report.benchmarks = self.run_benchmarks(component).await?;
        
        // Analyze profile data
        report.bottlenecks = self.analyze_bottlenecks(component).await?;
        
        // Check against targets
        report.violations = self.check_violations(&report.benchmarks);
        
        // Generate recommendations
        report.recommendations = self.generate_recommendations(&report);
        
        // Calculate overall score
        report.overall_score = self.calculate_score(&report);
        
        Ok(report)
    }
    
    async fn run_benchmarks(&self, component: &str) -> Result<Vec<BenchmarkResult>> {
        let mut benchmarks = Vec::new();
        
        // Run cargo bench for the component
        let bench_path = self.workspace_path.join("rust_core");
        let output = Command::new("cargo")
            .args(&["bench", "--bench", &format!("{}_bench", component), "--", "--output-format", "json"])
            .current_dir(&bench_path)
            .output()?;
        
        if output.status.success() {
            // Parse benchmark results
            let stdout = String::from_utf8_lossy(&output.stdout);
            
            for line in stdout.lines() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                    if json["type"] == "bench" {
                        let name = json["name"].as_str().unwrap_or("unknown").to_string();
                        let median = json["median"].as_f64().unwrap_or(0.0);
                        let unit = json["unit"].as_str().unwrap_or("ns").to_string();
                        
                        // Get target for this benchmark
                        let target = self.get_target_for_benchmark(&name);
                        
                        benchmarks.push(BenchmarkResult {
                            name: name.clone(),
                            measured_latency: median,
                            target_latency: target,
                            unit: unit.clone(),
                            passed: median <= target,
                            percentiles: LatencyPercentiles {
                                p50: json["percentiles"]["50"].as_f64().unwrap_or(median),
                                p95: json["percentiles"]["95"].as_f64().unwrap_or(median * 1.5),
                                p99: json["percentiles"]["99"].as_f64().unwrap_or(median * 2.0),
                                p999: json["percentiles"]["999"].as_f64().unwrap_or(median * 3.0),
                            },
                        });
                    }
                }
            }
        }
        
        // Fallback: Parse criterion output if available
        if benchmarks.is_empty() {
            benchmarks = self.parse_criterion_output(component).await?;
        }
        
        Ok(benchmarks)
    }
    
    async fn parse_criterion_output(&self, component: &str) -> Result<Vec<BenchmarkResult>> {
        let mut benchmarks = Vec::new();
        let criterion_path = self.workspace_path.join("target/criterion");
        
        if criterion_path.exists() {
            // Walk through criterion results
            for entry in walkdir::WalkDir::new(&criterion_path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name() == "estimates.json")
            {
                let content = fs::read_to_string(entry.path()).await?;
                if let Ok(estimates) = serde_json::from_str::<serde_json::Value>(&content) {
                    let bench_name = entry.path()
                        .parent()
                        .and_then(|p| p.file_name())
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    
                    if bench_name.contains(component) {
                        let median = estimates["median"]["point_estimate"].as_f64().unwrap_or(0.0);
                        let target = self.get_target_for_benchmark(&bench_name);
                        
                        benchmarks.push(BenchmarkResult {
                            name: bench_name,
                            measured_latency: median / 1000.0, // Convert to μs
                            target_latency: target,
                            unit: "μs".to_string(),
                            passed: median / 1000.0 <= target,
                            percentiles: LatencyPercentiles {
                                p50: median / 1000.0,
                                p95: (median * 1.5) / 1000.0,
                                p99: (median * 2.0) / 1000.0,
                                p999: (median * 3.0) / 1000.0,
                            },
                        });
                    }
                }
            }
        }
        
        Ok(benchmarks)
    }
    
    async fn analyze_bottlenecks(&self, component: &str) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        
        // Check for flamegraph data
        let flamegraph_path = self.workspace_path.join("flamegraph.svg");
        if flamegraph_path.exists() {
            bottlenecks.extend(self.parse_flamegraph(&flamegraph_path).await?);
        }
        
        // Check for perf data
        let perf_output = Command::new("perf")
            .args(&["report", "--stdio", "--no-header", "--percent-limit", "1"])
            .output();
        
        if let Ok(output) = perf_output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                bottlenecks.extend(self.parse_perf_output(&stdout, component));
            }
        }
        
        // Static analysis for common bottlenecks
        bottlenecks.extend(self.static_bottleneck_analysis(component).await?);
        
        Ok(bottlenecks)
    }
    
    async fn parse_flamegraph(&self, path: &Path) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        let content = fs::read_to_string(path).await?;
        
        // Parse SVG for stack frames with high sample counts
        let regex = Regex::new(r#"<title>([^<]+)\((\d+) samples"#)?;
        let mut total_samples = 0;
        let mut frame_samples: Vec<(String, usize)> = Vec::new();
        
        for cap in regex.captures_iter(&content) {
            let frame = cap.get(1).unwrap().as_str().to_string();
            let samples = cap.get(2).unwrap().as_str().parse::<usize>().unwrap_or(0);
            total_samples += samples;
            frame_samples.push((frame, samples));
        }
        
        // Find frames that consume >5% of time
        for (frame, samples) in frame_samples {
            let percentage = (samples as f64 / total_samples as f64) * 100.0;
            if percentage > 5.0 {
                bottlenecks.push(Bottleneck {
                    location: frame.clone(),
                    operation: self.extract_operation(&frame),
                    latency_ms: 0.0, // Not available from flamegraph
                    percentage_of_total: percentage,
                    call_count: samples,
                });
            }
        }
        
        Ok(bottlenecks)
    }
    
    fn parse_perf_output(&self, output: &str, component: &str) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();
        
        for line in output.lines() {
            if line.contains(component) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    if let Ok(percentage) = parts[0].trim_end_matches('%').parse::<f64>() {
                        if percentage > 5.0 {
                            bottlenecks.push(Bottleneck {
                                location: parts[3..].join(" "),
                                operation: parts[2].to_string(),
                                latency_ms: 0.0,
                                percentage_of_total: percentage,
                                call_count: 0,
                            });
                        }
                    }
                }
            }
        }
        
        bottlenecks
    }
    
    async fn static_bottleneck_analysis(&self, component: &str) -> Result<Vec<Bottleneck>> {
        let mut bottlenecks = Vec::new();
        let component_path = self.workspace_path.join("rust_core/crates").join(component);
        
        if component_path.exists() {
            for entry in walkdir::WalkDir::new(&component_path)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
            {
                let content = fs::read_to_string(entry.path()).await?;
                
                // Check for common performance issues
                let issues = [
                    (r"\.clone\(\)", "Excessive cloning"),
                    (r"Vec::new\(\).*push", "Vector reallocation"),
                    (r"String::from|\.to_string\(\)", "String allocation"),
                    (r"Mutex|RwLock", "Lock contention"),
                    (r"thread::sleep|tokio::time::sleep", "Blocking sleep"),
                    (r"\.collect::<Vec", "Unnecessary collection"),
                    (r"for.*in.*\.iter\(\)", "Potential SIMD opportunity"),
                ];
                
                for (pattern, operation) in &issues {
                    let regex = Regex::new(pattern)?;
                    let count = regex.find_iter(&content).count();
                    
                    if count > 10 {
                        bottlenecks.push(Bottleneck {
                            location: entry.path().display().to_string(),
                            operation: operation.to_string(),
                            latency_ms: 0.0,
                            percentage_of_total: 0.0,
                            call_count: count,
                        });
                    }
                }
            }
        }
        
        Ok(bottlenecks)
    }
    
    fn check_violations(&self, benchmarks: &[BenchmarkResult]) -> Vec<PerformanceViolation> {
        let mut violations = Vec::new();
        
        for benchmark in benchmarks {
            if !benchmark.passed {
                let severity = if benchmark.measured_latency > benchmark.target_latency * 2.0 {
                    "critical"
                } else if benchmark.measured_latency > benchmark.target_latency * 1.5 {
                    "high"
                } else {
                    "medium"
                };
                
                violations.push(PerformanceViolation {
                    component: benchmark.name.clone(),
                    requirement: format!("<{}{}", benchmark.target_latency, benchmark.unit),
                    measured: format!("{}{}", benchmark.measured_latency, benchmark.unit),
                    expected: format!("{}{}", benchmark.target_latency, benchmark.unit),
                    severity: severity.to_string(),
                });
            }
            
            // Check P99 latency
            if benchmark.percentiles.p99 > benchmark.target_latency * 3.0 {
                violations.push(PerformanceViolation {
                    component: benchmark.name.clone(),
                    requirement: "P99 latency spike".to_string(),
                    measured: format!("{}{}@p99", benchmark.percentiles.p99, benchmark.unit),
                    expected: format!("<{}{}@p99", benchmark.target_latency * 3.0, benchmark.unit),
                    severity: "high".to_string(),
                });
            }
        }
        
        violations
    }
    
    fn generate_recommendations(&self, report: &PerformanceReport) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Analyze bottlenecks
        for bottleneck in &report.bottlenecks {
            if bottleneck.operation.contains("cloning") {
                recommendations.push(format!(
                    "Consider using Arc or Cow instead of cloning in {}",
                    bottleneck.location
                ));
            }
            if bottleneck.operation.contains("allocation") {
                recommendations.push(format!(
                    "Pre-allocate capacity or use object pools in {}",
                    bottleneck.location
                ));
            }
            if bottleneck.operation.contains("Lock") {
                recommendations.push(format!(
                    "Consider lock-free data structures or sharding in {}",
                    bottleneck.location
                ));
            }
            if bottleneck.operation.contains("SIMD") {
                recommendations.push(format!(
                    "Consider SIMD optimization using packed_simd or std::simd in {}",
                    bottleneck.location
                ));
            }
        }
        
        // Analyze violations
        for violation in &report.violations {
            if violation.severity == "critical" {
                recommendations.push(format!(
                    "CRITICAL: {} requires immediate optimization (currently {})",
                    violation.component, violation.measured
                ));
            }
        }
        
        // General recommendations based on overall score
        if report.overall_score < 50.0 {
            recommendations.push("Consider profiling with cargo-flamegraph to identify hotspots".to_string());
            recommendations.push("Enable CPU-specific optimizations in Cargo.toml".to_string());
            recommendations.push("Review memory allocation patterns with heaptrack".to_string());
        }
        
        if report.benchmarks.is_empty() {
            recommendations.push("Add criterion benchmarks for performance-critical paths".to_string());
        }
        
        recommendations
    }
    
    fn calculate_score(&self, report: &PerformanceReport) -> f64 {
        let mut score = 100.0;
        
        // Deduct for violations
        for violation in &report.violations {
            match violation.severity.as_str() {
                "critical" => score -= 20.0,
                "high" => score -= 10.0,
                "medium" => score -= 5.0,
                _ => score -= 2.0,
            }
        }
        
        // Deduct for bottlenecks
        for bottleneck in &report.bottlenecks {
            if bottleneck.percentage_of_total > 20.0 {
                score -= 10.0;
            } else if bottleneck.percentage_of_total > 10.0 {
                score -= 5.0;
            }
        }
        
        // Bonus for meeting all targets
        if report.violations.is_empty() {
            score += 10.0;
        }
        
        score.max(0.0).min(100.0)
    }
    
    fn get_target_for_benchmark(&self, name: &str) -> f64 {
        // Map benchmark names to performance targets
        for (key, target) in &self.performance_targets {
            if name.to_lowercase().contains(key) {
                return *target;
            }
        }
        
        // Default targets based on operation type
        if name.contains("order") {
            100.0 // 100μs for order operations
        } else if name.contains("risk") {
            1000.0 // 1ms for risk calculations
        } else if name.contains("ml") || name.contains("inference") {
            1000.0 // 1ms for ML inference
        } else if name.contains("cache") {
            100.0 // 100μs for cache operations
        } else {
            500.0 // 500μs default
        }
    }
    
    fn extract_operation(&self, frame: &str) -> String {
        // Extract the main operation from a stack frame
        if let Some(pos) = frame.rfind("::") {
            frame[pos + 2..].to_string()
        } else {
            frame.to_string()
        }
    }
}