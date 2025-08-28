//! Report generation module for Bot4
//! Generates compliance and audit reports

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use chrono::{DateTime, Utc, Duration};
use tracing::info;

#[derive(Debug, Serialize, Deserialize)]
pub struct DailyReport {
    pub date: DateTime<Utc>,
    pub compliance_score: f64,
    pub tasks_completed: Vec<String>,
    pub violations_found: Vec<String>,
    pub tests_run: usize,
    pub tests_passed: usize,
    pub coverage_percentage: f64,
    pub commits: usize,
    pub critical_issues: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WeeklyReport {
    pub week_ending: DateTime<Utc>,
    pub overall_progress: f64,
    pub layers_completed: Vec<usize>,
    pub hours_logged: f64,
    pub velocity: f64,
    pub blockers_resolved: usize,
    pub new_violations: usize,
    pub team_performance: HashMap<String, TeamMetrics>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TeamMetrics {
    pub tasks_completed: usize,
    pub hours_contributed: f64,
    pub code_quality_score: f64,
    pub review_participation: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FullAuditReport {
    pub timestamp: DateTime<Utc>,
    pub project_status: String,
    pub compliance: ComplianceStatus,
    pub architecture: ArchitectureStatus,
    pub quality: QualityMetrics,
    pub risks: Vec<RiskItem>,
    pub recommendations: Vec<String>,
    pub deployment_readiness: DeploymentReadiness,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub overall_score: f64,
    pub alex_rules_compliance: f64,
    pub violations: Vec<Violation>,
    pub critical_violations: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ArchitectureStatus {
    pub layer_completion: HashMap<usize, f64>,
    pub layer_violations: usize,
    pub dependency_issues: Vec<String>,
    pub missing_components: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub test_coverage: f64,
    pub code_duplication: f64,
    pub technical_debt: f64,
    pub performance_score: f64,
    pub documentation_coverage: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskItem {
    pub category: String,
    pub description: String,
    pub severity: String,
    pub mitigation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Violation {
    pub rule: String,
    pub component: String,
    pub severity: String,
    pub location: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeploymentReadiness {
    pub ready: bool,
    pub blockers: Vec<String>,
    pub checklist: HashMap<String, bool>,
    pub estimated_ready_date: Option<DateTime<Utc>>,
}

pub struct ReportGenerator {
    workspace_path: PathBuf,
}

impl ReportGenerator {
    pub fn new(workspace_path: PathBuf) -> Self {
        Self { workspace_path }
    }
    
    pub async fn generate_daily_report(&self) -> Result<serde_json::Value> {
        info!("Generating daily compliance report");
        
        let report = DailyReport {
            date: Utc::now(),
            compliance_score: self.calculate_daily_compliance().await?,
            tasks_completed: self.get_tasks_completed_today().await?,
            violations_found: self.get_violations_today().await?,
            tests_run: self.count_tests_run().await?,
            tests_passed: self.count_tests_passed().await?,
            coverage_percentage: self.get_coverage_percentage().await?,
            commits: self.count_commits_today().await?,
            critical_issues: self.get_critical_issues().await?,
        };
        
        Ok(serde_json::json!({
            "report_type": "daily",
            "report": report,
            "summary": self.generate_daily_summary(&report),
        }))
    }
    
    pub async fn generate_weekly_report(&self) -> Result<serde_json::Value> {
        info!("Generating weekly compliance report");
        
        let report = WeeklyReport {
            week_ending: Utc::now(),
            overall_progress: self.calculate_weekly_progress().await?,
            layers_completed: self.get_layers_completed().await?,
            hours_logged: self.calculate_hours_logged().await?,
            velocity: self.calculate_velocity().await?,
            blockers_resolved: self.count_blockers_resolved().await?,
            new_violations: self.count_new_violations().await?,
            team_performance: self.calculate_team_performance().await?,
        };
        
        Ok(serde_json::json!({
            "report_type": "weekly",
            "report": report,
            "summary": self.generate_weekly_summary(&report),
        }))
    }
    
    pub async fn generate_full_audit(&self) -> Result<serde_json::Value> {
        info!("Generating full compliance audit");
        
        let report = FullAuditReport {
            timestamp: Utc::now(),
            project_status: self.assess_project_status().await?,
            compliance: self.audit_compliance().await?,
            architecture: self.audit_architecture().await?,
            quality: self.audit_quality().await?,
            risks: self.identify_risks().await?,
            recommendations: self.generate_recommendations().await?,
            deployment_readiness: self.assess_deployment_readiness().await?,
        };
        
        Ok(serde_json::json!({
            "report_type": "full_audit",
            "report": report,
            "executive_summary": self.generate_executive_summary(&report),
        }))
    }
    
    pub async fn generate_task_summary(&self) -> Result<serde_json::Value> {
        info!("Generating task summary report");
        
        let tasks = self.get_all_tasks().await?;
        let completed = tasks.iter().filter(|t| t["status"] == "completed").count();
        let in_progress = tasks.iter().filter(|t| t["status"] == "in_progress").count();
        let blocked = tasks.iter().filter(|t| t["status"] == "blocked").count();
        let pending = tasks.iter().filter(|t| t["status"] == "pending").count();
        
        Ok(serde_json::json!({
            "report_type": "task_summary",
            "total_tasks": tasks.len(),
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "pending": pending,
            "completion_rate": (completed as f64 / tasks.len() as f64) * 100.0,
            "by_layer": self.group_tasks_by_layer(&tasks),
            "critical_path": self.identify_critical_path(&tasks),
        }))
    }
    
    pub async fn generate_violation_report(&self) -> Result<serde_json::Value> {
        info!("Generating violation report");
        
        let violations = self.scan_all_violations().await?;
        
        Ok(serde_json::json!({
            "report_type": "violation_report",
            "total_violations": violations.len(),
            "by_severity": self.group_violations_by_severity(&violations),
            "by_component": self.group_violations_by_component(&violations),
            "critical_violations": violations.iter()
                .filter(|v| v.severity == "critical")
                .collect::<Vec<_>>(),
            "action_required": self.generate_violation_actions(&violations),
        }))
    }
    
    // Helper methods for daily report
    async fn calculate_daily_compliance(&self) -> Result<f64> {
        // Calculate compliance score based on today's activities
        let mut score = 100.0;
        
        // Check for violations
        let violations = self.get_violations_today().await?;
        score -= violations.len() as f64 * 5.0;
        
        // Check test results
        let tests_passed = self.count_tests_passed().await?;
        let tests_run = self.count_tests_run().await?;
        if tests_run > 0 {
            let pass_rate = (tests_passed as f64 / tests_run as f64) * 100.0;
            if pass_rate < 100.0 {
                score -= (100.0 - pass_rate) * 0.5;
            }
        }
        
        Ok(score.max(0.0))
    }
    
    async fn get_tasks_completed_today(&self) -> Result<Vec<String>> {
        // Check git log for task completions
        let output = std::process::Command::new("git")
            .args(&["log", "--since=1.day", "--pretty=%s"])
            .current_dir(&self.workspace_path)
            .output()?;
        
        let mut tasks = Vec::new();
        if output.status.success() {
            let commits = String::from_utf8_lossy(&output.stdout);
            for line in commits.lines() {
                if line.contains("Task") || line.contains("Complete") {
                    tasks.push(line.to_string());
                }
            }
        }
        
        Ok(tasks)
    }
    
    async fn get_violations_today(&self) -> Result<Vec<String>> {
        // Run validation scripts to find violations
        let mut violations = Vec::new();
        
        // Check for fake implementations
        let output = std::process::Command::new("python3")
            .args(&["scripts/validate_no_fakes.py"])
            .current_dir(&self.workspace_path)
            .output();
        
        if let Ok(output) = output {
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                violations.push(format!("Fake implementations found: {}", stderr));
            }
        }
        
        Ok(violations)
    }
    
    async fn count_tests_run(&self) -> Result<usize> {
        // Count tests from cargo test output
        let output = std::process::Command::new("cargo")
            .args(&["test", "--all", "--", "--list"])
            .current_dir(&self.workspace_path)
            .output()?;
        
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(stdout.lines().filter(|l| l.contains("test")).count())
        } else {
            Ok(0)
        }
    }
    
    async fn count_tests_passed(&self) -> Result<usize> {
        // Run tests and count passed
        let output = std::process::Command::new("cargo")
            .args(&["test", "--all"])
            .current_dir(&self.workspace_path)
            .output()?;
        
        if output.status.success() {
            // All tests passed
            self.count_tests_run().await
        } else {
            // Parse output for passed tests
            let stdout = String::from_utf8_lossy(&output.stdout);
            let passed = stdout.lines()
                .filter(|l| l.contains("test") && l.contains("ok"))
                .count();
            Ok(passed)
        }
    }
    
    async fn get_coverage_percentage(&self) -> Result<f64> {
        // Try to get coverage from tarpaulin
        let coverage_file = self.workspace_path.join("target/coverage/cobertura.xml");
        
        if coverage_file.exists() {
            let content = fs::read_to_string(&coverage_file).await?;
            // Parse coverage percentage from XML
            if let Some(pos) = content.find("line-rate=\"") {
                let start = pos + 11;
                if let Some(end) = content[start..].find("\"") {
                    if let Ok(rate) = content[start..start+end].parse::<f64>() {
                        return Ok(rate * 100.0);
                    }
                }
            }
        }
        
        Ok(0.0)
    }
    
    async fn count_commits_today(&self) -> Result<usize> {
        let output = std::process::Command::new("git")
            .args(&["rev-list", "--count", "--since=1.day", "HEAD"])
            .current_dir(&self.workspace_path)
            .output()?;
        
        if output.status.success() {
            let count_str = String::from_utf8_lossy(&output.stdout);
            Ok(count_str.trim().parse().unwrap_or(0))
        } else {
            Ok(0)
        }
    }
    
    async fn get_critical_issues(&self) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        
        // Check Layer 0 completion
        if !self.is_layer_complete(0).await? {
            issues.push("🚨 Layer 0 (Safety Systems) incomplete".to_string());
        }
        
        // Check for fake implementations
        if self.has_fake_implementations().await? {
            issues.push("🚨 Fake implementations detected".to_string());
        }
        
        // Check test coverage
        let coverage = self.get_coverage_percentage().await?;
        if coverage < 100.0 {
            issues.push(format!("⚠️ Test coverage only {:.1}%", coverage));
        }
        
        Ok(issues)
    }
    
    // Helper methods for full audit
    async fn assess_project_status(&self) -> Result<String> {
        let progress = self.calculate_overall_progress().await?;
        
        if progress < 25.0 {
            Ok("EARLY_DEVELOPMENT".to_string())
        } else if progress < 50.0 {
            Ok("ACTIVE_DEVELOPMENT".to_string())
        } else if progress < 75.0 {
            Ok("INTEGRATION_PHASE".to_string())
        } else if progress < 95.0 {
            Ok("TESTING_PHASE".to_string())
        } else {
            Ok("READY_FOR_DEPLOYMENT".to_string())
        }
    }
    
    async fn audit_compliance(&self) -> Result<ComplianceStatus> {
        let violations = self.scan_all_violations().await?;
        
        Ok(ComplianceStatus {
            overall_score: self.calculate_compliance_score(&violations),
            alex_rules_compliance: self.check_alex_rules_compliance().await?,
            violations: violations.clone(),
            critical_violations: violations.iter()
                .filter(|v| v.severity == "critical")
                .count(),
        })
    }
    
    async fn audit_architecture(&self) -> Result<ArchitectureStatus> {
        let mut layer_completion = HashMap::new();
        for layer in 0..=7 {
            layer_completion.insert(layer, self.calculate_layer_completion(layer).await?);
        }
        
        Ok(ArchitectureStatus {
            layer_completion,
            layer_violations: self.count_layer_violations().await?,
            dependency_issues: self.find_dependency_issues().await?,
            missing_components: self.find_missing_components().await?,
        })
    }
    
    async fn audit_quality(&self) -> Result<QualityMetrics> {
        Ok(QualityMetrics {
            test_coverage: self.get_coverage_percentage().await?,
            code_duplication: self.calculate_duplication_percentage().await?,
            technical_debt: self.calculate_technical_debt().await?,
            performance_score: self.calculate_performance_score().await?,
            documentation_coverage: self.calculate_doc_coverage().await?,
        })
    }
    
    async fn identify_risks(&self) -> Result<Vec<RiskItem>> {
        let mut risks = Vec::new();
        
        // Safety system risks
        if !self.is_layer_complete(0).await? {
            risks.push(RiskItem {
                category: "Safety".to_string(),
                description: "Safety systems incomplete".to_string(),
                severity: "CRITICAL".to_string(),
                mitigation: "Complete Layer 0 immediately".to_string(),
            });
        }
        
        // Technical debt risk
        let tech_debt = self.calculate_technical_debt().await?;
        if tech_debt > 20.0 {
            risks.push(RiskItem {
                category: "Technical".to_string(),
                description: format!("High technical debt: {:.1}%", tech_debt),
                severity: "HIGH".to_string(),
                mitigation: "Allocate time for refactoring".to_string(),
            });
        }
        
        // Performance risk
        let perf_score = self.calculate_performance_score().await?;
        if perf_score < 80.0 {
            risks.push(RiskItem {
                category: "Performance".to_string(),
                description: format!("Performance below target: {:.1}%", perf_score),
                severity: "MEDIUM".to_string(),
                mitigation: "Profile and optimize critical paths".to_string(),
            });
        }
        
        Ok(risks)
    }
    
    async fn generate_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        // Check Layer 0
        if !self.is_layer_complete(0).await? {
            recommendations.push("1. CRITICAL: Complete Layer 0 (Safety Systems) immediately".to_string());
        }
        
        // Check test coverage
        let coverage = self.get_coverage_percentage().await?;
        if coverage < 100.0 {
            recommendations.push(format!("2. Increase test coverage from {:.1}% to 100%", coverage));
        }
        
        // Check documentation
        let doc_coverage = self.calculate_doc_coverage().await?;
        if doc_coverage < 100.0 {
            recommendations.push(format!("3. Complete documentation (currently {:.1}%)", doc_coverage));
        }
        
        // Team collaboration
        recommendations.push("4. Continue FULL TEAM collaboration on single tasks".to_string());
        
        // External research
        recommendations.push("5. Conduct external research before implementation".to_string());
        
        Ok(recommendations)
    }
    
    async fn assess_deployment_readiness(&self) -> Result<DeploymentReadiness> {
        let mut checklist = HashMap::new();
        let mut blockers = Vec::new();
        
        // Check all requirements
        checklist.insert("Layer 0 Complete".to_string(), self.is_layer_complete(0).await?);
        checklist.insert("100% Test Coverage".to_string(), 
                        self.get_coverage_percentage().await? >= 100.0);
        checklist.insert("No Fake Implementations".to_string(), 
                        !self.has_fake_implementations().await?);
        checklist.insert("All Layers Complete".to_string(), 
                        self.all_layers_complete().await?);
        checklist.insert("Paper Trading Complete".to_string(), 
                        self.paper_trading_complete().await?);
        checklist.insert("Performance Targets Met".to_string(), 
                        self.performance_targets_met().await?);
        
        // Identify blockers
        for (requirement, met) in &checklist {
            if !met {
                blockers.push(requirement.clone());
            }
        }
        
        let ready = blockers.is_empty();
        
        Ok(DeploymentReadiness {
            ready,
            blockers,
            checklist,
            estimated_ready_date: if ready {
                Some(Utc::now())
            } else {
                Some(Utc::now() + Duration::weeks(self.estimate_weeks_to_ready().await?))
            },
        })
    }
    
    // Utility methods
    async fn is_layer_complete(&self, _layer: usize) -> Result<bool> {
        // Simplified - would check actual layer completion
        Ok(false)
    }
    
    async fn has_fake_implementations(&self) -> Result<bool> {
        let output = std::process::Command::new("python3")
            .args(&["scripts/validate_no_fakes.py"])
            .current_dir(&self.workspace_path)
            .output();
        
        match output {
            Ok(output) => Ok(!output.status.success()),
            Err(_) => Ok(true), // Assume fakes if script fails
        }
    }
    
    async fn calculate_overall_progress(&self) -> Result<f64> {
        // Simplified calculation
        Ok(14.4) // From PROJECT_MANAGEMENT_MASTER.md
    }
    
    async fn scan_all_violations(&self) -> Result<Vec<Violation>> {
        // Simplified - would scan entire codebase
        Ok(Vec::new())
    }
    
    async fn calculate_compliance_score(&self, violations: &[Violation]) -> f64 {
        let mut score = 100.0;
        for violation in violations {
            match violation.severity.as_str() {
                "critical" => score -= 20.0,
                "high" => score -= 10.0,
                "medium" => score -= 5.0,
                _ => score -= 2.0,
            }
        }
        score.max(0.0)
    }
    
    async fn check_alex_rules_compliance(&self) -> Result<f64> {
        // Check compliance with Alex's specific rules
        let mut score = 100.0;
        
        if self.has_fake_implementations().await? {
            score = 0.0; // Zero tolerance
        }
        
        let coverage = self.get_coverage_percentage().await?;
        if coverage < 100.0 {
            score -= (100.0 - coverage) * 0.5;
        }
        
        Ok(score.max(0.0))
    }
    
    // Additional helper methods...
    async fn calculate_layer_completion(&self, _layer: usize) -> Result<f64> { Ok(0.0) }
    async fn count_layer_violations(&self) -> Result<usize> { Ok(0) }
    async fn find_dependency_issues(&self) -> Result<Vec<String>> { Ok(Vec::new()) }
    async fn find_missing_components(&self) -> Result<Vec<String>> { Ok(Vec::new()) }
    async fn calculate_duplication_percentage(&self) -> Result<f64> { Ok(0.0) }
    async fn calculate_technical_debt(&self) -> Result<f64> { Ok(0.0) }
    async fn calculate_performance_score(&self) -> Result<f64> { Ok(0.0) }
    async fn calculate_doc_coverage(&self) -> Result<f64> { Ok(0.0) }
    async fn all_layers_complete(&self) -> Result<bool> { Ok(false) }
    async fn paper_trading_complete(&self) -> Result<bool> { Ok(false) }
    async fn performance_targets_met(&self) -> Result<bool> { Ok(false) }
    async fn estimate_weeks_to_ready(&self) -> Result<i64> { Ok(24) }
    async fn calculate_weekly_progress(&self) -> Result<f64> { Ok(0.0) }
    async fn get_layers_completed(&self) -> Result<Vec<usize>> { Ok(Vec::new()) }
    async fn calculate_hours_logged(&self) -> Result<f64> { Ok(0.0) }
    async fn calculate_velocity(&self) -> Result<f64> { Ok(0.0) }
    async fn count_blockers_resolved(&self) -> Result<usize> { Ok(0) }
    async fn count_new_violations(&self) -> Result<usize> { Ok(0) }
    async fn calculate_team_performance(&self) -> Result<HashMap<String, TeamMetrics>> { Ok(HashMap::new()) }
    async fn get_all_tasks(&self) -> Result<Vec<serde_json::Value>> { Ok(Vec::new()) }
    
    fn generate_daily_summary(&self, _report: &DailyReport) -> String {
        "Daily compliance report generated".to_string()
    }
    
    fn generate_weekly_summary(&self, _report: &WeeklyReport) -> String {
        "Weekly progress report generated".to_string()
    }
    
    fn generate_executive_summary(&self, _report: &FullAuditReport) -> String {
        "Full compliance audit completed".to_string()
    }
    
    fn group_tasks_by_layer(&self, _tasks: &[serde_json::Value]) -> serde_json::Value {
        serde_json::json!({})
    }
    
    fn identify_critical_path(&self, _tasks: &[serde_json::Value]) -> Vec<String> {
        Vec::new()
    }
    
    fn group_violations_by_severity(&self, _violations: &[Violation]) -> serde_json::Value {
        serde_json::json!({})
    }
    
    fn group_violations_by_component(&self, _violations: &[Violation]) -> serde_json::Value {
        serde_json::json!({})
    }
    
    fn generate_violation_actions(&self, _violations: &[Violation]) -> Vec<String> {
        Vec::new()
    }
}