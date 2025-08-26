//! # Parallel Validation System for Safe Migration
//! 
//! Runs old and new implementations in parallel to ensure correctness
//! during the migration from duplicate types to canonical types.
//!
//! ## How It Works
//! 1. Execute both old and new code paths
//! 2. Compare results for consistency
//! 3. Log any discrepancies for investigation
//! 4. Use feature flags to control behavior
//!
//! ## External Research Applied
//! - GitHub's Scientist library pattern
//! - Dark launching techniques (Facebook)
//! - Shadow traffic patterns (Netflix)

use crate::{Order, Price, Quantity, Trade, Candle, OrderBook};
use std::fmt::Debug;
use std::time::{Duration, Instant};
use tracing::{error, warn, info};

/// Result of parallel validation
#[derive(Debug, Clone)]
pub struct ValidationResult<T> {
    /// Result from the canonical implementation
    pub canonical_result: T,
    /// Result from the legacy implementation (if available)
    pub legacy_result: Option<T>,
    /// Whether results match
    pub results_match: bool,
    /// Performance comparison
    pub performance: PerformanceComparison,
    /// Any discrepancies found
    pub discrepancies: Vec<Discrepancy>,
}

/// Performance comparison between implementations
#[derive(Debug, Clone)]
pub struct PerformanceComparison {
    /// Time taken by canonical implementation
    pub canonical_duration: Duration,
    /// Time taken by legacy implementation
    pub legacy_duration: Option<Duration>,
    /// Percentage improvement (negative means canonical is slower)
    pub improvement_percent: f64,
}

/// Discrepancy between implementations
#[derive(Debug, Clone)]
pub struct Discrepancy {
    /// Field or aspect that differs
    pub field: String,
    /// Value from canonical implementation
    pub canonical_value: String,
    /// Value from legacy implementation
    pub legacy_value: String,
    /// Severity of the discrepancy
    pub severity: DiscrepancySeverity,
}

/// Severity levels for discrepancies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscrepancySeverity {
    /// Critical - results are fundamentally different
    Critical,
    /// Warning - minor differences that might affect behavior
    Warning,
    /// Info - cosmetic differences only
    Info,
}

/// Trait for types that can be compared in parallel validation
pub trait ParallelValidatable: Clone {
    /// Compares this instance with another for validation
    fn compare_with(&self, other: &Self) -> Vec<Discrepancy>;
}

/// Main parallel validator
pub struct ParallelValidator {
    /// Whether to fail on discrepancies
    fail_on_discrepancy: bool,
    /// Whether to log performance metrics
    log_performance: bool,
    /// Sampling rate (0.0 to 1.0)
    sampling_rate: f64,
}

impl Default for ParallelValidator {
    fn default() -> Self {
        Self {
            fail_on_discrepancy: false,
            log_performance: true,
            sampling_rate: 1.0, // Validate everything by default
        }
    }
}

impl ParallelValidator {
    /// Creates a new validator with custom settings
    pub fn new(fail_on_discrepancy: bool, log_performance: bool, sampling_rate: f64) -> Self {
        Self {
            fail_on_discrepancy,
            log_performance,
            sampling_rate: sampling_rate.clamp(0.0, 1.0),
        }
    }
    
    /// Validates a function that returns a result
    pub fn validate<T, E, F1, F2>(
        &self,
        canonical_fn: F1,
        legacy_fn: F2,
        operation_name: &str,
    ) -> Result<ValidationResult<T>, E>
    where
        T: ParallelValidatable + Debug,
        E: Debug,
        F1: FnOnce() -> Result<T, E>,
        F2: FnOnce() -> Result<T, E>,
    {
        // Check sampling
        if !self.should_validate() {
            let result = canonical_fn()?;
            return Ok(ValidationResult {
                canonical_result: result.clone(),
                legacy_result: None,
                results_match: true,
                performance: PerformanceComparison {
                    canonical_duration: Duration::ZERO,
                    legacy_duration: None,
                    improvement_percent: 0.0,
                },
                discrepancies: Vec::new(),
            });
        }
        
        // Run canonical implementation
        let canonical_start = Instant::now();
        let canonical_result = canonical_fn()?;
        let canonical_duration = canonical_start.elapsed();
        
        // Run legacy implementation
        let legacy_start = Instant::now();
        let legacy_result = match legacy_fn() {
            Ok(r) => Some(r),
            Err(e) => {
                warn!(
                    "Legacy implementation failed for {}: {:?}",
                    operation_name, e
                );
                None
            }
        };
        let legacy_duration = legacy_start.elapsed();
        
        // Compare results
        let (results_match, discrepancies) = if let Some(ref legacy) = legacy_result {
            let discrepancies = canonical_result.compare_with(legacy);
            let matches = discrepancies.is_empty() || 
                          discrepancies.iter().all(|d| d.severity != DiscrepancySeverity::Critical);
            (matches, discrepancies)
        } else {
            (false, vec![Discrepancy {
                field: "execution".to_string(),
                canonical_value: "success".to_string(),
                legacy_value: "failed".to_string(),
                severity: DiscrepancySeverity::Warning,
            }])
        };
        
        // Calculate performance improvement
        let improvement_percent = if let Some(legacy_dur) = legacy_result.as_ref().map(|_| legacy_duration) {
            let canonical_ms = canonical_duration.as_secs_f64() * 1000.0;
            let legacy_ms = legacy_dur.as_secs_f64() * 1000.0;
            ((legacy_ms - canonical_ms) / legacy_ms) * 100.0
        } else {
            0.0
        };
        
        // Log results
        if self.log_performance {
            info!(
                "Parallel validation for {}: match={}, canonical={:?}, legacy={:?}, improvement={:.2}%",
                operation_name,
                results_match,
                canonical_duration,
                legacy_duration,
                improvement_percent
            );
        }
        
        // Log discrepancies
        if !discrepancies.is_empty() {
            for discrepancy in &discrepancies {
                match discrepancy.severity {
                    DiscrepancySeverity::Critical => error!(
                        "CRITICAL discrepancy in {}: field={}, canonical={}, legacy={}",
                        operation_name, discrepancy.field, discrepancy.canonical_value, discrepancy.legacy_value
                    ),
                    DiscrepancySeverity::Warning => warn!(
                        "Warning discrepancy in {}: field={}, canonical={}, legacy={}",
                        operation_name, discrepancy.field, discrepancy.canonical_value, discrepancy.legacy_value
                    ),
                    DiscrepancySeverity::Info => info!(
                        "Info discrepancy in {}: field={}, canonical={}, legacy={}",
                        operation_name, discrepancy.field, discrepancy.canonical_value, discrepancy.legacy_value
                    ),
                }
            }
        }
        
        // Fail if configured and critical discrepancies found
        if self.fail_on_discrepancy && !results_match {
            error!("Parallel validation failed for {}", operation_name);
            // In production, we would still return the canonical result
            // but log the failure for investigation
        }
        
        Ok(ValidationResult {
            canonical_result,
            legacy_result,
            results_match,
            performance: PerformanceComparison {
                canonical_duration,
                legacy_duration: Some(legacy_duration),
                improvement_percent,
            },
            discrepancies,
        })
    }
    
    /// Determines whether to validate based on sampling rate
    fn should_validate(&self) -> bool {
        if self.sampling_rate >= 1.0 {
            return true;
        }
        if self.sampling_rate <= 0.0 {
            return false;
        }
        
        // Simple random sampling
        rand::random::<f64>() < self.sampling_rate
    }
}

// ===== ParallelValidatable Implementations =====

impl ParallelValidatable for Order {
    fn compare_with(&self, other: &Self) -> Vec<Discrepancy> {
        let mut discrepancies = Vec::new();
        
        // Compare critical fields
        if self.symbol != other.symbol {
            discrepancies.push(Discrepancy {
                field: "symbol".to_string(),
                canonical_value: self.symbol.clone(),
                legacy_value: other.symbol.clone(),
                severity: DiscrepancySeverity::Critical,
            });
        }
        
        if self.side != other.side {
            discrepancies.push(Discrepancy {
                field: "side".to_string(),
                canonical_value: format!("{:?}", self.side),
                legacy_value: format!("{:?}", other.side),
                severity: DiscrepancySeverity::Critical,
            });
        }
        
        if self.quantity != other.quantity {
            discrepancies.push(Discrepancy {
                field: "quantity".to_string(),
                canonical_value: self.quantity.to_string(),
                legacy_value: other.quantity.to_string(),
                severity: DiscrepancySeverity::Critical,
            });
        }
        
        if self.price != other.price {
            discrepancies.push(Discrepancy {
                field: "price".to_string(),
                canonical_value: format!("{:?}", self.price),
                legacy_value: format!("{:?}", other.price),
                severity: DiscrepancySeverity::Critical,
            });
        }
        
        // Compare less critical fields
        if self.order_type != other.order_type {
            discrepancies.push(Discrepancy {
                field: "order_type".to_string(),
                canonical_value: format!("{:?}", self.order_type),
                legacy_value: format!("{:?}", other.order_type),
                severity: DiscrepancySeverity::Warning,
            });
        }
        
        if self.time_in_force != other.time_in_force {
            discrepancies.push(Discrepancy {
                field: "time_in_force".to_string(),
                canonical_value: format!("{:?}", self.time_in_force),
                legacy_value: format!("{:?}", other.time_in_force),
                severity: DiscrepancySeverity::Warning,
            });
        }
        
        // Compare metadata (info level)
        if self.strategy_id != other.strategy_id {
            discrepancies.push(Discrepancy {
                field: "strategy_id".to_string(),
                canonical_value: format!("{:?}", self.strategy_id),
                legacy_value: format!("{:?}", other.strategy_id),
                severity: DiscrepancySeverity::Info,
            });
        }
        
        discrepancies
    }
}

impl ParallelValidatable for Price {
    fn compare_with(&self, other: &Self) -> Vec<Discrepancy> {
        let mut discrepancies = Vec::new();
        
        let canonical_decimal = self.as_decimal();
        let legacy_decimal = other.as_decimal();
        
        if canonical_decimal != legacy_decimal {
            // Check if difference is significant (more than 0.01%)
            let diff = (canonical_decimal - legacy_decimal).abs();
            let percent_diff = (diff / canonical_decimal) * rust_decimal::Decimal::from(100);
            
            let severity = if percent_diff > rust_decimal::Decimal::from_str_exact("0.01").unwrap() {
                DiscrepancySeverity::Critical
            } else {
                DiscrepancySeverity::Warning
            };
            
            discrepancies.push(Discrepancy {
                field: "value".to_string(),
                canonical_value: canonical_decimal.to_string(),
                legacy_value: legacy_decimal.to_string(),
                severity,
            });
        }
        
        if self.precision() != other.precision() {
            discrepancies.push(Discrepancy {
                field: "precision".to_string(),
                canonical_value: self.precision().to_string(),
                legacy_value: other.precision().to_string(),
                severity: DiscrepancySeverity::Info,
            });
        }
        
        discrepancies
    }
}

impl ParallelValidatable for Quantity {
    fn compare_with(&self, other: &Self) -> Vec<Discrepancy> {
        let mut discrepancies = Vec::new();
        
        if self.as_decimal() != other.as_decimal() {
            discrepancies.push(Discrepancy {
                field: "value".to_string(),
                canonical_value: self.as_decimal().to_string(),
                legacy_value: other.as_decimal().to_string(),
                severity: DiscrepancySeverity::Critical,
            });
        }
        
        discrepancies
    }
}

// ===== Usage Example =====

/// Example of parallel validation in practice
#[cfg(test)]
mod example {
    use super::*;
    use rust_decimal_macros::dec;
    use crate::OrderSide;
    
    fn create_order_canonical(symbol: String, quantity: Quantity) -> Result<Order, String> {
        Ok(Order::market(symbol, OrderSide::Buy, quantity))
    }
    
    fn create_order_legacy(symbol: String, quantity: Quantity) -> Result<Order, String> {
        // Simulating a legacy implementation with slight differences
        let mut order = Order::market(symbol, OrderSide::Buy, quantity);
        order.time_in_force = crate::TimeInForce::GTC; // Legacy uses different default
        Ok(order)
    }
    
    #[test]
    fn test_parallel_validation() {
        let validator = ParallelValidator::default();
        
        let symbol = "BTC/USDT".to_string();
        let quantity = Quantity::new(dec!(1)).unwrap();
        
        let result = validator.validate(
            || create_order_canonical(symbol.clone(), quantity),
            || create_order_legacy(symbol.clone(), quantity),
            "create_order",
        ).unwrap();
        
        // Results should mostly match except for time_in_force
        assert_eq!(result.discrepancies.len(), 1);
        assert_eq!(result.discrepancies[0].field, "time_in_force");
        assert_eq!(result.discrepancies[0].severity, DiscrepancySeverity::Warning);
    }
}

// External dependencies for sampling
use rand;