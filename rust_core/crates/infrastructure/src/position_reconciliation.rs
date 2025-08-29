use domain_types::CircuitBreaker;
use domain_types::order::OrderError;
//! Position Reconciliation Module - Layer 0.8.1
//! Module uses canonical Position type from domain_types
//! Cameron: "Single source of truth for Position struct"

use domain_types::order::{Order, OrderId, OrderStatus, OrderType};

pub use domain_types::position_canonical::{
    Position, PositionId, PositionSide, PositionStatus,
    PositionError, PositionUpdate
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type PositionResult<T> = Result<T, PositionError>;

// Avery: "Single source of truth for Order struct"

// Order types are already imported above
// Re-export for backward compatibility
pub type OrderResult<T> = Result<T, OrderError>;

// POSITION RECONCILIATION MODULE - Layer 0.8.1
// Full Team Implementation with External Research
// Team: All 8 members collaborating
// Purpose: Verify actual exchange positions match internal state after recovery
// External Research Applied:
// - "The Essential Guide to Crypto Custodian Reconciliation" - Bitwave (2024)
// - Blockchain reconciliation patterns from Chainalysis
// - "Distributed Systems: Principles and Paradigms" - Tanenbaum (2017)
// - Byzantine Fault Tolerance algorithms
// - Exchange API best practices (Binance, Kraken, Coinbase)
// - "Crash Recovery in Distributed Systems" - Lamport (1978)

use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime};
use anyhow::{Result, Context, bail};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};
use tokio::sync::{RwLock, Mutex, broadcast};
use rust_decimal::{Decimal, prelude::FromStr};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};

use crate::software_control_modes::ControlMode;
use crate::mode_persistence::ModePersistenceManager;

// ============================================================================
// CIRCUIT BREAKER (simplified for reconciliation)
// ============================================================================

/// Simple circuit breaker for reconciliation

#[derive(Debug, Clone)]
// ELIMINATED: use domain_types::CircuitBreaker
// pub struct CircuitBreaker {
    name: String,
    failure_count: Arc<Mutex<u32>>,
    max_failures: u32,
    tripped: Arc<Mutex<bool>>,
    reset_timeout: Duration,
    last_failure: Arc<Mutex<Option<SystemTime>>>,
}

impl CircuitBreaker {
    pub fn new(name: String, max_failures: u32, _window: Duration, reset_timeout: Duration) -> Self {
        Self {
            name,
            failure_count: Arc::new(Mutex::new(0)),
            max_failures,
            tripped: Arc::new(Mutex::new(false)),
            reset_timeout,
            last_failure: Arc::new(Mutex::new(None)),
        }
    }
    
    pub fn is_tripped(&self) -> bool {
        *self.tripped.blocking_lock()
    }
    
    pub fn record_failure(&self) {
        let mut count = self.failure_count.blocking_lock();
        *count += 1;
        *self.last_failure.blocking_lock() = Some(SystemTime::now());
        
        if *count >= self.max_failures {
            *self.tripped.blocking_lock() = true;
            warn!("Circuit breaker {} tripped after {} failures", self.name, count);
        }
    }
}

// ============================================================================
// POSITION DATA STRUCTURES
// ============================================================================

/// Represents a position on an exchange
/// Quinn: "Must capture ALL aspects of exposure"



#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum PositionSide {
    Long,
    Short,
}


#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum PositionStatus {
    Open,
    Closing,
    Closed,
    Liquidated,
    Unknown,
}

/// Risk metrics for a position

#[derive(Debug, Clone)]
// REMOVED: Duplicate
// pub struct RiskMetrics {
    /// Value at Risk (95% confidence)
    pub var_95: Decimal,
    
    /// Maximum potential loss
    pub max_loss: Decimal,
    
    /// Distance to liquidation (percentage)
    pub liquidation_distance: Decimal,
    
    /// Position weight in portfolio
    pub portfolio_weight: Decimal,
    
    /// Correlation with other positions
    pub correlation_score: Decimal,
}

// ============================================================================
// RECONCILIATION TYPES
// ============================================================================

/// Reconciliation report after comparing states
/// Alex: "Complete visibility into discrepancies"

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ReconciliationReport {
    /// Timestamp of reconciliation
    pub timestamp: DateTime<Utc>,
    
    /// Total positions found on exchanges
    pub exchange_positions: HashMap<String, Vec<Position>>,
    
    /// Total positions in internal state
    pub internal_positions: HashMap<String, Vec<Position>>,
    
    /// Discrepancies found
    pub discrepancies: Vec<Discrepancy>,
    
    /// Reconciliation status
    pub status: ReconciliationStatus,
    
    /// Total exposure on exchanges
    pub total_exchange_exposure: Decimal,
    
    /// Total exposure in internal state
    pub total_internal_exposure: Decimal,
    
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
    
    /// Recommended actions
    pub recommended_actions: Vec<RecommendedAction>,
    
    /// Integrity hash
    pub integrity_hash: String,
}

/// Types of discrepancies

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum Discrepancy {
    /// Position exists on exchange but not internally
    MissingInternal {
        exchange: String,
        position: Position,
        severity: Severity,
    },
    
    /// Position exists internally but not on exchange
    MissingExternal {
        exchange: String,
        position: Position,
        severity: Severity,
    },
    
    /// Size mismatch between exchange and internal
    SizeMismatch {
        exchange: String,
        symbol: String,
        exchange_size: Decimal,
        internal_size: Decimal,
        difference: Decimal,
        severity: Severity,
    },
    
    /// Price mismatch
    PriceMismatch {
        exchange: String,
        symbol: String,
        exchange_price: Decimal,
        internal_price: Decimal,
        difference: Decimal,
        severity: Severity,
    },
    
    /// Leverage mismatch
    LeverageMismatch {
        exchange: String,
        symbol: String,
        exchange_leverage: Decimal,
        internal_leverage: Decimal,
        severity: Severity,
    },
    
    /// Unknown position status
    UnknownStatus {
        exchange: String,
        symbol: String,
        position_id: String,
        severity: Severity,
    },
}


#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum Severity {
    Critical,  // Requires immediate action
    High,      // Significant risk
    Medium,    // Notable discrepancy
    Low,       // Minor issue
}


#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum ReconciliationStatus {
    /// All positions match perfectly
    FullyReconciled,
    
    /// Minor discrepancies within tolerance
    PartiallyReconciled,
    
    /// Critical discrepancies found
    Failed,
    
    /// Could not complete reconciliation
    Error,
}

/// Risk assessment from reconciliation
#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct RiskAssessment {
    /// Overall risk level
    pub risk_level: RiskLevel,
    
    /// Exposure discrepancy percentage
    pub exposure_discrepancy_pct: Decimal,
    
    /// Number of critical discrepancies
    pub critical_count: usize,
    
    /// Estimated potential loss
    pub potential_loss: Decimal,
    
    /// Margin at risk
    pub margin_at_risk: Decimal,
    
    /// Recommended mode
    pub recommended_mode: ControlMode,
}


#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum RiskLevel {
    Extreme,   // Force Emergency mode
    High,      // Downgrade to Manual
    Medium,    // Limit trading
    Low,       // Normal operation
}

/// Recommended actions based on reconciliation
#[derive(Debug, Clone)]
#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum RecommendedAction {
    /// Force emergency mode
    ForceEmergency(String),
    
    /// Close specific position
    ClosePosition {
        exchange: String,
        symbol: String,
        reason: String,
    },
    
    /// Reduce leverage
    ReduceLeverage {
        exchange: String,
        symbol: String,
        target_leverage: Decimal,
    },
    
    /// Sync internal state
    SyncInternalState {
        positions: Vec<Position>,
    },
    
    /// Manual review required
    ManualReview {
        description: String,
        positions: Vec<String>,
    },
}

// ============================================================================
// EXCHANGE CONNECTOR INTERFACE
// ============================================================================

/// Interface for exchange position queries
/// Casey: "Standardized across all exchanges"
#[async_trait::async_trait]
pub trait ExchangeConnector: Send + Sync {
    /// Get all open positions
    async fn get_open_positions(&self) -> Result<Vec<Position>>;
    
    /// Get position by symbol
    async fn get_position(&self, symbol: &str) -> Result<Option<Position>>;
    
    /// Get account balance
    async fn get_balance(&self) -> Result<HashMap<String, Decimal>>;
    
    /// Get open orders
    async fn get_open_orders(&self) -> Result<Vec<Order>>;
    
    /// Verify API connectivity
    async fn verify_connectivity(&self) -> Result<bool>;
    
    /// Get exchange name
    fn exchange_name(&self) -> String;
}

/// Order information


#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum OrderSide {
    Buy,
    Sell,
}


#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}


#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum OrderStatus {
    Pending,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
}

// ============================================================================
// POSITION RECONCILIATION ENGINE
// ============================================================================

/// Main reconciliation engine
/// Morgan: "Mathematical precision in state verification"
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct PositionReconciliationEngine {
    /// Exchange connectors
    exchanges: HashMap<String, Arc<dyn ExchangeConnector>>,
    
    /// Internal position state
    internal_state: Arc<RwLock<HashMap<String, Vec<Position>>>>,
    
    /// Persistence manager
    persistence: Arc<ModePersistenceManager>,
    
    /// Reconciliation configuration
    config: ReconciliationConfig,
    
    /// Circuit breaker for safety
    circuit_breaker: CircuitBreaker,
    
    /// Event broadcaster
    event_tx: broadcast::Sender<ReconciliationEvent>,
    
    /// Last reconciliation report
    last_report: Arc<RwLock<Option<ReconciliationReport>>>,
    
    /// Reconciliation history
    history: Arc<RwLock<Vec<ReconciliationReport>>>,
}

/// Reconciliation configuration

#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ReconciliationConfig {
    /// Maximum acceptable size discrepancy (percentage)
    pub size_tolerance_pct: Decimal,
    
    /// Maximum acceptable price discrepancy (percentage)
    pub price_tolerance_pct: Decimal,
    
    /// Force emergency if exposure discrepancy exceeds
    pub emergency_threshold_pct: Decimal,
    
    /// Downgrade mode if discrepancy exceeds
    pub downgrade_threshold_pct: Decimal,
    
    /// Maximum retries for API calls
    pub max_retries: u32,
    
    /// Timeout for exchange queries
    pub query_timeout: Duration,
    
    /// Enable auto-correction
    pub auto_correction_enabled: bool,
    
    /// Maximum auto-correction size
    pub max_auto_correction_size: Decimal,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            size_tolerance_pct: Decimal::from_str_exact("0.001").unwrap(), // 0.1%
            price_tolerance_pct: Decimal::from_str_exact("0.005").unwrap(), // 0.5%
            emergency_threshold_pct: Decimal::from_str_exact("0.05").unwrap(), // 5%
            downgrade_threshold_pct: Decimal::from_str_exact("0.02").unwrap(), // 2%
            max_retries: 3,
            query_timeout: Duration::from_secs(30),
            auto_correction_enabled: false,
            max_auto_correction_size: Decimal::from(10000), // $10k max
        }
    }
}

/// Reconciliation events

#[derive(Debug, Clone)]
/// TODO: Add docs
pub enum ReconciliationEvent {
    Started(DateTime<Utc>),
    Completed(ReconciliationReport),
    DiscrepancyFound(Discrepancy),
    ActionTaken(RecommendedAction),
    Failed(String),
}

impl PositionReconciliationEngine {
    /// Create new reconciliation engine
    pub fn new(
        exchanges: HashMap<String, Arc<dyn ExchangeConnector>>,
        persistence: Arc<ModePersistenceManager>,
        config: ReconciliationConfig,
    ) -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        
        Self {
            exchanges,
            internal_state: Arc::new(RwLock::new(HashMap::new())),
            persistence,
            config,
            circuit_breaker: CircuitBreaker::new(
                "position_reconciliation".to_string(),
                3,
                Duration::from_secs(60),
                Duration::from_secs(300),
            ),
            event_tx,
            last_report: Arc::new(RwLock::new(None)),
            history: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Perform full reconciliation
    /// Alex: "This is our safety net - must be bulletproof"
    pub async fn reconcile_all(&self) -> Result<ReconciliationReport> {
        info!("Starting position reconciliation across all exchanges");
        
        // Check circuit breaker
        if self.circuit_breaker.is_tripped() {
            bail!("Reconciliation circuit breaker is tripped");
        }
        
        // Send start event
        let _ = self.event_tx.send(ReconciliationEvent::Started(Utc::now()));
        
        // Collect positions from all exchanges
        let exchange_positions = self.query_all_exchanges().await?;
        
        // Get internal state
        let internal_positions = self.internal_state.read().await.clone();
        
        // Find discrepancies
        let discrepancies = self.find_discrepancies(
            &exchange_positions,
            &internal_positions
        ).await?;
        
        // Calculate risk assessment
        let risk_assessment = self.assess_risk(&discrepancies, &exchange_positions).await?;
        
        // Generate recommendations
        let recommended_actions = self.generate_recommendations(
            &discrepancies,
            &risk_assessment
        ).await?;
        
        // Calculate exposures
        let total_exchange_exposure = self.calculate_total_exposure(&exchange_positions);
        let total_internal_exposure = self.calculate_total_exposure(&internal_positions);
        
        // Determine status
        let status = self.determine_status(&discrepancies, &risk_assessment);
        
        // Create report
        let report = ReconciliationReport {
            timestamp: Utc::now(),
            exchange_positions: exchange_positions.clone(),
            internal_positions: internal_positions.clone(),
            discrepancies: discrepancies.clone(),
            status,
            total_exchange_exposure,
            total_internal_exposure,
            risk_assessment,
            recommended_actions: recommended_actions.clone(),
            integrity_hash: self.calculate_integrity_hash(&exchange_positions),
        };
        
        // Store report
        *self.last_report.write().await = Some(report.clone());
        self.history.write().await.push(report.clone());
        
        // Send completion event
        let _ = self.event_tx.send(ReconciliationEvent::Completed(report.clone()));
        
        // Execute critical actions if needed
        if report.risk_assessment.risk_level == RiskLevel::Extreme {
            self.execute_emergency_actions(&recommended_actions).await?;
        }
        
        info!("Reconciliation completed with status: {:?}", status);
        Ok(report)
    }
    
    /// Query all exchanges for positions
    async fn query_all_exchanges(&self) -> Result<HashMap<String, Vec<Position>>> {
        let mut all_positions = HashMap::new();
        
        for (name, connector) in &self.exchanges {
            debug!("Querying positions from exchange: {}", name);
            
            // Retry logic with exponential backoff
            let mut retries = 0;
            let mut positions = None;
            
            while retries < self.config.max_retries {
                match tokio::time::timeout(
                    self.config.query_timeout,
                    connector.get_open_positions()
                ).await {
                    Ok(Ok(pos)) => {
                        positions = Some(pos);
                        break;
                    }
                    Ok(Err(e)) => {
                        warn!("Failed to query {}: {}", name, e);
                        retries += 1;
                        tokio::time::sleep(Duration::from_secs(2_u64.pow(retries))).await;
                    }
                    Err(_) => {
                        warn!("Timeout querying {}", name);
                        retries += 1;
                    }
                }
            }
            
            if let Some(pos) = positions {
                info!("Found {} positions on {}", pos.len(), name);
                all_positions.insert(name.clone(), pos);
            } else {
                error!("Failed to query {} after {} retries", name, self.config.max_retries);
                self.circuit_breaker.record_failure();
                bail!("Failed to query exchange: {}", name);
            }
        }
        
        Ok(all_positions)
    }
    
    /// Find discrepancies between exchange and internal state
    /// Quinn: "Every discrepancy is a potential risk"
    async fn find_discrepancies(
        &self,
        exchange_positions: &HashMap<String, Vec<Position>>,
        internal_positions: &HashMap<String, Vec<Position>>,
    ) -> Result<Vec<Discrepancy>> {
        let mut discrepancies = Vec::new();
        
        // Check each exchange
        for (exchange, ex_positions) in exchange_positions {
            let int_positions = internal_positions.get(exchange)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            
            // Create position maps for efficient lookup
            let ex_map: HashMap<String, &Position> = ex_positions.iter()
                .map(|p| (p.symbol.clone(), p))
                .collect();
            
            let int_map: HashMap<String, &Position> = int_positions.iter()
                .map(|p| (p.symbol.clone(), p))
                .collect();
            
            // Check for missing internal positions
            for (symbol, ex_pos) in &ex_map {
                if !int_map.contains_key(symbol) {
                    discrepancies.push(Discrepancy::MissingInternal {
                        exchange: exchange.clone(),
                        position: (*ex_pos).clone(),
                        severity: self.calculate_severity(&(*ex_pos).risk_metrics),
                    });
                    
                    let _ = self.event_tx.send(ReconciliationEvent::DiscrepancyFound(
                        discrepancies.last().unwrap().clone()
                    ));
                }
            }
            
            // Check for missing external positions
            for (symbol, int_pos) in &int_map {
                if !ex_map.contains_key(symbol) {
                    discrepancies.push(Discrepancy::MissingExternal {
                        exchange: exchange.clone(),
                        position: (*int_pos).clone(),
                        severity: Severity::Critical, // Missing on exchange is critical
                    });
                    
                    let _ = self.event_tx.send(ReconciliationEvent::DiscrepancyFound(
                        discrepancies.last().unwrap().clone()
                    ));
                }
            }
            
            // Check for mismatches in existing positions
            for (symbol, ex_pos) in &ex_map {
                if let Some(int_pos) = int_map.get(symbol) {
                    // Size mismatch
                    let size_diff = (ex_pos.size - int_pos.size).abs();
                    let size_diff_pct = size_diff / ex_pos.size * Decimal::from(100);
                    
                    if size_diff_pct > self.config.size_tolerance_pct {
                        discrepancies.push(Discrepancy::SizeMismatch {
                            exchange: exchange.clone(),
                            symbol: symbol.clone(),
                            exchange_size: ex_pos.size,
                            internal_size: int_pos.size,
                            difference: size_diff,
                            severity: if size_diff_pct > Decimal::from(5) {
                                Severity::Critical
                            } else if size_diff_pct > Decimal::from(2) {
                                Severity::High
                            } else {
                                Severity::Medium
                            },
                        });
                    }
                    
                    // Price mismatch
                    let price_diff = (ex_pos.entry_price - int_pos.entry_price).abs();
                    let price_diff_pct = price_diff / ex_pos.entry_price * Decimal::from(100);
                    
                    if price_diff_pct > self.config.price_tolerance_pct {
                        discrepancies.push(Discrepancy::PriceMismatch {
                            exchange: exchange.clone(),
                            symbol: symbol.clone(),
                            exchange_price: ex_pos.entry_price,
                            internal_price: int_pos.entry_price,
                            difference: price_diff,
                            severity: if price_diff_pct > Decimal::from(2) {
                                Severity::High
                            } else {
                                Severity::Medium
                            },
                        });
                    }
                    
                    // Leverage mismatch
                    if ex_pos.leverage != int_pos.leverage {
                        discrepancies.push(Discrepancy::LeverageMismatch {
                            exchange: exchange.clone(),
                            symbol: symbol.clone(),
                            exchange_leverage: ex_pos.leverage,
                            internal_leverage: int_pos.leverage,
                            severity: if ex_pos.leverage > int_pos.leverage {
                                Severity::High // Higher leverage on exchange is risky
                            } else {
                                Severity::Medium
                            },
                        });
                    }
                }
            }
        }
        
        info!("Found {} discrepancies", discrepancies.len());
        Ok(discrepancies)
    }
    
    /// Assess risk based on discrepancies
    async fn assess_risk(
        &self,
        discrepancies: &[Discrepancy],
        exchange_positions: &HashMap<String, Vec<Position>>,
    ) -> Result<RiskAssessment> {
        let critical_count = discrepancies.iter()
            .filter(|d| matches!(d, 
                Discrepancy::MissingExternal { severity: Severity::Critical, .. } |
                Discrepancy::MissingInternal { severity: Severity::Critical, .. } |
                Discrepancy::SizeMismatch { severity: Severity::Critical, .. }
            ))
            .count();
        
        // Calculate exposure discrepancy
        let total_exchange_exposure = self.calculate_total_exposure(exchange_positions);
        let internal_state = self.internal_state.read().await;
        let total_internal_exposure = self.calculate_total_exposure(&*internal_state);
        
        let exposure_discrepancy = (total_exchange_exposure - total_internal_exposure).abs();
        let exposure_discrepancy_pct = if total_exchange_exposure > Decimal::ZERO {
            exposure_discrepancy / total_exchange_exposure * Decimal::from(100)
        } else {
            Decimal::ZERO
        };
        
        // Calculate potential loss
        let potential_loss = discrepancies.iter()
            .filter_map(|d| match d {
                Discrepancy::MissingInternal { position, .. } => {
                    Some(position.size * position.entry_price)
                }
                Discrepancy::SizeMismatch { difference, exchange, symbol, .. } => {
                    // Find the position to get price
                    exchange_positions.get(exchange)
                        .and_then(|positions| positions.iter()
                            .find(|p| p.symbol == *symbol))
                        .map(|p| *difference * p.entry_price)
                }
                _ => None,
            })
            .sum();
        
        // Calculate margin at risk
        let margin_at_risk = exchange_positions.values()
            .flat_map(|positions| positions.iter())
            .filter(|p| p.risk_metrics.liquidation_distance < Decimal::from(10)) // <10% from liquidation
            .map(|p| p.margin)
            .sum();
        
        // Determine risk level
        let risk_level = if critical_count > 2 || 
            exposure_discrepancy_pct > self.config.emergency_threshold_pct {
            RiskLevel::Extreme
        } else if critical_count > 0 || 
            exposure_discrepancy_pct > self.config.downgrade_threshold_pct {
            RiskLevel::High
        } else if !discrepancies.is_empty() {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        // Determine recommended mode
        let recommended_mode = match risk_level {
            RiskLevel::Extreme => ControlMode::Emergency,
            RiskLevel::High => ControlMode::Manual,
            RiskLevel::Medium => ControlMode::SemiAuto,
            RiskLevel::Low => ControlMode::FullAuto,
        };
        
        Ok(RiskAssessment {
            risk_level,
            exposure_discrepancy_pct,
            critical_count,
            potential_loss,
            margin_at_risk,
            recommended_mode,
        })
    }
    
    /// Generate recommended actions
    async fn generate_recommendations(
        &self,
        discrepancies: &[Discrepancy],
        risk_assessment: &RiskAssessment,
    ) -> Result<Vec<RecommendedAction>> {
        let mut actions = Vec::new();
        
        // Force emergency if extreme risk
        if risk_assessment.risk_level == RiskLevel::Extreme {
            actions.push(RecommendedAction::ForceEmergency(
                format!("Critical discrepancies: {} found, exposure discrepancy: {:.2}%",
                    risk_assessment.critical_count,
                    risk_assessment.exposure_discrepancy_pct)
            ));
        }
        
        // Handle specific discrepancies
        for discrepancy in discrepancies {
            match discrepancy {
                Discrepancy::MissingExternal { position, .. } => {
                    // Position exists internally but not on exchange - very dangerous
                    actions.push(RecommendedAction::ClosePosition {
                        exchange: position.exchange.clone(),
                        symbol: position.symbol.clone(),
                        reason: "Position not found on exchange - possible fill or liquidation".to_string(),
                    });
                }
                
                Discrepancy::MissingInternal { position, .. } => {
                    // Position exists on exchange but not internally
                    if self.config.auto_correction_enabled && 
                       position.size * position.entry_price <= self.config.max_auto_correction_size {
                        actions.push(RecommendedAction::SyncInternalState {
                            positions: vec![position.clone()],
                        });
                    } else {
                        actions.push(RecommendedAction::ManualReview {
                            description: format!("Unknown position found on exchange: {}", position.symbol),
                            positions: vec![position.exchange_position_id.clone()],
                        });
                    }
                }
                
                Discrepancy::LeverageMismatch { exchange, symbol, exchange_leverage, .. } => {
                    if *exchange_leverage > Decimal::from(5) {
                        actions.push(RecommendedAction::ReduceLeverage {
                            exchange: exchange.clone(),
                            symbol: symbol.clone(),
                            target_leverage: Decimal::from(3),
                        });
                    }
                }
                
                _ => {}
            }
        }
        
        Ok(actions)
    }
    
    /// Calculate total exposure across positions
    fn calculate_total_exposure(&self, positions: &HashMap<String, Vec<Position>>) -> Decimal {
        positions.values()
            .flat_map(|pos_vec| pos_vec.iter())
            .map(|p| p.size * p.entry_price)
            .sum()
    }
    
    /// Calculate severity based on risk metrics
    fn calculate_severity(&self, risk_metrics: &RiskMetrics) -> Severity {
        if risk_metrics.max_loss > Decimal::from(100000) || 
           risk_metrics.liquidation_distance < Decimal::from(5) {
            Severity::Critical
        } else if risk_metrics.max_loss > Decimal::from(50000) ||
                  risk_metrics.liquidation_distance < Decimal::from(10) {
            Severity::High
        } else if risk_metrics.max_loss > Decimal::from(10000) {
            Severity::Medium
        } else {
            Severity::Low
        }
    }
    
    /// Determine reconciliation status
    fn determine_status(
        &self,
        discrepancies: &[Discrepancy],
        risk_assessment: &RiskAssessment,
    ) -> ReconciliationStatus {
        if discrepancies.is_empty() {
            ReconciliationStatus::FullyReconciled
        } else if risk_assessment.risk_level == RiskLevel::Low {
            ReconciliationStatus::PartiallyReconciled
        } else if risk_assessment.risk_level == RiskLevel::Extreme {
            ReconciliationStatus::Failed
        } else {
            ReconciliationStatus::PartiallyReconciled
        }
    }
    
    /// Calculate integrity hash for positions
    fn calculate_integrity_hash(&self, positions: &HashMap<String, Vec<Position>>) -> String {
        let mut hasher = Sha256::new();
        
        // Sort for consistent hashing
        let mut sorted_exchanges: Vec<_> = positions.keys().collect();
        sorted_exchanges.sort();
        
        for exchange in sorted_exchanges {
            if let Some(pos_vec) = positions.get(exchange) {
                let mut sorted_positions = pos_vec.clone();
                sorted_positions.sort_by_key(|p| p.symbol.clone());
                
                for pos in sorted_positions {
                    hasher.update(exchange.as_bytes());
                    hasher.update(pos.symbol.as_bytes());
                    hasher.update(pos.size.to_string().as_bytes());
                    hasher.update(pos.entry_price.to_string().as_bytes());
                }
            }
        }
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Execute emergency actions
    async fn execute_emergency_actions(&self, actions: &[RecommendedAction]) -> Result<()> {
        error!("Executing emergency actions due to extreme risk");
        
        for action in actions {
            match action {
                RecommendedAction::ForceEmergency(reason) => {
                    // Force emergency mode through persistence
                    self.persistence.save_mode_state(
                        ControlMode::Emergency,
                        reason.clone(),
                        "PositionReconciliation".to_string(),
                        serde_json::json!({
                            "triggered_by": "reconciliation",
                            "risk_level": "extreme"
                        }),
                    ).await?;
                    
                    error!("FORCED EMERGENCY MODE: {}", reason);
                }
                
                RecommendedAction::ClosePosition { exchange, symbol, reason } => {
                    warn!("Position close required on {} for {}: {}", 
                          exchange, symbol, reason);
                    // TODO: Implement actual position closing through exchange connector
                }
                
                _ => {}
            }
            
            let _ = self.event_tx.send(ReconciliationEvent::ActionTaken(action.clone()));
        }
        
        Ok(())
    }
    
    /// Update internal state from exchange data
    pub async fn sync_from_exchanges(&self) -> Result<()> {
        info!("Syncing internal state from exchanges");
        
        let exchange_positions = self.query_all_exchanges().await?;
        
        *self.internal_state.write().await = exchange_positions;
        
        info!("Internal state synchronized");
        Ok(())
    }
    
    /// Get last reconciliation report
    pub async fn get_last_report(&self) -> Option<ReconciliationReport> {
        self.last_report.read().await.clone()
    }
    
    /// Subscribe to reconciliation events
    pub fn subscribe(&self) -> broadcast::Receiver<ReconciliationEvent> {
        self.event_tx.subscribe()
    }
}

// ============================================================================
// AUTOMATED RECONCILIATION SCHEDULER
// ============================================================================

/// Automated reconciliation scheduler
/// Riley: "Continuous verification at configurable intervals"
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct ReconciliationScheduler {
    engine: Arc<PositionReconciliationEngine>,
    interval: Duration,
    running: Arc<Mutex<bool>>,
}

impl ReconciliationScheduler {
    pub fn new(engine: Arc<PositionReconciliationEngine>, interval: Duration) -> Self {
        Self {
            engine,
            interval,
            running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Start automated reconciliation
    pub async fn start(&self) {
        let mut running = self.running.lock().await;
        if *running {
            warn!("Reconciliation scheduler already running");
            return;
        }
        *running = true;
        drop(running);
        
        let engine = self.engine.clone();
        let interval = self.interval;
        let running = self.running.clone();
        
        tokio::spawn(async move {
            info!("Starting reconciliation scheduler with interval: {:?}", interval);
            
            while *running.lock().await {
                match engine.reconcile_all().await {
                    Ok(report) => {
                        info!("Scheduled reconciliation completed: {:?}", report.status);
                    }
                    Err(e) => {
                        error!("Scheduled reconciliation failed: {}", e);
                    }
                }
                
                tokio::time::sleep(interval).await;
            }
        });
    }
    
    /// Stop automated reconciliation
    pub async fn stop(&self) {
        *self.running.lock().await = false;
        info!("Reconciliation scheduler stopped");
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[tokio::test]
    async fn test_severity_calculation() {
        // Create a mock persistence manager
        let persistence = Arc::new(ModePersistenceManager::new(
            "sqlite::memory:",
            Default::default(),
            "test".to_string(),
        ).await.unwrap());
        
        let engine = PositionReconciliationEngine::new(
            HashMap::new(),
            persistence,
            Default::default(),
        );
        
        let high_risk = RiskMetrics {
            var_95: dec!(50000),
            max_loss: dec!(150000),
            liquidation_distance: dec!(3),
            portfolio_weight: dec!(0.5),
            correlation_score: dec!(0.8),
        };
        
        assert_eq!(engine.calculate_severity(&high_risk), Severity::Critical);
        
        let low_risk = RiskMetrics {
            var_95: dec!(1000),
            max_loss: dec!(5000),
            liquidation_distance: dec!(50),
            portfolio_weight: dec!(0.05),
            correlation_score: dec!(0.2),
        };
        
        assert_eq!(engine.calculate_severity(&low_risk), Severity::Low);
    }
    
    #[tokio::test]
    async fn test_exposure_calculation() {
        let persistence = Arc::new(ModePersistenceManager::new(
            "sqlite::memory:",
            Default::default(),
            "test".to_string(),
        ).await.unwrap());
        
        let engine = PositionReconciliationEngine::new(
            HashMap::new(),
            persistence,
            Default::default(),
        );
        
        let mut positions = HashMap::new();
        positions.insert("binance".to_string(), vec![
            Position {
                exchange: "binance".to_string(),
                symbol: "BTC/USDT".to_string(),
                side: PositionSide::Long,
                size: dec!(0.5),
                entry_price: dec!(50000),
                mark_price: Some(dec!(51000)),
                unrealized_pnl: Some(dec!(500)),
                realized_pnl: dec!(0),
                margin: dec!(5000),
                leverage: dec!(5),
                liquidation_price: Some(dec!(45000)),
                exchange_position_id: "123".to_string(),
                last_updated: Utc::now(),
                status: PositionStatus::Open,
                open_orders: vec![],
                risk_metrics: RiskMetrics {
                    var_95: dec!(2500),
                    max_loss: dec!(5000),
                    liquidation_distance: dec!(10),
                    portfolio_weight: dec!(0.25),
                    correlation_score: dec!(0.5),
                },
            },
        ]);
        
        let exposure = engine.calculate_total_exposure(&positions);
        assert_eq!(exposure, dec!(25000)); // 0.5 * 50000
    }
    
    #[tokio::test]
    async fn test_reconciliation_status() {
        let config = ReconciliationConfig {
            size_tolerance_pct: dec!(0.001),
            price_tolerance_pct: dec!(0.005),
            emergency_threshold_pct: dec!(0.05),
            downgrade_threshold_pct: dec!(0.02),
            max_retries: 3,
            query_timeout: Duration::from_secs(30),
            auto_correction_enabled: false,
            max_auto_correction_size: dec!(10000),
        };
        
        // Test with no discrepancies
        let engine = PositionReconciliationEngine::new(
            HashMap::new(),
            Arc::new(ModePersistenceManager::new(
                "sqlite::memory:",
                Default::default(),
                "test".to_string(),
            ).await.unwrap()),
            config,
        );
        
        let status = engine.determine_status(
            &[],
            &RiskAssessment {
                risk_level: RiskLevel::Low,
                exposure_discrepancy_pct: dec!(0),
                critical_count: 0,
                potential_loss: dec!(0),
                margin_at_risk: dec!(0),
                recommended_mode: ControlMode::FullAuto,
            }
        );
        
        assert_eq!(status, ReconciliationStatus::FullyReconciled);
    }
}
