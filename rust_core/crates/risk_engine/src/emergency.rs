// Emergency Stop and Kill Switch
// Quinn's ultimate authority - when triggered, ALL trading stops

use parking_lot::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::{error, warn, info};

/// Trip conditions that trigger emergency stop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TripCondition {
    /// Daily loss exceeded
    DailyLossExceeded { loss: Decimal, limit: Decimal },
    
    /// Drawdown exceeded
    DrawdownExceeded { drawdown: Decimal, limit: Decimal },
    
    /// Consecutive losses
    ConsecutiveLosses { count: usize, limit: usize },
    
    /// System error
    SystemError { error: String },
    
    /// Manual trigger (Quinn's override)
    ManualTrigger { reason: String, triggered_by: String },
    
    /// Exchange issue
    ExchangeIssue { exchange: String, issue: String },
    
    /// Correlation breach
    CorrelationBreach { correlation: f64, limit: f64 },
    
    /// Circuit breaker cascade
    CircuitBreakerCascade { affected_components: Vec<String> },
}

/// Kill switch - the big red button
pub struct KillSwitch {
    is_active: Arc<AtomicBool>,
    triggered_at: Arc<RwLock<Option<Instant>>>,
    trigger_reason: Arc<RwLock<Option<TripCondition>>>,
    auto_reset_after: Option<Duration>,
    trigger_count: Arc<AtomicU64>,
}

impl KillSwitch {
    pub fn new(auto_reset_after: Option<Duration>) -> Self {
        Self {
            is_active: Arc::new(AtomicBool::new(false)),
            triggered_at: Arc::new(RwLock::new(None)),
            trigger_reason: Arc::new(RwLock::new(None)),
            auto_reset_after,
            trigger_count: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Activate kill switch
    pub fn activate(&self, reason: TripCondition) {
        error!("🚨 KILL SWITCH ACTIVATED: {:?}", reason);
        
        self.is_active.store(true, Ordering::SeqCst);
        *self.triggered_at.write() = Some(Instant::now());
        *self.trigger_reason.write() = Some(reason);
        self.trigger_count.fetch_add(1, Ordering::Relaxed);
        
        // Log to all systems
        error!("ALL TRADING HALTED - Quinn's kill switch activated");
    }
    
    /// Deactivate kill switch (requires authorization)
    pub fn deactivate(&self, authorized_by: &str) -> Result<(), String> {
        if !self.is_active.load(Ordering::SeqCst) {
            return Err("Kill switch is not active".to_string());
        }
        
        // In production, verify authorization
        if authorized_by != "Quinn" && authorized_by != "Alex" {
            return Err(format!("{} not authorized to deactivate kill switch", authorized_by));
        }
        
        info!("Kill switch deactivated by {}", authorized_by);
        self.is_active.store(false, Ordering::SeqCst);
        *self.triggered_at.write() = None;
        *self.trigger_reason.write() = None;
        
        Ok(())
    }
    
    /// Check if kill switch is active
    pub fn is_active(&self) -> bool {
        let active = self.is_active.load(Ordering::SeqCst);
        
        // Check auto-reset
        if active && self.auto_reset_after.is_some() {
            if let Some(triggered) = *self.triggered_at.read() {
                if triggered.elapsed() > self.auto_reset_after.unwrap() {
                    info!("Kill switch auto-reset after timeout");
                    self.is_active.store(false, Ordering::SeqCst);
                    return false;
                }
            }
        }
        
        active
    }
    
    /// Get trigger reason
    pub fn get_trigger_reason(&self) -> Option<TripCondition> {
        self.trigger_reason.read().clone()
    }
    
    /// Get trigger count
    pub fn get_trigger_count(&self) -> u64 {
        self.trigger_count.load(Ordering::Relaxed)
    }
}

/// Emergency stop system
pub struct EmergencyStop {
    kill_switch: Arc<KillSwitch>,
    conditions: Arc<RwLock<Vec<EmergencyCondition>>>,
    last_check: Arc<RwLock<Instant>>,
    check_interval: Duration,
}

struct EmergencyCondition {
    name: String,
    check_fn: Arc<dyn Fn() -> Option<TripCondition> + Send + Sync>,
    enabled: bool,
}

impl EmergencyStop {
    pub fn new(kill_switch: Arc<KillSwitch>) -> Self {
        Self {
            kill_switch,
            conditions: Arc::new(RwLock::new(Vec::new())),
            last_check: Arc::new(RwLock::new(Instant::now())),
            check_interval: Duration::from_millis(100), // Check every 100ms
        }
    }
    
    /// Add emergency condition
    pub fn add_condition<F>(&self, name: String, check_fn: F)
    where
        F: Fn() -> Option<TripCondition> + Send + Sync + 'static,
    {
        self.conditions.write().push(EmergencyCondition {
            name,
            check_fn: Arc::new(check_fn),
            enabled: true,
        });
    }
    
    /// Check all conditions
    pub fn check_conditions(&self) {
        // Rate limit checks
        let mut last_check = self.last_check.write();
        if last_check.elapsed() < self.check_interval {
            return;
        }
        *last_check = Instant::now();
        
        // Don't check if already triggered
        if self.kill_switch.is_active() {
            return;
        }
        
        // Check each condition
        for condition in self.conditions.read().iter() {
            if !condition.enabled {
                continue;
            }
            
            if let Some(trip_reason) = (condition.check_fn)() {
                warn!("Emergency condition '{}' triggered", condition.name);
                self.kill_switch.activate(trip_reason);
                break; // Stop checking after first trigger
            }
        }
    }
    
    /// Manual emergency stop (Quinn's panic button)
    pub fn manual_stop(&self, reason: String, triggered_by: String) {
        self.kill_switch.activate(TripCondition::ManualTrigger {
            reason,
            triggered_by,
        });
    }
    
    /// Check specific condition
    pub fn check_daily_loss(&self, current_loss: Decimal, limit: Decimal) {
        if current_loss.abs() > limit {
            self.kill_switch.activate(TripCondition::DailyLossExceeded {
                loss: current_loss,
                limit,
            });
        }
    }
    
    /// Check drawdown
    pub fn check_drawdown(&self, current_dd: Decimal, limit: Decimal) {
        if current_dd > limit {
            self.kill_switch.activate(TripCondition::DrawdownExceeded {
                drawdown: current_dd,
                limit,
            });
        }
    }
    
    /// Check consecutive losses
    pub fn check_consecutive_losses(&self, count: usize, limit: usize) {
        if count > limit {
            self.kill_switch.activate(TripCondition::ConsecutiveLosses {
                count,
                limit,
            });
        }
    }
    
    /// Report system error
    pub fn report_system_error(&self, error: String) {
        self.kill_switch.activate(TripCondition::SystemError { error });
    }
    
    /// Report exchange issue
    pub fn report_exchange_issue(&self, exchange: String, issue: String) {
        self.kill_switch.activate(TripCondition::ExchangeIssue { exchange, issue });
    }
    
    /// Get status
    pub fn get_status(&self) -> EmergencyStatus {
        EmergencyStatus {
            kill_switch_active: self.kill_switch.is_active(),
            trigger_reason: self.kill_switch.get_trigger_reason(),
            trigger_count: self.kill_switch.get_trigger_count(),
            conditions_count: self.conditions.read().len(),
            last_check: *self.last_check.read(),
        }
    }
}

/// Emergency system status
#[derive(Debug, Clone)]
pub struct EmergencyStatus {
    pub kill_switch_active: bool,
    pub trigger_reason: Option<TripCondition>,
    pub trigger_count: u64,
    pub conditions_count: usize,
    pub last_check: Instant,
}

/// Emergency recovery plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryPlan {
    pub steps: Vec<RecoveryStep>,
    pub estimated_time: Duration,
    pub requires_manual_approval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryStep {
    pub order: usize,
    pub description: String,
    pub action: RecoveryAction,
    pub requires_confirmation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    CloseAllPositions,
    CancelAllOrders,
    ReduceExposure { target_pct: Decimal },
    DisableStrategy { strategy_id: String },
    RestartComponent { component: String },
    ManualIntervention { instructions: String },
}

impl RecoveryPlan {
    /// Create standard recovery plan
    pub fn standard() -> Self {
        Self {
            steps: vec![
                RecoveryStep {
                    order: 1,
                    description: "Cancel all pending orders".to_string(),
                    action: RecoveryAction::CancelAllOrders,
                    requires_confirmation: false,
                },
                RecoveryStep {
                    order: 2,
                    description: "Close all positions".to_string(),
                    action: RecoveryAction::CloseAllPositions,
                    requires_confirmation: true,
                },
                RecoveryStep {
                    order: 3,
                    description: "Review and fix issue".to_string(),
                    action: RecoveryAction::ManualIntervention {
                        instructions: "Investigate root cause and implement fix".to_string(),
                    },
                    requires_confirmation: true,
                },
            ],
            estimated_time: Duration::from_secs(300), // 5 minutes
            requires_manual_approval: true,
        }
    }
    
    /// Create aggressive recovery (Quinn's nuclear option)
    pub fn aggressive() -> Self {
        Self {
            steps: vec![
                RecoveryStep {
                    order: 1,
                    description: "IMMEDIATE: Close all positions at market".to_string(),
                    action: RecoveryAction::CloseAllPositions,
                    requires_confirmation: false, // No confirmation in emergency
                },
                RecoveryStep {
                    order: 2,
                    description: "Cancel ALL orders across all exchanges".to_string(),
                    action: RecoveryAction::CancelAllOrders,
                    requires_confirmation: false,
                },
                RecoveryStep {
                    order: 3,
                    description: "Disconnect from all exchanges".to_string(),
                    action: RecoveryAction::ManualIntervention {
                        instructions: "Disconnect all exchange connections immediately".to_string(),
                    },
                    requires_confirmation: false,
                },
            ],
            estimated_time: Duration::from_secs(30), // 30 seconds
            requires_manual_approval: false, // Execute immediately
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    
    #[test]
    fn test_kill_switch() {
        let kill_switch = KillSwitch::new(None);
        
        assert!(!kill_switch.is_active());
        
        // Activate
        kill_switch.activate(TripCondition::ManualTrigger {
            reason: "Test".to_string(),
            triggered_by: "Test".to_string(),
        });
        
        assert!(kill_switch.is_active());
        assert_eq!(kill_switch.get_trigger_count(), 1);
        
        // Deactivate with authorization
        assert!(kill_switch.deactivate("Quinn").is_ok());
        assert!(!kill_switch.is_active());
        
        // Unauthorized deactivation should fail
        kill_switch.activate(TripCondition::SystemError {
            error: "Test error".to_string(),
        });
        assert!(kill_switch.deactivate("Unknown").is_err());
        assert!(kill_switch.is_active());
    }
    
    #[test]
    fn test_auto_reset() {
        let kill_switch = KillSwitch::new(Some(Duration::from_millis(100)));
        
        kill_switch.activate(TripCondition::ManualTrigger {
            reason: "Test".to_string(),
            triggered_by: "Test".to_string(),
        });
        
        assert!(kill_switch.is_active());
        
        // Wait for auto-reset
        std::thread::sleep(Duration::from_millis(150));
        assert!(!kill_switch.is_active());
    }
    
    #[test]
    fn test_emergency_conditions() {
        let kill_switch = Arc::new(KillSwitch::new(None));
        let emergency = EmergencyStop::new(kill_switch.clone());
        
        // Add condition that triggers
        emergency.add_condition(
            "test_condition".to_string(),
            || Some(TripCondition::SystemError {
                error: "Test error".to_string(),
            }),
        );
        
        emergency.check_conditions();
        assert!(kill_switch.is_active());
    }
}