// Central Emergency Coordinator - System-Wide Kill Switch
// Team: Alex (Lead) + Quinn (Risk) + Full Team
// CRITICAL: Single point of emergency control for entire system
// References:
// - "Building Reliable Trading Systems" - Aldridge (2013)
// - "Risk Controls for Algorithmic Trading" - SEC (2015)

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use parking_lot::RwLock;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tracing::{error, warn, info};
use tokio::sync::broadcast;
use serde::{Serialize, Deserialize};

/// Component that can be shutdown in emergency
#[async_trait]
pub trait Shutdownable: Send + Sync {
    /// Component name for logging
    fn name(&self) -> &str;
    
    /// Cancel all pending operations
    async fn cancel_all_orders(&self) -> Result<(), String>;
    
    /// Emergency liquidate positions
    async fn emergency_liquidate(&self) -> Result<(), String>;
    
    /// Graceful shutdown
    async fn shutdown(&self) -> Result<(), String>;
    
    /// Check if component is healthy
    fn is_healthy(&self) -> bool;
}

/// Emergency trigger reasons
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergencyReason {
    ManualTrigger,           // User initiated
    MaxDrawdownExceeded,     // Portfolio drawdown limit hit
    SystemFailure,           // Critical component failure
    ExchangeDisconnection,   // Lost connection to all exchanges
    DataFeedLoss,           // Lost market data
    RiskLimitBreach,        // Major risk limit violated
    CircuitBreakerCascade,  // Multiple circuit breakers tripped
    UnauthorizedAccess,     // Security breach detected
    MemoryExhaustion,       // System running out of memory
    LatencySpike,          // Extreme latency detected
}

/// Emergency coordinator state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergencyState {
    Normal,           // System operating normally
    Warning,          // Warning conditions detected
    Emergency,        // Emergency shutdown initiated
    Liquidating,      // Liquidating all positions
    Halted,          // System fully halted
}

/// CRITICAL: Central Emergency Coordinator
/// Alex: "This is our nuclear option - one button to stop everything"
pub struct EmergencyCoordinator {
    /// Current state
    state: Arc<RwLock<EmergencyState>>,
    
    /// Global kill switch
    kill_switch: Arc<AtomicBool>,
    
    /// Components to shutdown
    components: Arc<RwLock<Vec<Arc<dyn Shutdownable>>>>,
    
    /// Shutdown broadcast channel
    shutdown_tx: broadcast::Sender<EmergencyReason>,
    
    /// Emergency log
    emergency_log: Arc<RwLock<Vec<EmergencyEvent>>>,
    
    /// Statistics
    triggers_activated: Arc<AtomicU64>,
    false_positives: Arc<AtomicU64>,
    avg_shutdown_time_ms: Arc<AtomicU64>,
    
    /// Thresholds
    max_drawdown_pct: f64,
    max_latency_ms: u64,
    min_healthy_components_pct: f64,
}

/// Emergency event log entry
#[derive(Debug, Clone)]
pub struct EmergencyEvent {
    pub timestamp: DateTime<Utc>,
    pub reason: EmergencyReason,
    pub state_before: EmergencyState,
    pub state_after: EmergencyState,
    pub details: String,
    pub components_affected: usize,
    pub shutdown_time_ms: Option<u64>,
}

impl EmergencyCoordinator {
    pub fn new(
        max_drawdown_pct: f64,
        max_latency_ms: u64,
        min_healthy_components_pct: f64,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(100);
        
        Self {
            state: Arc::new(RwLock::new(EmergencyState::Normal)),
            kill_switch: Arc::new(AtomicBool::new(false)),
            components: Arc::new(RwLock::new(Vec::new())),
            shutdown_tx,
            emergency_log: Arc::new(RwLock::new(Vec::with_capacity(1000))),
            triggers_activated: Arc::new(AtomicU64::new(0)),
            false_positives: Arc::new(AtomicU64::new(0)),
            avg_shutdown_time_ms: Arc::new(AtomicU64::new(0)),
            max_drawdown_pct,
            max_latency_ms,
            min_healthy_components_pct,
        }
    }
    
    /// Register a component for emergency control
    pub fn register_component(&self, component: Arc<dyn Shutdownable>) {
        info!("Registered component for emergency control: {}", component.name());
        self.components.write().push(component);
    }
    
    /// Subscribe to shutdown events
    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<EmergencyReason> {
        self.shutdown_tx.subscribe()
    }
    
    /// CRITICAL: Trigger emergency shutdown
    /// Quinn: "This must work 100% of the time, no exceptions!"
    pub async fn trigger_emergency(&self, reason: EmergencyReason) -> Result<(), String> {
        error!("EMERGENCY SHUTDOWN TRIGGERED: {:?}", reason);
        
        let start_time = std::time::Instant::now();
        let state_before = *self.state.read();
        
        // Immediate state change
        *self.state.write() = EmergencyState::Emergency;
        
        // Activate kill switch
        self.kill_switch.store(true, Ordering::SeqCst);
        
        // Broadcast emergency to all subscribers
        let _ = self.shutdown_tx.send(reason);
        
        // Track activation
        self.triggers_activated.fetch_add(1, Ordering::Relaxed);
        
        // Phase 1: Stop all new operations
        info!("Phase 1: Stopping all new operations");
        self.stop_new_operations().await?;
        
        // Phase 2: Cancel all pending orders
        info!("Phase 2: Cancelling all pending orders");
        self.cancel_all_orders().await?;
        
        // Phase 3: Emergency liquidation (if needed)
        if matches!(reason, EmergencyReason::MaxDrawdownExceeded | 
                           EmergencyReason::RiskLimitBreach) {
            info!("Phase 3: Emergency liquidation");
            *self.state.write() = EmergencyState::Liquidating;
            self.emergency_liquidate_all().await?;
        }
        
        // Phase 4: Graceful shutdown
        info!("Phase 4: Graceful shutdown of all components");
        self.shutdown_all_components().await?;
        
        // Final state
        *self.state.write() = EmergencyState::Halted;
        
        let shutdown_time = start_time.elapsed().as_millis() as u64;
        
        // Update average shutdown time
        let old_avg = self.avg_shutdown_time_ms.load(Ordering::Relaxed);
        let new_avg = (old_avg * 9 + shutdown_time) / 10;
        self.avg_shutdown_time_ms.store(new_avg, Ordering::Relaxed);
        
        // Log the event
        self.emergency_log.write().push(EmergencyEvent {
            timestamp: Utc::now(),
            reason,
            state_before,
            state_after: EmergencyState::Halted,
            details: format!("Emergency shutdown completed in {}ms", shutdown_time),
            components_affected: self.components.read().len(),
            shutdown_time_ms: Some(shutdown_time),
        });
        
        error!(
            "EMERGENCY SHUTDOWN COMPLETE - {} components halted in {}ms",
            self.components.read().len(),
            shutdown_time
        );
        
        Ok(())
    }
    
    /// Stop all new operations
    async fn stop_new_operations(&self) -> Result<(), String> {
        // Kill switch already activated, components should check this
        info!("Kill switch activated - no new operations allowed");
        Ok(())
    }
    
    /// Cancel all pending orders across all components
    async fn cancel_all_orders(&self) -> Result<(), String> {
        let components = self.components.read().clone();
        let mut errors = Vec::new();
        
        for component in components {
            info!("Cancelling orders for: {}", component.name());
            
            if let Err(e) = component.cancel_all_orders().await {
                error!("Failed to cancel orders for {}: {}", component.name(), e);
                errors.push(format!("{}: {}", component.name(), e));
            }
        }
        
        if !errors.is_empty() {
            Err(format!("Order cancellation errors: {:?}", errors))
        } else {
            Ok(())
        }
    }
    
    /// Emergency liquidate all positions
    async fn emergency_liquidate_all(&self) -> Result<(), String> {
        let components = self.components.read().clone();
        let mut errors = Vec::new();
        
        for component in components {
            warn!("Emergency liquidating: {}", component.name());
            
            if let Err(e) = component.emergency_liquidate().await {
                error!("Failed to liquidate {}: {}", component.name(), e);
                errors.push(format!("{}: {}", component.name(), e));
            }
        }
        
        if !errors.is_empty() {
            Err(format!("Liquidation errors: {:?}", errors))
        } else {
            Ok(())
        }
    }
    
    /// Shutdown all components gracefully
    async fn shutdown_all_components(&self) -> Result<(), String> {
        let components = self.components.read().clone();
        let mut errors = Vec::new();
        
        for component in components {
            info!("Shutting down: {}", component.name());
            
            if let Err(e) = component.shutdown().await {
                error!("Failed to shutdown {}: {}", component.name(), e);
                errors.push(format!("{}: {}", component.name(), e));
            }
        }
        
        if !errors.is_empty() {
            Err(format!("Shutdown errors: {:?}", errors))
        } else {
            Ok(())
        }
    }
    
    /// Check system health and trigger emergency if needed
    pub async fn health_check(&self) -> HealthStatus {
        let total_components = self.components.read().len();
        if total_components == 0 {
            return HealthStatus {
                state: *self.state.read(),
                healthy_components: 0,
                total_components: 0,
                health_percentage: 0.0,
                warnings: vec!["No components registered".to_string()],
            };
        }
        
        let healthycount = self.components
            .read()
            .iter()
            .filter(|c| c.is_healthy())
            .count();
        
        let health_percentage = (healthycount as f64 / total_components as f64) * 100.0;
        let mut warnings = Vec::new();
        
        // Check component health threshold
        if health_percentage < self.min_healthy_components_pct {
            warnings.push(format!(
                "Only {:.1}% of components healthy (minimum: {:.1}%)",
                health_percentage, self.min_healthy_components_pct
            ));
            
            // Auto-trigger if too many components down
            if health_percentage < 50.0 {
                error!("CRITICAL: Less than 50% of components healthy!");
                let _ = self.trigger_emergency(EmergencyReason::SystemFailure).await;
            }
        }
        
        HealthStatus {
            state: *self.state.read(),
            healthy_components: healthycount,
            total_components,
            health_percentage,
            warnings,
        }
    }
    
    /// Check if emergency is active
    pub fn is_emergency_active(&self) -> bool {
        self.kill_switch.load(Ordering::SeqCst)
    }
    
    /// Get current state
    pub fn getstate(&self) -> EmergencyState {
        *self.state.read()
    }
    
    /// Reset after emergency (requires manual intervention)
    /// Alex: "Only after thorough investigation and fixes!"
    pub async fn reset(&self, admin_key: &str) -> Result<(), String> {
        // Simple admin check (in production, use proper auth)
        if admin_key != "ADMIN_RESET_KEY_2025" {
            return Err("Invalid admin key".to_string());
        }
        
        if *self.state.read() != EmergencyState::Halted {
            return Err("Can only reset from Halted state".to_string());
        }
        
        info!("System reset initiated by admin");
        
        // Reset state
        *self.state.write() = EmergencyState::Normal;
        self.kill_switch.store(false, Ordering::SeqCst);
        
        // Log reset
        self.emergency_log.write().push(EmergencyEvent {
            timestamp: Utc::now(),
            reason: EmergencyReason::ManualTrigger,
            state_before: EmergencyState::Halted,
            state_after: EmergencyState::Normal,
            details: "System reset by admin".to_string(),
            components_affected: 0,
            shutdown_time_ms: None,
        });
        
        Ok(())
    }
    
    /// Get emergency statistics
    pub fn get_statistics(&self) -> EmergencyStatistics {
        EmergencyStatistics {
            currentstate: *self.state.read(),
            kill_switch_active: self.kill_switch.load(Ordering::SeqCst),
            triggers_activated: self.triggers_activated.load(Ordering::Relaxed),
            false_positives: self.false_positives.load(Ordering::Relaxed),
            avg_shutdown_time_ms: self.avg_shutdown_time_ms.load(Ordering::Relaxed),
            registered_components: self.components.read().len(),
            recent_events: self.emergency_log.read().iter()
                .rev()
                .take(10)
                .cloned()
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub state: EmergencyState,
    pub healthy_components: usize,
    pub total_components: usize,
    pub health_percentage: f64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct EmergencyStatistics {
    pub currentstate: EmergencyState,
    pub kill_switch_active: bool,
    pub triggers_activated: u64,
    pub false_positives: u64,
    pub avg_shutdown_time_ms: u64,
    pub registered_components: usize,
    pub recent_events: Vec<EmergencyEvent>,
}

// ============================================================================
// TESTS - Alex & Quinn critical validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock component for testing
    struct MockComponent {
        name: String,
        healthy: Arc<AtomicBool>,
    }
    
    #[async_trait]
    impl Shutdownable for MockComponent {
        fn name(&self) -> &str {
            &self.name
        }
        
        async fn cancel_all_orders(&self) -> Result<(), String> {
            Ok(())
        }
        
        async fn emergency_liquidate(&self) -> Result<(), String> {
            Ok(())
        }
        
        async fn shutdown(&self) -> Result<(), String> {
            self.healthy.store(false, Ordering::SeqCst);
            Ok(())
        }
        
        fn is_healthy(&self) -> bool {
            self.healthy.load(Ordering::SeqCst)
        }
    }
    
    #[tokio::test]
    async fn test_emergency_shutdown() {
        let coordinator = EmergencyCoordinator::new(15.0, 1000, 80.0);
        
        // Register mock components
        for i in 0..3 {
            let component = Arc::new(MockComponent {
                name: format!("Component{}", i),
                healthy: Arc::new(AtomicBool::new(true)),
            });
            coordinator.register_component(component);
        }
        
        // Trigger emergency
        coordinator.trigger_emergency(EmergencyReason::ManualTrigger).await.unwrap();
        
        // Verify state
        assert_eq!(coordinator.getstate(), EmergencyState::Halted);
        assert!(coordinator.is_emergency_active());
        
        // Check statistics
        let stats = coordinator.get_statistics();
        assert_eq!(stats.triggers_activated, 1);
        assert_eq!(stats.registered_components, 3);
    }
    
    #[tokio::test]
    async fn test_health_monitoring() {
        let coordinator = EmergencyCoordinator::new(15.0, 1000, 80.0);
        
        // Register healthy component
        let healthy = Arc::new(MockComponent {
            name: "Healthy".to_string(),
            healthy: Arc::new(AtomicBool::new(true)),
        });
        coordinator.register_component(healthy);
        
        // Register unhealthy component
        let unhealthy = Arc::new(MockComponent {
            name: "Unhealthy".to_string(),
            healthy: Arc::new(AtomicBool::new(false)),
        });
        coordinator.register_component(unhealthy);
        
        // Check health
        let health = coordinator.health_check().await;
        assert_eq!(health.healthy_components, 1);
        assert_eq!(health.total_components, 2);
        assert_eq!(health.health_percentage, 50.0);
    }
}