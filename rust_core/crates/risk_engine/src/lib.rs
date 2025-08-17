// Risk Engine for Bot4 Trading Platform
// Phase: 1.5 - Risk Management Foundation
// Owner: Quinn (Risk Manager with VETO power)
// Performance target: <10μs for pre-trade checks

pub mod checks;
pub mod limits;
pub mod correlation;
pub mod correlation_simd;  // SIMD-optimized version (3x speedup per Nexus)
pub mod correlation_portable;  // Sophia Fix #6: Runtime SIMD detection with scalar fallback
pub mod monitor;
pub mod emergency;

pub use checks::{RiskChecker, PreTradeCheck, RiskCheckResult};
pub use limits::{RiskLimits, PositionLimits, LossLimits};
pub use correlation::{CorrelationAnalyzer, CorrelationMatrix};
pub use monitor::{RiskMonitor, RiskMetrics, DrawdownTracker};
pub use emergency::{EmergencyStop, KillSwitch, TripCondition};