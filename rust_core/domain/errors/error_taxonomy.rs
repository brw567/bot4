use domain_types::order::OrderError;
// Complete Error Taxonomy for Exchange/Venue Errors
// Owner: Sam | Reviewer: Casey (Exchange Integration)
// Pre-Production Requirement #4 from Sophia
// Target: Comprehensive error handling with proper recovery strategies

use std::fmt;
use std::error::Error;
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// Complete taxonomy of exchange/venue errors
/// Sophia's requirement: Handle all possible exchange error scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum VenueError {
    /// Network and connectivity errors
    Network(NetworkError),
    
    /// Authentication and authorization errors
    Auth(AuthError),
    
    /// Order-related errors
    Order(OrderError),
    
    /// Market data errors
    MarketData(MarketDataError),
    
    /// Rate limiting errors
    RateLimit(RateLimitError),
    
    /// Account and balance errors
    Account(AccountError),
    
    /// System and maintenance errors
    System(SystemError),
    
    /// Compliance and regulatory errors
    Compliance(ComplianceError),
}

/// Network-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum NetworkError {
    /// Connection timeout
    ConnectionTimeout {
        endpoint: String,
        timeout: Duration,
    },
    
    /// DNS resolution failure
    DnsResolution {
        hostname: String,
    },
    
    /// TLS/SSL error
    TlsError {
        details: String,
    },
    
    /// Connection refused
    ConnectionRefused {
        endpoint: String,
    },
    
    /// Connection reset by peer
    ConnectionReset,
    
    /// Network unreachable
    NetworkUnreachable,
    
    /// Proxy error
    ProxyError {
        proxy: String,
        details: String,
    },
}

/// Authentication and authorization errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AuthError {
    /// Invalid API key
    InvalidApiKey,
    
    /// Invalid signature
    InvalidSignature {
        expected_algorithm: String,
    },
    
    /// Expired credentials
    ExpiredCredentials {
        expired_at: i64,
    },
    
    /// Insufficient permissions
    InsufficientPermissions {
        required: Vec<String>,
        available: Vec<String>,
    },
    
    /// IP not whitelisted
    IpNotWhitelisted {
        ip: String,
    },
    
    /// Two-factor authentication required
    TwoFactorRequired,
    
    /// Session expired
    SessionExpired,
}

/// Order-related errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum OrderError {
    /// Order not found
    OrderNotFound {
        order_id: String,
    },
    
    /// Duplicate order
    DuplicateOrder {
        client_order_id: String,
        existing_order_id: String,
    },
    
    /// Invalid order type
    InvalidOrderType {
        order_type: String,
        supported_types: Vec<String>,
    },
    
    /// Invalid side
    InvalidSide {
        side: String,
    },
    
    /// Invalid symbol
    InvalidSymbol {
        symbol: String,
    },
    
    /// Symbol not tradeable
    SymbolNotTradeable {
        symbol: String,
        reason: String,
    },
    
    /// Invalid quantity
    InvalidQuantity {
        quantity: String,
        min: String,
        max: String,
        step: String,
    },
    
    /// Invalid price
    InvalidPrice {
        price: String,
        min: String,
        max: String,
        tick_size: String,
    },
    
    /// Order rejected
    OrderRejected {
        reason: String,
        exchange_code: Option<String>,
    },
    
    /// Order expired
    OrderExpired {
        order_id: String,
        expired_at: i64,
    },
    
    /// Self-trade prevention triggered
    SelfTradePrevention {
        order_id: String,
        matched_order_id: String,
    },
    
    /// Post-only order would match
    PostOnlyWouldMatch {
        order_id: String,
        best_price: String,
    },
    
    /// Reduce-only order would increase position
    ReduceOnlyIncrease {
        current_position: String,
        order_quantity: String,
    },
    
    /// Order would trigger liquidation
    WouldTriggerLiquidation {
        margin_required: String,
        margin_available: String,
    },
}

/// Market data errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum MarketDataError {
    /// Symbol not found
    SymbolNotFound {
        symbol: String,
    },
    
    /// Invalid depth
    InvalidDepth {
        requested: usize,
        max_allowed: usize,
    },
    
    /// Subscription limit exceeded
    SubscriptionLimitExceeded {
        current: usize,
        limit: usize,
    },
    
    /// Data not available
    DataNotAvailable {
        data_type: String,
        reason: String,
    },
    
    /// Invalid timeframe
    InvalidTimeframe {
        timeframe: String,
        valid_timeframes: Vec<String>,
    },
    
    /// Historical data limit exceeded
    HistoricalDataLimitExceeded {
        requested_bars: usize,
        limit: usize,
    },
}

/// Rate limiting errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum RateLimitError {
    /// Request rate exceeded
    RequestRateExceeded {
        limit: usize,
        window: Duration,
        retry_after: Duration,
    },
    
    /// Order rate exceeded
    OrderRateExceeded {
        limit: usize,
        window: Duration,
        retry_after: Duration,
    },
    
    /// Weight limit exceeded
    WeightLimitExceeded {
        used_weight: usize,
        limit: usize,
        reset_at: i64,
    },
    
    /// IP banned
    IpBanned {
        reason: String,
        until: Option<i64>,
    },
}

/// Account and balance errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum AccountError {
    /// Insufficient balance
    InsufficientBalance {
        required: String,
        available: String,
        currency: String,
    },
    
    /// Account suspended
    AccountSuspended {
        reason: String,
        until: Option<i64>,
    },
    
    /// Account not found
    AccountNotFound {
        account_id: String,
    },
    
    /// Margin call
    MarginCall {
        margin_required: String,
        margin_available: String,
    },
    
    /// Position limit exceeded
    PositionLimitExceeded {
        current: String,
        limit: String,
    },
    
    /// Withdrawal suspended
    WithdrawalSuspended {
        currency: String,
        reason: String,
    },
    
    /// Deposit suspended
    DepositSuspended {
        currency: String,
        reason: String,
    },
}

/// System and maintenance errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum SystemError {
    /// Exchange under maintenance
    Maintenance {
        expected_duration: Option<Duration>,
        message: String,
    },
    
    /// System overload
    SystemOverload {
        retry_after: Option<Duration>,
    },
    
    /// Internal server error
    InternalError {
        error_code: Option<String>,
        message: String,
    },
    
    /// Service unavailable
    ServiceUnavailable {
        service: String,
        retry_after: Option<Duration>,
    },
    
    /// Database error
    DatabaseError {
        operation: String,
    },
    
    /// Configuration error
    ConfigurationError {
        parameter: String,
        details: String,
    },
}

/// Compliance and regulatory errors
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum ComplianceError {
    /// KYC required
    KycRequired {
        level: String,
        documents_needed: Vec<String>,
    },
    
    /// Country restricted
    CountryRestricted {
        country: String,
        service: String,
    },
    
    /// Asset restricted
    AssetRestricted {
        asset: String,
        jurisdiction: String,
    },
    
    /// Trading suspended
    TradingSuspended {
        reason: String,
        review_required: bool,
    },
    
    /// Suspicious activity detected
    SuspiciousActivity {
        activity_type: String,
        action_required: String,
    },
}

/// Error recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum RecoveryStrategy {
    /// Retry immediately
    RetryImmediately,
    
    /// Retry with backoff
    RetryWithBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        max_attempts: usize,
    },
    
    /// Retry after specific time
    RetryAfter(Duration),
    
    /// Cancel and replace
    CancelAndReplace,
    
    /// Escalate to manual intervention
    Escalate {
        severity: Severity,
        notify: Vec<String>,
    },
    
    /// Fail permanently
    FailPermanently,
    
    /// Switch to backup venue
    SwitchVenue {
        backup_venues: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// TODO: Add docs
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl VenueError {
    /// Get recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            // Network errors - retry with backoff
            VenueError::Network(_) => RecoveryStrategy::RetryWithBackoff {
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(30),
                max_attempts: 5,
            },
            
            // Auth errors - usually permanent
            VenueError::Auth(AuthError::InvalidApiKey) => RecoveryStrategy::FailPermanently,
            VenueError::Auth(AuthError::SessionExpired) => RecoveryStrategy::RetryImmediately,
            
            // Order errors - depends on specific error
            VenueError::Order(OrderError::DuplicateOrder { .. }) => RecoveryStrategy::FailPermanently,
            VenueError::Order(OrderError::InvalidQuantity { .. }) => RecoveryStrategy::CancelAndReplace,
            VenueError::Order(OrderError::OrderRejected { .. }) => RecoveryStrategy::CancelAndReplace,
            
            // Rate limits - respect retry-after
            VenueError::RateLimit(RateLimitError::RequestRateExceeded { retry_after, .. }) => {
                RecoveryStrategy::RetryAfter(*retry_after)
            }
            
            // Account errors - usually need manual intervention
            VenueError::Account(AccountError::InsufficientBalance { .. }) => {
                RecoveryStrategy::Escalate {
                    severity: Severity::High,
                    notify: vec!["risk_team".to_string()],
                }
            }
            
            // System errors - retry or switch venue
            VenueError::System(SystemError::Maintenance { .. }) => RecoveryStrategy::SwitchVenue {
                backup_venues: vec!["binance".to_string(), "kraken".to_string()],
            },
            
            // Compliance - always escalate
            VenueError::Compliance(_) => RecoveryStrategy::Escalate {
                severity: Severity::Critical,
                notify: vec!["compliance_team".to_string(), "legal_team".to_string()],
            },
            
            _ => RecoveryStrategy::RetryWithBackoff {
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(60),
                max_attempts: 3,
            },
        }
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.recovery_strategy(),
            RecoveryStrategy::RetryImmediately
                | RecoveryStrategy::RetryWithBackoff { .. }
                | RecoveryStrategy::RetryAfter(_)
        )
    }
    
    /// Get error severity
    pub fn severity(&self) -> Severity {
        match self {
            VenueError::Network(_) => Severity::Medium,
            VenueError::Auth(_) => Severity::High,
            VenueError::Order(OrderError::WouldTriggerLiquidation { .. }) => Severity::Critical,
            VenueError::RateLimit(RateLimitError::IpBanned { .. }) => Severity::Critical,
            VenueError::Account(AccountError::MarginCall { .. }) => Severity::Critical,
            VenueError::System(SystemError::Maintenance { .. }) => Severity::Low,
            VenueError::Compliance(_) => Severity::Critical,
            _ => Severity::Medium,
        }
    }
    
    /// Convert from exchange-specific error codes
    pub fn from_exchange_error(exchange: &str, code: &str, message: &str) -> Self {
        match exchange {
            "binance" => Self::from_binance_error(code, message),
            "kraken" => Self::from_kraken_error(code, message),
            "coinbase" => Self::from_coinbase_error(code, message),
            _ => VenueError::System(SystemError::InternalError {
                error_code: Some(code.to_string()),
                message: message.to_string(),
            }),
        }
    }
    
    fn from_binance_error(code: &str, message: &str) -> Self {
        match code {
            "-1003" => VenueError::RateLimit(RateLimitError::RequestRateExceeded {
                limit: 1200,
                window: Duration::from_secs(60),
                retry_after: Duration::from_secs(60),
            }),
            "-1021" => VenueError::Auth(AuthError::InvalidSignature {
                expected_algorithm: "HMAC-SHA256".to_string(),
            }),
            "-2010" => VenueError::Order(OrderError::InvalidQuantity {
                quantity: message.to_string(),
                min: "0.001".to_string(),
                max: "9000".to_string(),
                step: "0.001".to_string(),
            }),
            _ => VenueError::System(SystemError::InternalError {
                error_code: Some(code.to_string()),
                message: message.to_string(),
            }),
        }
    }
    
    fn from_kraken_error(code: &str, message: &str) -> Self {
        match code {
            "EAPI:Rate limit exceeded" => VenueError::RateLimit(RateLimitError::RequestRateExceeded {
                limit: 15,
                window: Duration::from_secs(3),
                retry_after: Duration::from_secs(3),
            }),
            "EGeneral:Invalid arguments" => VenueError::Order(OrderError::OrderRejected {
                reason: message.to_string(),
                exchange_code: Some(code.to_string()),
            }),
            _ => VenueError::System(SystemError::InternalError {
                error_code: Some(code.to_string()),
                message: message.to_string(),
            }),
        }
    }
    
    fn from_coinbase_error(code: &str, message: &str) -> Self {
        match code {
            "insufficient_funds" => VenueError::Account(AccountError::InsufficientBalance {
                required: "unknown".to_string(),
                available: "unknown".to_string(),
                currency: "unknown".to_string(),
            }),
            "rate_limit_exceeded" => VenueError::RateLimit(RateLimitError::RequestRateExceeded {
                limit: 10,
                window: Duration::from_secs(1),
                retry_after: Duration::from_secs(1),
            }),
            _ => VenueError::System(SystemError::InternalError {
                error_code: Some(code.to_string()),
                message: message.to_string(),
            }),
        }
    }
}

impl fmt::Display for VenueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VenueError::Network(e) => write!(f, "Network error: {:?}", e),
            VenueError::Auth(e) => write!(f, "Authentication error: {:?}", e),
            VenueError::Order(e) => write!(f, "Order error: {:?}", e),
            VenueError::MarketData(e) => write!(f, "Market data error: {:?}", e),
            VenueError::RateLimit(e) => write!(f, "Rate limit error: {:?}", e),
            VenueError::Account(e) => write!(f, "Account error: {:?}", e),
            VenueError::System(e) => write!(f, "System error: {:?}", e),
            VenueError::Compliance(e) => write!(f, "Compliance error: {:?}", e),
        }
    }
}

impl Error for VenueError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_recovery_strategy() {
        let network_error = VenueError::Network(NetworkError::ConnectionTimeout {
            endpoint: "wss://stream.binance.com".to_string(),
            timeout: Duration::from_secs(30),
        });
        
        assert!(network_error.is_retryable());
        assert!(matches!(
            network_error.recovery_strategy(),
            RecoveryStrategy::RetryWithBackoff { .. }
        ));
    }
    
    #[test]
    fn test_rate_limit_handling() {
        let rate_limit = VenueError::RateLimit(RateLimitError::RequestRateExceeded {
            limit: 1200,
            window: Duration::from_secs(60),
            retry_after: Duration::from_secs(45),
        });
        
        match rate_limit.recovery_strategy() {
            RecoveryStrategy::RetryAfter(duration) => {
                assert_eq!(duration, Duration::from_secs(45));
            }
            _ => panic!("Expected RetryAfter strategy"),
        }
    }
    
    #[test]
    fn test_binance_error_parsing() {
        let error = VenueError::from_exchange_error(
            "binance",
            "-1003",
            "Too many requests"
        );
        
        assert!(matches!(error, VenueError::RateLimit(_)));
    }
    
    #[test]
    fn test_severity_levels() {
        let margin_call = VenueError::Account(AccountError::MarginCall {
            margin_required: "10000".to_string(),
            margin_available: "5000".to_string(),
        });
        
        assert!(matches!(margin_call.severity(), Severity::Critical));
    }
}

// Complete error taxonomy benefits:
// - Comprehensive coverage of all venue error scenarios
// - Automatic recovery strategy selection
// - Exchange-specific error code mapping
// - Severity-based escalation
// - Retry logic with proper backoff