// Value Object: Timestamp Validator
// Prevents replay attacks and ensures server time synchronization
// Addresses Sophia's #4 critical feedback on timestamp validation
// Owner: Casey | Reviewer: Quinn

use anyhow::{Result, bail};
use chrono::{DateTime, Utc, Duration};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

/// Timestamp validation configuration
#[derive(Debug, Clone)]
/// TODO: Add docs
pub struct TimestampConfig {
    /// Maximum allowed clock drift between client and server (default 1000ms)
    pub max_clock_drift: Duration,
    /// Maximum receive window for requests (default 5000ms)
    pub max_recv_window: Duration,
    /// Whether to enforce strict timestamp ordering
    pub enforce_ordering: bool,
    /// Grace period for network delays (default 500ms)
    pub network_grace_period: Duration,
}

impl Default for TimestampConfig {
    fn default() -> Self {
        Self {
            max_clock_drift: Duration::milliseconds(1000),
            max_recv_window: Duration::milliseconds(5000),
            enforce_ordering: true,
            network_grace_period: Duration::milliseconds(500),
        }
    }
}

impl TimestampConfig {
    /// Create a strict configuration (for production)
    pub fn strict() -> Self {
        Self {
            max_clock_drift: Duration::milliseconds(500),
            max_recv_window: Duration::milliseconds(3000),
            enforce_ordering: true,
            network_grace_period: Duration::milliseconds(200),
        }
    }
    
    /// Create a lenient configuration (for testing)
    pub fn lenient() -> Self {
        Self {
            max_clock_drift: Duration::milliseconds(5000),
            max_recv_window: Duration::milliseconds(60000),
            enforce_ordering: false,
            network_grace_period: Duration::milliseconds(2000),
        }
    }
}

/// Timestamp validator for preventing replay attacks
/// TODO: Add docs
pub struct TimestampValidator {
    /// Current server time (updated periodically)
    server_time: Arc<AtomicI64>,
    /// Last accepted timestamp (for ordering enforcement)
    last_timestamp: Arc<AtomicI64>,
    /// Validation configuration
    config: TimestampConfig,
    /// Statistics
    stats: Arc<ValidationStats>,
}

impl TimestampValidator {
    /// Create a new timestamp validator
    pub fn new(config: TimestampConfig) -> Self {
        let now = Utc::now().timestamp_millis();
        Self {
            server_time: Arc::new(AtomicI64::new(now)),
            last_timestamp: Arc::new(AtomicI64::new(now - 1000)),
            config,
            stats: Arc::new(ValidationStats::default()),
        }
    }
    
    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(TimestampConfig::default())
    }
    
    /// Update server time (should be called periodically)
    pub fn update_server_time(&self) {
        let now = Utc::now().timestamp_millis();
        self.server_time.store(now, Ordering::SeqCst);
    }
    
    /// Get current server time
    pub fn server_time(&self) -> DateTime<Utc> {
        let millis = self.server_time.load(Ordering::SeqCst);
        DateTime::from_timestamp_millis(millis).unwrap_or_else(Utc::now)
    }
    
    /// Validate a client timestamp
    pub fn validate_timestamp(&self, client_timestamp: i64) -> Result<()> {
        let server_millis = self.server_time.load(Ordering::SeqCst);
        
        // Check if timestamp is in the future
        if client_timestamp > server_millis + self.config.network_grace_period.num_milliseconds() {
            self.stats.future_timestamps.fetch_add(1, Ordering::Relaxed);
            bail!(
                "Timestamp {} is in the future (server time: {}). Clock drift?",
                client_timestamp,
                server_millis
            );
        }
        
        // Check if timestamp is too old (outside receive window)
        let min_allowed = server_millis - self.config.max_recv_window.num_milliseconds();
        if client_timestamp < min_allowed {
            self.stats.expired_timestamps.fetch_add(1, Ordering::Relaxed);
            bail!(
                "Timestamp {} is too old (expired). Must be within {} ms",
                client_timestamp,
                self.config.max_recv_window.num_milliseconds()
            );
        }
        
        // Check clock drift
        let drift = (server_millis - client_timestamp).abs();
        if drift > self.config.max_clock_drift.num_milliseconds() {
            self.stats.clock_drift_errors.fetch_add(1, Ordering::Relaxed);
            bail!(
                "Clock drift {} ms exceeds maximum {} ms. Sync your clock.",
                drift,
                self.config.max_clock_drift.num_milliseconds()
            );
        }
        
        // Check ordering (optional)
        if self.config.enforce_ordering {
            let last = self.last_timestamp.load(Ordering::SeqCst);
            if client_timestamp <= last {
                self.stats.ordering_violations.fetch_add(1, Ordering::Relaxed);
                bail!(
                    "Timestamp {} violates ordering (last: {}). Possible replay attack.",
                    client_timestamp,
                    last
                );
            }
            
            // Update last timestamp (CAS loop for thread safety)
            loop {
                let current = self.last_timestamp.load(Ordering::SeqCst);
                if client_timestamp <= current {
                    break; // Another thread updated it
                }
                if self.last_timestamp.compare_exchange_weak(
                    current,
                    client_timestamp,
                    Ordering::SeqCst,
                    Ordering::Relaxed,
                ).is_ok() {
                    break;
                }
            }
        }
        
        self.stats.valid_timestamps.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    /// Validate a request with timestamp and signature
    pub fn validate_request(
        &self,
        timestamp: i64,
        request_body: &str,
        signature: &str,
        api_secret: &str,
    ) -> Result<()> {
        // First validate timestamp
        self.validate_timestamp(timestamp)?;
        
        // Then validate signature (HMAC-SHA256)
        let expected_signature = self.calculate_signature(timestamp, request_body, api_secret);
        if signature != expected_signature {
            self.stats.signature_failures.fetch_add(1, Ordering::Relaxed);
            bail!("Invalid signature. Possible tampering or wrong API key.");
        }
        
        Ok(())
    }
    
    /// Calculate HMAC-SHA256 signature
    fn calculate_signature(&self, timestamp: i64, body: &str, secret: &str) -> String {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        let message = format!("{}{}", timestamp, body);
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        
        // Convert to hex string
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }
    
    /// Get validation statistics
    pub fn stats(&self) -> ValidationStats {
        ValidationStats {
            valid_timestamps: self.stats.valid_timestamps.load(Ordering::Relaxed),
            expired_timestamps: self.stats.expired_timestamps.load(Ordering::Relaxed),
            future_timestamps: self.stats.future_timestamps.load(Ordering::Relaxed),
            clock_drift_errors: self.stats.clock_drift_errors.load(Ordering::Relaxed),
            ordering_violations: self.stats.ordering_violations.load(Ordering::Relaxed),
            signature_failures: self.stats.signature_failures.load(Ordering::Relaxed),
        }
    }
    
    /// Reset statistics
    pub fn reset_stats(&self) {
        self.stats.valid_timestamps.store(0, Ordering::Relaxed);
        self.stats.expired_timestamps.store(0, Ordering::Relaxed);
        self.stats.future_timestamps.store(0, Ordering::Relaxed);
        self.stats.clock_drift_errors.store(0, Ordering::Relaxed);
        self.stats.ordering_violations.store(0, Ordering::Relaxed);
        self.stats.signature_failures.store(0, Ordering::Relaxed);
    }
}

/// Validation statistics
#[derive(Debug, Default)]
/// TODO: Add docs
pub struct ValidationStats {
    pub valid_timestamps: u64,
    pub expired_timestamps: u64,
    pub future_timestamps: u64,
    pub clock_drift_errors: u64,
    pub ordering_violations: u64,
    pub signature_failures: u64,
}

impl ValidationStats {
    /// Get total validations
    pub fn total(&self) -> u64 {
        self.valid_timestamps + self.expired_timestamps + self.future_timestamps
            + self.clock_drift_errors + self.ordering_violations
    }
    
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            100.0
        } else {
            (self.valid_timestamps as f64 / total as f64) * 100.0
        }
    }
    
    /// Get detailed error breakdown
    pub fn error_breakdown(&self) -> String {
        format!(
            "Expired: {}, Future: {}, Drift: {}, Ordering: {}, Signature: {}",
            self.expired_timestamps,
            self.future_timestamps,
            self.clock_drift_errors,
            self.ordering_violations,
            self.signature_failures,
        )
    }
}

/// Server time synchronization helper
/// TODO: Add docs
pub struct ServerTimeSync {
    validator: Arc<TimestampValidator>,
}

impl ServerTimeSync {
    /// Create new time sync helper
    pub fn new(validator: Arc<TimestampValidator>) -> Self {
        Self { validator }
    }
    
    /// Start periodic time updates
    pub async fn start_sync(self) {
        use tokio::time::{interval, Duration};
        
        let mut ticker = interval(Duration::from_secs(1));
        
        loop {
            ticker.tick().await;
            self.validator.update_server_time();
        }
    }
    
    /// Get time difference between client and server
    pub fn get_time_offset(&self, client_time: DateTime<Utc>) -> Duration {
        let server_time = self.validator.server_time();
        server_time - client_time
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_timestamp() {
        let validator = TimestampValidator::default();
        validator.update_server_time();
        
        let now = Utc::now().timestamp_millis();
        assert!(validator.validate_timestamp(now).is_ok());
    }
    
    #[test]
    fn test_expired_timestamp() {
        let validator = TimestampValidator::default();
        validator.update_server_time();
        
        let old = Utc::now().timestamp_millis() - 10000; // 10 seconds old
        assert!(validator.validate_timestamp(old).is_err());
    }
    
    #[test]
    fn test_future_timestamp() {
        let validator = TimestampValidator::default();
        validator.update_server_time();
        
        let future = Utc::now().timestamp_millis() + 2000; // 2 seconds in future
        let result = validator.validate_timestamp(future);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("future"));
    }
    
    #[test]
    fn test_clock_drift() {
        let config = TimestampConfig {
            max_clock_drift: Duration::milliseconds(100),
            ..Default::default()
        };
        let validator = TimestampValidator::new(config);
        validator.update_server_time();
        
        let drifted = Utc::now().timestamp_millis() - 200; // 200ms drift
        let result = validator.validate_timestamp(drifted);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("drift"));
    }
    
    #[test]
    fn test_ordering_enforcement() {
        let validator = TimestampValidator::default();
        validator.update_server_time();
        
        let t1 = Utc::now().timestamp_millis() - 1000;
        let t2 = t1 - 100; // Earlier timestamp
        
        // First should succeed
        assert!(validator.validate_timestamp(t1).is_ok());
        
        // Second should fail (violates ordering)
        let result = validator.validate_timestamp(t2);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ordering"));
    }
    
    #[test]
    fn test_signature_validation() {
        let validator = TimestampValidator::default();
        validator.update_server_time();
        
        let timestamp = Utc::now().timestamp_millis();
        let body = r#"{"symbol":"BTC/USDT","quantity":1.0}"#;
        let secret = "my_secret_key";
        
        // Calculate correct signature
        let signature = validator.calculate_signature(timestamp, body, secret);
        
        // Should validate successfully
        assert!(validator.validate_request(timestamp, body, &signature, secret).is_ok());
        
        // Wrong signature should fail
        let wrong_sig = "invalid_signature";
        assert!(validator.validate_request(timestamp, body, wrong_sig, secret).is_err());
    }
    
    #[test]
    fn test_statistics() {
        let validator = TimestampValidator::default();
        validator.update_server_time();
        
        // Generate some validations
        let now = Utc::now().timestamp_millis();
        let _ = validator.validate_timestamp(now);
        let _ = validator.validate_timestamp(now - 10000); // Expired
        let _ = validator.validate_timestamp(now + 2000);  // Future
        
        let stats = validator.stats();
        assert_eq!(stats.valid_timestamps, 1);
        assert_eq!(stats.expired_timestamps, 1);
        assert_eq!(stats.future_timestamps, 1);
        assert_eq!(stats.total(), 3);
    }
}