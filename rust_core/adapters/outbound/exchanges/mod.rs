// Exchange Adapters Module
// Team: Casey (Lead) + Full Team
// CRITICAL: Use REAL adapters in production, simulators for testing only!

pub mod exchange_adapter_trait;
pub mod exchange_simulator;
pub mod binance_real;  // REAL Binance implementation - NO MOCKS!
pub mod bounded_idempotency;

// Re-exports
pub use exchange_adapter_trait::{ExchangeAdapter, ExchangeConfig, ExchangeHealth, OrderStatus};
pub use exchange_simulator::{ExchangeSimulator, SimulationMode};
pub use binance_real::BinanceRealAdapter;
pub use bounded_idempotency::{BoundedIdempotencyCache, IdempotencyEntry};

use anyhow::Result;
use std::sync::Arc;

/// Factory for creating exchange adapters
/// Casey: "Always use REAL adapters in production!"
/// TODO: Add docs
pub struct ExchangeFactory;

impl ExchangeFactory {
    /// Create appropriate exchange adapter based on environment
    pub async fn create(
        exchange: &str,
        testnet: bool,
        api_key: Option<String>,
        api_secret: Option<String>,
    ) -> Result<Arc<dyn ExchangeAdapter>> {
        match exchange.to_lowercase().as_str() {
            "binance" => {
                if testnet || api_key.is_none() {
                    // Use simulator for testing
                    tracing::warn!("Using exchange simulator - NOT for production!");
                    Ok(Arc::new(ExchangeSimulator::new(
                        ExchangeConfig {
                            exchange_type: "binance".to_string(),
                            api_url: "https://testnet.binance.vision".to_string(),
                            ws_url: "wss://testnet.binance.vision".to_string(),
                            testnet: true,
                            rate_limit: 1200,
                        },
                        SimulationMode::Realistic,
                    )))
                } else {
                    // Use REAL adapter for production
                    tracing::info!("Using REAL Binance adapter - production mode");
                    let config = ExchangeConfig {
                        exchange_type: "binance".to_string(),
                        api_url: "https://api.binance.com".to_string(),
                        ws_url: "wss://stream.binance.com:9443".to_string(),
                        testnet: false,
                        rate_limit: 1200,
                    };
                    
                    Ok(Arc::new(
                        BinanceRealAdapter::new(
                            config,
                            api_key.unwrap(),
                            api_secret.unwrap_or_default(),
                        ).await?
                    ))
                }
            }
            _ => {
                anyhow::bail!("Unsupported exchange: {}", exchange)
            }
        }
    }
}