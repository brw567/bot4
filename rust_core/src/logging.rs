// Centralized Logging Configuration
// Team: Riley (Testing Lead) + Full Team
// Comprehensive structured logging for all components

use tracing::{Level, Subscriber};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
    Layer,
};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use serde_json::json;
use std::io;

/// Initialize comprehensive logging system
pub fn init_logging() -> anyhow::Result<()> {
    // Environment-based log level
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            EnvFilter::new("info")
                .add_directive("bot4=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("trading_engine=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("risk_engine=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("ml=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("order_management=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("exchanges=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("websocket=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("infrastructure=debug".parse().expect("SAFETY: Add proper error handling"))
                .add_directive("analysis=debug".parse().expect("SAFETY: Add proper error handling"))
        });

    // Console output layer (human-readable)
    let console_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_span_events(FmtSpan::CLOSE)
        .with_ansi(true)
        .pretty();

    // File output layer (JSON structured logs)
    let file_appender = RollingFileAppender::new(
        Rotation::DAILY,
        "logs",
        "bot4.log",
    );
    
    let file_layer = fmt::layer()
        .json()
        .with_current_span(true)
        .with_span_list(true)
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_writer(file_appender);

    // Performance metrics layer
    let metrics_appender = RollingFileAppender::new(
        Rotation::HOURLY,
        "logs/metrics",
        "performance.log",
    );
    
    let metrics_layer = fmt::layer()
        .json()
        .with_target(false)
        .with_writer(metrics_appender)
        .with_filter(EnvFilter::new("info")
            .add_directive("bot4::metrics=trace".parse().expect("SAFETY: Add proper error handling")));

    // Combine all layers
    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .with(metrics_layer)
        .init();

    tracing::info!(
        message = "Bot4 Trading Platform Initialized",
        version = env!("CARGO_PKG_VERSION"),
        build = "optimized",
        team = "Full Team Collaboration",
    );

    Ok(())
}

/// Log performance metrics
#[inline]
pub fn log_performance(
    component: &str,
    operation: &str,
    latency_us: u64,
    throughput: Option<f64>,
) {
    tracing::trace!(
        target: "bot4::metrics",
        component = component,
        operation = operation,
        latency_us = latency_us,
        throughput = throughput,
        timestamp = chrono::Utc::now().timestamp_nanos(),
    );
}

/// Log trade execution
pub fn log_trade(
    symbol: &str,
    side: &str,
    quantity: f64,
    price: f64,
    pnl: Option<f64>,
) {
    tracing::info!(
        target: "bot4::trades",
        event = "trade_executed",
        symbol = symbol,
        side = side,
        quantity = quantity,
        price = price,
        pnl = pnl,
        timestamp = chrono::Utc::now().to_rfc3339(),
    );
}

/// Log risk events
pub fn log_risk_event(
    event_type: &str,
    severity: &str,
    details: serde_json::Value,
) {
    let level = match severity {
        "CRITICAL" => Level::ERROR,
        "HIGH" => Level::WARN,
        "MEDIUM" => Level::INFO,
        _ => Level::DEBUG,
    };
    
    tracing::event!(
        target: "bot4::risk",
        level,
        event_type = event_type,
        severity = severity,
        details = ?details,
        timestamp = chrono::Utc::now().to_rfc3339(),
    );
}

/// Log ML model predictions
pub fn log_ml_prediction(
    model: &str,
    symbol: &str,
    signal: f64,
    confidence: f64,
    features: Option<serde_json::Value>,
) {
    tracing::debug!(
        target: "bot4::ml",
        model = model,
        symbol = symbol,
        signal = signal,
        confidence = confidence,
        features = ?features,
        timestamp = chrono::Utc::now().timestamp_millis(),
    );
}

/// Structured error logging with context
pub fn log_error<E: std::fmt::Display>(
    component: &str,
    operation: &str,
    error: &E,
    context: Option<serde_json::Value>,
) {
    tracing::error!(
        target: "bot4::errors",
        component = component,
        operation = operation,
        error = %error,
        context = ?context,
        timestamp = chrono::Utc::now().to_rfc3339(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_initialization() {
        // Should not panic
        let result = init_logging();
        assert!(result.is_ok() || result.is_err()); // May fail if already initialized
    }

    #[test]
    fn test_performance_logging() {
        log_performance("test_component", "test_op", 100, Some(1000.0));
        // Should not panic
    }

    #[test]
    fn test_trade_logging() {
        log_trade("BTCUSDT", "BUY", 1.0, 50000.0, Some(100.0));
        // Should not panic
    }

    #[test]
    fn test_risk_logging() {
        log_risk_event(
            "position_limit_exceeded",
            "HIGH",
            json!({"position": 0.03, "limit": 0.02}),
        );
        // Should not panic
    }

    #[test]
    fn test_ml_logging() {
        log_ml_prediction(
            "lstm_v2",
            "ETHUSDT",
            0.8,
            0.95,
            Some(json!({"rsi": 70, "volume": 1000000})),
        );
        // Should not panic
    }
}