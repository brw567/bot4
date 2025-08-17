// Bot4 Main Application
// Day 1 Sprint - Observability Integration
// Owner: Alex
// Exit Gate: Metrics accessible, dashboards populated

use anyhow::Result;
use tracing_subscriber;

mod observability;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("bot4=debug,tower_http=debug")
        .init();
    
    tracing::info!("Bot4 Trading Platform Starting...");
    tracing::info!("Day 1 Sprint - Observability Stack");
    
    // Initialize metrics
    observability::metrics::init_metrics();
    
    // Generate some sample metrics for testing
    generate_sample_metrics();
    
    // Start all metrics servers (ports 8080-8084)
    observability::server::start_all_metrics_servers().await;
    
    // Keep the main thread alive
    tokio::signal::ctrl_c().await?;
    tracing::info!("Shutting down...");
    
    Ok(())
}

fn generate_sample_metrics() {
    use observability::metrics::*;
    
    // Simulate some decision latencies
    for i in 0..10 {
        let timer = Timer::new();
        // Simulate work
        std::thread::sleep(std::time::Duration::from_micros(i % 3));
        DECISION_LATENCY
            .with_label_values(&["signal_generator", "momentum"])
            .observe(timer.elapsed_micros());
    }
    
    // Simulate risk checks
    for i in 0..5 {
        let timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_micros(i * 2));
        RISK_CHECK_LATENCY
            .with_label_values(&["position_size", "BTC/USDT"])
            .observe(timer.elapsed_micros());
    }
    
    // Set circuit breaker states
    CB_STATE.with_label_values(&["exchange_api"]).set(0); // Closed
    CB_STATE.with_label_values(&["risk_engine"]).set(0);  // Closed
    CB_STATE.with_label_values(&["order_pipeline"]).set(1); // Half-open
    
    // Set some failure rates
    CB_FAILURE_RATE.with_label_values(&["exchange_api"]).set(0.02);
    CB_FAILURE_RATE.with_label_values(&["risk_engine"]).set(0.01);
    CB_FAILURE_RATE.with_label_values(&["order_pipeline"]).set(0.35);
    
    // Simulate throughput
    for _ in 0..1000 {
        OPERATIONS_TOTAL.inc();
    }
    
    // Order metrics
    ORDERS_PROCESSED
        .with_label_values(&["success", "binance"])
        .inc_by(50);
    ORDERS_PROCESSED
        .with_label_values(&["failed", "binance"])
        .inc_by(2);
    ORDERS_RECEIVED.inc_by(52);
    
    // Memory pool metrics
    MEMORY_POOL_AVAILABLE
        .with_label_values(&["order_pool"])
        .set(8500);
    MEMORY_POOL_TOTAL
        .with_label_values(&["order_pool"])
        .set(10000);
    
    MEMORY_POOL_AVAILABLE
        .with_label_values(&["signal_pool"])
        .set(95000);
    MEMORY_POOL_TOTAL
        .with_label_values(&["signal_pool"])
        .set(100000);
    
    // Risk metrics
    MAX_DRAWDOWN_CURRENT.set(0.035); // 3.5% drawdown
    MAX_POSITION_SIZE_RATIO.set(0.018); // 1.8% position
    
    PORTFOLIO_CORRELATION
        .with_label_values(&["BTC_ETH"])
        .set(0.65);
    PORTFOLIO_CORRELATION
        .with_label_values(&["BTC_SOL"])
        .set(0.45);
    
    // Risk check results
    RISK_CHECK_PASS
        .with_label_values(&["position_size"])
        .inc_by(100);
    RISK_CHECK_FAIL
        .with_label_values(&["position_size", "exceeds_limit"])
        .inc_by(5);
    
    // Order queue
    ORDER_QUEUE_DEPTH.set(15);
    ORDERS_BY_TYPE.with_label_values(&["market"]).inc_by(30);
    ORDERS_BY_TYPE.with_label_values(&["limit"]).inc_by(20);
    ORDERS_SUCCESS.inc_by(48);
    ORDERS_FAILED.inc_by(2);
    
    // Exchange API latency
    for i in 0..5 {
        let timer = Timer::new();
        std::thread::sleep(std::time::Duration::from_millis(20 + i * 10));
        EXCHANGE_API_LATENCY
            .with_label_values(&["binance", "place_order"])
            .observe(timer.elapsed_millis());
    }
    
    // Market data freshness
    let now = chrono::Utc::now().timestamp() as f64;
    MARKET_DATA_LAST_UPDATE
        .with_label_values(&["BTC/USDT", "binance"])
        .set(now);
    MARKET_DATA_LAST_UPDATE
        .with_label_values(&["ETH/USDT", "binance"])
        .set(now - 2.0);
    
    // Order book depth
    ORDER_BOOK_DEPTH
        .with_label_values(&["BTC/USDT", "bid"])
        .set(250);
    ORDER_BOOK_DEPTH
        .with_label_values(&["BTC/USDT", "ask"])
        .set(245);
    
    tracing::info!("Sample metrics generated successfully");
}