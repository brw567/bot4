// Bot4 Observability - Metrics Server
// Day 1 Sprint - Prometheus metrics endpoint
// Owner: Avery
// Exit Gate: Metrics accessible at :8080-8084/metrics

use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use prometheus::{Encoder, TextEncoder};
use std::net::SocketAddr;
use tokio::net::TcpListener;

use super::metrics::{init_metrics, REGISTRY};

// Main metrics endpoint on port 8080
pub async fn start_main_metrics_server() {
    init_metrics();
    
    let app = Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let listener = TcpListener::bind(addr).await.unwrap();
    
    tracing::info!("Main metrics server listening on {}", addr);
    
    axum::serve(listener, app)
        .await
        .unwrap();
}

// Circuit breaker metrics on port 8081
pub async fn start_cb_metrics_server() {
    let app = Router::new()
        .route("/metrics", get(cb_metrics_handler));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 8081));
    let listener = TcpListener::bind(addr).await.unwrap();
    
    tracing::info!("Circuit breaker metrics server listening on {}", addr);
    
    axum::serve(listener, app)
        .await
        .unwrap();
}

// Risk engine metrics on port 8082
pub async fn start_risk_metrics_server() {
    let app = Router::new()
        .route("/metrics", get(risk_metrics_handler));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 8082));
    let listener = TcpListener::bind(addr).await.unwrap();
    
    tracing::info!("Risk metrics server listening on {}", addr);
    
    axum::serve(listener, app)
        .await
        .unwrap();
}

// Order pipeline metrics on port 8083
pub async fn start_order_metrics_server() {
    let app = Router::new()
        .route("/metrics", get(order_metrics_handler));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 8083));
    let listener = TcpListener::bind(addr).await.unwrap();
    
    tracing::info!("Order metrics server listening on {}", addr);
    
    axum::serve(listener, app)
        .await
        .unwrap();
}

// MiMalloc stats endpoint on port 8084
pub async fn start_memory_metrics_server() {
    let app = Router::new()
        .route("/metrics", get(memory_metrics_handler));
    
    let addr = SocketAddr::from(([0, 0, 0, 0], 8084));
    let listener = TcpListener::bind(addr).await.unwrap();
    
    tracing::info!("Memory metrics server listening on {}", addr);
    
    axum::serve(listener, app)
        .await
        .unwrap();
}

// Handler for main metrics endpoint
async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();
    
    (StatusCode::OK, buffer)
}

// Handler for health check
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "Bot4 Trading Engine - Healthy\n")
}

// Handler for CB-specific metrics
async fn cb_metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    
    // Filter to only CB metrics
    let cb_metrics: Vec<_> = metric_families
        .into_iter()
        .filter(|mf| mf.get_name().starts_with("cb_"))
        .collect();
    
    let mut buffer = vec![];
    encoder.encode(&cb_metrics, &mut buffer).unwrap();
    
    (StatusCode::OK, buffer)
}

// Handler for risk-specific metrics
async fn risk_metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    
    // Filter to only risk metrics
    let risk_metrics: Vec<_> = metric_families
        .into_iter()
        .filter(|mf| {
            let name = mf.get_name();
            name.starts_with("risk_") || 
            name.contains("drawdown") ||
            name.contains("position_size") ||
            name.contains("correlation")
        })
        .collect();
    
    let mut buffer = vec![];
    encoder.encode(&risk_metrics, &mut buffer).unwrap();
    
    (StatusCode::OK, buffer)
}

// Handler for order-specific metrics
async fn order_metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    
    // Filter to only order metrics
    let order_metrics: Vec<_> = metric_families
        .into_iter()
        .filter(|mf| {
            let name = mf.get_name();
            name.starts_with("order_") || 
            name.contains("orders_") ||
            name.contains("exchange_api")
        })
        .collect();
    
    let mut buffer = vec![];
    encoder.encode(&order_metrics, &mut buffer).unwrap();
    
    (StatusCode::OK, buffer)
}

// Handler for memory metrics
async fn memory_metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    
    // Filter to only memory metrics
    let memory_metrics: Vec<_> = metric_families
        .into_iter()
        .filter(|mf| {
            let name = mf.get_name();
            name.contains("memory_pool") || 
            name.contains("allocation")
        })
        .collect();
    
    let mut buffer = vec![];
    encoder.encode(&memory_metrics, &mut buffer).unwrap();
    
    // Add MiMalloc stats if available
    #[cfg(feature = "mimalloc")]
    {
        
        let stats = format!(
            "# HELP mimalloc_allocated_bytes Total allocated bytes\n\
             # TYPE mimalloc_allocated_bytes gauge\n\
             mimalloc_allocated_bytes {}\n\
             # HELP mimalloc_reserved_bytes Total reserved bytes\n\
             # TYPE mimalloc_reserved_bytes gauge\n\
             mimalloc_reserved_bytes {}\n",
            0, // Would need MiMalloc API for real stats
            0
        );
        buffer.extend_from_slice(stats.as_bytes());
    }
    
    (StatusCode::OK, buffer)
}

// Spawn all metrics servers
pub async fn start_all_metrics_servers() {
    init_metrics();
    
    // Spawn each server in its own task
    tokio::spawn(start_main_metrics_server());
    tokio::spawn(start_cb_metrics_server());
    tokio::spawn(start_risk_metrics_server());
    tokio::spawn(start_order_metrics_server());
    tokio::spawn(start_memory_metrics_server());
    
    tracing::info!("All metrics servers started on ports 8080-8084");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_endpoint() {
        init_metrics();
        
        // Record some test metrics
        use crate::observability::metrics::{DECISION_LATENCY, Timer};
        let timer = Timer::new();
        DECISION_LATENCY
            .with_label_values(&["test", "test"])
            .observe(timer.elapsed_micros());
        
        // Get metrics output
        let encoder = TextEncoder::new();
        let metric_families = REGISTRY.gather();
        let mut buffer = vec![];
        encoder.encode(&metric_families, &mut buffer).unwrap();
        
        let output = String::from_utf8(buffer).unwrap();
        assert!(output.contains("decision_latency_microseconds"));
    }
}