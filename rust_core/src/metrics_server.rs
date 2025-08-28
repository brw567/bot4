// Bot4 Metrics Server - Memory Metrics Extension
// Day 2 Sprint - Memory observability
// Owner: Jordan
// Port: 8081 (memory metrics endpoint)

use axum::{extract::State, http::StatusCode, response::IntoResponse, routing::get, Router};
use infrastructure::memory::metrics::metrics;
use std::sync::Arc;
use tokio::net::TcpListener;

/// Memory metrics server state
#[derive(Clone)]
struct MetricsState {
    start_time: std::time::Instant,
}

/// Create memory metrics router
pub fn create_memory_metrics_router() -> Router {
    let state = MetricsState {
        start_time: std::time::Instant::now(),
    };

    Router::new()
        .route("/metrics/memory", get(memory_metrics_handler))
        .route("/health", get(health_handler))
        .with_state(Arc::new(state))
}

/// Memory metrics handler - returns Prometheus format
async fn memory_metrics_handler(State(state): State<Arc<MetricsState>>) -> impl IntoResponse {
    let uptime = state.start_time.elapsed().as_secs();
    
    let mut response = String::with_capacity(4096);
    
    // Add uptime metric
    response.push_str(&format!(
        "# HELP bot4_memory_uptime_seconds Memory metrics server uptime\n\
         # TYPE bot4_memory_uptime_seconds counter\n\
         bot4_memory_uptime_seconds {}\n\n",
        uptime
    ));
    
    // Add memory metrics
    response.push_str(&metrics().export_prometheus_metrics());
    
    (StatusCode::OK, response)
}

/// Health check handler
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, "Memory metrics server healthy")
}

/// Start memory metrics server on port 8081
pub async fn start_memory_metrics_server() -> Result<(), Box<dyn std::error::Error>> {
    let app = create_memory_metrics_router();
    let listener = TcpListener::bind("0.0.0.0:8081").await?;
    
    tracing::info!("Memory metrics server listening on http://0.0.0.0:8081");
    tracing::info!("Memory metrics endpoint: http://0.0.0.0:8081/metrics/memory");
    
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use tower::ServiceExt;
    
    #[tokio::test]
    async fn test_memory_metrics_endpoint() {
        let app = create_memory_metrics_router();
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/metrics/memory")
                    .body(axum::body::Body::empty())
                    .expect("SAFETY: Add proper error handling"),
            )
            .await
            .expect("SAFETY: Add proper error handling");
        
        assert_eq!(response.status(), StatusCode::OK);
        
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("SAFETY: Add proper error handling");
        let body_str = String::from_utf8(body.to_vec()).expect("SAFETY: Add proper error handling");
        
        // Check for key metrics
        assert!(body_str.contains("bot4_memory_uptime_seconds"));
        assert!(body_str.contains("bot4_memory_allocations_total"));
        assert!(body_str.contains("bot4_pool_hit_rate_percent"));
    }
    
    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_memory_metrics_router();
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(axum::body::Body::empty())
                    .expect("SAFETY: Add proper error handling"),
            )
            .await
            .expect("SAFETY: Add proper error handling");
        
        assert_eq!(response.status(), StatusCode::OK);
    }
}