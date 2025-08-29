//! End-to-End Integration Tests
//! Team: IntegrationValidator + QualityGate
//! Coverage Target: 100%

use bot4_main::*;
use tokio;

#[tokio::test]
async fn test_full_trading_cycle() {
    // Initialize system
    let config = load_config("test_config.toml").unwrap();
    let mut system = TradingSystem::new(config).await.unwrap();
    
    // Start all components
    system.start().await.unwrap();
    
    // Simulate market data
    let tick = create_test_tick();
    system.process_tick(tick).await.unwrap();
    
    // Verify decision was made
    assert!(system.last_decision_latency_us() < 100);
    
    // Verify risk checks
    assert!(system.position_within_limits());
    
    // Shutdown gracefully
    system.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_multi_exchange_monitoring() {
    let exchanges = vec!["binance", "coinbase", "kraken", "okx", "bybit"];
    let mut connectors = Vec::new();
    
    for exchange in exchanges {
        let connector = ExchangeConnector::new(exchange).await.unwrap();
        connectors.push(connector);
    }
    
    // Verify all connected
    for connector in &connectors {
        assert!(connector.is_connected());
        assert!(connector.latency_ms() < 10.0);
    }
}

#[tokio::test]
async fn test_emergency_shutdown() {
    let system = TradingSystem::new_test().await.unwrap();
    
    // Trigger emergency stop
    system.emergency_stop().await.unwrap();
    
    // Verify all trading halted
    assert!(system.is_halted());
    assert_eq!(system.open_positions(), 0);
    assert_eq!(system.pending_orders(), 0);
}

#[tokio::test]
async fn test_risk_circuit_breakers() {
    let mut system = TradingSystem::new_test().await.unwrap();
    
    // Simulate large loss
    system.simulate_loss(0.16); // 16% loss
    
    // Should trigger soft limit (15%)
    assert!(system.risk_state() == RiskState::SoftLimit);
    
    // Additional loss
    system.simulate_loss(0.05); // Total 21%
    
    // Should trigger hard limit (20%)
    assert!(system.risk_state() == RiskState::HardLimit);
    assert!(system.is_halted());
}
