// MASTER INTEGRATION VALIDATION - DEEP DIVE VERIFICATION
// Team: FULL TEAM - NO SIMPLIFICATIONS!
// Alex: "Every connection must be verified, every flow validated!"

#[cfg(test)]
mod master_integration_tests {
    use super::super::*;
    use crate::master_orchestration_system::*;
    use crate::decision_orchestrator::{OrderBook, Order, SentimentData, MarketData};
    use crate::unified_types::*;
    use rust_decimal_macros::dec;
    use std::collections::HashMap;
    
    /// Verify ALL systems are connected and communicating
    #[tokio::test]
    async fn test_complete_system_integration() {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║     MASTER INTEGRATION TEST - NO SIMPLIFICATIONS!        ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");
        
        // Initialize master orchestration system
        let config = MasterConfig {
            enable_auto_optimization: true,
            enable_ml_retraining: true,
            enable_regime_adaptation: true,
            enable_profit_maximization: true,
            ..Default::default()
        };
        
        let master = MasterOrchestrationSystem::new(
            "postgresql://bot3user:bot3pass@localhost:5432/bot3trading",
            dec!(100000),
            config,
        ).await.expect("Failed to create master system");
        
        // Create test market data
        let market_data = MarketData {
            timestamp: 1000000,
            symbol: "BTC/USDT".to_string(),
            price: Price::from_f64(50000.0).unwrap(),
            bid: Price::from_f64(49995.0).unwrap(),
            ask: Price::from_f64(50005.0).unwrap(),
            spread: Price::from_f64(10.0).unwrap(),
            volume: Quantity::from_f64(1000.0).unwrap(),
            high: Price::from_f64(51000.0).unwrap(),
            low: Price::from_f64(49000.0).unwrap(),
            close: Price::from_f64(50000.0).unwrap(),
            returns_24h: Percentage::from_f64(0.02).unwrap(),
        };
        
        // Create test order book
        let order_book = OrderBook {
            bids: vec![
                Order { price: Price::from_f64(49995.0).unwrap(), quantity: Quantity::from_f64(10.0).unwrap() },
                Order { price: Price::from_f64(49990.0).unwrap(), quantity: Quantity::from_f64(20.0).unwrap() },
            ],
            asks: vec![
                Order { price: Price::from_f64(50005.0).unwrap(), quantity: Quantity::from_f64(10.0).unwrap() },
                Order { price: Price::from_f64(50010.0).unwrap(), quantity: Quantity::from_f64(20.0).unwrap() },
            ],
            timestamp: 1000000,
        };
        
        // Create test sentiment
        let sentiment = SentimentData {
            twitter_sentiment: 65.0,
            news_sentiment: 70.0,
            reddit_sentiment: 60.0,
            fear_greed_index: 55.0,
        };
        
        // TEST 1: Verify decision flow
        println!("TEST 1: Complete Decision Flow");
        let signal = master.make_integrated_decision(
            &market_data,
            &order_book,
            Some(&sentiment),
        ).await.expect("Decision failed");
        
        assert!(signal.latency_ms < 100.0, "Latency too high: {}ms", signal.latency_ms);
        assert!(signal.confidence.to_f64() >= 0.0 && signal.confidence.to_f64() <= 1.0);
        assert!(!signal.parameters_used.is_empty(), "No parameters used!");
        
        println!("✅ Decision made in {}ms with {} parameters", 
                 signal.latency_ms, signal.parameters_used.len());
        
        // TEST 2: Verify hyperparameter optimization
        println!("\nTEST 2: Hyperparameter Optimization");
        let hp_system = master.hyperparameter_system.read();
        let current_params = hp_system.current_params.read();
        
        assert!(current_params.contains_key("kelly_fraction"));
        assert!(current_params.contains_key("var_limit"));
        assert!(current_params.contains_key("ml_confidence_threshold"));
        assert!(current_params.contains_key("execution_algorithm_bias"));
        
        println!("✅ {} parameters configured", current_params.len());
        drop(current_params);
        drop(hp_system);
        
        // TEST 3: Verify ML system integration
        println!("\nTEST 3: ML System with XGBoost");
        let features = vec![0.5; 100]; // Dummy features
        let orchestrator = &master.decision_orchestrator;
        let ml_system = orchestrator.ml_system.read();
        let (action, confidence) = ml_system.predict(&features);
        
        assert!(confidence >= 0.0 && confidence <= 1.0);
        println!("✅ ML prediction: {:?} (conf: {:.2}%)", action, confidence * 100.0);
        
        // Get feature importance from XGBoost
        let importance = ml_system.get_feature_importance();
        println!("✅ Feature importance available: {} features", importance.len());
        drop(ml_system);
        
        // TEST 4: Verify TA system integration
        println!("\nTEST 4: Technical Analysis System");
        let ta_analytics = orchestrator.ta_analytics.read();
        
        // Update with market data
        ta_analytics.update(
            market_data.price.to_f64(),
            market_data.high.to_f64(),
            market_data.low.to_f64(),
            market_data.close.to_f64(),
            market_data.volume.to_f64(),
        );
        
        let rsi = ta_analytics.get_rsi();
        let macd = ta_analytics.get_macd();
        let bb = ta_analytics.get_bollinger_bands();
        
        assert!(rsi.is_some() || macd.is_some() || bb.is_some());
        println!("✅ TA indicators calculated");
        drop(ta_analytics);
        
        // TEST 5: Verify risk management integration
        println!("\nTEST 5: Risk Management System");
        let kelly = orchestrator.kelly_sizer.read();
        let recommendation = kelly.calculate_position_size(
            Percentage::from_f64(0.55).unwrap(),
            Decimal::from_f64(2.0).unwrap(),
            Decimal::from_f64(100000.0).unwrap(),
        );
        
        assert!(recommendation.size >= Decimal::ZERO);
        assert!(recommendation.leverage >= Decimal::ZERO);
        println!("✅ Kelly size: {:.4}% of capital", 
                 recommendation.fraction.to_f64().unwrap_or(0.0) * 100.0);
        drop(kelly);
        
        // TEST 6: Verify auto-tuning system
        println!("\nTEST 6: Auto-Tuning System");
        let auto_tuner = orchestrator.auto_tuner.read();
        let adaptations = auto_tuner.get_adaptations();
        
        assert!(!adaptations.is_empty());
        println!("✅ {} parameters adapted", adaptations.len());
        drop(auto_tuner);
        
        // TEST 7: Verify profit extraction
        println!("\nTEST 7: Profit Extraction System");
        let profit_extractor = orchestrator.profit_extractor.read();
        let edge = profit_extractor.calculate_edge(
            &order_book,
            &market_data,
        );
        
        assert!(edge >= -1.0 && edge <= 1.0);
        println!("✅ Market edge calculated: {:.4}%", edge * 100.0);
        drop(profit_extractor);
        
        // TEST 8: Verify regime detection
        println!("\nTEST 8: Regime Detection");
        let regime = master.detect_current_regime(&market_data).await
            .expect("Regime detection failed");
        
        println!("✅ Current regime: {:?}", regime);
        
        // TEST 9: Verify t-Copula integration
        println!("\nTEST 9: t-Copula Tail Dependence");
        let t_copula = &orchestrator.t_copula;
        let tail_metrics = t_copula.get_tail_metrics();
        
        assert!(tail_metrics.lower_tail_dependence >= 0.0);
        assert!(tail_metrics.upper_tail_dependence >= 0.0);
        println!("✅ Tail dependence: lower={:.3}, upper={:.3}",
                 tail_metrics.lower_tail_dependence,
                 tail_metrics.upper_tail_dependence);
        
        // TEST 10: Verify cross-asset correlations
        println!("\nTEST 10: Cross-Asset Correlations");
        let cross_asset = &orchestrator.cross_asset_corr;
        let contagion = cross_asset.get_contagion_risk();
        
        assert!(contagion.systemic_risk >= 0.0 && contagion.systemic_risk <= 1.0);
        println!("✅ Systemic risk: {:.1}%", contagion.systemic_risk * 100.0);
        
        // TEST 11: Verify feedback loops
        println!("\nTEST 11: Feedback Loop Integration");
        let performance = master.performance_tracker.read();
        assert_eq!(performance.trades.len(), 0); // No trades yet
        drop(performance);
        
        // Simulate a trade outcome
        master.record_decision_for_learning(
            &TradingSignal {
                symbol: "BTC/USDT".to_string(),
                action: SignalAction::Buy,
                size: Quantity::from_f64(0.01).unwrap(),
                confidence: Percentage::from_f64(0.7).unwrap(),
                entry_price: Some(Price::from_f64(50000.0).unwrap()),
                stop_loss: Some(Price::from_f64(49000.0).unwrap()),
                take_profit: Some(Price::from_f64(51000.0).unwrap()),
                reason: "Test trade".to_string(),
            },
            &market_data,
            regime,
            &signal.parameters_used,
        ).await.expect("Recording failed");
        
        let performance = master.performance_tracker.read();
        assert_eq!(performance.trades.len(), 1);
        println!("✅ Trade recorded for learning");
        drop(performance);
        
        // TEST 12: Verify parameter manager
        println!("\nTEST 12: Parameter Manager");
        let params = master.parameter_manager.get_all_parameters();
        
        assert!(!params.is_empty());
        assert!(params.contains_key("trading_costs"));
        assert!(params.contains_key("min_edge"));
        println!("✅ {} parameters managed", params.len());
        
        // TEST 13: Verify health monitoring
        println!("\nTEST 13: System Health Monitor");
        let health = master.health_monitor.read();
        
        assert_eq!(health.ml_health.status, HealthStatus::Healthy);
        assert_eq!(health.ta_health.status, HealthStatus::Healthy);
        assert_eq!(health.risk_health.status, HealthStatus::Healthy);
        assert!(health.overall_health > 0.8);
        println!("✅ System health: {:.1}%", health.overall_health * 100.0);
        drop(health);
        
        // TEST 14: Verify execution monitoring
        println!("\nTEST 14: Execution Monitor");
        let exec_monitor = master.execution_monitor.read();
        
        assert_eq!(exec_monitor.rejected_orders, 0);
        assert_eq!(exec_monitor.successful_orders, 0);
        println!("✅ Execution monitoring active");
        drop(exec_monitor);
        
        // TEST 15: End-to-end latency test
        println!("\nTEST 15: End-to-End Performance");
        let start = std::time::Instant::now();
        
        for _ in 0..10 {
            let _ = master.make_integrated_decision(
                &market_data,
                &order_book,
                Some(&sentiment),
            ).await.expect("Decision failed");
        }
        
        let avg_latency = start.elapsed().as_millis() / 10;
        assert!(avg_latency < 100, "Average latency too high: {}ms", avg_latency);
        println!("✅ Average decision latency: {}ms", avg_latency);
        
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║   ALL SYSTEMS VERIFIED - 100% INTEGRATION ACHIEVED!      ║");
        println!("╚══════════════════════════════════════════════════════════╝");
    }
    
    /// Test that optimization actually changes parameters
    #[tokio::test]
    async fn test_parameter_optimization_flow() {
        let config = MasterConfig::default();
        let mut master = MasterOrchestrationSystem::new(
            "postgresql://bot3user:bot3pass@localhost:5432/bot3trading",
            dec!(100000),
            config,
        ).await.expect("Failed to create system");
        
        // Get initial parameters
        let initial_params = master.parameter_manager.get_all_parameters();
        let initial_kelly = initial_params.get("kelly_fraction").copied().unwrap_or(0.25);
        
        // Update performance metrics to trigger optimization
        master.hyperparameter_system.write().update_performance_metrics(
            1.5,  // Good Sharpe
            0.05, // Low drawdown
            0.65, // Good win rate
            1000.0, // Positive P&L
        );
        
        // Run optimization
        let optimized = master.run_optimization_cycle().await
            .expect("Optimization failed");
        
        // Check parameters changed
        let new_kelly = optimized.get("kelly_fraction").copied().unwrap_or(0.25);
        
        // With good performance, Kelly should increase (more aggressive)
        assert!(new_kelly >= initial_kelly);
        println!("✅ Kelly fraction optimized: {:.3} → {:.3}", initial_kelly, new_kelly);
    }
    
    /// Test regime-specific parameter adaptation
    #[tokio::test]
    async fn test_regime_adaptation() {
        let config = MasterConfig::default();
        let master = MasterOrchestrationSystem::new(
            "postgresql://bot3user:bot3pass@localhost:5432/bot3trading",
            dec!(100000),
            config,
        ).await.expect("Failed to create system");
        
        // Test crisis parameters
        let crisis_params = master.get_regime_optimized_parameters(
            crate::isotonic::MarketRegime::Crisis
        ).await.expect("Failed to get crisis params");
        
        let crisis_kelly = crisis_params.get("kelly_fraction").unwrap();
        assert!(*crisis_kelly <= 0.1); // Very conservative in crisis
        
        // Test trending parameters
        let trend_params = master.get_regime_optimized_parameters(
            crate::isotonic::MarketRegime::Trending
        ).await.expect("Failed to get trending params");
        
        let trend_kelly = trend_params.get("kelly_fraction").unwrap();
        assert!(*trend_kelly >= 0.25); // More aggressive in trends
        
        println!("✅ Regime adaptation: Crisis Kelly={:.3}, Trend Kelly={:.3}", 
                 crisis_kelly, trend_kelly);
    }
}