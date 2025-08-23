// COMPREHENSIVE INTEGRATION TEST - Verify ALL Data Flows
// Team: Full collaboration - NO SIMPLIFICATIONS!

#[cfg(test)]
mod complete_integration_tests {
    use super::*;
    use crate::decision_orchestrator_enhanced::*;
    use crate::order_book_extensions::*;
    use crate::unified_types::*;
    use rust_decimal_macros::dec;
    use std::collections::HashMap;
    
    #[tokio::test]
    async fn test_complete_data_flow() {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║     DEEP DIVE INTEGRATION TEST - ALL DATA FLOWS         ║");
        println!("╚══════════════════════════════════════════════════════════╝\n");
        
        // Initialize orchestrator
        let orchestrator = EnhancedDecisionOrchestrator::new(
            "postgresql://bot3user:bot3pass@localhost:5432/bot3trading",
            dec!(100000)  // $100k initial equity
        ).await.expect("Failed to create orchestrator");
        
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
                Order { price: Price::from_f64(49985.0).unwrap(), quantity: Quantity::from_f64(30.0).unwrap() },
            ],
            asks: vec![
                Order { price: Price::from_f64(50005.0).unwrap(), quantity: Quantity::from_f64(10.0).unwrap() },
                Order { price: Price::from_f64(50010.0).unwrap(), quantity: Quantity::from_f64(20.0).unwrap() },
                Order { price: Price::from_f64(50015.0).unwrap(), quantity: Quantity::from_f64(30.0).unwrap() },
            ],
            timestamp: 1000000,
        };
        
        // Create historical data
        let historical_data: Vec<MarketData> = (0..100).map(|i| {
            MarketData {
                timestamp: 999900 + i,
                symbol: "BTC/USDT".to_string(),
                price: Price::from_f64(50000.0 + (i as f64 * 10.0)).unwrap(),
                bid: Price::from_f64(49995.0 + (i as f64 * 10.0)).unwrap(),
                ask: Price::from_f64(50005.0 + (i as f64 * 10.0)).unwrap(),
                spread: Price::from_f64(10.0).unwrap(),
                volume: Quantity::from_f64(1000.0 + (i as f64)).unwrap(),
                high: Price::from_f64(51000.0).unwrap(),
                low: Price::from_f64(49000.0).unwrap(),
                close: Price::from_f64(50000.0).unwrap(),
                returns_24h: Percentage::from_f64(0.02).unwrap(),
            }
        }).collect();
        
        // Create sentiment data
        let sentiment = SentimentData {
            twitter_sentiment: 65.0,
            news_sentiment: 70.0,
            reddit_sentiment: 60.0,
            fear_greed_index: 55.0,
        };
        
        // TEST 1: Feature Engineering
        println!("TEST 1: Feature Engineering Pipeline");
        let features = orchestrator.engineer_all_features(
            &market_data,
            &order_book,
            &historical_data
        ).await.expect("Feature engineering failed");
        
        assert!(!features.price_features.is_empty(), "Price features empty");
        assert!(!features.volume_features.is_empty(), "Volume features empty");
        assert!(!features.microstructure_features.is_empty(), "Microstructure features empty");
        println!("✅ Feature engineering: {} total features extracted", 
                 features.price_features.len() + features.volume_features.len() + 
                 features.microstructure_features.len() + features.technical_features.len());
        
        // TEST 2: ML Prediction
        println!("\nTEST 2: ML Prediction Pipeline");
        let ml_signal = orchestrator.get_ml_prediction_with_shap(&features)
            .await.expect("ML prediction failed");
        
        assert!(ml_signal.raw_confidence >= 0.0 && ml_signal.raw_confidence <= 1.0);
        assert!(ml_signal.calibrated_confidence >= 0.0 && ml_signal.calibrated_confidence <= 1.0);
        assert!(!ml_signal.shap_values.is_empty(), "SHAP values empty");
        println!("✅ ML prediction: {:?} (conf: {:.2}%, calibrated: {:.2}%)",
                 ml_signal.action, ml_signal.raw_confidence * 100.0,
                 ml_signal.calibrated_confidence * 100.0);
        
        // TEST 3: TA Analysis
        println!("\nTEST 3: Technical Analysis Pipeline");
        let ta_signal = orchestrator.get_advanced_ta_signal(&market_data, &historical_data)
            .await.expect("TA analysis failed");
        
        assert!(ta_signal.confidence >= 0.0 && ta_signal.confidence <= 1.0);
        assert!(!ta_signal.features.is_empty(), "TA features empty");
        println!("✅ TA analysis: {:?} (conf: {:.2}%, {} indicators)",
                 ta_signal.action, ta_signal.confidence * 100.0,
                 ta_signal.features[2] as usize);
        
        // TEST 4: Regime Detection
        println!("\nTEST 4: Regime Detection with HMM");
        let regime = orchestrator.detect_regime_with_hmm(&features)
            .await.expect("Regime detection failed");
        println!("✅ Regime detected: {:?}", regime);
        
        // TEST 5: VPIN Toxicity
        println!("\nTEST 5: VPIN Flow Toxicity");
        let vpin = orchestrator.calculate_vpin_toxicity(&order_book, &historical_data)
            .await.expect("VPIN calculation failed");
        
        assert!(vpin >= 0.0 && vpin <= 1.0);
        println!("✅ VPIN toxicity: {:.3}", vpin);
        
        // TEST 6: Systemic Risk Analysis
        println!("\nTEST 6: Systemic Risk Analysis");
        let (tail_risk, contagion_risk) = orchestrator.analyze_systemic_risks(&market_data)
            .await.expect("Risk analysis failed");
        
        assert!(tail_risk >= 0.0 && tail_risk <= 1.0);
        assert!(contagion_risk >= 0.0 && contagion_risk <= 1.0);
        println!("✅ Tail risk: {:.3}, Contagion: {:.3}", tail_risk, contagion_risk);
        
        // TEST 7: Ensemble Signal
        println!("\nTEST 7: Ensemble Signal Aggregation");
        let regime_signal = orchestrator.get_regime_adjusted_signal(regime)
            .await.expect("Regime signal failed");
        let sentiment_signal = orchestrator.analyze_sentiment_with_nlp(&sentiment)
            .await.expect("Sentiment analysis failed");
        
        let ensemble = orchestrator.create_ensemble_signal(
            ml_signal.clone(),
            ta_signal.clone(),
            regime_signal,
            Some(sentiment_signal),
            vpin,
            tail_risk,
            contagion_risk,
        ).await.expect("Ensemble failed");
        
        assert!(ensemble.confidence >= 0.0 && ensemble.confidence <= 1.0);
        println!("✅ Ensemble signal: {:?} (conf: {:.2}%)",
                 ensemble.action, ensemble.confidence * 100.0);
        
        // TEST 8: Kelly Sizing
        println!("\nTEST 8: Kelly Sizing with Risk Adjustment");
        let kelly_size = orchestrator.calculate_advanced_kelly_size(
            &ensemble,
            &market_data,
            regime,
            tail_risk,
        ).await.expect("Kelly sizing failed");
        
        assert!(kelly_size >= 0.0 && kelly_size <= 0.05);  // Max 5% position
        println!("✅ Kelly size: {:.4}% of capital", kelly_size * 100.0);
        
        // TEST 9: Risk Clamps
        println!("\nTEST 9: 8-Layer Risk Clamps");
        let clamped = orchestrator.apply_comprehensive_risk_clamps(
            ensemble.clone(),
            kelly_size,
            &market_data,
            vpin,
        ).await.expect("Risk clamps failed");
        
        assert!(clamped.size <= kelly_size);  // Should only reduce, not increase
        println!("✅ Risk-clamped size: {:.4}%", clamped.size * 100.0);
        
        // TEST 10: Monte Carlo Validation
        println!("\nTEST 10: Monte Carlo Validation");
        let mc_result = orchestrator.validate_with_monte_carlo(
            &clamped,
            &market_data,
            &historical_data,
        ).await.expect("Monte Carlo failed");
        
        assert!(mc_result.win_rate >= 0.0 && mc_result.win_rate <= 1.0);
        println!("✅ Monte Carlo: {:.1}% win rate, VaR95: {:.2}%",
                 mc_result.win_rate * 100.0, mc_result.var_95 * 100.0);
        
        // TEST 11: Profit Extraction
        println!("\nTEST 11: Profit Extraction Optimization");
        let optimized = orchestrator.optimize_for_profit_extraction(
            clamped.clone(),
            &market_data,
            &order_book,
        ).await.expect("Profit extraction failed");
        
        println!("✅ Profit-optimized: size adjusted from {:.4}% to {:.4}%",
                 clamped.size * 100.0, optimized.size * 100.0);
        
        // TEST 12: Execution Algorithm
        println!("\nTEST 12: Optimal Execution Selection");
        let exec_algo = orchestrator.select_optimal_execution(
            &optimized,
            &order_book,
            vpin,
        ).await.expect("Execution selection failed");
        
        println!("✅ Execution algorithm: {:?}", exec_algo);
        
        // TEST 13: Auto-Tuning
        println!("\nTEST 13: Hyperparameter Auto-Tuning");
        orchestrator.auto_tune_parameters(&ensemble, &market_data)
            .await.expect("Auto-tuning failed");
        println!("✅ Parameters auto-tuned successfully");
        
        // TEST 14: Full Decision
        println!("\nTEST 14: Complete Trading Decision");
        let final_signal = orchestrator.make_enhanced_trading_decision(
            &market_data,
            &order_book,
            Some(&sentiment),
            &historical_data,
        ).await.expect("Full decision failed");
        
        assert!(!final_signal.symbol.is_empty());
        assert!(final_signal.confidence.to_f64() >= 0.0);
        assert!(final_signal.size.to_f64() >= 0.0);
        
        println!("✅ Final decision: {:?} {:.4}% @ {:.2}% confidence",
                 final_signal.action,
                 final_signal.size.to_f64() * 100.0,
                 final_signal.confidence.to_f64() * 100.0);
        
        // TEST 15: Data Flow Verification
        println!("\nTEST 15: Complete Data Flow Verification");
        
        // Verify OrderBook extensions work
        let bid_vol = order_book.total_bid_volume();
        let ask_vol = order_book.total_ask_volume();
        let imbalance = order_book.volume_imbalance();
        let spread_bps = order_book.spread_bps();
        
        assert!(bid_vol > 0.0, "Bid volume calculation failed");
        assert!(ask_vol > 0.0, "Ask volume calculation failed");
        assert!(imbalance >= -1.0 && imbalance <= 1.0, "Imbalance out of range");
        assert!(spread_bps > 0.0, "Spread calculation failed");
        
        println!("✅ OrderBook extensions: bid_vol={:.2}, ask_vol={:.2}, imbalance={:.3}, spread={:.1}bps",
                 bid_vol, ask_vol, imbalance, spread_bps);
        
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║        ALL TESTS PASSED - SYSTEM FULLY INTEGRATED       ║");
        println!("╚══════════════════════════════════════════════════════════╝");
    }
    
    #[test]
    fn test_order_book_methods() {
        // Unit test for OrderBook extensions
        let order_book = OrderBook {
            bids: vec![
                Order { 
                    price: Price::from_f64(100.0).unwrap(), 
                    quantity: Quantity::from_f64(10.0).unwrap() 
                },
                Order { 
                    price: Price::from_f64(99.0).unwrap(), 
                    quantity: Quantity::from_f64(20.0).unwrap() 
                },
            ],
            asks: vec![
                Order { 
                    price: Price::from_f64(101.0).unwrap(), 
                    quantity: Quantity::from_f64(15.0).unwrap() 
                },
                Order { 
                    price: Price::from_f64(102.0).unwrap(), 
                    quantity: Quantity::from_f64(25.0).unwrap() 
                },
            ],
            timestamp: 1000000,
        };
        
        assert_eq!(order_book.total_bid_volume(), 30.0);
        assert_eq!(order_book.total_ask_volume(), 40.0);
        assert_eq!(order_book.mid_price(), 100.5);
        assert_eq!(order_book.bid_ask_spread(), 1.0);
        
        let imbalance = order_book.volume_imbalance();
        assert!(imbalance < 0.0);  // More asks than bids
        
        println!("✅ OrderBook unit tests passed");
    }
}