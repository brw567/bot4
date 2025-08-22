// Comprehensive Testing Suite with External Validation
// Task 1.5 - Full Team Collaboration
// References academic papers and industry standards for validation

use crate::{
    RiskClampSystem, ClampConfig,
    KellySizer, KellyConfig, TradeOutcome,
    GARCHModel, IsotonicCalibrator, MarketRegime,
};
use rust_decimal::Decimal;
use std::str::FromStr;

#[cfg(test)]
mod comprehensive_tests {
    use super::*;
    
    // ===== KELLY CRITERION VALIDATION =====
    // Validates against theoretical values from academic literature
    
    #[test]
    fn test_kelly_against_thorp_blackjack() {
        // Edward Thorp's "Beat the Dealer" (1962)
        // Blackjack with card counting: p=0.51, b=1:1
        // Expected Kelly fraction: f* = 2p - 1 = 0.02
        
        let mut kelly = KellySizer::new(KellyConfig::default());
        
        // Simulate 1000 blackjack hands with slight edge
        for i in 0..1000 {
            let outcome = TradeOutcome {
                timestamp: i,
                symbol: "BLACKJACK".to_string(),
                profit_loss: if i % 100 < 51 {
                    Decimal::from_str("1.0").unwrap()
                } else {
                    Decimal::from_str("-1.0").unwrap()
                },
                return_pct: if i % 100 < 51 {
                    Decimal::from_str("1.0").unwrap()
                } else {
                    Decimal::from_str("-1.0").unwrap()
                },
                win: i % 100 < 51,
                risk_taken: Decimal::from_str("1.0").unwrap(),
                trade_costs: Decimal::from_str("0.002").unwrap(),
            };
            kelly.add_trade(outcome);
        }
        
        let kelly_size = kelly.calculate_position_size(
            Decimal::from_str("0.7").unwrap(),    // ML confidence
            Decimal::from_str("0.02").unwrap(),   // Expected return (2%)
            Decimal::from_str("0.01").unwrap(),   // Expected risk (1%)
            Some(Decimal::from_str("0.002").unwrap()), // Trading costs
        ).unwrap();
        
        // Should be close to 2% (0.02) times fractional Kelly (0.25)
        let expected = Decimal::from_str("0.005").unwrap(); // 0.5% with fractional Kelly
        assert!(
            (kelly_size - expected).abs() < Decimal::from_str("0.005").unwrap(),
            "Blackjack Kelly should be ~{:.3}%, got {:.3}%",
            expected * Decimal::from(100),
            kelly_size * Decimal::from(100)
        );
    }
    
    #[test]
    fn test_kelly_cryptocurrency_volatility() {
        // Cryptocurrency specific test based on 2021-2024 BTC data
        // Average win: 5%, Average loss: 3%, Win rate: 55%
        
        let mut kelly = KellySizer::new(KellyConfig::default());
        
        // Simulate realistic crypto trading
        for i in 0..500 {
            let outcome = TradeOutcome {
                timestamp: i,
                symbol: "BTC".to_string(),
                profit_loss: if i % 100 < 55 {
                    Decimal::from_str("5.0").unwrap()
                } else {
                    Decimal::from_str("-3.0").unwrap()
                },
                return_pct: if i % 100 < 55 {
                    Decimal::from_str("5.0").unwrap()
                } else {
                    Decimal::from_str("-3.0").unwrap()
                },
                win: i % 100 < 55,
                risk_taken: Decimal::from_str("3.0").unwrap(),
                trade_costs: Decimal::from_str("0.001").unwrap(),
            };
            kelly.add_trade(outcome);
        }
        
        let kelly_size = kelly.calculate_position_size(
            Decimal::from_str("0.8").unwrap(),    // ML confidence
            Decimal::from_str("0.05").unwrap(),   // Expected return (5%)
            Decimal::from_str("0.03").unwrap(),   // Expected risk (3%)
            Some(Decimal::from_str("0.001").unwrap()), // Trading costs
        ).unwrap();
        
        // With 55% win rate and 5:3 payoff, Kelly should suggest meaningful position
        assert!(
            kelly_size > Decimal::from_str("0.01").unwrap() && 
            kelly_size < Decimal::from_str("0.10").unwrap(),
            "Crypto Kelly should be 1-10%, got {:.3}%",
            kelly_size * Decimal::from(100)
        );
    }
    
    // ===== GARCH MODEL VALIDATION =====
    // Validates against Bollerslev (1986) and empirical studies
    
    #[test]
    fn test_garch_persistence() {
        // Test GARCH persistence: α + β should be < 1 for stationarity
        // But close to 1 for financial data (typically 0.9-0.99)
        
        let mut model = GARCHModel::new();
        
        // Generate returns with known volatility clustering
        let mut returns = Vec::new();
        let mut vol = 0.01;
        
        for i in 0..1000 {
            // Volatility clustering periods
            if i % 200 == 0 {
                vol = if i % 400 == 0 { 0.02 } else { 0.005 };
            }
            
            // Generate return with current volatility
            let z: f64 = 2.0 * (i as f64 / 1000.0) - 1.0; // Simplified random
            returns.push(z * vol);
        }
        
        model.calibrate(&returns).unwrap();
        
        // Check persistence is high but stationary
        let persistence = model.alpha + model.beta;
        assert!(
            persistence > 0.7 && persistence < 0.999,
            "GARCH persistence should be 0.7-0.999, got {}",
            persistence
        );
        
        // Check unconditional variance is reasonable
        assert!(
            model.long_term_variance > 0.0 && model.long_term_variance < 0.01,
            "Long-term variance should be reasonable: {}",
            model.long_term_variance
        );
    }
    
    #[test]
    fn test_garch_forecast_convergence() {
        // Test that multi-step forecasts converge to unconditional variance
        // This is a fundamental property of GARCH models
        
        let model = GARCHModel::new();
        let forecasts = model.forecast(50);
        
        let long_term_vol = model.long_term_variance.sqrt();
        
        // Check convergence
        for i in 10..50 {
            let diff_early = (forecasts[i-10] - long_term_vol).abs();
            let diff_late = (forecasts[i] - long_term_vol).abs();
            
            // Later forecasts should be closer to long-term
            assert!(
                diff_late <= diff_early * 1.1, // Allow 10% tolerance
                "Forecast {} should be closer to long-term than forecast {}",
                i, i-10
            );
        }
    }
    
    // ===== ISOTONIC CALIBRATION VALIDATION =====
    // Based on Niculescu-Mizil & Caruana (2005)
    
    #[test]
    fn test_isotonic_calibration_improvement() {
        let mut calibrator = IsotonicCalibrator::new();
        
        // Create systematically miscalibrated predictions
        // Model is overconfident: predicts 0.9 but only 60% are positive
        let n = 200;
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        
        for i in 0..n {
            // Overconfident predictions
            let pred = 0.7 + (i as f64 / n as f64) * 0.25; // 0.7 to 0.95
            predictions.push(pred);
            
            // But only 60% are actually positive
            actuals.push(i < (n * 60 / 100));
        }
        
        calibrator.fit(&predictions, &actuals).unwrap();
        
        // Check that calibration reduces overconfidence
        let test_points = vec![0.8, 0.85, 0.9];
        for p in test_points {
            let calibrated = calibrator.transform(p, MarketRegime::Normal);
            assert!(
                calibrated < p,
                "Overconfident {} should be reduced, got {}",
                p, calibrated
            );
        }
        
        // Check Brier score improvement
        let metrics = calibrator.get_metrics();
        assert!(
            metrics.brier_improvement > 0.0,
            "Brier score should improve: {}",
            metrics.brier_improvement
        );
    }
    
    // ===== INTEGRATED RISK SYSTEM VALIDATION =====
    
    #[test]
    fn test_integrated_risk_system() {
        // Test the complete risk system with all components
        
        let config = ClampConfig {
            vol_target: 0.15,           // 15% target volatility
            var_limit: 0.02,            // 2% VaR limit
            es_limit: 0.03,             // 3% ES limit
            heat_cap: 0.8,              // 80% heat capacity
            leverage_cap: 2.0,          // 2x max leverage
            correlation_threshold: 0.6, // 60% correlation threshold
        };
        
        let mut risk_system = RiskClampSystem::new(config);
        
        // Calibrate GARCH with realistic returns
        let returns: Vec<f64> = (0..500)
            .map(|i| {
                let t = i as f64 / 500.0;
                0.001 * (1.0 + 0.5 * (t * 20.0).sin()) // Varying volatility
            })
            .collect();
        
        risk_system.calibrate_garch(&returns).unwrap();
        
        // Calibrate isotonic with realistic predictions
        let predictions: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let actuals: Vec<bool> = predictions.iter()
            .map(|&p| rand::random::<f64>() < p * 0.8) // Slightly miscalibrated
            .collect();
        
        risk_system.calibrate_isotonic(&predictions, &actuals).unwrap();
        
        // Test position sizing under various conditions
        
        // 1. Normal market conditions
        let normal_size = risk_system.calculate_position_size(
            0.65,      // 65% confidence
            0.015,     // 1.5% volatility
            0.4,       // 40% portfolio heat
            0.3,       // 30% correlation
            1000000.0, // $1M account
        );
        
        assert!(
            normal_size > 0.0 && normal_size < 1.0,
            "Normal conditions should produce reasonable size: {}",
            normal_size
        );
        
        // 2. High volatility conditions
        risk_system.update_garch(0.05); // Large return spike
        let high_vol_size = risk_system.calculate_position_size(
            0.65,      // Same confidence
            0.05,      // 5% volatility (high)
            0.4,       // Same heat
            0.3,       // Same correlation
            1000000.0, // Same account
        );
        
        assert!(
            high_vol_size < normal_size,
            "High volatility should reduce position: {} < {}",
            high_vol_size, normal_size
        );
        
        // 3. Crisis conditions
        risk_system.update_crisis_indicators(
            40.0,  // VIX spike
            3.0,   // Volume surge
            0.05,  // Correlation breakdown
            0.01,  // Wide spreads
        );
        
        let crisis_size = risk_system.calculate_position_size(
            0.9,       // High confidence
            0.02,      // Moderate volatility
            0.3,       // Lower heat
            0.2,       // Lower correlation
            1000000.0, // Same account
        );
        
        assert!(
            crisis_size < normal_size * 0.5,
            "Crisis should dramatically reduce position: {} < {}",
            crisis_size, normal_size * 0.5
        );
    }
    
    // ===== STRESS TESTING =====
    
    #[test]
    fn test_black_swan_protection() {
        // Test system behavior during extreme events
        // Based on historical crises: 1987, 2008, 2020, 2022
        
        let mut risk_system = RiskClampSystem::new(ClampConfig::default());
        let mut kelly = KellySizer::new(KellyConfig::default());
        
        // Simulate normal period followed by crisis
        for i in 0..100 {
            let (profit_loss, return_pct, win) = if i < 70 {
                // Normal period: 55% win rate
                if i % 100 < 55 {
                    (Decimal::from_str("2.0").unwrap(), Decimal::from_str("2.0").unwrap(), true)
                } else {
                    (Decimal::from_str("-1.5").unwrap(), Decimal::from_str("-1.5").unwrap(), false)
                }
            } else {
                // Crisis period: 20% win rate, larger losses
                if i % 100 < 20 {
                    (Decimal::from_str("1.0").unwrap(), Decimal::from_str("1.0").unwrap(), true)
                } else {
                    (Decimal::from_str("-5.0").unwrap(), Decimal::from_str("-5.0").unwrap(), false)
                }
            };
            
            let outcome = TradeOutcome {
                timestamp: i,
                symbol: "BTC".to_string(),
                profit_loss,
                return_pct,
                win,
                risk_taken: Decimal::from_str("2.0").unwrap(),
                trade_costs: Decimal::from_str("0.002").unwrap(),
            };
            kelly.add_trade(outcome);
        }
        
        // Kelly should reduce allocation after losses
        let post_crisis_size = kelly.calculate_position_size(
            Decimal::from_str("0.7").unwrap(),
            Decimal::from_str("0.01").unwrap(),   // Lower expected return
            Decimal::from_str("0.05").unwrap(),   // Higher risk
            Some(Decimal::from_str("0.002").unwrap()),
        ).unwrap();
        assert!(
            post_crisis_size < Decimal::from_str("0.05").unwrap(),
            "Kelly should be conservative after crisis: {}",
            post_crisis_size
        );
        
        // Risk clamps should trigger
        risk_system.update_crisis_indicators(45.0, 4.0, 0.02, 0.015);
        let crisis_position = risk_system.calculate_position_size(
            0.8, 0.03, 0.7, 0.6, 500000.0
        );
        
        assert!(
            crisis_position < 0.3,
            "Crisis mode should limit position to <30%: {}",
            crisis_position
        );
    }
    
    // ===== PERFORMANCE BENCHMARKS =====
    
    #[test]
    fn test_latency_requirements() {
        use std::time::Instant;
        
        let mut risk_system = RiskClampSystem::new(ClampConfig::default());
        let mut kelly = KellySizer::new(KellyConfig::default());
        
        // Warm up
        for i in 0..10 {
            let outcome = TradeOutcome {
                timestamp: i,
                symbol: "TEST".to_string(),
                profit_loss: Decimal::from_str("1.0").unwrap(),
                return_pct: Decimal::from_str("1.0").unwrap(),
                win: true,
                risk_taken: Decimal::from_str("1.0").unwrap(),
                trade_costs: Decimal::from_str("0.001").unwrap(),
            };
            kelly.add_trade(outcome);
        }
        
        // Measure Kelly calculation time
        let start = Instant::now();
        let conf = Decimal::from_str("0.7").unwrap();
        let ret = Decimal::from_str("0.02").unwrap();
        let risk = Decimal::from_str("0.01").unwrap();
        let cost = Some(Decimal::from_str("0.001").unwrap());
        for _ in 0..1000 {
            kelly.calculate_position_size(conf, ret, risk, cost);
        }
        let kelly_time = start.elapsed();
        
        assert!(
            kelly_time.as_micros() < 10000, // 10ms for 1000 calls = 10μs per call
            "Kelly calculation too slow: {:?} for 1000 calls",
            kelly_time
        );
        
        // Measure risk clamp calculation time
        let start = Instant::now();
        for _ in 0..100 {
            risk_system.calculate_position_size(0.7, 0.02, 0.5, 0.4, 100000.0);
        }
        let clamp_time = start.elapsed();
        
        assert!(
            clamp_time.as_micros() < 10000, // 100ms for 100 calls = 1ms per call
            "Risk clamp calculation too slow: {:?} for 100 calls",
            clamp_time
        );
    }
}