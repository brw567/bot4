//! Quantitative Finance Comprehensive Tests
//! Team: RiskQuant + MLEngineer
//! Coverage Target: 100%
//! Research: Black-Scholes, Heston, Greeks validation

use risk::quantitative_finance::*;
use approx::assert_relative_eq;

#[cfg(test)]
mod black_scholes_tests {
    use super::*;
    
    #[test]
    fn test_black_scholes_call_option() {
        let bs = BlackScholes::new(100.0, 110.0, 0.05, 1.0, 0.2, 0.0);
        let price = bs.call_price();
        
        // Validated against industry standard calculator
        assert_relative_eq!(price, 6.04, epsilon = 0.01);
    }
    
    #[test]
    fn test_black_scholes_put_option() {
        let bs = BlackScholes::new(100.0, 90.0, 0.05, 1.0, 0.2, 0.0);
        let price = bs.put_price();
        
        // Put-call parity validation
        let call = bs.call_price();
        let parity = call - bs.spot + bs.strike * (-bs.rate * bs.time).exp();
        assert_relative_eq!(price, parity, epsilon = 0.01);
    }
    
    #[test]
    fn test_complete_greeks() {
        let bs = BlackScholes::new(100.0, 100.0, 0.05, 0.5, 0.25, 0.02);
        
        let greeks = bs.calculate_all_greeks();
        
        // Delta should be around 0.5 for ATM
        assert!(greeks.delta > 0.4 && greeks.delta < 0.6);
        
        // Gamma should be positive
        assert!(greeks.gamma > 0.0);
        
        // Vega should be positive
        assert!(greeks.vega > 0.0);
        
        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0);
        
        // Rho should be positive for calls
        assert!(greeks.rho > 0.0);
    }
    
    #[test]
    fn test_advanced_greeks() {
        let bs = BlackScholes::new(100.0, 105.0, 0.05, 0.25, 0.3, 0.0);
        
        let adv_greeks = bs.calculate_advanced_greeks();
        
        // Vanna (dDelta/dVol) tests
        assert!(adv_greeks.vanna.is_finite());
        
        // Volga (dVega/dVol) should be positive
        assert!(adv_greeks.volga > 0.0);
        
        // Charm (dDelta/dTime) 
        assert!(adv_greeks.charm.is_finite());
        
        // Veta (dVega/dTime)
        assert!(adv_greeks.veta.is_finite());
    }
}

#[cfg(test)]
mod heston_model_tests {
    use super::*;
    
    #[test]
    fn test_heston_stochastic_volatility() {
        let heston = HestonModel::new(
            100.0,  // spot
            0.05,   // rate
            0.04,   // initial variance
            2.0,    // kappa (mean reversion)
            0.04,   // theta (long-term variance)
            0.3,    // sigma (vol of vol)
            -0.5    // rho (correlation)
        );
        
        let price = heston.call_price(110.0, 1.0);
        
        // Should differ from Black-Scholes due to stochastic vol
        let bs = BlackScholes::new(100.0, 110.0, 0.05, 1.0, 0.2, 0.0);
        let bs_price = bs.call_price();
        
        assert!((price - bs_price).abs() > 0.1);
        assert!(price > 0.0 && price < 100.0);
    }
    
    #[test]
    fn test_heston_monte_carlo_convergence() {
        let heston = HestonModel::new(100.0, 0.05, 0.04, 2.0, 0.04, 0.3, -0.5);
        
        let price_1k = heston.monte_carlo_price(100.0, 0.5, 1000);
        let price_10k = heston.monte_carlo_price(100.0, 0.5, 10000);
        let price_100k = heston.monte_carlo_price(100.0, 0.5, 100000);
        
        // Should converge as paths increase
        let diff_1 = (price_10k - price_1k).abs();
        let diff_2 = (price_100k - price_10k).abs();
        
        assert!(diff_2 < diff_1);
    }
}

#[cfg(test)]
mod local_volatility_tests {
    use super::*;
    
    #[test]
    fn test_dupire_local_volatility() {
        let dupire = DupireModel::new(100.0, 0.05);
        
        // Build implied volatility surface
        let mut iv_surface = ImpliedVolSurface::new();
        iv_surface.add_point(90.0, 0.25, 0.25);
        iv_surface.add_point(100.0, 0.25, 0.20);
        iv_surface.add_point(110.0, 0.25, 0.18);
        
        let local_vol = dupire.calculate_local_volatility(&iv_surface, 100.0, 0.25);
        
        assert!(local_vol > 0.0 && local_vol < 1.0);
    }
}

#[cfg(test)]
mod jump_diffusion_tests {
    use super::*;
    
    #[test]
    fn test_merton_jump_diffusion() {
        let merton = MertonJumpDiffusion::new(
            100.0,  // spot
            0.05,   // rate
            0.2,    // volatility
            0.1,    // jump intensity
            -0.1,   // mean jump size
            0.15    // jump volatility
        );
        
        let price = merton.call_price(110.0, 1.0);
        
        // Should account for jump risk
        assert!(price > 0.0);
        
        // Compare with Black-Scholes (no jumps)
        let bs = BlackScholes::new(100.0, 110.0, 0.05, 1.0, 0.2, 0.0);
        let bs_price = bs.call_price();
        
        // Jump risk should increase option value
        assert!(price > bs_price);
    }
}

#[cfg(test)]
mod simd_performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_simd_greeks_performance() {
        let spots = vec![100.0; 10000];
        let strikes = vec![105.0; 10000];
        
        // Non-SIMD calculation
        let start = Instant::now();
        for i in 0..10000 {
            let bs = BlackScholes::new(spots[i], strikes[i], 0.05, 0.5, 0.25, 0.0);
            let _ = bs.calculate_all_greeks();
        }
        let scalar_time = start.elapsed();
        
        // SIMD calculation
        let start = Instant::now();
        let _ = calculate_greeks_simd(&spots, &strikes, 0.05, 0.5, 0.25);
        let simd_time = start.elapsed();
        
        // SIMD should be at least 4x faster
        assert!(scalar_time.as_nanos() / simd_time.as_nanos() > 4);
    }
}
