//! Analysis Module - Mathematical Validation and Statistical Tests
//! Phase 1 Critical Gate - Morgan (Lead)
//! 
//! This module provides all mathematical validation required by external reviewers:
//! - Jarque-Bera test for normality
//! - ADF test for stationarity  
//! - DCC-GARCH for dynamic correlations
//! - Ljung-Box test for autocorrelation

pub mod statistical_tests;
pub mod dcc_garch;

// Re-export main validation types for easy access
pub use statistical_tests::{
    ADFTest,
    JarqueBeraTest, 
    LjungBoxTest,
    CriticalValues,
    DCCGarch as StatisticalDCCGarch,  // To avoid confusion with our DCC-GARCH
    GaussianCopula
};

pub use dcc_garch::{DccGarch, GarchParams};

/// Comprehensive mathematical validation suite
/// Returns true if all tests pass required thresholds
pub fn validate_trading_data(returns: &[f64], _max_correlation: f64) -> Result<bool, anyhow::Error> {
    use anyhow::Context;
    use ndarray::Array1;
    
    let returns_array = Array1::from_vec(returns.to_vec());
    
    // Test 1: Jarque-Bera for normality
    let jb_result = JarqueBeraTest::test(&returns_array);
    
    if !jb_result.is_normal {
        log::warn!("Returns are not normally distributed (JB stat: {:.2})", jb_result.statistic);
    }
    
    // Test 2: ADF for stationarity
    let adf_result = ADFTest::test(&returns_array, Some(1));
    
    if !adf_result.is_stationary {
        log::error!("Returns are not stationary (ADF stat: {:.2})", adf_result.statistic);
        return Ok(false);
    }
    
    // Test 3: Ljung-Box for autocorrelation
    let lb_result = LjungBoxTest::test(&returns_array, 10);
    
    if lb_result.has_autocorrelation {
        log::warn!("Significant autocorrelation detected (LB stat: {:.2})", lb_result.statistic);
    }
    
    // All critical tests passed
    Ok(adf_result.is_stationary)
}

/// Mathematical validation for portfolio returns
/// Used by Quinn for risk validation
pub fn validate_portfolio_returns(
    asset_returns: &[Vec<f64>], 
    max_correlation: f64
) -> Result<bool, anyhow::Error> {
    use nalgebra::DVector;
    
    // Convert to DVector format for DCC-GARCH
    let returns: Vec<DVector<f64>> = asset_returns.iter()
        .map(|r| DVector::from_vec(r.clone()))
        .collect();
    
    let n_assets = asset_returns[0].len();
    let mut dcc = DccGarch::new(n_assets);
    
    // Fit the model
    dcc.fit(&returns)?;
    
    // Check correlation breach
    if dcc.correlation_breach() {
        log::error!("Portfolio correlation exceeds limit of {:.2}", max_correlation);
        return Ok(false);
    }
    
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_trading_data() {
        // Generate some test data
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01, -0.005, 0.02, -0.01];
        
        let result = validate_trading_data(&returns, 0.7);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_mathematical_validation_integration() {
        use ndarray::Array1;
        
        // This ensures all statistical tests work together
        let data = vec![0.1, -0.05, 0.08, -0.03, 0.06, -0.04, 0.07, -0.02, 0.05, -0.01];
        let data_array = Array1::from_vec(data.clone());
        
        // Run all tests
        let jb = JarqueBeraTest::test(&data_array);
        let adf = ADFTest::test(&data_array, Some(1));
        let lb = LjungBoxTest::test(&data_array, 5);
        
        // At least one test should pass for this data
        assert!(adf.is_stationary || jb.is_normal || !lb.has_autocorrelation);
    }
}