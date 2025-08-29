//! Property-Based Testing Framework
//! Team: QualityGate
//! Research: QuickCheck, Hypothesis patterns

use proptest::prelude::*;
use risk::*;
use strategies::*;

proptest! {
    #[test]
    fn kelly_fraction_never_exceeds_cap(
        win_prob in 0.0..1.0,
        win_loss_ratio in 0.1..10.0,
    ) {
        let kelly = calculate_kelly_fraction(win_prob, win_loss_ratio);
        prop_assert!(kelly <= 0.25); // 25% cap
        prop_assert!(kelly >= 0.0);  // Never negative
    }
    
    #[test]
    fn position_sizing_preserves_capital(
        capital in 1000.0..1_000_000.0,
        kelly in 0.0..0.25,
        num_positions in 1usize..20,
    ) {
        let position_size = calculate_position_size(capital, kelly, num_positions);
        let total_allocated = position_size * num_positions as f64;
        
        prop_assert!(total_allocated <= capital);
        prop_assert!(position_size >= 0.0);
    }
    
    #[test]
    fn correlation_matrix_is_symmetric(
        size in 2usize..10,
        seed in any::<u64>(),
    ) {
        let matrix = generate_random_correlation_matrix(size, seed);
        
        for i in 0..size {
            for j in 0..size {
                prop_assert!((matrix[i][j] - matrix[j][i]).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn sharpe_ratio_calculation_stable(
        returns in prop::collection::vec(-0.1..0.1, 100..1000),
    ) {
        let sharpe = calculate_sharpe_ratio(&returns, 0.0);
        
        prop_assert!(sharpe.is_finite());
        prop_assert!(sharpe > -10.0 && sharpe < 10.0);
    }
}
