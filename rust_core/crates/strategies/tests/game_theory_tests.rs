//! Game Theory Comprehensive Tests
//! Team: MLEngineer + RiskQuant
//! Coverage Target: 100%

use strategies::game_theory_router::*;
use proptest::prelude::*;

#[cfg(test)]
mod nash_equilibrium_tests {
    use super::*;
    
    #[test]
    fn test_nash_equilibrium_convergence() {
        let mut solver = NashEquilibriumSolver::new(3);
        
        // Test payoff matrix for prisoner's dilemma
        let payoff_matrix = vec![
            vec![(-1.0, -1.0), (-3.0, 0.0)],
            vec![(0.0, -3.0), (-2.0, -2.0)],
        ];
        
        let equilibrium = solver.find_equilibrium(&payoff_matrix, 1000);
        
        // Nash equilibrium should be (Defect, Defect)
        assert!(equilibrium.player1_strategy[1] > 0.9);
        assert!(equilibrium.player2_strategy[1] > 0.9);
    }
    
    #[test]
    fn test_mixed_strategy_rock_paper_scissors() {
        let mut solver = NashEquilibriumSolver::new(3);
        
        // Symmetric zero-sum game
        let payoff_matrix = vec![
            vec![(0.0, 0.0), (-1.0, 1.0), (1.0, -1.0)],
            vec![(1.0, -1.0), (0.0, 0.0), (-1.0, 1.0)],
            vec![(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)],
        ];
        
        let equilibrium = solver.find_equilibrium(&payoff_matrix, 10000);
        
        // Should converge to 1/3, 1/3, 1/3
        for prob in &equilibrium.player1_strategy {
            assert!((prob - 0.333).abs() < 0.05);
        }
    }
    
    proptest! {
        #[test]
        fn test_nash_solver_doesnt_panic(
            size in 2usize..10,
            seed in any::<u64>()
        ) {
            let solver = NashEquilibriumSolver::new(size);
            // Should handle any matrix size without panicking
            assert!(solver.strategies.len() == size);
        }
    }
}

#[cfg(test)]
mod shapley_value_tests {
    use super::*;
    
    #[test]
    fn test_shapley_fair_allocation() {
        let allocator = ShapleyValueAllocator::new(3);
        
        // Coalition values
        let mut coalitions = std::collections::HashMap::new();
        coalitions.insert(vec![0], 100.0);
        coalitions.insert(vec![1], 150.0);
        coalitions.insert(vec![2], 200.0);
        coalitions.insert(vec![0, 1], 300.0);
        coalitions.insert(vec![0, 2], 350.0);
        coalitions.insert(vec![1, 2], 400.0);
        coalitions.insert(vec![0, 1, 2], 600.0);
        
        let values = allocator.calculate_shapley_values(&coalitions);
        
        // Sum should equal grand coalition value
        let sum: f64 = values.iter().sum();
        assert!((sum - 600.0).abs() < 0.01);
        
        // Values should be fair based on marginal contributions
        assert!(values[0] > 100.0 && values[0] < 200.0);
        assert!(values[1] > 150.0 && values[1] < 250.0);
        assert!(values[2] > 200.0 && values[2] < 300.0);
    }
    
    #[test]
    fn test_efficiency_property() {
        let allocator = ShapleyValueAllocator::new(4);
        let mut coalitions = std::collections::HashMap::new();
        
        // Random coalition values
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let grand_value = rng.gen_range(1000.0..10000.0);
        
        coalitions.insert(vec![0, 1, 2, 3], grand_value);
        
        let values = allocator.calculate_shapley_values(&coalitions);
        let sum: f64 = values.iter().sum();
        
        // Efficiency: sum of allocations equals total value
        assert!((sum - grand_value).abs() < 0.01);
    }
}

#[cfg(test)]
mod colonel_blotto_tests {
    use super::*;
    
    #[test]
    fn test_blotto_resource_allocation() {
        let strategy = ColonelBlottoStrategy::new(5, 100.0);
        
        let allocation = strategy.allocate_resources(
            &[20.0, 30.0, 25.0, 15.0, 10.0]
        );
        
        // Total allocation should equal total resources
        let sum: f64 = allocation.iter().sum();
        assert!((sum - 100.0).abs() < 0.01);
        
        // Should allocate more to valuable battlefields
        assert!(allocation[1] > allocation[4]);
    }
    
    #[test]
    fn test_mixed_strategy_randomization() {
        let strategy = ColonelBlottoStrategy::new(3, 60.0);
        
        let mut allocations = Vec::new();
        for _ in 0..100 {
            let alloc = strategy.allocate_resources(&[1.0, 1.0, 1.0]);
            allocations.push(alloc);
        }
        
        // Should have variation in allocations (mixed strategy)
        let first = &allocations[0];
        let different = allocations.iter()
            .any(|a| (a[0] - first[0]).abs() > 1.0);
        assert!(different);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_multi_exchange_routing_decision() {
        let router = GameTheoryRouter::new(5);
        
        let exchange_states = vec![
            ExchangeState {
                liquidity: 1_000_000.0,
                spread_bps: 5.0,
                fee_bps: 10.0,
                latency_ms: 1.5,
            },
            ExchangeState {
                liquidity: 500_000.0,
                spread_bps: 3.0,
                fee_bps: 15.0,
                latency_ms: 0.8,
            },
            ExchangeState {
                liquidity: 2_000_000.0,
                spread_bps: 8.0,
                fee_bps: 5.0,
                latency_ms: 2.0,
            },
        ];
        
        let routing = router.optimal_routing(100_000.0, &exchange_states);
        
        // Should distribute across exchanges
        assert!(routing.allocations.len() > 0);
        
        // Total should equal order size
        let total: f64 = routing.allocations.iter()
            .map(|a| a.size).sum();
        assert!((total - 100_000.0).abs() < 0.01);
    }
}
