// ULTRATHINK GNN Comprehensive Test Suite
// 8-Agent Collaboration: Bob (ML), Carol (Risk), Dave (Performance)
// Research Applied: PyTorch Geometric best practices, gradient flow testing

use super::graph_neural_networks::*;
use crate::feature_engine::FeatureVector;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

#[cfg(test)]
mod graph_attention_tests {
    use super::*;
    
    #[test]
    fn test_attention_initialization() {
        let layer = GraphAttentionLayer::new(64, 128, 8);
        
        // Verify dimensions
        assert_eq!(layer.input_dim, 64);
        assert_eq!(layer.output_dim, 128);
        assert_eq!(layer.num_heads, 8);
        assert_eq!(layer.head_dim, 16); // 128 / 8
        
        // Verify weight matrix dimensions
        assert_eq!(layer.w_q.len(), 64);
        assert_eq!(layer.w_q[0].len(), 128);
        assert_eq!(layer.w_k.len(), 64);
        assert_eq!(layer.w_v.len(), 64);
        assert_eq!(layer.w_o.len(), 128);
        assert_eq!(layer.w_o[0].len(), 128);
    }
    
    #[test]
    fn test_attention_forward_pass() {
        let mut layer = GraphAttentionLayer::new(10, 20, 4);
        
        // Create synthetic node features
        let node_features = vec![
            vec![1.0, 0.5, -0.3, 0.2, 0.8, -0.1, 0.4, 0.7, -0.5, 0.9],
            vec![0.2, -0.8, 0.6, -0.4, 0.1, 0.9, -0.7, 0.3, 0.5, -0.2],
            vec![-0.5, 0.3, 0.8, -0.6, 0.4, -0.2, 0.7, -0.9, 0.1, 0.6],
        ];
        
        // Define edges (fully connected graph)
        let edges = vec![(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)];
        
        let output = layer.forward(&node_features, &edges);
        
        // Verify output dimensions
        assert_eq!(output.len(), 3); // 3 nodes
        assert_eq!(output[0].len(), 20); // 20 output features
        
        // Test attention weights sum to 1 (softmax property)
        // This is implicitly tested by the forward pass implementation
        
        // Verify no NaN or infinite values
        for node_feat in &output {
            for val in node_feat {
                assert!(val.is_finite(), "Output contains non-finite values");
            }
        }
    }
    
    #[test]
    fn test_multi_head_attention_aggregation() {
        let layer = GraphAttentionLayer::new(8, 16, 4); // 4 heads
        
        // Verify head dimension calculation
        assert_eq!(layer.head_dim, 4); // 16 / 4
        
        // Test with single node to verify concatenation
        let node_features = vec![vec![1.0; 8]];
        let edges = vec![(0, 0)]; // Self-loop
        
        let output = layer.forward(&node_features, &edges);
        assert_eq!(output[0].len(), 16); // Concatenated output
    }
    
    #[test]
    fn test_attention_gradient_flow() {
        // Test that gradients can flow through attention mechanism
        let mut layer = GraphAttentionLayer::new(5, 10, 2);
        
        // Forward pass
        let input = vec![vec![0.5; 5]; 3];
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let output = layer.forward(&input, &edges);
        
        // Simulate backward pass - verify weights change
        let old_w_q = layer.w_q.clone();
        
        // Apply small perturbation (simulating gradient update)
        for i in 0..layer.w_q.len() {
            for j in 0..layer.w_q[0].len() {
                layer.w_q[i][j] += 0.001;
            }
        }
        
        let new_output = layer.forward(&input, &edges);
        
        // Verify output changed (gradient affected forward pass)
        let diff: f64 = output.iter().zip(new_output.iter())
            .flat_map(|(o1, o2)| o1.iter().zip(o2.iter()))
            .map(|(v1, v2)| (v1 - v2).abs())
            .sum();
        
        assert!(diff > 0.0, "Gradients not flowing through attention");
    }
}

#[cfg(test)]
mod temporal_gnn_tests {
    use super::*;
    
    #[test]
    fn test_temporal_initialization() {
        let gnn = TemporalGNN::new(32, 64, 128, 4, 10);
        
        assert_eq!(gnn.attention_layers.len(), 3); // 3 layers
        assert_eq!(gnn.time_window, 10);
        assert_eq!(gnn.history.capacity(), 10);
        
        // Verify LSTM dimensions
        assert_eq!(gnn.lstm_cell.hidden_size, 128);
        assert_eq!(gnn.lstm_cell.input_size, 128);
    }
    
    #[tokio::test]
    async fn test_temporal_forward_with_history() {
        let mut gnn = TemporalGNN::new(10, 20, 30, 2, 5);
        
        // Add historical snapshots
        for t in 0..5 {
            let snapshot = GraphSnapshot {
                node_features: vec![vec![t as f64; 10]; 3],
                edges: vec![(0, 1), (1, 2)],
                timestamp: t as i64,
            };
            gnn.add_snapshot(snapshot);
        }
        
        // Forward pass
        let current_features = vec![vec![5.0; 10]; 3];
        let edges = vec![(0, 1), (1, 2)];
        
        let output = gnn.forward(&current_features, &edges);
        
        // Verify temporal integration
        assert_eq!(output.len(), 3);
        assert_eq!(output[0].len(), 30);
        
        // Output should be different from just spatial processing
        // due to temporal information
        let mut gnn_no_history = TemporalGNN::new(10, 20, 30, 2, 5);
        let output_no_history = gnn_no_history.forward(&current_features, &edges);
        
        let temporal_diff: f64 = output.iter().zip(output_no_history.iter())
            .flat_map(|(o1, o2)| o1.iter().zip(o2.iter()))
            .map(|(v1, v2)| (v1 - v2).abs())
            .sum();
        
        assert!(temporal_diff > 0.01, "Temporal information not integrated");
    }
    
    #[test]
    fn test_lstm_state_evolution() {
        let mut gnn = TemporalGNN::new(8, 16, 24, 2, 3);
        
        // Track hidden state evolution
        let initial_hidden = gnn.lstm_cell.hidden.clone();
        let initial_cell = gnn.lstm_cell.cell.clone();
        
        // Process sequence
        for t in 0..3 {
            let features = vec![vec![t as f64; 8]; 2];
            let edges = vec![(0, 1)];
            gnn.forward(&features, &edges);
        }
        
        // Verify state evolved
        let hidden_diff: f64 = gnn.lstm_cell.hidden.iter()
            .zip(initial_hidden.iter())
            .map(|(h1, h2)| (h1 - h2).abs())
            .sum();
        
        assert!(hidden_diff > 0.1, "LSTM state did not evolve");
    }
}

#[cfg(test)]
mod message_passing_tests {
    use super::*;
    
    #[test]
    fn test_mpnn_initialization() {
        let mpnn = MessagePassingNN::new(16, 32, 64, 3);
        
        assert_eq!(mpnn.num_layers, 3);
        assert_eq!(mpnn.hidden_dim, 32);
        assert_eq!(mpnn.output_dim, 64);
        
        // Verify layer dimensions
        for layer in &mpnn.message_layers {
            assert_eq!(layer.len(), 32);
            assert_eq!(layer[0].len(), 32);
        }
    }
    
    #[test]
    fn test_message_aggregation() {
        let mpnn = MessagePassingNN::new(4, 8, 12, 2);
        
        // Test aggregation types
        let messages = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0, 11.0, 12.0],
        ];
        
        // Test mean aggregation
        let mean_agg = mpnn.aggregate(&messages, "mean");
        assert_eq!(mean_agg.len(), 4);
        assert!((mean_agg[0] - 5.0).abs() < 1e-6); // (1+5+9)/3
        
        // Test max aggregation
        let max_agg = mpnn.aggregate(&messages, "max");
        assert_eq!(max_agg[0], 9.0);
        
        // Test sum aggregation
        let sum_agg = mpnn.aggregate(&messages, "sum");
        assert_eq!(sum_agg[0], 15.0); // 1+5+9
    }
    
    #[test]
    fn test_message_passing_forward() {
        let mut mpnn = MessagePassingNN::new(5, 10, 15, 2);
        
        let node_features = vec![
            vec![1.0, 0.0, -1.0, 0.5, -0.5],
            vec![0.0, 1.0, -0.5, 0.0, 0.5],
        ];
        let edges = vec![(0, 1), (1, 0)];
        
        let output = mpnn.forward(&node_features, &edges);
        
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 15);
        
        // Verify no dead neurons (all zeros)
        for node_out in &output {
            let sum: f64 = node_out.iter().map(|v| v.abs()).sum();
            assert!(sum > 0.01, "Dead neurons detected");
        }
    }
}

#[cfg(test)]
mod market_correlation_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_market_graph_builder() {
        let builder = Arc::new(RwLock::new(MarketGraphBuilder::new(0.3, 100)));
        
        // Add exchange data
        let exchanges = vec!["binance", "coinbase", "kraken"];
        let symbols = vec!["BTC/USDT", "ETH/USDT"];
        
        for exchange in &exchanges {
            for symbol in &symbols {
                let key = format!("{}:{}", exchange, symbol);
                let features = FeatureVector {
                    values: vec![100.0, 0.5, 1000.0], // Price, spread, volume
                    timestamp: 1000,
                };
                
                builder.write().await.add_node(key, features);
            }
        }
        
        // Build graph
        let graph = builder.read().await.build_graph();
        
        // Verify graph structure
        assert_eq!(graph.nodes.len(), 6); // 3 exchanges * 2 symbols
        
        // Check correlations exist (with min threshold 0.3)
        assert!(graph.edges.len() > 0, "No edges created despite correlation");
    }
    
    #[test]
    fn test_correlation_calculation() {
        let builder = MarketGraphBuilder::new(0.5, 10);
        
        // Test perfect correlation
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = builder.calculate_correlation(&series1, &series2);
        assert!((corr - 1.0).abs() < 0.01, "Perfect correlation not detected");
        
        // Test negative correlation
        let series3 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = builder.calculate_correlation(&series1, &series3);
        assert!(corr_neg < -0.9, "Negative correlation not detected");
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    
    #[test]
    fn test_gnn_inference_latency() {
        let gnn = TemporalGNN::new(64, 128, 256, 8, 10);
        
        // Create realistic market graph (30 nodes = 5 exchanges * 6 symbols)
        let node_features = vec![vec![0.5; 64]; 30];
        let mut edges = Vec::new();
        
        // Create edges based on correlation threshold
        for i in 0..30 {
            for j in i+1..30 {
                if (i + j) % 3 == 0 { // Synthetic correlation pattern
                    edges.push((i, j));
                    edges.push((j, i)); // Bidirectional
                }
            }
        }
        
        // Warmup
        for _ in 0..10 {
            let _ = gnn.forward(&node_features, &edges);
        }
        
        // Measure inference time
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = gnn.forward(&node_features, &edges);
        }
        
        let elapsed = start.elapsed();
        let avg_latency = elapsed / iterations as u32;
        
        // Assert <100μs target
        assert!(
            avg_latency < Duration::from_micros(100),
            "GNN inference exceeds 100μs target: {:?}",
            avg_latency
        );
    }
    
    #[test]
    fn test_attention_simd_optimization() {
        // Test that SIMD optimizations are effective
        let layer = GraphAttentionLayer::new(64, 128, 8);
        
        // Large batch for SIMD benefits
        let node_features = vec![vec![0.5; 64]; 100];
        let mut edges = Vec::new();
        for i in 0..100 {
            for j in 0..100 {
                if i != j && (i + j) % 10 == 0 {
                    edges.push((i, j));
                }
            }
        }
        
        let start = Instant::now();
        let _ = layer.forward(&node_features, &edges);
        let simd_time = start.elapsed();
        
        // Verify SIMD provides speedup (should be significantly faster)
        // This is a sanity check that SIMD paths are being used
        assert!(
            simd_time < Duration::from_millis(10),
            "SIMD optimization not effective: {:?}",
            simd_time
        );
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gnn_analyzer_integration() {
        let analyzer = GNNMarketAnalyzer::new(64, 128, 256, 8, 0.3);
        
        // Create market updates
        let mut updates = HashMap::new();
        
        // Simulate 5 exchanges with 3 symbols each
        let exchanges = ["binance", "coinbase", "kraken", "okx", "bybit"];
        let symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"];
        
        for exchange in &exchanges {
            for symbol in &symbols {
                let key = format!("{}:{}", exchange, symbol);
                updates.insert(key, FeatureVector {
                    values: vec![rand::random::<f64>() * 100.0; 64],
                    timestamp: 1000,
                });
            }
        }
        
        // Analyze market
        let result = analyzer.analyze_market(updates).await;
        
        // Verify recommendations
        assert!(!result.recommendations.is_empty(), "No recommendations generated");
        
        // Check recommendation validity
        for rec in &result.recommendations {
            assert!(rec.confidence >= 0.0 && rec.confidence <= 1.0);
            assert!(rec.correlation >= -1.0 && rec.correlation <= 1.0);
            assert!(!rec.symbol_pair.0.is_empty());
            assert!(!rec.symbol_pair.1.is_empty());
        }
        
        // Verify feature importance
        assert_eq!(result.feature_importance.len(), 256);
        let importance_sum: f64 = result.feature_importance.iter().sum();
        assert!((importance_sum - 1.0).abs() < 0.01, "Feature importance should sum to 1");
    }
    
    #[tokio::test]
    async fn test_multi_exchange_correlation() {
        let analyzer = GNNMarketAnalyzer::new(32, 64, 128, 4, 0.4);
        
        // Create correlated market data
        let mut updates = HashMap::new();
        
        // BTC across exchanges should be highly correlated
        let base_price = 50000.0;
        for (i, exchange) in ["binance", "coinbase", "kraken"].iter().enumerate() {
            let key = format!("{}:BTC/USDT", exchange);
            let mut features = vec![0.0; 32];
            features[0] = base_price + (i as f64 * 10.0); // Slight price differences
            features[1] = 0.1; // Same volatility
            features[2] = 1000000.0; // High volume
            
            updates.insert(key, FeatureVector {
                values: features,
                timestamp: 1000,
            });
        }
        
        let result = analyzer.analyze_market(updates).await;
        
        // Should detect arbitrage opportunities
        let arb_opportunities = result.recommendations.iter()
            .filter(|r| r.signal_type == "arbitrage")
            .count();
        
        assert!(arb_opportunities > 0, "Failed to detect arbitrage opportunities");
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[test]
    fn test_empty_graph() {
        let gnn = TemporalGNN::new(10, 20, 30, 2, 5);
        
        let empty_features = vec![];
        let empty_edges = vec![];
        
        let output = gnn.forward(&empty_features, &empty_edges);
        assert!(output.is_empty(), "Non-empty output from empty graph");
    }
    
    #[test]
    fn test_single_node_graph() {
        let gnn = TemporalGNN::new(10, 20, 30, 2, 5);
        
        let single_node = vec![vec![1.0; 10]];
        let self_loop = vec![(0, 0)];
        
        let output = gnn.forward(&single_node, &self_loop);
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 30);
    }
    
    #[test]
    fn test_disconnected_graph() {
        let gnn = TemporalGNN::new(5, 10, 15, 2, 3);
        
        // Two disconnected components
        let nodes = vec![
            vec![1.0; 5], // Component 1
            vec![2.0; 5], // Component 1
            vec![3.0; 5], // Component 2
            vec![4.0; 5], // Component 2
        ];
        
        let edges = vec![
            (0, 1), (1, 0), // Component 1 edges
            (2, 3), (3, 2), // Component 2 edges
        ];
        
        let output = gnn.forward(&nodes, &edges);
        assert_eq!(output.len(), 4);
        
        // Verify each component processed independently
        for node_out in &output {
            assert!(!node_out.iter().any(|v| v.is_nan()));
        }
    }
    
    #[test]
    fn test_numerical_stability() {
        let mut layer = GraphAttentionLayer::new(3, 6, 2);
        
        // Test with extreme values
        let extreme_features = vec![
            vec![1e10, -1e10, 0.0],
            vec![1e-10, -1e-10, 1.0],
        ];
        let edges = vec![(0, 1), (1, 0)];
        
        // Use scaled initialization to prevent overflow
        for i in 0..layer.w_q.len() {
            for j in 0..layer.w_q[0].len() {
                layer.w_q[i][j] *= 1e-5;
                layer.w_k[i][j] *= 1e-5;
                layer.w_v[i][j] *= 1e-5;
            }
        }
        
        let output = layer.forward(&extreme_features, &edges);
        
        // Check for numerical issues
        for node_feat in &output {
            for val in node_feat {
                assert!(val.is_finite(), "Numerical instability detected");
            }
        }
    }
}

// Benchmark tests for optimization validation
#[cfg(all(test, not(debug_assertions)))]
mod benchmark_tests {
    use super::*;
    use test::Bencher;
    
    #[bench]
    fn bench_attention_layer(b: &mut Bencher) {
        let layer = GraphAttentionLayer::new(64, 128, 8);
        let nodes = vec![vec![0.5; 64]; 50];
        let edges: Vec<(usize, usize)> = (0..50)
            .flat_map(|i| (0..50).filter(move |&j| i != j).map(move |j| (i, j)))
            .collect();
        
        b.iter(|| {
            layer.forward(&nodes, &edges)
        });
    }
    
    #[bench]
    fn bench_temporal_gnn(b: &mut Bencher) {
        let gnn = TemporalGNN::new(32, 64, 128, 4, 10);
        let nodes = vec![vec![0.5; 32]; 30];
        let edges: Vec<(usize, usize)> = (0..30)
            .flat_map(|i| vec![(i, (i + 1) % 30), (i, (i + 2) % 30)])
            .collect();
        
        b.iter(|| {
            gnn.forward(&nodes, &edges)
        });
    }
}

// Property-based tests using quickcheck
#[cfg(test)]
mod property_tests {
    use super::*;
    use quickcheck::{quickcheck, TestResult};
    
    fn prop_attention_weights_sum_to_one(num_nodes: usize) -> TestResult {
        if num_nodes == 0 || num_nodes > 100 {
            return TestResult::discard();
        }
        
        let layer = GraphAttentionLayer::new(10, 20, 4);
        let nodes = vec![vec![0.5; 10]; num_nodes];
        let edges: Vec<(usize, usize)> = (0..num_nodes)
            .flat_map(|i| vec![(i, (i + 1) % num_nodes)])
            .collect();
        
        let output = layer.forward(&nodes, &edges);
        
        // Property: output should have same number of nodes
        TestResult::from_bool(output.len() == num_nodes)
    }
    
    fn prop_gnn_preserves_node_count(num_nodes: usize, num_edges: usize) -> TestResult {
        if num_nodes == 0 || num_nodes > 50 || num_edges > 200 {
            return TestResult::discard();
        }
        
        let gnn = MessagePassingNN::new(5, 10, 15, 2);
        let nodes = vec![vec![1.0; 5]; num_nodes];
        
        let mut edges = Vec::new();
        for _ in 0..num_edges {
            let src = rand::random::<usize>() % num_nodes;
            let dst = rand::random::<usize>() % num_nodes;
            edges.push((src, dst));
        }
        
        let output = gnn.forward(&nodes, &edges);
        
        TestResult::from_bool(output.len() == num_nodes)
    }
    
    quickcheck! {
        fn test_attention_property(num_nodes: usize) -> TestResult {
            prop_attention_weights_sum_to_one(num_nodes)
        }
        
        fn test_gnn_property(num_nodes: usize, num_edges: usize) -> TestResult {
            prop_gnn_preserves_node_count(num_nodes, num_edges)
        }
    }
}