//! Integration tests for Graph Neural Networks
//! Team: Full 8-Agent ULTRATHINK Collaboration
//! Validates GNN functionality for market correlation analysis

use ml::graph_neural_networks::*;
use petgraph::graph::DiGraph;
use nalgebra::DVector;
use std::collections::HashMap;

#[test]
fn test_end_to_end_market_prediction() {
    // Architect: System design validation
    // MLEngineer: Model implementation
    // RiskQuant: Risk propagation validation
    
    let mut gnn = TemporalGNN::new(10, 64, 16, 3);
    let builder = GraphBuilder::new(0.3, 10, 100);
    
    // Create realistic market data
    let assets = vec![
        ("BTC-USDT".to_string(), "Binance".to_string(), generate_price_series(50000.0, 0.02)),
        ("ETH-USDT".to_string(), "Binance".to_string(), generate_price_series(3000.0, 0.025)),
        ("SOL-USDT".to_string(), "Binance".to_string(), generate_price_series(100.0, 0.03)),
        ("AVAX-USDT".to_string(), "Binance".to_string(), generate_price_series(30.0, 0.035)),
        ("MATIC-USDT".to_string(), "Binance".to_string(), generate_price_series(1.0, 0.04)),
    ];
    
    // Build correlation graph
    let graph = builder.build_correlation_graph(&assets);
    
    // Generate prediction
    let prediction = gnn.forward(&graph, 1000000);
    
    // Validate prediction structure
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    assert!(!prediction.correlations.is_empty());
    assert!(!prediction.next_move_probabilities.is_empty());
    
    // Validate probabilities sum to ~1.0
    let prob_sum: f64 = prediction.next_move_probabilities.values().sum();
    assert!((prob_sum - 1.0).abs() < 0.01);
    
    // Validate risk propagation
    for (asset, risk) in &prediction.risk_propagation {
        assert!(*risk >= 0.0 && *risk <= 1.0, "Invalid risk for {}: {}", asset, risk);
    }
}

#[test]
fn test_arbitrage_detection() {
    // ExchangeSpec: Multi-exchange arbitrage
    // RiskQuant: Profit calculation validation
    
    let mut gnn = TemporalGNN::new(10, 64, 16, 2);
    let mut graph = DiGraph::new();
    
    // Create arbitrage opportunity scenario
    let btc_binance = AssetNode {
        symbol: "BTC-USDT".to_string(),
        exchange: "Binance".to_string(),
        features: DVector::from_vec(vec![50000.0, 1000000.0, 0.01, 50000.0, 100.0, 
                                         0.001, 51000.0, 49000.0, 0.0, 0.6]),
        hidden_state: DVector::zeros(64),
        attention_weights: vec![0.25; 4],
    };
    
    let btc_coinbase = AssetNode {
        symbol: "BTC-USD".to_string(),
        exchange: "Coinbase".to_string(),
        features: DVector::from_vec(vec![50100.0, 900000.0, 0.012, 50050.0, 110.0,
                                         0.002, 51100.0, 49100.0, 0.1, 0.62]),
        hidden_state: DVector::zeros(64),
        attention_weights: vec![0.25; 4],
    };
    
    let eth_binance = AssetNode {
        symbol: "ETH-USDT".to_string(),
        exchange: "Binance".to_string(),
        features: DVector::from_vec(vec![3000.0, 500000.0, 0.015, 3000.0, 50.0,
                                         0.003, 3100.0, 2900.0, 0.0, 0.58]),
        hidden_state: DVector::zeros(64),
        attention_weights: vec![0.25; 4],
    };
    
    let eth_kraken = AssetNode {
        symbol: "ETH-USD".to_string(),
        exchange: "Kraken".to_string(),
        features: DVector::from_vec(vec![3005.0, 450000.0, 0.018, 3002.0, 55.0,
                                         0.004, 3105.0, 2905.0, 0.05, 0.59]),
        hidden_state: DVector::zeros(64),
        attention_weights: vec![0.25; 4],
    };
    
    let idx1 = graph.add_node(btc_binance);
    let idx2 = graph.add_node(btc_coinbase);
    let idx3 = graph.add_node(eth_binance);
    let idx4 = graph.add_node(eth_kraken);
    
    // Add strong correlation edges
    let edge_strong = CorrelationEdge {
        weight: 0.95,
        edge_type: EdgeType::Arbitrage,
        features: DVector::from_vec(vec![0.95, 0.9, 0.01]),
        temporal_weights: vec![0.95; 10],
    };
    
    let edge_medium = CorrelationEdge {
        weight: 0.7,
        edge_type: EdgeType::PriceCorrelation,
        features: DVector::from_vec(vec![0.7, 0.6, 0.05]),
        temporal_weights: vec![0.7; 10],
    };
    
    graph.add_edge(idx1, idx2, edge_strong.clone());
    graph.add_edge(idx3, idx4, edge_strong);
    graph.add_edge(idx1, idx3, edge_medium.clone());
    graph.add_edge(idx2, idx4, edge_medium);
    
    let prediction = gnn.forward(&graph, 2000000);
    
    // Should detect arbitrage opportunities
    assert!(!prediction.arbitrage_opportunities.is_empty());
    
    // Validate arbitrage paths
    for arb in &prediction.arbitrage_opportunities {
        assert!(arb.strength > 0.5);
        assert!(arb.expected_profit > 0.0);
        println!("Arbitrage: {} -> {}, strength: {:.3}, profit: {:.2}", 
                 arb.from, arb.to, arb.strength, arb.expected_profit);
    }
}

#[test]
fn test_whale_detection() {
    // IntegrationValidator: Order flow validation
    // ComplianceAuditor: Whale tracking
    
    let mut network = OrderFlowNetwork::new(100000.0);
    
    // Simulate whale activity
    let orders = vec![
        ("0xwhale1".to_string(), 1500000.0, 1000000),  // Large whale order
        ("0xtrader1".to_string(), 1000.0, 1000010),
        ("0xtrader2".to_string(), 2000.0, 1000020),
        ("0xwhale2".to_string(), 800000.0, 1000030),  // Medium whale
        ("0xtrader3".to_string(), 500.0, 1000040),
        ("0xwhale1".to_string(), 2000000.0, 1000050),  // Same whale again
        ("0xbot1".to_string(), 100.0, 1000051),
        ("0xbot2".to_string(), 100.0, 1000052),
        ("0xbot3".to_string(), 100.0, 1000053),
        ("0xbot4".to_string(), 100.0, 1000054),
        ("0xbot5".to_string(), 100.0, 1000055),
    ];
    
    let analysis = network.analyze_flow(&orders);
    
    // Should detect whale movements
    assert!(analysis.whale_movements.len() >= 3);
    
    // Validate whale detection
    let total_whale_volume: f64 = analysis.whale_movements.iter()
        .map(|w| w.volume)
        .sum();
    assert!(total_whale_volume > 4000000.0);
    
    // Check for order clustering (potential bot activity)
    assert!(!analysis.order_clusters.is_empty());
    
    // Validate spoofing detection
    println!("Spoofing probability: {:.2}%", analysis.spoofing_probability * 100.0);
    
    // Check flow imbalance
    assert!(analysis.flow_imbalance.abs() <= 1.0);
}

#[test]
fn test_message_passing_convergence() {
    // InfraEngineer: Performance validation
    // MLEngineer: Convergence testing
    
    let mpnn = MessagePassingNN::new(5, 64);
    let mut graph = DiGraph::new();
    
    // Create a complex graph structure
    for i in 0..10 {
        let node = AssetNode {
            symbol: format!("ASSET{}", i),
            exchange: "TEST".to_string(),
            features: DVector::from_fn(64, |j| ((i + j) as f64).sin()),
            hidden_state: DVector::zeros(64),
            attention_weights: vec![0.1; 10],
        };
        graph.add_node(node);
    }
    
    // Add various edge types
    for i in 0..9 {
        let edge = CorrelationEdge {
            weight: 0.5 + (i as f64 * 0.05),
            edge_type: match i % 3 {
                0 => EdgeType::PriceCorrelation,
                1 => EdgeType::VolumeFlow,
                _ => EdgeType::LeadLag,
            },
            features: DVector::from_element(3, 0.5),
            temporal_weights: vec![0.5; 10],
        };
        graph.add_edge(petgraph::graph::NodeIndex::new(i), 
                      petgraph::graph::NodeIndex::new(i + 1), edge);
    }
    
    // Add some cycles
    let cycle_edge = CorrelationEdge {
        weight: 0.3,
        edge_type: EdgeType::Cointegration,
        features: DVector::from_element(3, 0.3),
        temporal_weights: vec![0.3; 10],
    };
    graph.add_edge(petgraph::graph::NodeIndex::new(9), 
                  petgraph::graph::NodeIndex::new(0), cycle_edge.clone());
    graph.add_edge(petgraph::graph::NodeIndex::new(4), 
                  petgraph::graph::NodeIndex::new(7), cycle_edge);
    
    // Run message passing
    let final_states = mpnn.propagate(&graph);
    
    // Validate convergence
    assert_eq!(final_states.len(), 10);
    
    // Check that information propagated
    for (i, state) in final_states.iter().enumerate() {
        let non_zero_count = state.iter().filter(|&&x| x.abs() > 1e-6).count();
        assert!(non_zero_count > 32, "Node {} didn't receive enough messages", i);
        
        // Check numerical stability
        assert!(state.iter().all(|&x| x.is_finite()), "Node {} has NaN/Inf values", i);
        assert!(state.norm() < 1000.0, "Node {} state exploded", i);
    }
}

#[test]
fn test_temporal_dynamics() {
    // QualityGate: Time series validation
    // RiskQuant: Temporal risk analysis
    
    let mut gnn = TemporalGNN::new(10, 32, 8, 2);
    let builder = GraphBuilder::new(0.2, 5, 50);
    
    // Simulate temporal evolution
    let mut predictions = Vec::new();
    
    for t in 0..10 {
        // Generate time-varying prices
        let assets = vec![
            ("BTC".to_string(), "Exchange1".to_string(), 
             generate_trending_series(50000.0, 0.01, t as f64 * 0.001)),
            ("ETH".to_string(), "Exchange1".to_string(),
             generate_trending_series(3000.0, 0.015, t as f64 * 0.0015)),
            ("SOL".to_string(), "Exchange1".to_string(),
             generate_trending_series(100.0, 0.02, t as f64 * 0.002)),
        ];
        
        let graph = builder.build_correlation_graph(&assets);
        let prediction = gnn.forward(&graph, 1000000 + t * 1000);
        predictions.push(prediction);
    }
    
    // Validate temporal consistency
    for i in 1..predictions.len() {
        let prev = &predictions[i - 1];
        let curr = &predictions[i];
        
        // Confidence shouldn't jump too much
        let conf_change = (curr.confidence - prev.confidence).abs();
        assert!(conf_change < 0.5, "Confidence jumped too much: {}", conf_change);
        
        // Risk should evolve smoothly
        for (asset, &curr_risk) in &curr.risk_propagation {
            if let Some(&prev_risk) = prev.risk_propagation.get(asset) {
                let risk_change = (curr_risk - prev_risk).abs();
                assert!(risk_change < 0.3, "Risk for {} jumped: {}", asset, risk_change);
            }
        }
    }
}

#[test]
fn test_multi_head_attention() {
    // Architect: Attention mechanism validation
    // MLEngineer: Multi-head implementation
    
    let layer = GraphAttentionLayer::new(16, 8, 8);  // 8 attention heads
    
    // Create input data
    let num_nodes = 20;
    let node_features = nalgebra::DMatrix::from_fn(num_nodes, 16, |i, j| {
        ((i * j) as f64 / 100.0).sin()
    });
    
    // Create fully connected adjacency
    let adjacency = nalgebra::DMatrix::from_fn(num_nodes, num_nodes, |i, j| {
        if i != j { 1.0 } else { 0.0 }
    });
    
    // Forward pass
    let output = layer.forward(&node_features, &adjacency);
    
    // Validate output dimensions
    assert_eq!(output.nrows(), num_nodes);
    assert_eq!(output.ncols(), 64);  // 8 heads * 8 features
    
    // Check that attention worked
    for i in 0..num_nodes {
        for j in 0..64 {
            assert!(output[(i, j)].is_finite());
            assert!(output[(i, j)].abs() <= 1.0);  // tanh activation
        }
    }
    
    // Verify different heads learned different representations
    let head_outputs: Vec<_> = (0..8).map(|h| {
        let start = h * 8;
        let end = (h + 1) * 8;
        output.columns(start, 8).clone_owned()
    }).collect();
    
    // Check that heads are not identical
    for i in 0..7 {
        for j in (i + 1)..8 {
            let diff = (&head_outputs[i] - &head_outputs[j]).norm();
            assert!(diff > 0.01, "Heads {} and {} are too similar", i, j);
        }
    }
}

#[test]
fn test_correlation_calculation() {
    // RiskQuant: Correlation validation
    // InfraEngineer: Numerical stability
    
    let builder = GraphBuilder::new(0.0, 10, 100);  // Accept all correlations
    
    // Test perfect correlation
    let prices1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let prices2 = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    
    let assets = vec![
        ("A".to_string(), "E1".to_string(), prices1.clone()),
        ("B".to_string(), "E1".to_string(), prices2),
    ];
    
    let graph = builder.build_correlation_graph(&assets);
    
    // Should have high correlation edge
    let mut found_high_correlation = false;
    for edge in graph.edge_references() {
        if edge.weight().weight > 0.99 {
            found_high_correlation = true;
            break;
        }
    }
    assert!(found_high_correlation, "Perfect correlation not detected");
    
    // Test anti-correlation
    let prices3 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let assets2 = vec![
        ("C".to_string(), "E1".to_string(), prices1),
        ("D".to_string(), "E1".to_string(), prices3),
    ];
    
    let graph2 = builder.build_correlation_graph(&assets2);
    
    // Should have negative correlation edge
    let mut found_negative_correlation = false;
    for edge in graph2.edge_references() {
        if edge.weight().weight < -0.99 {
            found_negative_correlation = true;
            break;
        }
    }
    assert!(found_negative_correlation, "Anti-correlation not detected");
}

// Helper functions
fn generate_price_series(base: f64, volatility: f64) -> Vec<f64> {
    use rand::distributions::{Distribution, Normal};
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, volatility).unwrap();
    
    let mut prices = vec![base];
    for _ in 1..100 {
        let last = *prices.last().unwrap();
        let change = normal.sample(&mut rng);
        prices.push(last * (1.0 + change));
    }
    prices
}

fn generate_trending_series(base: f64, volatility: f64, trend: f64) -> Vec<f64> {
    use rand::distributions::{Distribution, Normal};
    let mut rng = rand::thread_rng();
    let normal = Normal::new(trend, volatility).unwrap();
    
    let mut prices = vec![base];
    for _ in 1..50 {
        let last = *prices.last().unwrap();
        let change = normal.sample(&mut rng);
        prices.push(last * (1.0 + change));
    }
    prices
}