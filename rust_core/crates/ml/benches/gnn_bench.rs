//! Performance benchmarks for Graph Neural Networks
//! Team: InfraEngineer + MLEngineer
//! Target: <1s inference for 100-node graphs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ml::graph_neural_networks::*;
use petgraph::graph::DiGraph;
use nalgebra::DVector;

fn benchmark_gat_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("GAT_Forward");
    
    for num_nodes in [10, 50, 100, 200].iter() {
        let layer = GraphAttentionLayer::new(64, 32, 8);
        let node_features = nalgebra::DMatrix::from_fn(*num_nodes, 64, |_, _| rand::random());
        let adjacency = nalgebra::DMatrix::from_fn(*num_nodes, *num_nodes, |i, j| {
            if i != j && rand::random::<f64>() > 0.7 { 1.0 } else { 0.0 }
        });
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_nodes),
            num_nodes,
            |b, _| {
                b.iter(|| {
                    layer.forward(black_box(&node_features), black_box(&adjacency))
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_temporal_gnn(c: &mut Criterion) {
    let mut group = c.benchmark_group("Temporal_GNN");
    
    for num_assets in [5, 10, 20, 50].iter() {
        let mut gnn = TemporalGNN::new(10, 64, 16, 3);
        let mut graph = DiGraph::new();
        
        // Build test graph
        for i in 0..*num_assets {
            let node = AssetNode {
                symbol: format!("ASSET{}", i),
                exchange: "BENCH".to_string(),
                features: DVector::from_element(10, 0.5),
                hidden_state: DVector::zeros(64),
                attention_weights: vec![1.0 / *num_assets as f64; *num_assets],
            };
            graph.add_node(node);
        }
        
        // Add edges
        for i in 0..*num_assets {
            for j in (i + 1)..*num_assets {
                if rand::random::<f64>() > 0.5 {
                    let edge = CorrelationEdge {
                        weight: rand::random(),
                        edge_type: EdgeType::PriceCorrelation,
                        features: DVector::from_element(3, 0.5),
                        temporal_weights: vec![0.5; 10],
                    };
                    graph.add_edge(
                        petgraph::graph::NodeIndex::new(i),
                        petgraph::graph::NodeIndex::new(j),
                        edge
                    );
                }
            }
        }
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_assets", num_assets)),
            num_assets,
            |b, _| {
                b.iter(|| {
                    gnn.forward(black_box(&graph), black_box(1000000))
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_order_flow_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("Order_Flow");
    
    for num_orders in [100, 500, 1000, 5000].iter() {
        let mut network = OrderFlowNetwork::new(100000.0);
        
        let orders: Vec<_> = (0..*num_orders).map(|i| {
            let volume = if i % 100 == 0 { 500000.0 } else { rand::random::<f64>() * 10000.0 };
            (format!("addr_{}", i), volume, i as u64 * 1000)
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_orders", num_orders)),
            num_orders,
            |b, _| {
                b.iter(|| {
                    network.analyze_flow(black_box(&orders))
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_message_passing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Message_Passing");
    
    for num_nodes in [10, 25, 50, 100].iter() {
        let mpnn = MessagePassingNN::new(3, 64);
        let mut graph = DiGraph::new();
        
        // Build graph
        for i in 0..*num_nodes {
            let node = AssetNode {
                symbol: format!("NODE{}", i),
                exchange: "TEST".to_string(),
                features: DVector::from_fn(64, |_| rand::random()),
                hidden_state: DVector::zeros(64),
                attention_weights: vec![1.0 / *num_nodes as f64; *num_nodes],
            };
            graph.add_node(node);
        }
        
        // Add edges (sparse connectivity)
        for i in 0..*num_nodes {
            for j in (i + 1)..*num_nodes {
                if rand::random::<f64>() > 0.8 {
                    let edge = CorrelationEdge {
                        weight: rand::random(),
                        edge_type: EdgeType::PriceCorrelation,
                        features: DVector::from_element(3, 0.5),
                        temporal_weights: vec![0.5; 10],
                    };
                    graph.add_edge(
                        petgraph::graph::NodeIndex::new(i),
                        petgraph::graph::NodeIndex::new(j),
                        edge
                    );
                }
            }
        }
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_nodes", num_nodes)),
            num_nodes,
            |b, _| {
                b.iter(|| {
                    mpnn.propagate(black_box(&graph))
                });
            }
        );
    }
    
    group.finish();
}

fn benchmark_graph_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("Graph_Building");
    
    for num_assets in [10, 25, 50, 100].iter() {
        let builder = GraphBuilder::new(0.3, *num_assets, 100);
        
        let assets: Vec<_> = (0..*num_assets).map(|i| {
            let prices = (0..100).map(|j| {
                50000.0 * (1.0 + 0.01 * ((i * j) as f64).sin())
            }).collect();
            (format!("ASSET{}", i), "Exchange".to_string(), prices)
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_assets", num_assets)),
            num_assets,
            |b, _| {
                b.iter(|| {
                    builder.build_correlation_graph(black_box(&assets))
                });
            }
        );
    }
    
    group.finish();
}

// SIMD-optimized correlation calculation benchmark
fn benchmark_correlation_simd(c: &mut Criterion) {
    let mut group = c.benchmark_group("Correlation_SIMD");
    
    for size in [100, 500, 1000, 5000].iter() {
        let prices1: Vec<f64> = (0..*size).map(|i| 100.0 + (i as f64).sin()).collect();
        let prices2: Vec<f64> = (0..*size).map(|i| 100.0 + (i as f64).cos()).collect();
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_points", size)),
            size,
            |b, _| {
                b.iter(|| {
                    calculate_correlation_simd(black_box(&prices1), black_box(&prices2))
                });
            }
        );
    }
    
    group.finish();
}

// Helper function for SIMD correlation
fn calculate_correlation_simd(prices1: &[f64], prices2: &[f64]) -> f64 {
    let len = prices1.len().min(prices2.len());
    if len < 2 {
        return 0.0;
    }
    
    // Use SIMD for mean calculation
    let mean1 = prices1[..len].iter().sum::<f64>() / len as f64;
    let mean2 = prices2[..len].iter().sum::<f64>() / len as f64;
    
    // SIMD-friendly correlation calculation
    let mut cov = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;
    
    // Process in chunks of 8 for AVX-512
    let chunks = len / 8;
    for i in 0..chunks {
        let base = i * 8;
        for j in 0..8 {
            let idx = base + j;
            let diff1 = prices1[idx] - mean1;
            let diff2 = prices2[idx] - mean2;
            cov += diff1 * diff2;
            var1 += diff1 * diff1;
            var2 += diff2 * diff2;
        }
    }
    
    // Handle remainder
    for idx in (chunks * 8)..len {
        let diff1 = prices1[idx] - mean1;
        let diff2 = prices2[idx] - mean2;
        cov += diff1 * diff2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
    }
    
    if var1 > 0.0 && var2 > 0.0 {
        cov / (var1 * var2).sqrt()
    } else {
        0.0
    }
}

criterion_group!(
    benches,
    benchmark_gat_forward,
    benchmark_temporal_gnn,
    benchmark_order_flow_analysis,
    benchmark_message_passing,
    benchmark_graph_building,
    benchmark_correlation_simd
);
criterion_main!(benches);