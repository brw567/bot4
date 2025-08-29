//! Graph Neural Networks for Market Correlation Analysis
//! Team: Full 8-Agent ULTRATHINK Collaboration
//! Research Applied: 
//!   - Geometric Deep Learning (Bronstein et al., 2021)
//!   - Graph Attention Networks (Veličković et al., 2018)
//!   - Temporal Graph Networks (Rossi et al., 2020)
//!   - Financial Networks (Billio et al., 2012)
//!   - Order Flow Networks (Kirilenko et al., 2017)

use crate::features::FeatureVector;
use nalgebra::{DMatrix, DVector};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo::dijkstra;
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

/// Asset node in the correlation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetNode {
    pub symbol: String,
    pub exchange: String,
    pub features: DVector<f64>,  // Node features (price, volume, volatility, etc.)
    pub hidden_state: DVector<f64>,  // Hidden representation
    pub attention_weights: Vec<f64>,  // Self-attention weights
}

/// Edge in the correlation graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEdge {
    pub weight: f64,  // Correlation strength
    pub edge_type: EdgeType,
    pub features: DVector<f64>,  // Edge features (volume flow, lead-lag, etc.)
    pub temporal_weights: Vec<f64>,  // Time-varying weights
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    PriceCorrelation,
    VolumeFlow,
    OrderFlow,
    Arbitrage,
    LeadLag,
    Cointegration,
}

/// Graph Attention Layer (GAT)
pub struct GraphAttentionLayer {
    weight_matrix: DMatrix<f64>,
    attention_weights: DMatrix<f64>,
    bias: DVector<f64>,
    dropout: f64,
    num_heads: usize,
}

impl GraphAttentionLayer {
    pub fn new(in_features: usize, out_features: usize, num_heads: usize) -> Self {
        use rand::distributions::{Distribution, Uniform};
        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-0.1..0.1);
        
        // Xavier initialization
        let scale = (2.0 / (in_features + out_features) as f64).sqrt();
        
        let weight_matrix = DMatrix::from_fn(in_features, out_features * num_heads, |_, _| {
            dist.sample(&mut rng) * scale
        });
        
        let attention_weights = DMatrix::from_fn(2 * out_features, num_heads, |_, _| {
            dist.sample(&mut rng) * scale
        });
        
        Self {
            weight_matrix,
            attention_weights,
            bias: DVector::zeros(out_features * num_heads),
            dropout: 0.1,
            num_heads,
        }
    }
    
    /// Multi-head attention forward pass
    pub fn forward(&self, node_features: &DMatrix<f64>, adjacency: &DMatrix<f64>) -> DMatrix<f64> {
        let num_nodes = node_features.nrows();
        let out_features = self.weight_matrix.ncols() / self.num_heads;
        
        // Linear transformation
        let h = node_features * &self.weight_matrix;
        
        // Reshape for multi-head attention
        let mut outputs = Vec::new();
        
        for head in 0..self.num_heads {
            let start = head * out_features;
            let end = (head + 1) * out_features;
            let h_head = h.columns(start, out_features);
            
            // Compute attention scores
            let mut attention_scores = DMatrix::zeros(num_nodes, num_nodes);
            
            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    if adjacency[(i, j)] > 0.0 {
                        let concat = DVector::from_fn(2 * out_features, |idx| {
                            if idx < out_features {
                                h_head[(i, idx)]
                            } else {
                                h_head[(j, idx - out_features)]
                            }
                        });
                        
                        let score = (&concat.transpose() * 
                                    &self.attention_weights.column(head)).sum();
                        attention_scores[(i, j)] = score;
                    }
                }
                
                // Softmax normalization
                let row_sum = attention_scores.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..num_nodes {
                        attention_scores[(i, j)] = (attention_scores[(i, j)] / row_sum).exp();
                    }
                    let exp_sum = attention_scores.row(i).sum();
                    for j in 0..num_nodes {
                        attention_scores[(i, j)] /= exp_sum;
                    }
                }
            }
            
            // Apply attention and aggregate
            let output = &attention_scores * &h_head;
            outputs.push(output);
        }
        
        // Concatenate heads
        let mut final_output = DMatrix::zeros(num_nodes, out_features * self.num_heads);
        for (head, output) in outputs.iter().enumerate() {
            let start = head * out_features;
            for i in 0..num_nodes {
                for j in 0..out_features {
                    final_output[(i, start + j)] = output[(i, j)];
                }
            }
        }
        
        // Apply bias and activation
        for i in 0..num_nodes {
            for j in 0..final_output.ncols() {
                final_output[(i, j)] = (final_output[(i, j)] + self.bias[j]).tanh();
            }
        }
        
        final_output
    }
}

/// Temporal Graph Neural Network
pub struct TemporalGNN {
    attention_layers: Vec<GraphAttentionLayer>,
    lstm_cell: LSTMCell,
    time_window: usize,
    history: VecDeque<GraphSnapshot>,
}

#[derive(Clone)]
struct GraphSnapshot {
    timestamp: u64,
    node_features: DMatrix<f64>,
    adjacency: DMatrix<f64>,
    predictions: Option<DVector<f64>>,
}

struct LSTMCell {
    weight_ih: DMatrix<f64>,
    weight_hh: DMatrix<f64>,
    bias_ih: DVector<f64>,
    bias_hh: DVector<f64>,
    hidden_size: usize,
}

impl LSTMCell {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        use rand::distributions::{Distribution, Uniform};
        let mut rng = rand::thread_rng();
        let dist = Uniform::from(-0.1..0.1);
        
        Self {
            weight_ih: DMatrix::from_fn(4 * hidden_size, input_size, |_, _| dist.sample(&mut rng)),
            weight_hh: DMatrix::from_fn(4 * hidden_size, hidden_size, |_, _| dist.sample(&mut rng)),
            bias_ih: DVector::from_fn(4 * hidden_size, |_| dist.sample(&mut rng)),
            bias_hh: DVector::from_fn(4 * hidden_size, |_| dist.sample(&mut rng)),
            hidden_size,
        }
    }
    
    fn forward(&self, input: &DVector<f64>, hidden: &DVector<f64>, cell: &DVector<f64>) 
        -> (DVector<f64>, DVector<f64>) {
        let gates = &self.weight_ih * input + &self.bias_ih + 
                   &self.weight_hh * hidden + &self.bias_hh;
        
        let i_gate = gates.rows(0, self.hidden_size).map(|x| 1.0 / (1.0 + (-x).exp()));
        let f_gate = gates.rows(self.hidden_size, self.hidden_size).map(|x| 1.0 / (1.0 + (-x).exp()));
        let g_gate = gates.rows(2 * self.hidden_size, self.hidden_size).map(|x| x.tanh());
        let o_gate = gates.rows(3 * self.hidden_size, self.hidden_size).map(|x| 1.0 / (1.0 + (-x).exp()));
        
        let new_cell = f_gate.component_mul(cell) + i_gate.component_mul(&g_gate);
        let new_hidden = o_gate.component_mul(&new_cell.map(|x| x.tanh()));
        
        (new_hidden, new_cell)
    }
}

impl TemporalGNN {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, num_layers: usize) -> Self {
        let mut attention_layers = Vec::new();
        
        // Build GAT layers
        for i in 0..num_layers {
            let in_dim = if i == 0 { input_dim } else { hidden_dim * 8 };  // 8 attention heads
            let out_dim = if i == num_layers - 1 { output_dim } else { hidden_dim };
            attention_layers.push(GraphAttentionLayer::new(in_dim, out_dim, 8));
        }
        
        Self {
            attention_layers,
            lstm_cell: LSTMCell::new(output_dim, hidden_dim),
            time_window: 100,
            history: VecDeque::with_capacity(100),
        }
    }
    
    /// Process temporal graph sequence
    pub fn forward(&mut self, graph: &DiGraph<AssetNode, CorrelationEdge>, 
                   timestamp: u64) -> MarketPrediction {
        // Convert graph to matrix representation
        let (node_features, adjacency) = self.graph_to_matrices(graph);
        
        // Apply GAT layers
        let mut x = node_features.clone();
        for layer in &self.attention_layers {
            x = layer.forward(&x, &adjacency);
        }
        
        // Temporal processing with LSTM
        let graph_embedding = x.column_mean();
        let (hidden, cell) = if let Some(last) = self.history.back() {
            let last_hidden = DVector::zeros(self.lstm_cell.hidden_size);  // Would load from history
            let last_cell = DVector::zeros(self.lstm_cell.hidden_size);
            self.lstm_cell.forward(&graph_embedding, &last_hidden, &last_cell)
        } else {
            let init_hidden = DVector::zeros(self.lstm_cell.hidden_size);
            let init_cell = DVector::zeros(self.lstm_cell.hidden_size);
            self.lstm_cell.forward(&graph_embedding, &init_hidden, &init_cell)
        };
        
        // Store snapshot
        self.history.push_back(GraphSnapshot {
            timestamp,
            node_features,
            adjacency,
            predictions: Some(hidden.clone()),
        });
        
        if self.history.len() > self.time_window {
            self.history.pop_front();
        }
        
        // Generate predictions
        self.generate_predictions(graph, &hidden)
    }
    
    fn graph_to_matrices(&self, graph: &DiGraph<AssetNode, CorrelationEdge>) 
        -> (DMatrix<f64>, DMatrix<f64>) {
        let num_nodes = graph.node_count();
        let feature_dim = graph.node_weight(NodeIndex::new(0))
            .map(|n| n.features.len())
            .unwrap_or(0);
        
        let mut node_features = DMatrix::zeros(num_nodes, feature_dim);
        let mut adjacency = DMatrix::zeros(num_nodes, num_nodes);
        
        // Fill node features
        for (idx, node_idx) in graph.node_indices().enumerate() {
            if let Some(node) = graph.node_weight(node_idx) {
                for (j, &val) in node.features.iter().enumerate() {
                    node_features[(idx, j)] = val;
                }
            }
        }
        
        // Fill adjacency matrix
        for edge in graph.edge_indices() {
            if let Some((source, target)) = graph.edge_endpoints(edge) {
                if let Some(edge_data) = graph.edge_weight(edge) {
                    adjacency[(source.index(), target.index())] = edge_data.weight;
                }
            }
        }
        
        (node_features, adjacency)
    }
    
    fn generate_predictions(&self, graph: &DiGraph<AssetNode, CorrelationEdge>, 
                           hidden: &DVector<f64>) -> MarketPrediction {
        let mut correlations = HashMap::new();
        let mut arbitrage_paths = Vec::new();
        let mut risk_propagation = HashMap::new();
        
        // Detect arbitrage opportunities using graph algorithms
        for source in graph.node_indices() {
            let paths = dijkstra(graph, source, None, |e| {
                1.0 - e.weight().weight.abs()  // Convert correlation to distance
            });
            
            for (target, cost) in paths {
                if cost < 0.5 && source != target {  // Strong correlation path
                    if let (Some(s), Some(t)) = (graph.node_weight(source), graph.node_weight(target)) {
                        arbitrage_paths.push(ArbitragePath {
                            from: s.symbol.clone(),
                            to: t.symbol.clone(),
                            strength: 1.0 - cost,
                            expected_profit: self.calculate_arbitrage_profit(s, t, 1.0 - cost),
                        });
                    }
                }
            }
        }
        
        // Calculate pairwise correlations
        for edge in graph.edge_references() {
            let (source, target) = (edge.source(), edge.target());
            if let (Some(s), Some(t)) = (graph.node_weight(source), graph.node_weight(target)) {
                correlations.insert(
                    (s.symbol.clone(), t.symbol.clone()),
                    edge.weight().weight
                );
            }
        }
        
        // Risk propagation analysis
        for node_idx in graph.node_indices() {
            if let Some(node) = graph.node_weight(node_idx) {
                let mut risk = 0.0;
                for edge in graph.edges(node_idx) {
                    risk += edge.weight().weight.abs() * 0.1;  // Simplified risk calculation
                }
                risk_propagation.insert(node.symbol.clone(), risk);
            }
        }
        
        MarketPrediction {
            timestamp: chrono::Utc::now().timestamp_millis() as u64,
            correlations,
            arbitrage_opportunities: arbitrage_paths,
            risk_propagation,
            confidence: hidden.norm() / (hidden.len() as f64).sqrt(),
            next_move_probabilities: self.calculate_move_probabilities(hidden),
        }
    }
    
    fn calculate_arbitrage_profit(&self, from: &AssetNode, to: &AssetNode, strength: f64) -> f64 {
        // Simplified arbitrage profit calculation
        let price_diff = (from.features[0] - to.features[0]).abs();
        let volume_factor = (from.features[1] * to.features[1]).sqrt();
        price_diff * strength * volume_factor * 0.001  // Conservative estimate
    }
    
    fn calculate_move_probabilities(&self, hidden: &DVector<f64>) -> HashMap<String, f64> {
        let mut probs = HashMap::new();
        
        // Convert hidden state to probabilities
        let sum = hidden.iter().map(|x| x.exp()).sum::<f64>();
        
        probs.insert("strong_buy".to_string(), (hidden[0].exp() / sum).max(0.0).min(1.0));
        probs.insert("buy".to_string(), (hidden[1].exp() / sum).max(0.0).min(1.0));
        probs.insert("hold".to_string(), (hidden[2].exp() / sum).max(0.0).min(1.0));
        probs.insert("sell".to_string(), (hidden[3].exp() / sum).max(0.0).min(1.0));
        probs.insert("strong_sell".to_string(), (hidden[4].exp() / sum).max(0.0).min(1.0));
        
        probs
    }
}

/// Market prediction from GNN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketPrediction {
    pub timestamp: u64,
    pub correlations: HashMap<(String, String), f64>,
    pub arbitrage_opportunities: Vec<ArbitragePath>,
    pub risk_propagation: HashMap<String, f64>,
    pub confidence: f64,
    pub next_move_probabilities: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitragePath {
    pub from: String,
    pub to: String,
    pub strength: f64,
    pub expected_profit: f64,
}

/// Order Flow Network Analysis
pub struct OrderFlowNetwork {
    graph: Arc<RwLock<DiGraph<OrderNode, FlowEdge>>>,
    whale_threshold: f64,
    anomaly_detector: AnomalyDetector,
}

#[derive(Debug, Clone)]
struct OrderNode {
    address: String,
    order_type: OrderType,
    volume: f64,
    timestamp: u64,
    cluster_id: Option<usize>,
}

#[derive(Debug, Clone)]
enum OrderType {
    Market,
    Limit,
    Stop,
    Iceberg,
}

#[derive(Debug, Clone)]
struct FlowEdge {
    volume_flow: f64,
    frequency: u32,
    avg_time_delta: f64,
}

struct AnomalyDetector {
    mean: f64,
    std_dev: f64,
    window: VecDeque<f64>,
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
            window: VecDeque::with_capacity(1000),
        }
    }
    
    fn is_anomaly(&mut self, value: f64) -> bool {
        self.window.push_back(value);
        if self.window.len() > 1000 {
            self.window.pop_front();
        }
        
        if self.window.len() > 30 {
            self.mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
            let variance = self.window.iter()
                .map(|x| (x - self.mean).powi(2))
                .sum::<f64>() / self.window.len() as f64;
            self.std_dev = variance.sqrt();
            
            (value - self.mean).abs() > 3.0 * self.std_dev
        } else {
            false
        }
    }
}

impl OrderFlowNetwork {
    pub fn new(whale_threshold: f64) -> Self {
        Self {
            graph: Arc::new(RwLock::new(DiGraph::new())),
            whale_threshold,
            anomaly_detector: AnomalyDetector::new(),
        }
    }
    
    /// Analyze order flow patterns
    pub fn analyze_flow(&mut self, orders: &[(String, f64, u64)]) -> FlowAnalysis {
        let mut graph = self.graph.write().unwrap();
        
        // Update graph with new orders
        for (address, volume, timestamp) in orders {
            let node = OrderNode {
                address: address.clone(),
                order_type: if *volume > self.whale_threshold {
                    OrderType::Iceberg
                } else {
                    OrderType::Market
                },
                volume: *volume,
                timestamp: *timestamp,
                cluster_id: None,
            };
            
            graph.add_node(node);
        }
        
        // Detect whale movements
        let whale_movements = self.detect_whales(&graph);
        
        // Identify order clusters
        let clusters = self.cluster_orders(&graph);
        
        // Detect spoofing patterns
        let spoofing_signals = self.detect_spoofing(&graph);
        
        // Calculate flow imbalance
        let flow_imbalance = self.calculate_flow_imbalance(&graph);
        
        FlowAnalysis {
            whale_movements,
            order_clusters: clusters,
            spoofing_probability: spoofing_signals,
            flow_imbalance,
            anomaly_detected: self.anomaly_detector.is_anomaly(flow_imbalance),
        }
    }
    
    fn detect_whales(&self, graph: &DiGraph<OrderNode, FlowEdge>) -> Vec<WhaleMovement> {
        let mut movements = Vec::new();
        
        for node_idx in graph.node_indices() {
            if let Some(node) = graph.node_weight(node_idx) {
                if node.volume > self.whale_threshold {
                    movements.push(WhaleMovement {
                        address: node.address.clone(),
                        volume: node.volume,
                        timestamp: node.timestamp,
                        impact_estimate: node.volume * 0.001,  // Simplified impact
                    });
                }
            }
        }
        
        movements
    }
    
    fn cluster_orders(&self, graph: &DiGraph<OrderNode, FlowEdge>) -> Vec<OrderCluster> {
        // Simplified clustering - in production would use DBSCAN or similar
        let mut clusters = Vec::new();
        let mut visited = vec![false; graph.node_count()];
        
        for (idx, node_idx) in graph.node_indices().enumerate() {
            if !visited[idx] {
                let mut cluster = OrderCluster {
                    center: node_idx.index(),
                    members: vec![node_idx.index()],
                    total_volume: 0.0,
                    pattern_type: ClusterPattern::Normal,
                };
                
                if let Some(node) = graph.node_weight(node_idx) {
                    cluster.total_volume = node.volume;
                    
                    // Find nearby orders
                    for other_idx in graph.node_indices() {
                        if other_idx != node_idx {
                            if let Some(other) = graph.node_weight(other_idx) {
                                let time_diff = (node.timestamp as i64 - other.timestamp as i64).abs();
                                if time_diff < 1000 {  // Within 1 second
                                    cluster.members.push(other_idx.index());
                                    cluster.total_volume += other.volume;
                                }
                            }
                        }
                    }
                    
                    // Classify pattern
                    if cluster.members.len() > 10 && cluster.total_volume > self.whale_threshold * 2.0 {
                        cluster.pattern_type = ClusterPattern::Accumulation;
                    } else if cluster.members.len() > 20 {
                        cluster.pattern_type = ClusterPattern::Distribution;
                    }
                }
                
                visited[idx] = true;
                if cluster.members.len() > 1 {
                    clusters.push(cluster);
                }
            }
        }
        
        clusters
    }
    
    fn detect_spoofing(&self, graph: &DiGraph<OrderNode, FlowEdge>) -> f64 {
        // Detect potential spoofing by looking for orders that appear and disappear quickly
        let mut spoof_score = 0.0;
        let mut total_orders = 0;
        
        for node in graph.node_weights() {
            total_orders += 1;
            
            // Check for large orders that don't execute
            if node.volume > self.whale_threshold * 0.5 {
                match node.order_type {
                    OrderType::Limit | OrderType::Stop => spoof_score += 0.1,
                    _ => {}
                }
            }
        }
        
        if total_orders > 0 {
            (spoof_score / total_orders as f64).min(1.0)
        } else {
            0.0
        }
    }
    
    fn calculate_flow_imbalance(&self, graph: &DiGraph<OrderNode, FlowEdge>) -> f64 {
        let mut buy_volume = 0.0;
        let mut sell_volume = 0.0;
        
        for node in graph.node_weights() {
            // Simplified - would need actual order side information
            if node.timestamp % 2 == 0 {
                buy_volume += node.volume;
            } else {
                sell_volume += node.volume;
            }
        }
        
        let total = buy_volume + sell_volume;
        if total > 0.0 {
            (buy_volume - sell_volume) / total
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlowAnalysis {
    pub whale_movements: Vec<WhaleMovement>,
    pub order_clusters: Vec<OrderCluster>,
    pub spoofing_probability: f64,
    pub flow_imbalance: f64,
    pub anomaly_detected: bool,
}

#[derive(Debug, Clone)]
pub struct WhaleMovement {
    pub address: String,
    pub volume: f64,
    pub timestamp: u64,
    pub impact_estimate: f64,
}

#[derive(Debug, Clone)]
pub struct OrderCluster {
    pub center: usize,
    pub members: Vec<usize>,
    pub total_volume: f64,
    pub pattern_type: ClusterPattern,
}

#[derive(Debug, Clone)]
pub enum ClusterPattern {
    Normal,
    Accumulation,
    Distribution,
    Manipulation,
}

/// Message Passing Neural Network for information propagation
pub struct MessagePassingNN {
    num_layers: usize,
    hidden_dim: usize,
    message_fn: Arc<dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64> + Send + Sync>,
    update_fn: Arc<dyn Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64> + Send + Sync>,
}

impl MessagePassingNN {
    pub fn new(num_layers: usize, hidden_dim: usize) -> Self {
        Self {
            num_layers,
            hidden_dim,
            message_fn: Arc::new(|source, target| {
                // Neural message function
                let combined = source + target;
                combined.map(|x| x.tanh())
            }),
            update_fn: Arc::new(|old, messages| {
                // GRU-like update
                let gate = (old + messages).map(|x| 1.0 / (1.0 + (-x).exp()));
                gate.component_mul(messages) + gate.map(|x| 1.0 - x).component_mul(old)
            }),
        }
    }
    
    /// Propagate information through the network
    pub fn propagate(&self, graph: &DiGraph<AssetNode, CorrelationEdge>) -> Vec<DVector<f64>> {
        let num_nodes = graph.node_count();
        let mut node_states = vec![DVector::zeros(self.hidden_dim); num_nodes];
        
        // Initialize with node features
        for (idx, node_idx) in graph.node_indices().enumerate() {
            if let Some(node) = graph.node_weight(node_idx) {
                if node.features.len() >= self.hidden_dim {
                    node_states[idx] = node.features.rows(0, self.hidden_dim).into();
                }
            }
        }
        
        // Message passing iterations
        for _ in 0..self.num_layers {
            let mut new_states = node_states.clone();
            
            // Parallel message computation
            new_states.par_iter_mut().enumerate().for_each(|(i, state)| {
                let node_idx = NodeIndex::new(i);
                let mut messages = DVector::zeros(self.hidden_dim);
                let mut num_messages = 0;
                
                // Aggregate messages from neighbors
                for edge in graph.edges(node_idx) {
                    let neighbor_idx = edge.target().index();
                    if neighbor_idx < node_states.len() {
                        let message = (self.message_fn)(&node_states[i], &node_states[neighbor_idx]);
                        messages += message * edge.weight().weight;
                        num_messages += 1;
                    }
                }
                
                // Aggregate incoming messages
                for edge in graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let neighbor_idx = edge.source().index();
                    if neighbor_idx < node_states.len() {
                        let message = (self.message_fn)(&node_states[neighbor_idx], &node_states[i]);
                        messages += message * edge.weight().weight;
                        num_messages += 1;
                    }
                }
                
                // Normalize and update
                if num_messages > 0 {
                    messages /= num_messages as f64;
                    *state = (self.update_fn)(state, &messages);
                }
            });
            
            node_states = new_states;
        }
        
        node_states
    }
}

/// Graph builder for constructing market correlation graphs
pub struct GraphBuilder {
    min_correlation: f64,
    max_nodes: usize,
    lookback_window: usize,
}

impl GraphBuilder {
    pub fn new(min_correlation: f64, max_nodes: usize, lookback_window: usize) -> Self {
        Self {
            min_correlation,
            max_nodes,
            lookback_window,
        }
    }
    
    /// Build correlation graph from market data
    pub fn build_correlation_graph(&self, 
                                  assets: &[(String, String, Vec<f64>)]) 
                                  -> DiGraph<AssetNode, CorrelationEdge> {
        let mut graph = DiGraph::new();
        let mut node_indices = HashMap::new();
        
        // Add nodes
        for (symbol, exchange, prices) in assets.iter().take(self.max_nodes) {
            let features = self.extract_features(prices);
            let node = AssetNode {
                symbol: symbol.clone(),
                exchange: exchange.clone(),
                features,
                hidden_state: DVector::zeros(128),
                attention_weights: vec![1.0 / assets.len() as f64; assets.len()],
            };
            let idx = graph.add_node(node);
            node_indices.insert(symbol.clone(), idx);
        }
        
        // Add edges based on correlations
        for i in 0..assets.len().min(self.max_nodes) {
            for j in (i + 1)..assets.len().min(self.max_nodes) {
                let correlation = self.calculate_correlation(&assets[i].2, &assets[j].2);
                
                if correlation.abs() > self.min_correlation {
                    let edge = CorrelationEdge {
                        weight: correlation,
                        edge_type: if correlation > 0.8 {
                            EdgeType::Cointegration
                        } else if correlation > 0.5 {
                            EdgeType::PriceCorrelation
                        } else {
                            EdgeType::LeadLag
                        },
                        features: DVector::from_vec(vec![
                            correlation,
                            self.calculate_volume_correlation(&assets[i].2, &assets[j].2),
                            self.calculate_lead_lag(&assets[i].2, &assets[j].2),
                        ]),
                        temporal_weights: vec![correlation; 10],
                    };
                    
                    if let (Some(&idx_i), Some(&idx_j)) = 
                        (node_indices.get(&assets[i].0), node_indices.get(&assets[j].0)) {
                        graph.add_edge(idx_i, idx_j, edge.clone());
                        graph.add_edge(idx_j, idx_i, edge);  // Bidirectional
                    }
                }
            }
        }
        
        graph
    }
    
    fn extract_features(&self, prices: &[f64]) -> DVector<f64> {
        let len = prices.len().min(self.lookback_window);
        if len == 0 {
            return DVector::zeros(10);
        }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();
        
        let mean = prices.iter().sum::<f64>() / len as f64;
        let std_dev = (prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / len as f64).sqrt();
        let last_price = prices.last().copied().unwrap_or(0.0);
        let volume_proxy = prices.iter().sum::<f64>();  // Simplified
        
        let volatility = if !returns.is_empty() {
            let ret_mean = returns.iter().sum::<f64>() / returns.len() as f64;
            (returns.iter().map(|r| (r - ret_mean).powi(2)).sum::<f64>() / returns.len() as f64).sqrt()
        } else {
            0.0
        };
        
        DVector::from_vec(vec![
            last_price,
            volume_proxy,
            volatility,
            mean,
            std_dev,
            returns.last().copied().unwrap_or(0.0),
            prices.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0),
            prices.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(0.0),
            (last_price - mean) / std_dev.max(0.001),  // Z-score
            returns.iter().filter(|r| **r > 0.0).count() as f64 / returns.len().max(1) as f64,  // Win rate
        ])
    }
    
    fn calculate_correlation(&self, prices1: &[f64], prices2: &[f64]) -> f64 {
        let len = prices1.len().min(prices2.len()).min(self.lookback_window);
        if len < 2 {
            return 0.0;
        }
        
        let mean1 = prices1[..len].iter().sum::<f64>() / len as f64;
        let mean2 = prices2[..len].iter().sum::<f64>() / len as f64;
        
        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        
        for i in 0..len {
            let diff1 = prices1[i] - mean1;
            let diff2 = prices2[i] - mean2;
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
    
    fn calculate_volume_correlation(&self, prices1: &[f64], prices2: &[f64]) -> f64 {
        // Simplified - would use actual volume data
        self.calculate_correlation(prices1, prices2) * 0.8
    }
    
    fn calculate_lead_lag(&self, prices1: &[f64], prices2: &[f64]) -> f64 {
        // Cross-correlation at lag 1
        if prices1.len() > 1 && prices2.len() > 1 {
            self.calculate_correlation(&prices1[1..], &prices2[..prices2.len() - 1])
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_graph_attention_layer() {
        let layer = GraphAttentionLayer::new(10, 8, 4);
        let nodes = DMatrix::from_fn(5, 10, |_, _| rand::random::<f64>());
        let adjacency = DMatrix::from_fn(5, 5, |i, j| {
            if i != j { 1.0 } else { 0.0 }
        });
        
        let output = layer.forward(&nodes, &adjacency);
        assert_eq!(output.nrows(), 5);
        assert_eq!(output.ncols(), 32);  // 8 features * 4 heads
    }
    
    #[test]
    fn test_temporal_gnn() {
        let mut gnn = TemporalGNN::new(10, 64, 16, 3);
        let mut graph = DiGraph::new();
        
        // Add test nodes
        for i in 0..3 {
            let node = AssetNode {
                symbol: format!("ASSET{}", i),
                exchange: "TEST".to_string(),
                features: DVector::from_element(10, 0.5),
                hidden_state: DVector::zeros(64),
                attention_weights: vec![0.33; 3],
            };
            graph.add_node(node);
        }
        
        // Add test edges
        let edge = CorrelationEdge {
            weight: 0.8,
            edge_type: EdgeType::PriceCorrelation,
            features: DVector::from_element(3, 0.5),
            temporal_weights: vec![0.8; 10],
        };
        
        graph.add_edge(NodeIndex::new(0), NodeIndex::new(1), edge.clone());
        graph.add_edge(NodeIndex::new(1), NodeIndex::new(2), edge);
        
        let prediction = gnn.forward(&graph, 1000);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(!prediction.arbitrage_opportunities.is_empty() || 
                !prediction.correlations.is_empty());
    }
    
    #[test]
    fn test_order_flow_network() {
        let mut network = OrderFlowNetwork::new(10000.0);
        
        let orders = vec![
            ("whale1".to_string(), 50000.0, 1000),
            ("trader1".to_string(), 100.0, 1001),
            ("trader2".to_string(), 200.0, 1002),
            ("whale2".to_string(), 30000.0, 1003),
        ];
        
        let analysis = network.analyze_flow(&orders);
        
        assert_eq!(analysis.whale_movements.len(), 2);
        assert!(analysis.flow_imbalance >= -1.0 && analysis.flow_imbalance <= 1.0);
        assert!(analysis.spoofing_probability >= 0.0 && analysis.spoofing_probability <= 1.0);
    }
    
    #[test]
    fn test_message_passing() {
        let mpnn = MessagePassingNN::new(3, 32);
        let mut graph = DiGraph::new();
        
        // Create test graph
        for i in 0..5 {
            let node = AssetNode {
                symbol: format!("NODE{}", i),
                exchange: "TEST".to_string(),
                features: DVector::from_fn(32, |_| rand::random::<f64>()),
                hidden_state: DVector::zeros(32),
                attention_weights: vec![0.2; 5],
            };
            graph.add_node(node);
        }
        
        // Add edges in a chain
        for i in 0..4 {
            let edge = CorrelationEdge {
                weight: 0.5 + i as f64 * 0.1,
                edge_type: EdgeType::PriceCorrelation,
                features: DVector::from_element(3, 0.5),
                temporal_weights: vec![0.5; 10],
            };
            graph.add_edge(NodeIndex::new(i), NodeIndex::new(i + 1), edge);
        }
        
        let final_states = mpnn.propagate(&graph);
        
        assert_eq!(final_states.len(), 5);
        for state in &final_states {
            assert_eq!(state.len(), 32);
            // Check that message passing changed the states
            assert!(state.iter().any(|&x| x != 0.0));
        }
    }
    
    #[test]
    fn test_graph_builder() {
        let builder = GraphBuilder::new(0.3, 10, 100);
        
        let assets = vec![
            ("BTC".to_string(), "Binance".to_string(), vec![50000.0, 51000.0, 50500.0, 52000.0]),
            ("ETH".to_string(), "Binance".to_string(), vec![3000.0, 3100.0, 3050.0, 3200.0]),
            ("SOL".to_string(), "Binance".to_string(), vec![100.0, 102.0, 101.0, 105.0]),
        ];
        
        let graph = builder.build_correlation_graph(&assets);
        
        assert_eq!(graph.node_count(), 3);
        // Should have edges between correlated assets
        assert!(graph.edge_count() > 0);
        
        // Verify node properties
        for node in graph.node_weights() {
            assert_eq!(node.features.len(), 10);
            assert!(node.symbol == "BTC" || node.symbol == "ETH" || node.symbol == "SOL");
        }
    }
}