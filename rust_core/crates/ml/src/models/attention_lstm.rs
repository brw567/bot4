// Attention-Enhanced LSTM with AVX-512 Optimization
// Morgan (ML Lead) + Jordan (Performance) + Full Team
// References: Bahdanau (2014), Vaswani (2017), Informer (2021), Portfolio Transformer (2022)
// CRITICAL: Sophia Requirement #6 - Attention mechanism for temporal patterns

use std::arch::x86_64::*;
use ndarray::{Array1, Array2, Array3, Axis, s};
use std::collections::VecDeque;
use rand::prelude::*;
use rand_distr::{Normal, Distribution};

const XAVIER_SCALE: f32 = 1.0;  // Xavier initialization scale

/// Attention-Enhanced LSTM for Financial Time Series
/// Morgan: "Combines LSTM's memory with Transformer's attention - best of both worlds!"
#[derive(Debug, Clone)]
pub struct AttentionLSTM {
    // Architecture parameters
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    dropout_rate: f32,
    
    // LSTM weights (per layer)
    lstm_layers: Vec<LSTMLayer>,
    
    // Multi-head attention
    attention: MultiHeadAttention,
    
    // Output projection
    output_projection: Array2<f32>,
    output_bias: Array1<f32>,
    
    // Layer normalization
    layer_norms: Vec<LayerNorm>,
    
    // Residual connections
    use_residual: bool,
    
    // Optimization flags
    use_avx512: bool,
    
    // State tracking
    hidden_states: Vec<Array2<f32>>,
    cell_states: Vec<Array2<f32>>,
    
    // Performance metrics
    inference_times: VecDeque<u64>,
}

/// Single LSTM Layer with forget gate optimization
#[derive(Debug, Clone)]
struct LSTMLayer {
    // Input gate
    w_ii: Array2<f32>,  // input -> input gate
    w_hi: Array2<f32>,  // hidden -> input gate
    b_i: Array1<f32>,   // input gate bias
    
    // Forget gate
    w_if: Array2<f32>,  // input -> forget gate
    w_hf: Array2<f32>,  // hidden -> forget gate
    b_f: Array1<f32>,   // forget gate bias
    
    // Cell gate
    w_ig: Array2<f32>,  // input -> cell gate
    w_hg: Array2<f32>,  // hidden -> cell gate
    b_g: Array1<f32>,   // cell gate bias
    
    // Output gate
    w_io: Array2<f32>,  // input -> output gate
    w_ho: Array2<f32>,  // hidden -> output gate
    b_o: Array1<f32>,   // output gate bias
}

/// Multi-Head Scaled Dot-Product Attention
#[derive(Debug, Clone)]
struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    
    // Projections for Q, K, V
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>,  // Output projection
    
    // Positional encoding
    positional_encoding: Array2<f32>,
    
    // Temperature for scaling
    temperature: f32,
}

/// Layer Normalization
#[derive(Debug, Clone)]
struct LayerNorm {
    gamma: Array1<f32>,
    beta: Array1<f32>,
    epsilon: f32,
}

impl AttentionLSTM {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        let use_avx512 = is_x86_feature_detected!("avx512f") 
                      && is_x86_feature_detected!("avx512dq");
        
        // Initialize LSTM layers
        let mut lstm_layers = Vec::new();
        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            lstm_layers.push(LSTMLayer::new(layer_input_size, hidden_size));
        }
        
        // Initialize attention
        let attention = MultiHeadAttention::new(hidden_size, num_heads);
        
        // Initialize layer norms
        let mut layer_norms = Vec::new();
        for _ in 0..num_layers {
            layer_norms.push(LayerNorm::new(hidden_size));
        }
        
        // Output projection
        let output_projection = Self::xavier_init(hidden_size, input_size);
        let output_bias = Array1::zeros(input_size);
        
        // Initialize states
        let hidden_states = vec![Array2::zeros((1, hidden_size)); num_layers];
        let cell_states = vec![Array2::zeros((1, hidden_size)); num_layers];
        
        Self {
            input_size,
            hidden_size,
            num_layers,
            num_heads,
            dropout_rate: 0.1,
            lstm_layers,
            attention,
            output_projection,
            output_bias,
            layer_norms,
            use_residual: true,
            use_avx512,
            hidden_states,
            cell_states,
            inference_times: VecDeque::with_capacity(100),
        }
    }
    
    /// Forward pass with attention mechanism
    /// Jordan: "AVX-512 accelerates both LSTM and attention computations!"
    pub fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let start = std::time::Instant::now();
        let (batch_size, seq_len) = (input.nrows(), input.ncols() / self.input_size);
        
        // Reshape input to (batch, seq_len, features)
        let input_3d = input.clone().into_shape((batch_size, seq_len, self.input_size))
            .expect("Invalid input shape");
        
        // Process through LSTM layers
        let mut lstm_output = Array3::zeros((batch_size, seq_len, self.hidden_size));
        
        for t in 0..seq_len {
            let x_t = input_3d.slice(s![.., t, ..]).to_owned();
            let mut layer_input = x_t;
            
            for (layer_idx, lstm_layer) in self.lstm_layers.iter().enumerate() {
                let (h_new, c_new) = if self.use_avx512 && self.hidden_size >= 16 {
                    unsafe {
                        self.lstm_step_avx512(
                            &layer_input,
                            &self.hidden_states[layer_idx],
                            &self.cell_states[layer_idx],
                            lstm_layer,
                        )
                    }
                } else {
                    self.lstm_step_scalar(
                        &layer_input,
                        &self.hidden_states[layer_idx],
                        &self.cell_states[layer_idx],
                        lstm_layer,
                    )
                };
                
                // Update states
                self.hidden_states[layer_idx] = h_new.clone();
                self.cell_states[layer_idx] = c_new;
                
                // Apply layer norm
                layer_input = self.layer_norms[layer_idx].forward(&h_new);
                
                // Store output from last layer
                if layer_idx == self.num_layers - 1 {
                    lstm_output.slice_mut(s![.., t, ..]).assign(&layer_input);
                }
            }
        }
        
        // Apply multi-head attention
        let attended = self.attention.forward(&lstm_output);
        
        // Residual connection
        let output = if self.use_residual {
            &lstm_output + &attended
        } else {
            attended
        };
        
        // Final projection
        let final_output = self.project_output(&output);
        
        // Track performance
        self.inference_times.push_back(start.elapsed().as_nanos() as u64);
        if self.inference_times.len() > 100 {
            self.inference_times.pop_front();
        }
        
        final_output.into_shape((batch_size, seq_len * self.input_size))
            .expect("Output reshape failed")
    }
    
    /// LSTM step with AVX-512 optimization
    unsafe fn lstm_step_avx512(
        &self,
        x: &Array2<f32>,
        h_prev: &Array2<f32>,
        c_prev: &Array2<f32>,
        layer: &LSTMLayer,
    ) -> (Array2<f32>, Array2<f32>) {
        let batch_size = x.nrows();
        let hidden_size = self.hidden_size;
        
        let mut h_new = Array2::zeros((batch_size, hidden_size));
        let mut c_new = Array2::zeros((batch_size, hidden_size));
        
        // Process in chunks of 16 for AVX-512
        for b in 0..batch_size {
            for h_chunk in (0..hidden_size).step_by(16) {
                let chunk_size = (hidden_size - h_chunk).min(16);
                
                if chunk_size == 16 {
                    // Load previous states
                    let h_prev_vec = _mm512_loadu_ps(&h_prev[[b, h_chunk]]);
                    let c_prev_vec = _mm512_loadu_ps(&c_prev[[b, h_chunk]]);
                    
                    // Input gate: i = σ(W_ii @ x + W_hi @ h + b_i)
                    let mut i_gate = _mm512_setzero_ps();
                    for i in 0..x.ncols() {
                        let x_val = _mm512_set1_ps(x[[b, i]]);
                        let w_ii = _mm512_loadu_ps(&layer.w_ii[[i, h_chunk]]);
                        i_gate = _mm512_fmadd_ps(x_val, w_ii, i_gate);
                    }
                    let w_hi = _mm512_loadu_ps(&layer.w_hi[[0, h_chunk]]);
                    i_gate = _mm512_fmadd_ps(h_prev_vec, w_hi, i_gate);
                    let b_i = _mm512_loadu_ps(&layer.b_i[h_chunk]);
                    i_gate = _mm512_add_ps(i_gate, b_i);
                    i_gate = self.sigmoid_avx512(i_gate);
                    
                    // Forget gate: f = σ(W_if @ x + W_hf @ h + b_f)
                    let mut f_gate = _mm512_setzero_ps();
                    for i in 0..x.ncols() {
                        let x_val = _mm512_set1_ps(x[[b, i]]);
                        let w_if = _mm512_loadu_ps(&layer.w_if[[i, h_chunk]]);
                        f_gate = _mm512_fmadd_ps(x_val, w_if, f_gate);
                    }
                    let w_hf = _mm512_loadu_ps(&layer.w_hf[[0, h_chunk]]);
                    f_gate = _mm512_fmadd_ps(h_prev_vec, w_hf, f_gate);
                    let b_f = _mm512_loadu_ps(&layer.b_f[h_chunk]);
                    f_gate = _mm512_add_ps(f_gate, b_f);
                    f_gate = self.sigmoid_avx512(f_gate);
                    
                    // Cell gate: g = tanh(W_ig @ x + W_hg @ h + b_g)
                    let mut g_gate = _mm512_setzero_ps();
                    for i in 0..x.ncols() {
                        let x_val = _mm512_set1_ps(x[[b, i]]);
                        let w_ig = _mm512_loadu_ps(&layer.w_ig[[i, h_chunk]]);
                        g_gate = _mm512_fmadd_ps(x_val, w_ig, g_gate);
                    }
                    let w_hg = _mm512_loadu_ps(&layer.w_hg[[0, h_chunk]]);
                    g_gate = _mm512_fmadd_ps(h_prev_vec, w_hg, g_gate);
                    let b_g = _mm512_loadu_ps(&layer.b_g[h_chunk]);
                    g_gate = _mm512_add_ps(g_gate, b_g);
                    g_gate = self.tanh_avx512(g_gate);
                    
                    // Output gate: o = σ(W_io @ x + W_ho @ h + b_o)
                    let mut o_gate = _mm512_setzero_ps();
                    for i in 0..x.ncols() {
                        let x_val = _mm512_set1_ps(x[[b, i]]);
                        let w_io = _mm512_loadu_ps(&layer.w_io[[i, h_chunk]]);
                        o_gate = _mm512_fmadd_ps(x_val, w_io, o_gate);
                    }
                    let w_ho = _mm512_loadu_ps(&layer.w_ho[[0, h_chunk]]);
                    o_gate = _mm512_fmadd_ps(h_prev_vec, w_ho, o_gate);
                    let b_o = _mm512_loadu_ps(&layer.b_o[h_chunk]);
                    o_gate = _mm512_add_ps(o_gate, b_o);
                    o_gate = self.sigmoid_avx512(o_gate);
                    
                    // New cell state: c_t = f * c_{t-1} + i * g
                    let c_forget = _mm512_mul_ps(f_gate, c_prev_vec);
                    let c_input = _mm512_mul_ps(i_gate, g_gate);
                    let c_new_vec = _mm512_add_ps(c_forget, c_input);
                    
                    // New hidden state: h_t = o * tanh(c_t)
                    let c_tanh = self.tanh_avx512(c_new_vec);
                    let h_new_vec = _mm512_mul_ps(o_gate, c_tanh);
                    
                    // Store results
                    _mm512_storeu_ps(&mut c_new[[b, h_chunk]] as *mut f32, c_new_vec);
                    _mm512_storeu_ps(&mut h_new[[b, h_chunk]] as *mut f32, h_new_vec);
                } else {
                    // Handle remaining elements
                    for h in h_chunk..h_chunk + chunk_size {
                        // Scalar fallback for remainder
                        let (h_val, c_val) = self.lstm_cell_scalar(
                            x.row(b).to_owned(),
                            h_prev[[b, h]],
                            c_prev[[b, h]],
                            layer,
                            h,
                        );
                        h_new[[b, h]] = h_val;
                        c_new[[b, h]] = c_val;
                    }
                }
            }
        }
        
        (h_new, c_new)
    }
    
    /// Scalar LSTM step (fallback)
    fn lstm_step_scalar(
        &self,
        x: &Array2<f32>,
        h_prev: &Array2<f32>,
        c_prev: &Array2<f32>,
        layer: &LSTMLayer,
    ) -> (Array2<f32>, Array2<f32>) {
        let batch_size = x.nrows();
        let hidden_size = self.hidden_size;
        
        let mut h_new = Array2::zeros((batch_size, hidden_size));
        let mut c_new = Array2::zeros((batch_size, hidden_size));
        
        for b in 0..batch_size {
            for h in 0..hidden_size {
                let (h_val, c_val) = self.lstm_cell_scalar(
                    x.row(b).to_owned(),
                    h_prev[[b, h]],
                    c_prev[[b, h]],
                    layer,
                    h,
                );
                h_new[[b, h]] = h_val;
                c_new[[b, h]] = c_val;
            }
        }
        
        (h_new, c_new)
    }
    
    /// Single LSTM cell computation
    fn lstm_cell_scalar(
        &self,
        x: Array1<f32>,
        h_prev: f32,
        c_prev: f32,
        layer: &LSTMLayer,
        hidden_idx: usize,
    ) -> (f32, f32) {
        // Input gate
        let i = Self::sigmoid(
            x.dot(&layer.w_ii.column(hidden_idx)) +
            h_prev * layer.w_hi[[0, hidden_idx]] +
            layer.b_i[hidden_idx]
        );
        
        // Forget gate
        let f = Self::sigmoid(
            x.dot(&layer.w_if.column(hidden_idx)) +
            h_prev * layer.w_hf[[0, hidden_idx]] +
            layer.b_f[hidden_idx]
        );
        
        // Cell gate
        let g = Self::tanh(
            x.dot(&layer.w_ig.column(hidden_idx)) +
            h_prev * layer.w_hg[[0, hidden_idx]] +
            layer.b_g[hidden_idx]
        );
        
        // Output gate
        let o = Self::sigmoid(
            x.dot(&layer.w_io.column(hidden_idx)) +
            h_prev * layer.w_ho[[0, hidden_idx]] +
            layer.b_o[hidden_idx]
        );
        
        // Update cell and hidden states
        let c_new = f * c_prev + i * g;
        let h_new = o * Self::tanh(c_new);
        
        (h_new, c_new)
    }
    
    /// AVX-512 sigmoid approximation
    unsafe fn sigmoid_avx512(&self, x: __m512) -> __m512 {
        // Fast sigmoid: 1 / (1 + exp(-x))
        // Using approximation for speed
        let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        let exp_neg_x = self.exp_approx_avx512(neg_x);
        let one = _mm512_set1_ps(1.0);
        let denom = _mm512_add_ps(one, exp_neg_x);
        _mm512_div_ps(one, denom)
    }
    
    /// AVX-512 tanh approximation
    unsafe fn tanh_avx512(&self, x: __m512) -> __m512 {
        // Fast tanh using sigmoid: tanh(x) = 2*sigmoid(2*x) - 1
        let two = _mm512_set1_ps(2.0);
        let one = _mm512_set1_ps(1.0);
        let two_x = _mm512_mul_ps(two, x);
        let sig = self.sigmoid_avx512(two_x);
        _mm512_sub_ps(_mm512_mul_ps(two, sig), one)
    }
    
    /// Fast exponential approximation for AVX-512
    unsafe fn exp_approx_avx512(&self, x: __m512) -> __m512 {
        // Using polynomial approximation for exp(x)
        // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24
        let one = _mm512_set1_ps(1.0);
        let x2 = _mm512_mul_ps(x, x);
        let x3 = _mm512_mul_ps(x2, x);
        let x4 = _mm512_mul_ps(x3, x);
        
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(1.0 / 6.0);
        let c4 = _mm512_set1_ps(1.0 / 24.0);
        
        let term2 = _mm512_mul_ps(x2, c2);
        let term3 = _mm512_mul_ps(x3, c3);
        let term4 = _mm512_mul_ps(x4, c4);
        
        let sum = _mm512_add_ps(one, x);
        let sum = _mm512_add_ps(sum, term2);
        let sum = _mm512_add_ps(sum, term3);
        _mm512_add_ps(sum, term4)
    }
    
    /// Scalar activation functions
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    fn tanh(x: f32) -> f32 {
        x.tanh()
    }
    
    /// Project output to original dimension
    fn project_output(&self, x: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, _) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, self.input_size));
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let hidden = x.slice(s![b, t, ..]);
                let projected = hidden.dot(&self.output_projection.t()) + &self.output_bias;
                output.slice_mut(s![b, t, ..]).assign(&projected);
            }
        }
        
        output
    }
    
    /// Xavier initialization
    fn xavier_init(fan_in: usize, fan_out: usize) -> Array2<f32> {
        let mut rng = thread_rng();
        let scale = (2.0 / (fan_in + fan_out) as f32).sqrt() * XAVIER_SCALE;
        let normal = Normal::new(0.0, scale).unwrap();
        
        Array2::from_shape_fn((fan_in, fan_out), |_| normal.sample(&mut rng))
    }
    
    /// Get average inference time
    pub fn avg_inference_time_ns(&self) -> u64 {
        if self.inference_times.is_empty() {
            0
        } else {
            self.inference_times.iter().sum::<u64>() / self.inference_times.len() as u64
        }
    }
}

impl LSTMLayer {
    fn new(input_size: usize, hidden_size: usize) -> Self {
        let scale_i = (1.0 / input_size as f32).sqrt();
        let scale_h = (1.0 / hidden_size as f32).sqrt();
        
        Self {
            w_ii: AttentionLSTM::xavier_init(input_size, hidden_size) * scale_i,
            w_hi: AttentionLSTM::xavier_init(hidden_size, hidden_size) * scale_h,
            b_i: Array1::zeros(hidden_size),
            
            w_if: AttentionLSTM::xavier_init(input_size, hidden_size) * scale_i,
            w_hf: AttentionLSTM::xavier_init(hidden_size, hidden_size) * scale_h,
            b_f: Array1::ones(hidden_size),  // Forget gate bias = 1 (remember by default)
            
            w_ig: AttentionLSTM::xavier_init(input_size, hidden_size) * scale_i,
            w_hg: AttentionLSTM::xavier_init(hidden_size, hidden_size) * scale_h,
            b_g: Array1::zeros(hidden_size),
            
            w_io: AttentionLSTM::xavier_init(input_size, hidden_size) * scale_i,
            w_ho: AttentionLSTM::xavier_init(hidden_size, hidden_size) * scale_h,
            b_o: Array1::zeros(hidden_size),
        }
    }
}

impl MultiHeadAttention {
    fn new(hidden_size: usize, num_heads: usize) -> Self {
        assert!(hidden_size % num_heads == 0, "Hidden size must be divisible by num_heads");
        
        let head_dim = hidden_size / num_heads;
        let temperature = (head_dim as f32).sqrt();
        
        // Initialize projections
        let w_q = AttentionLSTM::xavier_init(hidden_size, hidden_size);
        let w_k = AttentionLSTM::xavier_init(hidden_size, hidden_size);
        let w_v = AttentionLSTM::xavier_init(hidden_size, hidden_size);
        let w_o = AttentionLSTM::xavier_init(hidden_size, hidden_size);
        
        // Positional encoding (sinusoidal)
        let max_len = 1000;
        let mut positional_encoding = Array2::zeros((max_len, hidden_size));
        for pos in 0..max_len {
            for i in 0..hidden_size {
                let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / hidden_size as f32);
                if i % 2 == 0 {
                    positional_encoding[[pos, i]] = angle.sin();
                } else {
                    positional_encoding[[pos, i]] = angle.cos();
                }
            }
        }
        
        Self {
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            positional_encoding,
            temperature,
        }
    }
    
    /// Multi-head attention forward pass
    fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, hidden_size) = x.dim();
        
        // Add positional encoding
        let mut x_pos = x.clone();
        for t in 0..seq_len.min(1000) {
            for b in 0..batch_size {
                let pos_enc = self.positional_encoding.row(t);
                x_pos.slice_mut(s![b, t, ..])
                    .zip_mut_with(&pos_enc, |x, &p| *x += p);
            }
        }
        
        // Linear projections in batch
        let q = self.project(&x_pos, &self.w_q);
        let k = self.project(&x_pos, &self.w_k);
        let v = self.project(&x_pos, &self.w_v);
        
        // Reshape for multi-head attention
        let q = self.reshape_for_attention(&q);
        let k = self.reshape_for_attention(&k);
        let v = self.reshape_for_attention(&v);
        
        // Scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(&q, &k, &v);
        
        // Concatenate heads
        let concat = self.concat_heads(&attention_output);
        
        // Final projection
        self.project(&concat, &self.w_o)
    }
    
    /// Project input using weight matrix
    fn project(&self, x: &Array3<f32>, w: &Array2<f32>) -> Array3<f32> {
        let (batch_size, seq_len, hidden_size) = x.dim();
        let mut output = Array3::zeros((batch_size, seq_len, hidden_size));
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let x_vec = x.slice(s![b, t, ..]);
                let projected = x_vec.dot(w);
                output.slice_mut(s![b, t, ..]).assign(&projected);
            }
        }
        
        output
    }
    
    /// Reshape for multi-head attention
    fn reshape_for_attention(&self, x: &Array3<f32>) -> Array3<f32> {
        let (batch_size, seq_len, hidden_size) = x.dim();
        
        // Reshape to (batch * num_heads, seq_len, head_dim)
        x.clone()
            .into_shape((batch_size * self.num_heads, seq_len, self.head_dim))
            .expect("Reshape for attention failed")
    }
    
    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        v: &Array3<f32>,
    ) -> Array3<f32> {
        let (batch_heads, seq_len, head_dim) = q.dim();
        
        // Q @ K^T / sqrt(d_k)
        let mut scores = Array3::zeros((batch_heads, seq_len, seq_len));
        for bh in 0..batch_heads {
            let q_batch = q.slice(s![bh, .., ..]);
            let k_batch = k.slice(s![bh, .., ..]);
            
            let scores_batch = q_batch.dot(&k_batch.t()) / self.temperature;
            scores.slice_mut(s![bh, .., ..]).assign(&scores_batch);
        }
        
        // Softmax
        let attention_weights = self.softmax(&scores);
        
        // Attention @ V
        let mut output = Array3::zeros((batch_heads, seq_len, head_dim));
        for bh in 0..batch_heads {
            let weights = attention_weights.slice(s![bh, .., ..]);
            let v_batch = v.slice(s![bh, .., ..]);
            
            let out = weights.dot(&v_batch);
            output.slice_mut(s![bh, .., ..]).assign(&out);
        }
        
        output
    }
    
    /// Softmax along last dimension
    fn softmax(&self, x: &Array3<f32>) -> Array3<f32> {
        let mut output = x.clone();
        
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                let row = x.slice(s![i, j, ..]);
                let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                
                let exp_sum: f32 = row.iter()
                    .map(|&val| (val - max).exp())
                    .sum();
                
                for k in 0..x.shape()[2] {
                    output[[i, j, k]] = ((x[[i, j, k]] - max).exp()) / exp_sum;
                }
            }
        }
        
        output
    }
    
    /// Concatenate attention heads
    fn concat_heads(&self, x: &Array3<f32>) -> Array3<f32> {
        let (batch_heads, seq_len, head_dim) = x.dim();
        let batch_size = batch_heads / self.num_heads;
        let hidden_size = self.num_heads * head_dim;
        
        x.clone()
            .into_shape((batch_size, seq_len, hidden_size))
            .expect("Concat heads failed")
    }
}

impl LayerNorm {
    fn new(hidden_size: usize) -> Self {
        Self {
            gamma: Array1::ones(hidden_size),
            beta: Array1::zeros(hidden_size),
            epsilon: 1e-5,
        }
    }
    
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mean = x.mean_axis(Axis(1))
            .expect("Mean calculation failed")
            .insert_axis(Axis(1));
        
        let variance = x.map(|v| v.powi(2))
            .mean_axis(Axis(1))
            .expect("Variance calculation failed")
            .insert_axis(Axis(1))
            - &mean.map(|v| v.powi(2));
        
        let normalized = (x - &mean) / variance.map(|v| (v + self.epsilon).sqrt());
        
        &normalized * &self.gamma + &self.beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_lstm_forward() {
        let mut model = AttentionLSTM::new(10, 32, 2, 4);
        
        // Test input (batch=2, features=10)
        let input = Array2::from_shape_fn((2, 10), |_| rand::random::<f32>());
        
        let output = model.forward(&input);
        
        assert_eq!(output.shape(), &[2, 10]);
        assert!(output.iter().all(|v| v.is_finite()));
    }
    
    #[test]
    fn test_multi_head_attention() {
        let attention = MultiHeadAttention::new(64, 8);
        
        // Test input (batch=2, seq_len=10, hidden=64)
        let input = Array3::from_shape_fn((2, 10, 64), |_| rand::random::<f32>());
        
        let output = attention.forward(&input);
        
        assert_eq!(output.shape(), &[2, 10, 64]);
        assert!(output.iter().all(|v| v.is_finite()));
    }
    
    #[test]
    fn test_layer_norm() {
        let layer_norm = LayerNorm::new(32);
        
        let input = Array2::from_shape_fn((4, 32), |_| rand::random::<f32>() * 10.0);
        let output = layer_norm.forward(&input);
        
        // Check normalization
        let mean = output.mean_axis(Axis(1)).unwrap();
        let var = output.map(|v| v.powi(2)).mean_axis(Axis(1)).unwrap()
            - mean.map(|v| v.powi(2));
        
        for m in mean.iter() {
            assert!(m.abs() < 0.1, "Mean should be near 0");
        }
        
        for v in var.iter() {
            assert!((v - 1.0).abs() < 0.1, "Variance should be near 1");
        }
    }
    
    #[test]
    fn test_avx512_consistency() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping AVX-512 test on non-supporting hardware");
            return;
        }
        
        let mut model_avx = AttentionLSTM::new(8, 16, 1, 2);
        model_avx.use_avx512 = true;
        
        let mut model_scalar = AttentionLSTM::new(8, 16, 1, 2);
        model_scalar.use_avx512 = false;
        
        // Use same weights
        model_scalar.lstm_layers = model_avx.lstm_layers.clone();
        
        let input = Array2::from_shape_fn((1, 8), |_| rand::random::<f32>());
        
        let output_avx = model_avx.forward(&input);
        let output_scalar = model_scalar.forward(&input);
        
        // Check outputs are similar
        for (a, s) in output_avx.iter().zip(output_scalar.iter()) {
            assert!((a - s).abs() < 1e-4, "AVX and scalar outputs should match");
        }
    }
}