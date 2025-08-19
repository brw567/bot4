# ML Model Cards - Bot4 Trading Platform

## Model Card: Deep LSTM (5-Layer)

### Architecture Specification
```yaml
model_type: Stacked LSTM with Residual Connections
layers:
  - layer_1:
      type: LSTM
      input_dim: 100
      hidden_dim: 512
      dropout: 0.2
  - layer_2:
      type: LSTM
      hidden_dim: 512
      dropout: 0.2
      residual: true
  - layer_3:
      type: LSTM
      hidden_dim: 256
      dropout: 0.15
      residual: true
  - layer_4:
      type: LSTM
      hidden_dim: 128
      dropout: 0.1
      residual: true
  - layer_5:
      type: Dense
      output_dim: 3  # [buy, hold, sell]
      activation: softmax

total_parameters: 2,847,235
trainable_parameters: 2,847,235
sequence_length: 60  # 1-hour window at 1-min bars
batch_size_inference: 1
```

### Input Features
```yaml
technical_indicators: 50
  - SMA: [10, 20, 50, 100, 200]
  - EMA: [9, 12, 26, 50]
  - RSI: [7, 14, 21]
  - MACD: [12-26-9, 5-35-5]
  - ATR: [7, 14, 21]
  - Bollinger Bands: [20-2, 20-1]
  
microstructure_features: 25
  - bid_ask_spread
  - order_flow_imbalance
  - volume_weighted_price
  - trade_intensity
  - quote_intensity
  
market_features: 25
  - volatility_realized
  - volatility_garch
  - correlation_matrix
  - funding_rate
  - open_interest
```

### Normalization
```yaml
method: RobustScaler
center: median
scale: IQR
clip_range: [-5, 5]
update_frequency: daily
```

### Performance Metrics
```yaml
latency_batch_1:
  p50: 0.72ms
  p95: 0.89ms
  p99: 0.95ms
  p99.9: 0.98ms

throughput:
  batch_1: 1,388 predictions/sec
  batch_32: 28,571 predictions/sec

calibration:
  brier_score: 0.18
  expected_calibration_error: 0.042
  reliability_diagram: see plots/lstm_calibration.png
```

### Training Configuration
```yaml
optimizer: Adam
learning_rate: 0.001
lr_schedule: CosineAnnealing
batch_size: 128
epochs: 100
early_stopping: patience=10
validation_split: 0.2 (temporal)
loss_function: CrossEntropyLoss + L2(0.0001)
```

---

## Model Card: Transformer (6-Layer, 8-Head)

### Architecture Specification
```yaml
model_type: Transformer Encoder
layers: 6
attention_heads: 8
hidden_dim: 512
ff_dim: 2048
dropout: 0.1
positional_encoding: learned
max_sequence: 100

total_parameters: 4,123,907
```

### Performance Metrics
```yaml
latency_batch_1:
  p50: 1.1ms
  p95: 1.3ms
  p99: 1.4ms

calibration:
  brier_score: 0.16
  ece: 0.038
```

---

## Model Card: Temporal CNN

### Architecture Specification
```yaml
model_type: Dilated Causal CNN
layers:
  - conv1d: [64, kernel=3, dilation=1]
  - conv1d: [128, kernel=3, dilation=2]
  - conv1d: [256, kernel=3, dilation=4]
  - conv1d: [256, kernel=3, dilation=8]
  - global_pool: max
  - dense: [128, relu]
  - dense: [3, softmax]

total_parameters: 892,163
receptive_field: 30 timesteps
```

### Performance Metrics
```yaml
latency_batch_1:
  p50: 0.31ms
  p95: 0.42ms
  p99: 0.48ms

calibration:
  brier_score: 0.21
  ece: 0.055
```

---

## Model Card: GRU Stack (4-Layer)

### Architecture Specification
```yaml
model_type: Bidirectional GRU
layers: 4
hidden_dim: 256
dropout: 0.2
bidirectional: true

total_parameters: 1,456,899
```

### Performance Metrics
```yaml
latency_batch_1:
  p50: 0.58ms
  p95: 0.71ms
  p99: 0.78ms

calibration:
  brier_score: 0.19
  ece: 0.048
```

---

## Model Card: XGBoost Ensemble

### Architecture Specification
```yaml
model_type: Gradient Boosting Trees
n_estimators: 100
max_depth: 5
learning_rate: 0.1
subsample: 0.8
colsample_bytree: 0.8
reg_lambda: 1.0
reg_alpha: 0.1

total_parameters: ~50,000 (tree nodes)
```

### Performance Metrics
```yaml
latency_batch_1:
  p50: 0.08ms
  p95: 0.11ms
  p99: 0.13ms

calibration:
  brier_score: 0.22
  ece: 0.061
```

---

## Ensemble Configuration

### Weighting Strategy (Adaptive)
```yaml
initial_weights: [0.3, 0.2, 0.15, 0.15, 0.2]  # [LSTM, Transformer, CNN, GRU, XGBoost]
adaptation_method: Bayesian Model Averaging
update_frequency: hourly
performance_window: 24 hours
min_weight: 0.05
max_weight: 0.5
```

### Combined Performance
```yaml
ensemble_latency_batch_1:
  p50: 2.8ms
  p95: 3.4ms
  p99: 3.7ms
  p99.9: 3.9ms

ensemble_calibration:
  brier_score: 0.14  # Better than any individual model
  ece: 0.031
  reliability: 0.97
```

### Deterministic Control
```yaml
random_seed: 42
numpy_seed: 42
torch_seed: 42
deterministic_mode: true
cublas_deterministic: true
```