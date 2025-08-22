#!/bin/bash

# Fix remaining ML variable issues
# Team: Morgan (ML Lead) + Jordan (Performance)

set -e

echo "Fixing remaining ML variable issues..."

# Fix indicators.rs SIMD issues
sed -i 's/_mm256_mul_ps(_prices,/_mm256_mul_ps(prices,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_mul_ps(_ema_vec,/_mm256_mul_ps(ema_vec,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_add_ps(_weighted_price,/_mm256_add_ps(weighted_price,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_extractf128_ps(_v,/_mm256_extractf128_ps(v,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm_add_ps(_high,/_mm_add_ps(high,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm_shuffle_ps(_sum64,/_mm_shuffle_ps(sum64,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_sub_ps(_curr,/_mm256_sub_ps(curr,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_max_ps(_change,/_mm256_max_ps(change,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_sub_ps(_zero,/_mm256_sub_ps(zero,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_add_ps(_gains,/_mm256_add_ps(gains,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm256_add_ps(_losses,/_mm256_add_ps(losses,/g' crates/ml/src/feature_engine/indicators.rs

# Fix type issues
sed -i 's/(_f64, f64)/(f64, f64)/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/if let Some((_min, max))/if let Some((min, max))/g' crates/ml/src/feature_engine/indicators.rs

# Fix calculate calls
sed -i 's/\.calculate(_data,/\.calculate(data,/g' crates/ml/src/**/*.rs
sed -i 's/\.calculate(_candles,/\.calculate(candles,/g' crates/ml/src/**/*.rs

# Fix cache_key
sed -i 's/self.cache.insert(_cache_key,/self.cache.insert(cache_key,/g' crates/ml/src/**/*.rs

# Fix pipeline.rs loop variables
sed -i 's/for (_name, value)/for (name, value)/g' crates/ml/src/feature_engine/pipeline.rs

# Fix scaler transform calls
sed -i 's/self.transform_minmax(_features,/self.transform_minmax(features,/g' crates/ml/src/**/*.rs
sed -i 's/scaled_value.clamp(_target_min,/scaled_value.clamp(target_min,/g' crates/ml/src/**/*.rs

# Fix selector type issues
sed -i 's/Vec<(_usize, f64)>/Vec<(usize, f64)>/g' crates/ml/src/**/*.rs

# Fix ARIMA
sed -i 's/let (_new_ar, new_ma,/let (new_ar, new_ma,/g' crates/ml/src/models/arima.rs
sed -i 's/Ok((_ar, ma,/Ok((ar, ma,/g' crates/ml/src/models/arima.rs

# Fix LSTM
sed -i 's/Array2::from_shape_fn((_rows,/Array2::from_shape_fn((rows,/g' crates/ml/src/**/*.rs
sed -i 's/init_weight(_hidden_size,/init_weight(hidden_size,/g' crates/ml/src/**/*.rs
sed -i 's/Array2::zeros((_hidden_size,/Array2::zeros((hidden_size,/g' crates/ml/src/**/*.rs
sed -i 's/LSTMCell::new(_input_dim,/LSTMCell::new(input_dim,/g' crates/ml/src/**/*.rs
sed -i 's/self.train_epoch(_train_data,/self.train_epoch(train_data,/g' crates/ml/src/**/*.rs
sed -i 's/self.validate(_val_data,/self.validate(val_data,/g' crates/ml/src/**/*.rs
sed -i 's/let (_new_hidden, new_cell)/let (new_hidden, new_cell)/g' crates/ml/src/**/*.rs
sed -i 's/(_new_hidden, new_cell)/(new_hidden, new_cell)/g' crates/ml/src/**/*.rs

# Fix GRU
sed -i 's/GRUCell::new(_input_dim,/GRUCell::new(input_dim,/g' crates/ml/src/**/*.rs
sed -i 's/Some(vd), Some(vl)) = (_val_data,/Some(vd), Some(vl)) = (val_data,/g' crates/ml/src/**/*.rs
sed -i 's/self.compute_loss(_vd,/self.compute_loss(vd,/g' crates/ml/src/**/*.rs
sed -i 's/self.compute_accuracy(_vd,/self.compute_accuracy(vd,/g' crates/ml/src/**/*.rs
sed -i 's/(_loss, acc)/(loss, acc)/g' crates/ml/src/**/*.rs
sed -i 's/(_train_loss, 0.0)/(train_loss, 0.0)/g' crates/ml/src/**/*.rs
sed -i 's/let (_val_loss, val_acc)/let (val_loss, val_acc)/g' crates/ml/src/**/*.rs

echo "Done fixing ML issues"