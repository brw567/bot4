#!/bin/bash

# Fix ML indicators SIMD variables
echo "Fixing ML crate SIMD variable errors..."

# Fix indicators.rs
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

# Fix calculate method parameter
sed -i 's/calculate(_data, _params)/calculate(data, params)/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/calculate(_candles,/calculate(candles,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/self.cache.insert(_cache_key,/self.cache.insert(cache_key,/g' crates/ml/src/feature_engine/indicators.rs

# Fix feature bounds
sed -i 's/HashMap<String, (_f64, f64)>/HashMap<String, (f64, f64)>/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/if let Some((_min, max))/if let Some((min, max))/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/if value < \*min/if value < *min/g' crates/ml/src/feature_engine/indicators.rs

# Fix indicators_extended.rs
sed -i 's/calculate(_data, params)/calculate(data, params)/g' crates/ml/src/feature_engine/indicators_extended.rs

# Fix pipeline.rs
sed -i 's/for (_name, value) in/for (name, value) in/g' crates/ml/src/feature_engine/pipeline.rs
sed -i 's/features.add_feature(&name,/features.add_feature(\&name,/g' crates/ml/src/feature_engine/pipeline.rs

# Fix scaler.rs
sed -i 's/self.fit_standard(_data,/self.fit_standard(data,/g' crates/ml/src/feature_engine/scaler.rs
sed -i 's/self.fit_minmax(_data,/self.fit_minmax(data,/g' crates/ml/src/feature_engine/scaler.rs
sed -i 's/self.fit_robust(_data,/self.fit_robust(data,/g' crates/ml/src/feature_engine/scaler.rs
sed -i 's/self.fit_maxabs(_data,/self.fit_maxabs(data,/g' crates/ml/src/feature_engine/scaler.rs
sed -i 's/self.transform_minmax(_features,/self.transform_minmax(features,/g' crates/ml/src/feature_engine/scaler.rs
sed -i 's/scaled_value.clamp(_target_min,/scaled_value.clamp(target_min,/g' crates/ml/src/feature_engine/scaler.rs

# Fix selector.rs
sed -i 's/Vec<(_usize, f64)>/Vec<(usize, f64)>/g' crates/ml/src/feature_engine/selector.rs
sed -i 's/self.calculate_correlations(_data,/self.calculate_correlations(data,/g' crates/ml/src/feature_engine/selector.rs
sed -i 's/self.calculate_variances(_data,/self.calculate_variances(data,/g' crates/ml/src/feature_engine/selector.rs

# Fix models/arima.rs
sed -i 's/self.difference_series(_data,/self.difference_series(data,/g' crates/ml/src/models/arima.rs
sed -i 's/let (_new_ar, new_ma, new_intercept)/let (new_ar, new_ma, new_intercept)/g' crates/ml/src/models/arima.rs
sed -i 's/self.ar_params = new_ar/self.ar_params = new_ar/g' crates/ml/src/models/arima.rs
sed -i 's/Ok((_ar, ma, intercept))/Ok((ar, ma, intercept))/g' crates/ml/src/models/arima.rs

# Fix models/lstm.rs
sed -i 's/Array2::from_shape_fn((_rows, cols)/Array2::from_shape_fn((rows, cols)/g' crates/ml/src/models/lstm.rs
sed -i 's/init_weight(_hidden_size,/init_weight(hidden_size,/g' crates/ml/src/models/lstm.rs
sed -i 's/Array2::zeros((_hidden_size,/Array2::zeros((hidden_size,/g' crates/ml/src/models/lstm.rs
sed -i 's/(_new_hidden, new_cell)/(new_hidden, new_cell)/g' crates/ml/src/models/lstm.rs
sed -i 's/LSTMCell::new(_input_dim,/LSTMCell::new(input_dim,/g' crates/ml/src/models/lstm.rs
sed -i 's/self.train_epoch(_train_data,/self.train_epoch(train_data,/g' crates/ml/src/models/lstm.rs
sed -i 's/self.validate(_val_data,/self.validate(val_data,/g' crates/ml/src/models/lstm.rs
sed -i 's/let (_new_hidden, new_cell)/let (new_hidden, new_cell)/g' crates/ml/src/models/lstm.rs
sed -i 's/hidden = new_hidden/hidden = new_hidden/g' crates/ml/src/models/lstm.rs

# Fix models/gru.rs
sed -i 's/Array2::from_shape_fn((_rows, cols)/Array2::from_shape_fn((rows, cols)/g' crates/ml/src/models/gru.rs
sed -i 's/init_weight(_hidden_size,/init_weight(hidden_size,/g' crates/ml/src/models/gru.rs
sed -i 's/Array2::zeros((_hidden_size,/Array2::zeros((hidden_size,/g' crates/ml/src/models/gru.rs
sed -i 's/GRUCell::new(_input_dim,/GRUCell::new(input_dim,/g' crates/ml/src/models/gru.rs
sed -i 's/self.train_epoch(_train_data,/self.train_epoch(train_data,/g' crates/ml/src/models/gru.rs
sed -i 's/if let (Some(vd), Some(vl)) = (_val_data, val_labels)/if let (Some(vd), Some(vl)) = (val_data, val_labels)/g' crates/ml/src/models/gru.rs
sed -i 's/self.compute_loss(_vd, vl)/self.compute_loss(vd, vl)/g' crates/ml/src/models/gru.rs
sed -i 's/self.compute_accuracy(_vd, vl)/self.compute_accuracy(vd, vl)/g' crates/ml/src/models/gru.rs
sed -i 's/(_loss, acc)/(loss, acc)/g' crates/ml/src/models/gru.rs
sed -i 's/(_train_loss, 0.0)/(train_loss, 0.0)/g' crates/ml/src/models/gru.rs
sed -i 's/let (_val_loss, val_acc)/let (val_loss, val_acc)/g' crates/ml/src/models/gru.rs
sed -i 's/.map(|(_p, l)|/.map(|(p, l)|/g' crates/ml/src/models/gru.rs

# Fix models/ensemble.rs
sed -i 's/self.arima_models.insert(_id,/self.arima_models.insert(id,/g' crates/ml/src/models/ensemble.rs
sed -i 's/self.model_weights.write().insert(_id,/self.model_weights.write().insert(id,/g' crates/ml/src/models/ensemble.rs
sed -i 's/self.model_performance.write().insert(_id,/self.model_performance.write().insert(id,/g' crates/ml/src/models/ensemble.rs
sed -i 's/self.lstm_models.insert(_id,/self.lstm_models.insert(id,/g' crates/ml/src/models/ensemble.rs
sed -i 's/self.gru_models.insert(_id,/self.gru_models.insert(id,/g' crates/ml/src/models/ensemble.rs
sed -i 's/for (_id, model) in/for (id, model) in/g' crates/ml/src/models/ensemble.rs
sed -i 's/weights.get(id)/weights.get(\&id)/g' crates/ml/src/models/ensemble.rs

echo "ML crate indicator fixes applied"