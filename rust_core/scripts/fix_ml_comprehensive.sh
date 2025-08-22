#!/bin/bash

echo "Comprehensive ML crate fix..."

# Fix function parameters with _params pattern
find crates/ml -name "*.rs" -exec sed -i 's/fn calculate(&self, data: &\[Candle\], _params: &IndicatorParams)/fn calculate(\&self, data: \&[Candle], params: \&IndicatorParams)/g' {} \;

# Fix models/ensemble.rs
sed -i 's/for (_pred, id) in/for (pred, id) in/g' crates/ml/src/models/ensemble.rs
sed -i 's/pred\[0\] \* weight/pred[0] * weight/g' crates/ml/src/models/ensemble.rs

# Fix models/registry.rs  
sed -i 's/self.mmap_cache.write().insert(_id,/self.mmap_cache.write().insert(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.model_sizes.write().insert(_id,/self.model_sizes.write().insert(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.get_metric_value(_baseline,/self.get_metric_value(baseline,/g' crates/ml/src/models/registry.rs
sed -i 's/self.get_metric_value(_current,/self.get_metric_value(current,/g' crates/ml/src/models/registry.rs
sed -i 's/.insert(_model_id,/.insert(model_id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.models.write().insert(_id,/self.models.write().insert(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.version_index.write().insert(_version_key,/self.version_index.write().insert(version_key,/g' crates/ml/src/models/registry.rs
sed -i 's/self.performance_history.write().insert(_id,/self.performance_history.write().insert(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.deploy_immediate(_id,/self.deploy_immediate(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.deploy_canary(_id,/self.deploy_canary(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.deploy_blue_green(_id,/self.deploy_blue_green(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.deploy_shadow(_id,/self.deploy_shadow(id,/g' crates/ml/src/models/registry.rs
sed -i 's/self.select_ab_model(_ab_test,/self.select_ab_model(ab_test,/g' crates/ml/src/models/registry.rs
sed -i 's/for (_model_id, percentage) in/for (model_id, percentage) in/g' crates/ml/src/models/registry.rs
sed -i 's/self.trigger_rollback(_purpose,/self.trigger_rollback(purpose,/g' crates/ml/src/models/registry.rs
sed -i 's/.insert(_purpose,/.insert(purpose,/g' crates/ml/src/models/registry.rs
sed -i 's/Option<(_f64, f64)>/Option<(f64, f64)>/g' crates/ml/src/models/registry.rs
sed -i 's/self.t_cdf(_t_stat,/self.t_cdf(t_stat,/g' crates/ml/src/models/registry.rs

# Fix models/deep_lstm.rs
sed -i 's/for (_layer_idx, layer) in/for (layer_idx, layer) in/g' crates/ml/src/models/deep_lstm.rs
sed -i 's/for (_idx, layer) in/for (idx, layer) in/g' crates/ml/src/models/deep_lstm.rs
sed -i 's/self.forward(_features,/self.forward(features,/g' crates/ml/src/models/deep_lstm.rs
sed -i 's/Array2::from_shape_fn((_rows, cols)/Array2::from_shape_fn((rows, cols)/g' crates/ml/src/models/deep_lstm.rs
sed -i 's/init_matrix(_input_size,/init_matrix(input_size,/g' crates/ml/src/models/deep_lstm.rs
sed -i 's/init_matrix(_hidden_size,/init_matrix(hidden_size,/g' crates/ml/src/models/deep_lstm.rs
sed -i 's/Array2::zeros((_batch_size,/Array2::zeros((batch_size,/g' crates/ml/src/models/deep_lstm.rs
sed -i 's/Array2::<f64>::zeros((_batch_size,/Array2::<f64>::zeros((batch_size,/g' crates/ml/src/models/deep_lstm.rs

# Fix models/ensemble_optimized.rs
sed -i 's/DeepLSTM::new(_input_size,/DeepLSTM::new(input_size,/g' crates/ml/src/models/ensemble_optimized.rs
sed -i 's/TransformerModel::new(_input_size,/TransformerModel::new(input_size,/g' crates/ml/src/models/ensemble_optimized.rs
sed -i 's/TemporalCNN::new(_input_size,/TemporalCNN::new(input_size,/g' crates/ml/src/models/ensemble_optimized.rs
sed -i 's/StackedGRU::new(_input_size,/StackedGRU::new(input_size,/g' crates/ml/src/models/ensemble_optimized.rs
sed -i 's/self.weighted_average(_predictions,/self.weighted_average(predictions,/g' crates/ml/src/models/ensemble_optimized.rs
sed -i 's/.map(|(_x, y)|/.map(|(x, y)|/g' crates/ml/src/models/ensemble_optimized.rs
sed -i 's/.map(|(_name, _)|/.map(|(name, _)|/g' crates/ml/src/models/ensemble_optimized.rs
sed -i 's/self.optimize_weights(_features,/self.optimize_weights(features,/g' crates/ml/src/models/ensemble_optimized.rs

# Fix models/xgboost_optimized.rs
sed -i 's/HashMap<(_usize, usize)/HashMap<(usize, usize)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/max_depth: (_u32, u32)/max_depth: (u32, u32)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/learning_rate: (_f32, f32)/learning_rate: (f32, f32)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/subsample: (_f32, f32)/subsample: (f32, f32)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/reg_alpha: (_f32, f32)/reg_alpha: (f32, f32)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/reg_lambda: (_f32, f32)/reg_lambda: (f32, f32)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/min_child_weight: (_f32, f32)/min_child_weight: (f32, f32)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/gradients.resize(_n_samples,/gradients.resize(n_samples,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/hessians.resize(_n_samples,/hessians.resize(n_samples,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/Array1::from_elem(_n_samples,/Array1::from_elem(n_samples,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/self.subsample_data(_n_samples,/self.subsample_data(n_samples,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/let (_sampled_indices, feature_indices)/let (sampled_indices, feature_indices)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/if let Some((_val_x, val_y))/if let Some((val_x, val_y))/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/self.calculate_sums_avx512(_gradients,/self.calculate_sums_avx512(gradients,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/self.calculate_sums_scalar(_gradients,/self.calculate_sums_scalar(gradients,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/let (_sum_grad, sum_hess)/let (sum_grad, sum_hess)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/let (_left_indices, right_indices)/let (left_indices, right_indices)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/Vec<(_f32, usize)>/Vec<(f32, usize)>/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/|&(_val, _)|/|&(val, _)|/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/self.calculate_gradients_avx512(_targets,/self.calculate_gradients_avx512(targets,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/_mm512_sub_ps(_pred,/_mm512_sub_ps(pred,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/) -> (_f32, f32)/) -> (f32, f32)/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/_mm512_add_ps(_sum_grad,/_mm512_add_ps(sum_grad,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/_mm512_add_ps(_sum_hess,/_mm512_add_ps(sum_hess,/g' crates/ml/src/models/xgboost_optimized.rs
sed -i 's/(_grad_sum, hess_sum)/(grad_sum, hess_sum)/g' crates/ml/src/models/xgboost_optimized.rs

echo "ML crate comprehensive fixes applied"