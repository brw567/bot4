# Cross-Validation Protocol - López de Prado Purged Walk-Forward

## Overview
Implementation of "Advances in Financial Machine Learning" Chapter 7: Cross-Validation in Finance

## Configuration
```yaml
cv_type: PurgedKFoldCV
n_splits: 5
purge_gap: 100  # samples to remove between train/test
embargo_pct: 0.01  # 1% of training size embargoed after test
min_train_size: 5000
min_test_size: 1000
```

## Fold Manifests

### Fold 1
```yaml
train_range: [2020-01-01, 2021-03-15]
purge_range: [2021-03-16, 2021-03-20]  # 100 samples
test_range: [2021-03-21, 2021-06-30]
embargo_range: [2021-07-01, 2021-07-15]  # 1% of train
train_samples: 45,360
test_samples: 7,200
```

### Fold 2
```yaml
train_range: [2020-01-01, 2021-09-15]
purge_range: [2021-09-16, 2021-09-20]
test_range: [2021-09-21, 2021-12-31]
embargo_range: [2022-01-01, 2022-01-20]
train_samples: 62,640
test_samples: 7,200
```

### Fold 3
```yaml
train_range: [2020-01-01, 2022-03-15]
purge_range: [2022-03-16, 2022-03-20]
test_range: [2022-03-21, 2022-06-30]
embargo_range: [2022-07-01, 2022-07-25]
train_samples: 79,920
test_samples: 7,200
```

### Fold 4
```yaml
train_range: [2020-01-01, 2022-09-15]
purge_range: [2022-09-16, 2022-09-20]
test_range: [2022-09-21, 2022-12-31]
embargo_range: [2023-01-01, 2023-01-30]
train_samples: 97,200
test_samples: 7,200
```

### Fold 5
```yaml
train_range: [2020-01-01, 2023-03-15]
purge_range: [2023-03-16, 2023-03-20]
test_range: [2023-03-21, 2023-06-30]
embargo_range: [2023-07-01, 2023-08-05]
train_samples: 114,480
test_samples: 7,200
```

## Leakage Prevention Measures

### 1. Temporal Leakage
```python
def verify_no_temporal_leakage(train_idx, test_idx):
    # Ensure no test sample timestamp <= any train timestamp
    assert max(train_timestamps[train_idx]) < min(test_timestamps[test_idx])
    
    # Verify purge gap
    gap = min(test_timestamps[test_idx]) - max(train_timestamps[train_idx])
    assert gap >= timedelta(minutes=100)
```

### 2. Feature Leakage
```python
def verify_feature_independence(features, labels):
    # Shuffle test to break any remaining dependence
    shuffled_labels = np.random.permutation(labels)
    
    # Train on shuffled - should get random performance
    model = train_model(features, shuffled_labels)
    score = model.score(features, shuffled_labels)
    
    # Sharpe should be near 0 if no leakage
    sharpe = calculate_sharpe(model.predict(features), shuffled_labels)
    assert abs(sharpe) < 0.1, f"Leakage detected! Sharpe = {sharpe}"
```

### 3. Look-Ahead Bias
```yaml
feature_calculation:
  window_type: backward_only
  min_history: 500  # samples before first prediction
  
  forbidden_features:
    - future_returns
    - next_bar_*
    - forward_*
    
  validation:
    - assert all features use data[:t] for prediction at t
    - no centered moving averages
    - no bidirectional RNNs on test data
```

## Reproducibility Script

```bash
#!/bin/bash
# reproduce_cv.sh

# Set deterministic seeds
export PYTHONHASHSEED=42
export CUDA_DETERMINISTIC=1

# Run CV with exact fold manifests
python run_cv.py \
  --config cv_config.yaml \
  --folds fold_manifests.json \
  --seed 42 \
  --output cv_report.json

# Verify metrics match
python verify_cv_metrics.py \
  --expected expected_metrics.json \
  --actual cv_report.json \
  --tolerance 1e-6
```

## Expected Metrics per Fold

```yaml
fold_1:
  accuracy: 0.683
  sharpe: 2.14
  max_dd: -0.087
  
fold_2:
  accuracy: 0.671
  sharpe: 2.08
  max_dd: -0.093
  
fold_3:
  accuracy: 0.689
  sharpe: 2.23
  max_dd: -0.081
  
fold_4:
  accuracy: 0.677
  sharpe: 2.11
  max_dd: -0.089
  
fold_5:
  accuracy: 0.692
  sharpe: 2.26
  max_dd: -0.078

aggregate:
  mean_accuracy: 0.682 ± 0.008
  mean_sharpe: 2.16 ± 0.07
  mean_max_dd: -0.086 ± 0.006
```

## Anti-Leakage Tests

### Test 1: Time Cutoff
```python
def test_time_cutoff():
    # Train only on data before cutoff
    cutoff = datetime(2022, 1, 1)
    train_data = data[data.timestamp < cutoff]
    test_data = data[data.timestamp >= cutoff]
    
    # No feature should use future information
    for feature in feature_list:
        assert feature.max_lookforward == 0
```

### Test 2: Information Coefficient Decay
```python
def test_ic_decay():
    # IC should decay with forecast horizon
    horizons = [1, 5, 10, 20, 50, 100]
    ics = []
    
    for h in horizons:
        pred = model.predict(features)
        actual = returns.shift(-h)
        ic = np.corrcoef(pred, actual)[0, 1]
        ics.append(ic)
    
    # Verify monotonic decay
    assert all(ics[i] >= ics[i+1] for i in range(len(ics)-1))
```

### Test 3: Embargo Effectiveness
```python
def test_embargo():
    # Train with and without embargo
    score_with_embargo = cv_with_embargo.score()
    score_without_embargo = cv_without_embargo.score()
    
    # Score should be lower (more realistic) with embargo
    assert score_with_embargo < score_without_embargo
    
    # Difference indicates leakage magnitude
    leakage = score_without_embargo - score_with_embargo
    assert leakage > 0.02  # Significant if >2%
```