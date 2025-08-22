#!/bin/bash

# Fix ML crate SIMD variable issues
# Team: Jordan (Performance) + Morgan (ML)
# Approach: Fix ONLY actual variable name issues, not SIMD intrinsics

set -e

echo "=========================================="
echo "Fixing ML crate SIMD variable issues"
echo "=========================================="

# Fix specific variable issues in indicators.rs
echo "Fixing indicators.rs..."
sed -i 's/_mm256_add_ps(_sum,/_mm256_add_ps(sum,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm_add_ps(_sum128,/_mm_add_ps(sum128,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm_movehl_ps(_sum128,/_mm_movehl_ps(sum128,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm_add_ps(_sum64,/_mm_add_ps(sum64,/g' crates/ml/src/feature_engine/indicators.rs
sed -i 's/_mm_movehl_ps(_sum64,/_mm_movehl_ps(sum64,/g' crates/ml/src/feature_engine/indicators.rs

# Fix common variable patterns
for file in crates/ml/src/**/*.rs crates/ml/src/*.rs; do
    if [[ -f "$file" ]]; then
        # Fix variable declarations that were incorrectly prefixed
        sed -i 's/let _prices =/let prices =/g' "$file"
        sed -i 's/let _ema_vec =/let ema_vec =/g' "$file"
        sed -i 's/let _weighted_price =/let weighted_price =/g' "$file"
        sed -i 's/let _v =/let v =/g' "$file"
        sed -i 's/let _high =/let high =/g' "$file"
        sed -i 's/let _curr =/let curr =/g' "$file"
        sed -i 's/let _change =/let change =/g' "$file"
        sed -i 's/let _zero =/let zero =/g' "$file"
        sed -i 's/let _gains =/let gains =/g' "$file"
        sed -i 's/let _losses =/let losses =/g' "$file"
        
        # Fix type issues
        sed -i 's/: _f64/: f64/g' "$file"
        
        # Fix min/max that got broken
        sed -i 's/\.min(/\.min(/g' "$file"
        sed -i 's/\.max(/\.max(/g' "$file"
    fi
done

echo "Done fixing ML crate"