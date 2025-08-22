#!/bin/bash

# Revert incorrect underscore prefixes from auto-fix
# Team: Sam (Code Quality Lead)

set -e

echo "=========================================="
echo "Reverting incorrect underscore prefixes"
echo "=========================================="

# Fix analysis crate - all variables are actually used
echo "Fixing analysis crate..."
find crates/analysis -name "*.rs" -type f | while read file; do
    # Revert all underscore prefixes for let bindings in this crate
    sed -i 's/let _\([a-z_][a-z0-9_]*\) =/let \1 =/g' "$file"
done

# Fix risk_engine crate  
echo "Fixing risk_engine crate..."
find crates/risk_engine -name "*.rs" -type f | while read file; do
    # Revert underscore prefixes for actually used variables
    sed -i 's/let _\(asset\|loss\|drawdown\|kelly_size\|var_limit\|actual_vol\|ruined\|t_dist\|daily_return\|true\|false\|expected_breaches\|actual_breaches\) =/let \1 =/g' "$file"
    sed -i 's/let _\(risk_checker\|limits\|checker\|result\|detector\|mm_id\|now\|event\|cancel\|retail_id\|rec\) =/let \1 =/g' "$file"
    sed -i 's/let _\(ftx_scenario\|activity\|quote_frequency\|time_span\|total_volume\|order_symmetry\|total_orders\) =/let \1 =/g' "$file"
    sed -i 's/let _\(cancellation_rate\|avg_order_lifetime_ms\|provides_liquidity_pct\|avg_spread_bps\|spread_stability\) =/let \1 =/g' "$file"
    sed -i 's/let _\(mean\|std_dev\|inventory_cycling\|profile\|cutoff\|metrics\|all_participants\|market_makers\) =/let \1 =/g' "$file"
    sed -i 's/let _\(mm_count\|avg_confidence\|avg_quote_frequency\|temp\|pivot\|factor\|lifetime\) =/let \1 =/g' "$file"
done

# Fix infrastructure circuit_breaker (already done manually, but ensure it's correct)
sed -i 's/let _is_half_open =/let is_half_open =/g' crates/infrastructure/src/circuit_breaker.rs

# Fix infrastructure stream_processing (already done manually)

# Fix ml crate
echo "Fixing ml crate..."
find crates/ml -name "*.rs" -type f | while read file; do
    sed -i 's/let _\([a-z_][a-z0-9_]*\) =/let \1 =/g' "$file"
done

# Fix order_management
echo "Fixing order_management crate..."
find crates/order_management -name "*.rs" -type f | while read file; do
    sed -i 's/let _\([a-z_][a-z0-9_]*\) =/let \1 =/g' "$file"
done

# Fix trading_engine
echo "Fixing trading_engine crate..."
find crates/trading_engine -name "*.rs" -type f | while read file; do
    sed -i 's/let _\([a-z_][a-z0-9_]*\) =/let \1 =/g' "$file"
done

# Fix websocket
echo "Fixing websocket crate..."
find crates/websocket -name "*.rs" -type f | while read file; do
    sed -i 's/let _\([a-z_][a-z0-9_]*\) =/let \1 =/g' "$file"
done

echo "=========================================="
echo "Revert complete, now compiling to check..."
echo "=========================================="

cargo build --all 2>&1 | tail -20