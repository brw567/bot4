#!/bin/bash

# Final cleanup of all underscore issues
# Team: Sam (Code Quality Lead)

set -e

echo "=========================================="
echo "Final underscore cleanup"
echo "=========================================="

# Fix common patterns across all crates
for crate in analysis infrastructure ml order_management risk_engine trading_engine websocket; do
    echo "Fixing $crate..."
    
    find crates/$crate -name "*.rs" -type f | while read file; do
        # Fix common variable patterns that are actually used
        sed -i 's/_nanos\b/nanos/g' "$file"
        sed -i 's/_true\b/true/g' "$file"
        sed -i 's/_false\b/false/g' "$file"
        sed -i 's/_state\b/state/g' "$file"
        sed -i 's/_new_global_state\b/new_global_state/g' "$file"
        sed -i 's/_new_value\b/new_value/g' "$file"
        sed -i 's/_pressure\b/pressure/g' "$file"
        sed -i 's/_core_id\b/core_id/g' "$file"
        sed -i 's/_ema_fixed\b/ema_fixed/g' "$file"
        sed -i 's/_clock\b/clock/g' "$file"
        sed -i 's/_stream\b/stream/g' "$file"
        sed -i 's/_count\b/count/g' "$file"
        sed -i 's/_elapsed\b/elapsed/g' "$file"
        sed -i 's/_length\b/length/g' "$file"
        sed -i 's/_total_latency\b/total_latency/g' "$file"
        sed -i 's/_message\b/message/g' "$file"
        
        # Fix specific issues in infrastructure
        if [[ "$file" == *"infrastructure"* ]]; then
            sed -i 's/let _instrument =/let instrument =/g' "$file"
        fi
    done
done

echo "=========================================="
echo "Cleanup complete, compiling to verify..."
echo "=========================================="

cargo build --all 2>&1 | tail -20