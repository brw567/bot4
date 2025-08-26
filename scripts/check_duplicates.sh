#!/bin/bash
# Check for duplicate functions across the codebase

echo "🔍 Checking for duplicate functions in Rust codebase..."
echo "========================================="
echo ""

# Common function patterns that might be duplicated
PATTERNS=(
    "calculate_ema"
    "calculate_sma"
    "calculate_rsi"
    "calculate_volatility"
    "calculate_correlation"
    "calculate_var"
    "calculate_kelly"
    "validate_order"
    "check_limits"
    "process_event"
    "update_position"
    "get_balance"
    "place_order"
    "cancel_order"
)

# Track duplicates
DUPLICATES_FOUND=0

for pattern in "${PATTERNS[@]}"; do
    # Find all occurrences
    FILES=$(grep -r "fn $pattern" rust_core --include="*.rs" -l 2>/dev/null)
    COUNT=$(echo "$FILES" | grep -c . 2>/dev/null || echo 0)
    
    if [ $COUNT -gt 1 ]; then
        echo "⚠️  DUPLICATE: '$pattern' found in $COUNT files:"
        echo "$FILES" | while read -r file; do
            LINE=$(grep -n "fn $pattern" "$file" | head -1 | cut -d: -f1)
            echo "   - $file:$LINE"
        done
        echo ""
        DUPLICATES_FOUND=$((DUPLICATES_FOUND + 1))
    fi
done

# Check for similar struct definitions
echo "Checking for duplicate structs..."
STRUCTS=(
    "OrderBook"
    "Position"
    "Order"
    "Trade"
    "Candle"
    "MarketData"
    "RiskLimits"
)

for struct in "${STRUCTS[@]}"; do
    FILES=$(grep -r "struct $struct" rust_core --include="*.rs" -l 2>/dev/null)
    COUNT=$(echo "$FILES" | grep -c . 2>/dev/null || echo 0)
    
    if [ $COUNT -gt 1 ]; then
        echo "⚠️  DUPLICATE STRUCT: '$struct' found in $COUNT files:"
        echo "$FILES" | while read -r file; do
            LINE=$(grep -n "struct $struct" "$file" | head -1 | cut -d: -f1)
            echo "   - $file:$LINE"
        done
        echo ""
        DUPLICATES_FOUND=$((DUPLICATES_FOUND + 1))
    fi
done

echo "========================================="
if [ $DUPLICATES_FOUND -eq 0 ]; then
    echo "✅ No duplicates found!"
else
    echo "❌ Found $DUPLICATES_FOUND potential duplicates"
    echo ""
    echo "Action Required:"
    echo "1. Review each duplicate"
    echo "2. Consolidate into single implementation"
    echo "3. Update imports to use shared version"
    echo "4. Update COMPLETE_CODEBASE_REGISTRY.md"
fi

exit $DUPLICATES_FOUND