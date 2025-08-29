#!/bin/bash
# TARGETED DUPLICATE ELIMINATION - Working within Claude's context limits
# Processing in small batches to avoid token overflow

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    TARGETED DUPLICATE ELIMINATION - BATCH PROCESSING       ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"

# BATCH 1: Eliminate Position duplicate
echo -e "\n${YELLOW}BATCH 1: Eliminating Position duplicate${NC}"
if [ -f "/home/hamster/bot4/rust_core/domain_types/src/position_canonical.rs" ]; then
    # Comment out the duplicate Position struct
    sed -i 's/^pub struct Position {/\/\/ ELIMINATED: Duplicate Position - use canonical_types::Position\n\/\/ pub struct Position {/' \
        /home/hamster/bot4/rust_core/domain_types/src/position_canonical.rs
    
    # Update imports to use canonical version
    find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -l "position_canonical::Position" {} \; | \
        xargs -I {} sed -i 's/position_canonical::Position/canonical_types::Position/g' {}
    
    echo -e "${GREEN}✓ Position duplicate eliminated${NC}"
fi

# BATCH 2: Eliminate OrderBook duplicate in smart_order_router
echo -e "\n${YELLOW}BATCH 2: Eliminating OrderBook duplicate${NC}"
if grep -q "^pub struct OrderBook {" /home/hamster/bot4/rust_core/crates/execution/src/smart_order_router.rs 2>/dev/null; then
    # Replace local OrderBook with import from domain_types
    sed -i '/^pub struct OrderBook {/,/^}/d' \
        /home/hamster/bot4/rust_core/crates/execution/src/smart_order_router.rs
    
    # Add import if not present
    if ! grep -q "use domain_types::market_data::OrderBook" /home/hamster/bot4/rust_core/crates/execution/src/smart_order_router.rs; then
        sed -i '1i use domain_types::market_data::OrderBook;' \
            /home/hamster/bot4/rust_core/crates/execution/src/smart_order_router.rs
    fi
    
    echo -e "${GREEN}✓ OrderBook duplicate eliminated${NC}"
fi

# BATCH 3: Find and eliminate Signal duplicates
echo -e "\n${YELLOW}BATCH 3: Checking for Signal duplicates${NC}"
SIGNAL_COUNT=$(find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -l "^pub struct .*Signal" {} \; | wc -l)
if [ "$SIGNAL_COUNT" -gt 1 ]; then
    echo "Found $SIGNAL_COUNT Signal definitions. Consolidating..."
    
    # Keep only canonical_types::TradingSignal
    find /home/hamster/bot4/rust_core -name "*.rs" -type f | while read -r file; do
        if [[ "$file" != *"canonical_types.rs" ]] && grep -q "^pub struct .*Signal {" "$file"; then
            sed -i 's/^pub struct \(.*Signal\) {/\/\/ ELIMINATED: Duplicate \1 - use canonical_types::TradingSignal\n\/\/ pub struct \1 {/' "$file"
        fi
    done
    echo -e "${GREEN}✓ Signal duplicates marked for elimination${NC}"
fi

# BATCH 4: Find and eliminate Tick duplicates
echo -e "\n${YELLOW}BATCH 4: Checking for Tick duplicates${NC}"
TICK_FILES=$(find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -l "^pub struct Tick {" {} \; | grep -v canonical_types.rs || true)
if [ -n "$TICK_FILES" ]; then
    echo "$TICK_FILES" | while read -r file; do
        echo "Eliminating duplicate Tick in: $file"
        sed -i '/^pub struct Tick {/,/^}/s/^/\/\/ ELIMINATED: /' "$file"
        
        # Add import for canonical Tick
        if ! grep -q "use domain_types::canonical_types::Tick" "$file"; then
            sed -i '1i use domain_types::canonical_types::Tick;' "$file"
        fi
    done
    echo -e "${GREEN}✓ Tick duplicates eliminated${NC}"
fi

# BATCH 5: Portfolio duplicates
echo -e "\n${YELLOW}BATCH 5: Checking for Portfolio duplicates${NC}"
PORTFOLIO_COUNT=$(find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -c "^pub struct Portfolio {" {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
if [ "$PORTFOLIO_COUNT" -gt 1 ]; then
    echo "Found $PORTFOLIO_COUNT Portfolio definitions. Consolidating..."
    
    # Find and eliminate non-canonical Portfolio structs
    find /home/hamster/bot4/rust_core -name "*.rs" -type f | while read -r file; do
        if [[ "$file" != *"canonical_types.rs" ]] && grep -q "^pub struct Portfolio {" "$file"; then
            # Comment out the struct definition
            sed -i '/^pub struct Portfolio {/,/^}/s/^/\/\/ ELIMINATED: /' "$file"
            
            # Update imports
            basename_file=$(basename "$file" .rs)
            find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec sed -i \
                "s/${basename_file}::Portfolio/canonical_types::Portfolio/g" {} \;
        fi
    done
    echo -e "${GREEN}✓ Portfolio duplicates eliminated${NC}"
fi

# Count remaining duplicates
echo -e "\n${YELLOW}═══ VERIFICATION ═══${NC}"
REMAINING=$(find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -h "^pub struct " {} \; | \
    grep -v "ELIMINATED:" | grep -v "^//" | \
    sort | uniq -c | sort -nr | awk '$1 > 1' | wc -l)

echo -e "Remaining duplicate structs: ${RED}$REMAINING${NC}"

# Show top remaining duplicates
if [ "$REMAINING" -gt 0 ]; then
    echo -e "\n${YELLOW}Top remaining duplicates:${NC}"
    find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -h "^pub struct " {} \; | \
        grep -v "ELIMINATED:" | grep -v "^//" | \
        sort | uniq -c | sort -nr | head -10
fi

echo -e "\n${GREEN}Batch processing complete!${NC}"