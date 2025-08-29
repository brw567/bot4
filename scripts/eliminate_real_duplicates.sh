#!/bin/bash
# ELIMINATE REAL BUSINESS LOGIC DUPLICATES (non-SQLite)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    ELIMINATING REAL BUSINESS LOGIC DUPLICATES              ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

# Function to find and consolidate duplicates
consolidate_struct() {
    local struct_name=$1
    local canonical_location=$2
    local canonical_module=$3
    
    echo -e "\n${YELLOW}Processing: $struct_name${NC}"
    
    # Find all files with this struct (excluding canonical)
    FILES=$(find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -l "^pub struct $struct_name {" {} \; | grep -v "$canonical_location" || true)
    
    if [ -n "$FILES" ]; then
        echo "$FILES" | while read -r file; do
            echo "  Eliminating in: $file"
            
            # Comment out the duplicate struct
            sed -i "/^pub struct $struct_name {/,/^}/s/^/\/\/ ELIMINATED: Duplicate - use $canonical_module\n\/\/ /" "$file"
            
            # Add import if needed
            if ! grep -q "use $canonical_module" "$file"; then
                sed -i "1i use $canonical_module;" "$file"
            fi
        done
        echo -e "  ${GREEN}✓ Consolidated to $canonical_module${NC}"
    else
        echo -e "  ${GREEN}✓ Already consolidated${NC}"
    fi
}

# BATCH 1: MicrostructureFeatures (4 instances)
consolidate_struct "MicrostructureFeatures" "ml/src/features.rs" "ml::features::MicrostructureFeatures"

# BATCH 2: ZeroCopyPipeline (3 instances)
consolidate_struct "ZeroCopyPipeline" "infrastructure/src/zero_copy.rs" "infrastructure::zero_copy::ZeroCopyPipeline"

# BATCH 3: WebSocketConfig (3 instances)
consolidate_struct "WebSocketConfig" "execution/src/websocket.rs" "execution::websocket::WebSocketConfig"

# BATCH 4: ValidationReport (3 instances)
consolidate_struct "ValidationReport" "infrastructure/src/validation.rs" "infrastructure::validation::ValidationReport"

# BATCH 5: TimescaleConfig (3 instances)
consolidate_struct "TimescaleConfig" "data_ingestion/src/timescale.rs" "data_ingestion::timescale::TimescaleConfig"

# BATCH 6: RateLimiter (3 instances)
consolidate_struct "RateLimiter" "execution/src/rate_limiter.rs" "execution::rate_limiter::RateLimiter"

# BATCH 7: PriceLevel (3 instances)
echo -e "\n${YELLOW}Special handling for PriceLevel (keeping in market_data)${NC}"
find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -l "^pub struct PriceLevel {" {} \; | \
    grep -v "domain_types/src/market_data.rs" | while read -r file; do
    echo "  Eliminating in: $file"
    sed -i '/^pub struct PriceLevel {/,/^}/d' "$file"
    
    # Add import
    if ! grep -q "use domain_types::market_data::PriceLevel" "$file"; then
        sed -i '1i use domain_types::market_data::PriceLevel;' "$file"
    fi
done

# BATCH 8: Prediction (3 instances)
consolidate_struct "Prediction" "ml/src/predictions.rs" "ml::predictions::Prediction"

# BATCH 9: OrderUpdate (3 instances)
consolidate_struct "OrderUpdate" "execution/src/order_updates.rs" "execution::order_updates::OrderUpdate"

# BATCH 10: OrderBookSnapshot (3 instances)
consolidate_struct "OrderBookSnapshot" "domain_types/src/market_data.rs" "domain_types::market_data::OrderBookSnapshot"

# BATCH 11: Model-related structs
echo -e "\n${YELLOW}Consolidating ML Model structs${NC}"
consolidate_struct "ModelStorage" "ml/src/model_storage.rs" "ml::model_storage::ModelStorage"
consolidate_struct "ModelMetrics" "ml/src/model_metrics.rs" "ml::model_metrics::ModelMetrics"
consolidate_struct "ModelMetadata" "ml/src/model_metadata.rs" "ml::model_metadata::ModelMetadata"

# Check remaining duplicates
echo -e "\n${BLUE}═══ VERIFICATION ═══${NC}"
REMAINING=$(find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -h "^pub struct " {} \; | \
    grep -v "ELIMINATED:" | grep -v "^//" | grep -v sqlite3 | grep -v fts5 | \
    sort | uniq -c | sort -nr | awk '$1 > 1' | wc -l)

echo -e "Non-SQLite duplicate structs remaining: ${RED}$REMAINING${NC}"

if [ "$REMAINING" -gt 0 ]; then
    echo -e "\n${YELLOW}Remaining duplicates:${NC}"
    find /home/hamster/bot4/rust_core -name "*.rs" -type f -exec grep -h "^pub struct " {} \; | \
        grep -v "ELIMINATED:" | grep -v "^//" | grep -v sqlite3 | grep -v fts5 | \
        sort | uniq -c | sort -nr | awk '$1 > 1' | head -10
fi

echo -e "\n${GREEN}═══ COMPLETE ═══${NC}"