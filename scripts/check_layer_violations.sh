#!/bin/bash
# Bot4 Layer Architecture Violation Detector
# Enforces strict layer dependencies at compile-time
# Version: 1.0

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Layer definitions (0 = lowest, 6 = highest)
declare -A LAYERS=(
    ["infrastructure"]=0
    ["data_ingestion"]=1
    ["risk"]=2
    ["ml"]=3
    ["strategies"]=4
    ["trading_engine"]=5
    ["integration"]=6
)

# Reverse mapping for display
declare -A LAYER_NAMES=(
    [0]="infrastructure"
    [1]="data_ingestion"
    [2]="risk"
    [3]="ml"
    [4]="strategies"
    [5]="trading_engine"
    [6]="integration"
)

RUST_CORE="/home/hamster/bot4/rust_core"

echo -e "${BLUE}🔍 Bot4 Layer Violation Detector v1.0${NC}"
echo "Checking for cross-layer violations..."
echo "----------------------------------------"

# Function to get layer number from path
get_layer() {
    local path=$1
    
    for layer in "${!LAYERS[@]}"; do
        if [[ "$path" == *"/$layer/"* ]] || [[ "$path" == *"/$layer."* ]]; then
            echo "${LAYERS[$layer]}"
            return
        fi
    done
    
    # Check crates directory
    if [[ "$path" == */crates/* ]]; then
        local crate_name=$(echo "$path" | sed -n 's|.*/crates/\([^/]*\)/.*|\1|p')
        for layer in "${!LAYERS[@]}"; do
            if [[ "$crate_name" == "$layer" ]]; then
                echo "${LAYERS[$layer]}"
                return
            fi
        done
    fi
    
    echo "-1"  # Unknown layer
}

# Function to check imports in a Rust file
check_file_imports() {
    local file=$1
    local file_layer=$(get_layer "$file")
    local violations=0
    
    if [ "$file_layer" -eq "-1" ]; then
        return 0  # Skip files not in known layers
    fi
    
    # Extract use statements
    local imports=$(grep -E "^use\s+" "$file" 2>/dev/null | grep -v "^use std" | grep -v "^use core" || true)
    
    while IFS= read -r import; do
        # Extract the module being imported
        local module=$(echo "$import" | sed -n 's/^use\s\+\([a-zA-Z0-9_:]*\).*/\1/p')
        
        # Check if it's an internal import
        if [[ "$module" == "crate::"* ]] || [[ "$module" == "super::"* ]]; then
            continue  # Internal imports are OK
        fi
        
        # Try to determine the layer of the imported module
        for layer in "${!LAYERS[@]}"; do
            if [[ "$module" == "$layer"* ]]; then
                local import_layer="${LAYERS[$layer]}"
                
                # Check for violation (importing from higher layer)
                if [ "$import_layer" -gt "$file_layer" ]; then
                    echo -e "${RED}❌ VIOLATION: ${LAYER_NAMES[$file_layer]} (layer $file_layer) imports from ${LAYER_NAMES[$import_layer]} (layer $import_layer)${NC}"
                    echo "   File: $file"
                    echo "   Import: $import"
                    ((violations++))
                fi
                break
            fi
        done
    done <<< "$imports"
    
    return $violations
}

# Function to check all Rust files
check_all_files() {
    local total_violations=0
    local files_checked=0
    
    echo -e "\n${YELLOW}Scanning Rust files for layer violations...${NC}"
    
    # Find all Rust files
    while IFS= read -r -d '' file; do
        ((files_checked++))
        if ! check_file_imports "$file"; then
            ((total_violations+=$?))
        fi
    done < <(find "$RUST_CORE" -name "*.rs" -type f -print0)
    
    echo -e "\n${BLUE}Summary:${NC}"
    echo "Files checked: $files_checked"
    echo "Violations found: $total_violations"
    
    return $total_violations
}

# Function to check specific known violations
check_known_violations() {
    echo -e "\n${YELLOW}Checking known problematic imports...${NC}"
    
    local violations=0
    
    # Known violations to check
    declare -a KNOWN_VIOLATIONS=(
        "risk.*order_management"
        "data.*from.*ml"
        "strategies.*from.*execution"
        "ml.*from.*trading"
        "data_ingestion.*from.*risk"
    )
    
    for pattern in "${KNOWN_VIOLATIONS[@]}"; do
        local source=$(echo "$pattern" | cut -d'.' -f1)
        local target=$(echo "$pattern" | awk -F'from.*' '{print $2}')
        
        echo -e "\n${YELLOW}Checking: $source should not import $target${NC}"
        
        if rg "use.*$target" "$RUST_CORE/crates/$source" -t rust --no-heading 2>/dev/null; then
            echo -e "${RED}❌ VIOLATION FOUND${NC}"
            ((violations++))
        else
            echo -e "${GREEN}✓ OK${NC}"
        fi
    done
    
    return $violations
}

# Function to generate layer dependency graph
generate_dependency_graph() {
    echo -e "\n${BLUE}Generating layer dependency graph...${NC}"
    
    local graph_file="/home/hamster/bot4/.mcp/layer_dependencies.dot"
    
    cat > "$graph_file" << 'EOF'
digraph LayerDependencies {
    rankdir=BT;
    node [shape=box, style=filled];
    
    // Define layers
    "Layer 0: Infrastructure" [fillcolor=lightblue];
    "Layer 1: Data" [fillcolor=lightgreen];
    "Layer 2: Risk" [fillcolor=yellow];
    "Layer 3: ML" [fillcolor=orange];
    "Layer 4: Strategies" [fillcolor=pink];
    "Layer 5: Execution" [fillcolor=lightgray];
    "Layer 6: Integration" [fillcolor=purple, fontcolor=white];
    
    // Allowed dependencies (bottom-up)
    "Layer 1: Data" -> "Layer 0: Infrastructure";
    "Layer 2: Risk" -> "Layer 0: Infrastructure";
    "Layer 2: Risk" -> "Layer 1: Data";
    "Layer 3: ML" -> "Layer 0: Infrastructure";
    "Layer 3: ML" -> "Layer 1: Data";
    "Layer 3: ML" -> "Layer 2: Risk";
    "Layer 4: Strategies" -> "Layer 0: Infrastructure";
    "Layer 4: Strategies" -> "Layer 1: Data";
    "Layer 4: Strategies" -> "Layer 2: Risk";
    "Layer 4: Strategies" -> "Layer 3: ML";
    "Layer 5: Execution" -> "Layer 0: Infrastructure";
    "Layer 5: Execution" -> "Layer 1: Data";
    "Layer 5: Execution" -> "Layer 2: Risk";
    "Layer 5: Execution" -> "Layer 3: ML";
    "Layer 5: Execution" -> "Layer 4: Strategies";
    "Layer 6: Integration" -> "Layer 0: Infrastructure";
    "Layer 6: Integration" -> "Layer 1: Data";
    "Layer 6: Integration" -> "Layer 2: Risk";
    "Layer 6: Integration" -> "Layer 3: ML";
    "Layer 6: Integration" -> "Layer 4: Strategies";
    "Layer 6: Integration" -> "Layer 5: Execution";
}
EOF
    
    echo -e "${GREEN}✓ Dependency graph saved to: $graph_file${NC}"
    
    # Generate visual if graphviz is installed
    if command -v dot &> /dev/null; then
        dot -Tpng "$graph_file" -o "/home/hamster/bot4/.mcp/layer_dependencies.png"
        echo -e "${GREEN}✓ Visual graph saved to: /home/hamster/bot4/.mcp/layer_dependencies.png${NC}"
    fi
}

# Main execution
main() {
    local exit_code=0
    
    # Ensure directory exists
    if [ ! -d "$RUST_CORE" ]; then
        echo -e "${RED}Error: rust_core directory not found at $RUST_CORE${NC}"
        exit 1
    fi
    
    # Check for violations
    if ! check_known_violations; then
        exit_code=1
    fi
    
    if ! check_all_files; then
        exit_code=1
    fi
    
    # Generate dependency graph
    generate_dependency_graph
    
    # Update shared context
    if [ -f "/home/hamster/bot4/.mcp/shared_context.json" ]; then
        echo "{\"layer_check\": {\"timestamp\": \"$(date -Iseconds)\", \"violations\": $exit_code}}" > /home/hamster/bot4/.mcp/last_layer_check.json
    fi
    
    # Final status
    echo -e "\n----------------------------------------"
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ No layer violations detected${NC}"
        echo "Architecture integrity maintained"
    else
        echo -e "${RED}⚠️  LAYER VIOLATIONS DETECTED!${NC}"
        echo "Action required: Refactor imports to respect layer boundaries"
        echo "Remember: Layer N can only import from layers 0 to N-1"
    fi
    
    exit $exit_code
}

# Run main
main