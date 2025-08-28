#!/bin/bash
# Bot4 Duplication Detection Script
# Uses AST analysis and pattern matching to find duplicate code
# Version: 2.0 - Enhanced for multi-agent architecture

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RUST_CORE_DIR="/home/hamster/bot4/rust_core"
THRESHOLD=${THRESHOLD:-0.8}  # Similarity threshold
COMPONENT=${1:-"all"}

echo -e "${BLUE}🔍 Bot4 Duplication Detector v2.0${NC}"
echo "Checking component: $COMPONENT"
echo "Similarity threshold: $THRESHOLD"
echo "----------------------------------------"

# Function to check for struct duplicates
check_struct_duplicates() {
    local struct_name=$1
    echo -e "\n${YELLOW}Checking for duplicate structs: $struct_name${NC}"
    
    # Use grep command (rg or grep)
    local count
    if command -v rg &> /dev/null; then
        count=$(rg "struct\s+$struct_name\s*\{" "$RUST_CORE_DIR" -t rust -c 2>/dev/null | wc -l || echo 0)
    else
        count=$(grep -r "struct[[:space:]]\+$struct_name[[:space:]]*{" "$RUST_CORE_DIR" --include='*.rs' 2>/dev/null | wc -l || echo 0)
    fi
    
    if [ "$count" -gt 1 ]; then
        echo -e "${RED}❌ DUPLICATE FOUND: $count instances of 'struct $struct_name'${NC}"
        if command -v rg &> /dev/null; then
            rg "struct\s+$struct_name\s*\{" "$RUST_CORE_DIR" -t rust --line-number --no-heading | head -5
        else
            grep -r "struct[[:space:]]\+$struct_name[[:space:]]*{" "$RUST_CORE_DIR" --include='*.rs' -n | head -5
        fi
        return 1
    else
        echo -e "${GREEN}✓ No duplicates found for struct $struct_name${NC}"
        return 0
    fi
}

# Function to check for function duplicates
check_function_duplicates() {
    local func_name=$1
    echo -e "\n${YELLOW}Checking for duplicate functions: $func_name${NC}"
    
    # Count occurrences
    local count
    if command -v rg &> /dev/null; then
        count=$(rg "fn\s+$func_name\s*\(" "$RUST_CORE_DIR" -t rust -c 2>/dev/null | wc -l || echo 0)
    else
        count=$(grep -r "fn[[:space:]]\+$func_name[[:space:]]*(" "$RUST_CORE_DIR" --include='*.rs' 2>/dev/null | wc -l || echo 0)
    fi
    
    if [ "$count" -gt 1 ]; then
        echo -e "${RED}❌ DUPLICATE FOUND: $count instances of 'fn $func_name'${NC}"
        if command -v rg &> /dev/null; then
            rg "fn\s+$func_name\s*\(" "$RUST_CORE_DIR" -t rust --line-number --no-heading | head -5
        else
            grep -r "fn[[:space:]]\+$func_name[[:space:]]*(" "$RUST_CORE_DIR" --include='*.rs' -n | head -5
        fi
        return 1
    else
        echo -e "${GREEN}✓ No duplicates found for fn $func_name${NC}"
        return 0
    fi
}

# Function to check known problematic duplicates
check_known_duplicates() {
    echo -e "\n${BLUE}Checking known problematic duplicates...${NC}"
    
    local has_duplicates=0
    
    # Known duplicate structs
    local known_structs=("Order" "Position" "Trade" "Candle" "OrderBook" "Fill" "Price" "Quantity")
    for struct in "${known_structs[@]}"; do
        if ! check_struct_duplicates "$struct"; then
            has_duplicates=1
        fi
    done
    
    # Known duplicate functions
    local known_functions=("calculate_correlation" "calculate_var" "calculate_ema" "calculate_rsi" "calculate_atr" "calculate_sharpe")
    for func in "${known_functions[@]}"; do
        if ! check_function_duplicates "$func"; then
            has_duplicates=1
        fi
    done
    
    return $has_duplicates
}

# Function to analyze specific component
analyze_component() {
    local component=$1
    
    case "$component" in
        "Order"|"order")
            check_struct_duplicates "Order"
            ;;
        "correlation")
            check_function_duplicates "calculate_correlation"
            ;;
        "var"|"VaR")
            check_function_duplicates "calculate_var"
            ;;
        "all")
            check_known_duplicates
            ;;
        *)
            # Try both struct and function
            check_struct_duplicates "$component" || check_function_duplicates "$component"
            ;;
    esac
}

# Function to generate duplication report
generate_report() {
    echo -e "\n${BLUE}Generating comprehensive duplication report...${NC}"
    
    local report_file="/home/hamster/bot4/.mcp/duplication_report_$(date +%Y%m%d_%H%M%S).json"
    
    # Use rust-code-analysis if available
    if command -v rust-code-analysis &> /dev/null; then
        rust-code-analysis --metrics loc,cyclomatic,cognitive --output json "$RUST_CORE_DIR" > "$report_file"
        echo -e "${GREEN}✓ Advanced analysis complete: $report_file${NC}"
    fi
    
    # Count total duplications
    local struct_dups
    local func_dups
    if command -v rg &> /dev/null; then
        struct_dups=$(rg "struct\s+\w+\s*\{" "$RUST_CORE_DIR" -t rust -o | sort | uniq -c | awk '$1>1' | wc -l)
        func_dups=$(rg "fn\s+\w+\s*\(" "$RUST_CORE_DIR" -t rust -o | sort | uniq -c | awk '$1>1' | wc -l)
    else
        struct_dups=$(grep -rho "struct[[:space:]]\+[[:alnum:]_]\+[[:space:]]*{" "$RUST_CORE_DIR" --include='*.rs' | sort | uniq -c | awk '$1>1' | wc -l)
        func_dups=$(grep -rho "fn[[:space:]]\+[[:alnum:]_]\+[[:space:]]*(" "$RUST_CORE_DIR" --include='*.rs' | sort | uniq -c | awk '$1>1' | wc -l)
    fi
    
    echo -e "\n${YELLOW}Summary:${NC}"
    echo "- Duplicate struct definitions: $struct_dups"
    echo "- Duplicate function definitions: $func_dups"
    echo "- Total duplications: $((struct_dups + func_dups))"
    
    # Update shared context
    if [ -f "/home/hamster/bot4/.mcp/shared_context.json" ]; then
        local total_dups=$((struct_dups + func_dups))
        # Update the duplication count in shared context (simplified update)
        echo "{\"duplication_check\": {\"timestamp\": \"$(date -Iseconds)\", \"total\": $total_dups, \"structs\": $struct_dups, \"functions\": $func_dups}}" > /home/hamster/bot4/.mcp/last_duplication_check.json
    fi
}

# Main execution
main() {
    local exit_code=0
    
    # Ensure we're in the right directory
    if [ ! -d "$RUST_CORE_DIR" ]; then
        echo -e "${RED}Error: rust_core directory not found at $RUST_CORE_DIR${NC}"
        exit 1
    fi
    
    # Check for required tools - use grep if rg not available
    if command -v rg &> /dev/null; then
        GREP_CMD="rg"
        GREP_ARGS="-t rust"
    else
        echo -e "${YELLOW}Warning: ripgrep not found, using grep (slower)${NC}"
        GREP_CMD="grep -r"
        GREP_ARGS="--include='*.rs'"
    fi
    
    # Perform analysis
    if ! analyze_component "$COMPONENT"; then
        exit_code=1
    fi
    
    # Generate report if checking all
    if [ "$COMPONENT" == "all" ]; then
        generate_report
    fi
    
    # Final status
    echo -e "\n----------------------------------------"
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ No duplications found for $COMPONENT${NC}"
        echo "Safe to proceed with implementation"
    else
        echo -e "${RED}⚠️  DUPLICATIONS DETECTED!${NC}"
        echo "Action required: Refactor to use existing implementations"
        echo "Check domain_types, mathematical_ops, and abstractions crates"
    fi
    
    exit $exit_code
}

# Run main function
main