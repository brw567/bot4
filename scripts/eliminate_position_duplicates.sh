#!/bin/bash
# Script to eliminate Position struct duplicates
# Cameron: "One Position struct to rule them all"

echo "🔧 Eliminating Position struct duplicates..."

# Files that have Position struct duplicates
FILES_TO_UPDATE=(
    "rust_core/crates/order_management/src/position.rs"
    "rust_core/crates/risk/src/unified_types.rs"
    "rust_core/crates/risk/src/clamps.rs"
    "rust_core/crates/trading_engine/src/liquidation_engine.rs"
    "rust_core/crates/trading_engine/src/transactions/compensator.rs"
    "rust_core/crates/infrastructure/src/circuit_breaker_layer_integration.rs"
    "rust_core/crates/infrastructure/src/position_reconciliation.rs"
)

for file in "${FILES_TO_UPDATE[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        
        # Check if it has its own Position struct
        if grep -q "pub struct Position {" "$file" || grep -q "^struct Position {" "$file"; then
            # Backup
            cp "$file" "${file}.backup"
            
            # Create header with canonical import
            cat > "${file}.tmp" << 'EOF'
//! Module uses canonical Position type from domain_types
//! Cameron: "Single source of truth for Position struct"

pub use domain_types::position_canonical::{
    Position, PositionId, PositionSide, PositionStatus,
    PositionError, PositionUpdate
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type PositionResult<T> = Result<T, PositionError>;

EOF
            
            # Extract non-struct content (impl blocks, traits, etc)
            awk '
                /^pub struct Position \{/,/^}/ { next }
                /^struct Position \{/,/^}/ { next }
                /^#\[derive.*\]$/ { 
                    getline
                    if ($0 ~ /struct Position/) next
                    else print prev"\n"$0
                    next
                }
                /^impl Position \{/,/^}/ { print; next }
                /^impl.*for Position/,/^}/ { print; next }
                { 
                    if (!/^use .*Position/ && !/Module uses canonical/) print 
                }
            ' "$file" >> "${file}.tmp"
            
            # Replace if successful
            if [ -s "${file}.tmp" ]; then
                mv "${file}.tmp" "$file"
                echo "  ✅ Updated $file"
            else
                echo "  ⚠️  Failed to update $file, restoring backup"
                mv "${file}.backup" "$file"
            fi
        fi
    fi
done

echo ""
echo "📝 Updating Cargo.toml files..."

# Update Cargo.toml files if needed
CARGO_FILES=(
    "rust_core/crates/order_management/Cargo.toml"
    "rust_core/crates/risk/Cargo.toml"
    "rust_core/crates/trading_engine/Cargo.toml"
    "rust_core/crates/infrastructure/Cargo.toml"
)

for cargo_file in "${CARGO_FILES[@]}"; do
    if [ -f "$cargo_file" ] && ! grep -q "domain_types" "$cargo_file"; then
        sed -i '/\[dependencies\]/a\domain_types = { path = "../../domain_types" }' "$cargo_file"
        echo "  ✅ Updated $cargo_file"
    fi
done

echo ""
echo "🔍 Checking remaining Position duplicates..."
grep -r "pub struct Position {" rust_core/ --include="*.rs" | grep -v "position_canonical.rs" | wc -l | xargs -I {} echo "Remaining Position struct duplicates: {}"

echo "✅ Position duplicate elimination complete!"