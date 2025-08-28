#!/bin/bash
# Script to eliminate Order struct duplicates
# Morgan: "Replace all Order duplicates with canonical version"

echo "🔧 Eliminating Order struct duplicates..."

# Files that should use the canonical Order from domain_types
FILES_TO_UPDATE=(
    "rust_core/crates/order_management/src/order.rs"
    "rust_core/crates/risk/src/optimal_execution.rs"
    "rust_core/crates/trading_engine/src/fees_slippage.rs"
    "rust_core/domain/entities/order.rs"
    "rust_core/crates/trading_engine/src/orders/oco.rs"
    "rust_core/crates/infrastructure/src/position_reconciliation.rs"
    "rust_core/crates/infrastructure/src/memory/pools_upgraded.rs"
    "rust_core/crates/infrastructure/src/memory/safe_pools.rs"
    "rust_core/crates/infrastructure/src/memory/pools.rs"
    "rust_core/crates/infrastructure/src/object_pools.rs"
    "rust_core/crates/infrastructure/src/circuit_breaker_layer_integration.rs"
)

for file in "${FILES_TO_UPDATE[@]}"; do
    if [ -f "$file" ]; then
        echo "Updating $file..."
        
        # Check if it has its own Order struct definition
        if grep -q "pub struct Order {" "$file"; then
            # Backup original
            cp "$file" "${file}.backup"
            
            # Get the module name from the file
            module=$(basename $(dirname "$file"))
            
            # Create updated file
            cat > "${file}.tmp" << 'EOF'
//! Module uses canonical Order type from domain_types
//! Avery: "Single source of truth for Order struct"

pub use domain_types::order::{
    Order, OrderId, OrderSide, OrderType, OrderStatus, TimeInForce,
    OrderError, Fill, FillId
};
pub use domain_types::{Price, Quantity, Symbol, Exchange};

// Re-export for backward compatibility
pub type OrderResult<T> = Result<T, OrderError>;

EOF
            
            # Extract any module-specific extensions (traits, impl blocks)
            # that don't define the struct itself
            awk '
                /^pub struct Order \{/,/^}/ { next }
                /^#\[derive.*\]$/ { 
                    getline
                    if ($0 ~ /^pub struct Order/) next
                    else print prev"\n"$0
                    next
                }
                /^impl Order \{/,/^}/ { print; next }
                /^impl.*for Order/,/^}/ { print; next }
                /^pub trait.*Order/,/^}/ { print; next }
                { 
                    if (!/^use .*Order/ && !/^\/\/ Order types/ && !/^\/\/ Designed for/) print 
                }
            ' "$file" >> "${file}.tmp"
            
            # Replace original if tmp file was created successfully
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
echo "📝 Updating Cargo.toml files to include domain_types dependency..."

# Update Cargo.toml files
CARGO_FILES=(
    "rust_core/crates/order_management/Cargo.toml"
    "rust_core/crates/risk/Cargo.toml"
    "rust_core/crates/trading_engine/Cargo.toml"
    "rust_core/crates/infrastructure/Cargo.toml"
)

for cargo_file in "${CARGO_FILES[@]}"; do
    if [ -f "$cargo_file" ]; then
        # Check if domain_types is already a dependency
        if ! grep -q "domain_types" "$cargo_file"; then
            echo "Adding domain_types to $cargo_file..."
            
            # Add domain_types dependency after [dependencies]
            sed -i '/\[dependencies\]/a\domain_types = { path = "../../domain_types" }' "$cargo_file"
            echo "  ✅ Updated $cargo_file"
        fi
    fi
done

echo ""
echo "🔍 Checking remaining Order duplicates..."
echo ""
grep -r "pub struct Order {" rust_core/ --include="*.rs" | grep -v "domain_types/src/order" | wc -l | xargs -I {} echo "Remaining Order struct duplicates: {}"

echo ""
echo "✅ Order duplicate elimination complete!"
echo "Next: Run 'cargo check' to verify all imports are correct"