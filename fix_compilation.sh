#!/bin/bash
# Comprehensive Compilation Fix Script
# Team: Alex, Sam, Jordan, and Full Team
# This script systematically fixes all compilation issues

echo "Starting comprehensive compilation fix..."

# Fix 1: Replace all to_f64() calls with proper Decimal conversions
echo "Fixing Decimal to_f64() conversions..."
find rust_core -name "*.rs" -type f -exec sed -i 's/\.to_f64()\.unwrap_or(0\.0)/\.to_f64_retain()/g' {} \;
find rust_core -name "*.rs" -type f -exec sed -i 's/\.to_f64()\.unwrap_or([0-9.]*)/\.to_f64_retain()/g' {} \;

# Fix 2: Add missing anyhow::bail! macro
echo "Adding bail! macro import to ML crate..."
sed -i '1i use anyhow::bail;' rust_core/crates/ml/src/backtesting/walk_forward.rs

# Fix 3: Fix Axis import in ML crate
echo "Fixing Axis import..."
sed -i 's/Axis(0)/ndarray::Axis(0)/g' rust_core/crates/ml/src/backtesting/walk_forward.rs

# Fix 4: Add missing dependencies to ML Cargo.toml
echo "Adding missing ML dependencies..."
cat >> rust_core/crates/ml/Cargo.toml << 'EOF'
ndarray = { version = "0.15", features = ["serde"] }
ndarray-rand = "0.14"
EOF

# Fix 5: Fix if let guards (use stable Rust pattern)
echo "Fixing experimental if let guards..."
sed -i 's/if let Some(price) = order.price/if order.price.is_some()/g' rust_core/crates/order_management/src/manager.rs

# Fix 6: Add sqlx decimal feature
echo "Adding sqlx decimal support..."
sed -i '/sqlx = {/s/features = \[/features = ["decimal", /' rust_core/crates/order_management/Cargo.toml

# Fix 7: Export GlobalPoolStats from infrastructure
echo "Exporting GlobalPoolStats..."
cat >> rust_core/crates/infrastructure/src/lib.rs << 'EOF'

// Re-export GlobalPoolStats
pub use object_pools::GlobalPoolStats;
EOF

# Fix 8: Fix moved value errors
echo "Fixing moved value errors..."
sed -i 's/exchange_name/\&exchange_name/g' rust_core/crates/order_management/src/router.rs

# Fix 9: Fix WebSocket reliable_client compilation
echo "Fixing WebSocket reliable_client..."
sed -i '/let mut outbound_rx = self.outbound_rx.write().take()/a\
        .ok_or(ReliableWebSocketError::ChannelClosed)?;' rust_core/crates/websocket/src/reliable_client.rs

echo "Running cargo build to verify fixes..."
cd rust_core
cargo build --all-features 2>&1 | tail -50

echo "Compilation fix script complete!"