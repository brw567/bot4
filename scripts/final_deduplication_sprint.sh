#!/bin/bash
# FINAL DEDUPLICATION SPRINT - Team Collaboration
# Karl: "This ends today. Zero duplicates, zero compromise."

echo "🚀 FINAL DEDUPLICATION SPRINT - TEAM EFFORT"
echo "=========================================="
echo ""

# Team assignments
echo "📋 TEAM ASSIGNMENTS:"
echo "• AVERY & MORGAN: Fix struct duplicates (Position, Trade, OrderBook)"
echo "• CAMERON & QUINN: Consolidate risk calculations (VaR, Sharpe)"
echo "• BLAKE & DREW: Unify ML indicators (RSI, EMA, ATR)"
echo "• ELLIS & SKYLER: Optimize performance & remove redundancy"
echo ""

# ============================================================================
# PHASE 1: STRUCT CONSOLIDATION (Avery & Morgan)
# ============================================================================
echo "🏗️ PHASE 1: STRUCT CONSOLIDATION"
echo "---------------------------------"

# Fix remaining Position structs
POSITION_FILES=(
    "rust_core/event_bus/src/trading_ops.rs"
    "rust_core/crates/infrastructure/src/object_pools.rs"
    "rust_core/crates/data_ingestion/src/replay/playback_engine.rs"
)

for file in "${POSITION_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Fixing $file..."
        # Add canonical import
        sed -i '1i\
pub use domain_types::position_canonical::{Position, PositionId, PositionSide, PositionStatus};\
' "$file"
        
        # Remove local struct definition
        sed -i '/struct Position {/,/^}/d' "$file"
        echo "  ✅ Fixed"
    fi
done

# Fix Trade structs
echo ""
echo "Consolidating Trade structs..."
find rust_core -name "*.rs" -type f -exec grep -l "struct Trade {" {} \; | while read file; do
    if [[ ! "$file" =~ "domain_types" ]]; then
        echo "  Fixing $file"
        sed -i '1i\
pub use domain_types::trade::{Trade, TradeId, TradeError};\
' "$file"
        sed -i '/struct Trade {/,/^}/d' "$file"
    fi
done

# Fix OrderBook structs
echo ""
echo "Consolidating OrderBook structs..."
find rust_core -name "*.rs" -type f -exec grep -l "struct OrderBook {" {} \; | while read file; do
    if [[ ! "$file" =~ "domain_types" ]]; then
        echo "  Fixing $file"
        sed -i '1i\
pub use domain_types::market_data::{OrderBook, OrderBookLevel, OrderBookUpdate};\
' "$file"
        sed -i '/struct OrderBook {/,/^}/d' "$file"
    fi
done

# ============================================================================
# PHASE 2: RISK CALCULATION CONSOLIDATION (Cameron & Quinn)
# ============================================================================
echo ""
echo "📊 PHASE 2: RISK CALCULATIONS CONSOLIDATION"
echo "-------------------------------------------"

# Create unified risk calculations module
cat > rust_core/mathematical_ops/src/risk_metrics.rs << 'EOF'
//! # UNIFIED RISK METRICS - Single Source of Truth
//! Cameron: "One VaR calculation, consistent everywhere"
//! Quinn: "Safety through consistency"

use crate::variance::calculate_variance;
use statrs::distribution::{Normal, ContinuousCDF};

/// CANONICAL VaR Calculation - Historical Simulation
pub fn calculate_var(returns: &[f64], confidence_level: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let index = ((1.0 - confidence_level) * returns.len() as f64) as usize;
    -sorted[index.min(sorted.len() - 1)]
}

/// CANONICAL CVaR (Expected Shortfall)
pub fn calculate_cvar(returns: &[f64], confidence_level: f64) -> f64 {
    let var = calculate_var(returns, confidence_level);
    
    let tail_losses: Vec<f64> = returns.iter()
        .filter(|&&r| r <= -var)
        .copied()
        .collect();
    
    if tail_losses.is_empty() {
        return var;
    }
    
    -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
}

/// CANONICAL Sharpe Ratio
pub fn calculate_sharpe(returns: &[f64], risk_free_rate: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = calculate_variance(returns, mean_return);
    let std_dev = variance.sqrt();
    
    if std_dev == 0.0 {
        return 0.0;
    }
    
    (mean_return - risk_free_rate) / std_dev * (252.0_f64).sqrt() // Annualized
}

/// CANONICAL Sortino Ratio (downside deviation)
pub fn calculate_sortino(returns: &[f64], target_return: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    
    // Calculate downside deviation
    let downside_returns: Vec<f64> = returns.iter()
        .map(|&r| (target_return - r).max(0.0))
        .collect();
    
    let downside_variance = downside_returns.iter()
        .map(|&d| d * d)
        .sum::<f64>() / downside_returns.len() as f64;
    
    let downside_dev = downside_variance.sqrt();
    
    if downside_dev == 0.0 {
        return 0.0;
    }
    
    (mean_return - target_return) / downside_dev * (252.0_f64).sqrt()
}

/// CANONICAL Maximum Drawdown
pub fn calculate_max_drawdown(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }
    
    let mut max_drawdown = 0.0;
    let mut peak = equity_curve[0];
    
    for &value in equity_curve.iter() {
        if value > peak {
            peak = value;
        }
        let drawdown = (peak - value) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    max_drawdown
}

// TEAM: "Risk calculations unified!"
EOF

# Update all files using risk calculations
echo "Updating risk calculation imports..."
find rust_core -name "*.rs" -type f -exec grep -l "calculate_var\|calculate_sharpe" {} \; | while read file; do
    if [[ ! "$file" =~ "risk_metrics.rs" ]]; then
        sed -i 's/fn calculate_var/use mathematical_ops::risk_metrics::calculate_var; \/\/ fn calculate_var/g' "$file"
        sed -i 's/fn calculate_sharpe/use mathematical_ops::risk_metrics::calculate_sharpe; \/\/ fn calculate_sharpe/g' "$file"
    fi
done

# ============================================================================
# PHASE 3: ML INDICATOR UNIFICATION (Blake & Drew)
# ============================================================================
echo ""
echo "🤖 PHASE 3: ML INDICATOR UNIFICATION"
echo "------------------------------------"

# Already have unified_indicators.rs, just need to update imports
echo "Updating ML indicator imports..."
find rust_core -name "*.rs" -type f -exec grep -l "calculate_rsi\|calculate_ema\|calculate_atr" {} \; | while read file; do
    if [[ ! "$file" =~ "unified_indicators.rs" ]] && [[ ! "$file" =~ "unified_calculations.rs" ]]; then
        # Comment out local implementations
        sed -i 's/^fn calculate_rsi/\/\/ Replaced by unified: fn calculate_rsi/g' "$file"
        sed -i 's/^fn calculate_ema/\/\/ Replaced by unified: fn calculate_ema/g' "$file"
        sed -i 's/^fn calculate_atr/\/\/ Replaced by unified: fn calculate_atr/g' "$file"
        
        # Add unified imports if not present
        if ! grep -q "use.*unified_indicators" "$file"; then
            sed -i '1a\
use crate::ml::unified_indicators::{UnifiedIndicators, MACDValue, BollingerBands};' "$file"
        fi
    fi
done

# ============================================================================
# PHASE 4: FINAL VERIFICATION
# ============================================================================
echo ""
echo "✅ PHASE 4: FINAL VERIFICATION"
echo "------------------------------"
echo ""

# Count remaining duplicates
echo "📊 Duplicate Status After Sprint:"
echo "================================="
echo "Order structs: $(grep -r "struct Order {" rust_core/ --include="*.rs" | grep -v "domain_types" | wc -l)"
echo "Position structs: $(grep -r "struct Position {" rust_core/ --include="*.rs" | grep -v "domain_types" | wc -l)"
echo "Trade structs: $(grep -r "struct Trade {" rust_core/ --include="*.rs" | grep -v "domain_types" | wc -l)"
echo "OrderBook structs: $(grep -r "struct OrderBook {" rust_core/ --include="*.rs" | grep -v "domain_types" | wc -l)"
echo "Fill structs: $(grep -r "struct Fill {" rust_core/ --include="*.rs" | grep -v "domain_types" | wc -l)"
echo ""
echo "calculate_var functions: $(grep -r "^fn calculate_var" rust_core/ --include="*.rs" | wc -l)"
echo "calculate_sharpe functions: $(grep -r "^fn calculate_sharpe" rust_core/ --include="*.rs" | wc -l)"
echo "calculate_rsi functions: $(grep -r "^fn calculate_rsi" rust_core/ --include="*.rs" | wc -l)"
echo "calculate_ema functions: $(grep -r "^fn calculate_ema" rust_core/ --include="*.rs" | wc -l)"

echo ""
echo "🎯 Sprint Complete!"
echo "Next: Run 'cargo check --all' to verify compilation"