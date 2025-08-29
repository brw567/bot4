#!/bin/bash

# Bot4 Paper Trading Startup Script
# Team: Full 8-Agent ULTRATHINK Collaboration
# Research Applied: Simulation patterns, backtesting best practices
# Target: 60-90 day validation with live data from 5 exchanges

set -e

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        BOT4 PAPER TRADING - LIVE DATA SIMULATION            ║${NC}"
echo -e "${BLUE}║           Team: Full 8-Agent ULTRATHINK                     ║${NC}"
echo -e "${BLUE}║           Target: 60-90 Day Validation                      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Configuration
CONFIG_FILE="configs/paper_trading_config.toml"
LOG_DIR="/home/hamster/bot4/paper_trading_logs"
REPORT_DIR="/home/hamster/bot4/paper_trading_reports"
DATA_DIR="/home/hamster/bot4/paper_trading_data"

# Create directories
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$DATA_DIR"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Paper trading config not found at $CONFIG_FILE${NC}"
    exit 1
fi

# Check if rust binary exists
if [ ! -f "rust_core/target/release/bot4-main" ]; then
    echo -e "${YELLOW}Building Bot4 in release mode...${NC}"
    cd rust_core
    RUSTFLAGS="-C target-cpu=native -C opt-level=3" cargo build --release --all-features
    cd ..
fi

# Export environment variables for paper trading
export RUST_LOG="info,bot4=debug"
export RUST_BACKTRACE=1
export BOT4_MODE="paper_trading"
export BOT4_CONFIG="$CONFIG_FILE"
export BOT4_LOG_DIR="$LOG_DIR"
export BOT4_REPORT_DIR="$REPORT_DIR"
export BOT4_DATA_DIR="$DATA_DIR"

# Performance optimizations
export MALLOC_ARENA_MAX=2
export MIMALLOC_LARGE_OS_PAGES=1
export MIMALLOC_RESERVE_HUGE_OS_PAGES=4

echo -e "\n${GREEN}═══ Paper Trading Configuration ═══${NC}"
echo "Config: $CONFIG_FILE"
echo "Logs: $LOG_DIR"
echo "Reports: $REPORT_DIR"
echo "Data: $DATA_DIR"
echo "Mode: PAPER_TRADING (live data, simulated execution)"

echo -e "\n${GREEN}═══ Exchange Connections ═══${NC}"
echo "1. Binance Testnet: wss://testnet.binance.vision/ws"
echo "2. Coinbase Sandbox: wss://ws-feed-public.sandbox.exchange.coinbase.com"
echo "3. Kraken Beta: wss://beta-ws.kraken.com"
echo "4. OKX Demo: wss://wspap.okx.com:8443/ws/v5/public"
echo "5. Bybit Testnet: wss://stream-testnet.bybit.com/v5/public/spot"

echo -e "\n${GREEN}═══ Risk Limits (Paper Trading) ═══${NC}"
echo "Max Position Size: 5%"
echo "Max Drawdown: 10%"
echo "Max Daily Loss: 3%"
echo "Kelly Fraction Cap: 15% (conservative)"
echo "Max Correlation: 0.6"

echo -e "\n${GREEN}═══ Strategies Enabled ═══${NC}"
echo "• Market Making (Avellaneda-Stoikov)"
echo "• Statistical Arbitrage (Pairs/Cointegration)"
echo "• Momentum (Trend Following)"
echo "• Mean Reversion (Bollinger/RSI)"
echo "• Game Theory Routing (Nash Equilibrium)"

echo -e "\n${GREEN}═══ Performance Tracking ═══${NC}"
echo "• Sharpe Ratio (target >2.0)"
echo "• Max Drawdown (limit 15%)"
echo "• Win Rate (target >60%)"
echo "• Profit Factor (target >1.5)"
echo "• Kelly Criterion Validation"
echo "• VaR/CVaR 95% Monitoring"

echo -e "\n${YELLOW}Starting paper trading engine...${NC}"

# Function to handle shutdown
cleanup() {
    echo -e "\n${YELLOW}Shutting down paper trading...${NC}"
    kill $PAPER_TRADING_PID 2>/dev/null || true
    
    # Generate final report
    echo -e "${YELLOW}Generating final report...${NC}"
    ./scripts/generate_paper_trading_report.sh
    
    echo -e "${GREEN}Paper trading stopped. Reports saved to $REPORT_DIR${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the paper trading engine
echo -e "\n${GREEN}Launching Bot4 Paper Trading Engine...${NC}"

# Run with optimal CPU affinity and NUMA settings
taskset -c 0-7 \
numactl --cpunodebind=0 --membind=0 \
./rust_core/target/release/bot4-main \
    --mode paper-trading \
    --config "$CONFIG_FILE" \
    --log-dir "$LOG_DIR" \
    --report-dir "$REPORT_DIR" \
    --data-dir "$DATA_DIR" \
    2>&1 | tee "$LOG_DIR/paper_trading_$(date +%Y%m%d_%H%M%S).log" &

PAPER_TRADING_PID=$!

echo -e "\n${GREEN}═══ Paper Trading Started ═══${NC}"
echo "Process ID: $PAPER_TRADING_PID"
echo "Start Time: $(date)"
echo "Initial Capital: \$100,000 (simulated)"

# Monitor initial startup
sleep 5

# Check if process is still running
if ! kill -0 $PAPER_TRADING_PID 2>/dev/null; then
    echo -e "${RED}Error: Paper trading engine failed to start${NC}"
    tail -n 50 "$LOG_DIR/paper_trading_$(date +%Y%m%d)*.log"
    exit 1
fi

echo -e "\n${GREEN}═══ Live Monitoring ═══${NC}"
echo "Dashboard: http://localhost:8080/paper-trading"
echo "Metrics: http://localhost:9090/metrics"
echo "Logs: tail -f $LOG_DIR/paper_trading_*.log"

# Function to display real-time stats
show_stats() {
    while kill -0 $PAPER_TRADING_PID 2>/dev/null; do
        clear
        echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║               PAPER TRADING - LIVE STATISTICS               ║${NC}"
        echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
        
        # Get latest metrics (simplified - would normally query the actual system)
        echo -e "\n${GREEN}Performance Metrics:${NC}"
        echo "├─ Sharpe Ratio: $(curl -s localhost:9090/metrics | grep sharpe_ratio | awk '{print $2}' || echo "N/A")"
        echo "├─ Win Rate: $(curl -s localhost:9090/metrics | grep win_rate | awk '{print $2}' || echo "N/A")%"
        echo "├─ Current Drawdown: $(curl -s localhost:9090/metrics | grep current_drawdown | awk '{print $2}' || echo "N/A")%"
        echo "├─ Total Trades: $(curl -s localhost:9090/metrics | grep total_trades | awk '{print $2}' || echo "N/A")"
        echo "└─ Decision Latency: $(curl -s localhost:9090/metrics | grep decision_latency_us | awk '{print $2}' || echo "N/A")μs"
        
        echo -e "\n${GREEN}Exchange Status:${NC}"
        echo "├─ Binance: $(curl -s localhost:8080/health/exchanges | jq -r '.binance.status' 2>/dev/null || echo "Checking...")"
        echo "├─ Coinbase: $(curl -s localhost:8080/health/exchanges | jq -r '.coinbase.status' 2>/dev/null || echo "Checking...")"
        echo "├─ Kraken: $(curl -s localhost:8080/health/exchanges | jq -r '.kraken.status' 2>/dev/null || echo "Checking...")"
        echo "├─ OKX: $(curl -s localhost:8080/health/exchanges | jq -r '.okx.status' 2>/dev/null || echo "Checking...")"
        echo "└─ Bybit: $(curl -s localhost:8080/health/exchanges | jq -r '.bybit.status' 2>/dev/null || echo "Checking...")"
        
        echo -e "\n${YELLOW}Press Ctrl+C to stop paper trading${NC}"
        echo -e "Next update in 30 seconds..."
        
        sleep 30
    done
}

# Ask user for monitoring preference
echo -e "\n${YELLOW}Would you like to monitor live statistics? (y/n)${NC}"
read -r MONITOR_CHOICE

if [[ "$MONITOR_CHOICE" == "y" || "$MONITOR_CHOICE" == "Y" ]]; then
    show_stats
else
    echo -e "\n${GREEN}Paper trading running in background.${NC}"
    echo "Process ID: $PAPER_TRADING_PID"
    echo "To monitor: tail -f $LOG_DIR/paper_trading_*.log"
    echo "To stop: kill $PAPER_TRADING_PID"
    
    # Keep script running
    wait $PAPER_TRADING_PID
fi

# Cleanup on exit
cleanup