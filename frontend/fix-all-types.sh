#!/bin/bash

echo "Fixing all TypeScript type issues..."

# 1. Fix test imports - add vitest imports to all test files
echo "Adding vitest imports to test files..."
find src -name "*.test.ts" -o -name "*.test.tsx" | while read file; do
  # Check if vitest import already exists
  if ! grep -q "from 'vitest'" "$file"; then
    # Add vitest import at the beginning
    sed -i "1i import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';" "$file"
  fi
done

# 2. Update StrategyPerformance type to include all fields
echo "Updating StrategyPerformance type..."
cat > src/types/strategy.ts << 'EOF'
export interface StrategyPerformance {
  name: string;
  strategy: string;  // Strategy identifier
  winRate: number;
  pnl: number;
  pnl24h: number;   // 24-hour P&L
  trades: number;
  totalTrades: number;  // Total trades count
  avgProfit: number;
  maxDrawdown: number;
  sharpeRatio: number;
  activePairs: string[];  // Active trading pairs
}
EOF

# 3. Update MetricData type to include all fields
echo "Fixing MetricData type..."
cat > src/types/index.ts << 'EOF'
// MetricData with all required fields
export interface MetricData {
  pair: string;
  price: number;
  volume: number;
  change: number;
  winRate: number;
  totalTrades: number;
  profitableTrades: number;
  regime: string;
  fundingRate: number;
  timestamp: string;
  
  // Additional fields for analytics
  volatility?: number;
  rsi?: number;
  adx?: number;
  orderImbalance?: number;
  openInterest?: number;
  activeStrategy?: string;
  mlConfidence?: number;
}

export interface ChangeEvent {
  pair: string;
  type: 'price' | 'volume' | 'regime';
  metric?: string;  // Which metric changed
  oldValue: number | string;
  newValue: number | string;
  timestamp: string;
  severity?: 'low' | 'medium' | 'high';
  changePercent?: number;
}

export interface StrategyPerformance {
  name: string;
  strategy: string;
  winRate: number;
  pnl: number;
  pnl24h: number;
  trades: number;
  totalTrades: number;
  avgProfit: number;
  maxDrawdown: number;
  sharpeRatio: number;
  activePairs: string[];
}

export interface SystemHealth {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  apiLatency: {
    binance: number;
    redis: number;
    [key: string]: number;
  };
  redisConnected: boolean;
}

export interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  actionable: boolean;
  category?: string;
  severity?: 'low' | 'medium' | 'high';
  pair?: string;
}

export interface WebSocketMessage {
  type: string;
  data: any;
}

// Redux state types
export interface AuthState {
  isAuthenticated: boolean;
  username: string | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

export interface MetricsState {
  metrics: MetricData[];
  changes: ChangeEvent[];
  strategies: StrategyPerformance[];
  winRate: number;
  totalPnl: number;
  activePairs: number;
  loading: boolean;
  error: string | null;
}

export interface SystemState {
  connected: boolean;
  botStatus: string;
  health: SystemHealth | null;
  lastUpdate: string | null;
}

export interface AlertsState {
  alerts: Alert[];
  unreadCount: number;
}

// Additional types
export interface WinRateData {
  timestamp: string;
  winRate: number;
  target: number;  // Target win rate
}

export interface ChartData {
  time: string;
  price: number;
  volume: number;
  buyVolume?: number;
  sellVolume?: number;
  signal?: number;
}

export interface OrderBookEntry {
  price: number;
  amount: number;
  size?: number;
  total: number;
}

export interface OrderBookData {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  spread: number;
  spreadPercentage: number;
}

export interface Trade {
  id: string;
  time: string;
  pair: string;
  side: 'buy' | 'sell';
  price: number;
  amount: number;
  total: number;
  fee?: number;
  profit?: number;
  profitPercent?: number;
  status: 'pending' | 'completed' | 'failed';
}
EOF

# 4. Fix authSlice exports
echo "Fixing authSlice actions..."
cat >> src/store/slices/authSlice.ts << 'EOF'

// Map old action names to new ones
export const setAuth = authSlice.actions.loginSuccess;
export const clearAuth = authSlice.actions.logout;
EOF

# 5. Fix component test imports
echo "Fixing component test imports..."

# Move misplaced test files to correct locations
if [ -d "src/hooks/__tests__" ]; then
  # These should be in components directories
  mv src/hooks/__tests__/AlertList.test.tsx src/components/AlertCenter/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/InstallPrompt.test.tsx src/components/PWA/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/MetricsCard.test.tsx src/components/MetricsGroup/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/ModelPerformance.test.tsx src/components/AIIntegration/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/OrderBook.test.tsx src/components/TradingInterface/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/SignalStrength.test.tsx src/components/AIIntegration/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/TradeHistory.test.tsx src/components/TradingInterface/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/TradingChart.test.tsx src/components/TradingInterface/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/WinRateChart.test.tsx src/components/Analytics/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/authSlice.test.ts src/store/slices/__tests__/ 2>/dev/null || true
  mv src/hooks/__tests__/setupTests.ts src/ 2>/dev/null || true
fi

# 6. Fix WinRateChart test data
echo "Fixing WinRateChart test data..."
find src -name "WinRateChart.test.tsx" -exec sed -i 's/winRate: \([0-9.]*\) }/winRate: \1, target: 80 }/g' {} \;

# 7. Fix OrderBook test props
echo "Fixing OrderBook test props..."
find src -name "OrderBook.test.tsx" -exec sed -i 's/<OrderBook orderBook=/<OrderBook bids={orderBook.bids} asks={orderBook.asks} spread={orderBook.spread}/g' {} \;

# 8. Fix Navigator.standalone
echo "Fixing Navigator type..."
cat > src/types/global.d.ts << 'EOF'
interface Navigator {
  standalone?: boolean;
}
EOF

# 9. Update tsconfig to include test types
echo "Updating tsconfig for test types..."
if [ -f "tsconfig.app.json" ]; then
  # Add vitest types
  sed -i '/"lib":/a\    "types": ["vitest/globals", "@testing-library/jest-dom"],' tsconfig.app.json
fi

echo "All type fixes applied!"