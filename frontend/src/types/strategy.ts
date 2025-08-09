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
