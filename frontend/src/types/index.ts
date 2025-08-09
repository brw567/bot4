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
