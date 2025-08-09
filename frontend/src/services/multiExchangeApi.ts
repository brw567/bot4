import axios from 'axios';
import { getAuthHeader } from './auth';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export interface ExchangeStatus {
  id: string;
  name: string;
  market_type?: string;
  status: string;
  test_mode: boolean;
  read_only: boolean;
}

export interface ArbitrageOpportunity {
  symbol: string;
  buy_exchange: string;
  sell_exchange: string;
  buy_price: number;
  sell_price: number;
  spread_percentage: number;
  volume_limit: number;
  potential_profit: number;
  timestamp: string;
}

export interface CrossExchangeAnalysis {
  symbol: string;
  exchanges_analyzed: number;
  price_range: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  spread_analysis: {
    min: number;
    max: number;
    mean: number;
  };
  volume_distribution: {
    total: number;
    by_exchange: Record<string, number>;
  };
  best_liquidity_exchange: string;
  tightest_spread_exchange: string;
  arbitrage_opportunities: ArbitrageOpportunity[];
}

export interface TradingRecommendation {
  symbol: string;
  exchange: string;
  action: string;
  confidence: number;
  reasons: string[];
  timestamp: string;
}

class MultiExchangeApi {
  private baseUrl = `${API_BASE_URL}/api/multi-exchange`;

  async getStatus() {
    const response = await axios.get(`${this.baseUrl}/status`, {
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getExchangeData(symbol: string) {
    const response = await axios.get(`${this.baseUrl}/data/${symbol}`, {
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getCrossExchangeAnalysis(symbol: string): Promise<CrossExchangeAnalysis> {
    const response = await axios.get(`${this.baseUrl}/analysis/${symbol}`, {
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getArbitrageOpportunities() {
    const response = await axios.get(`${this.baseUrl}/arbitrage`, {
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getTradingRecommendation(symbol: string, exchange: string = 'binance_spot'): Promise<TradingRecommendation> {
    const response = await axios.get(`${this.baseUrl}/recommendation/${symbol}`, {
      params: { exchange },
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getExchangeMetrics(exchange: string, symbol: string) {
    const response = await axios.get(`${this.baseUrl}/metrics/${exchange}/${symbol}`, {
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getPriceComparison(symbols: string[]) {
    const response = await axios.get(`${this.baseUrl}/price-comparison`, {
      params: { symbols: symbols.join(',') },
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getExchanges(): Promise<{ exchanges: ExchangeStatus[] }> {
    const response = await axios.get(`${API_BASE_URL}/api/exchanges`, {
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getMarketOverview() {
    const response = await axios.get(`${API_BASE_URL}/api/market-overview`, {
      headers: getAuthHeader()
    });
    return response.data;
  }

  async getDashboardSummary() {
    const response = await axios.get(`${API_BASE_URL}/api/dashboard/summary`, {
      headers: getAuthHeader()
    });
    return response.data;
  }
}

export const multiExchangeApi = new MultiExchangeApi();

// Helper function for auth header
function getAuthHeader() {
  const token = localStorage.getItem('token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}