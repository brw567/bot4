import axios from 'axios';
import type { AxiosInstance } from 'axios';
import { MetricData, StrategyPerformance, SystemHealth, Alert } from '../types';
import logger from './logger';

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth interceptor
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Add request interceptor for logging
    this.client.interceptors.request.use((config) => {
      config.metadata = { startTime: new Date().getTime() };
      return config;
    });

    // Add response interceptor for error handling and logging
    this.client.interceptors.response.use(
      (response) => {
        const duration = new Date().getTime() - response.config.metadata?.startTime || 0;
        logger.logApiCall(
          response.config.method?.toUpperCase() || 'GET',
          response.config.url || '',
          response.status,
          duration
        );
        return response;
      },
      (error) => {
        const duration = new Date().getTime() - error.config?.metadata?.startTime || 0;
        logger.logApiCall(
          error.config?.method?.toUpperCase() || 'GET',
          error.config?.url || '',
          error.response?.status || 0,
          duration
        );
        
        if (error.response?.status === 401) {
          logger.warn('Unauthorized access, redirecting to login');
          // Redirect to login
          localStorage.removeItem('token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async login(username: string, password: string) {
    logger.info('Attempting login', { username });
    
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    try {
      const response = await axios.post('/token', formData);
      const { access_token } = response.data;
      localStorage.setItem('token', access_token);
      logger.logAuth('login', true, { username });
      return access_token;
    } catch (error: any) {
      logger.logAuth('login', false, { 
        username, 
        error: error.response?.data?.detail || error.message 
      });
      throw error;
    }
  }

  async logout() {
    localStorage.removeItem('token');
  }

  // Bot Status
  async getBotStatus() {
    const response = await this.client.get('/bot/status');
    return response.data;
  }

  // Metrics
  async getMetrics(limit: number = 100): Promise<MetricData[]> {
    const response = await this.client.get('/monitoring/metrics', {
      params: { limit }
    });
    // API returns array directly
    return response.data || [];
  }

  async getPairMetrics(pair: string): Promise<MetricData> {
    const response = await this.client.get(`/monitoring/metrics/${pair}`);
    return response.data;
  }

  async getMetricsSummary() {
    const response = await this.client.get('/metrics/summary');
    return response.data;
  }

  // Strategies
  async getStrategyPerformance(timeframe: string = '24h'): Promise<StrategyPerformance[]> {
    const response = await this.client.get('/monitoring/strategies', {
      params: { timeframe }
    });
    return response.data;
  }

  // System Health
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.client.get('/monitoring/system');
    return response.data;
  }

  // Alerts
  async getAlerts(params?: {
    severity?: string;
    category?: string;
    resolved?: boolean;
    limit?: number;
  }): Promise<Alert[]> {
    const response = await this.client.get('/monitoring/alerts', { params });
    // API returns object with alerts array, extract it
    return response.data.alerts || [];
  }

  async resolveAlert(alertId: string) {
    const response = await this.client.post(`/monitoring/alerts/${alertId}/resolve`);
    return response.data;
  }

  // Win Rate History
  async getWinRateHistory(timeframe: string = '24h', pair?: string) {
    const response = await this.client.get('/monitoring/winrate/history', {
      params: { timeframe, pair }
    });
    return response.data;
  }

  // Active Pairs
  async getActivePairs() {
    const response = await this.client.get('/pairs');
    return response.data;
  }

  // Trades
  async getTrades(timeframe: string = '24h') {
    try {
      const response = await this.client.get('/trades', {
        params: { timeframe }
      });
      return response.data;
    } catch (error) {
      logger.error('Failed to fetch trades', error);
      return { trades: [] };
    }
  }

  // ML Model Status
  async getMLModelStatus() {
    try {
      const response = await this.client.get('/ml/model/status');
      return response.data;
    } catch (error) {
      logger.error('Failed to fetch ML model status', error);
      // Return default metrics if API fails
      return {
        accuracy: 0,
        precision: 0,
        recall: 0,
        f1Score: 0,
        auc: 0,
        lastTraining: new Date().toISOString(),
        nextTraining: new Date(Date.now() + 3600000).toISOString(),
        modelVersion: '1.0.0',
        dataPoints: 0
      };
    }
  }

  // ML Feature Importance
  async getMLFeatureImportance() {
    try {
      const response = await this.client.get('/ml/feature-importance');
      return response.data;
    } catch (error) {
      logger.error('Failed to fetch ML feature importance', error);
      return [];
    }
  }

  // System Logs
  async getSystemLogs(params?: {
    service?: string;
    level?: string;
    limit?: number;
  }) {
    try {
      const response = await this.client.get('/system/logs', { params });
      return response.data;
    } catch (error) {
      logger.error('Failed to fetch system logs', error);
      return { logs: [] };
    }
  }

  // TestNet Mode
  async getTestNetStatus() {
    try {
      const response = await this.client.get('/system/testnet-status');
      return response.data;
    } catch (error) {
      logger.error('Failed to fetch TestNet status', error);
      return { testnet_enabled: false, bot_running: false };
    }
  }

  async toggleTestNetMode(enable: boolean) {
    try {
      const response = await this.client.post('/system/testnet-toggle', null, {
        params: { enable }
      });
      return response.data;
    } catch (error) {
      logger.error('Failed to toggle TestNet mode', error);
      throw error;
    }
  }
}

export const apiService = new ApiService();
export default apiService;