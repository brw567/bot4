import React, { useState, useEffect } from 'react';
import { useAppSelector, useAppDispatch } from '../../hooks/redux';
import { setStrategies } from '../../store/slices/metricsSlice';
import WinRateChart from './WinRateChart';
import PnLVisualization from './PnLVisualization';
import StrategyPerformance from './StrategyPerformance';
import MLModelStatus from './MLModelStatus';
import MarketplaceStatus from '../MarketplaceStatus';
import apiService from '../../services/api';
import toast from 'react-hot-toast';

type TimeFrame = '1h' | '24h' | '7d' | '30d';
type TabView = 'overview' | 'strategies' | 'ml-model' | 'marketplaces';

const Analytics: React.FC = () => {
  const dispatch = useAppDispatch();
  const { strategies } = useAppSelector(state => state.metrics);
  const [timeframe, setTimeframe] = useState<TimeFrame>('24h');
  const [activeTab, setActiveTab] = useState<TabView>('overview');
  const [loading, setLoading] = useState(false);
  
  // Mock data - would come from API
  const [winRateData, setWinRateData] = useState<any[]>([]);
  const [pnlData, setPnlData] = useState<any[]>([]);
  const [mlMetrics, setMlMetrics] = useState<any>(null);

  useEffect(() => {
    loadAnalyticsData();
  }, [timeframe]);

  const loadAnalyticsData = async () => {
    setLoading(true);
    try {
      // Load win rate history from API
      const winRateResponse = await apiService.getWinRateHistory(timeframe);
      if (winRateResponse && winRateResponse.length > 0) {
        setWinRateData(winRateResponse);
      } else {
        // No data available yet
        setWinRateData([]);
      }
      
      // Load P&L data from trades
      const tradesResponse = await apiService.getTrades(timeframe);
      if (tradesResponse && tradesResponse.trades) {
        // Calculate P&L from real trades
        const pnlData = calculatePnLFromTrades(tradesResponse.trades, timeframe);
        setPnlData(pnlData);
      } else {
        setPnlData([]);
      }
      
      // Load strategy performance
      const strategiesResponse = await apiService.getStrategyPerformance(timeframe);
      dispatch(setStrategies(strategiesResponse));
      
      // Load ML metrics from real ML model
      const mlResponse = await apiService.getMLModelStatus();
      setMlMetrics(mlResponse);
      
    } catch (error) {
      toast.error('Failed to load analytics data');
      console.error('Analytics data error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate P&L from real trades
  const calculatePnLFromTrades = (trades: any[], tf: TimeFrame) => {
    if (!trades || trades.length === 0) return [];
    
    // Group trades by time period
    const groupedTrades = new Map<string, { pnl: number, trades: number }>();
    const now = new Date();
    
    trades.forEach(trade => {
      const tradeDate = new Date(trade.timestamp);
      let key: string;
      
      if (tf === '1h') {
        key = tradeDate.toISOString().slice(0, 13); // Group by hour
      } else if (tf === '24h') {
        key = tradeDate.toISOString().slice(0, 13); // Group by hour
      } else {
        key = tradeDate.toISOString().slice(0, 10); // Group by day
      }
      
      const existing = groupedTrades.get(key) || { pnl: 0, trades: 0 };
      existing.pnl += trade.profit || 0;
      existing.trades += 1;
      groupedTrades.set(key, existing);
    });
    
    // Convert to array and calculate cumulative
    const data = [];
    let cumulative = 0;
    const sortedKeys = Array.from(groupedTrades.keys()).sort();
    
    sortedKeys.forEach(key => {
      const group = groupedTrades.get(key)!;
      cumulative += group.pnl;
      data.push({
        timestamp: key,
        pnl: group.pnl,
        cumulative: cumulative,
        trades: group.trades,
      });
    });
    
    return data;
  };

  // Feature importance will come from ML API
  const [featureImportance, setFeatureImportance] = useState<any[]>([]);
  const [modelPerformanceHistory, setModelPerformanceHistory] = useState<any[]>([]);
  
  useEffect(() => {
    // Load feature importance from API
    apiService.getMLFeatureImportance().then(data => {
      if (data && data.length > 0) {
        setFeatureImportance(data);
      }
    }).catch(err => {
      console.error('Failed to load feature importance:', err);
    });
    
    // Load model performance history
    if (winRateData && winRateData.length > 0) {
      setModelPerformanceHistory(winRateData.map(d => ({
        timestamp: d.timestamp,
        accuracy: d.winRate || 0,
        loss: d.winRate ? 100 - d.winRate : 0,
      })));
    }
  }, [winRateData]);

  const timeframeOptions: { value: TimeFrame; label: string }[] = [
    { value: '1h', label: '1 Hour' },
    { value: '24h', label: '24 Hours' },
    { value: '7d', label: '7 Days' },
    { value: '30d', label: '30 Days' },
  ];

  const tabs = [
    { id: 'overview' as TabView, label: 'Overview' },
    { id: 'strategies' as TabView, label: 'Strategies' },
    { id: 'ml-model' as TabView, label: 'ML Model' },
    { id: 'marketplaces' as TabView, label: 'Marketplaces' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Analytics Dashboard
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Performance insights and ML model monitoring
          </p>
        </div>
        
        {/* Timeframe Selector */}
        <div className="flex items-center space-x-2">
          {timeframeOptions.map(option => (
            <button
              key={option.value}
              onClick={() => setTimeframe(option.value)}
              className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
                timeframe === option.value
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center h-96">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      ) : (
        <>
          {activeTab === 'overview' && (
            <div className="space-y-6">
              <WinRateChart 
                data={winRateData} 
                timeframe={timeframe}
                showTarget={true}
              />
              <PnLVisualization 
                data={pnlData} 
                timeframe={timeframe}
              />
            </div>
          )}

          {activeTab === 'strategies' && (
            <StrategyPerformance data={strategies} />
          )}

          {activeTab === 'ml-model' && mlMetrics && (
            <MLModelStatus
              metrics={mlMetrics}
              featureImportance={featureImportance}
              performanceHistory={modelPerformanceHistory}
            />
          )}

          {activeTab === 'marketplaces' && (
            <MarketplaceStatus compact={false} showFunds={true} />
          )}
        </>
      )}
    </div>
  );
};

export default Analytics;