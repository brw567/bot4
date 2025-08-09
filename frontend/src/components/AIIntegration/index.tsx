import React, { useState, useEffect } from 'react';
import { useAppSelector } from '../../hooks/redux';
import ModelPerformance from './ModelPerformance';
import PredictionInsights from './PredictionInsights';
import FeatureImportance from './FeatureImportance';
import BacktestResults from './BacktestResults';
import { SparklesIcon, ChartBarIcon, BeakerIcon, CpuChipIcon } from '@heroicons/react/24/outline';

interface AIIntegrationProps {
  selectedModel?: string;
}

const AIIntegration: React.FC<AIIntegrationProps> = ({ selectedModel = 'arbitrage_predictor' }) => {
  const [activeTab, setActiveTab] = useState('performance');
  const [modelData, setModelData] = useState<any>({
    performance: {},
    predictions: [],
    features: {},
    backtest: {},
  });

  useEffect(() => {
    // Fetch real ML data from API
    const fetchModelData = async () => {
      try {
        const token = localStorage.getItem('token');
        const headers = {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        };

        // Fetch performance metrics
        const perfResponse = await fetch('/api/ml/model/performance', { headers });
        const performance = perfResponse.ok ? await perfResponse.json() : {
          accuracy: 0,
          precision: 0,
          recall: 0,
          f1Score: 0,
          rocAuc: 0,
          lastTrained: new Date().toISOString(),
          totalPredictions: 0,
          profitableTrades: 0,
        };

        // Fetch recent predictions
        const predResponse = await fetch('/api/ml/predictions/recent', { headers });
        const predictions = predResponse.ok ? await predResponse.json() : [];

        // Fetch feature importance
        const featResponse = await fetch('/api/ml/model/features', { headers });
        const featuresArray = featResponse.ok ? await featResponse.json() : [];
        
        // Convert features array to object format expected by component
        const features: any = {};
        featuresArray.forEach((feat: any) => {
          features[feat.feature] = {
            importance: feat.importance,
            trend: feat.trend,
            correlation: feat.correlation,
          };
        });

        // Backtest results - fetch from trades API
        const tradesResponse = await fetch('/api/trades?limit=1000', { headers });
        const tradesData = tradesResponse.ok ? await tradesResponse.json() : { trades: [] };
        const trades = tradesData.trades || [];
        
        // Calculate backtest data from real trades
        const backtestData = [];
        const dailyProfits = new Map<string, { profit: number, count: number, successful: number }>();
        
        // Group trades by day
        trades.forEach((trade: any) => {
          const date = new Date(trade.timestamp).toISOString().split('T')[0];
          if (!dailyProfits.has(date)) {
            dailyProfits.set(date, { profit: 0, count: 0, successful: 0 });
          }
          const day = dailyProfits.get(date)!;
          day.profit += trade.profit || 0;
          day.count += 1;
          if (trade.profit > 0) day.successful += 1;
        });
        
        // Convert to array and sort by date
        let cumulativeProfit = 0;
        const sortedDates = Array.from(dailyProfits.keys()).sort();
        
        sortedDates.forEach(date => {
          const day = dailyProfits.get(date)!;
          cumulativeProfit += day.profit;
          backtestData.push({
            date: new Date(date).toISOString(),
            profit: day.profit,
            cumulativeProfit,
            predictions: day.count,
            successRate: day.count > 0 ? day.successful / day.count : 0,
            sharpeRatio: 0, // Would need to calculate from returns
          });
        });

        setModelData({
          performance,
          predictions,
          features,
          backtest: {
            data: backtestData,
            summary: {
              totalProfit: cumulativeProfit,
              avgDailyProfit: backtestData.length > 0 ? cumulativeProfit / backtestData.length : 0,
              winRate: trades.length > 0 ? trades.filter((t: any) => t.profit > 0).length / trades.length : 0,
              sharpeRatio: 0, // Would need returns data to calculate
              maxDrawdown: 0, // Would need to calculate from equity curve
              profitFactor: 0, // Would need to calculate from wins/losses
            },
          },
        });
      } catch (error) {
        console.error('Failed to fetch ML data:', error);
      }
    };

    fetchModelData();
    const interval = setInterval(fetchModelData, 30000);
    return () => clearInterval(interval);
  }, [selectedModel]);

  const tabs = [
    { id: 'performance', name: 'Model Performance', icon: ChartBarIcon },
    { id: 'predictions', name: 'Live Predictions', icon: SparklesIcon },
    { id: 'features', name: 'Feature Analysis', icon: BeakerIcon },
    { id: 'backtest', name: 'Backtest Results', icon: CpuChipIcon },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">AI Integration</h2>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Machine Learning insights and predictions for {selectedModel}
        </p>
      </div>

      {/* Model Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Accuracy</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {(modelData.performance.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-green-600">
              <ChartBarIcon className="h-8 w-8" />
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">ROC AUC</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {modelData.performance.rocAuc?.toFixed(3)}
              </p>
            </div>
            <div className="text-blue-600">
              <SparklesIcon className="h-8 w-8" />
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Win Rate</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {((modelData.performance.profitableTrades / modelData.performance.totalPredictions) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="text-purple-600">
              <BeakerIcon className="h-8 w-8" />
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">F1 Score</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {modelData.performance.f1Score?.toFixed(3)}
              </p>
            </div>
            <div className="text-orange-600">
              <CpuChipIcon className="h-8 w-8" />
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center py-2 px-1 border-b-2 font-medium text-sm transition-colors
                ${activeTab === tab.id
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
                }
              `}
            >
              <tab.icon className="h-5 w-5 mr-2" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="mt-6">
        {activeTab === 'performance' && (
          <ModelPerformance data={modelData.performance} />
        )}
        {activeTab === 'predictions' && (
          <PredictionInsights predictions={modelData.predictions} />
        )}
        {activeTab === 'features' && (
          <FeatureImportance features={modelData.features} />
        )}
        {activeTab === 'backtest' && (
          <BacktestResults data={modelData.backtest} />
        )}
      </div>
    </div>
  );
};

export default AIIntegration;