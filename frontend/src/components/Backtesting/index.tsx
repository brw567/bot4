import React, { useState, useEffect, useCallback } from 'react';
import BacktestConfiguration from './BacktestConfiguration';
import BacktestResultsVisualization from './BacktestResultsVisualization';
import StrategyComparison from './StrategyComparison';
import { BacktestConfig } from './BacktestConfiguration';
import {
  ArrowPathIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';

interface BacktestStatus {
  backtestId: string;
  status: 'running' | 'completed' | 'failed';
  progress: number;
  message: string;
  startedAt: string;
  completedAt?: string;
  error?: string;
}

interface BacktestResult {
  config: any;
  metrics: any;
  equityCurve: any[];
  dailyReturns: number[];
  monthlyReturns: any[];
  drawdownSeries: any[];
  trades: any[];
  tradeAnalysis?: any;
  riskAnalysis?: any;
  performanceAttribution?: any;
  monteCarloResults?: any;
}

const BacktestingDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'configure' | 'results' | 'compare'>('configure');
  const [runningBacktests, setRunningBacktests] = useState<BacktestStatus[]>([]);
  const [completedBacktests, setCompletedBacktests] = useState<Map<string, BacktestResult>>(new Map());
  const [selectedResult, setSelectedResult] = useState<string | null>(null);
  const [comparisonResults, setComparisonResults] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Poll for backtest status updates
  useEffect(() => {
    const pollInterval = setInterval(() => {
      runningBacktests.forEach(backtest => {
        if (backtest.status === 'running') {
          checkBacktestStatus(backtest.backtestId);
        }
      });
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [runningBacktests]);

  const runBacktest = useCallback(async (config: BacktestConfig) => {
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/backtest/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          strategy_name: config.strategyName,
          start_date: config.startDate,
          end_date: config.endDate,
          initial_capital: config.initialCapital,
          symbols: config.symbols,
          timeframe: config.timeframe,
          mode: config.mode,
          parameters: config.parameters,
          stop_loss_pct: config.stopLossPct,
          take_profit_pct: config.takeProfitPct,
          max_position_size: config.maxPositionSize,
          max_leverage: config.maxLeverage,
          max_drawdown: config.maxDrawdown,
          slippage_pct: config.slippagePct,
          commission_pct: config.commissionPct,
          walk_forward_windows: config.walkForwardWindows,
          in_sample_pct: config.inSamplePct,
          monte_carlo_runs: config.monteCarloRuns,
          confidence_level: config.confidenceLevel
        })
      });

      if (response.ok) {
        const status: BacktestStatus = await response.json();
        setRunningBacktests(prev => [...prev, status]);
        setActiveTab('results');
      } else {
        const error = await response.json();
        console.error('Failed to start backtest:', error);
        alert(`Failed to start backtest: ${error.detail}`);
      }
    } catch (error) {
      console.error('Error starting backtest:', error);
      alert('Failed to start backtest. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const checkBacktestStatus = useCallback(async (backtestId: string) => {
    try {
      const response = await fetch(`/api/backtest/status/${backtestId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const status: BacktestStatus = await response.json();
        
        // Update status in running list
        setRunningBacktests(prev => 
          prev.map(bt => bt.backtestId === backtestId ? status : bt)
        );

        // If completed, fetch results
        if (status.status === 'completed') {
          fetchBacktestResults(backtestId);
        }
      }
    } catch (error) {
      console.error('Error checking backtest status:', error);
    }
  }, []);

  const fetchBacktestResults = useCallback(async (backtestId: string) => {
    try {
      const response = await fetch(`/api/backtest/results/${backtestId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const results = await response.json();
        
        // Store results
        setCompletedBacktests(prev => {
          const newMap = new Map(prev);
          newMap.set(backtestId, results);
          return newMap;
        });

        // Auto-select if no result selected
        if (!selectedResult) {
          setSelectedResult(backtestId);
        }

        // Remove from running list
        setRunningBacktests(prev => 
          prev.filter(bt => bt.backtestId !== backtestId)
        );
      }
    } catch (error) {
      console.error('Error fetching backtest results:', error);
    }
  }, [selectedResult]);

  const exportResults = useCallback(async (backtestId?: string) => {
    const id = backtestId || selectedResult;
    if (!id) return;

    try {
      const response = await fetch(`/api/backtest/export/${id}?format=json`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        
        // Create download
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `backtest-${id}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Error exporting results:', error);
    }
  }, [selectedResult]);

  const loadComparisonData = useCallback(async () => {
    // Get all completed backtests for comparison
    const results = Array.from(completedBacktests.values()).map(result => ({
      strategyName: result.config.strategy_name,
      totalReturn: result.metrics.total_return_pct,
      annualReturn: result.metrics.annual_return,
      sharpeRatio: result.metrics.sharpe_ratio,
      sortinoRatio: result.metrics.sortino_ratio,
      calmarRatio: result.metrics.calmar_ratio,
      maxDrawdown: result.metrics.max_drawdown,
      winRate: result.metrics.win_rate,
      profitFactor: result.metrics.profit_factor,
      totalTrades: result.metrics.total_trades,
      avgTradeDuration: result.metrics.avg_trade_duration_hours,
      expectancy: result.metrics.expectancy,
      var95: result.metrics.var_95,
      stopLossTriggers: result.metrics.stop_loss_triggers
    }));

    setComparisonResults(results);
    setActiveTab('compare');
  }, [completedBacktests]);

  const currentResult = selectedResult ? completedBacktests.get(selectedResult) : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Backtesting Dashboard
            </h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Test and validate trading strategies with historical data
            </p>
          </div>
          <div className="flex items-center space-x-4">
            {/* Status Indicators */}
            <div className="flex items-center space-x-2">
              {runningBacktests.length > 0 && (
                <div className="flex items-center px-3 py-1 bg-blue-100 dark:bg-blue-900 rounded-lg">
                  <ArrowPathIcon className="h-4 w-4 text-blue-600 dark:text-blue-400 animate-spin mr-2" />
                  <span className="text-sm text-blue-600 dark:text-blue-400">
                    {runningBacktests.length} Running
                  </span>
                </div>
              )}
              {completedBacktests.size > 0 && (
                <div className="flex items-center px-3 py-1 bg-green-100 dark:bg-green-900 rounded-lg">
                  <CheckCircleIcon className="h-4 w-4 text-green-600 dark:text-green-400 mr-2" />
                  <span className="text-sm text-green-600 dark:text-green-400">
                    {completedBacktests.size} Completed
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="flex space-x-8 px-6">
            <button
              onClick={() => setActiveTab('configure')}
              className={`py-3 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'configure'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              Configure
            </button>
            <button
              onClick={() => setActiveTab('results')}
              className={`py-3 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'results'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              Results ({completedBacktests.size + runningBacktests.length})
            </button>
            <button
              onClick={() => {
                if (completedBacktests.size >= 2) {
                  loadComparisonData();
                }
              }}
              disabled={completedBacktests.size < 2}
              className={`py-3 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'compare'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : completedBacktests.size < 2
                  ? 'border-transparent text-gray-300 cursor-not-allowed'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              Compare
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {activeTab === 'configure' && (
            <BacktestConfiguration 
              onRunBacktest={runBacktest}
              isRunning={isLoading}
            />
          )}

          {activeTab === 'results' && (
            <div className="space-y-6">
              {/* Running Backtests */}
              {runningBacktests.length > 0 && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                    Running Backtests
                  </h3>
                  <div className="space-y-3">
                    {runningBacktests.map(backtest => (
                      <div 
                        key={backtest.backtestId}
                        className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-lg"
                      >
                        <div className="flex items-center">
                          <ArrowPathIcon className="h-5 w-5 text-blue-600 animate-spin mr-3" />
                          <div>
                            <p className="font-medium text-gray-900 dark:text-white">
                              Backtest {backtest.backtestId.slice(0, 8)}...
                            </p>
                            <p className="text-sm text-gray-500">{backtest.message}</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4">
                          <div className="w-32 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                            <div 
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${backtest.progress}%` }}
                            />
                          </div>
                          <span className="text-sm text-gray-500">{backtest.progress}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Completed Backtests Selector */}
              {completedBacktests.size > 0 && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                    Completed Backtests
                  </h3>
                  <div className="flex items-center space-x-4 mb-4">
                    <select
                      value={selectedResult || ''}
                      onChange={(e) => setSelectedResult(e.target.value)}
                      className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700"
                    >
                      <option value="">Select a backtest...</option>
                      {Array.from(completedBacktests.entries()).map(([id, result]) => (
                        <option key={id} value={id}>
                          {result.config.strategy_name} - {id.slice(0, 8)}... 
                          ({result.metrics.total_return_pct.toFixed(2)}%)
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              {/* Results Visualization */}
              {currentResult && (
                <BacktestResultsVisualization 
                  result={currentResult}
                  onExport={() => exportResults()}
                />
              )}

              {/* No Results Message */}
              {completedBacktests.size === 0 && runningBacktests.length === 0 && (
                <div className="text-center py-12">
                  <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500 dark:text-gray-400">
                    No backtest results yet. Configure and run a backtest to see results.
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'compare' && comparisonResults.length >= 2 && (
            <StrategyComparison 
              results={comparisonResults}
              onExportComparison={() => {
                // Export comparison data
                const blob = new Blob([JSON.stringify(comparisonResults, null, 2)], { 
                  type: 'application/json' 
                });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'strategy-comparison.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
              }}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default BacktestingDashboard;