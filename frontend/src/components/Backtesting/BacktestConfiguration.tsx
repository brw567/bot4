import React, { useState, useEffect } from 'react';
import { format } from 'date-fns';
import {
  CalendarIcon,
  CogIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  PlayIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';

interface Strategy {
  name: string;
  description: string;
  parameters: Record<string, any>;
  supportedSymbols: string[];
}

interface BacktestConfigurationProps {
  onRunBacktest: (config: BacktestConfig) => void;
  isRunning: boolean;
}

export interface BacktestConfig {
  strategyName: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  symbols: string[];
  timeframe: string;
  mode: 'fast' | 'standard' | 'detailed' | 'walk_forward' | 'monte_carlo';
  
  // Strategy parameters
  parameters: Record<string, any>;
  
  // Risk parameters (Quinn's requirements)
  stopLossPct: number;
  takeProfitPct?: number;
  maxPositionSize: number;
  maxLeverage: number;
  maxDrawdown: number;
  
  // Execution parameters
  slippagePct: number;
  commissionPct: number;
  
  // Advanced parameters
  walkForwardWindows?: number;
  inSamplePct?: number;
  monteCarloRuns?: number;
  confidenceLevel?: number;
}

const BacktestConfiguration: React.FC<BacktestConfigurationProps> = ({ 
  onRunBacktest, 
  isRunning 
}) => {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('');
  const [config, setConfig] = useState<BacktestConfig>({
    strategyName: '',
    startDate: format(new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), 'yyyy-MM-dd'),
    endDate: format(new Date(), 'yyyy-MM-dd'),
    initialCapital: 10000,
    symbols: ['BTC/USDT', 'ETH/USDT'],
    timeframe: '1h',
    mode: 'standard',
    parameters: {},
    stopLossPct: 2.0,
    maxPositionSize: 2.0,
    maxLeverage: 3.0,
    maxDrawdown: 15.0,
    slippagePct: 0.1,
    commissionPct: 0.1
  });
  
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Available options
  const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];
  const availableSymbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
    'ADA/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT'
  ];
  
  const backtestModes = [
    { value: 'fast', label: 'Fast', description: 'Quick validation with basic metrics' },
    { value: 'standard', label: 'Standard', description: 'Normal backtesting with full metrics' },
    { value: 'detailed', label: 'Detailed', description: 'Comprehensive analysis with trade log' },
    { value: 'walk_forward', label: 'Walk Forward', description: 'Optimization with out-of-sample testing' },
    { value: 'monte_carlo', label: 'Monte Carlo', description: 'Statistical simulation for robustness' }
  ];

  useEffect(() => {
    // Fetch available strategies
    const fetchStrategies = async () => {
      try {
        const response = await fetch('/api/strategies/available', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          setStrategies(data.strategies || [
            {
              name: 'statistical_arbitrage',
              description: 'Statistical arbitrage using cointegration',
              parameters: {
                zscore_threshold: 2.0,
                lookback_period: 20,
                min_correlation: 0.7
              },
              supportedSymbols: ['BTC/USDT', 'ETH/USDT']
            },
            {
              name: 'grid_trading',
              description: 'Grid trading with dynamic levels',
              parameters: {
                grid_levels: 10,
                grid_spacing_pct: 1.0,
                rebalance_threshold: 0.1
              },
              supportedSymbols: availableSymbols
            },
            {
              name: 'momentum_breakout',
              description: 'Momentum-based breakout strategy',
              parameters: {
                momentum_period: 14,
                breakout_threshold: 1.5,
                volume_filter: true
              },
              supportedSymbols: availableSymbols
            }
          ]);
        }
      } catch (error) {
        console.error('Failed to fetch strategies:', error);
      }
    };
    
    fetchStrategies();
  }, []);

  const validateConfig = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!config.strategyName) {
      newErrors.strategy = 'Please select a strategy';
    }
    
    if (new Date(config.startDate) >= new Date(config.endDate)) {
      newErrors.dates = 'End date must be after start date';
    }
    
    if (config.initialCapital <= 0) {
      newErrors.capital = 'Initial capital must be positive';
    }
    
    if (config.symbols.length === 0) {
      newErrors.symbols = 'Select at least one symbol';
    }
    
    // Risk validation (Quinn's requirements)
    if (config.stopLossPct <= 0 || config.stopLossPct > 10) {
      newErrors.stopLoss = 'Stop loss must be between 0.1% and 10%';
    }
    
    if (config.maxPositionSize <= 0 || config.maxPositionSize > 10) {
      newErrors.positionSize = 'Position size must be between 0.1% and 10%';
    }
    
    if (config.maxLeverage < 1 || config.maxLeverage > 10) {
      newErrors.leverage = 'Leverage must be between 1x and 10x';
    }
    
    if (config.maxDrawdown <= 0 || config.maxDrawdown > 50) {
      newErrors.drawdown = 'Max drawdown must be between 1% and 50%';
    }
    
    // Mode-specific validation
    if (config.mode === 'walk_forward') {
      if (!config.walkForwardWindows || config.walkForwardWindows < 2) {
        newErrors.walkForward = 'Walk forward requires at least 2 windows';
      }
    }
    
    if (config.mode === 'monte_carlo') {
      if (!config.monteCarloRuns || config.monteCarloRuns < 100) {
        newErrors.monteCarlo = 'Monte Carlo requires at least 100 runs';
      }
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleStrategyChange = (strategyName: string) => {
    setSelectedStrategy(strategyName);
    const strategy = strategies.find(s => s.name === strategyName);
    
    if (strategy) {
      setConfig({
        ...config,
        strategyName,
        parameters: strategy.parameters,
        symbols: strategy.supportedSymbols.slice(0, 2)
      });
    }
  };

  const handleSymbolToggle = (symbol: string) => {
    const newSymbols = config.symbols.includes(symbol)
      ? config.symbols.filter(s => s !== symbol)
      : [...config.symbols, symbol];
    
    setConfig({ ...config, symbols: newSymbols });
  };

  const handleParameterChange = (key: string, value: any) => {
    setConfig({
      ...config,
      parameters: {
        ...config.parameters,
        [key]: value
      }
    });
  };

  const handleRunBacktest = () => {
    if (validateConfig()) {
      onRunBacktest(config);
    }
  };

  const selectedStrategyData = strategies.find(s => s.name === selectedStrategy);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
          Backtest Configuration
        </h2>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Configure and run comprehensive strategy backtests
        </p>
      </div>

      {/* Strategy Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Strategy
        </label>
        <select
          value={selectedStrategy}
          onChange={(e) => handleStrategyChange(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500"
        >
          <option value="">Select a strategy...</option>
          {strategies.map(strategy => (
            <option key={strategy.name} value={strategy.name}>
              {strategy.name} - {strategy.description}
            </option>
          ))}
        </select>
        {errors.strategy && (
          <p className="mt-1 text-sm text-red-600">{errors.strategy}</p>
        )}
      </div>

      {/* Date Range */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <CalendarIcon className="inline h-4 w-4 mr-1" />
            Start Date
          </label>
          <input
            type="date"
            value={config.startDate}
            onChange={(e) => setConfig({ ...config, startDate: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            <CalendarIcon className="inline h-4 w-4 mr-1" />
            End Date
          </label>
          <input
            type="date"
            value={config.endDate}
            onChange={(e) => setConfig({ ...config, endDate: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
        </div>
        {errors.dates && (
          <p className="col-span-2 text-sm text-red-600">{errors.dates}</p>
        )}
      </div>

      {/* Capital and Timeframe */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Initial Capital ($)
          </label>
          <input
            type="number"
            value={config.initialCapital}
            onChange={(e) => setConfig({ ...config, initialCapital: parseFloat(e.target.value) })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          />
          {errors.capital && (
            <p className="mt-1 text-sm text-red-600">{errors.capital}</p>
          )}
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Timeframe
          </label>
          <select
            value={config.timeframe}
            onChange={(e) => setConfig({ ...config, timeframe: e.target.value })}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
          >
            {timeframes.map(tf => (
              <option key={tf} value={tf}>{tf}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Symbol Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Symbols
        </label>
        <div className="grid grid-cols-4 gap-2">
          {availableSymbols.map(symbol => (
            <label key={symbol} className="flex items-center">
              <input
                type="checkbox"
                checked={config.symbols.includes(symbol)}
                onChange={() => handleSymbolToggle(symbol)}
                className="mr-2 rounded text-primary-600"
                disabled={selectedStrategyData && !selectedStrategyData.supportedSymbols.includes(symbol)}
              />
              <span className="text-sm text-gray-700 dark:text-gray-300">{symbol}</span>
            </label>
          ))}
        </div>
        {errors.symbols && (
          <p className="mt-1 text-sm text-red-600">{errors.symbols}</p>
        )}
      </div>

      {/* Backtest Mode */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Backtest Mode
        </label>
        <div className="space-y-2">
          {backtestModes.map(mode => (
            <label key={mode.value} className="flex items-start">
              <input
                type="radio"
                value={mode.value}
                checked={config.mode === mode.value}
                onChange={(e) => setConfig({ ...config, mode: e.target.value as any })}
                className="mt-1 mr-3"
              />
              <div>
                <div className="text-sm font-medium text-gray-900 dark:text-white">
                  {mode.label}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {mode.description}
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Risk Parameters (Quinn's requirements) */}
      <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
        <h3 className="text-sm font-medium text-red-800 dark:text-red-200 mb-3 flex items-center">
          <ExclamationTriangleIcon className="h-4 w-4 mr-1" />
          Risk Management (Required)
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
              Stop Loss (%)
            </label>
            <input
              type="number"
              value={config.stopLossPct}
              onChange={(e) => setConfig({ ...config, stopLossPct: parseFloat(e.target.value) })}
              step="0.1"
              min="0.1"
              max="10"
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
            />
            {errors.stopLoss && (
              <p className="mt-1 text-xs text-red-600">{errors.stopLoss}</p>
            )}
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
              Take Profit (%) - Optional
            </label>
            <input
              type="number"
              value={config.takeProfitPct || ''}
              onChange={(e) => setConfig({ ...config, takeProfitPct: e.target.value ? parseFloat(e.target.value) : undefined })}
              step="0.1"
              min="0.1"
              max="50"
              placeholder="None"
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
              Max Position Size (%)
            </label>
            <input
              type="number"
              value={config.maxPositionSize}
              onChange={(e) => setConfig({ ...config, maxPositionSize: parseFloat(e.target.value) })}
              step="0.1"
              min="0.1"
              max="10"
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
            />
            {errors.positionSize && (
              <p className="mt-1 text-xs text-red-600">{errors.positionSize}</p>
            )}
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
              Max Leverage
            </label>
            <input
              type="number"
              value={config.maxLeverage}
              onChange={(e) => setConfig({ ...config, maxLeverage: parseFloat(e.target.value) })}
              step="0.5"
              min="1"
              max="10"
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
            />
            {errors.leverage && (
              <p className="mt-1 text-xs text-red-600">{errors.leverage}</p>
            )}
          </div>
          <div className="col-span-2">
            <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
              Max Drawdown (%)
            </label>
            <input
              type="number"
              value={config.maxDrawdown}
              onChange={(e) => setConfig({ ...config, maxDrawdown: parseFloat(e.target.value) })}
              step="1"
              min="1"
              max="50"
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
            />
            {errors.drawdown && (
              <p className="mt-1 text-xs text-red-600">{errors.drawdown}</p>
            )}
          </div>
        </div>
      </div>

      {/* Strategy Parameters */}
      {selectedStrategyData && Object.keys(selectedStrategyData.parameters).length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
            <CogIcon className="h-4 w-4 mr-1" />
            Strategy Parameters
          </h3>
          <div className="grid grid-cols-2 gap-4">
            {Object.entries(selectedStrategyData.parameters).map(([key, value]) => (
              <div key={key}>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </label>
                <input
                  type={typeof value === 'boolean' ? 'checkbox' : 'number'}
                  checked={typeof value === 'boolean' ? config.parameters[key] : undefined}
                  value={typeof value !== 'boolean' ? config.parameters[key] : undefined}
                  onChange={(e) => handleParameterChange(
                    key,
                    typeof value === 'boolean' ? e.target.checked : parseFloat(e.target.value)
                  )}
                  step={typeof value === 'number' && value < 1 ? '0.01' : '1'}
                  className={
                    typeof value === 'boolean'
                      ? 'mr-2 rounded text-primary-600'
                      : 'w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700'
                  }
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Advanced Settings */}
      <div className="mb-6">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-primary-600 hover:text-primary-700 flex items-center"
        >
          <CogIcon className="h-4 w-4 mr-1" />
          {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
        </button>
        
        {showAdvanced && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Slippage (%)
                </label>
                <input
                  type="number"
                  value={config.slippagePct}
                  onChange={(e) => setConfig({ ...config, slippagePct: parseFloat(e.target.value) })}
                  step="0.01"
                  className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Commission (%)
                </label>
                <input
                  type="number"
                  value={config.commissionPct}
                  onChange={(e) => setConfig({ ...config, commissionPct: parseFloat(e.target.value) })}
                  step="0.01"
                  className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                />
              </div>
              
              {config.mode === 'walk_forward' && (
                <>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Walk Forward Windows
                    </label>
                    <input
                      type="number"
                      value={config.walkForwardWindows || 10}
                      onChange={(e) => setConfig({ ...config, walkForwardWindows: parseInt(e.target.value) })}
                      min="2"
                      max="50"
                      className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                    />
                    {errors.walkForward && (
                      <p className="mt-1 text-xs text-red-600">{errors.walkForward}</p>
                    )}
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      In-Sample (%)
                    </label>
                    <input
                      type="number"
                      value={(config.inSamplePct || 0.7) * 100}
                      onChange={(e) => setConfig({ ...config, inSamplePct: parseFloat(e.target.value) / 100 })}
                      min="50"
                      max="90"
                      className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                    />
                  </div>
                </>
              )}
              
              {config.mode === 'monte_carlo' && (
                <>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Monte Carlo Runs
                    </label>
                    <input
                      type="number"
                      value={config.monteCarloRuns || 1000}
                      onChange={(e) => setConfig({ ...config, monteCarloRuns: parseInt(e.target.value) })}
                      min="100"
                      max="10000"
                      step="100"
                      className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                    />
                    {errors.monteCarlo && (
                      <p className="mt-1 text-xs text-red-600">{errors.monteCarlo}</p>
                    )}
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Confidence Level (%)
                    </label>
                    <input
                      type="number"
                      value={(config.confidenceLevel || 0.95) * 100}
                      onChange={(e) => setConfig({ ...config, confidenceLevel: parseFloat(e.target.value) / 100 })}
                      min="80"
                      max="99"
                      className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
                    />
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Run Button */}
      <div className="flex justify-end space-x-4">
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg flex items-center"
        >
          <ArrowPathIcon className="h-5 w-5 mr-2" />
          Reset
        </button>
        <button
          onClick={handleRunBacktest}
          disabled={isRunning || !selectedStrategy}
          className={`px-6 py-2 rounded-lg flex items-center font-medium ${
            isRunning || !selectedStrategy
              ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 cursor-not-allowed'
              : 'bg-primary-600 hover:bg-primary-700 text-white'
          }`}
        >
          {isRunning ? (
            <>
              <ArrowPathIcon className="h-5 w-5 mr-2 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <PlayIcon className="h-5 w-5 mr-2" />
              Run Backtest
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default BacktestConfiguration;