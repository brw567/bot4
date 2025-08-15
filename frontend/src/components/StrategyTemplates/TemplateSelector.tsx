import React, { useState, useMemo } from 'react';
import {
  ChartBarIcon,
  CogIcon,
  SparklesIcon,
  ScaleIcon,
  ArrowTrendingUpIcon,
  BeakerIcon,
  ShieldCheckIcon,
  ClockIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';

interface StrategyTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  riskLevel: 'conservative' | 'moderate' | 'aggressive';
  timeframe: string;
  minCapital: number;
  expectedSharpe: number;
  expectedWinRate: number;
  expectedProfitFactor: number;
  recommendedPairs: string[];
  requiredIndicators: string[];
  requiresML: boolean;
  requiresOrderbook: boolean;
  suitableForBeginners: boolean;
  parameters: {
    stopLossPct: number;
    takeProfitPct?: number;
    maxPositionSize: number;
    maxLeverage: number;
    maxDrawdown: number;
    technicalParams: Record<string, any>;
  };
}

interface TemplateSelectorProps {
  onSelectTemplate: (template: StrategyTemplate) => void;
  userCapital?: number;
  userExperience?: 'beginner' | 'intermediate' | 'advanced';
}

const TemplateSelector: React.FC<TemplateSelectorProps> = ({
  onSelectTemplate,
  userCapital = 10000,
  userExperience = 'intermediate'
}) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedRiskLevel, setSelectedRiskLevel] = useState<string>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [showDetails, setShowDetails] = useState<string | null>(null);

  // Available strategy templates
  const templates: StrategyTemplate[] = [
    {
      id: 'grid_trading',
      name: 'Grid Trading',
      description: 'Places buy and sell orders at regular price intervals',
      category: 'market_making',
      riskLevel: 'moderate',
      timeframe: '15m',
      minCapital: 5000,
      expectedSharpe: 1.2,
      expectedWinRate: 65,
      expectedProfitFactor: 1.4,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
      requiredIndicators: ['price_levels', 'volatility'],
      requiresML: false,
      requiresOrderbook: false,
      suitableForBeginners: true,
      parameters: {
        stopLossPct: 2.0,
        maxPositionSize: 2.0,
        maxLeverage: 2.0,
        maxDrawdown: 15.0,
        technicalParams: {
          gridLevels: 10,
          gridSpacingPct: 1.0,
          rebalanceThreshold: 0.1
        }
      }
    },
    {
      id: 'mean_reversion',
      name: 'Mean Reversion',
      description: 'Trades on the assumption that prices revert to their mean',
      category: 'mean_reversion',
      riskLevel: 'conservative',
      timeframe: '1h',
      minCapital: 3000,
      expectedSharpe: 1.5,
      expectedWinRate: 70,
      expectedProfitFactor: 1.6,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
      requiredIndicators: ['bollinger_bands', 'rsi'],
      requiresML: false,
      requiresOrderbook: false,
      suitableForBeginners: true,
      parameters: {
        stopLossPct: 2.0,
        takeProfitPct: 3.0,
        maxPositionSize: 2.0,
        maxLeverage: 1.0,
        maxDrawdown: 10.0,
        technicalParams: {
          bbPeriod: 20,
          bbStd: 2.0,
          rsiPeriod: 14,
          rsiOversold: 30,
          rsiOverbought: 70
        }
      }
    },
    {
      id: 'momentum_breakout',
      name: 'Momentum Breakout',
      description: 'Trades breakouts with strong momentum confirmation',
      category: 'trend_following',
      riskLevel: 'moderate',
      timeframe: '1h',
      minCapital: 5000,
      expectedSharpe: 1.3,
      expectedWinRate: 45,
      expectedProfitFactor: 2.2,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT'],
      requiredIndicators: ['momentum', 'volume', 'breakout_levels'],
      requiresML: false,
      requiresOrderbook: false,
      suitableForBeginners: false,
      parameters: {
        stopLossPct: 2.0,
        takeProfitPct: 6.0,
        maxPositionSize: 2.0,
        maxLeverage: 2.0,
        maxDrawdown: 12.0,
        technicalParams: {
          momentumPeriod: 14,
          breakoutPeriod: 20,
          volumeMultiplier: 1.5,
          atrMultiplier: 2.0
        }
      }
    },
    {
      id: 'statistical_arbitrage',
      name: 'Statistical Arbitrage',
      description: 'Exploits price deviations between correlated assets',
      category: 'arbitrage',
      riskLevel: 'conservative',
      timeframe: '15m',
      minCapital: 10000,
      expectedSharpe: 1.8,
      expectedWinRate: 75,
      expectedProfitFactor: 1.7,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT'],
      requiredIndicators: ['zscore', 'correlation', 'spread'],
      requiresML: false,
      requiresOrderbook: true,
      suitableForBeginners: false,
      parameters: {
        stopLossPct: 1.0,
        takeProfitPct: 2.0,
        maxPositionSize: 2.0,
        maxLeverage: 2.0,
        maxDrawdown: 8.0,
        technicalParams: {
          lookbackPeriod: 60,
          zscoreThreshold: 2.0,
          minCorrelation: 0.7,
          halfLife: 20
        }
      }
    },
    {
      id: 'ml_ensemble',
      name: 'ML Ensemble',
      description: 'Uses multiple ML models for prediction',
      category: 'machine_learning',
      riskLevel: 'moderate',
      timeframe: '1h',
      minCapital: 10000,
      expectedSharpe: 1.6,
      expectedWinRate: 60,
      expectedProfitFactor: 1.8,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT'],
      requiredIndicators: ['features', 'predictions'],
      requiresML: true,
      requiresOrderbook: false,
      suitableForBeginners: false,
      parameters: {
        stopLossPct: 2.0,
        takeProfitPct: 4.0,
        maxPositionSize: 2.0,
        maxLeverage: 2.0,
        maxDrawdown: 15.0,
        technicalParams: {
          featureWindow: 50,
          predictionHorizon: 24,
          modelConfidenceThreshold: 0.65,
          ensembleVoting: 'soft'
        }
      }
    },
    {
      id: 'scalping',
      name: 'Scalping',
      description: 'High-frequency trading for small, quick profits',
      category: 'scalping',
      riskLevel: 'aggressive',
      timeframe: '1m',
      minCapital: 5000,
      expectedSharpe: 1.1,
      expectedWinRate: 80,
      expectedProfitFactor: 1.2,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT'],
      requiredIndicators: ['orderflow', 'microstructure'],
      requiresML: false,
      requiresOrderbook: true,
      suitableForBeginners: false,
      parameters: {
        stopLossPct: 0.5,
        takeProfitPct: 0.5,
        maxPositionSize: 2.0,
        maxLeverage: 3.0,
        maxDrawdown: 10.0,
        technicalParams: {
          emaFast: 9,
          emaSlow: 21,
          volumeThreshold: 2.0,
          tickSize: 0.01
        }
      }
    },
    {
      id: 'swing_trading',
      name: 'Swing Trading',
      description: 'Captures medium-term price swings',
      category: 'swing_trading',
      riskLevel: 'moderate',
      timeframe: '4h',
      minCapital: 3000,
      expectedSharpe: 1.4,
      expectedWinRate: 55,
      expectedProfitFactor: 2.0,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'],
      requiredIndicators: ['swing_highs', 'swing_lows', 'trend'],
      requiresML: false,
      requiresOrderbook: false,
      suitableForBeginners: true,
      parameters: {
        stopLossPct: 2.0,
        takeProfitPct: 8.0,
        maxPositionSize: 2.0,
        maxLeverage: 1.5,
        maxDrawdown: 15.0,
        technicalParams: {
          swingPeriod: 10,
          trendPeriod: 50,
          macdFast: 12,
          macdSlow: 26,
          macdSignal: 9
        }
      }
    },
    {
      id: 'dca',
      name: 'Dollar Cost Averaging',
      description: 'Systematic periodic buying regardless of price',
      category: 'accumulation',
      riskLevel: 'conservative',
      timeframe: '1d',
      minCapital: 1000,
      expectedSharpe: 0.8,
      expectedWinRate: 60,
      expectedProfitFactor: 1.3,
      recommendedPairs: ['BTC/USDT', 'ETH/USDT'],
      requiredIndicators: ['schedule'],
      requiresML: false,
      requiresOrderbook: false,
      suitableForBeginners: true,
      parameters: {
        stopLossPct: 10.0,
        maxPositionSize: 2.0,
        maxLeverage: 1.0,
        maxDrawdown: 30.0,
        technicalParams: {
          buyFrequencyDays: 7,
          buyAmountPct: 1.0,
          useValueAveraging: false,
          buyOnDipThreshold: 5.0
        }
      }
    }
  ];

  // Filter templates based on criteria
  const filteredTemplates = useMemo(() => {
    return templates.filter(template => {
      // Category filter
      if (selectedCategory !== 'all' && template.category !== selectedCategory) {
        return false;
      }

      // Risk level filter
      if (selectedRiskLevel !== 'all' && template.riskLevel !== selectedRiskLevel) {
        return false;
      }

      // Capital requirement filter
      if (template.minCapital > userCapital) {
        return false;
      }

      // Experience filter
      if (userExperience === 'beginner' && !template.suitableForBeginners) {
        return false;
      }

      // Search filter
      if (searchTerm) {
        const searchLower = searchTerm.toLowerCase();
        return (
          template.name.toLowerCase().includes(searchLower) ||
          template.description.toLowerCase().includes(searchLower) ||
          template.category.toLowerCase().includes(searchLower)
        );
      }

      return true;
    });
  }, [selectedCategory, selectedRiskLevel, searchTerm, userCapital, userExperience, templates]);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'market_making': return <ScaleIcon className="h-5 w-5" />;
      case 'mean_reversion': return <ArrowTrendingUpIcon className="h-5 w-5" />;
      case 'trend_following': return <ChartBarIcon className="h-5 w-5" />;
      case 'arbitrage': return <SparklesIcon className="h-5 w-5" />;
      case 'machine_learning': return <BeakerIcon className="h-5 w-5" />;
      case 'scalping': return <ClockIcon className="h-5 w-5" />;
      case 'swing_trading': return <CogIcon className="h-5 w-5" />;
      case 'accumulation': return <CurrencyDollarIcon className="h-5 w-5" />;
      default: return <ChartBarIcon className="h-5 w-5" />;
    }
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'conservative': return 'text-green-600 bg-green-100';
      case 'moderate': return 'text-yellow-600 bg-yellow-100';
      case 'aggressive': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
          Strategy Templates
        </h2>
        <p className="text-gray-500 dark:text-gray-400">
          Choose from pre-configured strategies optimized for different market conditions
        </p>
      </div>

      {/* Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Search */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Search
            </label>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search strategies..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                       bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Category Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Category
            </label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                       bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="all">All Categories</option>
              <option value="market_making">Market Making</option>
              <option value="mean_reversion">Mean Reversion</option>
              <option value="trend_following">Trend Following</option>
              <option value="arbitrage">Arbitrage</option>
              <option value="machine_learning">Machine Learning</option>
              <option value="scalping">Scalping</option>
              <option value="swing_trading">Swing Trading</option>
              <option value="accumulation">Accumulation</option>
            </select>
          </div>

          {/* Risk Level Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Risk Level
            </label>
            <select
              value={selectedRiskLevel}
              onChange={(e) => setSelectedRiskLevel(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg 
                       bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="all">All Risk Levels</option>
              <option value="conservative">Conservative</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>

          {/* User Info */}
          <div className="text-sm text-gray-500 dark:text-gray-400">
            <div>Capital: ${userCapital.toLocaleString()}</div>
            <div>Experience: {userExperience}</div>
            <div className="text-primary-600 dark:text-primary-400">
              {filteredTemplates.length} strategies available
            </div>
          </div>
        </div>
      </div>

      {/* Template Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredTemplates.map((template) => (
          <div
            key={template.id}
            className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 
                     hover:border-primary-500 transition-colors cursor-pointer"
            onClick={() => setShowDetails(showDetails === template.id ? null : template.id)}
          >
            {/* Template Header */}
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center">
                  {getCategoryIcon(template.category)}
                  <h3 className="ml-2 text-lg font-semibold text-gray-900 dark:text-white">
                    {template.name}
                  </h3>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRiskLevelColor(template.riskLevel)}`}>
                  {template.riskLevel}
                </span>
              </div>

              <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
                {template.description}
              </p>

              {/* Key Metrics */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                <div className="text-center">
                  <div className="text-xs text-gray-500 dark:text-gray-400">Sharpe</div>
                  <div className="text-sm font-semibold text-gray-900 dark:text-white">
                    {template.expectedSharpe}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-500 dark:text-gray-400">Win Rate</div>
                  <div className="text-sm font-semibold text-gray-900 dark:text-white">
                    {template.expectedWinRate}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-500 dark:text-gray-400">Profit Factor</div>
                  <div className="text-sm font-semibold text-gray-900 dark:text-white">
                    {template.expectedProfitFactor}
                  </div>
                </div>
              </div>

              {/* Requirements */}
              <div className="flex flex-wrap gap-2 mb-4">
                <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                  {template.timeframe}
                </span>
                <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                  Min ${template.minCapital}
                </span>
                {template.requiresML && (
                  <span className="px-2 py-1 text-xs bg-purple-100 dark:bg-purple-900 text-purple-600 dark:text-purple-400 rounded">
                    ML Required
                  </span>
                )}
                {template.requiresOrderbook && (
                  <span className="px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-400 rounded">
                    Orderbook
                  </span>
                )}
              </div>

              {/* Expanded Details */}
              {showDetails === template.id && (
                <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4 space-y-3">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Risk Parameters (Quinn's Requirements)
                    </h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Stop Loss:</span>
                        <span className="font-medium">{template.parameters.stopLossPct}%</span>
                      </div>
                      {template.parameters.takeProfitPct && (
                        <div className="flex justify-between">
                          <span className="text-gray-500">Take Profit:</span>
                          <span className="font-medium">{template.parameters.takeProfitPct}%</span>
                        </div>
                      )}
                      <div className="flex justify-between">
                        <span className="text-gray-500">Max Position:</span>
                        <span className="font-medium">{template.parameters.maxPositionSize}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Max Leverage:</span>
                        <span className="font-medium">{template.parameters.maxLeverage}x</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Recommended Pairs
                    </h4>
                    <div className="flex flex-wrap gap-1">
                      {template.recommendedPairs.map(pair => (
                        <span key={pair} className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                          {pair}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Required Indicators
                    </h4>
                    <div className="flex flex-wrap gap-1">
                      {template.requiredIndicators.map(indicator => (
                        <span key={indicator} className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                          {indicator}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Select Button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectTemplate(template);
                }}
                className="w-full mt-4 px-4 py-2 bg-primary-600 text-white rounded-lg 
                         hover:bg-primary-700 transition-colors flex items-center justify-center"
              >
                <ShieldCheckIcon className="h-5 w-5 mr-2" />
                Use This Template
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* No Results */}
      {filteredTemplates.length === 0 && (
        <div className="text-center py-12 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">
            No strategies match your criteria. Try adjusting the filters.
          </p>
        </div>
      )}
    </div>
  );
};

export default TemplateSelector;