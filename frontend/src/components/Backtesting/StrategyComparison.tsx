import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter
} from 'recharts';
import {
  ScaleIcon,
  ChartBarIcon,
  TrophyIcon,
  ExclamationTriangleIcon,
  DocumentArrowDownIcon
} from '@heroicons/react/24/outline';

interface StrategyResult {
  strategyName: string;
  totalReturn: number;
  annualReturn: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTradeDuration: number;
  expectancy: number;
  var95: number;
  stopLossTriggers: number;
}

interface StrategyComparisonProps {
  results: StrategyResult[];
  onExportComparison?: () => void;
}

const StrategyComparison: React.FC<StrategyComparisonProps> = ({
  results,
  onExportComparison
}) => {
  const [selectedMetric, setSelectedMetric] = useState<'returns' | 'risk' | 'efficiency'>('returns');
  const [sortBy, setSortBy] = useState<keyof StrategyResult>('totalReturn');
  const [showTopN, setShowTopN] = useState<number>(10);

  // Color palette for strategies
  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316'];

  // Sort strategies
  const sortedResults = useMemo(() => {
    return [...results].sort((a, b) => {
      const aVal = a[sortBy] as number;
      const bVal = b[sortBy] as number;
      return bVal - aVal;
    }).slice(0, showTopN);
  }, [results, sortBy, showTopN]);

  // Prepare comparison data
  const comparisonMetrics = [
    { key: 'totalReturn', label: 'Total Return', format: 'percent' },
    { key: 'sharpeRatio', label: 'Sharpe Ratio', format: 'number' },
    { key: 'maxDrawdown', label: 'Max Drawdown', format: 'percent', inverse: true },
    { key: 'winRate', label: 'Win Rate', format: 'percent' },
    { key: 'profitFactor', label: 'Profit Factor', format: 'number' },
    { key: 'expectancy', label: 'Expectancy', format: 'currency' }
  ];

  // Prepare radar chart data for top strategies
  const radarData = useMemo(() => {
    const metrics = ['Sharpe', 'Win Rate', 'Profit Factor', 'Risk Adj.', 'Consistency'];
    return metrics.map(metric => {
      const point: any = { metric };
      sortedResults.slice(0, 5).forEach(result => {
        let value = 0;
        switch (metric) {
          case 'Sharpe':
            value = Math.min(result.sharpeRatio / 3, 1);
            break;
          case 'Win Rate':
            value = result.winRate / 100;
            break;
          case 'Profit Factor':
            value = Math.min(result.profitFactor / 3, 1);
            break;
          case 'Risk Adj.':
            value = Math.min(result.calmarRatio / 3, 1);
            break;
          case 'Consistency':
            value = 1 - Math.min(Math.abs(result.maxDrawdown) / 30, 1);
            break;
        }
        point[result.strategyName] = value;
      });
      return point;
    });
  }, [sortedResults]);

  // Risk-Return scatter plot data
  const scatterData = useMemo(() => {
    return results.map((result, index) => ({
      x: Math.abs(result.maxDrawdown),
      y: result.totalReturn,
      z: result.sharpeRatio,
      name: result.strategyName,
      fill: COLORS[index % COLORS.length]
    }));
  }, [results]);

  // Efficiency metrics
  const efficiencyData = useMemo(() => {
    return sortedResults.map(result => ({
      strategy: result.strategyName,
      tradesPerProfit: result.totalTrades > 0 ? result.totalReturn / result.totalTrades : 0,
      riskAdjustedReturn: result.sharpeRatio * (1 - Math.abs(result.maxDrawdown) / 100),
      efficiency: result.profitFactor * result.winRate / 100
    }));
  }, [sortedResults]);

  // Calculate best strategy for each metric
  const bestStrategies = useMemo(() => {
    const metrics: (keyof StrategyResult)[] = [
      'totalReturn', 'sharpeRatio', 'winRate', 'profitFactor', 'calmarRatio'
    ];
    
    return metrics.map(metric => {
      const best = results.reduce((prev, current) => {
        if (metric === 'maxDrawdown') {
          return Math.abs(current[metric] as number) < Math.abs(prev[metric] as number) ? current : prev;
        }
        return (current[metric] as number) > (prev[metric] as number) ? current : prev;
      });
      
      return {
        metric: metric.replace(/([A-Z])/g, ' $1').trim(),
        strategy: best.strategyName,
        value: best[metric]
      };
    });
  }, [results]);

  const formatValue = (value: number, format: string) => {
    switch (format) {
      case 'percent':
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD'
        }).format(value);
      case 'number':
      default:
        return value.toFixed(2);
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
              <ScaleIcon className="h-6 w-6 mr-2" />
              Strategy Comparison
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Compare performance across {results.length} strategies
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <select
              value={showTopN}
              onChange={(e) => setShowTopN(Number(e.target.value))}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-sm"
            >
              <option value={5}>Top 5</option>
              <option value={10}>Top 10</option>
              <option value={20}>Top 20</option>
              <option value={results.length}>All</option>
            </select>
            <button
              onClick={onExportComparison}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center"
            >
              <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
              Export
            </button>
          </div>
        </div>
      </div>

      {/* Best Performers */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4 flex items-center">
          <TrophyIcon className="h-5 w-5 mr-2 text-yellow-500" />
          Best Performers
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          {bestStrategies.map((best, index) => (
            <div key={index} className="text-center p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
              <div className="text-xs text-gray-500 dark:text-gray-400 uppercase">
                {best.metric}
              </div>
              <div className="text-sm font-bold text-gray-900 dark:text-white mt-1">
                {best.strategy}
              </div>
              <div className="text-sm text-primary-600 mt-1">
                {formatValue(best.value as number, 
                  best.metric.includes('Return') || best.metric.includes('Rate') ? 'percent' : 'number'
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Metric Selector */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">View:</span>
          {(['returns', 'risk', 'efficiency'] as const).map(metric => (
            <button
              key={metric}
              onClick={() => setSelectedMetric(metric)}
              className={`px-4 py-2 text-sm rounded-lg capitalize ${
                selectedMetric === metric
                  ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                  : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
              }`}
            >
              {metric}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        {selectedMetric === 'returns' && (
          <div className="space-y-6">
            {/* Returns Comparison Bar Chart */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Returns Comparison
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={sortedResults}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="strategyName" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip formatter={(value: number) => formatValue(value, 'percent')} />
                  <Legend />
                  <Bar dataKey="totalReturn" fill="#3b82f6" name="Total Return %" />
                  <Bar dataKey="annualReturn" fill="#10b981" name="Annual Return %" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Cumulative Comparison */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Key Metrics Comparison
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-900">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                        Strategy
                      </th>
                      {comparisonMetrics.map(metric => (
                        <th key={metric.key} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                          {metric.label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                    {sortedResults.map((result, index) => (
                      <tr key={index}>
                        <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-white">
                          {result.strategyName}
                        </td>
                        {comparisonMetrics.map(metric => (
                          <td key={metric.key} className={`px-4 py-3 text-sm ${
                            metric.inverse 
                              ? result[metric.key as keyof StrategyResult] as number > 0 ? 'text-red-600' : 'text-green-600'
                              : result[metric.key as keyof StrategyResult] as number > 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {formatValue(result[metric.key as keyof StrategyResult] as number, metric.format)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {selectedMetric === 'risk' && (
          <div className="space-y-6">
            {/* Risk-Return Scatter Plot */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Risk-Return Profile
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="x" 
                    name="Max Drawdown %" 
                    label={{ value: 'Max Drawdown %', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    dataKey="y" 
                    name="Total Return %" 
                    label={{ value: 'Total Return %', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload[0]) {
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white dark:bg-gray-800 p-3 rounded shadow-lg border border-gray-200 dark:border-gray-700">
                            <p className="font-medium">{data.name}</p>
                            <p className="text-sm">Return: {formatValue(data.y, 'percent')}</p>
                            <p className="text-sm">Drawdown: {formatValue(data.x, 'percent')}</p>
                            <p className="text-sm">Sharpe: {data.z.toFixed(2)}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Scatter name="Strategies" data={scatterData} fill="#8884d8">
                    {scatterData.map((entry, index) => (
                      <cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
              <p className="text-xs text-gray-500 dark:text-gray-400 text-center mt-2">
                Better strategies appear in the upper-left (high return, low drawdown)
              </p>
            </div>

            {/* Risk Metrics Radar */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Risk Profile Comparison (Top 5)
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis angle={90} domain={[0, 1]} />
                  {sortedResults.slice(0, 5).map((result, index) => (
                    <Radar
                      key={result.strategyName}
                      name={result.strategyName}
                      dataKey={result.strategyName}
                      stroke={COLORS[index % COLORS.length]}
                      fill={COLORS[index % COLORS.length]}
                      fillOpacity={0.3}
                    />
                  ))}
                  <Legend />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Risk Metrics Table */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Risk Metrics Detail
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {sortedResults.slice(0, 6).map((result, index) => (
                  <div key={index} className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
                    <h4 className="font-medium text-gray-900 dark:text-white mb-3">
                      {result.strategyName}
                    </h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Sharpe Ratio:</span>
                        <span className="font-medium">{result.sharpeRatio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Sortino Ratio:</span>
                        <span className="font-medium">{result.sortinoRatio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Calmar Ratio:</span>
                        <span className="font-medium">{result.calmarRatio.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Max Drawdown:</span>
                        <span className="font-medium text-red-600">
                          {formatValue(Math.abs(result.maxDrawdown), 'percent')}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">VaR (95%):</span>
                        <span className="font-medium text-orange-600">
                          {formatValue(result.var95, 'percent')}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Stop Loss Hits:</span>
                        <span className="font-medium">{result.stopLossTriggers}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedMetric === 'efficiency' && (
          <div className="space-y-6">
            {/* Efficiency Metrics */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Trading Efficiency
              </h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={efficiencyData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="strategy" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="tradesPerProfit" fill="#3b82f6" name="Return per Trade" />
                  <Bar dataKey="riskAdjustedReturn" fill="#10b981" name="Risk-Adjusted Return" />
                  <Bar dataKey="efficiency" fill="#f59e0b" name="Overall Efficiency" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Trade Statistics */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Trade Statistics
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sortedResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="strategyName" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="totalTrades" fill="#8b5cf6" name="Total Trades" />
                  </BarChart>
                </ResponsiveContainer>
                
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={sortedResults}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="strategyName" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip formatter={(value: number) => `${value.toFixed(1)} hours`} />
                    <Bar dataKey="avgTradeDuration" fill="#ec4899" name="Avg Trade Duration" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Efficiency Rankings */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Efficiency Rankings
              </h3>
              <div className="space-y-2">
                {sortedResults.map((result, index) => {
                  const efficiency = result.profitFactor * result.winRate / 100;
                  const maxEfficiency = Math.max(...results.map(r => r.profitFactor * r.winRate / 100));
                  const percentage = (efficiency / maxEfficiency) * 100;
                  
                  return (
                    <div key={index} className="flex items-center space-x-4">
                      <div className="w-32 text-sm font-medium text-gray-900 dark:text-white">
                        {result.strategyName}
                      </div>
                      <div className="flex-1">
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-6">
                          <div 
                            className="bg-primary-600 h-6 rounded-full flex items-center justify-end pr-2"
                            style={{ width: `${percentage}%` }}
                          >
                            <span className="text-xs text-white font-medium">
                              {efficiency.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      </div>
                      <div className="text-sm text-gray-500 w-20 text-right">
                        Rank #{index + 1}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer with Sort Options */}
      <div className="p-6 border-t border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-500">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as keyof StrategyResult)}
              className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
            >
              <option value="totalReturn">Total Return</option>
              <option value="sharpeRatio">Sharpe Ratio</option>
              <option value="maxDrawdown">Max Drawdown</option>
              <option value="winRate">Win Rate</option>
              <option value="profitFactor">Profit Factor</option>
              <option value="totalTrades">Total Trades</option>
            </select>
          </div>
          <div className="text-sm text-gray-500">
            Showing {Math.min(showTopN, results.length)} of {results.length} strategies
          </div>
        </div>
      </div>
    </div>
  );
};

export default StrategyComparison;