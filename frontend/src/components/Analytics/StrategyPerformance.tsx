import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Cell,
} from 'recharts';
import { StrategyPerformance as StrategyData } from '../../types';

interface StrategyPerformanceProps {
  data: StrategyData[];
  height?: number;
}

const COLORS = [
  '#10b981', '#3b82f6', '#f59e0b', '#ef4444', 
  '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'
];

const StrategyPerformance: React.FC<StrategyPerformanceProps> = ({ 
  data,
  height = 400 
}) => {
  const radarData = useMemo(() => {
    return data.map(strategy => ({
      strategy: strategy.strategy,
      winRate: strategy.winRate,
      avgProfit: strategy.avgProfit * 100, // Convert to percentage
      sharpeRatio: strategy.sharpeRatio * 20, // Scale for visibility
      trades: Math.min(strategy.totalTrades / 10, 100), // Scale trades
    }));
  }, [data]);

  const barData = useMemo(() => {
    return data.map((strategy, index) => ({
      ...strategy,
      color: COLORS[index % COLORS.length],
    }));
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const strategy = data.find(s => s.strategy === label);
      if (!strategy) return null;

      return (
        <div className="bg-white dark:bg-gray-800 p-4 rounded shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-medium text-gray-900 dark:text-white mb-2">
            {strategy.strategy}
          </p>
          <div className="space-y-1">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Win Rate: <span className="font-medium text-green-600">{strategy.winRate.toFixed(1)}%</span>
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Avg Profit: <span className="font-medium">{strategy.avgProfit.toFixed(2)}%</span>
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Total Trades: <span className="font-medium">{strategy.totalTrades}</span>
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              24h P&L: <span className={`font-medium ${strategy.pnl24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                ${strategy.pnl24h.toFixed(2)}
              </span>
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Sharpe Ratio: <span className="font-medium">{strategy.sharpeRatio.toFixed(2)}</span>
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  const totalPnL = data.reduce((sum, s) => sum + s.pnl24h, 0);
  const totalTrades = data.reduce((sum, s) => sum + s.totalTrades, 0);
  const avgWinRate = data.reduce((sum, s) => sum + s.winRate, 0) / data.length;

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Active Strategies</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{data.length}</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Total 24h P&L</p>
          <p className={`text-2xl font-bold ${totalPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            ${totalPnL.toFixed(2)}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Total Trades</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">{totalTrades}</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Avg Win Rate</p>
          <p className="text-2xl font-bold text-green-600">{avgWinRate.toFixed(1)}%</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Bar Chart - Win Rate & P&L */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Strategy Performance Comparison
          </h3>
          <ResponsiveContainer width="100%" height={height}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis 
                dataKey="strategy" 
                angle={-45}
                textAnchor="end"
                height={80}
                className="text-gray-600 dark:text-gray-400"
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                yAxisId="left"
                className="text-gray-600 dark:text-gray-400"
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                yAxisId="right"
                orientation="right"
                className="text-gray-600 dark:text-gray-400"
                tick={{ fontSize: 12 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              
              <Bar yAxisId="left" dataKey="winRate" name="Win Rate (%)" fill="#10b981">
                {barData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
              <Bar yAxisId="right" dataKey="pnl24h" name="24h P&L ($)" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Radar Chart - Multi-dimensional Analysis */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Strategy Characteristics
          </h3>
          <ResponsiveContainer width="100%" height={height}>
            <RadarChart data={radarData}>
              <PolarGrid className="stroke-gray-300 dark:stroke-gray-600" />
              <PolarAngleAxis 
                dataKey="strategy" 
                className="text-gray-600 dark:text-gray-400"
                tick={{ fontSize: 11 }}
              />
              <PolarRadiusAxis 
                angle={90} 
                domain={[0, 100]}
                className="text-gray-600 dark:text-gray-400"
                tick={{ fontSize: 10 }}
              />
              <Radar 
                name="Win Rate" 
                dataKey="winRate" 
                stroke="#10b981" 
                fill="#10b981" 
                fillOpacity={0.3} 
              />
              <Radar 
                name="Avg Profit" 
                dataKey="avgProfit" 
                stroke="#3b82f6" 
                fill="#3b82f6" 
                fillOpacity={0.3} 
              />
              <Radar 
                name="Sharpe Ratio" 
                dataKey="sharpeRatio" 
                stroke="#f59e0b" 
                fill="#f59e0b" 
                fillOpacity={0.3} 
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Strategy Details Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Strategy Details
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Strategy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Active Pairs
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Win Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  Sharpe Ratio
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                  24h P&L
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-800 dark:divide-gray-700">
              {data.map((strategy, index) => (
                <tr key={strategy.strategy}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    <div className="flex items-center">
                      <div 
                        className="h-3 w-3 rounded-full mr-2" 
                        style={{ backgroundColor: COLORS[index % COLORS.length] }}
                      />
                      {strategy.strategy}
                    </div>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex flex-wrap gap-1">
                      {(strategy.activePairs || []).slice(0, 3).map(pair => (
                        <span key={pair} className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                          {pair}
                        </span>
                      ))}
                      {(strategy.activePairs?.length || 0) > 3 && (
                        <span className="px-2 py-0.5 text-xs bg-gray-100 dark:bg-gray-700 rounded">
                          +{strategy.activePairs.length - 3}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    <div className="flex items-center">
                      <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full" 
                          style={{ width: `${strategy.winRate}%` }}
                        />
                      </div>
                      <span className="text-sm font-medium">{strategy.winRate.toFixed(1)}%</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {strategy.sharpeRatio.toFixed(2)}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                    strategy.pnl24h >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    ${strategy.pnl24h.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default StrategyPerformance;