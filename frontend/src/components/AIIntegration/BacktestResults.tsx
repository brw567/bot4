import React, { useState } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
} from 'recharts';
import { CalendarIcon, ChartBarIcon, CurrencyDollarIcon, ArrowTrendingUpIcon } from '@heroicons/react/24/outline';

interface BacktestResultsProps {
  data: {
    data: Array<{
      date: string;
      profit: number;
      cumulativeProfit: number;
      predictions: number;
      successRate: number;
      sharpeRatio: number;
    }>;
    summary: {
      totalProfit: number;
      avgDailyProfit: number;
      winRate: number;
      sharpeRatio: number;
      maxDrawdown: number;
      profitFactor: number;
    };
  };
}

const BacktestResults: React.FC<BacktestResultsProps> = ({ data }) => {
  const [showBrush, setShowBrush] = useState(true);
  const [selectedMetric, setSelectedMetric] = useState<'profit' | 'successRate' | 'sharpe'>('profit');

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return `${date.getMonth() + 1}/${date.getDate()}`;
  };

  const summaryCards = [
    {
      title: 'Total Profit',
      value: `$${data.summary.totalProfit.toFixed(2)}`,
      icon: CurrencyDollarIcon,
      color: data.summary.totalProfit > 0 ? 'text-green-600' : 'text-red-600',
    },
    {
      title: 'Win Rate',
      value: `${(data.summary.winRate * 100).toFixed(1)}%`,
      icon: ChartBarIcon,
      color: data.summary.winRate > 0.8 ? 'text-green-600' : 'text-yellow-600',
    },
    {
      title: 'Sharpe Ratio',
      value: data.summary.sharpeRatio.toFixed(2),
      icon: ArrowTrendingUpIcon,
      color: data.summary.sharpeRatio > 1.5 ? 'text-green-600' : 'text-yellow-600',
    },
    {
      title: 'Max Drawdown',
      value: `${(data.summary.maxDrawdown * 100).toFixed(1)}%`,
      icon: CalendarIcon,
      color: data.summary.maxDrawdown > -0.1 ? 'text-yellow-600' : 'text-red-600',
    },
  ];

  // Calculate additional metrics
  const winningDays = data.data.filter(d => d.profit > 0).length;
  const losingDays = data.data.filter(d => d.profit < 0).length;
  const avgWin = data.data.filter(d => d.profit > 0).reduce((sum, d) => sum + d.profit, 0) / winningDays || 0;
  const avgLoss = data.data.filter(d => d.profit < 0).reduce((sum, d) => sum + Math.abs(d.profit), 0) / losingDays || 0;

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {summaryCards.map((card) => (
          <div key={card.title} className="metric-card">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">{card.title}</p>
                <p className={`text-2xl font-semibold ${card.color}`}>{card.value}</p>
              </div>
              <card.icon className="h-8 w-8 text-gray-400" />
            </div>
          </div>
        ))}
      </div>

      {/* Cumulative Profit Chart */}
      <div className="metric-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">Cumulative Profit</h3>
          <button
            onClick={() => setShowBrush(!showBrush)}
            className="text-sm text-primary-600 hover:text-primary-700"
          >
            {showBrush ? 'Hide' : 'Show'} Timeline Brush
          </button>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={data.data}>
            <defs>
              <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#10B981" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="date" 
              tick={{ fill: '#9CA3AF' }}
              tickFormatter={formatDate}
            />
            <YAxis tick={{ fill: '#9CA3AF' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '0.375rem',
              }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: any) => `$${value.toFixed(2)}`}
              labelFormatter={(label) => new Date(label).toLocaleDateString()}
            />
            <ReferenceLine y={0} stroke="#6B7280" strokeDasharray="3 3" />
            <Area
              type="monotone"
              dataKey="cumulativeProfit"
              stroke="#10B981"
              strokeWidth={2}
              fill="url(#profitGradient)"
            />
            {showBrush && (
              <Brush 
                dataKey="date" 
                height={30} 
                stroke="#374151"
                fill="#1F2937"
                tickFormatter={formatDate}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Daily Performance */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Daily Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="date" 
                tick={{ fill: '#9CA3AF' }}
                tickFormatter={formatDate}
              />
              <YAxis tick={{ fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.375rem',
                }}
                labelStyle={{ color: '#9CA3AF' }}
                formatter={(value: any) => `$${value.toFixed(2)}`}
                labelFormatter={(label) => new Date(label).toLocaleDateString()}
              />
              <ReferenceLine y={0} stroke="#6B7280" />
              <Bar 
                dataKey="profit" 
                fill="#3B82F6"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics Comparison */}
        <div className="metric-card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">Key Metrics</h3>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value as any)}
              className="text-sm px-3 py-1 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600"
            >
              <option value="profit">Daily Profit</option>
              <option value="successRate">Success Rate</option>
              <option value="sharpe">Sharpe Ratio</option>
            </select>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data.data}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="date" 
                tick={{ fill: '#9CA3AF' }}
                tickFormatter={formatDate}
              />
              <YAxis 
                tick={{ fill: '#9CA3AF' }}
                domain={selectedMetric === 'successRate' ? [0, 1] : undefined}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.375rem',
                }}
                labelStyle={{ color: '#9CA3AF' }}
                formatter={(value: any) => 
                  selectedMetric === 'successRate' 
                    ? `${(value * 100).toFixed(1)}%`
                    : selectedMetric === 'profit'
                    ? `$${value.toFixed(2)}`
                    : value.toFixed(2)
                }
                labelFormatter={(label) => new Date(label).toLocaleDateString()}
              />
              {selectedMetric === 'successRate' && (
                <ReferenceLine y={0.8} stroke="#10B981" strokeDasharray="3 3" label="Target (80%)" />
              )}
              <Line
                type="monotone"
                dataKey={selectedMetric === 'sharpe' ? 'sharpeRatio' : selectedMetric}
                stroke="#8B5CF6"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Performance Statistics</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Trading Days</span>
              <span className="font-medium text-gray-900 dark:text-white">{data.data.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Winning Days</span>
              <span className="font-medium text-green-600">{winningDays}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Losing Days</span>
              <span className="font-medium text-red-600">{losingDays}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Win/Loss Ratio</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {(winningDays / losingDays).toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Profit Analysis</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Average Win</span>
              <span className="font-medium text-green-600">${avgWin.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Average Loss</span>
              <span className="font-medium text-red-600">${avgLoss.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Profit Factor</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {data.summary.profitFactor.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Daily Avg</span>
              <span className="font-medium text-gray-900 dark:text-white">
                ${data.summary.avgDailyProfit.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Risk Metrics</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Sharpe Ratio</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {data.summary.sharpeRatio.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Max Drawdown</span>
              <span className="font-medium text-red-600">
                {(data.summary.maxDrawdown * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Daily Volatility</span>
              <span className="font-medium text-gray-900 dark:text-white">2.3%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Risk/Reward</span>
              <span className="font-medium text-gray-900 dark:text-white">1:2.1</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BacktestResults;