import React, { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
} from 'recharts';
import {
  ArrowUpIcon,
  ArrowDownIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  InformationCircleIcon,
  DocumentArrowDownIcon
} from '@heroicons/react/24/outline';
import { format } from 'date-fns';

interface BacktestResult {
  config: {
    strategyName: string;
    startDate: string;
    endDate: string;
    initialCapital: number;
    symbols: string[];
    mode: string;
  };
  metrics: {
    totalReturn: number;
    totalReturnPct: number;
    annualReturn: number;
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;
    maxDrawdown: number;
    maxDrawdownDurationDays: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    avgWin: number;
    avgLoss: number;
    profitFactor: number;
    expectancy: number;
    bestTrade: number;
    worstTrade: number;
    avgTrade: number;
    avgTradeDurationHours: number;
    var95: number;
    cvar95: number;
    kellyCriterion: number;
    returnSkewness: number;
    returnKurtosis: number;
    totalCommission: number;
    totalSlippage: number;
    stopLossTriggers: number;
    stopLossSaved: number;
  };
  equityCurve: Array<{ date: string; value: number }>;
  dailyReturns: number[];
  monthlyReturns: Array<{ month: string; return: number }>;
  drawdownSeries: Array<{ date: string; drawdown: number }>;
  trades: Array<{
    timestamp: string;
    symbol: string;
    side: string;
    entryPrice: number;
    exitPrice: number;
    pnl: number;
    pnlPct: number;
    exitReason: string;
  }>;
  tradeAnalysis?: {
    bySymbol: Record<string, { count: number; winRate: number; avgPnl: number }>;
    byHour: Record<number, { count: number; avgReturn: number }>;
    maxConsecutiveWins: number;
    maxConsecutiveLosses: number;
  };
  riskAnalysis?: {
    dailyVar99: number;
    dailyVar95: number;
    dailyCvar99: number;
    dailyCvar95: number;
    downsideDeviation: number;
    upsideDeviation: number;
    tailRatio: number;
    maxDailyLoss: number;
    maxDailyGain: number;
  };
  performanceAttribution?: Record<string, {
    count: number;
    totalPnl: number;
    contributionPct: number;
    avgPnl: number;
  }>;
  monteCarloResults?: {
    returnStatistics: Record<string, number>;
    drawdownStatistics: Record<string, number>;
    probabilityOfProfit: number;
    probabilityOfLoss: number;
  };
}

interface BacktestResultsVisualizationProps {
  result: BacktestResult;
  onExport?: () => void;
}

const BacktestResultsVisualization: React.FC<BacktestResultsVisualizationProps> = ({
  result,
  onExport
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'trades' | 'risk' | 'analysis'>('overview');
  const [selectedMetric, setSelectedMetric] = useState<'equity' | 'returns' | 'drawdown'>('equity');

  // Color scheme
  const COLORS = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#8b5cf6', '#ec4899'];

  // Format numbers
  const formatNumber = (num: number, decimals: number = 2) => {
    return num?.toFixed(decimals) || '0.00';
  };

  const formatPercent = (num: number) => {
    return `${num >= 0 ? '+' : ''}${formatNumber(num)}%`;
  };

  const formatCurrency = (num: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(num);
  };

  // Calculate additional metrics
  const recoveryFactor = useMemo(() => {
    if (result.metrics.maxDrawdown === 0) return Infinity;
    return result.metrics.totalReturnPct / Math.abs(result.metrics.maxDrawdown);
  }, [result]);

  const payoffRatio = useMemo(() => {
    if (result.metrics.avgLoss === 0) return Infinity;
    return Math.abs(result.metrics.avgWin / result.metrics.avgLoss);
  }, [result]);

  // Prepare chart data
  const equityChartData = result.equityCurve?.map(point => ({
    date: format(new Date(point.date), 'MMM dd'),
    value: point.value,
    baseline: result.config.initialCapital
  })) || [];

  const returnsDistribution = useMemo(() => {
    const returns = result.dailyReturns || [];
    const bins = 20;
    const min = Math.min(...returns);
    const max = Math.max(...returns);
    const binSize = (max - min) / bins;
    
    const histogram: Record<string, number> = {};
    for (let i = 0; i < bins; i++) {
      const binStart = min + i * binSize;
      const binEnd = binStart + binSize;
      const binLabel = `${formatNumber(binStart * 100, 1)}%`;
      histogram[binLabel] = returns.filter(r => r >= binStart && r < binEnd).length;
    }
    
    return Object.entries(histogram).map(([bin, count]) => ({ bin, count }));
  }, [result.dailyReturns]);

  const monthlyReturnsData = result.monthlyReturns?.map(m => ({
    month: m.month,
    return: m.return * 100,
    positive: m.return >= 0
  })) || [];

  // Risk metrics for radar chart
  const riskRadarData = [
    { metric: 'Sharpe', value: Math.min(result.metrics.sharpeRatio, 3), fullMark: 3 },
    { metric: 'Sortino', value: Math.min(result.metrics.sortinoRatio, 3), fullMark: 3 },
    { metric: 'Calmar', value: Math.min(result.metrics.calmarRatio, 3), fullMark: 3 },
    { metric: 'Win Rate', value: result.metrics.winRate / 100, fullMark: 1 },
    { metric: 'Profit Factor', value: Math.min(result.metrics.profitFactor / 3, 1), fullMark: 1 },
    { metric: 'Recovery', value: Math.min(recoveryFactor / 3, 1), fullMark: 1 }
  ];

  // Trade distribution by exit reason
  const exitReasonDistribution = useMemo(() => {
    const distribution: Record<string, number> = {};
    result.trades?.forEach(trade => {
      distribution[trade.exitReason] = (distribution[trade.exitReason] || 0) + 1;
    });
    return Object.entries(distribution).map(([reason, count]) => ({
      name: reason,
      value: count
    }));
  }, [result.trades]);

  // Performance by time of day
  const hourlyPerformance = useMemo(() => {
    if (!result.tradeAnalysis?.byHour) return [];
    return Object.entries(result.tradeAnalysis.byHour).map(([hour, data]) => ({
      hour: `${hour}:00`,
      avgReturn: data.avgReturn,
      count: data.count
    }));
  }, [result.tradeAnalysis]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              Backtest Results - {result.config.strategyName}
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {format(new Date(result.config.startDate), 'MMM dd, yyyy')} - 
              {format(new Date(result.config.endDate), 'MMM dd, yyyy')} | 
              {result.config.symbols.join(', ')} | 
              Mode: {result.config.mode}
            </p>
          </div>
          <button
            onClick={onExport}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 flex items-center"
          >
            <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
            Export Report
          </button>
        </div>
      </div>

      {/* Key Metrics Summary */}
      <div className="p-6 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="text-center">
          <div className={`text-2xl font-bold ${result.metrics.totalReturnPct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {formatPercent(result.metrics.totalReturnPct)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Total Return</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {formatNumber(result.metrics.sharpeRatio)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Sharpe Ratio</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">
            {formatPercent(Math.abs(result.metrics.maxDrawdown))}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Max Drawdown</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {formatPercent(result.metrics.winRate)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Win Rate</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {formatNumber(result.metrics.profitFactor)}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Profit Factor</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {result.metrics.totalTrades}
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Total Trades</div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200 dark:border-gray-700">
        <nav className="flex space-x-8 px-6">
          {(['overview', 'trades', 'risk', 'analysis'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-3 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 dark:text-gray-400'
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="p-6">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Equity Curve */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white">
                  Equity Curve
                </h3>
                <div className="flex space-x-2">
                  {(['equity', 'returns', 'drawdown'] as const).map(metric => (
                    <button
                      key={metric}
                      onClick={() => setSelectedMetric(metric)}
                      className={`px-3 py-1 text-sm rounded ${
                        selectedMetric === metric
                          ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                          : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                      }`}
                    >
                      {metric.charAt(0).toUpperCase() + metric.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
              
              <ResponsiveContainer width="100%" height={400}>
                {selectedMetric === 'equity' ? (
                  <AreaChart data={equityChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip formatter={(value: number) => formatCurrency(value)} />
                    <ReferenceLine y={result.config.initialCapital} stroke="#666" strokeDasharray="3 3" />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#3b82f6" 
                      fill="#3b82f6" 
                      fillOpacity={0.3} 
                    />
                    <Brush dataKey="date" height={30} />
                  </AreaChart>
                ) : selectedMetric === 'returns' ? (
                  <BarChart data={monthlyReturnsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip formatter={(value: number) => `${formatNumber(value)}%`} />
                    <ReferenceLine y={0} stroke="#666" />
                    <Bar dataKey="return" fill={(entry: any) => entry.positive ? '#10b981' : '#ef4444'} />
                  </BarChart>
                ) : (
                  <AreaChart data={result.drawdownSeries?.map(d => ({
                    date: format(new Date(d.date), 'MMM dd'),
                    drawdown: Math.abs(d.drawdown)
                  })) || []}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip formatter={(value: number) => `${formatNumber(value)}%`} />
                    <Area 
                      type="monotone" 
                      dataKey="drawdown" 
                      stroke="#ef4444" 
                      fill="#ef4444" 
                      fillOpacity={0.3} 
                    />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>

            {/* Performance Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Returns Distribution */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Returns Distribution
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={returnsDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="bin" angle={-45} textAnchor="end" height={60} />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Risk Radar */}
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Risk Profile
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={riskRadarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis angle={90} domain={[0, 1]} />
                    <Radar name="Metrics" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'trades' && (
          <div className="space-y-6">
            {/* Trade Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Winning Trades</div>
                <div className="text-xl font-bold text-green-600">{result.metrics.winningTrades}</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Losing Trades</div>
                <div className="text-xl font-bold text-red-600">{result.metrics.losingTrades}</div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Avg Win</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {formatCurrency(result.metrics.avgWin)}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Avg Loss</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {formatCurrency(Math.abs(result.metrics.avgLoss))}
                </div>
              </div>
            </div>

            {/* Exit Reason Distribution */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Exit Reasons
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={exitReasonDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={entry => `${entry.name}: ${entry.value}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {exitReasonDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Hourly Performance */}
              {hourlyPerformance.length > 0 && (
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                    Performance by Hour
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={hourlyPerformance}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="hour" />
                      <YAxis />
                      <Tooltip formatter={(value: number) => `${formatNumber(value)}%`} />
                      <Bar dataKey="avgReturn" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            {/* Trades Table */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Recent Trades
              </h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-900">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Symbol</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Side</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entry</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Exit</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">P&L</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Return</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Exit Reason</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                    {result.trades?.slice(-10).reverse().map((trade, index) => (
                      <tr key={index}>
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                          {format(new Date(trade.timestamp), 'MMM dd HH:mm')}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">{trade.symbol}</td>
                        <td className="px-4 py-3 text-sm">
                          <span className={`px-2 py-1 text-xs rounded ${
                            trade.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                          }`}>
                            {trade.side.toUpperCase()}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                          {formatCurrency(trade.entryPrice)}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-900 dark:text-white">
                          {formatCurrency(trade.exitPrice)}
                        </td>
                        <td className={`px-4 py-3 text-sm font-medium ${
                          trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatCurrency(trade.pnl)}
                        </td>
                        <td className={`px-4 py-3 text-sm font-medium ${
                          trade.pnlPct >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatPercent(trade.pnlPct * 100)}
                        </td>
                        <td className="px-4 py-3 text-sm text-gray-500">
                          {trade.exitReason}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'risk' && (
          <div className="space-y-6">
            {/* Risk Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Sharpe Ratio</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(result.metrics.sharpeRatio)}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Sortino Ratio</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(result.metrics.sortinoRatio)}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Calmar Ratio</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {formatNumber(result.metrics.calmarRatio)}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">VaR (95%)</div>
                <div className="text-xl font-bold text-red-600">
                  {formatPercent(result.metrics.var95 * 100)}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">CVaR (95%)</div>
                <div className="text-xl font-bold text-red-600">
                  {formatPercent(result.metrics.cvar95 * 100)}
                </div>
              </div>
              <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                <div className="text-sm text-gray-500 dark:text-gray-400">Kelly Criterion</div>
                <div className="text-xl font-bold text-gray-900 dark:text-white">
                  {formatPercent(result.metrics.kellyCriterion * 100)}
                </div>
              </div>
            </div>

            {/* Stop Loss Analysis (Quinn's requirement) */}
            <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
              <h3 className="text-lg font-medium text-red-800 dark:text-red-200 mb-4">
                Stop Loss Analysis
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                <div>
                  <div className="text-sm text-red-600 dark:text-red-400">Stop Loss Triggers</div>
                  <div className="text-xl font-bold text-red-800 dark:text-red-200">
                    {result.metrics.stopLossTriggers}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-red-600 dark:text-red-400">Amount Saved</div>
                  <div className="text-xl font-bold text-red-800 dark:text-red-200">
                    {formatCurrency(Math.abs(result.metrics.stopLossSaved))}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-red-600 dark:text-red-400">% of Total Trades</div>
                  <div className="text-xl font-bold text-red-800 dark:text-red-200">
                    {formatPercent((result.metrics.stopLossTriggers / result.metrics.totalTrades) * 100)}
                  </div>
                </div>
              </div>
            </div>

            {/* Advanced Risk Metrics */}
            {result.riskAnalysis && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Advanced Risk Analysis
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Downside Deviation</div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {formatPercent(result.riskAnalysis.downsideDeviation * 100)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Upside Deviation</div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {formatPercent(result.riskAnalysis.upsideDeviation * 100)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Tail Ratio</div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {formatNumber(result.riskAnalysis.tailRatio)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Max Daily Loss</div>
                    <div className="text-lg font-bold text-red-600">
                      {formatPercent(result.riskAnalysis.maxDailyLoss * 100)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Max Daily Gain</div>
                    <div className="text-lg font-bold text-green-600">
                      {formatPercent(result.riskAnalysis.maxDailyGain * 100)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Skewness</div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {formatNumber(result.metrics.returnSkewness)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="space-y-6">
            {/* Performance Attribution */}
            {result.performanceAttribution && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Performance Attribution
                </h3>
                <div className="space-y-2">
                  {Object.entries(result.performanceAttribution).map(([reason, data]) => (
                    <div key={reason} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                      <div className="flex items-center">
                        <span className="font-medium text-gray-900 dark:text-white capitalize">
                          {reason.replace('_', ' ')}
                        </span>
                        <span className="ml-2 text-sm text-gray-500">
                          ({data.count} trades)
                        </span>
                      </div>
                      <div className="flex items-center space-x-4">
                        <span className={`font-medium ${data.totalPnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {formatCurrency(data.totalPnl)}
                        </span>
                        <span className="text-sm text-gray-500">
                          {formatPercent(data.contributionPct)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Monte Carlo Results */}
            {result.monteCarloResults && (
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <h3 className="text-lg font-medium text-blue-800 dark:text-blue-200 mb-4">
                  Monte Carlo Simulation Results
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <div className="text-sm text-blue-600">Probability of Profit</div>
                    <div className="text-xl font-bold text-blue-800 dark:text-blue-200">
                      {formatPercent(result.monteCarloResults.probabilityOfProfit)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-blue-600">Mean Return</div>
                    <div className="text-xl font-bold text-blue-800 dark:text-blue-200">
                      {formatPercent(result.monteCarloResults.returnStatistics.mean)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-blue-600">Worst Case (5%)</div>
                    <div className="text-xl font-bold text-blue-800 dark:text-blue-200">
                      {formatPercent(result.monteCarloResults.returnStatistics.percentile_5 || 0)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-blue-600">Best Case (95%)</div>
                    <div className="text-xl font-bold text-blue-800 dark:text-blue-200">
                      {formatPercent(result.monteCarloResults.returnStatistics.percentile_95 || 0)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Trade Consistency */}
            {result.tradeAnalysis && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Trade Consistency
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Max Consecutive Wins</div>
                    <div className="text-xl font-bold text-green-600">
                      {result.tradeAnalysis.maxConsecutiveWins}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Max Consecutive Losses</div>
                    <div className="text-xl font-bold text-red-600">
                      {result.tradeAnalysis.maxConsecutiveLosses}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Expectancy</div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">
                      {formatCurrency(result.metrics.expectancy)}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-gray-500">Payoff Ratio</div>
                    <div className="text-xl font-bold text-gray-900 dark:text-white">
                      {formatNumber(payoffRatio)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Costs Analysis */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                Trading Costs
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="text-sm text-gray-500">Total Commission</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {formatCurrency(result.metrics.totalCommission)}
                  </div>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg">
                  <div className="text-sm text-gray-500">Total Slippage</div>
                  <div className="text-xl font-bold text-gray-900 dark:text-white">
                    {formatCurrency(result.metrics.totalSlippage)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BacktestResultsVisualization;