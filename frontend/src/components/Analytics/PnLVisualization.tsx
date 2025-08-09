import React, { useMemo } from 'react';
import {
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';
import { CurrencyDollarIcon, ArrowTrendingUpIcon, ArrowTrendingDownIcon } from '@heroicons/react/24/outline';

interface PnLData {
  timestamp: string;
  pnl: number;
  cumulative: number;
  trades: number;
}

interface PnLVisualizationProps {
  data: PnLData[];
  timeframe: string;
  height?: number;
}

const PnLVisualization: React.FC<PnLVisualizationProps> = ({ 
  data, 
  timeframe,
  height = 400 
}) => {
  const formattedData = useMemo(() => {
    return data.map(item => {
      const date = new Date(item.timestamp);
      const isValidDate = !isNaN(date.getTime());
      
      return {
        ...item,
        time: isValidDate ? format(date, 
          timeframe === '1h' ? 'HH:mm' : 
          timeframe === '24h' ? 'HH:mm' :
          timeframe === '7d' ? 'MM/dd' :
          'MM/dd'
        ) : 'N/A',
        color: item.pnl >= 0 ? '#10b981' : '#ef4444',
      };
    });
  }, [data, timeframe]);

  const stats = useMemo(() => {
    const totalPnL = data[data.length - 1]?.cumulative || 0;
    const profitDays = data.filter(d => d.pnl > 0).length;
    const lossDays = data.filter(d => d.pnl < 0).length;
    const maxProfit = Math.max(...data.map(d => d.pnl));
    const maxLoss = Math.min(...data.map(d => d.pnl));
    const avgPnL = data.reduce((sum, d) => sum + d.pnl, 0) / data.length;

    return {
      totalPnL,
      profitDays,
      lossDays,
      maxProfit,
      maxLoss,
      avgPnL,
      winRate: (profitDays / (profitDays + lossDays)) * 100,
    };
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const pnl = payload[0]?.value || 0;
      const cumulative = payload[1]?.value || 0;
      const trades = payload[0]?.payload?.trades || 0;

      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-medium text-gray-900 dark:text-white mb-1">
            {label}
          </p>
          <p className={`text-sm font-medium ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            P&L: ${pnl.toFixed(2)}
          </p>
          <p className="text-sm text-blue-600 dark:text-blue-400">
            Cumulative: ${cumulative.toFixed(2)}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Trades: {trades}
          </p>
        </div>
      );
    }
    return null;
  };

  const CustomBar = (props: any) => {
    const { fill, x, y, width, height } = props;
    return (
      <g>
        <rect
          x={x}
          y={y}
          width={width}
          height={height}
          fill={props.payload.pnl >= 0 ? '#10b981' : '#ef4444'}
          opacity={0.8}
        />
      </g>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Profit & Loss Analysis
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          {timeframe} performance overview
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Total P&L</p>
              <p className={`text-xl font-bold ${
                stats.totalPnL >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                ${stats.totalPnL.toFixed(2)}
              </p>
            </div>
            <CurrencyDollarIcon className="h-8 w-8 text-gray-400" />
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Win Rate</p>
              <p className="text-xl font-bold text-green-600">
                {stats.winRate.toFixed(1)}%
              </p>
            </div>
            <ArrowTrendingUpIcon className="h-8 w-8 text-green-400" />
          </div>
        </div>

        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-gray-500 dark:text-gray-400">Avg P&L</p>
              <p className={`text-xl font-bold ${
                stats.avgPnL >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                ${stats.avgPnL.toFixed(2)}
              </p>
            </div>
            <div className="text-2xl">📊</div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={formattedData}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
          <XAxis 
            dataKey="time" 
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
          
          {/* P&L Bars */}
          <Bar
            yAxisId="left"
            dataKey="pnl"
            name="P&L"
            shape={<CustomBar />}
          />
          
          {/* Cumulative Line */}
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="cumulative"
            stroke="#3b82f6"
            strokeWidth={3}
            dot={false}
            name="Cumulative"
          />
          
          {/* Zero Reference Line */}
          <ReferenceLine 
            y={0} 
            stroke="#6b7280"
            strokeDasharray="3 3"
            yAxisId="left"
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Detailed Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Profit Days</p>
          <p className="text-lg font-semibold text-green-600">
            {stats.profitDays}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Loss Days</p>
          <p className="text-lg font-semibold text-red-600">
            {stats.lossDays}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Max Profit</p>
          <p className="text-lg font-semibold text-green-600">
            ${stats.maxProfit.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Max Loss</p>
          <p className="text-lg font-semibold text-red-600">
            ${Math.abs(stats.maxLoss).toFixed(2)}
          </p>
        </div>
      </div>
    </div>
  );
};

export default PnLVisualization;