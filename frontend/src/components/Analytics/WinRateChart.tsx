import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { format } from 'date-fns';

interface WinRateData {
  timestamp: string;
  winRate: number;
  target: number;
}

interface WinRateChartProps {
  data: WinRateData[];
  timeframe: string;
  showTarget?: boolean;
  height?: number;
}

const WinRateChart: React.FC<WinRateChartProps> = ({ 
  data, 
  timeframe, 
  showTarget = true,
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
      };
    });
  }, [data, timeframe]);

  const avgWinRate = useMemo(() => {
    if (data.length === 0) return 0;
    return data.reduce((sum, item) => sum + item.winRate, 0) / data.length;
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {label}
          </p>
          <p className="text-sm text-green-600 dark:text-green-400">
            Win Rate: {payload[0].value.toFixed(2)}%
          </p>
          {showTarget && payload[1] && (
            <p className="text-sm text-blue-600 dark:text-blue-400">
              Target: {payload[1].value.toFixed(2)}%
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Win Rate Trend
        </h3>
        <div className="flex items-center justify-between mt-1">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {timeframe} timeframe
          </p>
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              Average: 
              <span className={`ml-1 font-medium ${
                avgWinRate >= 80 ? 'text-green-600' : 'text-yellow-600'
              }`}>
                {avgWinRate.toFixed(2)}%
              </span>
            </span>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={formattedData}>
          <defs>
            <linearGradient id="colorWinRate" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.1}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
          <XAxis 
            dataKey="time" 
            className="text-gray-600 dark:text-gray-400"
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            domain={[70, 95]}
            className="text-gray-600 dark:text-gray-400"
            tick={{ fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          
          {/* Win Rate Area */}
          <Area
            type="monotone"
            dataKey="winRate"
            stroke="#10b981"
            fillOpacity={1}
            fill="url(#colorWinRate)"
            strokeWidth={2}
            name="Win Rate"
          />
          
          {/* Target Line */}
          {showTarget && (
            <Line
              type="monotone"
              dataKey="target"
              stroke="#3b82f6"
              strokeDasharray="5 5"
              dot={false}
              name="Target"
            />
          )}
          
          {/* Reference Lines */}
          <ReferenceLine 
            y={80} 
            stroke="#ef4444"
            strokeDasharray="3 3"
            label={{ value: "Min Target", position: "right" }}
          />
          <ReferenceLine 
            y={avgWinRate} 
            stroke="#f59e0b"
            strokeDasharray="3 3"
            label={{ value: "Average", position: "right" }}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Stats Summary */}
      <div className="grid grid-cols-4 gap-4 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Current</p>
          <p className={`text-lg font-semibold ${
            data[data.length - 1]?.winRate >= 80 ? 'text-green-600' : 'text-yellow-600'
          }`}>
            {data[data.length - 1]?.winRate.toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">High</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {Math.max(...data.map(d => d.winRate)).toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Low</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {Math.min(...data.map(d => d.winRate)).toFixed(2)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Volatility</p>
          <p className="text-lg font-semibold text-gray-900 dark:text-white">
            {(Math.max(...data.map(d => d.winRate)) - Math.min(...data.map(d => d.winRate))).toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
};

export default WinRateChart;