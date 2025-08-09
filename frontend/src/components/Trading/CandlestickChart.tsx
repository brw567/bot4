import React, { useMemo, useState } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
} from 'recharts';
import { format } from 'date-fns';

interface CandleData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  sma20?: number;
  sma50?: number;
  rsi?: number;
}

interface TradeSignal {
  timestamp: string;
  type: 'buy' | 'sell';
  price: number;
  strategy: string;
}

interface CandlestickChartProps {
  data: CandleData[];
  signals?: TradeSignal[];
  height?: number;
  showIndicators?: boolean;
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  signals = [],
  height = 500,
  showIndicators = true,
}) => {
  const [brushDomain, setBrushDomain] = useState<[number, number] | null>(null);

  const formattedData = useMemo(() => {
    return data.map(candle => {
      // Check if there's a signal for this candle
      const signal = signals.find(s => s.timestamp === candle.timestamp);
      
      return {
        ...candle,
        time: format(new Date(candle.timestamp), 'HH:mm'),
        color: candle.close >= candle.open ? '#10b981' : '#ef4444',
        bodyHeight: Math.abs(candle.close - candle.open),
        bodyY: Math.max(candle.open, candle.close),
        wickHeight: candle.high - candle.low,
        wickY: candle.high,
        // Add signal data
        buySignal: signal?.type === 'buy' ? candle.low * 0.995 : null,
        sellSignal: signal?.type === 'sell' ? candle.high * 1.005 : null,
      };
    });
  }, [data, signals]);

  const BuyMarker = (props: any) => {
    const { cx, cy } = props;
    if (!cy || isNaN(cy)) return null;
    
    return (
      <g>
        <polygon
          points={`${cx},${cy-8} ${cx-6},${cy+4} ${cx+6},${cy+4}`}
          fill="#10b981"
          stroke="#10b981"
          strokeWidth="2"
        />
        <text
          x={cx}
          y={cy-12}
          fill="#10b981"
          fontSize="12"
          fontWeight="bold"
          textAnchor="middle"
        >
          BUY
        </text>
      </g>
    );
  };

  const SellMarker = (props: any) => {
    const { cx, cy } = props;
    if (!cy || isNaN(cy)) return null;
    
    return (
      <g>
        <polygon
          points={`${cx},${cy+8} ${cx-6},${cy-4} ${cx+6},${cy-4}`}
          fill="#ef4444"
          stroke="#ef4444"
          strokeWidth="2"
        />
        <text
          x={cx}
          y={cy+20}
          fill="#ef4444"
          fontSize="12"
          fontWeight="bold"
          textAnchor="middle"
        >
          SELL
        </text>
      </g>
    );
  };

  const CustomCandle = (props: any) => {
    const { x, y, width, height, payload } = props;
    const candleWidth = width * 0.8;
    const wickWidth = 2;
    const xCenter = x + width / 2;

    return (
      <g>
        {/* Wick */}
        <rect
          x={xCenter - wickWidth / 2}
          y={y}
          width={wickWidth}
          height={height}
          fill={payload.color}
        />
        {/* Body */}
        <rect
          x={x + (width - candleWidth) / 2}
          y={payload.bodyY}
          width={candleWidth}
          height={Math.max(payload.bodyHeight, 1)}
          fill={payload.color}
          stroke={payload.color}
        />
      </g>
    );
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const candle = payload[0].payload;
      return (
        <div className="bg-white dark:bg-gray-800 p-4 rounded shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-medium text-gray-900 dark:text-white mb-2">
            {label}
          </p>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Open:</span>
              <span className="font-medium">${candle.open.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">High:</span>
              <span className="font-medium">${candle.high.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Low:</span>
              <span className="font-medium">${candle.low.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Close:</span>
              <span className={`font-medium ${candle.color === '#10b981' ? 'text-green-600' : 'text-red-600'}`}>
                ${candle.close.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Volume:</span>
              <span className="font-medium">{(candle.volume / 1000).toFixed(2)}K</span>
            </div>
            {candle.rsi && (
              <div className="flex justify-between pt-2 border-t">
                <span className="text-gray-500">RSI:</span>
                <span className="font-medium">{candle.rsi.toFixed(1)}</span>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  const priceMin = Math.min(...data.map(d => d.low)) * 0.995;
  const priceMax = Math.max(...data.map(d => d.high)) * 1.005;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Price Chart
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Candlestick chart with technical indicators
        </p>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={formattedData} margin={{ top: 20, right: 30, bottom: 70, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
          <XAxis 
            dataKey="time" 
            className="text-gray-600 dark:text-gray-400"
            tick={{ fontSize: 11 }}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis 
            domain={[priceMin, priceMax]}
            className="text-gray-600 dark:text-gray-400"
            tick={{ fontSize: 11 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />

          {/* Candlesticks */}
          <Bar
            dataKey="wickHeight"
            shape={<CustomCandle />}
            isAnimationActive={false}
          />

          {/* Moving Averages */}
          {showIndicators && (
            <>
              <Line
                type="monotone"
                dataKey="sma20"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="SMA 20"
              />
              <Line
                type="monotone"
                dataKey="sma50"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                name="SMA 50"
              />
            </>
          )}

          {/* Buy Signals */}
          <Scatter
            dataKey="buySignal"
            fill="#10b981"
            shape={BuyMarker}
            name="BUY"
          />
          
          {/* Sell Signals */}
          <Scatter
            dataKey="sellSignal"
            fill="#ef4444"
            shape={SellMarker}
            name="SELL"
          />

          <Brush 
            dataKey="time" 
            height={30} 
            stroke="#8884d8"
            onChange={(domain: any) => setBrushDomain(domain)}
          />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Indicators Panel */}
      {showIndicators && data.length > 0 && (
        <div className="mt-6 grid grid-cols-4 gap-4">
          <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <p className="text-xs text-gray-500 dark:text-gray-400">Current Price</p>
            <p className={`text-lg font-semibold ${
              data[data.length - 1].close >= data[data.length - 1].open ? 'text-green-600' : 'text-red-600'
            }`}>
              ${data[data.length - 1].close.toFixed(2)}
            </p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <p className="text-xs text-gray-500 dark:text-gray-400">24h Change</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              {((data[data.length - 1].close - data[0].close) / data[0].close * 100).toFixed(2)}%
            </p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <p className="text-xs text-gray-500 dark:text-gray-400">24h Volume</p>
            <p className="text-lg font-semibold text-gray-900 dark:text-white">
              ${(data.reduce((sum, d) => sum + d.volume, 0) / 1000000).toFixed(2)}M
            </p>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded p-3">
            <p className="text-xs text-gray-500 dark:text-gray-400">RSI</p>
            <p className={`text-lg font-semibold ${
              data[data.length - 1].rsi && data[data.length - 1].rsi! > 70 ? 'text-red-600' :
              data[data.length - 1].rsi && data[data.length - 1].rsi! < 30 ? 'text-green-600' :
              'text-gray-900 dark:text-white'
            }`}>
              {data[data.length - 1].rsi?.toFixed(1) || 'N/A'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default CandlestickChart;