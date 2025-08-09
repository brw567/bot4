import React, { useMemo } from 'react';
import { Box, Typography } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { useTheme } from '@mui/material/styles';

interface CrossExchangeChartProps {
  data: Record<string, any>;
}

const CrossExchangeChart: React.FC<CrossExchangeChartProps> = ({ data }) => {
  const theme = useTheme();

  // Process real data from API
  const chartData = useMemo(() => {
    if (!data || !data.history) {
      return [];
    }
    
    // Transform API data into chart format
    return data.history.map((point: any) => {
      const time = new Date(point.timestamp);
      const timeStr = time.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
      
      // Extract prices for each exchange
      const chartPoint: any = { time: timeStr };
      if (data.exchanges) {
        data.exchanges.forEach((exchange: any) => {
          chartPoint[exchange.name.toLowerCase()] = point.prices?.[exchange.name] || exchange.price || 0;
        });
      }
      
      return chartPoint;
    });
  }, [data]);

  const exchanges = useMemo(() => {
    if (!data || !data.exchanges) {
      return [
        { key: 'binance', color: theme.palette.primary.main },
        { key: 'coinbase', color: theme.palette.secondary.main },
        { key: 'kraken', color: theme.palette.warning.main },
        { key: 'okx', color: theme.palette.info.main },
      ];
    }
    
    const colors = [
      theme.palette.primary.main,
      theme.palette.secondary.main,
      theme.palette.error.main,
      theme.palette.warning.main,
      theme.palette.success.main,
      theme.palette.info.main,
    ];
    
    return data.exchanges.map((exchange: any, index: number) => ({
      key: exchange.name.toLowerCase(),
      color: colors[index % colors.length],
    }));
  }, [data, theme]);

  const formatYAxis = (value: number) => {
    return `$${(value / 1000).toFixed(1)}k`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Box
          sx={{
            bgcolor: 'background.paper',
            p: 1.5,
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            boxShadow: 2
          }}
        >
          <Typography variant="caption" fontWeight="medium">
            {label}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.5 }}>
              <Box
                sx={{
                  width: 12,
                  height: 12,
                  bgcolor: entry.color,
                  borderRadius: '50%'
                }}
              />
              <Typography variant="caption">
                {entry.name}: ${entry.value.toFixed(2)}
              </Typography>
            </Box>
          ))}
          {payload.length > 1 && (
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              Spread: ${(Math.max(...payload.map((p: any) => p.value)) - 
                       Math.min(...payload.map((p: any) => p.value))).toFixed(2)}
            </Typography>
          )}
        </Box>
      );
    }
    return null;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        BTC/USDT Price Across Exchanges
      </Typography>
      
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
          <XAxis 
            dataKey="time" 
            stroke={theme.palette.text.secondary}
            style={{ fontSize: 12 }}
          />
          <YAxis 
            stroke={theme.palette.text.secondary}
            style={{ fontSize: 12 }}
            tickFormatter={formatYAxis}
            domain={['dataMin - 20', 'dataMax + 20']}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            wrapperStyle={{ fontSize: 12 }}
            iconType="line"
          />
          
          {exchanges.map((exchange) => (
            <Line
              key={exchange.key}
              type="monotone"
              dataKey={exchange.key}
              stroke={exchange.color}
              strokeWidth={2}
              dot={false}
              name={exchange.key.charAt(0).toUpperCase() + exchange.key.slice(1)}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        Real-time price tracking across multiple exchanges. Lines closer together indicate better market efficiency.
      </Typography>
    </Box>
  );
};

export default CrossExchangeChart;