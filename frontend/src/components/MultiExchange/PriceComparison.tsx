import React, { useMemo } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Tooltip,
  Chip
} from '@mui/material';
import { TrendingUp, TrendingDown } from '@mui/icons-material';
import { formatCurrency } from '../../utils/formatters';

interface PriceData {
  price_range: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
  best_exchange: string;
  volume_leader: string;
  arbitrage: boolean;
}

interface PriceComparisonProps {
  priceData: Record<string, PriceData>;
}

const PriceComparison: React.FC<PriceComparisonProps> = ({ priceData }) => {
  const sortedSymbols = useMemo(() => {
    return Object.entries(priceData)
      .sort(([, a], [, b]) => b.price_range.mean - a.price_range.mean)
      .slice(0, 10); // Top 10 symbols
  }, [priceData]);

  const getPriceSpreadPercentage = (data: PriceData) => {
    const spread = data.price_range.max - data.price_range.min;
    return (spread / data.price_range.min * 100).toFixed(2);
  };

  const getPriceBarPosition = (data: PriceData) => {
    const range = data.price_range.max - data.price_range.min;
    if (range === 0) return 50;
    return ((data.price_range.mean - data.price_range.min) / range) * 100;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Cross-Exchange Price Comparison
      </Typography>

      <TableContainer sx={{ maxHeight: 320 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>Symbol</TableCell>
              <TableCell align="right">Avg Price</TableCell>
              <TableCell>Price Range</TableCell>
              <TableCell align="right">Spread</TableCell>
              <TableCell>Best Price</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedSymbols.map(([symbol, data]) => (
              <TableRow key={symbol} hover>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    {data.arbitrage ? (
                      <TrendingUp color="success" fontSize="small" />
                    ) : (
                      <TrendingDown color="action" fontSize="small" />
                    )}
                    <Typography variant="body2" fontWeight="medium">
                      {symbol}
                    </Typography>
                  </Box>
                </TableCell>
                <TableCell align="right">
                  <Typography variant="body2">
                    {formatCurrency(data.price_range.mean)}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Box sx={{ width: 120 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                      <Typography variant="caption">
                        {formatCurrency(data.price_range.min)}
                      </Typography>
                      <Typography variant="caption">
                        {formatCurrency(data.price_range.max)}
                      </Typography>
                    </Box>
                    <Box sx={{ position: 'relative', height: 4, bgcolor: 'grey.300', borderRadius: 2 }}>
                      <Box
                        sx={{
                          position: 'absolute',
                          left: `${getPriceBarPosition(data)}%`,
                          transform: 'translateX(-50%)',
                          width: 8,
                          height: 8,
                          bgcolor: 'primary.main',
                          borderRadius: '50%',
                          top: -2
                        }}
                      />
                    </Box>
                  </Box>
                </TableCell>
                <TableCell align="right">
                  <Tooltip title="Price difference between exchanges">
                    <Chip
                      label={`${getPriceSpreadPercentage(data)}%`}
                      size="small"
                      color={parseFloat(getPriceSpreadPercentage(data)) > 0.5 ? 'warning' : 'default'}
                    />
                  </Tooltip>
                </TableCell>
                <TableCell>
                  <Typography variant="caption" color="text.secondary">
                    {data.best_exchange}
                  </Typography>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="caption" color="text.secondary">
          <TrendingUp fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5, color: 'success.main' }} />
          Has arbitrage opportunity
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Spread % = (Max - Min) / Min × 100
        </Typography>
      </Box>
    </Box>
  );
};

export default PriceComparison;