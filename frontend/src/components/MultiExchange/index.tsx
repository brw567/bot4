import React, { useState, useEffect } from 'react';
import { Grid, Paper, Typography, Box } from '@mui/material';
import { useTheme } from '@mui/material/styles';
import ExchangeOverview from './ExchangeOverview';
import ArbitrageOpportunities from './ArbitrageOpportunities';
import PriceComparison from './PriceComparison';
import CrossExchangeChart from './CrossExchangeChart';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useAuth } from '../../contexts/AuthContext';

interface MultiExchangeData {
  exchanges: Array<{
    id: string;
    name: string;
    status: string;
    test_mode: boolean;
    read_only: boolean;
  }>;
  arbitrage: Array<{
    symbol: string;
    profit: string;
    buy: string;
    sell: string;
    volume: number;
  }>;
  priceUpdates: Record<string, any>;
}

const MultiExchange: React.FC = () => {
  const theme = useTheme();
  const { token } = useAuth();
  const [data, setData] = useState<MultiExchangeData>({
    exchanges: [],
    arbitrage: [],
    priceUpdates: {}
  });

  // WebSocket connections for real-time updates
  const { data: multiExchangeData } = useWebSocket('/ws/multi-exchange', {
    enabled: !!token
  });

  const { data: arbitrageData } = useWebSocket('/ws/arbitrage', {
    enabled: !!token
  });

  // Fetch initial data
  useEffect(() => {
    fetchExchangeData();
  }, [token]);

  // Handle WebSocket updates
  useEffect(() => {
    if (multiExchangeData?.type === 'multi_exchange_update') {
      setData(prev => ({
        ...prev,
        priceUpdates: multiExchangeData.data
      }));
    }
  }, [multiExchangeData]);

  useEffect(() => {
    if (arbitrageData?.type === 'arbitrage_alert') {
      setData(prev => ({
        ...prev,
        arbitrage: arbitrageData.opportunities
      }));
    }
  }, [arbitrageData]);

  const fetchExchangeData = async () => {
    try {
      const response = await fetch('/api/exchanges', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const result = await response.json();
        setData(prev => ({
          ...prev,
          exchanges: result.exchanges
        }));
      }
    } catch (error) {
      console.error('Failed to fetch exchange data:', error);
    }
  };

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Multi-Exchange Analytics
      </Typography>
      
      <Grid container spacing={3}>
        {/* Exchange Overview */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <ExchangeOverview exchanges={data.exchanges} />
          </Paper>
        </Grid>

        {/* Price Comparison */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <PriceComparison priceData={data.priceUpdates} />
          </Paper>
        </Grid>

        {/* Cross-Exchange Chart */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 2, height: 400 }}>
            <CrossExchangeChart data={data.priceUpdates} />
          </Paper>
        </Grid>

        {/* Arbitrage Opportunities */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <ArbitrageOpportunities opportunities={data.arbitrage} />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default MultiExchange;