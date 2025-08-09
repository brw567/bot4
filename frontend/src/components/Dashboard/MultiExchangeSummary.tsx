import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  Button
} from '@mui/material';
import {
  CompareArrows,
  TrendingUp,
  AccountBalance,
  Speed
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { formatCurrency } from '../../utils/formatters';

interface DashboardSummary {
  exchanges_connected: number;
  arbitrage_opportunities: number;
  price_disparities: Array<{
    symbol: string;
    min: number;
    max: number;
    disparity: string;
  }>;
}

const MultiExchangeSummary: React.FC = () => {
  const navigate = useNavigate();
  const { token } = useAuth();
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSummary();
  }, [token]);

  const fetchSummary = async () => {
    try {
      const response = await fetch('/api/dashboard/summary', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setSummary(data);
      } else {
        throw new Error('Failed to fetch summary');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </Card>
    );
  }

  if (!summary) return null;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6">
            Multi-Exchange Overview
          </Typography>
          <Button
            size="small"
            onClick={() => navigate('/multi-exchange')}
            endIcon={<CompareArrows />}
          >
            View Details
          </Button>
        </Box>

        <Grid container spacing={3}>
          {/* Connected Exchanges */}
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <AccountBalance sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4">
                {summary.exchanges_connected}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Connected Exchanges
              </Typography>
            </Box>
          </Grid>

          {/* Arbitrage Opportunities */}
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center' }}>
              <TrendingUp sx={{ fontSize: 40, color: 'success.main', mb: 1 }} />
              <Typography variant="h4">
                {summary.arbitrage_opportunities}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Arbitrage Opportunities
              </Typography>
            </Box>
          </Grid>

          {/* Price Disparities */}
          <Grid item xs={12} md={6}>
            <Box>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Speed sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="subtitle1">
                  Price Disparities
                </Typography>
              </Box>
              
              {summary.price_disparities.length > 0 ? (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {summary.price_disparities.slice(0, 3).map((item, index) => (
                    <Box
                      key={index}
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        p: 1,
                        bgcolor: 'background.default',
                        borderRadius: 1
                      }}
                    >
                      <Typography variant="body2" fontWeight="medium">
                        {item.symbol}
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          {formatCurrency(item.min)} - {formatCurrency(item.max)}
                        </Typography>
                        <Chip
                          label={item.disparity}
                          size="small"
                          color={parseFloat(item.disparity) > 1 ? 'warning' : 'default'}
                        />
                      </Box>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No significant price disparities detected
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>

        {/* Alert for arbitrage opportunities */}
        {summary.arbitrage_opportunities > 0 && (
          <Alert
            severity="info"
            action={
              <Button
                color="inherit"
                size="small"
                onClick={() => navigate('/multi-exchange')}
              >
                View
              </Button>
            }
            sx={{ mt: 2 }}
          >
            {summary.arbitrage_opportunities} profitable arbitrage {summary.arbitrage_opportunities === 1 ? 'opportunity' : 'opportunities'} detected across exchanges
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default MultiExchangeSummary;