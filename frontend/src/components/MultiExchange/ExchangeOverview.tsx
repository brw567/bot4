import React from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Tooltip
} from '@mui/material';
import { CheckCircle, Error, Warning, Lock, FlashOn } from '@mui/icons-material';

interface Exchange {
  id: string;
  name: string;
  market_type?: string;
  status: string;
  test_mode: boolean;
  read_only: boolean;
}

interface ExchangeOverviewProps {
  exchanges: Exchange[];
}

const ExchangeOverview: React.FC<ExchangeOverviewProps> = ({ exchanges }) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected':
        return <CheckCircle color="success" fontSize="small" />;
      case 'disconnected':
        return <Error color="error" fontSize="small" />;
      default:
        return <Warning color="warning" fontSize="small" />;
    }
  };

  const getModeChip = (exchange: Exchange) => {
    if (exchange.test_mode) {
      return (
        <Chip
          label="TESTNET"
          size="small"
          color="warning"
          icon={<FlashOn />}
        />
      );
    } else if (exchange.read_only) {
      return (
        <Chip
          label="READ-ONLY"
          size="small"
          color="info"
          icon={<Lock />}
        />
      );
    } else {
      return (
        <Chip
          label="LIVE"
          size="small"
          color="success"
        />
      );
    }
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Connected Exchanges
      </Typography>
      
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Exchange</TableCell>
              <TableCell>Market Type</TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Mode</TableCell>
              <TableCell>Trading</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {exchanges.map((exchange) => (
              <TableRow key={exchange.id}>
                <TableCell>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getStatusIcon(exchange.status)}
                    {exchange.name}
                  </Box>
                </TableCell>
                <TableCell>
                  {exchange.market_type || 'spot'}
                </TableCell>
                <TableCell>
                  <Typography
                    variant="body2"
                    color={exchange.status === 'connected' ? 'success.main' : 'error.main'}
                  >
                    {exchange.status.toUpperCase()}
                  </Typography>
                </TableCell>
                <TableCell>
                  {getModeChip(exchange)}
                </TableCell>
                <TableCell>
                  <Tooltip title={
                    exchange.read_only 
                      ? "Only market data access allowed" 
                      : exchange.test_mode 
                        ? "Trading on testnet with test funds"
                        : "Live trading enabled"
                  }>
                    <Typography
                      variant="body2"
                      color={exchange.read_only ? 'text.secondary' : 'primary'}
                    >
                      {exchange.read_only ? 'Data Only' : 'Enabled'}
                    </Typography>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <Box sx={{ mt: 2, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        <Typography variant="caption" color="text.secondary">
          <CheckCircle fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
          Connected: Receiving real-time data
        </Typography>
        <Typography variant="caption" color="text.secondary">
          <Lock fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
          Read-Only: Market data access only
        </Typography>
        <Typography variant="caption" color="text.secondary">
          <FlashOn fontSize="small" sx={{ verticalAlign: 'middle', mr: 0.5 }} />
          TestNet: Using test credentials
        </Typography>
      </Box>
    </Box>
  );
};

export default ExchangeOverview;