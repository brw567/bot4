import React, { useState } from 'react';
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
  Button,
  Tooltip,
  IconButton,
  Alert
} from '@mui/material';
import { TrendingUp, Info, PlayArrow } from '@mui/icons-material';
import { formatCurrency } from '../../utils/formatters';

interface ArbitrageOpportunity {
  symbol: string;
  profit: string;
  buy: string;
  sell: string;
  volume: number;
}

interface ArbitrageOpportunitiesProps {
  opportunities: ArbitrageOpportunity[];
}

const ArbitrageOpportunities: React.FC<ArbitrageOpportunitiesProps> = ({ opportunities }) => {
  const [selectedOpp, setSelectedOpp] = useState<ArbitrageOpportunity | null>(null);

  const getProfitColor = (profit: string) => {
    const profitValue = parseFloat(profit.replace('%', ''));
    if (profitValue >= 1) return 'success';
    if (profitValue >= 0.5) return 'warning';
    return 'default';
  };

  const handleExecute = (opportunity: ArbitrageOpportunity) => {
    // In real implementation, this would trigger arbitrage execution
    console.log('Execute arbitrage:', opportunity);
    setSelectedOpp(opportunity);
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <TrendingUp sx={{ mr: 1 }} />
        <Typography variant="h6">
          Arbitrage Opportunities
        </Typography>
        <Chip
          label={`${opportunities.length} Active`}
          size="small"
          color="primary"
          sx={{ ml: 2 }}
        />
      </Box>

      {opportunities.length === 0 ? (
        <Alert severity="info">
          No arbitrage opportunities detected at the moment. 
          The system continuously monitors price discrepancies across exchanges.
        </Alert>
      ) : (
        <>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Buy From</TableCell>
                  <TableCell>Sell To</TableCell>
                  <TableCell align="right">Profit</TableCell>
                  <TableCell align="right">Max Volume</TableCell>
                  <TableCell align="center">Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {opportunities.map((opp, index) => (
                  <TableRow key={index} hover>
                    <TableCell>
                      <Typography variant="body2" fontWeight="medium">
                        {opp.symbol}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="primary">
                        {opp.buy}
                      </Typography>
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="secondary">
                        {opp.sell}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Chip
                        label={opp.profit}
                        size="small"
                        color={getProfitColor(opp.profit)}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2">
                        {opp.volume.toFixed(4)}
                      </Typography>
                    </TableCell>
                    <TableCell align="center">
                      <Tooltip title="Execute arbitrage (requires funds on both exchanges)">
                        <IconButton
                          size="small"
                          color="primary"
                          onClick={() => handleExecute(opp)}
                        >
                          <PlayArrow />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Info fontSize="small" sx={{ mr: 1 }} />
              <Typography variant="subtitle2">
                How Arbitrage Works
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              Arbitrage opportunities arise when the same asset has different prices on different exchanges. 
              The bot can buy on the cheaper exchange and simultaneously sell on the more expensive one, 
              capturing the price difference as profit. All opportunities shown are after accounting for trading fees.
            </Typography>
          </Box>

          {selectedOpp && (
            <Alert severity="info" sx={{ mt: 2 }}>
              Selected: {selectedOpp.symbol} - Buy from {selectedOpp.buy}, Sell to {selectedOpp.sell} for {selectedOpp.profit} profit
            </Alert>
          )}
        </>
      )}
    </Box>
  );
};

export default ArbitrageOpportunities;