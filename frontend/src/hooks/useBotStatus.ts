import { useEffect } from 'react';
import { useAppDispatch } from './redux';
import { setBotStatus, setConnected } from '../store/slices/systemSlice';
import { updateMetric } from '../store/slices/metricsSlice';

export const useBotStatus = () => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    const fetchBotStatus = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/system/bot/status', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (response.ok) {
          const data = await response.json();
          
          // Update bot status
          dispatch(setBotStatus(data.status || 'unknown'));
          
          // Update connection status based on bot status
          dispatch(setConnected(data.status === 'running'));
          
          // Update active pairs metric
          if (data.active_pairs) {
            dispatch(updateMetric({
              pair: 'TOTAL',
              price: 0,
              volume: 0,
              change: 0,
              winRate: 0,
              totalTrades: 0,
              profitableTrades: 0,
              regime: 'unknown',
              fundingRate: 0,
              timestamp: new Date().toISOString(),
              activePairs: data.active_pairs
            }));
          }
        }
      } catch (error) {
        console.error('Failed to fetch bot status:', error);
        dispatch(setBotStatus('unknown'));
        dispatch(setConnected(false));
      }
    };

    // Fetch immediately
    fetchBotStatus();

    // Then fetch every 5 seconds
    const interval = setInterval(fetchBotStatus, 5000);

    return () => clearInterval(interval);
  }, [dispatch]);
};