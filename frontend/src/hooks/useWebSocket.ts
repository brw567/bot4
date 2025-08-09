import { useEffect, useCallback } from 'react';
import { useAppDispatch } from './redux';
import wsService from '../services/websocket';
import { 
  updateMetric, 
  addChange, 
  setPerformance 
} from '../store/slices/metricsSlice';
import { addAlert } from '../store/slices/alertsSlice';
import { setHealth, setConnected } from '../store/slices/systemSlice';
import { MetricData, ChangeEvent, Alert } from '../types';
import toast from 'react-hot-toast';

export const useWebSocket = () => {
  const dispatch = useAppDispatch();

  const handleMetricsUpdate = useCallback((data: MetricData[]) => {
    data.forEach(metric => {
      dispatch(updateMetric(metric));
    });
  }, [dispatch]);

  const handleChanges = useCallback((changes: ChangeEvent[]) => {
    changes.forEach(change => {
      dispatch(addChange(change));
      
      // Show toast for critical changes
      if (change.severity === 'critical') {
        toast.error(`Critical change in ${change.pair}: ${change.metric} changed by ${change.changePercent.toFixed(1)}%`);
      }
    });
  }, [dispatch]);

  const handlePerformance = useCallback((data: { winRate: number; totalPnl: number }) => {
    dispatch(setPerformance(data));
  }, [dispatch]);

  const handleAlert = useCallback((alert: Alert) => {
    dispatch(addAlert(alert));
    
    // Show toast based on severity
    const message = alert.pair ? `${alert.pair}: ${alert.message}` : alert.message;
    switch (alert.severity) {
      case 'critical':
        toast.error(message);
        break;
      case 'error':
        toast.error(message);
        break;
      case 'warning':
        toast(message, { icon: '⚠️' });
        break;
      case 'info':
        toast(message, { icon: 'ℹ️' });
        break;
    }
  }, [dispatch]);

  const handleHealth = useCallback((health: any) => {
    dispatch(setHealth(health));
  }, [dispatch]);

  const handleConnection = useCallback((connected: boolean) => {
    dispatch(setConnected(connected));
    if (connected) {
      toast.success('Connected to trading system');
    } else {
      toast.error('Disconnected from trading system');
    }
  }, [dispatch]);

  useEffect(() => {
    // Subscribe to WebSocket events
    wsService.on('metrics', handleMetricsUpdate);
    wsService.on('changes', handleChanges);
    wsService.on('performance', handlePerformance);
    wsService.on('alert', handleAlert);
    wsService.on('health', handleHealth);
    wsService.on('connected', handleConnection);

    // Connect native WebSocket endpoints
    const metricsWs = wsService.connectMetrics();
    const tradesWs = wsService.connectTrades();
    const alertsWs = wsService.connectAlerts();
    const systemWs = wsService.connectSystem();

    // Cleanup
    return () => {
      wsService.off('metrics', handleMetricsUpdate);
      wsService.off('changes', handleChanges);
      wsService.off('performance', handlePerformance);
      wsService.off('alert', handleAlert);
      wsService.off('health', handleHealth);
      wsService.off('connected', handleConnection);

      metricsWs.close();
      tradesWs.close();
      alertsWs.close();
      systemWs.close();
    };
  }, [
    handleMetricsUpdate,
    handleChanges,
    handlePerformance,
    handleAlert,
    handleHealth,
    handleConnection
  ]);
};