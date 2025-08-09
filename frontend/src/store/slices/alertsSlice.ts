import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Alert } from '../../components/AlertCenter';

interface AlertState {
  alerts: Alert[];
  unreadCount: number;
}

const initialState: AlertState = {
  alerts: [
    {
      id: 'alert-1',
      type: 'critical',
      category: 'trading',
      title: 'Win Rate Below Target',
      message: 'Current win rate has dropped to 78.5%, below the 80% target threshold',
      timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
      read: false,
      actionable: true,
      actions: [
        { label: 'Pause Trading', action: 'pause_trading' },
        { label: 'Adjust Parameters', action: 'adjust_params' },
      ],
      metadata: {
        currentWinRate: 78.5,
        targetWinRate: 80,
        affectedPairs: ['BTC/USDT', 'ETH/USDT'],
      },
    },
    {
      id: 'alert-2',
      type: 'warning',
      category: 'system',
      title: 'High Memory Usage',
      message: 'Memory usage has exceeded 85% for the last 10 minutes',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      read: false,
      actionable: true,
      actions: [
        { label: 'Restart Service', action: 'restart_service' },
        { label: 'Clear Cache', action: 'clear_cache' },
      ],
      metadata: {
        memoryUsage: 85.3,
        duration: '10 minutes',
      },
    },
    {
      id: 'alert-3',
      type: 'info',
      category: 'trading',
      title: 'New Strategy Available',
      message: 'ML model has identified a new profitable strategy pattern',
      timestamp: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(),
      read: true,
      actionable: false,
    },
    {
      id: 'alert-4',
      type: 'success',
      category: 'performance',
      title: 'Performance Milestone',
      message: 'System has achieved 99.9% uptime for the past 30 days',
      timestamp: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      read: true,
      actionable: false,
    },
  ],
  unreadCount: 2,
};

const alertSlice = createSlice({
  name: 'alerts',
  initialState,
  reducers: {
    addAlert: (state, action: PayloadAction<Alert>) => {
      state.alerts.unshift(action.payload);
      if (!action.payload.read) {
        state.unreadCount++;
      }
    },
    markAsRead: (state, action: PayloadAction<string>) => {
      const alert = state.alerts.find(a => a.id === action.payload);
      if (alert && !alert.read) {
        alert.read = true;
        state.unreadCount = Math.max(0, state.unreadCount - 1);
      }
    },
    markAllAsRead: (state) => {
      state.alerts.forEach(alert => {
        alert.read = true;
      });
      state.unreadCount = 0;
    },
    deleteAlert: (state, action: PayloadAction<string>) => {
      const index = state.alerts.findIndex(a => a.id === action.payload);
      if (index !== -1) {
        const alert = state.alerts[index];
        if (!alert.read) {
          state.unreadCount = Math.max(0, state.unreadCount - 1);
        }
        state.alerts.splice(index, 1);
      }
    },
    updateAlerts: (state, action: PayloadAction<Alert[]>) => {
      state.alerts = action.payload;
      state.unreadCount = action.payload.filter(a => !a.read).length;
    },
  },
});

export const { addAlert, markAsRead, markAllAsRead, deleteAlert, updateAlerts } = alertSlice.actions;
export default alertSlice.reducer;