import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { MetricData, ChangeEvent, StrategyPerformance } from '../../types';

interface MetricsState {
  metrics: MetricData[];
  changes: ChangeEvent[];
  strategies: StrategyPerformance[];
  winRate: number;
  totalPnl: number;
  activePairs: number;
  loading: boolean;
  error: string | null;
}

const initialState: MetricsState = {
  metrics: [],
  changes: [],
  strategies: [],
  winRate: 0,
  totalPnl: 0,
  activePairs: 0,
  loading: false,
  error: null,
};

const metricsSlice = createSlice({
  name: 'metrics',
  initialState,
  reducers: {
    setMetrics: (state, action: PayloadAction<MetricData[]>) => {
      state.metrics = action.payload;
      state.activePairs = action.payload.length;
    },
    updateMetric: (state, action: PayloadAction<MetricData>) => {
      const index = state.metrics.findIndex(m => m.pair === action.payload.pair);
      if (index !== -1) {
        state.metrics[index] = action.payload;
      } else {
        state.metrics.push(action.payload);
      }
    },
    addChange: (state, action: PayloadAction<ChangeEvent>) => {
      state.changes.unshift(action.payload);
      // Keep only last 100 changes
      if (state.changes.length > 100) {
        state.changes = state.changes.slice(0, 100);
      }
    },
    setStrategies: (state, action: PayloadAction<StrategyPerformance[]>) => {
      state.strategies = action.payload;
    },
    setPerformance: (state, action: PayloadAction<{ winRate: number; totalPnl: number }>) => {
      state.winRate = action.payload.winRate;
      state.totalPnl = action.payload.totalPnl;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const {
  setMetrics,
  updateMetric,
  addChange,
  setStrategies,
  setPerformance,
  setLoading,
  setError,
} = metricsSlice.actions;

export default metricsSlice.reducer;