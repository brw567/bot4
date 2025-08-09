import { configureStore } from '@reduxjs/toolkit';
import metricsReducer from './slices/metricsSlice';
import alertsReducer from './slices/alertsSlice';
import authReducer from './slices/authSlice';
import systemReducer from './slices/systemSlice';

export const store = configureStore({
  reducer: {
    metrics: metricsReducer,
    alerts: alertsReducer,
    auth: authReducer,
    system: systemReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;