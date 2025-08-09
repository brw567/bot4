import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { SystemHealth } from '../../types';

interface SystemState {
  health: SystemHealth | null;
  botStatus: 'running' | 'stopped' | 'unknown';
  connected: boolean;
  lastUpdate: Date | null;
}

const initialState: SystemState = {
  health: null,
  botStatus: 'unknown',
  connected: false,
  lastUpdate: null,
};

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setHealth: (state, action: PayloadAction<SystemHealth>) => {
      state.health = action.payload;
      state.lastUpdate = new Date();
    },
    setBotStatus: (state, action: PayloadAction<'running' | 'stopped' | 'unknown'>) => {
      state.botStatus = action.payload;
    },
    setConnected: (state, action: PayloadAction<boolean>) => {
      state.connected = action.payload;
    },
  },
});

export const { setHealth, setBotStatus, setConnected } = systemSlice.actions;
export default systemSlice.reducer;