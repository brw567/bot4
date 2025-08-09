import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface AuthState {
  isAuthenticated: boolean;
  username: string | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

const initialState: AuthState = {
  isAuthenticated: !!localStorage.getItem('token'),
  username: null,
  token: localStorage.getItem('token'),
  loading: false,
  error: null,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    loginSuccess: (state, action: PayloadAction<{ username: string; token: string }>) => {
      state.isAuthenticated = true;
      state.username = action.payload.username;
      state.token = action.payload.token;
      state.loading = false;
      state.error = null;
    },
    loginFailure: (state, action: PayloadAction<string>) => {
      state.isAuthenticated = false;
      state.username = null;
      state.token = null;
      state.loading = false;
      state.error = action.payload;
    },
    logout: (state) => {
      state.isAuthenticated = false;
      state.username = null;
      state.token = null;
      state.loading = false;
      state.error = null;
      localStorage.removeItem('token');
    },
  },
});

export const { loginStart, loginSuccess, loginFailure, logout } = authSlice.actions;

// Map old action names to new ones for backward compatibility
export const setAuth = loginSuccess;
export const clearAuth = logout;

export default authSlice.reducer;
export type { AuthState };
