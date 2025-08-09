import { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from './redux';
import { setMetrics, setLoading, setError, updateMetric } from '../store/slices/metricsSlice';
import apiService from '../services/api';

export const useMetricsPolling = (interval: number = 5000) => {
  const dispatch = useAppDispatch();
  const isAuthenticated = useAppSelector(state => state.auth.isAuthenticated);

  useEffect(() => {
    if (!isAuthenticated) return;

    const fetchMetrics = async () => {
      try {
        const metrics = await apiService.getMetrics();
        if (metrics && metrics.length > 0) {
          dispatch(setMetrics(metrics));
        }
      } catch (error: any) {
        // Only log error, don't set error state to avoid UI disruption
        console.error('Failed to fetch metrics:', error);
        
        // If 401, user needs to re-login
        if (error.response?.status === 401) {
          dispatch(setError('Session expired. Please login again.'));
        }
      }
    };

    // Initial fetch
    fetchMetrics();

    // Set up polling
    const pollInterval = setInterval(fetchMetrics, interval);

    return () => clearInterval(pollInterval);
  }, [dispatch, interval, isAuthenticated]);
};

export default useMetricsPolling;