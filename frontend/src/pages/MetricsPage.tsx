import React, { useEffect } from 'react';
import { useAppDispatch } from '../hooks/redux';
import { setMetrics, setLoading, setError } from '../store/slices/metricsSlice';
import MetricsGroup from '../components/MetricsGroup';
import apiService from '../services/api';

const MetricsPage: React.FC = () => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    // Load initial metrics
    const loadMetrics = async () => {
      dispatch(setLoading(true));
      try {
        const metrics = await apiService.getMetrics();
        dispatch(setMetrics(metrics));
      } catch (error) {
        dispatch(setError('Failed to load metrics'));
      } finally {
        dispatch(setLoading(false));
      }
    };

    loadMetrics();
  }, [dispatch]);

  return <MetricsGroup />;
};

export default MetricsPage;