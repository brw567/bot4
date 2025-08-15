import React, { useEffect, useState } from 'react';
import { ExclamationTriangleIcon, ShieldCheckIcon, BoltIcon } from '@heroicons/react/24/outline';
import { format } from 'date-fns';

interface CircuitBreakerState {
  name: string;
  type: string;
  state: 'open' | 'closed' | 'half_open';
  failure_count: number;
  last_failure_time: string | null;
  recovery_time: string | null;
  trigger_value: number | null;
  threshold_value: number | null;
  cooldown_seconds: number;
}

interface CircuitBreakerPattern {
  breaker_name: string;
  pattern_type: string;
  confidence: number;
  predicted_trigger_time: string | null;
  recommendation: string;
}

const CircuitBreakerStatus: React.FC = () => {
  const [breakers, setBreakers] = useState<CircuitBreakerState[]>([]);
  const [patterns, setPatterns] = useState<CircuitBreakerPattern[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchBreakerStatus = async () => {
      try {
        const token = localStorage.getItem('token');
        const headers = { 'Authorization': `Bearer ${token}` };

        // Fetch current breaker states
        const statesResponse = await fetch('/api/circuit-breakers/states', { headers });
        if (statesResponse.ok) {
          const data = await statesResponse.json();
          setBreakers(data.breakers || []);
        }

        // Fetch detected patterns
        const patternsResponse = await fetch('/api/circuit-breakers/patterns', { headers });
        if (patternsResponse.ok) {
          const data = await patternsResponse.json();
          setPatterns(data.patterns || []);
        }

        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch circuit breaker status:', err);
        setError('Failed to load circuit breaker status');
        setLoading(false);
      }
    };

    fetchBreakerStatus();
    const interval = setInterval(fetchBreakerStatus, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const getBreakerIcon = (state: string) => {
    switch (state) {
      case 'open':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      case 'half_open':
        return <BoltIcon className="h-5 w-5 text-yellow-500" />;
      case 'closed':
        return <ShieldCheckIcon className="h-5 w-5 text-green-500" />;
      default:
        return <ShieldCheckIcon className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStateColor = (state: string) => {
    switch (state) {
      case 'open':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'half_open':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'closed':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-red-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-gray-600';
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-600 dark:text-red-400 text-center p-4">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Circuit Breaker States */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Circuit Breakers
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {breakers.map((breaker) => (
            <div
              key={breaker.name}
              className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  {getBreakerIcon(breaker.state)}
                  <span className="ml-2 font-medium text-gray-900 dark:text-white">
                    {breaker.name}
                  </span>
                </div>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStateColor(breaker.state)}`}>
                  {breaker.state.toUpperCase()}
                </span>
              </div>
              
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Type:</span>
                  <span className="text-gray-900 dark:text-white">{breaker.type}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500 dark:text-gray-400">Failures:</span>
                  <span className="text-gray-900 dark:text-white">{breaker.failure_count}</span>
                </div>
                {breaker.trigger_value !== null && (
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Trigger:</span>
                    <span className="text-gray-900 dark:text-white">
                      {breaker.trigger_value.toFixed(2)} / {breaker.threshold_value?.toFixed(2)}
                    </span>
                  </div>
                )}
                {breaker.state === 'open' && breaker.recovery_time && (
                  <div className="flex justify-between">
                    <span className="text-gray-500 dark:text-gray-400">Recovery:</span>
                    <span className="text-gray-900 dark:text-white">
                      {format(new Date(breaker.recovery_time), 'HH:mm:ss')}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pattern Detection */}
      {patterns.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Detected Patterns & Predictions
          </h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-900">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Breaker
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Pattern
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Confidence
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Predicted Trigger
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Recommendation
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {patterns.map((pattern, index) => (
                    <tr key={index}>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                        {pattern.breaker_name}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {pattern.pattern_type}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <span className={`font-medium ${getConfidenceColor(pattern.confidence)}`}>
                          {(pattern.confidence * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {pattern.predicted_trigger_time 
                          ? format(new Date(pattern.predicted_trigger_time), 'HH:mm:ss')
                          : '-'}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                        {pattern.recommendation}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {breakers.filter(b => b.state === 'open').length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Open Breakers</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {breakers.reduce((sum, b) => sum + b.failure_count, 0)}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Total Failures</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {patterns.length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Patterns Detected</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {patterns.filter(p => p.confidence >= 0.8).length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">High Risk Alerts</div>
        </div>
      </div>
    </div>
  );
};

export default CircuitBreakerStatus;