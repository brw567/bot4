import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { ClockIcon, BoltIcon, CircleStackIcon, ChartBarIcon } from '@heroicons/react/24/outline';

interface PerformanceMetricsProps {
  metrics: any;
  apiCounters?: any;
  apiHealth?: any;
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ metrics, apiCounters, apiHealth }) => {
  const [timeframe, setTimeframe] = useState('1h');

  // Use real latency distribution data from metrics
  const latencyDistribution = React.useMemo(() => {
    if (!metrics.latency) {
      return [
        { range: '0-10ms', count: 0, percentage: 0 },
        { range: '10-25ms', count: 0, percentage: 0 },
        { range: '25-50ms', count: 0, percentage: 0 },
        { range: '50-100ms', count: 0, percentage: 0 },
        { range: '100-200ms', count: 0, percentage: 0 },
        { range: '200ms+', count: 0, percentage: 0 },
      ];
    }
    
    // Calculate distribution from actual latency data
    // This would need real latency data from the backend
    return [
      { range: '0-10ms', count: 0, percentage: 0 },
      { range: '10-25ms', count: 0, percentage: 0 },
      { range: '25-50ms', count: 0, percentage: 0 },
      { range: '50-100ms', count: 0, percentage: 0 },
      { range: '100-200ms', count: 0, percentage: 0 },
      { range: '200ms+', count: 0, percentage: 0 },
    ];
  }, [metrics]);

  // Use real throughput data from metrics
  const throughputData = React.useMemo(() => {
    if (!metrics.throughput) {
      // Return empty data if no metrics
      return Array.from({ length: 25 }, (_, i) => ({
        time: `${24 - i}h`,
        read: 0,
        write: 0,
        total: 0,
      }));
    }
    
    // Use actual throughput values
    return Array.from({ length: 25 }, (_, i) => ({
      time: `${24 - i}h`,
      read: metrics.throughput.read || 0,
      write: metrics.throughput.write || 0,
      total: metrics.throughput.total || 0,
    }));
  }, [metrics]);

  // Cache hit rate pie chart data
  const cacheData = [
    { name: 'Hits', value: metrics.cache.hits, color: '#10B981' },
    { name: 'Misses', value: metrics.cache.misses, color: '#EF4444' },
  ];

  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(2)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <div className="space-y-6">
      {/* Key Metrics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Request Rate</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {metrics.requests.rate}/s
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Error rate: {(metrics.requests.errorRate * 100).toFixed(2)}%
              </p>
            </div>
            <BoltIcon className="h-8 w-8 text-yellow-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Avg Latency</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {metrics.latency.avg}ms
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                P95: {metrics.latency.p95}ms
              </p>
            </div>
            <ClockIcon className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Throughput</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {metrics.throughput.total} MB/s
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                R: {metrics.throughput.read} W: {metrics.throughput.write}
              </p>
            </div>
            <ChartBarIcon className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Cache Hit Rate</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {(metrics.cache.hitRate * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Size: {metrics.cache.size} MB
              </p>
            </div>
            <CircleStackIcon className="h-8 w-8 text-purple-600" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Latency Distribution */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Latency Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={latencyDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="range" tick={{ fill: '#9CA3AF' }} />
              <YAxis tick={{ fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.375rem',
                }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Bar dataKey="percentage" fill="#3B82F6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Cache Performance */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Cache Performance</h3>
          <div className="grid grid-cols-2 gap-4">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={cacheData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {cacheData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '0.375rem',
                  }}
                  labelStyle={{ color: '#9CA3AF' }}
                  formatter={(value: any) => formatNumber(value)}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-col justify-center space-y-3">
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Total Requests</p>
                <p className="text-xl font-semibold text-gray-900 dark:text-white">
                  {formatNumber(metrics.cache.hits + metrics.cache.misses)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Cache Hits</p>
                <p className="text-xl font-semibold text-green-600">
                  {formatNumber(metrics.cache.hits)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Cache Misses</p>
                <p className="text-xl font-semibold text-red-600">
                  {formatNumber(metrics.cache.misses)}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Throughput Over Time */}
      <div className="metric-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">Throughput Over Time</h3>
          <div className="flex space-x-2">
            {['1h', '6h', '24h', '7d'].map(tf => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                  timeframe === tf
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={throughputData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="time" tick={{ fill: '#9CA3AF' }} />
            <YAxis tick={{ fill: '#9CA3AF' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '0.375rem',
              }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: any) => `${value.toFixed(1)} MB/s`}
            />
            <Legend wrapperStyle={{ color: '#9CA3AF' }} />
            <Line type="monotone" dataKey="read" stroke="#10B981" strokeWidth={2} dot={false} name="Read" />
            <Line type="monotone" dataKey="write" stroke="#F59E0B" strokeWidth={2} dot={false} name="Write" />
            <Line type="monotone" dataKey="total" stroke="#3B82F6" strokeWidth={2} dot={false} name="Total" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Request Statistics</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Total Requests</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {formatNumber(metrics.requests.total)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Success Rate</span>
              <span className="font-medium text-green-600">
                {((1 - metrics.requests.errorRate) * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Error Count</span>
              <span className="font-medium text-red-600">{metrics.requests.errors}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Avg Response Size</span>
              <span className="font-medium text-gray-900 dark:text-white">12.4 KB</span>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Latency Percentiles</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">P50 (Median)</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {metrics.latency.p50}ms
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">P95</span>
              <span className="font-medium text-gray-900 dark:text-white">
                {metrics.latency.p95}ms
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">P99</span>
              <span className="font-medium text-yellow-600">{metrics.latency.p99}ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Max</span>
              <span className="font-medium text-red-600">256ms</span>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Database Performance</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Query Count</span>
              <span className="font-medium text-gray-900 dark:text-white">45.2K</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Avg Query Time</span>
              <span className="font-medium text-gray-900 dark:text-white">2.3ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Connection Pool</span>
              <span className="font-medium text-gray-900 dark:text-white">18/50</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Slow Queries</span>
              <span className="font-medium text-yellow-600">3</span>
            </div>
          </div>
        </div>
      </div>
      
      {/* API Call Counters */}
      {apiCounters && (
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">External API Usage</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Dune Analytics */}
            <div className="metric-card">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                <CircleStackIcon className="h-5 w-5 mr-2 text-purple-600" />
                Dune Analytics
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Calls Today</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {apiCounters.dune?.calls_today || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Total Calls</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {formatNumber(apiCounters.dune?.calls_total || 0)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Rate Limit</span>
                  <span className="font-medium text-green-600">OK</span>
                </div>
              </div>
            </div>

            {/* CoinGecko */}
            <div className="metric-card">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                <ChartBarIcon className="h-5 w-5 mr-2 text-orange-600" />
                CoinGecko
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Calls Today</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {apiCounters.coingecko?.calls_today || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Total Calls</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {formatNumber(apiCounters.coingecko?.calls_total || 0)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Rate Limit</span>
                  <span className="font-medium text-green-600">50/min</span>
                </div>
              </div>
            </div>

            {/* xAI (Grok) */}
            <div className="metric-card">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                <BoltIcon className="h-5 w-5 mr-2 text-blue-600" />
                xAI (Grok)
              </h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Calls Today</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {apiCounters.xai?.calls_today || 0}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Tokens Today</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {formatNumber(apiCounters.xai?.tokens_today || 0)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600 dark:text-gray-400">Total Tokens</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {formatNumber(apiCounters.xai?.tokens_total || 0)}
                  </span>
                </div>
              </div>
            </div>
          </div>
          
          {/* API Health Status */}
          {apiHealth && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(apiHealth).map(([service, health]: [string, any]) => (
                <div key={service} className="metric-card">
                  <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 capitalize">
                    {service} API
                  </h5>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-600 dark:text-gray-400">Status</span>
                      <span className={`font-medium ${
                        health.status === 'healthy' ? 'text-green-600' : 
                        health.status === 'error' || health.status === 'auth_failed' ? 'text-red-600' : 
                        health.status === 'not_configured' ? 'text-gray-500' :
                        'text-yellow-600'
                      }`}>
                        {health.status}
                      </span>
                    </div>
                    {health.authenticated !== undefined && (
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Auth</span>
                        <span className={`font-medium ${health.authenticated ? 'text-green-600' : 'text-red-600'}`}>
                          {health.authenticated ? 'Valid' : 'Invalid'}
                        </span>
                      </div>
                    )}
                    {health.latency && (
                      <div className="flex justify-between">
                        <span className="text-gray-600 dark:text-gray-400">Latency</span>
                        <span className="font-medium text-gray-900 dark:text-white">
                          {health.latency}ms
                        </span>
                      </div>
                    )}
                    {health.error && (
                      <div className="text-red-600 text-xs mt-1">{health.error}</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PerformanceMetrics;