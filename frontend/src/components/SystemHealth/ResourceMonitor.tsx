import React, { useState } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import { CpuChipIcon, ServerStackIcon, CircleStackIcon, WifiIcon } from '@heroicons/react/24/outline';

interface ResourceMonitorProps {
  data: any[];
  currentHealth: any;
}

const ResourceMonitor: React.FC<ResourceMonitorProps> = ({ data, currentHealth }) => {
  const [selectedMetrics, setSelectedMetrics] = useState(['cpu', 'memory']);
  const [timeRange, setTimeRange] = useState('1h');

  const metrics = [
    { key: 'cpu', name: 'CPU', color: '#3B82F6', icon: CpuChipIcon },
    { key: 'memory', name: 'Memory', color: '#10B981', icon: ServerStackIcon },
    { key: 'disk', name: 'Disk', color: '#F59E0B', icon: CircleStackIcon },
    { key: 'network', name: 'Network', color: '#8B5CF6', icon: WifiIcon },
  ];

  const toggleMetric = (metric: string) => {
    setSelectedMetrics(prev =>
      prev.includes(metric)
        ? prev.filter(m => m !== metric)
        : [...prev, metric]
    );
  };

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Filter data based on time range
  const filteredData = data.slice(timeRange === '5m' ? -5 : timeRange === '15m' ? -15 : -60);

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {metrics.map(metric => (
            <button
              key={metric.key}
              onClick={() => toggleMetric(metric.key)}
              className={`flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                selectedMetrics.includes(metric.key)
                  ? 'bg-primary-100 text-primary-700 dark:bg-primary-900 dark:text-primary-300'
                  : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              <metric.icon className="h-4 w-4 mr-2" />
              {metric.name}
            </button>
          ))}
        </div>
        
        <div className="flex items-center space-x-2">
          {['5m', '15m', '1h'].map(range => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                timeRange === range
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Main Chart */}
      <div className="metric-card">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Resource Usage Over Time</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="timestamp"
              tick={{ fill: '#9CA3AF' }}
              tickFormatter={formatTime}
            />
            <YAxis tick={{ fill: '#9CA3AF' }} domain={[0, 100]} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '0.375rem',
              }}
              labelStyle={{ color: '#9CA3AF' }}
              formatter={(value: any) => `${(value || 0).toFixed(1)}%`}
              labelFormatter={(label) => new Date(label).toLocaleString()}
            />
            <Legend wrapperStyle={{ color: '#9CA3AF' }} />
            
            {/* Warning lines */}
            <ReferenceLine y={80} stroke="#EF4444" strokeDasharray="3 3" label="Warning" />
            <ReferenceLine y={90} stroke="#DC2626" strokeDasharray="3 3" label="Critical" />
            
            {metrics.map(metric => 
              selectedMetrics.includes(metric.key) && (
                <Line
                  key={metric.key}
                  type="monotone"
                  dataKey={metric.key}
                  stroke={metric.color}
                  strokeWidth={2}
                  dot={false}
                  name={metric.name}
                />
              )
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Current Usage */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Current Usage</h3>
          <div className="space-y-4">
            {metrics.map(metric => {
              const value = metric.key === 'cpu' ? currentHealth?.cpuUsage :
                           metric.key === 'memory' ? currentHealth?.memoryUsage :
                           metric.key === 'disk' ? currentHealth?.diskUsage || 70 :
                           data[data.length - 1]?.[metric.key] || 0;
              
              const isWarning = value > 80;
              const isCritical = value > 90;
              
              return (
                <div key={metric.key}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center">
                      <metric.icon className="h-5 w-5 mr-2 text-gray-400" />
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {metric.name}
                      </span>
                    </div>
                    <span className={`text-sm font-semibold ${
                      isCritical ? 'text-red-600' :
                      isWarning ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>
                      {(value || 0).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all ${
                        isCritical ? 'bg-red-600' :
                        isWarning ? 'bg-yellow-600' :
                        'bg-green-600'
                      }`}
                      style={{ width: `${Math.min(100, value)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Additional Metrics */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">System Metrics</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Active Threads</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {data[data.length - 1]?.threads || 0}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Connections</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {data[data.length - 1]?.connections || 0}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Load Average</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {((currentHealth?.cpuUsage || 0) / 25).toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500 dark:text-gray-400">Uptime</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                14d 3h
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Breakdown */}
      <div className="metric-card">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Process Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Process
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  CPU %
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Memory (MB)
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Threads
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              {[
                { name: 'scalping_bot.py', cpu: 15.2, memory: 128, threads: 12, status: 'running' },
                { name: 'ml_pipeline', cpu: 25.7, memory: 256, threads: 8, status: 'running' },
                { name: 'redis-server', cpu: 2.1, memory: 64, threads: 4, status: 'running' },
                { name: 'data_collector', cpu: 8.9, memory: 96, threads: 6, status: 'running' },
                { name: 'websocket_server', cpu: 5.3, memory: 32, threads: 10, status: 'running' },
              ].map((process) => (
                <tr key={process.name}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {process.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center">
                      <span className={process.cpu > 20 ? 'text-yellow-600 font-medium' : ''}>
                        {(process.cpu || 0).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {process.memory}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {process.threads}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      {process.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ResourceMonitor;