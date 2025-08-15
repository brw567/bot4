import React, { useEffect, useState } from 'react';
import { ClockIcon, CpuChipIcon, ServerStackIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';
import { format } from 'date-fns';
import { Bar, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface PerformanceStats {
  function_name: string;
  call_count: number;
  avg_time_ms: number;
  min_time_ms: number;
  max_time_ms: number;
  p50_time_ms: number;
  p95_time_ms: number;
  p99_time_ms: number;
  avg_memory_mb: number;
  max_memory_mb: number;
  error_count: number;
  last_called: string;
}

interface ExchangeLatency {
  exchange: string;
  operation: string;
  count: number;
  avg_ms: number;
  min_ms: number;
  max_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

interface MemorySnapshot {
  timestamp: string;
  total_mb: number;
  baseline_mb: number;
  delta_mb: number;
  top_allocations: Array<{
    file: string;
    size_mb: number;
    count: number;
  }>;
}

interface Anomaly {
  function: string;
  time: string;
  duration_ms: number;
  memory_delta_mb: number;
  exception: string | null;
}

const PerformanceProfiler: React.FC = () => {
  const [stats, setStats] = useState<Record<string, PerformanceStats>>({});
  const [slowestFunctions, setSlowestFunctions] = useState<PerformanceStats[]>([]);
  const [exchangeLatencies, setExchangeLatencies] = useState<Record<string, ExchangeLatency>>({});
  const [memorySnapshot, setMemorySnapshot] = useState<MemorySnapshot | null>(null);
  const [anomalies, setAnomalies] = useState<Anomaly[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFunction, setSelectedFunction] = useState<string | null>(null);

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        const token = localStorage.getItem('token');
        const headers = { 'Authorization': `Bearer ${token}` };

        // Fetch performance stats
        const statsResponse = await fetch('/api/performance/stats', { headers });
        if (statsResponse.ok) {
          const data = await statsResponse.json();
          setStats(data.stats || {});
        }

        // Fetch slowest functions
        const slowestResponse = await fetch('/api/performance/slowest?top_n=10', { headers });
        if (slowestResponse.ok) {
          const data = await slowestResponse.json();
          setSlowestFunctions(data.functions || []);
        }

        // Fetch exchange latencies
        const latencyResponse = await fetch('/api/performance/exchange-latency', { headers });
        if (latencyResponse.ok) {
          const data = await latencyResponse.json();
          setExchangeLatencies(data.latencies || {});
        }

        // Fetch memory snapshot
        const memoryResponse = await fetch('/api/performance/memory', { headers });
        if (memoryResponse.ok) {
          const data = await memoryResponse.json();
          setMemorySnapshot(data.snapshot);
        }

        // Fetch anomalies
        const anomaliesResponse = await fetch('/api/performance/anomalies?hours=1', { headers });
        if (anomaliesResponse.ok) {
          const data = await anomaliesResponse.json();
          setAnomalies(data.anomalies || []);
        }

        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch performance data:', err);
        setError('Failed to load performance data');
        setLoading(false);
      }
    };

    fetchPerformanceData();
    const interval = setInterval(fetchPerformanceData, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const getLatencyColor = (latency: number) => {
    if (latency < 50) return 'text-green-600';
    if (latency < 100) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getMemoryColor = (delta: number) => {
    if (Math.abs(delta) < 50) return 'text-green-600';
    if (Math.abs(delta) < 100) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-96 bg-gray-200 dark:bg-gray-700 rounded"></div>
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

  // Prepare chart data for slowest functions
  const slowestFunctionsChartData = {
    labels: slowestFunctions.slice(0, 5).map(f => 
      f.function_name.split('.').pop() || f.function_name
    ),
    datasets: [
      {
        label: 'Avg Time (ms)',
        data: slowestFunctions.slice(0, 5).map(f => f.avg_time_ms),
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
      {
        label: 'P95 Time (ms)',
        data: slowestFunctions.slice(0, 5).map(f => f.p95_time_ms),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
      },
      {
        label: 'P99 Time (ms)',
        data: slowestFunctions.slice(0, 5).map(f => f.p99_time_ms),
        backgroundColor: 'rgba(255, 206, 86, 0.5)',
      },
    ],
  };

  // Prepare exchange latency chart data
  const exchangeLatencyData = {
    labels: Object.keys(exchangeLatencies).slice(0, 10),
    datasets: [
      {
        label: 'Average Latency (ms)',
        data: Object.values(exchangeLatencies).slice(0, 10).map(l => l.avg_ms),
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
    ],
  };

  return (
    <div className="space-y-6">
      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Functions</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {Object.keys(stats).length}
              </p>
            </div>
            <CpuChipIcon className="h-8 w-8 text-primary-600" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Calls</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {Object.values(stats).reduce((sum, s) => sum + s.call_count, 0).toLocaleString()}
              </p>
            </div>
            <ClockIcon className="h-8 w-8 text-primary-600" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Memory Usage</p>
              <p className={`text-2xl font-semibold ${getMemoryColor(memorySnapshot?.delta_mb || 0)}`}>
                {memorySnapshot?.total_mb.toFixed(0)} MB
              </p>
            </div>
            <ServerStackIcon className="h-8 w-8 text-primary-600" />
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Anomalies</p>
              <p className={`text-2xl font-semibold ${anomalies.length > 0 ? 'text-red-600' : 'text-gray-900 dark:text-white'}`}>
                {anomalies.length}
              </p>
            </div>
            <ExclamationCircleIcon className={`h-8 w-8 ${anomalies.length > 0 ? 'text-red-600' : 'text-gray-400'}`} />
          </div>
        </div>
      </div>

      {/* Slowest Functions Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Slowest Functions
        </h3>
        <div className="h-64">
          <Bar
            data={slowestFunctionsChartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  position: 'top' as const,
                },
                title: {
                  display: false,
                },
              },
              scales: {
                y: {
                  beginAtZero: true,
                  title: {
                    display: true,
                    text: 'Time (ms)',
                  },
                },
              },
            }}
          />
        </div>
      </div>

      {/* Exchange Latencies */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Exchange Latencies
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="h-64">
            <Bar
              data={exchangeLatencyData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y' as const,
                plugins: {
                  legend: {
                    display: false,
                  },
                  title: {
                    display: false,
                  },
                },
                scales: {
                  x: {
                    beginAtZero: true,
                    title: {
                      display: true,
                      text: 'Latency (ms)',
                    },
                  },
                },
              }}
            />
          </div>
          <div className="space-y-2">
            {Object.entries(exchangeLatencies).slice(0, 5).map(([key, latency]) => (
              <div key={key} className="flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-900 rounded">
                <span className="text-sm font-medium text-gray-900 dark:text-white">{key}</span>
                <div className="text-right">
                  <div className={`text-sm font-medium ${getLatencyColor(latency.avg_ms)}`}>
                    {latency.avg_ms.toFixed(1)} ms
                  </div>
                  <div className="text-xs text-gray-500">
                    P95: {latency.p95_ms.toFixed(1)} ms
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Function Details Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Function Performance Details
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Function
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Calls
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Avg Time
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  P95 Time
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Memory
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Errors
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Last Called
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {slowestFunctions.map((func) => (
                <tr key={func.function_name} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                  <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {func.function_name.split('.').slice(-2).join('.')}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {func.call_count.toLocaleString()}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    <span className={getLatencyColor(func.avg_time_ms)}>
                      {func.avg_time_ms.toFixed(1)} ms
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    <span className={getLatencyColor(func.p95_time_ms)}>
                      {func.p95_time_ms.toFixed(1)} ms
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {func.avg_memory_mb.toFixed(1)} MB
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    {func.error_count > 0 ? (
                      <span className="text-red-600">{func.error_count}</span>
                    ) : (
                      <span className="text-gray-400">0</span>
                    )}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {format(new Date(func.last_called), 'HH:mm:ss')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Memory Analysis */}
      {memorySnapshot && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Memory Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {memorySnapshot.total_mb.toFixed(0)} MB
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">Current Usage</div>
            </div>
            <div className="text-center">
              <div className={`text-2xl font-bold ${getMemoryColor(memorySnapshot.delta_mb)}`}>
                {memorySnapshot.delta_mb > 0 ? '+' : ''}{memorySnapshot.delta_mb.toFixed(0)} MB
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">Delta from Baseline</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 dark:text-white">
                {memorySnapshot.baseline_mb.toFixed(0)} MB
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">Baseline</div>
            </div>
          </div>
          
          {memorySnapshot.top_allocations && memorySnapshot.top_allocations.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Top Memory Allocations
              </h4>
              <div className="space-y-1">
                {memorySnapshot.top_allocations.slice(0, 5).map((alloc, index) => (
                  <div key={index} className="flex justify-between items-center text-sm">
                    <span className="text-gray-600 dark:text-gray-400 truncate flex-1 mr-2">
                      {alloc.file}
                    </span>
                    <span className="font-medium text-gray-900 dark:text-white">
                      {alloc.size_mb.toFixed(2)} MB
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Anomalies */}
      {anomalies.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Recent Anomalies
          </h3>
          <div className="space-y-2">
            {anomalies.map((anomaly, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                <div className="flex items-center">
                  <ExclamationCircleIcon className="h-5 w-5 text-red-600 mr-3" />
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      {anomaly.function}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                      {format(new Date(anomaly.time), 'HH:mm:ss')} - Duration: {anomaly.duration_ms.toFixed(0)}ms
                      {anomaly.memory_delta_mb && `, Memory: ${anomaly.memory_delta_mb.toFixed(1)}MB`}
                    </div>
                    {anomaly.exception && (
                      <div className="text-sm text-red-600 dark:text-red-400 mt-1">
                        Error: {anomaly.exception}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceProfiler;