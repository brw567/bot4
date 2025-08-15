import React, { useEffect, useState } from 'react';
import { BeakerIcon, ArrowUpIcon, ArrowDownIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';
import { format } from 'date-fns';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ModelVersion {
  model_name: string;
  version: string;
  state: 'development' | 'staging' | 'production' | 'deprecated';
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sharpe_ratio: number;
  max_drawdown: number;
  overfitting_gap: number;
  created_at: string;
  promoted_at: string | null;
  rollback_count: number;
}

interface ABTestResult {
  test_id: string;
  model_a: string;
  model_b: string;
  winner: string | null;
  confidence: number;
  sample_size: number;
  metrics_comparison: {
    accuracy_diff: number;
    sharpe_diff: number;
    drawdown_diff: number;
  };
  status: 'running' | 'completed' | 'failed';
}

const ModelVersioning: React.FC = () => {
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [abTests, setABTests] = useState<ABTestResult[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [performanceHistory, setPerformanceHistory] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelData = async () => {
      try {
        const token = localStorage.getItem('token');
        const headers = { 'Authorization': `Bearer ${token}` };

        // Fetch model versions
        const versionsResponse = await fetch('/api/models/versions', { headers });
        if (versionsResponse.ok) {
          const data = await versionsResponse.json();
          setModels(data.versions || []);
        }

        // Fetch A/B test results
        const abTestResponse = await fetch('/api/models/ab-tests', { headers });
        if (abTestResponse.ok) {
          const data = await abTestResponse.json();
          setABTests(data.tests || []);
        }

        // Fetch performance history for selected model
        if (selectedModel) {
          const historyResponse = await fetch(`/api/models/${selectedModel}/history`, { headers });
          if (historyResponse.ok) {
            const data = await historyResponse.json();
            setPerformanceHistory(data);
          }
        }

        setLoading(false);
      } catch (err) {
        console.error('Failed to fetch model data:', err);
        setError('Failed to load model versioning data');
        setLoading(false);
      }
    };

    fetchModelData();
    const interval = setInterval(fetchModelData, 10000); // Update every 10 seconds

    return () => clearInterval(interval);
  }, [selectedModel]);

  const getStateColor = (state: string) => {
    switch (state) {
      case 'production':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'staging':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
      case 'development':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'deprecated':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const getMetricColor = (value: number, threshold: number, inverted: boolean = false) => {
    const good = inverted ? value < threshold : value > threshold;
    return good ? 'text-green-600' : 'text-red-600';
  };

  const handlePromote = async (modelName: string, version: string, targetState: string) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/models/promote', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_name: modelName, version, target_state: targetState })
      });

      if (response.ok) {
        // Refresh data
        window.location.reload();
      }
    } catch (err) {
      console.error('Failed to promote model:', err);
    }
  };

  const handleRollback = async (modelName: string) => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/models/rollback', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ model_name: modelName })
      });

      if (response.ok) {
        // Refresh data
        window.location.reload();
      }
    } catch (err) {
      console.error('Failed to rollback model:', err);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-64 bg-gray-200 dark:bg-gray-700 rounded"></div>
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

  // Group models by name
  const modelGroups = models.reduce((acc, model) => {
    if (!acc[model.model_name]) {
      acc[model.model_name] = [];
    }
    acc[model.model_name].push(model);
    return acc;
  }, {} as Record<string, ModelVersion[]>);

  return (
    <div className="space-y-6">
      {/* Model Version Overview */}
      <div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Model Versions
        </h3>
        <div className="space-y-4">
          {Object.entries(modelGroups).map(([modelName, versions]) => {
            const productionVersion = versions.find(v => v.state === 'production');
            const stagingVersion = versions.find(v => v.state === 'staging');
            
            return (
              <div key={modelName} className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <BeakerIcon className="h-6 w-6 text-primary-600 mr-2" />
                    <h4 className="text-lg font-medium text-gray-900 dark:text-white">{modelName}</h4>
                  </div>
                  <button
                    onClick={() => setSelectedModel(selectedModel === modelName ? null : modelName)}
                    className="text-sm text-primary-600 hover:text-primary-700"
                  >
                    {selectedModel === modelName ? 'Hide Details' : 'Show Details'}
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {versions.slice(0, 3).map(version => (
                    <div key={version.version} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-gray-900 dark:text-white">
                          v{version.version}
                        </span>
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStateColor(version.state)}`}>
                          {version.state.toUpperCase()}
                        </span>
                      </div>
                      
                      <div className="space-y-1 text-sm">
                        <div className="flex justify-between">
                          <span className="text-gray-500 dark:text-gray-400">Accuracy:</span>
                          <span className={getMetricColor(version.accuracy, 0.7)}>
                            {(version.accuracy * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500 dark:text-gray-400">Sharpe:</span>
                          <span className={getMetricColor(version.sharpe_ratio, 1.0)}>
                            {version.sharpe_ratio.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500 dark:text-gray-400">Max DD:</span>
                          <span className={getMetricColor(version.max_drawdown, 0.15, true)}>
                            {(version.max_drawdown * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-500 dark:text-gray-400">Overfit Gap:</span>
                          <span className={getMetricColor(version.overfitting_gap, 0.05, true)}>
                            {(version.overfitting_gap * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      {version.state === 'staging' && (
                        <div className="mt-3 flex space-x-2">
                          <button
                            onClick={() => handlePromote(modelName, version.version, 'production')}
                            className="flex-1 px-2 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700"
                          >
                            Promote
                          </button>
                          <button
                            onClick={() => handleRollback(modelName)}
                            className="flex-1 px-2 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700"
                          >
                            Rollback
                          </button>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* A/B Test Results */}
      {abTests.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            A/B Test Results
          </h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                <thead className="bg-gray-50 dark:bg-gray-900">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Test ID
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Model A vs B
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Winner
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Confidence
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Metrics Δ
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                  {abTests.map((test) => (
                    <tr key={test.test_id}>
                      <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                        {test.test_id.slice(0, 8)}...
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {test.model_a} vs {test.model_b}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        {test.winner ? (
                          <span className="font-medium text-green-600">{test.winner}</span>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                        {(test.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-sm">
                        <div className="space-y-1">
                          <div className="flex items-center">
                            {test.metrics_comparison.accuracy_diff > 0 ? (
                              <ArrowUpIcon className="h-3 w-3 text-green-500 mr-1" />
                            ) : (
                              <ArrowDownIcon className="h-3 w-3 text-red-500 mr-1" />
                            )}
                            <span className="text-xs">Acc: {(test.metrics_comparison.accuracy_diff * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex items-center">
                            {test.metrics_comparison.sharpe_diff > 0 ? (
                              <ArrowUpIcon className="h-3 w-3 text-green-500 mr-1" />
                            ) : (
                              <ArrowDownIcon className="h-3 w-3 text-red-500 mr-1" />
                            )}
                            <span className="text-xs">Sharpe: {test.metrics_comparison.sharpe_diff.toFixed(2)}</span>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        {test.status === 'running' && (
                          <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                            Running
                          </span>
                        )}
                        {test.status === 'completed' && (
                          <span className="px-2 py-1 text-xs font-medium rounded-full bg-green-100 text-green-800">
                            Completed
                          </span>
                        )}
                        {test.status === 'failed' && (
                          <span className="px-2 py-1 text-xs font-medium rounded-full bg-red-100 text-red-800">
                            Failed
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Performance History Chart */}
      {selectedModel && performanceHistory && (
        <div>
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Performance History - {selectedModel}
          </h3>
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
            <Line
              data={{
                labels: performanceHistory.dates,
                datasets: [
                  {
                    label: 'Accuracy',
                    data: performanceHistory.accuracy,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                  },
                  {
                    label: 'Sharpe Ratio',
                    data: performanceHistory.sharpe_ratio,
                    borderColor: 'rgb(54, 162, 235)',
                    backgroundColor: 'rgba(54, 162, 235, 0.2)',
                  },
                ],
              }}
              options={{
                responsive: true,
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
                  },
                },
              }}
            />
          </div>
        </div>
      )}

      {/* Model Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {models.filter(m => m.state === 'production').length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Production Models</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {models.filter(m => m.state === 'staging').length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Staging Models</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {abTests.filter(t => t.status === 'running').length}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Active A/B Tests</div>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {models.reduce((sum, m) => sum + m.rollback_count, 0)}
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">Total Rollbacks</div>
        </div>
      </div>
    </div>
  );
};

export default ModelVersioning;