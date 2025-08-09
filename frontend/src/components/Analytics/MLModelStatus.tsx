import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { CheckCircleIcon, XCircleIcon, ClockIcon } from '@heroicons/react/24/outline';
import { format, isValid } from 'date-fns';

interface MLMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  auc: number;
  lastTraining: string;
  nextTraining: string;
  modelVersion: string;
  dataPoints: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

interface ModelPerformanceData {
  timestamp: string;
  accuracy: number;
  loss: number;
}

interface MLModelStatusProps {
  metrics: MLMetrics;
  featureImportance: FeatureImportance[];
  performanceHistory: ModelPerformanceData[];
}

const MLModelStatus: React.FC<MLModelStatusProps> = ({
  metrics,
  featureImportance,
  performanceHistory,
}) => {
  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  // Helper function to safely format dates
  const safeFormatDate = (dateValue: string | number | Date, formatString: string, fallback: string = 'Invalid Date'): string => {
    if (!dateValue) return fallback;
    
    const date = new Date(dateValue);
    if (!isValid(date)) {
      console.warn('Invalid date value:', dateValue);
      return fallback;
    }
    
    try {
      return format(date, formatString);
    } catch (error) {
      console.error('Date formatting error:', error, 'for value:', dateValue);
      return fallback;
    }
  };

  const pieData = featureImportance.map((item, index) => ({
    ...item,
    color: COLORS[index % COLORS.length],
  }));

  const metricsData = [
    { name: 'Accuracy', value: metrics.accuracy, target: 85 },
    { name: 'Precision', value: metrics.precision, target: 85 },
    { name: 'Recall', value: metrics.recall, target: 80 },
    { name: 'F1 Score', value: metrics.f1Score, target: 82 },
    { name: 'AUC', value: metrics.auc, target: 90 },
  ];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white dark:bg-gray-800 p-3 rounded shadow-lg border border-gray-200 dark:border-gray-700">
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {label}
          </p>
          {payload.map((entry: any, index: number) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value.toFixed(2)}%
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Model Overview */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              ML Model Status
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Version {metrics.modelVersion} • {metrics.dataPoints.toLocaleString()} data points
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <CheckCircleIcon className="h-5 w-5 text-green-500" />
            <span className="text-sm font-medium text-green-600">Healthy</span>
          </div>
        </div>

        {/* Training Schedule */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Last Training</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {safeFormatDate(metrics.lastTraining, 'MMM dd, HH:mm', 'N/A')}
                </p>
              </div>
              <CheckCircleIcon className="h-8 w-8 text-green-400" />
            </div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Next Training</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {safeFormatDate(metrics.nextTraining, 'MMM dd, HH:mm', 'N/A')}
                </p>
              </div>
              <ClockIcon className="h-8 w-8 text-blue-400" />
            </div>
          </div>
        </div>

        {/* Metrics Bars */}
        <div className="space-y-3">
          {metricsData.map((metric) => (
            <div key={metric.name}>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600 dark:text-gray-400">{metric.name}</span>
                <span className={`font-medium ${
                  metric.value >= metric.target ? 'text-green-600' : 'text-yellow-600'
                }`}>
                  {metric.value.toFixed(1)}%
                </span>
              </div>
              <div className="relative">
                <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full">
                  <div
                    className={`h-2 rounded-full ${
                      metric.value >= metric.target ? 'bg-green-500' : 'bg-yellow-500'
                    }`}
                    style={{ width: `${metric.value}%` }}
                  />
                </div>
                <div
                  className="absolute top-0 h-2 w-0.5 bg-gray-600"
                  style={{ left: `${metric.target}%` }}
                  title={`Target: ${metric.target}%`}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance History */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Model Performance History
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceHistory}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(value) => safeFormatDate(value, 'MM/dd', '--')}
                className="text-gray-600 dark:text-gray-400"
                tick={{ fontSize: 12 }}
              />
              <YAxis 
                className="text-gray-600 dark:text-gray-400"
                tick={{ fontSize: 12 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#10b981" 
                strokeWidth={2}
                name="Accuracy"
              />
              <Line 
                type="monotone" 
                dataKey="loss" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Loss"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importance */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Feature Importance
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="importance"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          
          {/* Feature List */}
          <div className="mt-4 space-y-2">
            {featureImportance.map((feature, index) => (
              <div key={feature.feature} className="flex items-center justify-between">
                <div className="flex items-center">
                  <div 
                    className="h-3 w-3 rounded-full mr-2"
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    {feature.feature}
                  </span>
                </div>
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {(feature.importance * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Actions */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Model Management
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors">
            Force Retrain
          </button>
          <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
            Download Model
          </button>
          <button className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors">
            View Logs
          </button>
        </div>
      </div>
    </div>
  );
};

export default MLModelStatus;