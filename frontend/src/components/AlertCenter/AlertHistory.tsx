import React, { useState } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
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
import { CalendarIcon, ChartBarIcon } from '@heroicons/react/24/outline';
import { Alert } from './index';

interface AlertHistoryProps {
  alerts: Alert[];
}

const AlertHistory: React.FC<AlertHistoryProps> = ({ alerts }) => {
  const [timeRange, setTimeRange] = useState('7d');
  const [selectedCategory, setSelectedCategory] = useState('all');

  // Filter alerts by time range
  const getFilteredAlerts = () => {
    const now = new Date();
    const ranges: { [key: string]: number } = {
      '24h': 24 * 60 * 60 * 1000,
      '7d': 7 * 24 * 60 * 60 * 1000,
      '30d': 30 * 24 * 60 * 60 * 1000,
      '90d': 90 * 24 * 60 * 60 * 1000,
    };

    return alerts.filter(alert => {
      const alertTime = new Date(alert.timestamp).getTime();
      const inRange = now.getTime() - alertTime <= ranges[timeRange];
      const inCategory = selectedCategory === 'all' || alert.category === selectedCategory;
      return inRange && inCategory;
    });
  };

  const filteredAlerts = getFilteredAlerts();

  // Calculate statistics
  const stats = {
    total: filteredAlerts.length,
    critical: filteredAlerts.filter(a => a.type === 'critical').length,
    warning: filteredAlerts.filter(a => a.type === 'warning').length,
    info: filteredAlerts.filter(a => a.type === 'info').length,
    success: filteredAlerts.filter(a => a.type === 'success').length,
  };

  // Prepare data for charts
  const prepareTimeSeriesData = () => {
    const data: { [key: string]: any } = {};
    
    filteredAlerts.forEach(alert => {
      const date = new Date(alert.timestamp).toLocaleDateString();
      if (!data[date]) {
        data[date] = {
          date,
          total: 0,
          critical: 0,
          warning: 0,
          info: 0,
          success: 0,
        };
      }
      data[date].total++;
      data[date][alert.type]++;
    });

    return Object.values(data).sort((a, b) => 
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );
  };

  const prepareCategoryData = () => {
    const data: { [key: string]: number } = {};
    
    filteredAlerts.forEach(alert => {
      if (!data[alert.category]) {
        data[alert.category] = 0;
      }
      data[alert.category]++;
    });

    return Object.entries(data).map(([category, count]) => ({
      category: category.charAt(0).toUpperCase() + category.slice(1),
      count,
    }));
  };

  const timeSeriesData = prepareTimeSeriesData();
  const categoryData = prepareCategoryData();

  const pieData = [
    { name: 'Critical', value: stats.critical, color: '#EF4444' },
    { name: 'Warning', value: stats.warning, color: '#F59E0B' },
    { name: 'Info', value: stats.info, color: '#3B82F6' },
    { name: 'Success', value: stats.success, color: '#10B981' },
  ].filter(d => d.value > 0);

  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({
    cx, cy, midAngle, innerRadius, outerRadius, percent
  }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text 
        x={x} 
        y={y} 
        fill="white" 
        textAnchor={x > cx ? 'start' : 'end'} 
        dominantBaseline="central"
        className="text-xs font-medium"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex items-center space-x-2">
          <CalendarIcon className="h-5 w-5 text-gray-400" />
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
          >
            <option value="24h">Last 24 hours</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
            <option value="90d">Last 90 days</option>
          </select>
        </div>

        <select
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800"
        >
          <option value="all">All Categories</option>
          <option value="system">System</option>
          <option value="trading">Trading</option>
          <option value="performance">Performance</option>
          <option value="security">Security</option>
        </select>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Alerts</p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-white">{stats.total}</p>
        </div>
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Critical</p>
          <p className="text-2xl font-semibold text-red-600">{stats.critical}</p>
        </div>
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Warning</p>
          <p className="text-2xl font-semibold text-yellow-600">{stats.warning}</p>
        </div>
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Info</p>
          <p className="text-2xl font-semibold text-blue-600">{stats.info}</p>
        </div>
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Success</p>
          <p className="text-2xl font-semibold text-green-600">{stats.success}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Alert Trend */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Alert Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="date" 
                tick={{ fill: '#9CA3AF' }}
                tickFormatter={(value) => new Date(value).toLocaleDateString([], { month: 'short', day: 'numeric' })}
              />
              <YAxis tick={{ fill: '#9CA3AF' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.375rem',
                }}
                labelStyle={{ color: '#9CA3AF' }}
              />
              <Legend wrapperStyle={{ color: '#9CA3AF' }} />
              <Line type="monotone" dataKey="total" stroke="#8B5CF6" strokeWidth={2} name="Total" />
              <Line type="monotone" dataKey="critical" stroke="#EF4444" strokeWidth={2} name="Critical" />
              <Line type="monotone" dataKey="warning" stroke="#F59E0B" strokeWidth={2} name="Warning" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Alert Type Distribution */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Type Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={renderCustomizedLabel}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
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
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Category Breakdown */}
      <div className="metric-card">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Alerts by Category</h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={categoryData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="category" tick={{ fill: '#9CA3AF' }} />
            <YAxis tick={{ fill: '#9CA3AF' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '0.375rem',
              }}
              labelStyle={{ color: '#9CA3AF' }}
            />
            <Bar dataKey="count" fill="#3B82F6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Recent History Table */}
      <div className="metric-card">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Recent Alert History</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Title
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Message
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
              {filteredAlerts.slice(0, 10).map((alert) => (
                <tr key={alert.id}>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {new Date(alert.timestamp).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                      alert.type === 'critical' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                      alert.type === 'warning' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                      alert.type === 'info' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' :
                      'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                    }`}>
                      {alert.type}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400 capitalize">
                    {alert.category}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {alert.title}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400">
                    <div className="max-w-xs truncate">{alert.message}</div>
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

export default AlertHistory;