import React from 'react';
import { useAppSelector } from '../hooks/redux';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowUpIcon, 
  ArrowDownIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ChartPieIcon,
  ClockIcon
} from '@heroicons/react/24/outline';
import MarketplaceStatus from '../components/MarketplaceStatus';
import BalanceDisplay from '../components/Portfolio/BalanceDisplay';
import TestNetToggle from '../components/TestNetToggle';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { winRate, totalPnl, activePairs, metrics } = useAppSelector(state => state.metrics);
  const { health } = useAppSelector(state => state.system);

  const stats = [
    {
      name: 'Win Rate',
      value: `${winRate.toFixed(1)}%`,
      change: '+2.3%',
      changeType: 'positive',
      icon: ChartPieIcon,
    },
    {
      name: 'Total P&L',
      value: `$${totalPnl.toFixed(2)}`,
      change: '+$234.56',
      changeType: 'positive',
      icon: CurrencyDollarIcon,
    },
    {
      name: 'Active Pairs',
      value: activePairs.toString(),
      change: '+5',
      changeType: 'positive',
      icon: ChartBarIcon,
    },
    {
      name: 'System Uptime',
      value: '99.9%',
      change: '24h',
      changeType: 'neutral',
      icon: ClockIcon,
    },
  ];

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Real-time overview of your trading system performance
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-8">
        {stats.map((stat) => (
          <div key={stat.name} className="metric-card">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <stat.icon className="h-6 w-6 text-gray-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 truncate">
                    {stat.name}
                  </dt>
                  <dd className="flex items-baseline">
                    <div className="text-2xl font-semibold text-gray-900 dark:text-white">
                      {stat.value}
                    </div>
                    <div className={`ml-2 flex items-baseline text-sm font-semibold ${
                      stat.changeType === 'positive' ? 'text-green-600' : 
                      stat.changeType === 'negative' ? 'text-red-600' : 
                      'text-gray-500'
                    }`}>
                      {stat.changeType === 'positive' && <ArrowUpIcon className="h-4 w-4 mr-1" />}
                      {stat.changeType === 'negative' && <ArrowDownIcon className="h-4 w-4 mr-1" />}
                      {stat.change}
                    </div>
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* TestNet Mode Toggle */}
      <div className="mb-8">
        <TestNetToggle />
      </div>

      {/* Portfolio and Marketplace */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Portfolio Balance */}
        <div>
          <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Portfolio Balance
          </h2>
          <BalanceDisplay compact={true} />
        </div>
        
        {/* Marketplace Status */}
        <div>
          <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Marketplace Status
          </h2>
          <MarketplaceStatus compact={true} showFunds={false} />
        </div>
      </div>

      {/* Quick Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Top Performing Pairs
          </h3>
          <div className="space-y-3">
            {metrics.slice(0, 5).map((metric) => (
              <div key={metric.pair} className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {metric.pair}
                </span>
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-500">
                    Win Rate: {metric.winRate.toFixed(1)}%
                  </span>
                  <span className={`text-sm font-medium ${
                    metric.change > 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {metric.change > 0 ? '+' : ''}{metric.change.toFixed(2)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            System Health
          </h3>
          {health && (
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">CPU Usage</span>
                <span className="text-sm font-medium">{health.cpuUsage.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Memory Usage</span>
                <span className="text-sm font-medium">{health.memoryUsage.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">API Latency</span>
                <span className="text-sm font-medium">
                  {Object.values(health.apiLatency).reduce((a, b) => a + b, 0) / Object.values(health.apiLatency).length} ms
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-500">Redis Status</span>
                <span className={`text-sm font-medium ${health.redisConnected ? 'text-green-600' : 'text-red-600'}`}>
                  {health.redisConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;