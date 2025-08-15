import React, { useState } from 'react';
import { Tab } from '@headlessui/react';
import { 
  ShieldCheckIcon, 
  BeakerIcon, 
  ChartBarIcon,
  CpuChipIcon,
  ServerStackIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';
import CircuitBreakerStatus from './CircuitBreakerStatus';
import ModelVersioning from './ModelVersioning';
import PerformanceProfiler from './PerformanceProfiler';
import { useAppSelector } from '../../hooks/redux';

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

const IntegratedSystemDashboard: React.FC = () => {
  const { health, botStatus } = useAppSelector(state => state.system);
  const [selectedTab, setSelectedTab] = useState(0);

  const tabs = [
    { name: 'Overview', icon: ChartBarIcon },
    { name: 'Circuit Breakers', icon: ShieldCheckIcon },
    { name: 'Model Versioning', icon: BeakerIcon },
    { name: 'Performance', icon: CpuChipIcon },
  ];

  // Calculate system health score
  const systemScore = health ? (
    (100 - health.cpuUsage) * 0.25 +
    (100 - health.memoryUsage) * 0.25 +
    (health.redisConnected ? 100 : 0) * 0.25 +
    (botStatus === 'running' ? 100 : 0) * 0.25
  ) : 0;

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100 dark:bg-green-900/20';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20';
    return 'text-red-600 bg-red-100 dark:bg-red-900/20';
  };

  const getHealthLabel = (score: number) => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Critical';
  };

  return (
    <div className="space-y-6">
      {/* Header with System Status */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
              Integrated System Dashboard
            </h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Complete overview of Bot3 trading platform health and performance
            </p>
          </div>
          <div className="text-center">
            <div className={`inline-flex items-center px-4 py-2 rounded-lg ${getHealthColor(systemScore)}`}>
              <div>
                <div className="text-3xl font-bold">
                  {systemScore.toFixed(0)}%
                </div>
                <div className="text-sm font-medium">
                  {getHealthLabel(systemScore)}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${
              health?.cpuUsage > 80 ? 'bg-red-100 dark:bg-red-900/20' : 
              health?.cpuUsage > 60 ? 'bg-yellow-100 dark:bg-yellow-900/20' : 
              'bg-green-100 dark:bg-green-900/20'
            }`}>
              <CpuChipIcon className={`h-6 w-6 ${
                health?.cpuUsage > 80 ? 'text-red-600' : 
                health?.cpuUsage > 60 ? 'text-yellow-600' : 
                'text-green-600'
              }`} />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">CPU</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {(health?.cpuUsage || 0).toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${
              health?.memoryUsage > 80 ? 'bg-red-100 dark:bg-red-900/20' : 
              health?.memoryUsage > 60 ? 'bg-yellow-100 dark:bg-yellow-900/20' : 
              'bg-green-100 dark:bg-green-900/20'
            }`}>
              <ServerStackIcon className={`h-6 w-6 ${
                health?.memoryUsage > 80 ? 'text-red-600' : 
                health?.memoryUsage > 60 ? 'text-yellow-600' : 
                'text-green-600'
              }`} />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Memory</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {(health?.memoryUsage || 0).toFixed(1)}%
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${
              botStatus === 'running' ? 'bg-green-100 dark:bg-green-900/20' : 
              botStatus === 'stopped' ? 'bg-red-100 dark:bg-red-900/20' : 
              'bg-yellow-100 dark:bg-yellow-900/20'
            }`}>
              <ChartBarIcon className={`h-6 w-6 ${
                botStatus === 'running' ? 'text-green-600' : 
                botStatus === 'stopped' ? 'text-red-600' : 
                'text-yellow-600'
              }`} />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Bot Status</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white capitalize">
                {botStatus || 'Unknown'}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <div className={`p-2 rounded-lg ${
              health?.redisConnected ? 'bg-green-100 dark:bg-green-900/20' : 'bg-red-100 dark:bg-red-900/20'
            }`}>
              <ServerStackIcon className={`h-6 w-6 ${
                health?.redisConnected ? 'text-green-600' : 'text-red-600'
              }`} />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Redis</p>
              <p className="text-lg font-semibold text-gray-900 dark:text-white">
                {health?.redisConnected ? 'Connected' : 'Disconnected'}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabbed Content */}
      <Tab.Group selectedIndex={selectedTab} onChange={setSelectedTab}>
        <Tab.List className="flex space-x-1 rounded-xl bg-gray-100 dark:bg-gray-900 p-1">
          {tabs.map((tab) => (
            <Tab
              key={tab.name}
              className={({ selected }) =>
                classNames(
                  'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                  'ring-white ring-opacity-60 ring-offset-2 ring-offset-primary-400 focus:outline-none focus:ring-2',
                  selected
                    ? 'bg-white dark:bg-gray-800 text-primary-700 dark:text-primary-400 shadow'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-white/[0.12] hover:text-gray-900 dark:hover:text-white'
                )
              }
            >
              <div className="flex items-center justify-center">
                <tab.icon className="h-5 w-5 mr-2" />
                {tab.name}
              </div>
            </Tab>
          ))}
        </Tab.List>
        <Tab.Panels className="mt-6">
          {/* Overview Tab */}
          <Tab.Panel className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* System Alerts */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  System Alerts
                </h3>
                <div className="space-y-2">
                  {/* This would be populated with real alerts */}
                  <div className="flex items-center p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                    <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600 mr-3" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-gray-900 dark:text-white">
                        High memory usage detected
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        Memory usage above 80% threshold
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                  Quick Actions
                </h3>
                <div className="grid grid-cols-2 gap-2">
                  <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm">
                    Reset Circuit Breakers
                  </button>
                  <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm">
                    Rollback Model
                  </button>
                  <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm">
                    Clear Cache
                  </button>
                  <button className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 text-sm">
                    Force GC
                  </button>
                </div>
              </div>
            </div>

            {/* System Metrics Summary */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
                System Metrics Summary
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Uptime</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">99.9%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Requests/sec</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">1,234</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Avg Latency</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">45ms</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Error Rate</p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">0.01%</p>
                </div>
              </div>
            </div>
          </Tab.Panel>

          {/* Circuit Breakers Tab */}
          <Tab.Panel>
            <CircuitBreakerStatus />
          </Tab.Panel>

          {/* Model Versioning Tab */}
          <Tab.Panel>
            <ModelVersioning />
          </Tab.Panel>

          {/* Performance Tab */}
          <Tab.Panel>
            <PerformanceProfiler />
          </Tab.Panel>
        </Tab.Panels>
      </Tab.Group>
    </div>
  );
};

export default IntegratedSystemDashboard;