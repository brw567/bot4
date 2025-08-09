import React from 'react';
import { ArrowTrendingUpIcon, CurrencyDollarIcon, ChartBarIcon } from '@heroicons/react/24/outline';

interface ArbitrageMetricsProps {
  metrics: {
    opportunitiesFound: number;
    opportunitiesExecuted: number;
    totalProfit: number;
    avgSpread: number;
    bestSpread: number;
    successRate: number;
    exchangesInvolved: string[];
  };
}

const ArbitrageMetrics: React.FC<ArbitrageMetricsProps> = ({ metrics }) => {
  const executionRate = metrics.opportunitiesFound > 0 
    ? ((metrics.opportunitiesExecuted / metrics.opportunitiesFound) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Arbitrage Performance
        </h3>
        <ArrowTrendingUpIcon className="h-6 w-6 text-primary-600" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Opportunities Found</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {metrics.opportunitiesFound}
              </p>
            </div>
            <ChartBarIcon className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Executed</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {metrics.opportunitiesExecuted}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {executionRate}% execution rate
              </p>
            </div>
            <div className="text-green-600">
              <svg className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Profit</p>
              <p className="text-2xl font-bold text-green-600">
                ${metrics.totalProfit.toFixed(2)}
              </p>
            </div>
            <CurrencyDollarIcon className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Avg Spread</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {metrics.avgSpread.toFixed(3)}%
              </p>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Best Spread</p>
              <p className="text-2xl font-bold text-primary-600">
                {metrics.bestSpread.toFixed(3)}%
              </p>
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                {metrics.successRate.toFixed(1)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
        <p className="text-sm text-gray-600 dark:text-gray-400">Exchanges Involved</p>
        <div className="flex flex-wrap gap-2 mt-2">
          {metrics.exchangesInvolved.map((exchange) => (
            <span
              key={exchange}
              className="px-3 py-1 text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-200 rounded-full capitalize"
            >
              {exchange}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ArbitrageMetrics;