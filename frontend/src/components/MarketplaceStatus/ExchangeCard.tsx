import React from 'react';
import { CheckCircleIcon, XCircleIcon, ClockIcon } from '@heroicons/react/24/solid';

interface ExchangeCardProps {
  exchange: {
    exchange: string;
    status: 'healthy' | 'degraded' | 'error';
    uptime: number;
    avgLatency: number;
    errorRate: number;
    requestsTotal: number;
    requestsSuccess: number;
    lastUpdate: string;
  };
}

const ExchangeCard: React.FC<ExchangeCardProps> = ({ exchange }) => {
  const getStatusIcon = () => {
    switch (exchange.status) {
      case 'healthy':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'degraded':
        return <ClockIcon className="h-5 w-5 text-yellow-500" />;
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-red-500" />;
    }
  };

  const getStatusColor = () => {
    switch (exchange.status) {
      case 'healthy':
        return 'border-green-500 bg-green-50 dark:bg-green-900/20';
      case 'degraded':
        return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
      case 'error':
        return 'border-red-500 bg-red-50 dark:bg-red-900/20';
    }
  };

  const successRate = exchange.requestsTotal > 0 
    ? ((exchange.requestsSuccess / exchange.requestsTotal) * 100).toFixed(1)
    : '0.0';

  return (
    <div className={`p-4 rounded-lg border-2 ${getStatusColor()}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-lg font-semibold capitalize text-gray-900 dark:text-white">
          {exchange.exchange}
        </h3>
        {getStatusIcon()}
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <p className="text-gray-600 dark:text-gray-400">Uptime</p>
          <p className="font-medium text-gray-900 dark:text-white">{exchange.uptime.toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-gray-600 dark:text-gray-400">Latency</p>
          <p className="font-medium text-gray-900 dark:text-white">{exchange.avgLatency.toFixed(0)}ms</p>
        </div>
        <div>
          <p className="text-gray-600 dark:text-gray-400">Success Rate</p>
          <p className="font-medium text-gray-900 dark:text-white">{successRate}%</p>
        </div>
        <div>
          <p className="text-gray-600 dark:text-gray-400">Error Rate</p>
          <p className="font-medium text-gray-900 dark:text-white">{exchange.errorRate.toFixed(2)}%</p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-500 dark:text-gray-400">
            Total Requests: {exchange.requestsTotal.toLocaleString()}
          </span>
          <span className="text-gray-500 dark:text-gray-400">
            Last Update: {new Date(exchange.lastUpdate).toLocaleTimeString()}
          </span>
        </div>
      </div>
    </div>
  );
};

export default ExchangeCard;