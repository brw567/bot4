import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  ClockIcon,
} from '@heroicons/react/24/solid';

interface ServiceStatusProps {
  services: { [key: string]: any };
  apiLatency?: { [key: string]: number };
}

const ServiceStatus: React.FC<ServiceStatusProps> = ({ services, apiLatency }) => {
  const [selectedService, setSelectedService] = useState<string | null>(null);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon className="h-6 w-6 text-green-600" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-6 w-6 text-yellow-600" />;
      case 'error':
        return <XCircleIcon className="h-6 w-6 text-red-600" />;
      case 'stopped':
        return <XCircleIcon className="h-6 w-6 text-gray-600" />;
      default:
        return <ClockIcon className="h-6 w-6 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      case 'error':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'stopped':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200';
    }
  };

  const handleRestart = async (serviceName: string) => {
    try {
      const response = await fetch(`/api/system/services/${encodeURIComponent(serviceName)}/restart`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log(`Service restart response:`, data);
        // Could show a toast notification here
        alert(`${serviceName} restart initiated`);
      } else {
        alert(`Failed to restart ${serviceName}`);
      }
    } catch (error) {
      console.error(`Error restarting ${serviceName}:`, error);
      alert(`Error restarting ${serviceName}`);
    }
  };
  
  const handleViewLogs = async (serviceName: string) => {
    try {
      const response = await fetch(`/api/system/services/${encodeURIComponent(serviceName)}/logs?lines=50`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      
      if (response.ok) {
        const data = await response.json();
        // Open logs in a modal or new window
        const logsWindow = window.open('', `${serviceName} Logs`, 'width=800,height=600');
        if (logsWindow) {
          logsWindow.document.write(`
            <html>
              <head>
                <title>${serviceName} Logs</title>
                <style>
                  body { 
                    font-family: monospace; 
                    background: #1a1a1a; 
                    color: #f0f0f0; 
                    padding: 20px;
                    margin: 0;
                  }
                  pre { 
                    white-space: pre-wrap; 
                    word-wrap: break-word;
                    margin: 0;
                  }
                  h2 {
                    color: #4ade80;
                    margin-bottom: 20px;
                  }
                </style>
              </head>
              <body>
                <h2>${serviceName} Logs (Last 50 lines)</h2>
                <pre>${data.logs.join('')}</pre>
              </body>
            </html>
          `);
        }
      } else {
        alert(`Failed to fetch logs for ${serviceName}`);
      }
    } catch (error) {
      console.error(`Error fetching logs for ${serviceName}:`, error);
      alert(`Error fetching logs for ${serviceName}`);
    }
  };

  // Group services by status
  const servicesByStatus = Object.entries(services).reduce((acc, [name, service]) => {
    const status = service.status;
    if (!acc[status]) acc[status] = [];
    acc[status].push({ name, ...service });
    return acc;
  }, {} as { [key: string]: any[] });

  return (
    <div className="space-y-6">
      {/* Status Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Healthy</p>
              <p className="text-2xl font-semibold text-green-600">
                {servicesByStatus.healthy?.length || 0}
              </p>
            </div>
            <CheckCircleIcon className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Warning</p>
              <p className="text-2xl font-semibold text-yellow-600">
                {servicesByStatus.warning?.length || 0}
              </p>
            </div>
            <ExclamationTriangleIcon className="h-8 w-8 text-yellow-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Error</p>
              <p className="text-2xl font-semibold text-red-600">
                {servicesByStatus.error?.length || 0}
              </p>
            </div>
            <XCircleIcon className="h-8 w-8 text-red-600" />
          </div>
        </div>

        <div className="metric-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Services</p>
              <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                {Object.keys(services).length}
              </p>
            </div>
            <div className="text-primary-600">
              <ArrowPathIcon className="h-8 w-8" />
            </div>
          </div>
        </div>
      </div>

      {/* Services Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {Object.entries(services).map(([name, service]) => (
          <motion.div
            key={name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`metric-card cursor-pointer transition-all ${
              selectedService === name ? 'ring-2 ring-primary-500' : ''
            }`}
            onClick={() => setSelectedService(selectedService === name ? null : name)}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                {getStatusIcon(service.status)}
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">{name}</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Uptime: {service.uptime}
                  </p>
                </div>
              </div>
              <span className={`px-3 py-1 text-xs font-medium rounded-full ${getStatusColor(service.status)}`}>
                {service.status.toUpperCase()}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4">
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">CPU</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {service.cpu}%
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Memory</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {service.memory} MB
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500 dark:text-gray-400">Restarts</p>
                <p className="text-sm font-medium text-gray-900 dark:text-white">
                  {service.restarts}
                </p>
              </div>
            </div>

            {selectedService === name && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4"
              >
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Last Check</span>
                    <span className="text-gray-900 dark:text-white">
                      {new Date(service.lastCheck).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Process ID</span>
                    <span className="text-gray-900 dark:text-white">
                      {service.pid || 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600 dark:text-gray-400">Port</span>
                    <span className="text-gray-900 dark:text-white">
                      {name === 'Redis' ? '6379' : 
                       name === 'WebSocket Server' ? '8080' :
                       name === 'API Gateway' ? '8000' : 'N/A'}
                    </span>
                  </div>
                </div>

                <div className="mt-4 flex space-x-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRestart(name);
                    }}
                    className="flex-1 px-3 py-2 bg-primary-600 text-white text-sm font-medium rounded-md hover:bg-primary-700 transition-colors"
                  >
                    <ArrowPathIcon className="h-4 w-4 inline mr-1" />
                    Restart
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleViewLogs(name);
                    }}
                    className="flex-1 px-3 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
                  >
                    View Logs
                  </button>
                </div>
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Dependencies Graph */}
      <div className="metric-card">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Service Dependencies</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-green-600 rounded-full"></div>
              <span className="text-sm font-medium text-gray-900 dark:text-white">Trading Bot</span>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
              <span>depends on</span>
              <ArrowPathIcon className="h-4 w-4" />
              <span className="font-medium">Redis, ML Pipeline, API Gateway</span>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-green-600 rounded-full"></div>
              <span className="text-sm font-medium text-gray-900 dark:text-white">ML Pipeline</span>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
              <span>depends on</span>
              <ArrowPathIcon className="h-4 w-4" />
              <span className="font-medium">Redis, Data Collector</span>
            </div>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-2 h-2 bg-yellow-600 rounded-full"></div>
              <span className="text-sm font-medium text-gray-900 dark:text-white">Data Collector</span>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
              <span>depends on</span>
              <ArrowPathIcon className="h-4 w-4" />
              <span className="font-medium">API Gateway</span>
            </div>
          </div>
        </div>
      </div>

      {/* Multi-Exchange API Performance */}
      {apiLatency && Object.keys(apiLatency).length > 0 && (
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Exchange API Performance
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(apiLatency).map(([exchange, latency]) => (
              <div key={exchange} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {exchange}
                  </span>
                  <span className={`text-sm font-bold ${
                    latency < 100 ? 'text-green-600' : 
                    latency < 200 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {latency.toFixed(0)}ms
                  </span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      latency < 100 ? 'bg-green-500' : 
                      latency < 200 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min((latency / 300) * 100, 100)}%` }}
                  />
                </div>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  {latency < 100 ? 'Excellent' : latency < 200 ? 'Good' : 'Poor'}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ServiceStatus;