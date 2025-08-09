import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FunnelIcon,
  MagnifyingGlassIcon,
  ArrowDownTrayIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import {
  InformationCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  BugAntIcon,
} from '@heroicons/react/24/solid';
import apiService from '../../services/api';

interface Log {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  service: string;
  message: string;
  details?: string;
}

interface SystemLogsProps {
  logs: Log[];
}

const SystemLogs: React.FC<SystemLogsProps> = ({ logs: initialLogs }) => {
  const [logs, setLogs] = useState<Log[]>([]);
  const [filter, setFilter] = useState({
    level: 'all',
    service: 'all',
    search: '',
  });
  const [autoScroll, setAutoScroll] = useState(true);
  const [isLive, setIsLive] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const pollingInterval = useRef<NodeJS.Timeout | null>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  // Fetch real logs from API
  const fetchLogs = useCallback(async () => {
    try {
      setIsLoading(true);
      const params: any = { limit: 100 };
      
      if (filter.service !== 'all') {
        params.service = filter.service.toLowerCase().replace(' ', '_');
      }
      
      if (filter.level !== 'all') {
        params.level = filter.level;
      }
      
      const response = await apiService.getSystemLogs(params);
      
      if (response.logs && Array.isArray(response.logs)) {
        setLogs(response.logs);
      }
    } catch (error) {
      console.error('Failed to fetch system logs:', error);
    } finally {
      setIsLoading(false);
    }
  }, [filter.service, filter.level]);

  // Initial fetch and polling
  useEffect(() => {
    fetchLogs();

    if (isLive) {
      // Poll for new logs every 5 seconds
      pollingInterval.current = setInterval(fetchLogs, 5000);
    }

    return () => {
      if (pollingInterval.current) {
        clearInterval(pollingInterval.current);
      }
    };
  }, [fetchLogs, isLive]);

  const getLevelIcon = (level: string) => {
    switch (level) {
      case 'info':
        return <InformationCircleIcon className="h-5 w-5 text-blue-600" />;
      case 'warning':
        return <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600" />;
      case 'error':
        return <XCircleIcon className="h-5 w-5 text-red-600" />;
      case 'debug':
        return <BugAntIcon className="h-5 w-5 text-gray-600" />;
      default:
        return null;
    }
  };

  const getLevelBg = (level: string) => {
    switch (level) {
      case 'info':
        return 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800';
      case 'warning':
        return 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800';
      case 'error':
        return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
      case 'debug':
        return 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
      default:
        return 'bg-gray-50 dark:bg-gray-900/20 border-gray-200 dark:border-gray-800';
    }
  };

  const filteredLogs = logs.filter(log => {
    // Apply search filter
    if (filter.search && !log.message.toLowerCase().includes(filter.search.toLowerCase())) return false;
    return true;
  });

  const uniqueServices = Array.from(new Set(logs.map(log => log.service)));

  const handleExport = () => {
    const csvContent = [
      ['Timestamp', 'Level', 'Service', 'Message', 'Details'].join(','),
      ...filteredLogs.map(log => 
        [log.timestamp, log.level, log.service, `"${log.message}"`, log.details || ''].join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `system-logs-${new Date().toISOString()}.csv`;
    a.click();
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="metric-card">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search logs..."
                value={filter.search}
                onChange={(e) => setFilter({ ...filter, search: e.target.value })}
                className="w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-primary-500 focus:border-primary-500"
              />
            </div>
          </div>

          {/* Filters */}
          <div className="flex gap-2">
            <select
              value={filter.level}
              onChange={(e) => setFilter({ ...filter, level: e.target.value })}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            >
              <option value="all">All Levels</option>
              <option value="info">Info</option>
              <option value="warning">Warning</option>
              <option value="error">Error</option>
              <option value="debug">Debug</option>
            </select>

            <select
              value={filter.service}
              onChange={(e) => setFilter({ ...filter, service: e.target.value })}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            >
              <option value="all">All Services</option>
              {uniqueServices.map(service => (
                <option key={service} value={service}>{service}</option>
              ))}
            </select>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <button
              onClick={() => setAutoScroll(!autoScroll)}
              className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                autoScroll
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              Auto-scroll
            </button>
            
            <button
              onClick={() => setIsLive(!isLive)}
              className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                isLive
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              {isLive ? 'Live' : 'Paused'}
            </button>

            <button
              onClick={fetchLogs}
              disabled={isLoading}
              className="px-3 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors disabled:opacity-50"
            >
              <ArrowPathIcon className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={handleExport}
              className="px-3 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm font-medium rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              <ArrowDownTrayIcon className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Log Stats */}
      <div className="grid grid-cols-4 gap-4">
        {['info', 'warning', 'error', 'debug'].map(level => {
          const count = logs.filter(log => log.level === level).length;
          return (
            <div key={level} className="metric-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-500 dark:text-gray-400 capitalize">
                    {level}
                  </p>
                  <p className="text-xl font-semibold text-gray-900 dark:text-white">
                    {count}
                  </p>
                </div>
                {getLevelIcon(level)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Logs List */}
      <div className="metric-card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            System Logs ({filteredLogs.length})
          </h3>
          <div className="flex items-center space-x-4">
            {isLoading && (
              <span className="text-sm text-gray-500 dark:text-gray-400">Loading...</span>
            )}
            {isLive && (
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-500 dark:text-gray-400">Live</span>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-2 max-h-[600px] overflow-y-auto">
          <AnimatePresence>
            {filteredLogs.map((log) => (
              <motion.div
                key={log.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className={`p-3 rounded-lg border ${getLevelBg(log.level)}`}
              >
                <div className="flex items-start space-x-3">
                  {getLevelIcon(log.level)}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {log.service}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <span className={`text-xs font-medium px-2 py-1 rounded-full uppercase ${
                        log.level === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200' :
                        log.level === 'warning' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                        log.level === 'debug' ? 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200' :
                        'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                      }`}>
                        {log.level}
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">{log.message}</p>
                    {log.details && (
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{log.details}</p>
                    )}
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  );
};

export default SystemLogs;