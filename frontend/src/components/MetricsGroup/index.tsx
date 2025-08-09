import React, { useState, useEffect, useMemo } from 'react';
import { useAppSelector } from '../../hooks/redux';
import MetricsTable from './MetricsTable';
import MetricCard from './MetricCard';
import ChangeLog from './ChangeLog';
import AlertBanner from './AlertBanner';
import { TableCellsIcon, Squares2X2Icon } from '@heroicons/react/24/outline';
import clsx from 'clsx';
import toast from 'react-hot-toast';

type ViewMode = 'table' | 'grid';

const MetricsGroup: React.FC = () => {
  const { metrics, changes } = useAppSelector(state => state.metrics);
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [selectedPair, setSelectedPair] = useState<string | null>(null);
  const [showAlertBanner, setShowAlertBanner] = useState(false);
  
  const HIGHLIGHT_THRESHOLD = 0.1; // 10% change threshold

  // Check for critical changes
  useEffect(() => {
    const hasCriticalChanges = changes.some(
      change => change.severity === 'critical' && 
      new Date(change.timestamp).getTime() > Date.now() - 60000 // Last minute
    );
    setShowAlertBanner(hasCriticalChanges);
  }, [changes]);

  // Filter metrics based on selection
  const displayMetrics = useMemo(() => {
    if (selectedPair) {
      return metrics.filter(m => m.pair === selectedPair);
    }
    return metrics;
  }, [metrics, selectedPair]);

  const handlePairClick = (pair: string) => {
    setSelectedPair(pair);
    toast.success(`Selected ${pair}`);
  };

  const handleDismissAlert = () => {
    setShowAlertBanner(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Metrics Dashboard
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Real-time metrics for {metrics.length} trading pairs
            {selectedPair && (
              <span className="ml-2">
                • Filtered: <span className="font-medium">{selectedPair}</span>
                <button
                  onClick={() => setSelectedPair(null)}
                  className="ml-2 text-primary-600 hover:text-primary-700 dark:text-primary-400"
                >
                  Clear
                </button>
              </span>
            )}
          </p>
        </div>
        
        {/* View Mode Toggle */}
        <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          <button
            onClick={() => setViewMode('table')}
            className={clsx(
              'flex items-center px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
              viewMode === 'table'
                ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            )}
          >
            <TableCellsIcon className="h-4 w-4 mr-1.5" />
            Table
          </button>
          <button
            onClick={() => setViewMode('grid')}
            className={clsx(
              'flex items-center px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
              viewMode === 'grid'
                ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
            )}
          >
            <Squares2X2Icon className="h-4 w-4 mr-1.5" />
            Grid
          </button>
        </div>
      </div>

      {/* Alert Banner */}
      <AlertBanner 
        active={showAlertBanner}
        onDismiss={handleDismissAlert}
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2">
          {viewMode === 'table' ? (
            <MetricsTable
              data={displayMetrics}
              highlightThreshold={HIGHLIGHT_THRESHOLD}
              onPairClick={handlePairClick}
            />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {displayMetrics.map((metric) => (
                <MetricCard
                  key={metric.pair}
                  metric={metric}
                  highlightThreshold={HIGHLIGHT_THRESHOLD}
                  onClick={() => handlePairClick(metric.pair)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Change Log Sidebar */}
        <div className="lg:col-span-1">
          <ChangeLog changes={changes} />
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Avg Win Rate</p>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {metrics.length > 0 
              ? (metrics.reduce((sum, m) => sum + m.winRate, 0) / metrics.length).toFixed(1)
              : '0.0'}%
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Bull Markets</p>
          <p className="text-2xl font-bold text-green-600 dark:text-green-400">
            {metrics.filter(m => m.regime === 'bull').length}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Bear Markets</p>
          <p className="text-2xl font-bold text-red-600 dark:text-red-400">
            {metrics.filter(m => m.regime === 'bear').length}
          </p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
          <p className="text-sm text-gray-500 dark:text-gray-400">Critical Changes</p>
          <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
            {changes.filter(c => c.severity === 'critical').length}
          </p>
        </div>
      </div>
    </div>
  );
};

export default MetricsGroup;