import React, { useState, useMemo } from 'react';
import { MetricData } from '../../types';
import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid';
import { ChevronUpIcon, ChevronDownIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';

interface MetricsTableProps {
  data: MetricData[];
  highlightThreshold: number;
  onPairClick?: (pair: string) => void;
}

type SortKey = keyof MetricData;
type SortDirection = 'asc' | 'desc';

const MetricsTable: React.FC<MetricsTableProps> = ({ 
  data, 
  highlightThreshold, 
  onPairClick 
}) => {
  const [sortKey, setSortKey] = useState<SortKey>('winRate');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [filter, setFilter] = useState('');

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortKey(key);
      setSortDirection('desc');
    }
  };

  const filteredAndSortedData = useMemo(() => {
    let filtered = data;
    
    if (filter) {
      filtered = data.filter(item => 
        item.pair.toLowerCase().includes(filter.toLowerCase())
      );
    }

    return [...filtered].sort((a, b) => {
      const aValue = a[sortKey];
      const bValue = b[sortKey];
      
      if (typeof aValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue as string)
          : (bValue as string).localeCompare(aValue);
      }
      
      return sortDirection === 'asc' 
        ? (aValue as number) - (bValue as number)
        : (bValue as number) - (aValue as number);
    });
  }, [data, filter, sortKey, sortDirection]);

  const getChangeColor = (change: number) => {
    if (Math.abs(change) > highlightThreshold) {
      return change > 0 ? 'text-red-600 font-bold' : 'text-red-600 font-bold';
    }
    return change > 0 ? 'text-green-600' : 'text-red-600';
  };

  const getRegimeEmoji = (regime: string) => {
    switch (regime) {
      case 'bull': return '🐂';
      case 'bear': return '🐻';
      default: return '😐';
    }
  };

  const columns = [
    { key: 'pair', label: 'Pair', sortable: true },
    { key: 'regime', label: 'Regime', sortable: true },
    { key: 'volatility', label: 'Volatility', sortable: true },
    { key: 'rsi', label: 'RSI', sortable: true },
    { key: 'adx', label: 'ADX', sortable: true },
    { key: 'orderImbalance', label: 'Order Imb.', sortable: true },
    { key: 'openInterest', label: 'OI', sortable: true },
    { key: 'fundingRate', label: 'Funding', sortable: true },
    { key: 'change', label: 'Change %', sortable: true },
    { key: 'winRate', label: 'Win Rate', sortable: true },
    { key: 'activeStrategy', label: 'Strategy', sortable: true },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Real-time Metrics ({filteredAndSortedData.length} pairs)
          </h3>
          <input
            type="text"
            placeholder="Filter pairs..."
            className="px-3 py-1 text-sm border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              {columns.map((column) => (
                <th
                  key={column.key}
                  className={clsx(
                    'px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider',
                    column.sortable && 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600'
                  )}
                  onClick={() => column.sortable && handleSort(column.key as SortKey)}
                >
                  <div className="flex items-center space-x-1">
                    <span>{column.label}</span>
                    {column.sortable && sortKey === column.key && (
                      sortDirection === 'asc' ? 
                        <ChevronUpIcon className="h-4 w-4" /> : 
                        <ChevronDownIcon className="h-4 w-4" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-800 dark:divide-gray-700">
            {filteredAndSortedData.map((metric) => (
              <tr 
                key={metric.pair}
                className="hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
                onClick={() => onPairClick?.(metric.pair)}
              >
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                  {metric.pair}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  <span className="text-lg">{getRegimeEmoji(metric.regime)}</span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {(metric.volatility || 0).toFixed(3)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {(metric.rsi || 0).toFixed(1)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {(metric.adx || 0).toFixed(1)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {(metric.orderImbalance || 0).toFixed(3)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {((metric.openInterest || 0) / 1000000).toFixed(2)}M
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {((metric.fundingRate || 0) * 100).toFixed(3)}%
                </td>
                <td className={clsx(
                  'px-6 py-4 whitespace-nowrap text-sm font-medium',
                  getChangeColor(metric.change)
                )}>
                  <div className="flex items-center">
                    {metric.change > 0 ? (
                      <ArrowUpIcon className="h-4 w-4 mr-1" />
                    ) : (
                      <ArrowDownIcon className="h-4 w-4 mr-1" />
                    )}
                    {Math.abs(metric.change || 0).toFixed(2)}%
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-green-600 dark:text-green-400">
                  {(metric.winRate || 0).toFixed(1)}%
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  <span className="px-2 py-1 text-xs font-medium bg-blue-100 text-blue-800 rounded-full dark:bg-blue-900 dark:text-blue-200">
                    {metric.activeStrategy}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MetricsTable;