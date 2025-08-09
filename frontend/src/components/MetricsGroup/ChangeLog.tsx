import { ChangeEvent } from '../../types';
import { formatDistanceToNow } from 'date-fns';
import { ExclamationTriangleIcon, ExclamationCircleIcon } from '@heroicons/react/24/solid';
import clsx from 'clsx';
import { motion, AnimatePresence } from 'framer-motion';

interface ChangeLogProps {
  changes: ChangeEvent[];
  maxItems?: number;
}

const ChangeLog: React.FC<ChangeLogProps> = ({ changes, maxItems = 20 }) => {
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      case 'high':
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200';
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
      default:
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200';
    }
  };

  const getSeverityIcon = (severity: string) => {
    if (severity === 'critical' || severity === 'high') {
      return <ExclamationTriangleIcon className="h-4 w-4" />;
    }
    return <ExclamationCircleIcon className="h-4 w-4" />;
  };

  const displayChanges = changes.slice(0, maxItems);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Change Log
        </h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Significant metric changes (&gt;{10}% threshold)
        </p>
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        <AnimatePresence>
          {displayChanges.length === 0 ? (
            <div className="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
              No significant changes detected
            </div>
          ) : (
            <ul className="divide-y divide-gray-200 dark:divide-gray-700">
              {displayChanges.map((change, index) => (
                <motion.li
                  key={`${change.pair}-${change.metric}-${change.timestamp}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: 20 }}
                  transition={{ delay: index * 0.05 }}
                  className="px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                  <div className="flex items-start space-x-3">
                    <div className={clsx(
                      'flex-shrink-0 p-1 rounded-full',
                      getSeverityColor(change.severity)
                    )}>
                      {getSeverityIcon(change.severity)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium text-gray-900 dark:text-white">
                          {change.pair} - {change.metric}
                        </p>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {formatDistanceToNow(new Date(change.timestamp), { addSuffix: true })}
                        </span>
                      </div>
                      
                      <div className="mt-1 flex items-center text-sm">
                        <span className="text-gray-500 dark:text-gray-400">
                          {change.oldValue.toFixed(3)} → {change.newValue.toFixed(3)}
                        </span>
                        <span className={clsx(
                          'ml-2 font-medium',
                          change.changePercent > 0 ? 'text-green-600' : 'text-red-600'
                        )}>
                          ({change.changePercent > 0 ? '+' : ''}{change.changePercent.toFixed(1)}%)
                        </span>
                      </div>
                      
                      <span className={clsx(
                        'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mt-1',
                        getSeverityColor(change.severity)
                      )}>
                        {change.severity}
                      </span>
                    </div>
                  </div>
                </motion.li>
              ))}
            </ul>
          )}
        </AnimatePresence>
      </div>
      
      {changes.length > maxItems && (
        <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 text-center">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Showing {maxItems} of {changes.length} changes
          </p>
        </div>
      )}
    </div>
  );
};

export default ChangeLog;