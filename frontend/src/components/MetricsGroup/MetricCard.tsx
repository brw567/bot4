import { MetricData } from '../../types';
import { ArrowUpIcon, ArrowDownIcon, SparklesIcon } from '@heroicons/react/24/solid';
import clsx from 'clsx';
import { motion } from 'framer-motion';

interface MetricCardProps {
  metric: MetricData;
  highlightThreshold: number;
  onClick?: () => void;
}

const MetricCard: React.FC<MetricCardProps> = ({ metric, highlightThreshold, onClick }) => {
  const isHighChange = Math.abs(metric.change) > highlightThreshold;
  
  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'bull':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
      case 'bear':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200';
    }
  };

  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      className={clsx(
        'metric-card cursor-pointer transition-all duration-200',
        isHighChange && 'ring-2 ring-red-500 animate-pulse-slow'
      )}
      onClick={onClick}
    >
      <div className="flex justify-between items-start mb-4">
        <div>
          <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
            {metric.pair}
          </h4>
          <span className={clsx(
            'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mt-1',
            getRegimeColor(metric.regime)
          )}>
            {metric.regime}
          </span>
        </div>
        <div className="text-right">
          <div className="flex items-center">
            <SparklesIcon className="h-4 w-4 text-yellow-400 mr-1" />
            <span className="text-lg font-bold text-green-600 dark:text-green-400">
              {metric.winRate.toFixed(1)}%
            </span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400">win rate</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Volatility</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {metric.volatility.toFixed(3)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">RSI</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {metric.rsi.toFixed(1)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Funding Rate</p>
          <p className="text-sm font-medium text-gray-900 dark:text-white">
            {(metric.fundingRate * 100).toFixed(3)}%
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Change</p>
          <p className={clsx(
            'text-sm font-medium flex items-center',
            metric.change > 0 ? 'text-green-600' : 'text-red-600',
            isHighChange && 'font-bold animate-pulse'
          )}>
            {metric.change > 0 ? (
              <ArrowUpIcon className="h-3 w-3 mr-1" />
            ) : (
              <ArrowDownIcon className="h-3 w-3 mr-1" />
            )}
            {Math.abs(metric.change).toFixed(2)}%
          </p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
        <div className="flex justify-between items-center">
          <span className="text-xs text-gray-500 dark:text-gray-400">
            Strategy: {metric.activeStrategy}
          </span>
          <span className={clsx(
            'text-xs font-medium',
            metric.mlConfidence > 0.8 ? 'text-green-600' : 'text-yellow-600'
          )}>
            ML: {(metric.mlConfidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default MetricCard;