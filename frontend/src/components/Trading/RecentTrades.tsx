import { format } from 'date-fns';
import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid';
import { motion, AnimatePresence } from 'framer-motion';

interface Trade {
  id: string;
  timestamp: string;
  pair: string;
  side: 'buy' | 'sell';
  price: number;
  quantity: number;
  total: number;
  strategy: string;
  profit?: number;
  status: 'completed' | 'pending' | 'cancelled' | 'simulated' | 'stage_test';
  is_test?: boolean;
  test_mode?: 'stage' | 'testnet' | null;
}

interface RecentTradesProps {
  trades: Trade[];
  limit?: number;
}

const RecentTrades: React.FC<RecentTradesProps> = ({ trades, limit = 20 }) => {
  const displayTrades = trades.slice(0, limit);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-200';
      case 'pending':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-200';
      case 'cancelled':
        return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-200';
      case 'simulated':
        return 'text-blue-600 bg-blue-100 dark:bg-blue-900 dark:text-blue-200';
      case 'stage_test':
        return 'text-purple-600 bg-purple-100 dark:bg-purple-900 dark:text-purple-200';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-300';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Recent Trades
          </h3>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            Last {displayTrades.length} trades
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-700">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Time
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Pair
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Side
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Price
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Quantity
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Total
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Profit
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Strategy
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200 dark:bg-gray-800 dark:divide-gray-700">
            <AnimatePresence>
              {displayTrades.map((trade, index) => (
                <motion.tr
                  key={trade.id}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  transition={{ delay: index * 0.05 }}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {format(new Date(trade.timestamp), 'HH:mm:ss')}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white">
                    {trade.pair}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                      trade.side === 'buy' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' : 
                      'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {trade.side.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    ${trade.price.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    {trade.quantity.toFixed(6)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                    ${trade.total.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    {trade.profit !== undefined && (
                      <div className={`flex items-center ${trade.profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {trade.profit >= 0 ? (
                          <ArrowUpIcon className="h-3 w-3 mr-1" />
                        ) : (
                          <ArrowDownIcon className="h-3 w-3 mr-1" />
                        )}
                        ${Math.abs(trade.profit).toFixed(2)}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                    {trade.strategy}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                      getStatusColor(trade.status)
                    }`}>
                      {trade.is_test ? (
                        trade.test_mode === 'stage' ? 'STAGE TEST' : 'TEST'
                      ) : (
                        trade.status === 'completed' ? 'LIVE' : trade.status
                      )}
                    </span>
                  </td>
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
      </div>

      {trades.length === 0 && (
        <div className="p-8 text-center text-gray-500 dark:text-gray-400">
          No trades yet
        </div>
      )}

      {/* Summary Footer */}
      <div className="px-4 py-3 bg-gray-50 dark:bg-gray-700 border-t border-gray-200 dark:border-gray-600">
        <div className="flex items-center justify-between text-sm">
          <div>
            <span className="text-gray-500 dark:text-gray-400">Total Trades: </span>
            <span className="font-medium text-gray-900 dark:text-white">{trades.length}</span>
          </div>
          <div>
            <span className="text-gray-500 dark:text-gray-400">Total Volume: </span>
            <span className="font-medium text-gray-900 dark:text-white">
              ${trades.reduce((sum, t) => sum + t.total, 0).toFixed(2)}
            </span>
          </div>
          <div>
            <span className="text-gray-500 dark:text-gray-400">Total Profit: </span>
            <span className={`font-medium ${
              trades.reduce((sum, t) => sum + (t.profit || 0), 0) >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              ${trades.reduce((sum, t) => sum + (t.profit || 0), 0).toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecentTrades;