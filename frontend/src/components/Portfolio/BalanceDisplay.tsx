import React, { useState, useEffect } from 'react';
import { useAppSelector } from '../../hooks/redux';
import { formatCurrency, formatNumber } from '../../utils/formatters';
import { ArrowUpIcon, ArrowDownIcon, WalletIcon } from '@heroicons/react/24/outline';

interface Balance {
  symbol: string;
  total: number;
  free: number;
  used: number;
  exchanges: Record<string, {
    total: number;
    free: number;
    used: number;
  }>;
  usdc_price: number;
  usdc_value: number;
}

interface BalanceDisplayProps {
  compact?: boolean;
}

const BalanceDisplay: React.FC<BalanceDisplayProps> = ({ compact = false }) => {
  const [balances, setBalances] = useState<Balance[]>([]);
  const [totalValue, setTotalValue] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Fetch balances
  const fetchBalances = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('/api/portfolio/balances', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch balances');
      }

      const data = await response.json();
      setBalances(data.balances || []);
      setTotalValue(data.total_usdc_value || 0);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch balances');
    } finally {
      setLoading(false);
    }
  };

  // Setup WebSocket for real-time updates
  useEffect(() => {
    fetchBalances();
    
    // Refresh every 10 seconds
    const interval = setInterval(fetchBalances, 10000);

    // Setup WebSocket for real-time updates
    const ws = new WebSocket(`ws://${window.location.host}/api/portfolio/ws/balances`);
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'balance_update') {
        setBalances(message.data.balances || []);
        setTotalValue(message.data.total_usdc_value || 0);
        setLastUpdate(new Date());
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, []);

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-48 mb-4"></div>
        <div className="space-y-2">
          <div className="h-12 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-12 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-12 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 dark:text-red-400 p-4 rounded-lg bg-red-50 dark:bg-red-900/20">
        <p className="font-medium">Error loading balances</p>
        <p className="text-sm mt-1">{error}</p>
      </div>
    );
  }

  if (compact) {
    // Compact view for dashboard
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white flex items-center">
            <WalletIcon className="h-5 w-5 mr-2 text-gray-500 dark:text-gray-400" />
            Portfolio Value
          </h3>
          <span className="text-2xl font-bold text-gray-900 dark:text-white">
            {formatCurrency(totalValue)}
          </span>
        </div>
        
        <div className="space-y-2">
          {balances.slice(0, 5).map((balance) => (
            <div key={balance.symbol} className="flex items-center justify-between text-sm">
              <div className="flex items-center">
                <span className="font-medium text-gray-700 dark:text-gray-300">
                  {balance.symbol}
                </span>
                <span className="ml-2 text-gray-500 dark:text-gray-400">
                  {formatNumber(balance.total, 6)}
                </span>
              </div>
              <span className="text-gray-700 dark:text-gray-300">
                {formatCurrency(balance.usdc_value)}
              </span>
            </div>
          ))}
          
          {balances.length > 5 && (
            <div className="text-sm text-gray-500 dark:text-gray-400 text-center pt-2">
              +{balances.length - 5} more assets
            </div>
          )}
        </div>
      </div>
    );
  }

  // Full view for portfolio page
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Portfolio Balances
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Live balances across all connected exchanges
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-500 dark:text-gray-400">Total Value</p>
            <p className="text-3xl font-bold text-gray-900 dark:text-white">
              {formatCurrency(totalValue)}
            </p>
            {lastUpdate && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Updated {lastUpdate.toLocaleTimeString()}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Balance Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Asset
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Balance
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Available
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                In Orders
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Price (USDC)
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Value (USDC)
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                % of Portfolio
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {balances.map((balance) => {
              const percentage = (balance.usdc_value / totalValue) * 100;
              
              return (
                <tr key={balance.symbol} className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
                  <td className="px-6 py-4">
                    <div>
                      <div className="text-sm font-medium text-gray-900 dark:text-white">
                        {balance.symbol}
                      </div>
                      <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        {Object.entries(balance.exchanges).map(([exchange, amounts], idx) => (
                          <span key={exchange}>
                            {idx > 0 && ' • '}
                            {exchange}: {formatNumber(amounts.total, 8)}
                          </span>
                        ))}
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900 dark:text-white">
                    {formatNumber(balance.total, 8)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900 dark:text-white">
                    {formatNumber(balance.free, 8)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-500 dark:text-gray-400">
                    {formatNumber(balance.used, 8)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm text-gray-900 dark:text-white">
                    {balance.usdc_price > 0 ? formatCurrency(balance.usdc_price) : '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium text-gray-900 dark:text-white">
                    {formatCurrency(balance.usdc_value)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right">
                    <div className="flex items-center justify-end">
                      <span className="text-sm text-gray-900 dark:text-white">
                        {percentage.toFixed(2)}%
                      </span>
                      <div className="ml-2 w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-primary-600 h-2 rounded-full"
                          style={{ width: `${Math.min(percentage, 100)}%` }}
                        />
                      </div>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
          <tfoot className="bg-gray-50 dark:bg-gray-900">
            <tr>
              <td colSpan={5} className="px-6 py-4 text-right text-sm font-medium text-gray-900 dark:text-white">
                Total Portfolio Value
              </td>
              <td className="px-6 py-4 text-right text-lg font-bold text-gray-900 dark:text-white">
                {formatCurrency(totalValue)}
              </td>
              <td className="px-6 py-4 text-right text-sm text-gray-500 dark:text-gray-400">
                100.00%
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* Summary Statistics */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm p-6">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Portfolio Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Total Assets</p>
            <p className="text-xl font-semibold text-gray-900 dark:text-white">{balances.length}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Total Value</p>
            <p className="text-xl font-semibold text-gray-900 dark:text-white">{formatCurrency(totalValue)}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Connected Exchanges</p>
            <p className="text-xl font-semibold text-gray-900 dark:text-white">
              {new Set(balances.flatMap(b => Object.keys(b.exchanges))).size}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-500 dark:text-gray-400">Last Update</p>
            <p className="text-xl font-semibold text-gray-900 dark:text-white">
              {lastUpdate ? lastUpdate.toLocaleTimeString() : '-'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BalanceDisplay;