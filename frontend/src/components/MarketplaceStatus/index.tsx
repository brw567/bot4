import React, { useState, useEffect } from 'react';
import { useAppSelector } from '../../hooks/redux';
import ExchangeCard from './ExchangeCard';
import ArbitrageMetrics from './ArbitrageMetrics';
import CrossExchangeChart from './CrossExchangeChart';
import { BuildingLibraryIcon, ArrowsRightLeftIcon } from '@heroicons/react/24/outline';

interface MarketplaceStatusProps {
  compact?: boolean;
  showFunds?: boolean;
}

const MarketplaceStatus: React.FC<MarketplaceStatusProps> = ({ compact = false, showFunds = true }) => {
  const [exchanges, setExchanges] = useState<any[]>([]);
  const [arbitrageMetrics, setArbitrageMetrics] = useState<any>(null);
  const [crossExchangeData, setCrossExchangeData] = useState<any>(null);
  const [selectedPair, setSelectedPair] = useState('BTC/USDT');
  const [loading, setLoading] = useState(true);

  // Fetch exchange status
  useEffect(() => {
    const fetchExchangeStatus = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/monitoring/exchanges', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          setExchanges(data.exchanges);
        }
      } catch (error) {
        console.error('Failed to fetch exchange status:', error);
      }
    };

    fetchExchangeStatus();
    const interval = setInterval(fetchExchangeStatus, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  // Fetch arbitrage metrics
  useEffect(() => {
    const fetchArbitrageMetrics = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/monitoring/arbitrage?timeframe=1h', {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          setArbitrageMetrics(data);
        }
      } catch (error) {
        console.error('Failed to fetch arbitrage metrics:', error);
      }
    };

    fetchArbitrageMetrics();
    const interval = setInterval(fetchArbitrageMetrics, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Fetch cross-exchange analysis
  useEffect(() => {
    const fetchCrossExchangeData = async () => {
      try {
        const token = localStorage.getItem('token');
        const symbol = selectedPair.replace('/', '-');
        const response = await fetch(`/api/monitoring/cross-exchange/${symbol}`, {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          setCrossExchangeData(data);
        }
      } catch (error) {
        console.error('Failed to fetch cross-exchange data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchCrossExchangeData();
    const interval = setInterval(fetchCrossExchangeData, 15000); // Update every 15 seconds
    return () => clearInterval(interval);
  }, [selectedPair]);

  const tradingPairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];

  const healthyExchanges = exchanges.filter(e => e.status === 'healthy').length;
  const totalExchanges = exchanges.length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Multi-Exchange Status</h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Monitor performance across all connected exchanges
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-center">
            <p className="text-sm text-gray-500 dark:text-gray-400">Active Exchanges</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              {healthyExchanges}/{totalExchanges}
            </p>
          </div>
          <BuildingLibraryIcon className="h-8 w-8 text-primary-600" />
        </div>
      </div>

      {/* Exchange Grid */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Exchange Health Status
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {exchanges.map((exchange) => (
            <ExchangeCard key={exchange.exchange} exchange={exchange} />
          ))}
        </div>
      </div>

      {/* Arbitrage Metrics */}
      {arbitrageMetrics && (
        <ArbitrageMetrics metrics={arbitrageMetrics} />
      )}

      {/* Cross-Exchange Analysis */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Cross-Exchange Analysis
          </h3>
          <div className="flex items-center space-x-2">
            <label htmlFor="pair-select" className="text-sm text-gray-600 dark:text-gray-400">
              Trading Pair:
            </label>
            <select
              id="pair-select"
              value={selectedPair}
              onChange={(e) => setSelectedPair(e.target.value)}
              className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-md 
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              {tradingPairs.map((pair) => (
                <option key={pair} value={pair}>{pair}</option>
              ))}
            </select>
          </div>
        </div>

        {crossExchangeData && !loading && (
          <CrossExchangeChart data={crossExchangeData} />
        )}
      </div>

      {/* Exchange Comparison Table */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Exchange Comparison
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead className="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Exchange
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Uptime
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Avg Latency
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Success Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Total Requests
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              {exchanges.map((exchange) => {
                const successRate = exchange.requestsTotal > 0 
                  ? ((exchange.requestsSuccess / exchange.requestsTotal) * 100).toFixed(1)
                  : '0.0';
                
                return (
                  <tr key={exchange.exchange}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-white capitalize">
                      {exchange.exchange}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full
                        ${exchange.status === 'healthy' ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' :
                          exchange.status === 'degraded' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' :
                          'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'}`}>
                        {exchange.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {exchange.uptime.toFixed(1)}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {exchange.avgLatency.toFixed(0)}ms
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {successRate}%
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-white">
                      {exchange.requestsTotal.toLocaleString()}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default MarketplaceStatus;