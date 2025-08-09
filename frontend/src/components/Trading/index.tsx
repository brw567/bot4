import React, { useState, useEffect } from 'react';
import CandlestickChart from './CandlestickChart';
import OrderBook from './OrderBook';
import TradePanel from './TradePanel';
import RecentTrades from './RecentTrades';
import MarketplaceStatus from '../MarketplaceStatus';
import { useAppSelector } from '../../hooks/redux';
import toast from 'react-hot-toast';

interface TradingInterfaceProps {
  selectedPair?: string;
}

const TradingInterface: React.FC<TradingInterfaceProps> = ({ selectedPair = 'BTC/USDC' }) => {
  const [pair, setPair] = useState(selectedPair);
  const [timeframe, setTimeframe] = useState('1h');
  const [showIndicators, setShowIndicators] = useState(true);
  
  // Real data from API
  const [candleData, setCandleData] = useState<any[]>([]);
  const [orderBook, setOrderBook] = useState({ bids: [], asks: [] });
  const [trades, setTrades] = useState<any[]>([]);
  const [currentPrice, setCurrentPrice] = useState(0);
  const [marketData, setMarketData] = useState<any>(null);

  // Fetch real trades from API
  const fetchRealTrades = async () => {
    try {
      const response = await fetch('/api/trades?limit=50', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        if (data.trades && data.trades.length > 0) {
          setTrades(data.trades);
        }
      }
    } catch (error) {
      console.error('Failed to fetch trades:', error);
    }
  };

  useEffect(() => {
    // Fetch real trades on mount and periodically
    fetchRealTrades();
    const tradesInterval = setInterval(fetchRealTrades, 5000);
    
    return () => clearInterval(tradesInterval);
  }, []);

  // Fetch candle data
  const fetchCandleData = async () => {
    try {
      const response = await fetch(`/api/trading/candles/${pair}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        if (data.candles && data.candles.length > 0) {
          setCandleData(data.candles);
          // Set current price from last candle
          const lastCandle = data.candles[data.candles.length - 1];
          setCurrentPrice(lastCandle.close);
        }
      }
    } catch (error) {
      console.error('Failed to fetch candle data:', error);
    }
  };

  // Fetch market data
  const fetchMarketData = async () => {
    try {
      const response = await fetch(`/api/trading/market/${pair}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setMarketData(data);
        if (data.price > 0) {
          setCurrentPrice(data.price);
        }
      }
    } catch (error) {
      console.error('Failed to fetch market data:', error);
    }
  };

  // Fetch order book
  const fetchOrderBook = async () => {
    try {
      const response = await fetch(`/api/trading/orderbook/${pair}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        setOrderBook({ bids: data.bids || [], asks: data.asks || [] });
      }
    } catch (error) {
      console.error('Failed to fetch order book:', error);
    }
  };

  useEffect(() => {
    // Fetch all data when pair or timeframe changes
    fetchCandleData();
    fetchMarketData();
    fetchOrderBook();
    
    // Set up intervals for real-time updates
    const candleInterval = setInterval(fetchCandleData, 10000); // Every 10 seconds
    const marketInterval = setInterval(fetchMarketData, 2000); // Every 2 seconds
    const orderBookInterval = setInterval(fetchOrderBook, 5000); // Every 5 seconds
    
    return () => {
      clearInterval(candleInterval);
      clearInterval(marketInterval);
      clearInterval(orderBookInterval);
    };
  }, [pair, timeframe]);

  // Removed fake order book generation - will get real data from WebSocket

  const handlePlaceOrder = (order: any) => {
    // Add to trades
    const newTrade = {
      ...order,
      id: `trade-${Date.now()}`,
      status: 'pending',
    };
    setTrades([newTrade, ...trades]);
  };

  const [tradingPairs, setTradingPairs] = useState<string[]>([]);
  
  // Fetch available trading pairs from Redis
  const fetchTradingPairs = async () => {
    try {
      const response = await fetch('/api/trading/pairs', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
        },
      });
      if (response.ok) {
        const data = await response.json();
        const pairs = data.pairs || [];
        setTradingPairs(pairs);
        
        // If current pair is not in the list, set to first available pair
        if (pairs.length > 0 && !pairs.includes(pair)) {
          setPair(pairs[0]);
        }
      }
    } catch (error) {
      console.error('Failed to fetch trading pairs:', error);
      // Don't set fake pairs - show empty if no real data
      setTradingPairs([]);
    }
  };
  
  useEffect(() => {
    fetchTradingPairs();
  }, []);

  const timeframes = [
    { value: '1m', label: '1M' },
    { value: '5m', label: '5M' },
    { value: '15m', label: '15M' },
    { value: '1h', label: '1H' },
    { value: '4h', label: '4H' },
    { value: '1d', label: '1D' },
  ];

  // Real trade signals will come from the bot
  const signals: any[] = [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <select
            value={pair}
            onChange={(e) => setPair(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            {tradingPairs.map(p => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
          
          <div className="flex items-center space-x-2">
            {timeframes.map(tf => (
              <button
                key={tf.value}
                onClick={() => setTimeframe(tf.value)}
                className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                  timeframe === tf.value
                    ? 'bg-primary-600 text-white'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
          
          <button
            onClick={() => setShowIndicators(!showIndicators)}
            className={`px-4 py-2 text-sm font-medium rounded transition-colors ${
              showIndicators
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Indicators
          </button>
        </div>
        
        <div className="flex items-center space-x-4">
          <div className="text-sm text-gray-500 dark:text-gray-400">
            {timeframe === '1d' ? '24h' : timeframe} Vol: <span className="font-medium text-gray-900 dark:text-white">
              ${(candleData.reduce((sum, d) => sum + d.volume, 0) / 1000000).toFixed(2)}M
            </span>
          </div>
          <div className="text-sm text-gray-500 dark:text-gray-400">
            {timeframe === '1d' ? '24h' : timeframe} Change: <span className={`font-medium ${
              candleData.length > 0 && candleData[candleData.length - 1].close > candleData[0].close
                ? 'text-green-600' : 'text-red-600'
            }`}>
              {candleData.length > 0 
                ? ((candleData[candleData.length - 1].close - candleData[0].close) / candleData[0].close * 100).toFixed(2)
                : '0.00'}%
            </span>
          </div>
        </div>
      </div>

      {/* Main Trading Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart Area */}
        <div className="lg:col-span-2 space-y-6">
          {candleData.length === 0 && (
            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 mb-4">
              <p className="text-sm text-yellow-800 dark:text-yellow-200">
                <strong>Note:</strong> Loading chart data for {pair}...
              </p>
            </div>
          )}
          <CandlestickChart
            data={candleData}
            signals={signals}
            showIndicators={showIndicators}
            height={500}
          />
          
          <RecentTrades trades={trades} />
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          <OrderBook
            bids={orderBook.bids}
            asks={orderBook.asks}
            currentPrice={currentPrice}
            pair={pair}
          />
          
          <TradePanel
            pair={pair}
            currentPrice={currentPrice}
            onPlaceOrder={handlePlaceOrder}
          />
        </div>
      </div>

      {/* Marketplace Status Section */}
      <div className="mt-8">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Marketplace Status & Funds
        </h3>
        <MarketplaceStatus compact={true} showFunds={true} />
      </div>
    </div>
  );
};

export default TradingInterface;