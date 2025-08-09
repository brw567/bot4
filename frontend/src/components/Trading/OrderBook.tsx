import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

interface OrderLevel {
  price: number;
  quantity: number;
  total: number;
}

interface OrderBookProps {
  bids: OrderLevel[];
  asks: OrderLevel[];
  currentPrice: number;
  pair: string;
}

const OrderBook: React.FC<OrderBookProps> = ({ bids, asks, currentPrice, pair }) => {
  const maxTotal = useMemo(() => {
    const allOrders = [...bids, ...asks];
    if (allOrders.length === 0) return 0;
    
    const totals = allOrders.map(order => {
      const total = order?.total || ((order?.price || 0) * (order?.quantity || 0));
      return total;
    });
    
    return Math.max(...totals) || 0;
  }, [bids, asks]);

  const spread = useMemo(() => {
    if (asks.length > 0 && bids.length > 0) {
      return asks[0].price - bids[0].price;
    }
    return 0;
  }, [asks, bids]);

  const spreadPercentage = useMemo(() => {
    if (currentPrice > 0) {
      return (spread / currentPrice) * 100;
    }
    return 0;
  }, [spread, currentPrice]);

  const OrderRow = ({ order, type, index }: { order: OrderLevel; type: 'bid' | 'ask'; index: number }) => {
    // Ensure all values are defined with fallbacks
    const price = order?.price || 0;
    const quantity = order?.quantity || 0;
    const total = order?.total || (price * quantity);
    const depthPercentage = maxTotal > 0 ? (total / maxTotal) * 100 : 0;
    
    return (
      <motion.div
        initial={{ opacity: 0, x: type === 'bid' ? -20 : 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: index * 0.02 }}
        className="relative group"
      >
        <div className="flex justify-between items-center py-1 px-2 text-xs relative z-10">
          <span className={`font-medium ${type === 'bid' ? 'text-green-600' : 'text-red-600'}`}>
            ${price.toFixed(2)}
          </span>
          <span className="text-gray-600 dark:text-gray-400">
            {quantity.toFixed(4)}
          </span>
          <span className="text-gray-700 dark:text-gray-300 font-medium">
            ${total.toFixed(2)}
          </span>
        </div>
        <div
          className={`absolute inset-0 opacity-20 ${
            type === 'bid' ? 'bg-green-500' : 'bg-red-500'
          }`}
          style={{ width: `${depthPercentage}%` }}
        />
      </motion.div>
    );
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">
            Order Book
          </h3>
          <span className="text-sm text-gray-500 dark:text-gray-400">
            {pair}
          </span>
        </div>
      </div>

      <div className="p-4">
        {/* Header */}
        <div className="flex justify-between items-center text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">
          <span>Price (USD)</span>
          <span>Amount</span>
          <span>Total</span>
        </div>

        {/* Asks (Sells) */}
        <div className="space-y-0.5 max-h-48 overflow-y-auto">
          {asks.slice().reverse().map((ask, index) => (
            <OrderRow key={index} order={ask} type="ask" index={index} />
          ))}
        </div>

        {/* Current Price & Spread */}
        <div className="my-3 py-3 border-y border-gray-200 dark:border-gray-700">
          <div className="flex justify-between items-center">
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                ${currentPrice.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Current Price
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-gray-900 dark:text-white">
                ${spread.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Spread ({spreadPercentage.toFixed(3)}%)
              </p>
            </div>
          </div>
        </div>

        {/* Bids (Buys) */}
        <div className="space-y-0.5 max-h-48 overflow-y-auto">
          {bids.map((bid, index) => (
            <OrderRow key={index} order={bid} type="bid" index={index} />
          ))}
        </div>

        {/* Summary Stats */}
        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-500 dark:text-gray-400">Bid Volume</p>
              <p className="font-medium text-green-600">
                ${bids.reduce((sum, bid) => sum + bid.total, 0).toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-gray-500 dark:text-gray-400">Ask Volume</p>
              <p className="font-medium text-red-600">
                ${asks.reduce((sum, ask) => sum + ask.total, 0).toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OrderBook;