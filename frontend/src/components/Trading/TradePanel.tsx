import React, { useState } from 'react';
import { useAppSelector } from '../../hooks/redux';
import { CalculatorIcon, ChartBarIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

interface TradePanelProps {
  pair: string;
  currentPrice: number;
  onPlaceOrder?: (order: any) => void;
}

type OrderType = 'market' | 'limit' | 'stop';
type OrderSide = 'buy' | 'sell';

const TradePanel: React.FC<TradePanelProps> = ({ pair, currentPrice, onPlaceOrder }) => {
  const [orderSide, setOrderSide] = useState<OrderSide>('buy');
  const [orderType, setOrderType] = useState<OrderType>('market');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState(currentPrice.toString());
  const [stopPrice, setStopPrice] = useState('');
  const [usePercentage, setUsePercentage] = useState(false);
  const [percentage, setPercentage] = useState(25);

  const balance = 10000; // Mock balance

  const calculateTotal = () => {
    const qty = parseFloat(quantity) || 0;
    const prc = orderType === 'market' ? currentPrice : (parseFloat(price) || 0);
    return qty * prc;
  };

  const handlePercentageClick = (pct: number) => {
    setPercentage(pct);
    if (orderType === 'market') {
      const amount = (balance * pct) / 100;
      const qty = amount / currentPrice;
      setQuantity(qty.toFixed(6));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!quantity || parseFloat(quantity) <= 0) {
      toast.error('Please enter a valid quantity');
      return;
    }

    if (orderType === 'limit' && (!price || parseFloat(price) <= 0)) {
      toast.error('Please enter a valid limit price');
      return;
    }

    if (orderType === 'stop' && (!stopPrice || parseFloat(stopPrice) <= 0)) {
      toast.error('Please enter a valid stop price');
      return;
    }

    const order = {
      pair,
      side: orderSide,
      type: orderType,
      quantity: parseFloat(quantity),
      price: orderType === 'market' ? currentPrice : parseFloat(price),
      stopPrice: orderType === 'stop' ? parseFloat(stopPrice) : undefined,
      total: calculateTotal(),
      timestamp: new Date().toISOString(),
    };

    toast.success(`${orderSide.toUpperCase()} order placed successfully`);
    onPlaceOrder?.(order);
    
    // Reset form
    setQuantity('');
    setPrice(currentPrice.toString());
    setStopPrice('');
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">
          Place Order
        </h3>
      </div>

      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        {/* Order Side Tabs */}
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setOrderSide('buy')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${
              orderSide === 'buy'
                ? 'bg-green-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Buy
          </button>
          <button
            type="button"
            onClick={() => setOrderSide('sell')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-colors ${
              orderSide === 'sell'
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
            }`}
          >
            Sell
          </button>
        </div>

        {/* Order Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Order Type
          </label>
          <select
            value={orderType}
            onChange={(e) => setOrderType(e.target.value as OrderType)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          >
            <option value="market">Market</option>
            <option value="limit">Limit</option>
            <option value="stop">Stop</option>
          </select>
        </div>

        {/* Price Input (for limit/stop orders) */}
        {orderType !== 'market' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              {orderType === 'limit' ? 'Limit Price' : 'Trigger Price'}
            </label>
            <div className="relative">
              <input
                type="number"
                value={orderType === 'limit' ? price : stopPrice}
                onChange={(e) => orderType === 'limit' ? setPrice(e.target.value) : setStopPrice(e.target.value)}
                step="0.01"
                className="w-full px-3 py-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                placeholder="0.00"
              />
              <span className="absolute right-3 top-2 text-gray-500 dark:text-gray-400">
                USD
              </span>
            </div>
          </div>
        )}

        {/* Quantity Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Quantity
          </label>
          <div className="relative">
            <input
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              step="0.000001"
              className="w-full px-3 py-2 border border-gray-300 rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
              placeholder="0.000000"
            />
            <span className="absolute right-3 top-2 text-gray-500 dark:text-gray-400">
              {pair.split('/')[0]}
            </span>
          </div>
        </div>

        {/* Quick Amount Buttons */}
        <div className="flex space-x-2">
          {[25, 50, 75, 100].map(pct => (
            <button
              key={pct}
              type="button"
              onClick={() => handlePercentageClick(pct)}
              className="flex-1 py-1 px-2 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
            >
              {pct}%
            </button>
          ))}
        </div>

        {/* Order Summary */}
        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-500 dark:text-gray-400">Available Balance</span>
            <span className="font-medium text-gray-900 dark:text-white">
              ${balance.toFixed(2)} USD
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-500 dark:text-gray-400">Order Value</span>
            <span className="font-medium text-gray-900 dark:text-white">
              ${calculateTotal().toFixed(2)} USD
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-500 dark:text-gray-400">Fee (0.1%)</span>
            <span className="font-medium text-gray-900 dark:text-white">
              ${(calculateTotal() * 0.001).toFixed(2)} USD
            </span>
          </div>
          <div className="pt-2 border-t border-gray-200 dark:border-gray-600">
            <div className="flex justify-between text-sm">
              <span className="text-gray-700 dark:text-gray-300 font-medium">Total</span>
              <span className="font-bold text-gray-900 dark:text-white">
                ${(calculateTotal() * 1.001).toFixed(2)} USD
              </span>
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          className={`w-full py-3 px-4 rounded-md font-medium text-white transition-colors ${
            orderSide === 'buy'
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-red-600 hover:bg-red-700'
          }`}
        >
          {orderSide === 'buy' ? 'Buy' : 'Sell'} {pair.split('/')[0]}
        </button>

        {/* Additional Options */}
        <div className="flex items-center justify-between text-sm">
          <button
            type="button"
            className="flex items-center text-blue-600 hover:text-blue-700 dark:text-blue-400"
          >
            <CalculatorIcon className="h-4 w-4 mr-1" />
            Calculator
          </button>
          <button
            type="button"
            className="flex items-center text-blue-600 hover:text-blue-700 dark:text-blue-400"
          >
            <ChartBarIcon className="h-4 w-4 mr-1" />
            Strategy
          </button>
        </div>
      </form>
    </div>
  );
};

export default TradePanel;