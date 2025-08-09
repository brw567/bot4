import React, { useState, useEffect } from 'react';
import { Switch } from '@headlessui/react';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import apiService from '../../services/api';

interface TestNetToggleProps {
  className?: string;
}

const TestNetToggle: React.FC<TestNetToggleProps> = ({ className = '' }) => {
  const [testNetEnabled, setTestNetEnabled] = useState(false);
  const [loading, setLoading] = useState(false);
  const [botRunning, setBotRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch current TestNet status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const status = await apiService.getTestNetStatus();
        setTestNetEnabled(status.testnet_enabled);
        setBotRunning(status.bot_running);
      } catch (error) {
        console.error('Failed to fetch TestNet status:', error);
        setError('Failed to fetch TestNet status');
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const handleToggle = async () => {
    setLoading(true);
    setError(null);

    try {
      const newState = !testNetEnabled;
      const result = await apiService.toggleTestNetMode(newState);
      
      if (result.success) {
        setTestNetEnabled(result.testnet_enabled);
        setBotRunning(result.bot_running);
      } else {
        setError(result.error || 'Failed to toggle TestNet mode');
      }
    } catch (error: any) {
      console.error('Failed to toggle TestNet mode:', error);
      setError(error.message || 'Failed to toggle TestNet mode');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-4 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-900 dark:text-white">
            TestNet Mode
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {testNetEnabled 
              ? 'Trading on Binance TestNet (paper trading)'
              : 'Trading on live exchange (real money)'}
          </p>
        </div>
        
        <Switch
          checked={testNetEnabled}
          onChange={handleToggle}
          disabled={loading}
          className={`
            ${testNetEnabled ? 'bg-primary-600' : 'bg-gray-200 dark:bg-gray-700'}
            relative inline-flex h-6 w-11 items-center rounded-full
            transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
            ${loading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          <span
            className={`
              ${testNetEnabled ? 'translate-x-6' : 'translate-x-1'}
              inline-block h-4 w-4 transform rounded-full bg-white transition-transform
            `}
          />
        </Switch>
      </div>

      {/* Warning if bot is running */}
      {botRunning && (
        <div className="mt-3 flex items-start space-x-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
          <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600 dark:text-yellow-500 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-yellow-800 dark:text-yellow-300 font-medium">
              Bot is currently running
            </p>
            <p className="text-xs text-yellow-700 dark:text-yellow-400 mt-1">
              Restart the bot for TestNet mode changes to take effect
            </p>
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
          <p className="text-sm text-red-800 dark:text-red-300">
            {error}
          </p>
        </div>
      )}

      {/* Status information */}
      <div className="mt-4 text-xs text-gray-500 dark:text-gray-400">
        <div className="flex items-center justify-between">
          <span>Status:</span>
          <span className={`font-medium ${testNetEnabled ? 'text-blue-600 dark:text-blue-400' : 'text-green-600 dark:text-green-400'}`}>
            {testNetEnabled ? 'TestNet' : 'Live Trading'}
          </span>
        </div>
        {testNetEnabled && (
          <div className="mt-1">
            <span className="text-gray-400">TestNet allows risk-free practice trading</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default TestNetToggle;