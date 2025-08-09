import React from 'react';
import Portfolio from '../components/Portfolio';
import { WalletIcon } from '@heroicons/react/24/outline';

const PortfolioPage: React.FC = () => {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <div className="flex items-center">
          <WalletIcon className="h-8 w-8 text-primary-600 dark:text-primary-400 mr-3" />
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Portfolio</h1>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              View and manage your cryptocurrency holdings across all exchanges
            </p>
          </div>
        </div>
      </div>

      <Portfolio />
    </div>
  );
};

export default PortfolioPage;