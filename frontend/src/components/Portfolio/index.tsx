import React from 'react';
import BalanceDisplay from './BalanceDisplay';

const Portfolio: React.FC = () => {
  return (
    <div className="space-y-6">
      <BalanceDisplay />
    </div>
  );
};

export default Portfolio;