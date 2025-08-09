#!/bin/bash

echo "Fixing OrderBook test..."

# Fix the OrderBook test to use proper props
cat > src/components/TradingInterface/__tests__/OrderBook.test.tsx << 'EOF'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import OrderBook from '../OrderBook';

describe('OrderBook', () => {
  const mockOrderBook = {
    bids: [
      { price: 45000, size: 1.5, total: 67500 },
      { price: 44999, size: 2.0, total: 89998 },
      { price: 44998, size: 0.5, total: 22499 },
    ],
    asks: [
      { price: 45001, size: 1.0, total: 45001 },
      { price: 45002, size: 1.2, total: 54002.4 },
      { price: 45003, size: 0.8, total: 36002.4 },
    ],
    spread: 1,
    spreadPercentage: 0.0022,
  };

  it('renders order book with bids and asks', () => {
    render(<OrderBook bids={mockOrderBook.bids} asks={mockOrderBook.asks} spread={mockOrderBook.spread} />);

    expect(screen.getByText('Order Book')).toBeInTheDocument();
    expect(screen.getByText(/Spread:/)).toBeInTheDocument();
  });

  it('displays bid information', () => {
    render(<OrderBook bids={mockOrderBook.bids} asks={mockOrderBook.asks} spread={mockOrderBook.spread} />);
    
    expect(screen.getByText('Bids: 3')).toBeInTheDocument();
  });

  it('displays ask information', () => {
    render(<OrderBook bids={mockOrderBook.bids} asks={mockOrderBook.asks} spread={mockOrderBook.spread} />);
    
    expect(screen.getByText('Asks: 3')).toBeInTheDocument();
  });

  it('displays spread correctly', () => {
    render(<OrderBook bids={mockOrderBook.bids} asks={mockOrderBook.asks} spread={mockOrderBook.spread} />);
    
    expect(screen.getByText('Spread: 0.00%')).toBeInTheDocument();
  });

  it('handles empty order book', () => {
    render(<OrderBook bids={[]} asks={[]} spread={0} />);
    
    expect(screen.getByText('Order Book')).toBeInTheDocument();
    expect(screen.getByText('Bids: 0')).toBeInTheDocument();
    expect(screen.getByText('Asks: 0')).toBeInTheDocument();
  });

  it('calculates totals correctly', () => {
    const customBook = {
      bids: [
        { price: 100, size: 1, total: 100 },
        { price: 99, size: 2, total: 198 },
      ],
      asks: [
        { price: 101, size: 1, total: 101 },
        { price: 102, size: 2, total: 204 },
      ],
      spread: 1,
      spreadPercentage: 1,
    };

    render(<OrderBook bids={customBook.bids} asks={customBook.asks} spread={customBook.spread} />);
    
    expect(screen.getByText('Order Book')).toBeInTheDocument();
  });

  it('updates when props change', () => {
    const { rerender } = render(<OrderBook bids={mockOrderBook.bids} asks={mockOrderBook.asks} spread={mockOrderBook.spread} />);
    
    expect(screen.getByText('Bids: 3')).toBeInTheDocument();

    const newBids = [{ price: 46000, size: 5, total: 230000 }];
    rerender(<OrderBook bids={newBids} asks={mockOrderBook.asks} spread={2} />);
    
    expect(screen.getByText('Bids: 1')).toBeInTheDocument();
  });
});
EOF

echo "OrderBook test fixed!"