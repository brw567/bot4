interface Trade {
  id: string;
  time: string;
  pair: string;
  side: 'buy' | 'sell';
  price: number;
  amount: number;
  total: number;
  status: 'pending' | 'completed';
}

interface TradeHistoryProps {
  trades?: Trade[];
}

export default function TradeHistory({ trades = [] }: TradeHistoryProps) {
  return (
    <div className="p-4">
      <h3 className="text-lg font-semibold mb-4">Trade History</h3>
      <div className="text-sm text-gray-500">
        {trades.length > 0 ? `${trades.length} trades` : 'No trades yet'}
      </div>
    </div>
  );
}
