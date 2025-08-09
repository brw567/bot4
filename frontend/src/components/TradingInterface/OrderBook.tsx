interface OrderBookProps {
  bids?: Array<{ price: number; amount: number; total: number }>;
  asks?: Array<{ price: number; amount: number; total: number }>;
  spread?: number;
}

export default function OrderBook({ bids = [], asks = [], spread = 0 }: OrderBookProps) {
  return (
    <div className="p-4">
      <h3 className="text-lg font-semibold mb-4">Order Book</h3>
      <div className="space-y-2">
        <div>Spread: {spread.toFixed(2)}%</div>
        <div>Bids: {bids.length}</div>
        <div>Asks: {asks.length}</div>
      </div>
    </div>
  );
}
