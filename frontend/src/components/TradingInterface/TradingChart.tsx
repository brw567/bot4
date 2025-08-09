interface ChartData {
  time: string;
  price: number;
  volume: number;
  buyVolume?: number;
  sellVolume?: number;
  signal?: number;
}

interface TradingChartProps {
  data?: ChartData[];
  height?: number;
}

export default function TradingChart({ data = [], height = 400 }: TradingChartProps) {
  return (
    <div className="p-4" style={{ height }}>
      <h3 className="text-lg font-semibold mb-4">Trading Chart</h3>
      <div className="border rounded p-4 text-center text-gray-500">
        {data.length > 0 ? `Chart with ${data.length} data points` : 'No data available'}
      </div>
    </div>
  );
}
