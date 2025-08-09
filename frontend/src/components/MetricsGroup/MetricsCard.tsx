interface MetricsCardProps {
  title: string;
  value: string | number;
  change?: number;
}

export default function MetricsCard({ title, value, change }: MetricsCardProps) {
  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className="text-2xl font-semibold">{value}</p>
      {change !== undefined && (
        <p className={`text-sm ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {change >= 0 ? '+' : ''}{change}%
        </p>
      )}
    </div>
  );
}
