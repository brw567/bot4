import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
} from 'recharts';
import { ArrowTrendingUpIcon, ArrowTrendingDownIcon, MinusIcon } from '@heroicons/react/24/solid';

interface FeatureImportanceProps {
  features: {
    [key: string]: {
      importance: number;
      trend: 'up' | 'down' | 'stable';
      correlation: number;
    };
  };
}

const FeatureImportance: React.FC<FeatureImportanceProps> = ({ features }) => {
  const featureData = Object.entries(features).map(([name, data]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    importance: data.importance * 100,
    correlation: data.correlation,
    trend: data.trend,
    fullName: name,
  })).sort((a, b) => b.importance - a.importance);

  const pieData = featureData.map((f, index) => ({
    name: f.name,
    value: f.importance,
    color: ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6', '#EF4444', '#6B7280'][index],
  }));

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up':
        return <ArrowTrendingUpIcon className="h-4 w-4 text-green-600" />;
      case 'down':
        return <ArrowTrendingDownIcon className="h-4 w-4 text-red-600" />;
      default:
        return <MinusIcon className="h-4 w-4 text-gray-600" />;
    }
  };

  const featureDescriptions: { [key: string]: string } = {
    volume: 'Trading volume ratio between spot and futures markets',
    book_imbalance: 'Order book bid/ask imbalance indicator',
    momentum: 'Price momentum over multiple timeframes',
    open_interest: 'Futures open interest changes',
    funding_rate: 'Current funding rate for perpetual futures',
    volatility: 'Recent price volatility measurement',
  };

  const getCorrelationColor = (correlation: number) => {
    if (correlation >= 0.7) return 'text-green-600 bg-green-100 dark:bg-green-900';
    if (correlation >= 0.5) return 'text-blue-600 bg-blue-100 dark:bg-blue-900';
    if (correlation >= 0.3) return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900';
    return 'text-gray-600 bg-gray-100 dark:bg-gray-900';
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Importance Bar Chart */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={featureData} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis type="number" tick={{ fill: '#9CA3AF' }} domain={[0, 30]} />
              <YAxis type="category" dataKey="name" tick={{ fill: '#9CA3AF' }} width={100} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.375rem',
                }}
                labelStyle={{ color: '#9CA3AF' }}
                formatter={(value: any) => `${value.toFixed(1)}%`}
              />
              <Bar dataKey="importance" fill="#3B82F6">
                {featureData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={pieData[index].color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Importance Pie Chart */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Importance Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={(entry) => `${entry.name}: ${entry.value.toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.375rem',
                }}
                labelStyle={{ color: '#9CA3AF' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Feature Details */}
      <div className="metric-card">
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Feature Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(features).map(([name, data]) => (
            <div key={name} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h4 className="font-medium text-gray-900 dark:text-white capitalize">
                    {name.replace(/_/g, ' ')}
                  </h4>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {featureDescriptions[name]}
                  </p>
                </div>
                {getTrendIcon(data.trend)}
              </div>
              
              <div className="grid grid-cols-3 gap-2 mt-3">
                <div className="text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Importance</p>
                  <p className="text-lg font-semibold text-gray-900 dark:text-white">
                    {(data.importance * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Correlation</p>
                  <p className={`text-lg font-semibold px-2 py-1 rounded ${getCorrelationColor(data.correlation)}`}>
                    {data.correlation.toFixed(2)}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-gray-500 dark:text-gray-400">Trend</p>
                  <p className={`text-lg font-semibold ${
                    data.trend === 'up' ? 'text-green-600' :
                    data.trend === 'down' ? 'text-red-600' :
                    'text-gray-600'
                  }`}>
                    {data.trend.toUpperCase()}
                  </p>
                </div>
              </div>
              
              {/* Visual importance bar */}
              <div className="mt-3">
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className="h-2 rounded-full transition-all"
                    style={{
                      width: `${data.importance * 100}%`,
                      backgroundColor: pieData.find(p => p.name.toLowerCase().includes(name.split('_')[0]))?.color || '#6B7280',
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Feature Insights */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Key Insights</h4>
          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <li className="flex items-start">
              <span className="text-primary-600 mr-2">•</span>
              Volume ratio is the strongest predictor with {(features.volume.importance * 100).toFixed(1)}% importance
            </li>
            <li className="flex items-start">
              <span className="text-primary-600 mr-2">•</span>
              Book imbalance shows high correlation ({features.book_imbalance.correlation.toFixed(2)}) with profitable trades
            </li>
            <li className="flex items-start">
              <span className="text-primary-600 mr-2">•</span>
              {Object.entries(features).filter(([_, data]) => data.trend === 'up').length} features showing upward trends
            </li>
            <li className="flex items-start">
              <span className="text-primary-600 mr-2">•</span>
              Combined top 3 features account for {
                (Object.values(features)
                  .sort((a, b) => b.importance - a.importance)
                  .slice(0, 3)
                  .reduce((sum, f) => sum + f.importance, 0) * 100
                ).toFixed(1)
              }% of model decisions
            </li>
          </ul>
        </div>

        <div className="metric-card">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Recommendations</h4>
          <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <li className="flex items-start">
              <span className="text-yellow-600 mr-2">⚡</span>
              Monitor volume spikes as they're the strongest profit indicator
            </li>
            <li className="flex items-start">
              <span className="text-yellow-600 mr-2">⚡</span>
              Consider adding more weight to book imbalance in live trading
            </li>
            <li className="flex items-start">
              <span className="text-yellow-600 mr-2">⚡</span>
              Review features with declining trends for potential model drift
            </li>
            <li className="flex items-start">
              <span className="text-yellow-600 mr-2">⚡</span>
              Implement real-time feature monitoring for anomaly detection
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default FeatureImportance;