import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface CrossExchangeChartProps {
  data: {
    symbol: string;
    exchanges_analyzed: number;
    price_range: {
      min: number;
      max: number;
      mean: number;
      std: number;
    };
    volume_distribution: {
      total: number;
      by_exchange: Record<string, number>;
    };
  };
}

const CrossExchangeChart: React.FC<CrossExchangeChartProps> = ({ data }) => {
  const exchanges = Object.keys(data.volume_distribution.by_exchange);
  const volumes = Object.values(data.volume_distribution.by_exchange);

  const chartData = {
    labels: exchanges.map(e => e.charAt(0).toUpperCase() + e.slice(1)),
    datasets: [
      {
        label: 'Trading Volume',
        data: volumes,
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: `Volume Distribution for ${data.symbol}`,
        color: '#9CA3AF',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const value = context.parsed.y;
            return `Volume: $${value.toLocaleString()}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(tickValue) {
            const value = tickValue as number;
            return '$' + (value / 1000000).toFixed(1) + 'M';
          },
          color: '#9CA3AF',
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.2)',
        },
      },
      x: {
        ticks: {
          color: '#9CA3AF',
        },
        grid: {
          display: false,
        },
      },
    },
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
          Cross-Exchange Analysis
        </h3>
        <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <p className="text-gray-600 dark:text-gray-400">Price Range</p>
            <p className="font-medium text-gray-900 dark:text-white">
              ${data.price_range.min.toFixed(2)} - ${data.price_range.max.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">Mean Price</p>
            <p className="font-medium text-gray-900 dark:text-white">
              ${data.price_range.mean.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">Price Std Dev</p>
            <p className="font-medium text-gray-900 dark:text-white">
              ${data.price_range.std.toFixed(2)}
            </p>
          </div>
          <div>
            <p className="text-gray-600 dark:text-gray-400">Total Volume</p>
            <p className="font-medium text-gray-900 dark:text-white">
              ${(data.volume_distribution.total / 1000000).toFixed(1)}M
            </p>
          </div>
        </div>
      </div>

      <div className="h-64">
        <Bar data={chartData} options={options} />
      </div>
    </div>
  );
};

export default CrossExchangeChart;