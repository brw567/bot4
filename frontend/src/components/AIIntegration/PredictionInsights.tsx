import React, { useState } from 'react';
import { CheckCircleIcon, XCircleIcon, ClockIcon } from '@heroicons/react/24/solid';
import { motion, AnimatePresence } from 'framer-motion';

interface Prediction {
  id: string;
  timestamp: string;
  pair: string;
  strategy: string;
  confidence: number;
  predictedProfit: number;
  features: {
    volume: number;
    book_imbalance: number;
    momentum: number;
    open_interest: number;
    funding_rate: number;
    volatility: number;
  };
  outcome: 'success' | 'failure' | 'pending';
  actualProfit: number | null;
}

interface PredictionInsightsProps {
  predictions: Prediction[];
}

const PredictionInsights: React.FC<PredictionInsightsProps> = ({ predictions }) => {
  const [selectedPrediction, setSelectedPrediction] = useState<Prediction | null>(null);
  const [filterOutcome, setFilterOutcome] = useState<'all' | 'success' | 'failure' | 'pending'>('all');

  const filteredPredictions = predictions.filter(p => 
    filterOutcome === 'all' || p.outcome === filterOutcome
  );

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.8) return 'text-blue-600';
    if (confidence >= 0.7) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getConfidenceBg = (confidence: number) => {
    if (confidence >= 0.9) return 'bg-green-100 dark:bg-green-900';
    if (confidence >= 0.8) return 'bg-blue-100 dark:bg-blue-900';
    if (confidence >= 0.7) return 'bg-yellow-100 dark:bg-yellow-900';
    return 'bg-red-100 dark:bg-red-900';
  };

  const formatFeatureValue = (key: string, value: number) => {
    switch (key) {
      case 'volume':
        return value.toFixed(3);
      case 'book_imbalance':
        return value.toFixed(4);
      case 'momentum':
        return `${(value * 100).toFixed(2)}%`;
      case 'open_interest':
        return `$${(value / 1000000).toFixed(2)}M`;
      case 'funding_rate':
        return `${(value * 100).toFixed(3)}%`;
      case 'volatility':
        return `${(value * 100).toFixed(2)}%`;
      default:
        return value.toFixed(4);
    }
  };

  const featureLabels = {
    volume: 'Volume Ratio',
    book_imbalance: 'Book Imbalance',
    momentum: 'Momentum',
    open_interest: 'Open Interest',
    funding_rate: 'Funding Rate',
    volatility: 'Volatility',
  };

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Predictions</p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-white">{predictions.length}</p>
        </div>
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Success Rate</p>
          <p className="text-2xl font-semibold text-green-600">
            {((predictions.filter(p => p.outcome === 'success').length / 
              predictions.filter(p => p.outcome !== 'pending').length) * 100).toFixed(1)}%
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Avg Confidence</p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-white">
            {(predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length * 100).toFixed(1)}%
          </p>
        </div>
        <div className="metric-card">
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400">Total Profit</p>
          <p className="text-2xl font-semibold text-gray-900 dark:text-white">
            ${predictions
              .filter(p => p.actualProfit !== null)
              .reduce((sum, p) => sum + (p.actualProfit || 0), 0)
              .toFixed(2)}
          </p>
        </div>
      </div>

      {/* Filter Buttons */}
      <div className="flex space-x-2">
        {(['all', 'success', 'failure', 'pending'] as const).map(outcome => (
          <button
            key={outcome}
            onClick={() => setFilterOutcome(outcome)}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${
              filterOutcome === outcome
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            {outcome.charAt(0).toUpperCase() + outcome.slice(1)}
            {outcome !== 'all' && (
              <span className="ml-2">
                ({predictions.filter(p => p.outcome === outcome).length})
              </span>
            )}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Predictions List */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Recent Predictions</h3>
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            <AnimatePresence>
              {filteredPredictions.map((prediction) => (
                <motion.div
                  key={prediction.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className={`p-4 rounded-lg border cursor-pointer transition-all ${
                    selectedPrediction?.id === prediction.id
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                      : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                  onClick={() => setSelectedPrediction(prediction)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      {prediction.outcome === 'success' && (
                        <CheckCircleIcon className="h-5 w-5 text-green-600" />
                      )}
                      {prediction.outcome === 'failure' && (
                        <XCircleIcon className="h-5 w-5 text-red-600" />
                      )}
                      {prediction.outcome === 'pending' && (
                        <ClockIcon className="h-5 w-5 text-yellow-600" />
                      )}
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">{prediction.pair}</p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">{prediction.strategy}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`text-sm font-medium ${getConfidenceColor(prediction.confidence)}`}>
                        {(prediction.confidence * 100).toFixed(1)}% confidence
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {new Date(prediction.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-gray-400">Predicted</p>
                      <p className="font-medium text-gray-900 dark:text-white">
                        ${prediction.predictedProfit.toFixed(2)}
                      </p>
                    </div>
                    {prediction.actualProfit !== null && (
                      <div>
                        <p className="text-sm text-gray-500 dark:text-gray-400">Actual</p>
                        <p className={`font-medium ${
                          prediction.actualProfit > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          ${prediction.actualProfit.toFixed(2)}
                        </p>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Prediction Details */}
        <div className="metric-card">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Prediction Details</h3>
          {selectedPrediction ? (
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {selectedPrediction.pair} - {selectedPrediction.strategy}
                  </h4>
                  <span className={`px-3 py-1 text-sm font-medium rounded-full ${getConfidenceBg(selectedPrediction.confidence)} ${getConfidenceColor(selectedPrediction.confidence)}`}>
                    {(selectedPrediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {new Date(selectedPrediction.timestamp).toLocaleString()}
                </p>
              </div>

              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Feature Values</h5>
                <div className="space-y-3">
                  {Object.entries(selectedPrediction.features).map(([key, value]) => {
                    const normalizedValue = key === 'book_imbalance' 
                      ? (value + 1) / 2  // Normalize from [-1, 1] to [0, 1]
                      : key === 'momentum'
                      ? (value + 0.05) / 0.1  // Normalize from [-0.05, 0.05] to [0, 1]
                      : key === 'funding_rate'
                      ? (value + 0.005) / 0.01  // Normalize from [-0.005, 0.005] to [0, 1]
                      : value;  // Already normalized [0, 1]
                    
                    return (
                      <div key={key}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-600 dark:text-gray-400">
                            {featureLabels[key as keyof typeof featureLabels]}
                          </span>
                          <span className="font-medium text-gray-900 dark:text-white">
                            {formatFeatureValue(key, value)}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full transition-all"
                            style={{ width: `${Math.min(100, Math.max(0, normalizedValue * 100))}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
                <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Outcome</h5>
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <p className="text-sm text-gray-500 dark:text-gray-400">Predicted Profit</p>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white">
                      ${selectedPrediction.predictedProfit.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                    <p className="text-sm text-gray-500 dark:text-gray-400">Actual Profit</p>
                    <p className={`text-xl font-semibold ${
                      selectedPrediction.actualProfit === null
                        ? 'text-gray-400'
                        : selectedPrediction.actualProfit > 0
                        ? 'text-green-600'
                        : 'text-red-600'
                    }`}>
                      {selectedPrediction.actualProfit === null
                        ? 'Pending'
                        : `$${selectedPrediction.actualProfit.toFixed(2)}`}
                    </p>
                  </div>
                </div>
                
                {selectedPrediction.outcome !== 'pending' && (
                  <div className="mt-3 text-center">
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                      selectedPrediction.outcome === 'success'
                        ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                        : 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                    }`}>
                      {selectedPrediction.outcome === 'success' ? (
                        <>
                          <CheckCircleIcon className="h-4 w-4 mr-1" />
                          Success
                        </>
                      ) : (
                        <>
                          <XCircleIcon className="h-4 w-4 mr-1" />
                          Failed
                        </>
                      )}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <p className="text-gray-500 dark:text-gray-400">
                Select a prediction to view details
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionInsights;