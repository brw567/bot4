interface SignalStrengthProps {
  signals: {
    technical: number;
    sentiment: number;
    ml_prediction: number;
    volume: number;
    overall: number;
  };
}

export default function SignalStrength({ signals }: SignalStrengthProps) {
  const getColorClass = (value: number) => {
    if (value >= 70) return 'bg-green-500';
    if (value >= 50) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="p-4 space-y-3">
      <h3 className="text-lg font-semibold">Signal Strength</h3>
      
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span>Overall</span>
          <div className="flex items-center space-x-2">
            <div className="w-32 bg-gray-200 rounded-full h-2.5">
              <div 
                className={`h-2.5 rounded-full ${getColorClass(signals.overall)}`}
                style={{ width: `${signals.overall}%` }}
              ></div>
            </div>
            <span className="text-sm w-10">{signals.overall}%</span>
          </div>
        </div>

        <div className="text-sm space-y-1">
          <div className="flex justify-between">
            <span>Technical</span>
            <span>{signals.technical}%</span>
          </div>
          <div className="flex justify-between">
            <span>Sentiment</span>
            <span>{signals.sentiment}%</span>
          </div>
          <div className="flex justify-between">
            <span>ML Prediction</span>
            <span>{signals.ml_prediction}%</span>
          </div>
          <div className="flex justify-between">
            <span>Volume</span>
            <span>{signals.volume}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
