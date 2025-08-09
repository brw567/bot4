from .base_strategy import BaseStrategy
import pandas as pd

class BollingerHeatmapStrategy(BaseStrategy):
    """Bollinger Bands with simple volatility measure."""
    def __init__(self, period=20, mult=2, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.mult = mult

    def generate_signal(self, data: pd.DataFrame) -> str:
        if len(data) < self.period:
            return 'hold'
        data = data.copy()
        ma = data['close'].rolling(self.period).mean()
        std = data['close'].rolling(self.period).std()
        upper = ma + std * self.mult
        lower = ma - std * self.mult
        price = data['close'].iloc[-1]
        if price > upper.iloc[-1]:
            return 'sell'
        if price < lower.iloc[-1]:
            return 'buy'
        return 'hold'
