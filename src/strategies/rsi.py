from .base_strategy import BaseStrategy
import pandas as pd

class RSIStrategy(BaseStrategy):
    """Relative Strength Index strategy."""
    def __init__(self, overbought=70, oversold=30, period=14, **kwargs):
        super().__init__(**kwargs)
        self.overbought = overbought
        self.oversold = oversold
        self.period = period

    def generate_signal(self, data: pd.DataFrame) -> str:
        if len(data) < self.period:
            return 'hold'
        rsi = data['close'].rolling(self.period).apply(lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).sum() / (-x.diff().clip(upper=0).sum())))))
        value = rsi.iloc[-1]
        if value > self.overbought:
            return 'sell'
        if value < self.oversold:
            return 'buy'
        return 'hold'
