from .base_strategy import BaseStrategy
import pandas as pd

class EMAStrategy(BaseStrategy):
    """Simple EMA crossover strategy."""
    def __init__(self, short_period=12, long_period=26, **kwargs):
        super().__init__(**kwargs)
        self.short_period = short_period
        self.long_period = long_period

    def generate_signal(self, data: pd.DataFrame) -> str:
        alpha_short = 2 / (self.short_period + 1)
        alpha_long = 2 / (self.long_period + 1)
        data = data.copy()
        data['ema_short'] = data['close'].ewm(alpha=alpha_short, adjust=False).mean()
        data['ema_long'] = data['close'].ewm(alpha=alpha_long, adjust=False).mean()
        if len(data) < 2:
            return 'hold'
        if data['ema_short'].iloc[-1] > data['ema_long'].iloc[-1] and data['ema_short'].iloc[-2] <= data['ema_long'].iloc[-2]:
            return 'buy'
        if data['ema_short'].iloc[-1] < data['ema_long'].iloc[-1] and data['ema_short'].iloc[-2] >= data['ema_long'].iloc[-2]:
            return 'sell'
        return 'hold'
