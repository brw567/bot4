from .base_strategy import BaseStrategy
import pandas as pd

class FVGStrategy(BaseStrategy):
    """Fair value gap detection."""
    def generate_signal(self, data: pd.DataFrame) -> str:
        if len(data) < 3:
            return 'hold'
        last = data.iloc[-1]
        prev = data.iloc[-2]
        gap_up = prev['high'] < data.iloc[-3]['low']
        gap_down = prev['low'] > data.iloc[-3]['high']
        if gap_up and last['close'] > prev['high']:
            return 'buy'
        if gap_down and last['close'] < prev['low']:
            return 'sell'
        return 'hold'
