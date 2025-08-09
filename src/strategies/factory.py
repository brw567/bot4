from .ema import EMAStrategy
from .rsi import RSIStrategy
from .fvg import FVGStrategy
from .bollinger_heatmap import BollingerHeatmapStrategy
from .triangular_arb import TriangularArbStrategy
from .delta_neutral import DeltaNeutralStrategy
from .arbitrage_strategy import ArbitrageStrategy
from .grid_strategy import GridStrategy
from .mev import MEVStrategy
from .strategy_wrapper import StrategyWrapper

class StrategyFactory:
    mapping = {
        'EMA': EMAStrategy,
        'RSI': RSIStrategy,
        'FVG': FVGStrategy,
        'BOLLINGER': BollingerHeatmapStrategy,
        'TRIANGULAR_ARB': TriangularArbStrategy,
        'DELTA_NEUTRAL': DeltaNeutralStrategy,
        'ARBITRAGE': ArbitrageStrategy,
        'GRID': GridStrategy,
        'MEV': MEVStrategy,
    }

    @classmethod
    def create(cls, name: str, **kwargs):
        klass = cls.mapping.get(name.upper())
        if not klass:
            raise ValueError(f"Unknown strategy {name}")
        strategy = klass(**kwargs)
        # Wrap the strategy to provide consistent interface
        return StrategyWrapper(strategy)
    
    @classmethod
    def get_available_strategies(cls):
        """Get list of available strategy names"""
        return list(cls.mapping.keys())
    
    @classmethod
    def create_strategy(cls, name: str, config: dict):
        """Create a strategy instance with config"""
        return cls.create(name, **config)
