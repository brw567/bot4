from .base_strategy import BaseStrategy

class MEVStrategy(BaseStrategy):
    """Simple mempool-density based MEV strategy."""

    def __init__(self, density_threshold: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.density_threshold = density_threshold

    def generate_signal(self, data: dict, params: dict | None = None) -> str:
        """Return 'buy' when mempool density exceeds threshold."""
        params = params or {}
        density = float(data.get("mempool_density", 0))
        threshold = float(params.get("density_threshold", self.density_threshold))
        return "buy" if density > threshold else "hold"

    def is_arbitrage_opportunity(self, price_diff: float, sentiment: str) -> bool:
        return price_diff > 0.001 and sentiment == "positive"

    def is_sandwich_risk(self, mempool_density: float) -> bool:
        return mempool_density > self.density_threshold
