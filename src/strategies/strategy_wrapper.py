"""
Strategy wrapper to provide consistent interface for all strategies
"""
import pandas as pd
from typing import Dict, Any, Optional
import logging


class StrategyWrapper:
    """Wrapper to provide consistent analyze/get_signal interface for strategies"""
    
    def __init__(self, strategy):
        self.strategy = strategy
        self._last_signal = None
        
    def analyze(self, data: Dict[str, list], params: Dict[str, Any] = None) -> Optional[str]:
        """
        Analyze market data and generate trading signal
        
        Args:
            data: Dictionary with keys 'close', 'high', 'low', 'volume' containing lists
            params: Optional parameters for the strategy
            
        Returns:
            Signal: 'buy', 'sell', 'hold', or None
        """
        try:
            # Convert dict data to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
                
            # Call the appropriate method based on what the strategy has
            if hasattr(self.strategy, 'generate_signal'):
                signal = self.strategy.generate_signal(df)
            elif hasattr(self.strategy, 'get_signal'):
                signal = self.strategy.get_signal(df)
            elif hasattr(self.strategy, 'analyze'):
                signal = self.strategy.analyze(df, params or {})
            else:
                # Try to call the strategy directly if it's callable
                if callable(self.strategy):
                    signal = self.strategy(df)
                else:
                    signal = None
                    
            self._last_signal = signal
            return signal
            
        except Exception as e:
            logger.error(f"Strategy analysis error: {e}")
            return None
            
    def get_signal(self, data: Dict[str, list] = None) -> Optional[str]:
        """
        Get the last generated signal or analyze new data
        
        Args:
            data: Optional new data to analyze
            
        Returns:
            Signal: 'buy', 'sell', 'hold', or None
        """
        if data is not None:
            return self.analyze(data)
        return self._last_signal
    
    def generate_signal(self, data):
        """
        Generate trading signal - alias for analyze method
        
        Args:
            data: Market data (DataFrame or dict)
            
        Returns:
            Signal: 'buy', 'sell', 'hold', or None
        """
        return self.analyze(data)