import logging
from logging.handlers import RotatingFileHandler
import sqlite3
import pandas as pd
from config import DB_PATH
from strategies.base_strategy import BaseStrategy
from utils.ccxt_utils import calculate_spread  # No circular: dynamic imports in ccxt_utils
try:
    from core.grok_api import get_grok_api
    get_risk_assessment = get_grok_api().get_risk_assessment
except ImportError:
    from utils.grok_utils import get_risk_assessment
from utils.binance_utils import get_binance_client
from utils.ml_utils import get_ml_predictor

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

class ArbitrageStrategy(BaseStrategy):
    """
    Arbitrage strategy for spot-futures price differences.
    Executes buy/sell if spread exceeds threshold after fees/slippage.

    Inherits BaseStrategy for risk management (position sizing, SL/TP).
    Note: Fixed threshold (0.002) balances profitability; configurable if needed.
    """
    def __init__(self, threshold=0.002, **kwargs):
        """
        Initialize arbitrage strategy.

        Args:
            threshold (float): Minimum spread to trigger trade (default 0.2%).
            **kwargs: Passed to BaseStrategy (e.g., capital, risk_per_trade).
        """
        super().__init__(**kwargs)
        self.threshold = threshold

    def generate_signal(self, spread: float) -> str:
        """Return 'buy' if spread exceeds threshold."""
        try:
            return 'buy' if spread > self.threshold else 'hold'
        except Exception:
            return 'hold'

    def run(self, symbol_spot, symbol_futures):
        """
        Execute arbitrage if spread exceeds threshold, risk is approved, and ML probability > 0.8.

        Args:
            symbol_spot (str): Spot pair (e.g., 'BTC/USDT').
            symbol_futures (str): Futures pair (e.g., 'BTC/USDT:USDT').

        Note: Enhanced with ML arbitrage predictor for 80% win rate target.
        """
        try:
            spread = calculate_spread(symbol_spot, symbol_futures)
            client = get_binance_client()  # Dynamic import from binance_utils
            ticker = client.fetch_ticker(symbol_spot)
            price = ticker['last']
            vol = (ticker['high'] - ticker['low']) / ticker['low']  # Simple volatility

            # ML prediction check - Phase 3 enhancement
            ml_predictor = get_ml_predictor()
            should_trade_ml, ml_probability, ml_details = ml_predictor.should_trade(symbol_spot, symbol_futures)
            
            if not should_trade_ml:
                logging.info(f"Arbitrage skipped for {symbol_spot}/{symbol_futures}: ML probability {ml_probability:.3f} < {ml_predictor.probability_threshold}")
                return

            # Risk check via Grok
            risk = get_risk_assessment(symbol_spot, price, vol, 0.80)  # Updated to 80% win rate
            if risk.trade != 'yes':
                logging.info(f"Arbitrage skipped for {symbol_spot}: Risk not approved")
                return

            # Position sizing
            sl, tp = self.get_dynamic_sl_tp(symbol_spot, price, vol)
            if not sl or not tp:
                logging.info(f"Arbitrage skipped for {symbol_spot}: Invalid SL/TP")
                return
            size = self.calculate_position_size(price, (price - sl) / price, vol=vol)

            # Execute if spread exceeds threshold AND ML approval
            if spread > self.threshold:
                from utils.binance_utils import execute_trade  # Dynamic import
                execute_trade(symbol_spot, 'buy', size)
                execute_trade(symbol_futures, 'sell', size)
                
                # Enhanced logging with ML details
                logging.info(f"ML-Enhanced Arbitrage executed:")
                logging.info(f"  Pair: {symbol_spot}/{symbol_futures}")
                logging.info(f"  Spread: {spread:.4f}")
                logging.info(f"  Size: {size:.6f}")
                logging.info(f"  ML Probability: {ml_probability:.3f}")
                logging.info(f"  ML Features: {ml_details['features']}")
                
                # Enhanced trade logging to DB with ML data
                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    INSERT INTO trades (symbol, profit, timestamp, strategy, spread, 
                                      ml_probability, volume_spot, volume_futures, volatility, 
                                      momentum, funding_rate, open_interest) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol_spot, 
                    spread * size * price,  # Estimated profit
                    pd.Timestamp.now(),
                    'arbitrage',
                    spread,
                    ml_probability,
                    ml_details['features'].get('volume_ratio', 1.0) * ticker.get('quoteVolume', 0),  # Spot volume
                    ticker.get('quoteVolume', 0),  # Futures volume (approximation)
                    ml_details['features'].get('volatility', vol),
                    ml_details['features'].get('momentum_1h', 0.0),
                    ml_details['features'].get('funding_rate', 0.0),
                    ml_details['features'].get('open_interest_change', 0.0)
                ))
                conn.commit()
                conn.close()
                
            else:
                logging.info(f"Arbitrage spread {spread:.4f} below threshold {self.threshold:.4f} for {symbol_spot}")
                
        except Exception as e:
            logging.error(f"ML-Enhanced Arbitrage run failed for {symbol_spot}: {e}")
    
    def get_ml_status(self):
        """Get ML predictor status and performance metrics."""
        try:
            ml_predictor = get_ml_predictor()
            return ml_predictor.get_model_status()
        except Exception as e:
            logging.error(f"Failed to get ML status: {e}")
            return {'error': str(e)}
    
    def schedule_ml_retraining(self):
        """Schedule ML model retraining if needed."""
        try:
            ml_predictor = get_ml_predictor()
            success = ml_predictor.schedule_retraining()
            if success:
                logging.info("ML model retraining check completed successfully")
            else:
                logging.warning("ML model retraining failed")
            return success
        except Exception as e:
            logging.error(f"ML retraining scheduling failed: {e}")
            return False