import logging
from logging.handlers import RotatingFileHandler
try:
    from core.grok_api import get_grok_api
    get_risk_assessment = get_grok_api().get_risk_assessment
except ImportError:
    from utils.grok_utils import get_risk_assessment
from config import MAX_DEAL_PERCENT, MAX_DEAL_ABSOLUTE

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

class BaseStrategy:
    """
    Base class for trading strategies, providing shared risk management logic.
    Includes position sizing (Kelly approximation) and dynamic SL/TP via Grok.

    Note: Used by arbitrage, grid, and MEV strategies to ensure consistent risk handling.
    """
    def __init__(self, capital=10000, risk_per_trade=0.01):
        """
        Initialize base strategy with trading parameters.

        Args:
            capital (float): Initial trading capital (default 10000 USDT).
            risk_per_trade (float): Risk per trade as fraction of capital (default 1%).

        Note: Parameters can be overridden via DB (config.DEFAULT_PARAMS).
        """
        self.capital = capital
        self.risk_per_trade = risk_per_trade

    def scale_by_volatility(self, size, vol):
        """Scale position size based on market volatility."""
        try:
            factor = 1 / (1 + max(vol, 0) * 10)
            return size * factor
        except Exception as e:
            logging.error(f"Volatility scaling failed: {e}")
            return size

    def cap_position_size(self, size, price):
        """Cap position size using MAX_DEAL_PERCENT and MAX_DEAL_ABSOLUTE."""
        try:
            if size <= 0 or price <= 0:
                return 0.0

            cap = min(
                self.capital * MAX_DEAL_PERCENT / price,
                MAX_DEAL_ABSOLUTE / price,
            )
            final_size = min(size, cap)
            if final_size < size:
                logging.info(
                    f"Position size capped from {size:.6f} ({size * price:.2f} USDT) to {final_size:.6f}"
                )
            return final_size
        except Exception as e:
            logging.error(f"cap_position_size failed: {e}")
            return 0.0

    def calculate_position_size(self, price, sl_distance, winrate=0.6, vol=0.0):
        """
        Calculate position size using Kelly criterion approximation.

        Args:
            price (float): Current asset price.
            sl_distance (float): Stop-loss distance as fraction of price.
            winrate (float): Expected winrate (default 0.6 per Trader requirements).

        Returns:
            float: Position size (in units of asset).

        Note: Uses simplified R_ratio=1 for scalping; capped by
        MAX_DEAL_PERCENT and MAX_DEAL_ABSOLUTE from config.
        """
        try:
            if sl_distance <= 0 or price <= 0:
                logging.error(f"Invalid inputs for position size: price={price}, sl_distance={sl_distance}")
                return 0.0
            r_ratio = 1  # Simplified for scalping (risk/reward ratio)
            kelly = winrate - (1 - winrate) / r_ratio
            risk_amount = self.capital * self.risk_per_trade * kelly
            size = risk_amount / (price * sl_distance)
            size = self.scale_by_volatility(size, vol)

            final_size = self.cap_position_size(size, price)
            if final_size <= 0:
                logging.warning(f"Calculated position size non-positive: {final_size}")
                return 0.0
            logging.info(f"Position size calculated: {final_size:.6f} for price={price}, sl_distance={sl_distance}")
            return final_size
        except Exception as e:
            logging.error(f"Position size calculation failed: {e}")
            return 0.0

    def get_dynamic_sl_tp(self, symbol, price, vol):
        """
        Get dynamic stop-loss (SL) and take-profit (TP) using Grok risk assessment.

        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            price (float): Current asset price.
            vol (float): Volatility (e.g., (high-low)/low).

        Returns:
            tuple: (sl, tp) as floats; None if trade not approved.

        Note: Uses structured Grok prompts (from grok_utils) for consistent outputs.
        """
        try:
            risk = get_risk_assessment(symbol, price, vol, 0.65)  # Mock winrate
            if risk.trade != 'yes':
                logging.info(f"Trade not approved by Grok for {symbol}")
                return None, None
            sl = price * (1 - vol * risk.sl_mult)
            tp = price * (1 + vol * risk.tp_mult)
            logging.info(f"SL/TP for {symbol}: SL={sl:.2f}, TP={tp:.2f}")
            return sl, tp
        except Exception as e:
            logging.error(f"SL/TP calculation failed for {symbol}: {e}")
            return None, None

    def switch_strategy(self, name: str):
        """Placeholder for dynamic strategy switching."""
        logging.info(f"Switching strategy to {name}")