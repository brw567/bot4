import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
# Database imports moved to db_adapter
from config import DB_PATH
from strategies.base_strategy import BaseStrategy
from utils.binance_utils import get_binance_client
from utils.db_adapter import get_db_connection, get_db_cursor, USE_POSTGRES

handler = RotatingFileHandler('bot.log', maxBytes=1_000_000, backupCount=5)
logging.basicConfig(level=logging.INFO, handlers=[handler],
                    format='%(asctime)s - %(message)s')

class GridStrategy(BaseStrategy):
    """
    Grid trading strategy placing buy/sell orders around VWAP for scalping.
    Inherits BaseStrategy for risk management (position sizing, SL/TP).

    Note: VWAP centering reduces asymmetry in volatile markets; 10 levels and 1% spacing
    chosen for scalping profitability (0.3-0.5% per trade).
    """
    def __init__(self, levels=10, spacing=0.01, **kwargs):
        """
        Initialize grid strategy with trading parameters.

        Args:
            levels (int): Number of grid levels on each side (default 10).
            spacing (float): Percentage spacing between levels (default 1%).
            **kwargs: Passed to BaseStrategy (e.g., capital, risk_per_trade).
        """
        super().__init__(**kwargs)
        self.levels = levels
        self.spacing = spacing

    def generate_signal(self, data: pd.DataFrame) -> str:
        """Return a signal based on price vs. rolling VWAP."""
        try:
            if len(data) < 20:
                return 'hold'
            typ = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typ * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
            price = data['close'].iloc[-1]
            if price < vwap.iloc[-1] * (1 - self.spacing):
                return 'buy'
            if price > vwap.iloc[-1] * (1 + self.spacing):
                return 'sell'
            return 'hold'
        except Exception:
            return 'hold'

    def run(self, symbol, current_price):
        """
        Execute grid trading by placing buy/sell orders around VWAP.

        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            current_price (float): Current market price.

        Note: VWAP calculated from 1hr 1m OHLCV to center grid; uses dynamic import
        of execute_trade to avoid circular dependencies.
        """
        try:
            # Calculate VWAP from last 1hr 1m data
            client = get_binance_client()
            ohlcv = client.fetch_ohlcv(symbol, '1m', limit=60)
            df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
            typ = (df['h'] + df['l'] + df['c']) / 3
            # VWAP over the entire hour: sum(typical*vol) / sum(vol)
            vwap = (typ * df['v']).sum() / df['v'].sum()
            center = vwap
            logging.info(f"VWAP calculated for {symbol}: {center:.2f}")

            # Risk check via Grok
            vol = 0.02  # Mock volatility; replace with statsmodels in prod
            sl, tp = self.get_dynamic_sl_tp(symbol, center, vol)
            if not sl or not tp:
                logging.info(f"Grid skipped for {symbol}: Invalid SL/TP")
                return

            # Position sizing
            size = self.calculate_position_size(center, (center - sl) / center, vol=vol)
            if size <= 0:
                logging.info(f"Grid skipped for {symbol}: Invalid position size")
                return

            # Place grid orders
            for i in range(1, self.levels + 1):
                buy_price = center * (1 - i * self.spacing)
                sell_price = center * (1 + i * self.spacing)
                try:
                    from utils.binance_utils import execute_trade
                    execute_trade(symbol, 'buy', size / self.levels, price=buy_price)
                    execute_trade(symbol, 'sell', size / self.levels, price=sell_price)
                    logging.info(f"Grid order placed: {symbol}, buy={buy_price:.2f}, sell={sell_price:.2f}, size={size / self.levels:.6f}")
                except Exception as e:
                    logging.error(f"Grid order failed for {symbol}: {e}")

            # Log trade to DB (mock profit for simplicity)
            conn = get_db_connection()
            conn.execute("INSERT INTO trades (symbol, profit, timestamp) VALUES (?, ?, ?)",
                         (symbol, 0.0, pd.Timestamp.now()))  # Mock profit
            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Grid run failed for {symbol}: {e}")