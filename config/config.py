import os
from dotenv import load_dotenv
from utils.db_utils import get_param, load_currency
try:
    from utils.security import SecureConfig, SecurityManager
    security_manager = SecurityManager()
    secure_config = SecureConfig()
except ImportError:
    # Fallback if security module not available
    security_manager = None
    secure_config = None

load_dotenv()

"""
Central configuration file for the Ultimate Crypto Scalping Bot.
Loads environment variables for secure API access and defines default parameters.
All secrets are stored in .env to prevent hardcoding.
"""

# Exchange API credentials (used by ccxt, backtrader for trading)
if secure_config:
    BINANCE_API_KEY = secure_config.get_secure_value("BINANCE_API_KEY")
    BINANCE_API_SECRET = secure_config.get_secure_value("BINANCE_API_SECRET")
else:
    BINANCE_API_KEY = get_param("BINANCE_API_KEY", os.getenv("BINANCE_API_KEY"))
    BINANCE_API_SECRET = get_param("BINANCE_API_SECRET", os.getenv("BINANCE_API_SECRET"))

# TestNet API credentials
if secure_config:
    BINANCE_TESTNET_SPOT_API_KEY = secure_config.get_secure_value("BINANCE_TESTNET_SPOT_API_KEY")
    BINANCE_TESTNET_SPOT_API_SECRET = secure_config.get_secure_value("BINANCE_TESTNET_SPOT_API_SECRET")
    BINANCE_TESTNET_FUTURES_API_KEY = secure_config.get_secure_value("BINANCE_TESTNET_FUTURES_API_KEY")
    BINANCE_TESTNET_FUTURES_API_SECRET = secure_config.get_secure_value("BINANCE_TESTNET_FUTURES_API_SECRET")
else:
    BINANCE_TESTNET_SPOT_API_KEY = get_param("BINANCE_TESTNET_SPOT_API_KEY", os.getenv("BINANCE_TESTNET_SPOT_API_KEY"))
    BINANCE_TESTNET_SPOT_API_SECRET = get_param("BINANCE_TESTNET_SPOT_API_SECRET", os.getenv("BINANCE_TESTNET_SPOT_API_SECRET"))
    BINANCE_TESTNET_FUTURES_API_KEY = get_param("BINANCE_TESTNET_FUTURES_API_KEY", os.getenv("BINANCE_TESTNET_FUTURES_API_KEY"))
    BINANCE_TESTNET_FUTURES_API_SECRET = get_param("BINANCE_TESTNET_FUTURES_API_SECRET", os.getenv("BINANCE_TESTNET_FUTURES_API_SECRET"))

# Enable Binance TestNet via CCXT sandbox when True
TEST_MODE = get_param("TEST_MODE", os.getenv("TEST_MODE", "False")).lower() == "true"
# Stage mode: Use production API but only generate test trades
STAGE_MODE = get_param("STAGE_MODE", os.getenv("STAGE_MODE", "False")).lower() == "true"
# Spot or futures trading
BINANCE_DEFAULT_TYPE = os.getenv("BINANCE_DEFAULT_TYPE", "spot")

# Telegram API credentials for notifications and sentiment analysis (telethon)
TELEGRAM_TOKEN = get_param("TELEGRAM_TOKEN", os.getenv("TELEGRAM_TOKEN"))
telegram_api_id_str = get_param("TELEGRAM_API_ID", os.getenv("TELEGRAM_API_ID", "0"))
TELEGRAM_API_ID = int(telegram_api_id_str) if telegram_api_id_str and telegram_api_id_str.isdigit() else 0
TELEGRAM_API_HASH = get_param("TELEGRAM_API_HASH", os.getenv("TELEGRAM_API_HASH"))
TELEGRAM_SESSION = get_param("TELEGRAM_SESSION", os.getenv("TELEGRAM_SESSION"))
NOTIFICATIONS_ENABLED = (
    get_param(
        "NOTIFICATIONS_ENABLED", os.getenv("NOTIFICATIONS_ENABLED", "True")
    ).lower()
    == "true"
)

# Grok API for AI-driven risk assessment and parameter tuning (requests)
GROK_API_KEY = get_param("GROK_API_KEY", os.getenv("GROK_API_KEY"))
GROK_TIMEOUT = int(get_param("GROK_TIMEOUT", os.getenv("GROK_TIMEOUT", 10)))

# Dune API for on-chain metrics (e.g., STH RPL via dune-client)
DUNE_API_KEY = get_param("DUNE_API_KEY", os.getenv("DUNE_API_KEY"))
DUNE_QUERY_ID = get_param(
    "DUNE_QUERY_ID", os.getenv("DUNE_QUERY_ID")
)  # Default query ID

# CoinGecko API for market data (optional, enhances ML features)
COINGECKO_API_KEY = get_param("COINGECKO_API_KEY", os.getenv("COINGECKO_API_KEY"))

# CryptoCompare API for historical data (required for ML training)
CRYPTOCOMPARE_API_KEY = get_param("CRYPTOCOMPARE_API_KEY", os.getenv("CRYPTOCOMPARE_API_KEY"))

# Optional per-symbol query IDs for on-chain metrics
DUNE_QUERY_ID_BTC = get_param(
    "DUNE_QUERY_ID_BTC", os.getenv("DUNE_QUERY_ID_BTC", DUNE_QUERY_ID)
)
DUNE_QUERY_ID_ETH = get_param(
    "DUNE_QUERY_ID_ETH", os.getenv("DUNE_QUERY_ID_ETH", DUNE_QUERY_ID)
)
DUNE_QUERY_ID_SOL = get_param(
    "DUNE_QUERY_ID_SOL", os.getenv("DUNE_QUERY_ID_SOL", DUNE_QUERY_ID)
)
DUNE_QUERY_IDS = {
    "BTC": DUNE_QUERY_ID_BTC,
    "ETH": DUNE_QUERY_ID_ETH,
    "SOL": DUNE_QUERY_ID_SOL,
}

# Redis configuration for pub/sub messaging (winrate, ML updates via redis)
REDIS_HOST = get_param("REDIS_HOST", os.getenv("REDIS_HOST", "localhost"))
REDIS_PORT = int(get_param("REDIS_PORT", os.getenv("REDIS_PORT", 6379)))
REDIS_DB = int(get_param("REDIS_DB", os.getenv("REDIS_DB", 0)))
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "True").lower() == "true"

# Main trading currencies for portfolio management
MAIN_CURRENCIES = os.getenv("MAIN_CURRENCIES", "USDC,BTC,ETH").split(",")
MAIN_CURRENCIES = [load_currency(c.strip()) for c in MAIN_CURRENCIES]

# Rebalancing configuration
REBALANCE_ENABLED = os.getenv("REBALANCE_ENABLED", "True").lower() == "true"
REBALANCE_THRESHOLD = float(os.getenv("REBALANCE_THRESHOLD", "5.0"))  # Percentage
REBALANCE_INTERVAL = int(os.getenv("REBALANCE_INTERVAL", "3600"))  # Seconds
REBALANCE_TARGETS = {
    "USDC": float(os.getenv("REBALANCE_TARGET_USDC", "50.0")),
    "BTC": float(os.getenv("REBALANCE_TARGET_BTC", "25.0")),
    "ETH": float(os.getenv("REBALANCE_TARGET_ETH", "25.0"))
}

# Legacy support - use first currency as default
MAIN_CURRENCY = MAIN_CURRENCIES[0] if MAIN_CURRENCIES else "USDC"

# Binance API rate limit (weights per minute, per 2025 docs)
# Used for quota monitoring in scalping_bot.py
BINANCE_WEIGHT_LIMIT = 6000

# Database and storage paths
DB_PATH = "/home/hamster/sbot/bot2/bot.db"  # SQLite database for parameters and trade logs

# Default trading parameters (stored in DB, can be overridden via GUI)
# Note: Chosen to balance risk/reward for scalping; adjustable in settings tab
DEFAULT_PARAMS = {
    "win_rate_threshold": 0.6,  # Minimum winrate to continue trading
    "max_consec_losses": 3,  # Pause after 3 consecutive losses
    "slippage_tolerance": 0.001,  # Max slippage (0.1%) for trade execution
    "risk_per_trade": 0.01,
    "grok_timeout": GROK_TIMEOUT,
    "auto_pair_limit": 20,  # Number of pairs to auto-trade (monitor 5x for analytics)
    "swap_pair_multiplier": 10,
    "volatility_check_interval": 4 * 60 * 60,
    "volatility_threshold_percent": 50.0,
    "grok_interval": 4 * 60 * 60,
    "dune_interval": 10 * 60,
    "analytics_interval": 60,
    "swap_threshold": 1.5,
    "cooldown": 45 * 60,
    "forecast_period": 4 * 60 * 60,
    "history_period": 48 * 60 * 60,
    "oi_threshold": 15,
    "profit_threshold": 0.16,
    "main_currency": "USDC",
    # Missing parameters added for trading logic
    "trade_interval": 10,  # Seconds between trading cycles
    "max_pairs": 10,  # Maximum number of pairs to trade simultaneously
    "min_volatility": 0.01,  # Minimum volatility threshold for pair selection
    "stop_loss_pct": 0.02,  # Stop loss percentage (2%)
    "take_profit_pct": 0.03,  # Take profit percentage (3%)
    "max_position_size": 0.1,  # Maximum position size as % of balance
    "max_drawdown": 0.1,  # Maximum drawdown before stopping
}

# Analytics settings for ContinuousAnalyzer
ANALYTICS_INTERVAL = int(
    get_param("ANALYTICS_INTERVAL", os.getenv("ANALYTICS_INTERVAL", 300))
)  # seconds
VOL_THRESHOLD = float(get_param("VOL_THRESHOLD", os.getenv("VOL_THRESHOLD", 0.05)))
GARCH_FLAG = get_param("GARCH_FLAG", os.getenv("GARCH_FLAG", "False")).lower() == "true"
GROK_PAIRS_INTERVAL = int(
    get_param("GROK_PAIRS_INTERVAL", os.getenv("GROK_PAIRS_INTERVAL", 3600))
)
GROK_SENTIMENT_INTERVAL = int(
    get_param("GROK_SENTIMENT_INTERVAL", os.getenv("GROK_SENTIMENT_INTERVAL", 600))
)

# Position sizing limits
MAX_DEAL_PERCENT = float(
    get_param("MAX_DEAL_PERCENT", os.getenv("MAX_DEAL_PERCENT", 0.2))
)
MAX_DEAL_ABSOLUTE = float(
    get_param("MAX_DEAL_ABSOLUTE", os.getenv("MAX_DEAL_ABSOLUTE", 10000))
)

# Minimum asset balance (in units) to monitor automatically
MIN_BALANCE_THRESHOLD = float(
    get_param("MIN_BALANCE_THRESHOLD", os.getenv("MIN_BALANCE_THRESHOLD", 10))
)

# Dynamic trading pair management
SWAP_PAIR_MULTIPLIER = int(
    get_param("SWAP_PAIR_MULTIPLIER", os.getenv("SWAP_PAIR_MULTIPLIER", 10))
)
VOLATILITY_CHECK_INTERVAL = int(
    get_param(
        "VOLATILITY_CHECK_INTERVAL", os.getenv("VOLATILITY_CHECK_INTERVAL", 4 * 60 * 60)
    )
)
VOLATILITY_THRESHOLD_PERCENT = float(
    get_param(
        "VOLATILITY_THRESHOLD_PERCENT", os.getenv("VOLATILITY_THRESHOLD_PERCENT", 50.0)
    )
)

# AnalyticsEngine configuration
_default_pairs = f"BTC/{MAIN_CURRENCY},ETH/{MAIN_CURRENCY}"
ANALYTICS_PAIRS = os.getenv("ANALYTICS_PAIRS", _default_pairs).split(",")
ANALYTICS_TIMEFRAME = os.getenv("ANALYTICS_TIMEFRAME", "1m")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
OI_THRESHOLD = int(get_param("OI_THRESHOLD", os.getenv("OI_THRESHOLD", 15)))

# Per-pair strategy configuration
# Default per-pair strategy combos used if no DB entry exists. Traders can
# override these on the fly via ``core.selector`` JSON inputs.
PAIR_STRATEGIES = {
    f"BTC/{MAIN_CURRENCY}": {
        "active": True,
        "strategies": ["EMA", "RSI"],
        "rule": "AND",
    },
    f"ETH/{MAIN_CURRENCY}": {"active": False, "strategies": ["GRID"], "rule": "AND"},
}

# Strategy matrices for spot and futures trading
SPOT_MATRIX = [
    {"condition": "std_dev < 0.2", "combo": ["Grid", "RSI"]},
    {"condition": "std_dev >= 0.2 and std_dev < 0.4", "combo": ["EMA", "RSI", "FVG"]},
    {"condition": "std_dev >= 0.4", "combo": ["Triangular Arb", "Bid-Ask Spread"]},
]

FUTURES_MATRIX = [
    {"condition": "std_dev < 0.2", "combo": ["Grid", "Arbitrage"]},
    {
        "condition": "std_dev >= 0.2 and std_dev < 0.4",
        "combo": ["EMA", "RSI", "Delta-Neutral"],
    },
    {
        "condition": "std_dev >= 0.4 and oi_change > 0.15",
        "combo": ["Bollinger", "Heatmap", "FVG"],
    },
]

# Switch triggers for adaptive strategy changes
TRIGGERS = {
    "common": [
        {
            "name": "drawdown_high",
            "cond": "drawdown > 0.02",
            "action": "add_defensive_filter",
        }
    ],
    "spot": [
        {"name": "mempool_high", "cond": "mempool_density > 0.8", "action": "add_MEV"}
    ],
    "futures": [
        {
            "name": "oi_surge",
            "cond": "oi_change > 0.15 and funding_rate > 0.0001",
            "action": "add_Heatmap",
        }
    ],
}

# Default strategy combos for regime switching
BULL_COMBO = ["EMA", "RSI", "MACD"]
NEUTRAL_COMBO = ["Grid", "RSI"]
BEAR_COMBO = ["Delta-Neutral", "Arbitrage"]

# Threshold for triggering immediate Grok queries when metrics change
CHANGE_THRESHOLD = 0.1

# Weights for hourly hybrid regime detection
HYBRID_LOCAL_WEIGHT = 0.7
HYBRID_GROK_WEIGHT = 0.3

# Exchange-specific configurations
EXCHANGE_CONFIG = {
    'binance': {
        'maker_fee': 0.001,  # 0.1%
        'taker_fee': 0.001,  # 0.1%
        'withdraw_fee': 0.0005,
        'min_order_size': 10,
        'rate_limit': 1200  # requests per minute
    },
    'coinbase': {
        'maker_fee': 0.005,  # 0.5%
        'taker_fee': 0.005,  # 0.5%
        'withdraw_fee': 0.0,
        'min_order_size': 10,
        'rate_limit': 600
    },
    'kraken': {
        'maker_fee': 0.0016,  # 0.16%
        'taker_fee': 0.0026,  # 0.26%
        'withdraw_fee': 0.0005,
        'min_order_size': 10,
        'rate_limit': 900
    },
    'okx': {
        'maker_fee': 0.001,  # 0.1%
        'taker_fee': 0.0015,  # 0.15%
        'withdraw_fee': 0.0004,
        'min_order_size': 10,
        'rate_limit': 600
    },
    'kucoin': {
        'maker_fee': 0.001,  # 0.1%
        'taker_fee': 0.001,  # 0.1%
        'withdraw_fee': 0.0005,
        'min_order_size': 10,
        'rate_limit': 1800
    },
    'bybit': {
        'maker_fee': 0.001,  # 0.1%
        'taker_fee': 0.001,  # 0.1%
        'withdraw_fee': 0.0005,
        'min_order_size': 10,
        'rate_limit': 1200
    }
}

# Multi-exchange parameters
ARBITRAGE_MIN_PROFIT = float(get_param("ARBITRAGE_MIN_PROFIT", os.getenv("ARBITRAGE_MIN_PROFIT", 0.001)))  # 0.1% after fees
VOLUME_LIMIT_PCT = float(get_param("VOLUME_LIMIT_PCT", os.getenv("VOLUME_LIMIT_PCT", 0.001)))  # 0.1% of exchange volume
CROSS_EXCHANGE_WEIGHT = float(get_param("CROSS_EXCHANGE_WEIGHT", os.getenv("CROSS_EXCHANGE_WEIGHT", 2.0)))  # 2x weight for signals
EXCHANGE_TIMEOUT = int(get_param("EXCHANGE_TIMEOUT", os.getenv("EXCHANGE_TIMEOUT", 30)))  # 30 seconds
MAX_ARBITRAGE_VOLUME = float(get_param("MAX_ARBITRAGE_VOLUME", os.getenv("MAX_ARBITRAGE_VOLUME", 10000)))  # $10k per opportunity
