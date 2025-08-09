"""
Headless Scalping Bot - Works with React/FastAPI monitoring dashboard
Streamlit GUI removed - all monitoring through Phase 4 dashboard
"""
# Database imports moved to db_adapter
import threading
import time
import asyncio
import logging
import json
import pandas as pd
import redis
from datetime import datetime
from strategies.factory import StrategyFactory
from utils.performance_optimizer import PerformanceOptimizer, OptimizedRedisPublisher
from utils.config_validator import ConfigValidator
from utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, get_error_handler
from utils.trading_errors import TradingErrorHandler, TradingError
from config import (
    DB_PATH,
    DEFAULT_PARAMS,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    BINANCE_WEIGHT_LIMIT,
    ANALYTICS_PAIRS,
    ANALYTICS_TIMEFRAME,
    MAIN_CURRENCY,
    TEST_MODE,
    STAGE_MODE,
    PAIR_STRATEGIES,
)
from utils.db_adapter import (
    init_db,
    get_param,
    save_param,
    get_state,
    set_state,
    check_user,
    get_pair_config,
    log_trade,
    get_db_connection,
    get_db_cursor,
    USE_POSTGRES
)
from utils.binance_utils import get_binance_client, latency_monitor
from core.monitoring_bridge import start_monitoring, stop_monitoring

# Setup logging using centralized configuration
from utils.logging_config import get_bot_logger
logger = get_bot_logger()

try:
    # Use new centralized Grok API
    from core.grok_api import get_grok_api
    _grok_api = get_grok_api()
    logger.info("Grok API initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize Grok API: {e}")
    _grok_api = None

# Import ML and analytics
from utils.ml_utils import ArbitrageMLPredictor as MLPredictor, get_ml_predictor
from core.analytics_engine import AnalyticsEngine

# Analytics engine will be initialized later with pairs
analytics_engine = None

class HeadlessScalpingBot:
    """Headless version of the scalping bot for use with React dashboard"""
    
    def __init__(self):
        self.client = None
        self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self.running = False
        self.params = {}
        self.ml_model = None
        self.active_pairs = []
        self.swap_pairs = []
        self.cooldown_until = {}
        self.pair_scores = {}
        self.strategies = {}
        self.thread = None
        # Performance optimizers
        self.perf_optimizer = PerformanceOptimizer(self.redis_client)
        self.redis_publisher = OptimizedRedisPublisher(self.redis_client)
        # Error handling
        self.error_handler = get_error_handler(self.redis_client)
        self.trading_error_handler = TradingErrorHandler(self.error_handler)
        
    def initialize(self):
        """Initialize the bot"""
        try:
            # Validate configuration first
            validator = ConfigValidator()
            is_valid, errors, warnings = validator.validate()
            
            if warnings:
                for warning in warnings:
                    logger.warning(f"Config warning: {warning}")
            
            if not is_valid:
                logger.error("Configuration validation failed:")
                for error in errors:
                    logger.error(f"  - {error}")
                raise ValueError("Invalid configuration. Please fix errors before starting.")
            
            logger.info("Configuration validated successfully")
            
            # Initialize database
            init_db(DB_PATH)
            
            # Load parameters
            self.params = {k: get_param(k, v) for k, v in DEFAULT_PARAMS.items()}
            logger.info(f"Loaded params: {self.params}")
            
            # Initialize ML predictor
            self.ml_model = get_ml_predictor()
            # Train the model if needed
            self.ml_model.train_model()
            logger.info("ML model initialized and trained")
            
            # Initialize Binance client
            self.client = get_binance_client()
            logger.info("Binance client initialized")
            
            # Initialize strategies
            strategy_names = ['EMA', 'RSI', 'FVG', 'BOLLINGER', 'ARBITRAGE', 'GRID', 'MEV']
            self.strategies = {}
            for name in strategy_names:
                try:
                    self.strategies[name.lower()] = StrategyFactory.create(name)
                except Exception as e:
                    logger.warning(f"Failed to load strategy {name}: {e}")
            logger.info(f"Loaded {len(self.strategies)} strategies")
            
            # Start monitoring bridge
            start_monitoring()
            logger.info("Monitoring bridge started")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize bot: {e}")
            return False
    
    def start(self):
        """Start the bot"""
        logger.debug("Bot start() called")
        if not self.running:
            self.running = True
            set_state('running')
            self.thread = threading.Thread(target=self._bot_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("Bot started")
            
            # Publish status update
            self._publish_status("running")
    
    def stop(self):
        """Stop the bot"""
        if self.running:
            self.running = False
            set_state('stopped')
            if self.thread:
                self.thread.join()
            logger.info("Bot stopped")
            
            # Publish status update
            self._publish_status("stopped")
    
    def _bot_loop(self):
        """Main bot loop"""
        logger.debug("Bot loop started")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.running:
            try:
                # Get trading pairs
                self._update_trading_pairs()
                
                # Process pairs in parallel for better performance
                # Active pairs (with trading)
                active_results = self.perf_optimizer.parallel_process_pairs(
                    self.active_pairs,
                    lambda pair: self._safe_process_pair(pair, trade=True),
                    max_workers=5
                )
                
                # Monitored pairs (analytics only) - process in batches
                monitored_batch = self.swap_pairs[:20]
                if monitored_batch:
                    monitor_results = self.perf_optimizer.parallel_process_pairs(
                        monitored_batch,
                        lambda pair: self._safe_process_pair(pair, trade=False),
                        max_workers=3
                    )
                
                # Update analytics
                self._update_analytics()
                
                # Check for pair swapping opportunities
                self._check_pair_swaps()
                
                # Reset error counter on successful iteration
                consecutive_errors = 0
                
                # Sleep
                time.sleep(int(self.params.get('trade_interval', 10)))
                
            except Exception as e:
                consecutive_errors += 1
                context = self.error_handler.handle_error(
                    e, ErrorCategory.SYSTEM, ErrorSeverity.HIGH,
                    "bot_loop", {"consecutive_errors": consecutive_errors}
                )
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical(f"Too many consecutive errors ({consecutive_errors}), stopping bot")
                    self.stop()
                    break
                
                # Exponential backoff
                sleep_time = min(30 * (2 ** (consecutive_errors - 1)), 300)
                logger.error(f"Bot loop error (attempt {consecutive_errors}): {e}. Sleeping {sleep_time}s")
                time.sleep(sleep_time)
    
    def _update_trading_pairs(self):
        """Update active trading pairs"""
        logger.debug("Updating trading pairs")
        try:
            # Load markets using ccxt
            markets = self.client.load_markets()
            
            # Filter for active pairs with our quote currency
            tradeable = []
            for symbol, market in markets.items():
                if (market['active'] and 
                    market['quote'] == MAIN_CURRENCY and
                    market['type'] in ['spot', 'future']):
                    tradeable.append(symbol)
            
            # Get max pairs from settings (should be 20)
            max_pairs = int(self.params.get('auto_pair_limit', self.params.get('max_pairs', 10)))
            
            # Select top pairs based on volume or other criteria
            # For now, just take the first max_pairs
            self.active_pairs = tradeable[:max_pairs]
            
            # Select additional pairs for monitoring (5x active pairs)
            monitor_count = max_pairs * 5  # 20 * 5 = 100
            all_monitored = tradeable[:monitor_count]
            
            # Swap pairs are the ones we monitor but don't actively trade
            self.swap_pairs = [p for p in all_monitored if p not in self.active_pairs]
            
            logger.info(f"Trading pairs updated: {len(self.active_pairs)} active, {len(self.swap_pairs)} monitored")
            logger.info(f"Active pairs: {', '.join(self.active_pairs[:5])}...")
            logger.info(f"Total monitored: {len(self.active_pairs) + len(self.swap_pairs)} pairs")
            
            # Publish trading pairs to Redis
            self._publish_trading_pairs()
            
            # Also publish monitored pairs
            self._publish_monitored_pairs()
            
        except Exception as e:
            logger.error(f"Failed to update trading pairs: {e}")
    
    def _safe_process_pair(self, pair, trade=True):
        """Safely process a trading pair with error handling"""
        try:
            return self._process_pair(pair, trade)
        except Exception as e:
            # Handle exchange-specific errors
            if hasattr(e, '__module__') and 'ccxt' in str(e.__module__):
                handled = self.trading_error_handler.handle_exchange_error(
                    'binance', e, 'process_pair', pair
                )
                if not handled:
                    logger.error(f"Failed to process {pair}: {e}")
            else:
                # Handle general errors
                self.error_handler.handle_error(
                    e, ErrorCategory.TRADING, ErrorSeverity.MEDIUM,
                    f"process_pair_{pair}", {"pair": pair, "trade": trade}
                )
            return None
    
    def _process_pair(self, pair, trade=True):
        """Process a single trading pair"""
        logger.debug(f"Processing pair: {pair} (trade={trade})")
        try:
            # Get market data using ccxt
            ticker = self.client.fetch_ticker(pair)
            logger.debug(f"Ticker data for {pair}: {ticker}")
            price = float(ticker['last'])
            volume = float(ticker['baseVolume'])
            
            # Publish market data to Redis
            self._publish_market_data(pair, ticker)
            logger.debug(f"Published market data for {pair}")
            
            # Store pair score for analytics
            self.pair_scores[pair] = {
                'price': price,
                'volume': volume,
                'change24h': float(ticker.get('percentage', 0)),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Only continue with trading logic if trade=True
            if not trade:
                return
            
            # Get strategy for this pair
            pair_config = PAIR_STRATEGIES.get(pair, {})
            if isinstance(pair_config, dict):
                # Handle dictionary configuration
                strategy_names = pair_config.get('strategies', ['RSI'])
                strategy_name = strategy_names[0].lower() if strategy_names else 'rsi'
            else:
                # Handle legacy string configuration
                strategy_name = pair_config.lower() if pair_config else 'rsi'
            
            strategy = self.strategies.get(strategy_name)
            
            if not strategy:
                return
            
            # Get historical data using ccxt
            timeframe = self.params.get('timeframe', '5m')
            ohlcv = self.client.fetch_ohlcv(
                pair,
                timeframe=timeframe,
                limit=100
            )
            
            # Convert OHLCV data to DataFrame (ccxt format)
            df = pd.DataFrame(ohlcv, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Publish candle data to Redis for charting
            self._publish_candle_data(pair, ohlcv)
            
            # Data is already numeric from ccxt, just ensure float type
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Apply strategy
            signal_str = strategy.generate_signal(df)
            # Convert signal to numeric: buy=1, sell=-1, hold=0
            signal = 1 if signal_str == 'buy' else (-1 if signal_str == 'sell' else 0)
            
            # Execute trade if signal
            if signal != 0:
                self._execute_trade(pair, signal, price, volume)
            
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
    
    def _execute_trade(self, pair, signal, price, volume):
        """Execute a trade"""
        logger.debug(f"Executing trade for {pair}: signal={signal}, price={price}, volume={volume}")
        try:
            # Check risk management
            if not self._check_risk_management(pair, price, volume):
                return
            
            # Determine trade size
            trade_size = self._calculate_trade_size(pair, price)
            
            if TEST_MODE or STAGE_MODE:
                # Simulate trade (TestNet or Stage mode)
                mode_label = "[STAGE]" if STAGE_MODE else "[TEST]"
                logger.info(f"{mode_label} Would {'BUY' if signal > 0 else 'SELL'} {trade_size} {pair} at {price}")
                
                # Record simulated trade with appropriate status
                trade_status = 'stage_test' if STAGE_MODE else 'simulated'
                self._record_trade(pair, 'buy' if signal > 0 else 'sell', 
                                 price, trade_size, 0, 0, trade_status)
                
                # Clear portfolio cache after trade
                self._clear_portfolio_cache()
            else:
                # Execute real trade using ccxt
                side = 'buy' if signal > 0 else 'sell'
                order = self.client.create_order(
                    symbol=pair,
                    side=side,
                    type='market',
                    amount=trade_size
                )
                
                # Record trade
                self._record_trade(pair, side.lower(), price, trade_size, 
                                 float(order.get('cummulativeQuoteQty', 0)) * 0.001,
                                 0, 'completed')
                
                logger.info(f"Executed {side} {trade_size} {pair} at {price}")
                
                # Clear portfolio cache after trade
                self._clear_portfolio_cache()
                
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
    
    def _check_risk_management(self, pair, price, volume):
        """Check risk management rules"""
        # Implement risk checks
        return True
    
    def _calculate_trade_size(self, pair, price):
        """Calculate appropriate trade size"""
        # Implement position sizing
        return 0.001  # Placeholder
    
    def _record_trade(self, pair, side, price, amount, fee, profit, status):
        """Record trade in database"""
        try:
            if USE_POSTGRES:
                # Use context manager for PostgreSQL
                from utils.db_adapter import get_db_cursor
                with get_db_cursor() as cursor:
                    # Get strategy name from current pair
                    pair_config = PAIR_STRATEGIES.get(pair, {})
                    if isinstance(pair_config, dict):
                        strategy = pair_config.get('strategies', ['Manual'])[0]
                    else:
                        strategy = 'Manual'
                    
                    trade_data = {
                        'symbol': pair,
                        'side': side,
                        'price': price,
                        'amount': amount,
                        'fee': fee,
                        'profit': profit,
                        'status': status,
                        'strategy': strategy,
                        'timestamp': time.time()
                    }
                    log_trade(trade_data)
                    return
            else:
                conn = get_db_connection()
                cursor = conn.cursor()
            
            # Get strategy name from current pair
            pair_config = PAIR_STRATEGIES.get(pair, {})
            if isinstance(pair_config, dict):
                strategy_names = pair_config.get('strategies', ['unknown'])
                strategy_name = strategy_names[0] if strategy_names else 'unknown'
            else:
                strategy_name = pair_config if pair_config else 'unknown'
            
            cursor.execute("""
                INSERT INTO trades (symbol, side, price, amount, fee, profit, timestamp, status, strategy)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), ?, ?)
            """, (pair, side, price, amount, fee, profit, status, strategy_name))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
    
    def _update_analytics(self):
        """Update analytics data"""
        try:
            # Initialize analytics engine if needed
            global analytics_engine
            if analytics_engine is None and self.active_pairs:
                analytics_engine = AnalyticsEngine(self.active_pairs)
            
            # Calculate metrics if engine is available
            if analytics_engine:
                # Use compute_metrics instead of calculate_metrics
                try:
                    # Get current market data for metrics computation
                    df = pd.DataFrame()  # Empty for now, analytics engine handles fetching
                    metrics = analytics_engine.compute_metrics(df)
                    # Publish to Redis for dashboard
                    self.redis_client.publish('analytics', json.dumps(metrics))
                except Exception as e:
                    logger.debug(f"Analytics metrics computation: {e}")
            
        except Exception as e:
            logger.error(f"Failed to update analytics: {e}")
    
    def _publish_status(self, status):
        """Publish bot status"""
        try:
            # Calculate monitored pairs (5x active pairs)
            monitored_pairs = len(self.swap_pairs) if self.swap_pairs else len(self.active_pairs) * 5
            
            status_data = {
                "status": status,
                "timestamp": time.time(),
                "active_pairs": len(self.active_pairs),
                "monitored_pairs": monitored_pairs,
                "version": "4.0.0",
                "bot_name": "Scalping Bot",
                "exchange": "binance",
                "testmode": TEST_MODE
            }
            
            # Publish to channel
            self.redis_client.publish('bot_status', json.dumps(status_data))
            
            # Also store in Redis for persistence
            self.redis_client.set('bot:status', json.dumps(status_data))
            
        except Exception as e:
            logger.error(f"Failed to publish status: {e}")
    
    def _publish_trading_pairs(self):
        """Publish active trading pairs to Redis"""
        try:
            pairs_list = []
            for pair in self.active_pairs:
                pair_data = {
                    "symbol": pair,
                    "enabled": True,
                    "exchange": "binance",
                    "lastUpdate": datetime.utcnow().isoformat()
                }
                pairs_list.append(pair_data)
            
            # Use batch operations for better performance
            operations = [
                ('set', 'trading_pairs', json.dumps(pairs_list)),
                ('publish', 'trading_pairs', json.dumps(pairs_list))
            ]
            self.perf_optimizer.batch_redis_operations(operations)
            logger.info(f"Published {len(pairs_list)} trading pairs to Redis")
            
        except Exception as e:
            logger.error(f"Failed to publish trading pairs: {e}")
    
    def _publish_market_data(self, pair, ticker):
        """Publish market data for a trading pair"""
        try:
            market_data = {
                "pair": pair,
                "price": float(ticker.get('last', 0)),
                "volume": float(ticker.get('baseVolume', 0)),
                "high24h": float(ticker.get('high', 0)),
                "low24h": float(ticker.get('low', 0)),
                "change24h": float(ticker.get('percentage', 0)),
                "bid": float(ticker.get('bid', 0)),
                "ask": float(ticker.get('ask', 0)),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis with key format market:PAIR
            market_key = f"market:{pair.replace('/', '_')}"
            
            # Use batch publisher for efficiency
            self.redis_publisher.publish_batch('market_data', market_data)
            
            # Store with TTL to prevent stale data
            self.redis_client.setex(market_key, 300, json.dumps(market_data))  # 5 min TTL
            
        except Exception as e:
            logger.error(f"Failed to publish market data for {pair}: {e}")
    
    def _publish_candle_data(self, pair, ohlcv):
        """Publish candle data for charting"""
        try:
            # Format candles for frontend
            candles = []
            for candle in ohlcv[-50:]:  # Last 50 candles for charting
                candles.append({
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5]
                })
            
            candle_data = {
                "pair": pair,
                "timeframe": self.params.get('timeframe', '5m'),
                "candles": candles,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis with key format candles:PAIR
            candle_key = f"candles:{pair.replace('/', '_')}"
            self.redis_client.set(candle_key, json.dumps(candle_data))
            
        except Exception as e:
            logger.error(f"Failed to publish candle data for {pair}: {e}")
    
    def _clear_portfolio_cache(self):
        """Clear portfolio cache after trades - optimized"""
        try:
            # Use pipeline for batch deletion
            pipeline = self.redis_client.pipeline()
            keys_to_delete = []
            
            for key in self.redis_client.scan_iter("portfolio:balances:*", count=100):
                keys_to_delete.append(key)
                if len(keys_to_delete) >= 100:
                    pipeline.delete(*keys_to_delete)
                    keys_to_delete = []
            
            if keys_to_delete:
                pipeline.delete(*keys_to_delete)
            
            pipeline.execute()
            logger.debug("Cleared portfolio cache after trade")
        except Exception as e:
            logger.error(f"Failed to clear portfolio cache: {e}")
    
    def get_state(self):
        """Get current bot state"""
        return get_state()
    
    def get_metrics(self):
        """Get current metrics"""
        return {
            "active_pairs": self.active_pairs,
            "swap_pairs": self.swap_pairs,
            "cooldown_until": self.cooldown_until,
            "pair_scores": self.pair_scores
        }
    
    def _publish_monitored_pairs(self):
        """Publish monitored pairs to Redis"""
        try:
            monitored_list = []
            
            # Add all monitored pairs (active + swap)
            all_monitored = self.active_pairs + self.swap_pairs
            
            for pair in all_monitored:
                pair_data = {
                    "symbol": pair,
                    "enabled": True,
                    "active": pair in self.active_pairs,
                    "exchange": "binance",
                    "score": self.pair_scores.get(pair, {}).get('change24h', 0),
                    "lastUpdate": datetime.utcnow().isoformat()
                }
                monitored_list.append(pair_data)
            
            # Store in Redis
            self.redis_client.set('monitored_pairs', json.dumps(monitored_list))
            logger.info(f"Published {len(monitored_list)} monitored pairs to Redis")
            
        except Exception as e:
            logger.error(f"Failed to publish monitored pairs: {e}")
    
    def _check_pair_swaps(self):
        """Check if we should swap any active pairs with monitored ones"""
        try:
            # Only check every 5 minutes
            if not hasattr(self, '_last_swap_check'):
                self._last_swap_check = 0
            
            if time.time() - self._last_swap_check < 300:  # 5 minutes
                return
            
            self._last_swap_check = time.time()
            
            # Score all pairs
            scored_pairs = []
            
            # Score active pairs
            for pair in self.active_pairs:
                if pair in self.pair_scores:
                    score = self._calculate_pair_score(pair)
                    scored_pairs.append((pair, score, True))  # True = currently active
            
            # Score swap pairs
            for pair in self.swap_pairs:
                if pair in self.pair_scores:
                    score = self._calculate_pair_score(pair)
                    scored_pairs.append((pair, score, False))  # False = not active
            
            # Sort by score descending
            scored_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Check if any swaps should be made
            max_pairs = int(self.params.get('auto_pair_limit', self.params.get('max_pairs', 10)))
            top_pairs = scored_pairs[:max_pairs]
            
            # Get pairs that should be active
            new_active = [p[0] for p in top_pairs]
            
            # Check if different from current active
            if set(new_active) != set(self.active_pairs):
                logger.info(f"Swapping trading pairs based on performance")
                
                # Update active and swap pairs
                self.active_pairs = new_active
                self.swap_pairs = [p[0] for p in scored_pairs[max_pairs:] if p[0] not in new_active]
                
                # Publish updates
                self._publish_trading_pairs()
                self._publish_monitored_pairs()
                
                logger.info(f"New active pairs: {', '.join(self.active_pairs[:5])}...")
                
        except Exception as e:
            logger.error(f"Failed to check pair swaps: {e}")
    
    def _calculate_pair_score(self, pair):
        """Calculate a score for a trading pair based on various metrics"""
        try:
            if pair not in self.pair_scores:
                return 0
            
            data = self.pair_scores[pair]
            
            # Score based on:
            # - Volume (40%)
            # - 24h change magnitude (30%)
            # - Recent volatility (30%)
            
            volume_score = min(data['volume'] / 1000, 100)  # Normalize to 0-100
            change_score = abs(data['change24h']) * 10  # Higher change = more opportunity
            volatility_score = 50  # Placeholder - would calculate from price movements
            
            total_score = (volume_score * 0.4) + (change_score * 0.3) + (volatility_score * 0.3)
            
            return total_score
            
        except Exception as e:
            logger.error(f"Failed to calculate score for {pair}: {e}")
            return 0


# Global bot instance
bot = None

def get_bot():
    """Get or create bot instance"""
    global bot
    if bot is None:
        bot = HeadlessScalpingBot()
    return bot

def main():
    """Main entry point"""
    logger.info("Starting Headless Scalping Bot...")
    
    # Initialize bot
    bot = get_bot()
    if not bot.initialize():
        logger.error("Failed to initialize bot")
        return
    
    # Start bot
    bot.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bot.stop()
        stop_monitoring()
        

if __name__ == '__main__':
    main()