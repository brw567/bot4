import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Iterable, Dict
import time
import os

import pandas as pd

try:
    from ta import add_all_ta_features
except Exception:  # pragma: no cover - optional dep

    def add_all_ta_features(
        df, open="open", high="high", low="low", close="close", volume="volume"
    ):
        return df


from utils.binance_utils import get_binance_client

try:
    from utils.ml_utils import lstm_predict
except Exception:  # pragma: no cover

    def lstm_predict(*a, **k):
        return {}


try:
    from utils.onchain_utils import get_oi_funding, get_dune_data
except Exception:  # pragma: no cover - optional in tests

    def get_oi_funding(*a, **k):
        return 0.0, 0.0

    def get_dune_data(*a, **k):
        return {}


from sympy.parsing.sympy_parser import parse_expr

try:
    # Use new centralized Grok API
    from core.grok_api import get_grok_api
    _grok_api = get_grok_api()
    get_grok_pairs = _grok_api.get_recommended_pairs
    # Map old function to new API - grok_query is more complex, handled below
    daily_sentiment_query = None  # Will be handled below
    get_cached_sentiment = None   # Will be handled below

    # For grok_query, we need to handle the legacy interface
    async def grok_query(prompt, ttl=3600, force=False):
        """Legacy grok_query wrapper for backward compatibility."""
        try:
            result = _grok_api.query_grok(prompt, cache_ttl=ttl)
            # Parse the result to match expected format
            import re
            score_match = re.search(r"score\s*[:=]\s*([0-9.]+)", result)
            score = float(score_match.group(1)) if score_match else 0.5

            reasoning = ""
            if "reasoning" in result.lower():
                reasoning_match = re.search(r"reasoning[:\s]+(.*?)(?:decision|$)", result, re.IGNORECASE)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            decision = ""
            if "decision" in result.lower():
                decision_match = re.search(r"decision[:\s]+(\w+)", result, re.IGNORECASE)
                decision = decision_match.group(1) if decision_match else ""

            return {"score": score, "reasoning": reasoning, "decision": decision}
        except Exception as e:
            return {"score": 0.5, "reasoning": f"Error: {e}", "decision": "neutral"}

    async def daily_sentiment_query(pair, metrics):
        """Enhanced daily_sentiment_query using new daily sentiment engine."""
        try:
            # Try daily sentiment engine first (preferred)
            from core.sentiment_engine import get_daily_sentiment_for_pair
            result = await get_daily_sentiment_for_pair(pair)
            
            # Return in legacy format with required fields
            return {
                "score": result.get('score', 0.5),  # 0-1 as required
                "reasoning": result.get('reasoning', ''),
                "decision": result.get('decision', 'hold')  # buy/sell/hold as required
            }
        except ImportError:
            # Fallback to sentiment pipeline
            try:
                from core.sentiment_pipeline import get_sentiment_for_analytics
                result = await get_sentiment_for_analytics(pair, metrics)
                
                return {
                    "score": result.get('score', 0.5),
                    "reasoning": result.get('reasoning', ''),
                    "decision": result.get('decision', 'neutral')
                }
            except ImportError:
                # Final fallback to legacy implementation
                try:
                    prompt = (
                        f"Measure crypto market sentiment (positive/neutral/negative score 0-1) "
                        f"for {pair} with local vol={metrics.get('std_dev', 0)}, "
                        f"RSI={metrics.get('rsi', 0)}, ADX={metrics.get('adx', 0)}, "
                        f"imbalance={metrics.get('imbalance', 0)}. Combine with latest "
                        "sentiment/news/vol for risk-averse decision, e.g., avoid leverage if "
                        "<0.5 bearish/neutral. Provide score, reasoning, decision."
                    )
                    return await grok_query(prompt, ttl=86400, force=True)
                except Exception:
                    return {"score": 0.5, "reasoning": "", "decision": ""}

    def get_cached_sentiment():
        """Legacy cached sentiment wrapper."""
        try:
            stats = _grok_api.get_service_stats()
            if stats.get('cache_hits', 0) > 0:
                # Return a dummy cached result format
                return {"score": 0.5, "reasoning": "cached", "decision": "neutral"}
            return None
        except Exception:
            return None

except Exception:  # pragma: no cover - optional in tests

    def get_grok_pairs(*a, **k):
        return []

    async def grok_query(*a, **k):
        return None

    async def daily_sentiment_query(*a, **k):
        return {"score": 0.5, "reasoning": "", "decision": ""}

    def get_cached_sentiment():
        return None


try:
    from utils.db_utils import save_pair_config
except Exception:  # pragma: no cover - optional in tests

    def save_pair_config(*a, **k):
        pass


from utils.config import load_config_from_db

try:
    from utils.db_utils import (
        store_var,
        get_var,
        get_trade_type,
        set_trade_type,
        get_combo,
        update_combo,
        get_last_switch_ts,
        update_last_switch_ts,
        get_prev_oi,
        update_prev_oi,
        get_trade_history,
        get_equity,
        get_prev_metrics,
        update_prev_metrics,
    )
except Exception:  # pragma: no cover - tests may stub
    store_var = lambda *a, **k: None
    get_var = lambda k, d=None: d
    get_trade_type = lambda p: "spot"
    set_trade_type = lambda p, t: None
    get_combo = lambda p: []
    update_combo = lambda p, c: None
    get_last_switch_ts = lambda p: 0.0
    update_last_switch_ts = lambda p, t: None
    get_prev_oi = lambda p: 0.0
    update_prev_oi = lambda p, v: None
    get_trade_history = lambda p, l=20: []
    get_equity = lambda p: []
    get_prev_metrics = lambda p: {}
    update_prev_metrics = lambda p, m: None
try:
    from utils.db_utils import (
        get_last_grok_ts,
        update_last_grok_ts,
        cache_grok,
        get_cached_grok,
        get_leverage,
        set_leverage,
    )
except Exception:  # pragma: no cover - tests may stub
    get_last_grok_ts = lambda: 0.0
    update_last_grok_ts = lambda ts: None
    cache_grok = lambda *a, **k: None
    get_cached_grok = lambda: (None, None, None)
    get_leverage = lambda p: 1
    set_leverage = lambda p, l: None
from core.selector import StrategySelector

try:
    import redis
except Exception:  # pragma: no cover
    redis = None
import config

handler = RotatingFileHandler("bot.log", maxBytes=1_000_000, backupCount=5)
logging.basicConfig(
    level=logging.INFO, handlers=[handler], format="%(asctime)s - %(message)s"
)

STRATEGY_MAP = {"trending": "momentum", "sideways": "grid", "volatile": "arbitrage"}
FUTURES_ONLY = {"Heatmap"}


class AnalyticsEngine:
    """Asynchronous engine for continuous multi-pair analytics."""

    def __init__(self, pairs: Iterable[str], timeframe: str = "1m"):
        cfg = load_config_from_db()
        self.max_active = int(
            cfg.get("max_active_pairs", cfg.get("auto_pair_limit", 5))
        )
        self.swap_multiplier = int(cfg.get("swap_pair_multiplier", 10))
        self.grok_interval = int(cfg.get("grok_interval", 4 * 60 * 60))
        self.dune_interval = int(cfg.get("dune_interval", 600))
        self.analytics_interval = int(cfg.get("analytics_interval", 60))
        self.swap_threshold = float(cfg.get("swap_threshold", 1.5))
        self.cooldown = int(cfg.get("cooldown", 45 * 60))
        self.forecast_period = int(cfg.get("forecast_period", 4 * 60 * 60))
        self.history_period = int(cfg.get("history_period", 48 * 60 * 60))
        self._limit = self.history_period // 60
        self.vol_threshold = get_var("vol_threshold")
        if self.vol_threshold is None:
            self.vol_threshold = int(
                os.getenv("OI_THRESHOLD", getattr(config, "OI_THRESHOLD", 15))
            )
            store_var("vol_threshold", self.vol_threshold)
        self.timeframe = timeframe
        self.metrics: Dict[str, dict] = {}
        self.active_pairs = list(pairs)[: self.max_active]
        self.swap_pairs = list(pairs)[
            self.max_active : self.max_active * self.swap_multiplier
        ]
        self.pairs = self.active_pairs + self.swap_pairs
        self.cooldowns: Dict[str, float] = {}
        self.volatile = False
        try:
            self.redis = redis.Redis(
                host=config.REDIS_HOST, port=config.REDIS_PORT, db=config.REDIS_DB
            )
        except Exception:
            self.redis = None
        
        # Initialize sentiment pipeline
        self.sentiment_pipeline = None
        try:
            from core.sentiment_pipeline import initialize_sentiment_pipeline
            self.sentiment_pipeline = initialize_sentiment_pipeline(self.pairs)
            logging.info("Sentiment pipeline initialized for analytics engine")
        except Exception as e:
            logging.warning(f"Sentiment pipeline initialization failed: {e}")
            self.sentiment_pipeline = None
        
        # Initialize daily sentiment engine
        self.daily_sentiment_engine = None
        try:
            from core.sentiment_engine import initialize_daily_sentiment_engine
            self.daily_sentiment_engine = initialize_daily_sentiment_engine(self.pairs)
            self.daily_sentiment_engine.start_scheduler()
            logging.info("Daily sentiment engine initialized and scheduled")
        except Exception as e:
            logging.warning(f"Daily sentiment engine initialization failed: {e}")
            self.daily_sentiment_engine = None
        required = [
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET",
            "GROK_API_KEY",
        ]
        # In TEST_MODE, Telegram is optional
        if not os.getenv("TEST_MODE", "False").lower() == "true":
            required.extend([
                "TELEGRAM_TOKEN",
                "TELEGRAM_API_ID",
                "TELEGRAM_API_HASH",
            ])
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise ValueError(f"Missing env key: {', '.join(missing)}")

    async def fetch_data(
        self, pair: str, limit: int | None = None, timeout: float = 15.0
    ) -> pd.DataFrame:
        """Return OHLCV dataframe for the pair with Grok fallback.

        The optional ``timeout`` controls how long to wait for Binance before
        falling back to Grok.
        """
        from utils.perf_utils import check_cpu_usage

        limit = limit or self._limit
        try:
            client = get_binance_client()
            ohlcv = await asyncio.wait_for(
                asyncio.to_thread(
                    client.fetch_ohlcv, pair, self.timeframe, limit=limit
                ),
                timeout=timeout,
            )
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["source"] = "binance"
            df = add_all_ta_features(
                df, open="open", high="high", low="low", close="close", volume="volume"
            )
        except (asyncio.TimeoutError, TimeoutError):
            logging.error(f"Fetch timeout for {pair}")
            try:
                from utils.telegram_utils import send_alert

                await send_alert(f"Timeout fetching data for {pair}")
            except Exception as err:
                logging.error(f"Alert failed: {err}")
            try:
                from core.grok_api import get_grok_api
                ohlcv = await get_grok_api().grok_fetch_ohlcv(pair, self.timeframe, limit)
            except ImportError:
                from utils.grok_utils import grok_fetch_ohlcv
                ohlcv = await grok_fetch_ohlcv(pair, self.timeframe, limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["source"] = "grok"
            df = add_all_ta_features(
                df, open="open", high="high", low="low", close="close", volume="volume"
            )
        except Exception as e:
            logging.error(f"Binance fetch failed for {pair}: {e}")
            try:
                from utils.telegram_utils import send_alert

                await send_alert(f"Binance fetch failed for {pair}: {e}")
            except Exception as err:
                logging.error(f"Alert failed: {err}")
            try:
                from core.grok_api import get_grok_api
                ohlcv = await get_grok_api().grok_fetch_ohlcv(pair, self.timeframe, limit)
            except ImportError:
                from utils.grok_utils import grok_fetch_ohlcv
                ohlcv = await grok_fetch_ohlcv(pair, self.timeframe, limit)
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["source"] = "grok"
            df = add_all_ta_features(
                df, open="open", high="high", low="low", close="close", volume="volume"
            )
        check_cpu_usage(threshold=10.0)
        return df

    def compute_metrics(self, df: pd.DataFrame) -> dict:
        try:
            rsi = float(df["momentum_rsi"].iloc[-1]) if "momentum_rsi" in df else 50.0
            macd_diff = (
                float(df["trend_macd_diff"].iloc[-1])
                if "trend_macd_diff" in df
                else 0.0
            )
            atr = (
                float(df["volatility_atr"].iloc[-1]) if "volatility_atr" in df else 0.0
            )
            sharpe = 0.0
            returns = df["close"].pct_change().dropna()
            if returns.std() != 0:
                sharpe = float((returns.mean() / returns.std()) * (252**0.5))
            return {
                "rsi": rsi,
                "macd_diff": macd_diff,
                "atr": atr,
                "sharpe": sharpe,
                "avg_atr": (
                    float(df["volatility_atr"].mean())
                    if "volatility_atr" in df
                    else atr
                ),
            }
        except Exception as e:  # pragma: no cover - unexpected df issues
            logging.error(f"Metric calculation failed: {e}")
            return {
                "rsi": 50.0,
                "macd_diff": 0.0,
                "atr": 0.0,
                "sharpe": 0.0,
                "avg_atr": 0.0,
            }

    async def analyze_once(self):
        """Analyze all pairs once and update metrics."""
        for pair in self.pairs:
            try:
                df = await self.fetch_data(pair)
                metrics = self.compute_metrics(df)
                oi, funding = get_oi_funding(pair)
                oi_change = oi.get("change", 0) if isinstance(oi, dict) else 0
                if metrics["macd_diff"] > 0 and metrics["rsi"] > 50:
                    pattern = "trending"
                elif metrics["atr"] < metrics["avg_atr"]:
                    pattern = "sideways"
                else:
                    pattern = "volatile"
                strategy = STRATEGY_MAP.get(pattern, "hold")
                prediction = lstm_predict(df)
                if prediction.get("confidence", 0) > 0.7:
                    metrics.update(
                        {
                            "pattern": pattern,
                            "strategy": strategy,
                            "oi_change": oi_change,
                        }
                    )
                else:
                    metrics.update(
                        {"pattern": "hold", "strategy": "hold", "oi_change": oi_change}
                    )
                metrics["funding_rate"] = funding
                metrics["data_source"] = (
                    df["source"].iloc[-1] if "source" in df else "binance"
                )
                metrics["rating"] = self.smart_rating(
                    metrics, forecast=self.forecast_period
                )
                prev = self.metrics.get(pair, {})
                self.metrics[pair] = metrics
                self.volatile = self.volatile or self.detect_volatility(
                    metrics, oi_change
                )
                if metrics.get("oi_change", 0) > self.vol_threshold:
                    cfg = {
                        "rule": {
                            "rule": "OR",
                            "subrules": [
                                {
                                    "rule": "AND",
                                    "strategies": ["BOLLINGER", "ARBITRAGE"],
                                },
                                {"strategies": ["MEV"]},
                            ],
                        },
                        "params": {},
                    }
                    save_pair_config(pair, cfg)
                if prev.get("strategy") and prev.get("strategy") != metrics["strategy"]:
                    try:
                        from utils.telegram_utils import send_notification

                        await send_notification(
                            f"Switch to {metrics['strategy']} on {pair}"
                        )
                    except Exception as e:
                        logging.error(f"Notification failed: {e}")
                if self.redis:
                    try:
                        self.redis.set(f"metrics:{pair}", json.dumps(metrics))
                    except Exception as e:  # pragma: no cover - Redis optional
                        logging.error(f"Redis store failed: {e}")
            except Exception as e:  # pragma: no cover - network issues
                logging.error(f"Analytics error for {pair}: {e}")
                try:
                    from utils.telegram_utils import send_alert

                    await send_alert(f"Analysis failed for {pair}: {e}")
                except Exception as err:
                    logging.error(f"Alert failed: {err}")

    def smart_rating(self, metrics: dict, forecast: int) -> float:
        """Return a simple combined rating for a pair."""
        return (
            0.4 * metrics.get("sharpe", 0)
            + 0.3 * (metrics.get("rsi", 50) / 100)
            + 0.3 * (metrics.get("oi_change", 0) / 100)
        )

    def detect_volatility(self, metrics: dict, oi_change: float) -> bool:
        """Return True if market is volatile based on ATR or open interest."""
        threshold = max(self.vol_threshold, 15)
        return (
            metrics.get("atr", 0) > metrics.get("avg_atr", 1) * 2
            or oi_change > threshold
        )

    def handle_swapping(self):
        if not self.swap_pairs:
            return
        ratings = {p: self.metrics.get(p, {}).get("rating", 0) for p in self.pairs}
        if not ratings:
            return
        low = (
            min(self.active_pairs, key=lambda p: ratings.get(p, 0))
            if self.active_pairs
            else None
        )
        high = (
            max(self.swap_pairs, key=lambda p: ratings.get(p, 0))
            if self.swap_pairs
            else None
        )
        if not low or not high:
            return
        promote = ratings.get(high, 0) > ratings.get(low, 0) * self.swap_threshold
        high_metrics = self.metrics.get(high, {})
        if high_metrics.get("oi_change", 0) > self.vol_threshold:
            promote = True
        if promote and self.cooldowns.get(low, 0) <= time.time():
            self.cooldowns[low] = time.time() + self.cooldown
            self.active_pairs.remove(low)
            self.swap_pairs.remove(high)
            self.swap_pairs.append(low)
            self.active_pairs.append(high)
            self.pairs = self.active_pairs + self.swap_pairs
            store_var("last_swap", f"{low}->{high}")

    async def update_pairs(self):
        while True:
            all_pairs = get_grok_pairs(self.max_active * self.swap_multiplier)
            self.active_pairs = all_pairs[: self.max_active]
            self.swap_pairs = all_pairs[self.max_active :]
            self.pairs = self.active_pairs + self.swap_pairs
            await asyncio.sleep(
                self.grok_interval / 2 if self.volatile else self.grok_interval
            )

    async def dune_cache(self):
        while True:
            data = get_dune_data()
            if isinstance(data, dict):
                self.volatile = self.volatile or data.get("oi_change", 0) > 10
            if self.redis:
                try:
                    self.redis.set(
                        "market_volatile", json.dumps({"volatile": self.volatile})
                    )
                except Exception:
                    pass
            await asyncio.sleep(self.dune_interval)

    async def fetch_new_threshold(self):
        """Query Grok for the latest volatility threshold and update state."""
        try:
            prompt = "Based on current BTC annualized volatility, suggest open interest change threshold."  # noqa: E501
            data = await grok_query(prompt, force=True)
            result = data.get("decision", "") if isinstance(data, dict) else str(data)
            val = int("".join(filter(str.isdigit, result)) or self.vol_threshold)
            store_var("vol_threshold", val)
            self.vol_threshold = val
        except Exception as e:  # pragma: no cover - network parsing errors
            logging.error(f"Threshold update failed: {e}")

    async def update_vol_threshold(self):
        while True:
            await self.fetch_new_threshold()
            await asyncio.sleep(24 * 60 * 60)

    async def hourly_hybrid_loop(self):
        """Periodically apply cached hybrid regime weighting."""
        while True:
            try:
                for pair in self.active_pairs:
                    await self.monitor_and_switch(pair)
            except Exception as e:
                logging.error(f"hourly_hybrid_loop error: {e}")
            await asyncio.sleep(3600)

    async def continuous_analyze(self, interval: int | None = None):
        """Run continuous analysis loop with dynamic intervals."""
        if not getattr(self, "_tasks_started", False):
            self._tasks_started = True
            asyncio.create_task(self.update_pairs())
            asyncio.create_task(self.dune_cache())
            asyncio.create_task(self.update_vol_threshold())
            asyncio.create_task(self.hourly_hybrid_loop())
        interval = interval or self.analytics_interval
        while True:
            try:
                self.volatile = False
                await self.analyze_once()
                self.handle_swapping()
                
                # Update sentiment pipeline with market data
                if self.sentiment_pipeline:
                    try:
                        market_data = {pair: None for pair in self.pairs}  # Could pass actual DataFrames
                        await self.sentiment_pipeline.update_all_sentiments(market_data)
                    except Exception as e:
                        logging.error(f"Sentiment pipeline update failed: {e}")
                        
            except Exception as e:  # pragma: no cover - top level errors
                logging.error(f"continuous_analyze failed: {e}")
                try:
                    from utils.telegram_utils import send_alert

                    await send_alert(f"Continuous analyze failure: {e}")
                except Exception as err:
                    logging.error(f"Alert failed: {err}")
            await asyncio.sleep(30 if self.volatile else interval)

    def _calculate_cached_indicators(self, pair: str, closes: pd.Series) -> Dict[str, float]:
        """Calculate technical indicators with caching"""
        import hashlib
        
        # Generate cache key based on closing prices
        cache_key = hashlib.md5(closes.tail(20).to_json().encode()).hexdigest()
        cache_full_key = f"analytics:indicators:{pair}:{cache_key}"
        
        try:
            # Try to get from Redis cache
            if self.redis:
                cached = self.redis.get(cache_full_key)
                if cached:
                    logging.debug(f"Indicator cache hit for {pair}")
                    return json.loads(cached)
        except Exception as e:
            logging.error(f"Cache read error: {e}")
        
        # Calculate indicators
        indicators = {}
        
        indicators['std_dev'] = float(closes.std())
        indicators['ma50'] = (
            float(closes.rolling(50).mean().iloc[-1])
            if len(closes) >= 50
            else float(closes.mean())
        )
        indicators['ma200'] = (
            float(closes.rolling(200).mean().iloc[-1]) 
            if len(closes) >= 200 
            else indicators['ma50']
        )
        
        # RSI calculation
        diff = closes.diff().dropna()
        up = diff.clip(lower=0)
        down = -diff.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / (roll_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = float(rsi.iloc[-1]) if not rsi.empty else 50.0
        indicators['adx'] = float(indicators['std_dev'] * 100)
        
        # Cache the results
        try:
            if self.redis:
                self.redis.setex(cache_full_key, 300, json.dumps(indicators))  # 5 min TTL
                logging.debug(f"Cached indicators for {pair}")
        except Exception as e:
            logging.error(f"Cache write error: {e}")
        
        return indicators

    async def fetch_metrics(self, pair: str, trade_type: str) -> dict:
        """Return metrics required for the switch mechanism."""
        df_task = asyncio.create_task(self.fetch_data(pair, limit=10))
        hist_task = asyncio.to_thread(get_trade_history, pair, 20)
        eq_task = asyncio.to_thread(get_equity, pair)
        dune_task = asyncio.to_thread(get_dune_data)
        extra_task = (
            asyncio.to_thread(get_oi_funding, pair) if trade_type == "futures" else None
        )

        df = await df_task
        closes = pd.Series(df["close"]).astype(float)
        
        # Get cached indicators
        indicators = self._calculate_cached_indicators(pair, closes)
        
        book_task = asyncio.to_thread(
            lambda: get_binance_client().fetch_order_book(pair, limit=10)
        )
        hist, equity, dune_data, extra = await asyncio.gather(
            hist_task,
            eq_task,
            dune_task,
            extra_task or asyncio.sleep(0),
        )
        try:
            book = await book_task
        except Exception:
            book = {"bids": [], "asks": []}

        profits = [h.get("profit", 0) for h in hist]
        winrate = sum(p > 0 for p in profits) / len(profits) if profits else 0
        if equity:
            peak = max(equity)
            drawdown = max(1 - equity[-1] / peak, 0)
        else:
            drawdown = 0.0

        bids = sum(b[1] for b in book.get("bids", []))
        asks = sum(a[1] for a in book.get("asks", []))
        imbalance = (bids / (asks or 1)) - 1

        metrics = {
            "std_dev": indicators['std_dev'],
            "winrate": winrate,
            "drawdown": drawdown,
            "rsi": indicators['rsi'],
            "adx": indicators['adx'],
            "ma50": indicators['ma50'],
            "ma200": indicators['ma200'],
            "imbalance": imbalance,
            "vol": indicators['std_dev'],
        }

        metrics["mempool_density"] = dune_data.get("mempool_density", 0)
        if trade_type == "futures":
            oi = float(dune_data.get("oi", 0))
            logging.info(f"Dune OI fetched: {oi}")
            prev = get_prev_oi(pair)
            metrics["oi_change"] = (oi - prev) / prev if prev else 0.0
            update_prev_oi(pair, oi)
            if extra:
                _, funding = extra
                metrics["funding_rate"] = funding

        return metrics

    async def cooldown_ok(self, pair: str) -> bool:
        last_ts = get_last_switch_ts(pair)
        return time.time() - last_ts > 60

    def calc_metrics_change(self, current: dict, previous: dict) -> float:
        """Return maximum absolute change among key metrics."""
        if not previous:
            return 0.0
        keys = ["rsi", "adx", "std_dev", "imbalance"]
        diffs = [abs(current.get(k, 0) - previous.get(k, 0)) for k in keys]
        return max(diffs) if diffs else 0.0

    async def monitor_and_switch(self, pair: str):
        trade_type = get_trade_type(pair)
        metrics = await self.fetch_metrics(pair, trade_type)
        try:
            await daily_sentiment_query(pair, metrics)
        except Exception:
            pass
        prev_metrics = get_prev_metrics(pair)
        change = self.calc_metrics_change(metrics, prev_metrics)
        force = change > config.CHANGE_THRESHOLD
        if force:
            logging.info(
                f"Metric change {change:.2f} detected for {pair}; querying Grok"
            )
        selector = StrategySelector()
        combo = selector.select_by_matrix(metrics, trade_type)
        current = get_combo(pair)
        if trade_type == "spot" and any(s in current for s in FUTURES_ONLY):
            raise ValueError("Futures-only strategy in spot pair")
        changed = False
        if combo and combo != current and await self.cooldown_ok(pair):
            if trade_type == "spot" and any(s in combo for s in FUTURES_ONLY):
                raise ValueError("Futures-only strategy in spot pair")
            update_combo(pair, combo)
            update_last_switch_ts(pair, time.time())
            logging.info(f"Switched {pair} ({trade_type}) to {combo}")
            changed = True
        # triggers
        for trig in config.TRIGGERS["common"] + config.TRIGGERS.get(trade_type, []):
            if self._eval_condition(trig["cond"], metrics) and await self.cooldown_ok(
                pair
            ):
                new_combo = self.apply_trigger_action(current, trig["action"])
                if trade_type == "spot" and any(s in new_combo for s in FUTURES_ONLY):
                    raise ValueError("Futures-only strategy in spot pair")
                if new_combo != current:
                    update_combo(pair, new_combo)
                    update_last_switch_ts(pair, time.time())
                    logging.info(f"Triggered {trig['name']} on {pair}")
                    changed = True
        if not changed:
            local_regime = self.local_regime(metrics)
            local_score = self._score_from_regime(local_regime)
            grok_data = get_cached_sentiment()
            hybrid, score = self.calc_hybrid(local_score, grok_data)
            regime = (
                "bullish" if hybrid > 0.5 else "bearish" if hybrid < -0.5 else "neutral"
            )
            leverage = 2 if regime == "bullish" and score >= 0.5 else 1
            combo = (
                config.BULL_COMBO
                if regime == "bullish"
                else config.BEAR_COMBO if regime == "bearish" else config.NEUTRAL_COMBO
            )
            if hybrid < 0 or score < 0.5:
                leverage = 1
                combo = ["Arbitrage", "Delta-Neutral"]
                logging.info("Hybrid <0: Switched arb+delta no leverage")
            if grok_data is None:
                logging.info("No Grok cache: Fallback local")
            if get_combo(pair) != combo or get_leverage(pair) != leverage:
                update_combo(pair, combo)
                set_leverage(pair, leverage)
                logging.info(
                    f"Hourly hybrid regime {regime} for {pair}: hybrid {hybrid}, score {score}"
                )
                try:
                    from utils.telegram_utils import send_notification

                    await send_notification(
                        f"Hourly hybrid regime {regime} for {pair}: hybrid {hybrid:.2f}, score {score:.2f}"
                    )
                except Exception as e:
                    logging.error(f"Notification failed: {e}")
                changed = True
        if not changed:
            ttl = 7200 if metrics.get("vol", 0) > 0.4 else 14400
            regime, grok_data = await self.detect_regime(
                pair, metrics, ttl=ttl, force=force
            )
            await self.switch_strategy(pair, regime, grok_data, metrics)
        update_prev_metrics(pair, metrics)

    def apply_trigger_action(self, combo: list, action: str) -> list:
        if action.startswith("add_"):
            strat = action.split("_", 1)[1]
            if strat not in combo:
                return combo + [strat]
        return combo

    def _eval_condition(self, condition: str, metrics: dict) -> bool:
        expr_str = condition
        for k, v in metrics.items():
            expr_str = expr_str.replace(k, str(v))
        try:
            return bool(parse_expr(expr_str))
        except Exception:
            try:
                return bool(eval(expr_str, {"__builtins__": {}}))
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Grok helpers

    def build_prompt(self, pair: str, metrics: dict) -> str:
        return (
            "Measure crypto market sentiment (positive/neutral/negative score 0-1) "
            f"for {pair} with local metrics vol={metrics.get('std_dev', 0)}, "
            f"RSI={metrics.get('rsi', 0)}, ADX={metrics.get('adx', 0)}, "
            f"imbalance={metrics.get('imbalance', 0)}. Combine with latest "
            "sentiment/news/vol for risk-averse decision, e.g., avoid leverage if "
            "<0.5 bearish/neutral. Provide score, reasoning, decision."
        )

    async def grok_query(self, prompt: str, ttl: int = 14400, force: bool = False):
        """Wrapper around utils.grok_utils.grok_query with TTL."""
        return await grok_query(prompt, ttl=ttl, force=force)

    def local_regime(self, metrics: dict) -> str:
        if (
            metrics.get("ma50", 0) > metrics.get("ma200", 0)
            and metrics.get("adx", 0) > 25
        ):
            return "bullish"
        if (
            metrics.get("ma50", 0) < metrics.get("ma200", 0)
            and metrics.get("adx", 0) > 25
        ):
            return "bearish"
        return "neutral"

    def _score_from_regime(self, regime: str) -> int:
        return 1 if regime == "bullish" else -1 if regime == "bearish" else 0

    def calc_hybrid(
        self, local_score: int, grok_data: dict | None
    ) -> tuple[float, float]:
        """Return hybrid score and Grok score with caching fallback."""
        if grok_data:
            score = grok_data.get("score", 0.5)
            g_norm = (score - 0.5) * 2
            hybrid = (
                config.HYBRID_LOCAL_WEIGHT * local_score
                + config.HYBRID_GROK_WEIGHT * g_norm
            )
            return hybrid, score
        return float(local_score), 0.5

    # ------------------------------------------------------------------
    # Regime detection and adaptive switching
    async def detect_regime(
        self, pair: str, metrics: dict, ttl: int = 14400, force: bool = False
    ):
        """Return regime with hybrid local/Grok sentiment."""
        local = self.local_regime(metrics)
        grok_data = await self.grok_query(
            self.build_prompt(pair, metrics), ttl=ttl, force=force
        )
        local_score = self._score_from_regime(local)
        if grok_data:
            g_norm = (grok_data["score"] - 0.5) * 2
            hybrid_score = 0.7 * local_score + 0.3 * g_norm
            regime = (
                "bullish"
                if hybrid_score > 0.5
                else "bearish" if hybrid_score < -0.5 else "neutral"
            )
            cat = (
                "positive"
                if grok_data["score"] > 0.66
                else "negative" if grok_data["score"] < 0.33 else "neutral"
            )
            logging.info(
                f"4-hour Grok sentiment {grok_data['score']:.1f} {cat} for {pair}: {grok_data.get('reasoning','')}, {grok_data.get('decision','')}"
            )
            return regime, grok_data
        return local, {"score": 0.5, "reasoning": "local", "decision": ""}

    async def switch_strategy(
        self, pair: str, regime: str, grok_data: dict, metrics: dict
    ):
        combo = (
            config.BULL_COMBO
            if regime == "bullish"
            else config.BEAR_COMBO if regime == "bearish" else config.NEUTRAL_COMBO
        )
        leverage = 2 if regime == "bullish" else 1
        if (
            grok_data.get("score", 0.5) < 0.5
            or metrics.get("rsi", 50) < 40
            or metrics.get("adx", 0) < 25
            or "avoid leverage" in grok_data.get("decision", "").lower()
        ):
            leverage = 1
            combo = ["Arbitrage", "Delta-Neutral"]
        if metrics.get("drawdown", 0) > 0.03:
            leverage = 1
        if grok_data.get("score", 0.5) < 0.5 and metrics.get("std_dev", 0) > 1.5:
            combo = config.BEAR_COMBO
        update_combo(pair, combo)
        set_leverage(pair, leverage)
        logging.info(
            f"{regime.title()} score {grok_data.get('score',0.5):.2f}: set {pair} {combo} leverage {leverage}"
        )

# Apply extensions to add missing methods
try:
    from core.analytics_engine_extensions import add_analytics_methods
    add_analytics_methods(AnalyticsEngine)
except ImportError:
    pass