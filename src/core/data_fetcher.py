#!/usr/bin/env python3
"""
Real Data Fetcher Service with Binance Integration
Fetches actual market data instead of random values
"""

import os
import time
import json
import asyncio
import aiohttp
import redis
import psycopg2
from datetime import datetime, timedelta
import logging
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
import websocket
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        
        # Initialize Binance exchange
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY', ''),
            'secret': os.getenv('BINANCE_SECRET', ''),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # Expanded pairs list with USD pairs
        self.pairs = [
            # Major USD pairs
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT',
            'ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'AVAX/USDT',
            'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT',
            # BTC pairs
            'ETH/BTC', 'BNB/BTC', 'SOL/BTC', 'ADA/BTC',
            'XRP/BTC', 'DOT/BTC', 'AVAX/BTC', 'LINK/BTC',
            # ETH pairs
            'BNB/ETH', 'SOL/ETH', 'MATIC/ETH', 'LINK/ETH'
        ]
        
        self.fetch_interval = int(os.getenv('FETCH_INTERVAL', 5))  # 5 seconds for scalping
        self.orderbook_depth = 20
        self.historical_data = {}  # Store historical data for TA
        
        # WebSocket connections for real-time data
        self.ws_connections = {}
        self.ws_data = {}
        
        # Macroeconomic data cache
        self.macro_data = {}
        self.macro_update_interval = 3600  # Update every hour
        self.last_macro_update = 0
        
    async def fetch_market_data(self, pair: str) -> Optional[Dict]:
        """Fetch real market data for a pair"""
        try:
            # Fetch multiple data points in parallel
            tasks = [
                self.exchange.fetch_ticker(pair),
                self.exchange.fetch_order_book(pair, self.orderbook_depth),
                self.exchange.fetch_trades(pair, limit=100),
                self.exchange.fetch_ohlcv(pair, '1m', limit=100)
            ]
            
            ticker, orderbook, trades, ohlcv = await asyncio.gather(*tasks)
            
            # Calculate additional metrics
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate spread metrics
            bid_ask_spread = orderbook['asks'][0][0] - orderbook['bids'][0][0]
            spread_percentage = (bid_ask_spread / ticker['last']) * 100
            
            # Calculate liquidity metrics
            bid_liquidity = sum([bid[0] * bid[1] for bid in orderbook['bids'][:10]])
            ask_liquidity = sum([ask[0] * ask[1] for ask in orderbook['asks'][:10]])
            total_liquidity = bid_liquidity + ask_liquidity
            
            # Calculate order book imbalance
            bid_volume = sum([bid[1] for bid in orderbook['bids'][:10]])
            ask_volume = sum([ask[1] for ask in orderbook['asks'][:10]])
            order_imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
            
            # Calculate volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(1440)  # Daily volatility
            
            # Calculate volume profile
            volume_mean = df['volume'].mean()
            volume_std = df['volume'].std()
            current_volume_zscore = (df['volume'].iloc[-1] - volume_mean) / volume_std if volume_std > 0 else 0
            
            # Detect support and resistance levels
            support_resistance = self.calculate_support_resistance(df)
            
            data = {
                'pair': pair,
                'timestamp': datetime.now().isoformat(),
                
                # Price data
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low'],
                'vwap': ticker['vwap'] if 'vwap' in ticker else ticker['last'],
                
                # Volume data
                'volume_24h': ticker['quoteVolume'],
                'base_volume_24h': ticker['baseVolume'],
                'volume_zscore': current_volume_zscore,
                
                # Change metrics
                'change_24h': ticker['percentage'],
                'change_1h': self.calculate_change(df, 60),
                'change_5m': self.calculate_change(df, 5),
                
                # Spread and liquidity
                'spread': bid_ask_spread,
                'spread_percentage': spread_percentage,
                'bid_liquidity': bid_liquidity,
                'ask_liquidity': ask_liquidity,
                'total_liquidity': total_liquidity,
                'order_imbalance': order_imbalance,
                
                # Volatility metrics
                'volatility': volatility,
                'atr': self.calculate_atr(df),
                
                # Market depth
                'orderbook_bids': len(orderbook['bids']),
                'orderbook_asks': len(orderbook['asks']),
                'bid_depth_10': sum([b[1] for b in orderbook['bids'][:10]]),
                'ask_depth_10': sum([a[1] for a in orderbook['asks'][:10]]),
                
                # Support and resistance
                'support_levels': support_resistance['support'],
                'resistance_levels': support_resistance['resistance'],
                'nearest_support': support_resistance['nearest_support'],
                'nearest_resistance': support_resistance['nearest_resistance'],
                
                # Trade flow
                'buy_volume': sum([t['amount'] for t in trades if t['side'] == 'buy']),
                'sell_volume': sum([t['amount'] for t in trades if t['side'] == 'sell']),
                'trade_count': len(trades),
                'avg_trade_size': sum([t['amount'] for t in trades]) / len(trades) if trades else 0,
                
                # Technical indicators
                'rsi': self.calculate_rsi(df),
                'macd': self.calculate_macd(df),
                'bollinger_bands': self.calculate_bollinger_bands(df),
                'ema_short': df['close'].ewm(span=9).mean().iloc[-1],
                'ema_long': df['close'].ewm(span=21).mean().iloc[-1],
                'sma_50': df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else df['close'].mean(),
                'sma_200': df['close'].rolling(200).mean().iloc[-1] if len(df) >= 200 else df['close'].mean(),
            }
            
            # Store in Redis with TTL
            key = f"market_data:{pair}"
            self.redis_client.setex(key, 60, json.dumps(data))
            
            # Store historical data
            if pair not in self.historical_data:
                self.historical_data[pair] = deque(maxlen=1440)  # 24 hours of minute data
            self.historical_data[pair].append(data)
            
            logger.info(f"Fetched real data for {pair}: ${data['price']:.4f} "
                       f"(spread: {data['spread_percentage']:.3f}%, "
                       f"liquidity: ${data['total_liquidity']:.0f})")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching {pair}: {e}")
            return None
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        try:
            # Use pivot points
            pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
            
            # Calculate support levels
            s1 = 2 * pivot - df['high'].iloc[-1]
            s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
            s3 = df['low'].iloc[-1] - 2 * (df['high'].iloc[-1] - pivot)
            
            # Calculate resistance levels
            r1 = 2 * pivot - df['low'].iloc[-1]
            r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
            r3 = df['high'].iloc[-1] + 2 * (pivot - df['low'].iloc[-1])
            
            current_price = df['close'].iloc[-1]
            
            return {
                'support': [s3, s2, s1],
                'resistance': [r1, r2, r3],
                'pivot': pivot,
                'nearest_support': max([s for s in [s3, s2, s1] if s < current_price], default=s1),
                'nearest_resistance': min([r for r in [r1, r2, r3] if r > current_price], default=r1)
            }
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return {
                'support': [], 'resistance': [], 'pivot': 0,
                'nearest_support': 0, 'nearest_resistance': 0
            }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return float(atr)
        except Exception:
            return 0.0
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0
    
    def calculate_macd(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD"""
        try:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except Exception:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        try:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            return {
                'upper': float(upper.iloc[-1]),
                'middle': float(sma.iloc[-1]),
                'lower': float(lower.iloc[-1]),
                'width': float((upper - lower).iloc[-1])
            }
        except Exception:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0}
    
    def calculate_change(self, df: pd.DataFrame, minutes: int) -> float:
        """Calculate price change over specified minutes"""
        try:
            if len(df) > minutes:
                old_price = df['close'].iloc[-minutes]
                current_price = df['close'].iloc[-1]
                return ((current_price - old_price) / old_price) * 100
            return 0.0
        except Exception:
            return 0.0
    
    async def fetch_macro_data(self):
        """Fetch macroeconomic data"""
        try:
            current_time = time.time()
            if current_time - self.last_macro_update < self.macro_update_interval:
                return self.macro_data
            
            # Fetch DXY (Dollar Index) - using EURUSD as proxy
            try:
                eurusd = await self.exchange.fetch_ticker('EUR/USDT')
                dxy_proxy = 1 / eurusd['last'] * 100  # Inverse correlation
            except:
                dxy_proxy = 100
            
            # Fetch commodity correlations
            try:
                gold = await self.exchange.fetch_ticker('PAXG/USDT')  # Gold token
            except:
                gold = {'last': 2000}
            
            # Calculate crypto market cap index (top coins weighted)
            market_cap_index = 0
            weights = {'BTC/USDT': 0.4, 'ETH/USDT': 0.3, 'BNB/USDT': 0.1, 
                      'SOL/USDT': 0.1, 'ADA/USDT': 0.05, 'XRP/USDT': 0.05}
            
            for pair, weight in weights.items():
                try:
                    ticker = await self.exchange.fetch_ticker(pair)
                    market_cap_index += ticker['last'] * weight
                except:
                    pass
            
            self.macro_data = {
                'dxy_proxy': dxy_proxy,
                'gold_price': gold['last'],
                'crypto_index': market_cap_index,
                'timestamp': datetime.now().isoformat()
            }
            
            self.last_macro_update = current_time
            
            # Store in Redis
            self.redis_client.setex('macro_data', 3600, json.dumps(self.macro_data))
            
            logger.info(f"Updated macro data: DXY={dxy_proxy:.2f}, Gold=${gold['last']:.2f}")
            
            return self.macro_data
            
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            return self.macro_data
    
    def setup_websocket(self, pair: str):
        """Setup WebSocket connection for real-time data"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                self.ws_data[pair] = {
                    'price': float(data.get('p', 0)),
                    'volume': float(data.get('v', 0)),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"WebSocket message error for {pair}: {e}")
        
        def on_error(ws, error):
            logger.error(f"WebSocket error for {pair}: {error}")
        
        def on_close(ws):
            logger.info(f"WebSocket closed for {pair}")
        
        def on_open(ws):
            logger.info(f"WebSocket opened for {pair}")
        
        # Convert pair format for Binance WebSocket
        symbol = pair.replace('/', '').lower()
        ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
        
        ws = websocket.WebSocketApp(ws_url,
                                    on_message=on_message,
                                    on_error=on_error,
                                    on_close=on_close,
                                    on_open=on_open)
        
        # Run in separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        self.ws_connections[pair] = ws
    
    async def calculate_slippage(self, pair: str, size: float, side: str = 'buy') -> Dict:
        """Calculate expected slippage for a given order size"""
        try:
            orderbook = await self.exchange.fetch_order_book(pair, 50)
            
            if side == 'buy':
                orders = orderbook['asks']
            else:
                orders = orderbook['bids']
            
            total_cost = 0
            total_amount = 0
            levels_consumed = 0
            
            for price, amount in orders:
                if total_amount >= size:
                    break
                
                take_amount = min(amount, size - total_amount)
                total_cost += price * take_amount
                total_amount += take_amount
                levels_consumed += 1
            
            if total_amount > 0:
                avg_price = total_cost / total_amount
                market_price = orders[0][0]
                slippage_pct = abs((avg_price - market_price) / market_price) * 100
                
                return {
                    'avg_price': avg_price,
                    'market_price': market_price,
                    'slippage_pct': slippage_pct,
                    'levels_consumed': levels_consumed,
                    'total_cost': total_cost
                }
            
            return {'slippage_pct': 0, 'avg_price': 0, 'levels_consumed': 0}
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return {'slippage_pct': 0, 'avg_price': 0, 'levels_consumed': 0}
    
    async def run(self):
        """Main async data fetching loop"""
        logger.info("Real Data Fetcher Service started")
        logger.info(f"Fetching {len(self.pairs)} pairs every {self.fetch_interval}s")
        
        # Setup WebSocket connections for top pairs
        for pair in self.pairs[:10]:  # Limit WebSocket connections
            self.setup_websocket(pair)
        
        while True:
            try:
                # Fetch macro data
                await self.fetch_macro_data()
                
                # Fetch market data for all pairs
                tasks = [self.fetch_market_data(pair) for pair in self.pairs]
                results = await asyncio.gather(*tasks)
                
                # Calculate correlations
                await self.calculate_correlations()
                
                logger.info(f"Completed fetch cycle for {len([r for r in results if r])} pairs")
                
                await asyncio.sleep(self.fetch_interval)
                
            except KeyboardInterrupt:
                logger.info("Data fetcher shutting down")
                break
            except Exception as e:
                logger.error(f"Data fetcher error: {e}")
                await asyncio.sleep(10)
        
        # Cleanup
        await self.exchange.close()
        for ws in self.ws_connections.values():
            ws.close()
    
    async def calculate_correlations(self):
        """Calculate pair correlations"""
        try:
            if len(self.historical_data) < 2:
                return
            
            correlations = {}
            
            for pair1 in self.pairs[:10]:  # Limit to top 10 for performance
                if pair1 not in self.historical_data:
                    continue
                    
                prices1 = [d['price'] for d in self.historical_data[pair1] if d]
                if len(prices1) < 20:
                    continue
                
                for pair2 in self.pairs[:10]:
                    if pair1 == pair2 or pair2 not in self.historical_data:
                        continue
                    
                    prices2 = [d['price'] for d in self.historical_data[pair2] if d]
                    if len(prices2) < 20:
                        continue
                    
                    # Align lengths
                    min_len = min(len(prices1), len(prices2))
                    p1 = pd.Series(prices1[-min_len:])
                    p2 = pd.Series(prices2[-min_len:])
                    
                    # Calculate returns
                    returns1 = p1.pct_change().dropna()
                    returns2 = p2.pct_change().dropna()
                    
                    if len(returns1) > 0 and len(returns2) > 0:
                        correlation = returns1.corr(returns2)
                        correlations[f"{pair1}_{pair2}"] = correlation
            
            # Store correlations in Redis
            if correlations:
                self.redis_client.setex('pair_correlations', 300, json.dumps(correlations))
                logger.info(f"Updated {len(correlations)} pair correlations")
                
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")

if __name__ == "__main__":
    fetcher = DataFetcher()
    asyncio.run(fetcher.run())