"""
Unified Exchange Manager - Binance and Coinbase Integration
Version: 2.0.0
Features: Unified interface, rate limiting, error handling, websocket streams
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import json
import hmac
import hashlib
import time

import ccxt.async_support as ccxt
import aiohttp
from aiohttp import ClientSession
import websockets
from prometheus_client import Counter, Histogram, Gauge
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
exchange_requests_total = Counter('exchange_requests_total', 'Total exchange API requests', ['exchange', 'endpoint'])
exchange_errors_total = Counter('exchange_errors_total', 'Total exchange API errors', ['exchange', 'error_type'])
exchange_latency = Histogram('exchange_request_duration_seconds', 'Exchange API latency', ['exchange', 'endpoint'])
websocket_connections = Gauge('exchange_websocket_connections', 'Active WebSocket connections', ['exchange'])

# ============================================================================
# DATA MODELS
# ============================================================================

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"

class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    name: str
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None  # For Coinbase
    testnet: bool = False
    rate_limit: int = 50  # requests per second
    ws_url: Optional[str] = None
    rest_url: Optional[str] = None

@dataclass
class Ticker:
    """Unified ticker data"""
    symbol: str
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal
    volume_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    change_24h: Decimal
    change_percent_24h: Decimal

@dataclass
class OrderBook:
    """Unified order book"""
    symbol: str
    timestamp: datetime
    bids: List[tuple[Decimal, Decimal]]  # price, quantity
    asks: List[tuple[Decimal, Decimal]]
    
    @property
    def spread(self) -> Decimal:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return Decimal('0')
    
    @property
    def mid_price(self) -> Decimal:
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2
        return Decimal('0')

@dataclass
class Trade:
    """Unified trade data"""
    id: str
    symbol: str
    timestamp: datetime
    side: OrderSide
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_currency: str

@dataclass
class Order:
    """Unified order data"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    status: OrderStatus
    price: Optional[Decimal]
    quantity: Decimal
    filled: Decimal
    remaining: Decimal
    timestamp: datetime
    client_order_id: Optional[str] = None

@dataclass
class Balance:
    """Account balance"""
    currency: str
    free: Decimal
    used: Decimal
    total: Decimal

# ============================================================================
# BASE EXCHANGE CONNECTOR
# ============================================================================

class BaseExchangeConnector:
    """Base class for exchange connectors"""
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange: Optional[ccxt.Exchange] = None
        self.ws_connection = None
        self.ws_subscriptions: Dict[str, List[Callable]] = {}
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize exchange connection"""
        raise NotImplementedError
        
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker data"""
        raise NotImplementedError
        
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get order book"""
        raise NotImplementedError
        
    async def place_order(self, symbol: str, side: OrderSide, type: OrderType,
                          quantity: Decimal, price: Optional[Decimal] = None) -> Order:
        """Place an order"""
        raise NotImplementedError
        
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order"""
        raise NotImplementedError
        
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Get order status"""
        raise NotImplementedError
        
    async def get_balances(self) -> List[Balance]:
        """Get account balances"""
        raise NotImplementedError
        
    async def subscribe_ticker(self, symbol: str, callback: Callable):
        """Subscribe to ticker updates"""
        raise NotImplementedError
        
    async def subscribe_order_book(self, symbol: str, callback: Callable):
        """Subscribe to order book updates"""
        raise NotImplementedError

# ============================================================================
# BINANCE CONNECTOR
# ============================================================================

class BinanceConnector(BaseExchangeConnector):
    """Binance exchange connector"""
    
    async def initialize(self):
        """Initialize Binance connection"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # or 'future' for futures
                    'adjustForTimeDifference': True
                }
            })
            
            if self.config.testnet:
                self.exchange.set_sandbox_mode(True)
                
            # Test connection
            await self.exchange.load_markets()
            
            # Initialize WebSocket
            if self.config.ws_url:
                asyncio.create_task(self._ws_connect())
                
            logger.info("Binance connector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance connector: {e}")
            raise
            
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get Binance ticker"""
        with exchange_latency.labels(exchange='binance', endpoint='ticker').time():
            try:
                await self.rate_limiter.acquire()
                
                ticker = await self.exchange.fetch_ticker(symbol)
                
                exchange_requests_total.labels(
                    exchange='binance',
                    endpoint='ticker'
                ).inc()
                
                return Ticker(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    bid=Decimal(str(ticker['bid'])),
                    ask=Decimal(str(ticker['ask'])),
                    last=Decimal(str(ticker['last'])),
                    volume_24h=Decimal(str(ticker['quoteVolume'])),
                    high_24h=Decimal(str(ticker['high'])),
                    low_24h=Decimal(str(ticker['low'])),
                    change_24h=Decimal(str(ticker['change'])),
                    change_percent_24h=Decimal(str(ticker['percentage']))
                )
                
            except Exception as e:
                exchange_errors_total.labels(
                    exchange='binance',
                    error_type=type(e).__name__
                ).inc()
                logger.error(f"Error fetching Binance ticker: {e}")
                raise
                
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get Binance order book"""
        with exchange_latency.labels(exchange='binance', endpoint='orderbook').time():
            try:
                await self.rate_limiter.acquire()
                
                book = await self.exchange.fetch_order_book(symbol, limit)
                
                exchange_requests_total.labels(
                    exchange='binance',
                    endpoint='orderbook'
                ).inc()
                
                return OrderBook(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(book['timestamp'] / 1000),
                    bids=[(Decimal(str(bid[0])), Decimal(str(bid[1]))) for bid in book['bids']],
                    asks=[(Decimal(str(ask[0])), Decimal(str(ask[1]))) for ask in book['asks']]
                )
                
            except Exception as e:
                exchange_errors_total.labels(
                    exchange='binance',
                    error_type=type(e).__name__
                ).inc()
                logger.error(f"Error fetching Binance order book: {e}")
                raise
                
    async def place_order(self, symbol: str, side: OrderSide, type: OrderType,
                          quantity: Decimal, price: Optional[Decimal] = None) -> Order:
        """Place order on Binance"""
        with exchange_latency.labels(exchange='binance', endpoint='place_order').time():
            try:
                await self.rate_limiter.acquire()
                
                # Convert to ccxt format
                ccxt_side = side.value
                ccxt_type = self._map_order_type(type)
                
                params = {}
                if type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
                    params['stopPrice'] = float(price)
                    
                if ccxt_type == 'market':
                    order = await self.exchange.create_market_order(
                        symbol, ccxt_side, float(quantity), params
                    )
                else:
                    order = await self.exchange.create_limit_order(
                        symbol, ccxt_side, float(quantity), float(price), params
                    )
                    
                exchange_requests_total.labels(
                    exchange='binance',
                    endpoint='place_order'
                ).inc()
                
                return self._parse_order(order)
                
            except Exception as e:
                exchange_errors_total.labels(
                    exchange='binance',
                    error_type=type(e).__name__
                ).inc()
                logger.error(f"Error placing Binance order: {e}")
                raise
                
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order on Binance"""
        try:
            await self.rate_limiter.acquire()
            
            result = await self.exchange.cancel_order(order_id, symbol)
            
            exchange_requests_total.labels(
                exchange='binance',
                endpoint='cancel_order'
            ).inc()
            
            return result.get('status') == 'canceled'
            
        except Exception as e:
            exchange_errors_total.labels(
                exchange='binance',
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Error cancelling Binance order: {e}")
            return False
            
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Get order from Binance"""
        try:
            await self.rate_limiter.acquire()
            
            order = await self.exchange.fetch_order(order_id, symbol)
            
            exchange_requests_total.labels(
                exchange='binance',
                endpoint='get_order'
            ).inc()
            
            return self._parse_order(order)
            
        except Exception as e:
            exchange_errors_total.labels(
                exchange='binance',
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Error fetching Binance order: {e}")
            raise
            
    async def get_balances(self) -> List[Balance]:
        """Get Binance account balances"""
        try:
            await self.rate_limiter.acquire()
            
            balance = await self.exchange.fetch_balance()
            
            exchange_requests_total.labels(
                exchange='binance',
                endpoint='balance'
            ).inc()
            
            balances = []
            for currency, data in balance['total'].items():
                if data > 0:
                    balances.append(Balance(
                        currency=currency,
                        free=Decimal(str(balance['free'].get(currency, 0))),
                        used=Decimal(str(balance['used'].get(currency, 0))),
                        total=Decimal(str(data))
                    ))
                    
            return balances
            
        except Exception as e:
            exchange_errors_total.labels(
                exchange='binance',
                error_type=type(e).__name__
            ).inc()
            logger.error(f"Error fetching Binance balances: {e}")
            raise
            
    async def _ws_connect(self):
        """Connect to Binance WebSocket"""
        ws_url = self.config.ws_url or "wss://stream.binance.com:9443/ws"
        
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    self.ws_connection = ws
                    websocket_connections.labels(exchange='binance').inc()
                    logger.info("Connected to Binance WebSocket")
                    
                    # Subscribe to streams
                    await self._ws_subscribe()
                    
                    # Handle messages
                    async for message in ws:
                        await self._ws_handle_message(json.loads(message))
                        
            except Exception as e:
                logger.error(f"Binance WebSocket error: {e}")
                websocket_connections.labels(exchange='binance').dec()
                await asyncio.sleep(5)  # Reconnect after 5 seconds
                
    async def _ws_subscribe(self):
        """Subscribe to Binance WebSocket streams"""
        if not self.ws_connection:
            return
            
        # Subscribe to configured streams
        streams = []
        for symbol in self.ws_subscriptions.keys():
            # Convert symbol format (BTC/USDT -> btcusdt)
            stream_symbol = symbol.replace('/', '').lower()
            streams.extend([
                f"{stream_symbol}@ticker",
                f"{stream_symbol}@depth20"
            ])
            
        if streams:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": 1
            }
            await self.ws_connection.send(json.dumps(subscribe_msg))
            
    async def _ws_handle_message(self, message: Dict):
        """Handle Binance WebSocket message"""
        if 'e' not in message:
            return
            
        event_type = message['e']
        
        if event_type == '24hrTicker':
            # Ticker update
            symbol = self._parse_ws_symbol(message['s'])
            ticker = Ticker(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(message['E'] / 1000),
                bid=Decimal(message['b']),
                ask=Decimal(message['a']),
                last=Decimal(message['c']),
                volume_24h=Decimal(message['q']),
                high_24h=Decimal(message['h']),
                low_24h=Decimal(message['l']),
                change_24h=Decimal(message['p']),
                change_percent_24h=Decimal(message['P'])
            )
            
            # Call registered callbacks
            for callback in self.ws_subscriptions.get(f"{symbol}:ticker", []):
                await callback(ticker)
                
        elif event_type == 'depthUpdate':
            # Order book update
            symbol = self._parse_ws_symbol(message['s'])
            # Process order book update
            # ... (implementation details)
            
    def _map_order_type(self, type: OrderType) -> str:
        """Map order type to ccxt format"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.STOP_LOSS: 'stop_loss',
            OrderType.STOP_LIMIT: 'stop_loss_limit',
            OrderType.TAKE_PROFIT: 'take_profit',
            OrderType.TAKE_PROFIT_LIMIT: 'take_profit_limit'
        }
        return mapping.get(type, 'limit')
        
    def _parse_order(self, order: Dict) -> Order:
        """Parse ccxt order to unified format"""
        return Order(
            id=order['id'],
            symbol=order['symbol'],
            side=OrderSide(order['side']),
            type=self._parse_order_type(order['type']),
            status=self._parse_order_status(order['status']),
            price=Decimal(str(order['price'])) if order['price'] else None,
            quantity=Decimal(str(order['amount'])),
            filled=Decimal(str(order['filled'])),
            remaining=Decimal(str(order['remaining'])),
            timestamp=datetime.fromtimestamp(order['timestamp'] / 1000),
            client_order_id=order.get('clientOrderId')
        )
        
    def _parse_order_type(self, type_str: str) -> OrderType:
        """Parse order type string"""
        for order_type in OrderType:
            if order_type.value == type_str:
                return order_type
        return OrderType.LIMIT
        
    def _parse_order_status(self, status: str) -> OrderStatus:
        """Parse order status"""
        status_map = {
            'open': OrderStatus.NEW,
            'closed': OrderStatus.FILLED,
            'canceled': OrderStatus.CANCELLED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED
        }
        return status_map.get(status, OrderStatus.NEW)
        
    def _parse_ws_symbol(self, symbol: str) -> str:
        """Parse WebSocket symbol format"""
        # BTCUSDT -> BTC/USDT
        if 'USDT' in symbol:
            base = symbol.replace('USDT', '')
            return f"{base}/USDT"
        return symbol

# ============================================================================
# COINBASE CONNECTOR
# ============================================================================

class CoinbaseConnector(BaseExchangeConnector):
    """Coinbase exchange connector"""
    
    async def initialize(self):
        """Initialize Coinbase connection"""
        try:
            self.exchange = ccxt.coinbase({
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'password': self.config.passphrase,  # Coinbase requires passphrase
                'enableRateLimit': True
            })
            
            if self.config.testnet:
                self.exchange.urls['api'] = 'https://api-public.sandbox.exchange.coinbase.com'
                
            # Test connection
            await self.exchange.load_markets()
            
            # Initialize WebSocket
            if self.config.ws_url:
                asyncio.create_task(self._ws_connect())
                
            logger.info("Coinbase connector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Coinbase connector: {e}")
            raise
            
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get Coinbase ticker"""
        with exchange_latency.labels(exchange='coinbase', endpoint='ticker').time():
            try:
                await self.rate_limiter.acquire()
                
                ticker = await self.exchange.fetch_ticker(symbol)
                
                exchange_requests_total.labels(
                    exchange='coinbase',
                    endpoint='ticker'
                ).inc()
                
                return Ticker(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(ticker['timestamp'] / 1000),
                    bid=Decimal(str(ticker['bid'])),
                    ask=Decimal(str(ticker['ask'])),
                    last=Decimal(str(ticker['last'])),
                    volume_24h=Decimal(str(ticker['quoteVolume'])),
                    high_24h=Decimal(str(ticker['high'])),
                    low_24h=Decimal(str(ticker['low'])),
                    change_24h=Decimal(str(ticker.get('change', 0))),
                    change_percent_24h=Decimal(str(ticker.get('percentage', 0)))
                )
                
            except Exception as e:
                exchange_errors_total.labels(
                    exchange='coinbase',
                    error_type=type(e).__name__
                ).inc()
                logger.error(f"Error fetching Coinbase ticker: {e}")
                raise
                
    async def get_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get Coinbase order book"""
        with exchange_latency.labels(exchange='coinbase', endpoint='orderbook').time():
            try:
                await self.rate_limiter.acquire()
                
                book = await self.exchange.fetch_order_book(symbol, limit)
                
                exchange_requests_total.labels(
                    exchange='coinbase',
                    endpoint='orderbook'
                ).inc()
                
                return OrderBook(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(book['timestamp'] / 1000),
                    bids=[(Decimal(str(bid[0])), Decimal(str(bid[1]))) for bid in book['bids']],
                    asks=[(Decimal(str(ask[0])), Decimal(str(ask[1]))) for ask in book['asks']]
                )
                
            except Exception as e:
                exchange_errors_total.labels(
                    exchange='coinbase',
                    error_type=type(e).__name__
                ).inc()
                logger.error(f"Error fetching Coinbase order book: {e}")
                raise
                
    # ... (Similar implementations for other methods)
    
    async def _ws_connect(self):
        """Connect to Coinbase WebSocket"""
        ws_url = self.config.ws_url or "wss://ws-feed.exchange.coinbase.com"
        
        while True:
            try:
                async with websockets.connect(ws_url) as ws:
                    self.ws_connection = ws
                    websocket_connections.labels(exchange='coinbase').inc()
                    logger.info("Connected to Coinbase WebSocket")
                    
                    # Authenticate
                    await self._ws_authenticate()
                    
                    # Subscribe to channels
                    await self._ws_subscribe()
                    
                    # Handle messages
                    async for message in ws:
                        await self._ws_handle_message(json.loads(message))
                        
            except Exception as e:
                logger.error(f"Coinbase WebSocket error: {e}")
                websocket_connections.labels(exchange='coinbase').dec()
                await asyncio.sleep(5)
                
    async def _ws_authenticate(self):
        """Authenticate Coinbase WebSocket"""
        if not self.ws_connection:
            return
            
        timestamp = str(time.time())
        message = timestamp + 'GET' + '/users/self/verify'
        
        signature = hmac.new(
            self.config.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        auth_msg = {
            "type": "subscribe",
            "product_ids": [],
            "channels": ["user"],
            "signature": signature,
            "key": self.config.api_key,
            "passphrase": self.config.passphrase,
            "timestamp": timestamp
        }
        
        await self.ws_connection.send(json.dumps(auth_msg))

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, rate: int):
        self.rate = rate  # requests per second
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self):
        """Acquire a token for making a request"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                sleep_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(sleep_time)
                self.tokens = 0
            else:
                self.tokens -= 1

# ============================================================================
# UNIFIED EXCHANGE MANAGER
# ============================================================================

class ExchangeManager:
    """Manages multiple exchange connections"""
    
    def __init__(self):
        self.exchanges: Dict[str, BaseExchangeConnector] = {}
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize exchange manager"""
        # Connect to Redis
        self.redis_client = await redis.from_url(
            "redis://redis-master.rc5-trading.svc.cluster.local:6379",
            decode_responses=True
        )
        
        # Load exchange configurations from environment or config file
        configs = self._load_configs()
        
        # Initialize exchanges
        for config in configs:
            await self.add_exchange(config)
            
        logger.info(f"Exchange Manager initialized with {len(self.exchanges)} exchanges")
        
    def _load_configs(self) -> List[ExchangeConfig]:
        """Load exchange configurations"""
        # This would load from environment variables or config file
        configs = []
        
        # Binance config
        binance_config = ExchangeConfig(
            name="binance",
            api_key="",  # Load from environment
            api_secret="",  # Load from environment
            testnet=False,
            rate_limit=50,
            ws_url="wss://stream.binance.com:9443/ws"
        )
        configs.append(binance_config)
        
        # Coinbase config
        coinbase_config = ExchangeConfig(
            name="coinbase",
            api_key="",  # Load from environment
            api_secret="",  # Load from environment
            passphrase="",  # Load from environment
            testnet=False,
            rate_limit=10,
            ws_url="wss://ws-feed.exchange.coinbase.com"
        )
        configs.append(coinbase_config)
        
        return configs
        
    async def add_exchange(self, config: ExchangeConfig):
        """Add an exchange"""
        if config.name == "binance":
            connector = BinanceConnector(config)
        elif config.name == "coinbase":
            connector = CoinbaseConnector(config)
        else:
            raise ValueError(f"Unknown exchange: {config.name}")
            
        await connector.initialize()
        self.exchanges[config.name] = connector
        
        logger.info(f"Added exchange: {config.name}")
        
    async def get_best_price(self, symbol: str, side: OrderSide) -> tuple[str, Decimal]:
        """Get best price across all exchanges"""
        best_exchange = None
        best_price = None
        
        for name, exchange in self.exchanges.items():
            try:
                ticker = await exchange.get_ticker(symbol)
                
                if side == OrderSide.BUY:
                    price = ticker.ask
                    if best_price is None or price < best_price:
                        best_price = price
                        best_exchange = name
                else:
                    price = ticker.bid
                    if best_price is None or price > best_price:
                        best_price = price
                        best_exchange = name
                        
            except Exception as e:
                logger.error(f"Error getting price from {name}: {e}")
                
        return best_exchange, best_price
        
    async def execute_smart_order(self, symbol: str, side: OrderSide, 
                                 quantity: Decimal, type: OrderType = OrderType.MARKET) -> Order:
        """Execute order on best exchange"""
        # Find best price
        best_exchange, best_price = await self.get_best_price(symbol, side)
        
        if not best_exchange:
            raise ValueError("No exchange available for order")
            
        # Place order on best exchange
        exchange = self.exchanges[best_exchange]
        
        logger.info(f"Executing order on {best_exchange} at {best_price}")
        
        return await exchange.place_order(
            symbol=symbol,
            side=side,
            type=type,
            quantity=quantity,
            price=best_price if type == OrderType.LIMIT else None
        )
        
    async def get_aggregated_order_book(self, symbol: str, limit: int = 20) -> OrderBook:
        """Get aggregated order book from all exchanges"""
        all_bids = []
        all_asks = []
        
        for name, exchange in self.exchanges.items():
            try:
                book = await exchange.get_order_book(symbol, limit)
                all_bids.extend(book.bids)
                all_asks.extend(book.asks)
            except Exception as e:
                logger.error(f"Error getting order book from {name}: {e}")
                
        # Sort and aggregate
        all_bids.sort(key=lambda x: x[0], reverse=True)  # Highest bid first
        all_asks.sort(key=lambda x: x[0])  # Lowest ask first
        
        return OrderBook(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            bids=all_bids[:limit],
            asks=all_asks[:limit]
        )
        
    async def get_total_balance(self, currency: str) -> Decimal:
        """Get total balance across all exchanges"""
        total = Decimal('0')
        
        for name, exchange in self.exchanges.items():
            try:
                balances = await exchange.get_balances()
                for balance in balances:
                    if balance.currency == currency:
                        total += balance.total
                        break
            except Exception as e:
                logger.error(f"Error getting balance from {name}: {e}")
                
        return total

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Exchange Manager API", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exchange manager
manager = ExchangeManager()

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    await manager.initialize()

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "exchanges": list(manager.exchanges.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/exchanges")
async def get_exchanges():
    """Get available exchanges"""
    return {"exchanges": list(manager.exchanges.keys())}

@app.get("/ticker/{exchange}/{symbol}")
async def get_ticker(exchange: str, symbol: str):
    """Get ticker from specific exchange"""
    if exchange not in manager.exchanges:
        raise HTTPException(status_code=404, detail="Exchange not found")
        
    try:
        ticker = await manager.exchanges[exchange].get_ticker(symbol)
        return ticker.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/best-price/{symbol}")
async def get_best_price(symbol: str, side: str):
    """Get best price across exchanges"""
    try:
        order_side = OrderSide(side)
        exchange, price = await manager.get_best_price(symbol, order_side)
        return {
            "exchange": exchange,
            "price": str(price),
            "side": side
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart-order")
async def place_smart_order(symbol: str, side: str, quantity: float):
    """Place smart order on best exchange"""
    try:
        order = await manager.execute_smart_order(
            symbol=symbol,
            side=OrderSide(side),
            quantity=Decimal(str(quantity))
        )
        return order.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)