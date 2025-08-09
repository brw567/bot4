"""
Bridge module to connect existing Streamlit bot with Phase 4 React/FastAPI monitoring
"""
import redis
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, List
import sqlite3
from utils.db_utils import _get_conn, get_param, get_state
import logging



class MonitoringBridge:
    """Bridges the existing bot data with the new monitoring dashboard"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the monitoring bridge in a background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitoring_loop)
            self.thread.daemon = True
            self.thread.start()
            logger.info("✅ Monitoring bridge started")
    
    def stop(self):
        """Stop the monitoring bridge"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("⏹️  Monitoring bridge stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that publishes data to Redis"""
        while self.running:
            try:
                # Publish metrics
                self._publish_metrics()
                
                # Publish trading data
                self._publish_trades()
                
                # Publish system health
                self._publish_system_health()
                
                # Publish alerts
                self._publish_alerts()
                
                # Publish exchange status
                self._publish_exchange_status()
                
                # Publish trading pairs
                self._publish_trading_pairs()
                
                # Publish fee data every 60 seconds
                if int(time.time()) % 60 == 0:
                    self._publish_fee_data()
                
                time.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Monitoring bridge error: {e}")
                time.sleep(5)
    
    def _publish_metrics(self):
        """Extract and publish trading metrics from the database"""
        try:
            # Get all market data keys from Redis
            market_keys = []
            for key in self.redis_client.scan_iter("market:*"):
                market_keys.append(key)
            
            metrics = []
            
            # If no market data yet, try to get from database
            if not market_keys:
                conn = _get_conn()
                cursor = conn.cursor()
                
                # Get pairs from database
                cursor.execute("""
                    SELECT DISTINCT symbol FROM trades 
                    WHERE timestamp > datetime('now', '-24 hours')
                """)
                pairs = [row[0] for row in cursor.fetchall()]
                
                if not pairs:
                    self.redis_client.publish('metrics', json.dumps(metrics))
                    conn.close()
                    return
                    
                for pair in pairs:
                    # Get metrics for each pair
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_trades,
                            SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as profitable_trades,
                            AVG(price) as avg_price,
                            SUM(amount) as volume,
                            AVG(profit) as avg_profit
                        FROM trades
                        WHERE symbol = ? AND timestamp > datetime('now', '-24 hours')
                    """, (pair,))
                    
                    row = cursor.fetchone()
                    if row:
                        total_trades = row[0] or 0
                        profitable_trades = row[1] or 0
                        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                        
                        metric = {
                            "pair": pair,
                            "price": 0.0,  # Will be updated from market data
                            "volume": float(row[3]) if row[3] is not None else 0.0,
                            "change": 0.0,
                            "winRate": float(win_rate) if win_rate is not None else 0.0,
                            "totalTrades": int(total_trades) if total_trades is not None else 0,
                            "profitableTrades": int(profitable_trades) if profitable_trades is not None else 0,
                            "regime": "unknown",
                            "fundingRate": 0.01,  # Placeholder
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        metrics.append(metric)
                
                conn.close()
            else:
                # Use real-time market data from Redis
                conn = _get_conn()
                cursor = conn.cursor()
                
                for key in market_keys:
                    try:
                        market_data_str = self.redis_client.get(key)
                        if market_data_str:
                            market_data = json.loads(market_data_str)
                            pair = market_data.get('pair', '')
                            
                            # Get trade metrics from database
                            cursor.execute("""
                                SELECT 
                                    COUNT(*) as total_trades,
                                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as profitable_trades
                                FROM trades
                                WHERE symbol = ? AND timestamp > datetime('now', '-24 hours')
                            """, (pair,))
                            
                            row = cursor.fetchone()
                            total_trades = row[0] if row and row[0] else 0
                            profitable_trades = row[1] if row and row[1] else 0
                            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
                            
                            metric = {
                                "pair": pair,
                                "price": float(market_data.get('price', 0)),
                                "volume": float(market_data.get('volume', 0)),
                                "change": float(market_data.get('change24h', 0)),
                                "winRate": float(win_rate),
                                "totalTrades": int(total_trades),
                                "profitableTrades": int(profitable_trades),
                                "regime": "trending" if abs(market_data.get('change24h', 0)) > 2 else "ranging",
                                "fundingRate": 0.01,  # Placeholder
                                "timestamp": datetime.utcnow().isoformat()
                            }
                            metrics.append(metric)
                    except Exception as e:
                        logger.error(f"Error parsing market data from {key}: {e}")
                
                conn.close()
            
            # Publish to Redis
            if metrics:
                self.redis_client.publish('metrics', json.dumps(metrics))
            
        except Exception as e:
            logger.error(f"Error publishing metrics: {e}")
    
    def _publish_trades(self):
        """Extract and publish recent trades"""
        try:
            conn = _get_conn()
            cursor = conn.cursor()
            
            # Get recent trades
            cursor.execute("""
                SELECT 
                    id, symbol as pair, side, price, amount, fee, profit, timestamp, status
                FROM trades
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 50
            """)
            
            trades = []
            rows = cursor.fetchall()
            
            # If no trades yet, return empty list
            if not rows:
                self.redis_client.publish('trading', json.dumps(trades))
                conn.close()
                return
                
            for row in rows:
                price = float(row[3]) if row[3] is not None else 0.0
                amount = float(row[4]) if row[4] is not None else 0.0
                profit = float(row[6]) if row[6] is not None else 0.0
                total = price * amount
                profit_percent = (profit / total * 100) if total > 0 and profit != 0 else 0.0
                
                trade = {
                    "id": str(row[0]),
                    "pair": row[1],
                    "side": row[2] or "buy",
                    "price": price,
                    "amount": amount,
                    "total": total,
                    "fee": float(row[5]) if row[5] is not None else 0.0,
                    "profit": profit,
                    "profitPercent": profit_percent,
                    "time": row[7],
                    "status": row[8] or "completed"
                }
                trades.append(trade)
            
            if trades:
                self.redis_client.publish('trading', json.dumps(trades))
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error publishing trades: {e}")
    
    def _publish_system_health(self):
        """Publish system health metrics"""
        try:
            import psutil
            
            # Get bot status
            bot_status = get_state() or 'stopped'
            
            # Get real performance metrics from database
            conn = _get_conn()
            cursor = conn.cursor()
            
            # Get win rate from last 24 hours
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as profitable_trades,
                    SUM(profit) as total_profit
                FROM trades
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            row = cursor.fetchone()
            total_trades = row[0] if row and row[0] else 0
            profitable_trades = row[1] if row and row[1] else 0
            total_profit = row[2] if row and row[2] else 0
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Get API latency from Redis (if stored)
            binance_latency = 25  # Default
            try:
                # Try to get actual latency from bot
                from scalping_bot_headless import get_bot
                bot_instance = get_bot()
                if bot_instance and hasattr(bot_instance, 'client'):
                    # Measure actual API latency
                    start_time = time.time()
                    try:
                        bot_instance.client.fetch_ticker('BTC/USDC')
                        binance_latency = int((time.time() - start_time) * 1000)  # ms
                    except:
                        pass
            except:
                pass
            
            health = {
                "cpuUsage": psutil.cpu_percent(interval=1),
                "memoryUsage": psutil.virtual_memory().percent,
                "diskUsage": psutil.disk_usage('/').percent,
                "apiLatency": {
                    "binance": binance_latency,
                    "172.18.0.3": self._get_redis_latency()
                },
                "redisConnected": self._check_redis_connection(),
                "botStatus": bot_status,
                "performance": {
                    "winRate": round(win_rate, 2),
                    "totalTrades": total_trades,
                    "profitableTrades": profitable_trades,
                    "totalProfit": round(total_profit, 4) if total_profit else 0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            self.redis_client.publish('system', json.dumps(health))
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error publishing system health: {e}")
    
    def _publish_alerts(self):
        """Publish system alerts"""
        try:
            conn = _get_conn()
            cursor = conn.cursor()
            
            # Check for high-severity conditions
            alerts = []
            
            # Check for low win rate
            cursor.execute("""
                SELECT symbol as pair, 
                    COUNT(*) as total,
                    SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins
                FROM trades
                WHERE timestamp > datetime('now', '-1 hour')
                GROUP BY symbol
            """)
            
            rows = cursor.fetchall()
            if not rows:
                # No trades yet, no alerts
                self.redis_client.publish('alerts', json.dumps(alerts))
                conn.close()
                return
                
            for row in rows:
                pair = row[0]
                total = row[1] if row[1] is not None else 0
                wins = row[2] if row[2] is not None else 0
                win_rate = (wins / total * 100) if total > 0 else 0
                
                if win_rate < 40 and total > 10:
                    alert = {
                        "id": f"low-wr-{pair}-{int(time.time())}",
                        "type": "warning",
                        "category": "performance",
                        "severity": "high",
                        "title": f"Low Win Rate Alert - {pair}",
                        "message": f"Win rate for {pair} has dropped to {win_rate:.1f}%",
                        "timestamp": datetime.utcnow().isoformat(),
                        "pair": pair,
                        "read": False
                    }
                    alerts.append(alert)
            
            # Check for high drawdown
            cursor.execute("""
                WITH running_pnl AS (
                    SELECT 
                        timestamp,
                        profit,
                        SUM(profit) OVER (ORDER BY timestamp) as cumulative_pnl
                    FROM trades
                    WHERE timestamp > datetime('now', '-24 hours')
                )
                SELECT 
                    SUM(profit) as total_pnl,
                    MIN(cumulative_pnl) as max_drawdown
                FROM running_pnl
            """)
            
            row = cursor.fetchone()
            if row and row[0] and row[1]:
                drawdown_pct = abs(row[1] / row[0] * 100) if row[0] != 0 else 0
                if drawdown_pct > 10:
                    alert = {
                        "id": f"high-dd-{int(time.time())}",
                        "type": "danger",
                        "category": "risk",
                        "severity": "critical",
                        "title": "High Drawdown Alert",
                        "message": f"Portfolio drawdown has reached {drawdown_pct:.1f}%",
                        "timestamp": datetime.utcnow().isoformat(),
                        "read": False
                    }
                    alerts.append(alert)
            
            if alerts:
                self.redis_client.publish('alerts', json.dumps(alerts))
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error publishing alerts: {e}")
    
    def _publish_exchange_status(self):
        """Publish exchange status metrics"""
        try:
            # Try to get exchange status from bot
            exchange_status = []
            
            try:
                from scalping_bot_headless import get_bot
                bot_instance = get_bot()
                
                if bot_instance and hasattr(bot_instance, 'client'):
                    # Bot has single client (Binance)
                    client = bot_instance.client
                    if client:
                        # Get exchange status
                        status = {
                            "exchange": "binance",
                            "status": "healthy",
                            "uptime": 99.9,  # Will be calculated from bot uptime
                            "avgLatency": 100,  # Will be measured from actual API calls
                            "errorRate": 0,
                            "requestsTotal": 0,
                            "requestsSuccess": 0,
                            "lastUpdate": datetime.utcnow().isoformat()
                        }
                        
                        # Try to get real metrics
                        if hasattr(client, 'last_request_time'):
                            status["avgLatency"] = getattr(client, 'last_request_time', 100)
                        
                        exchange_status.append(status)
                else:
                    # No bot instance, return offline status
                    status = {
                        "exchange": "binance",
                        "status": "offline",
                        "uptime": 0,
                        "avgLatency": 0,
                        "errorRate": 0,
                        "requestsTotal": 0,
                        "requestsSuccess": 0,
                        "lastUpdate": datetime.utcnow().isoformat()
                    }
                    exchange_status.append(status)
            except Exception as e:
                # Import or other error
                status = {
                    "exchange": "binance",
                    "status": "error",
                    "uptime": 0,
                    "avgLatency": 0,
                    "errorRate": 100,
                    "requestsTotal": 0,
                    "requestsSuccess": 0,
                    "lastUpdate": datetime.utcnow().isoformat()
                }
                exchange_status.append(status)
            
            # Publish to Redis
            if exchange_status:
                self.redis_client.set('exchange_status', json.dumps(exchange_status))
                
        except Exception as e:
            logger.error(f"Error publishing exchange status: {e}")
    
    def _get_redis_latency(self) -> float:
        """Measure Redis latency"""
        import time
        start = time.time()
        try:
            self.redis_client.ping()
            return (time.time() - start) * 1000  # Convert to ms
        except:
            return 0
    
    def _check_redis_connection(self) -> bool:
        """Check if Redis is connected"""
        try:
            return self.redis_client.ping()
        except:
            return False
    
    def _publish_trading_pairs(self):
        """Publish active trading pairs from the bot"""
        try:
            # Try to get trading pairs from bot
            pairs_list = []
            
            try:
                from scalping_bot_headless import get_bot
                bot_instance = get_bot()
                
                if bot_instance and hasattr(bot_instance, 'active_pairs'):
                    # Get active pairs from bot
                    active_pairs = bot_instance.active_pairs
                    
                    for pair in active_pairs:
                        pair_data = {
                            "symbol": pair,
                            "enabled": True,
                            "exchange": "binance",
                            "lastUpdate": datetime.utcnow().isoformat()
                        }
                        pairs_list.append(pair_data)
                else:
                    # No bot instance, try to get from database
                    conn = _get_conn()
                    cursor = conn.cursor()
                    
                    # Get distinct pairs from recent trades
                    cursor.execute("""
                        SELECT DISTINCT symbol FROM trades 
                        WHERE timestamp > datetime('now', '-24 hours')
                    """)
                    pairs = [row[0] for row in cursor.fetchall()]
                    
                    # If no trades, use default pairs
                    if not pairs:
                        pairs = ['BNB/USDC', 'BTC/USDC', 'ETH/USDC', 'XRP/USDC', 'XLM/USDC', 
                                'LINK/USDC', 'LTC/USDC', 'TRX/USDC', 'ADA/USDC', 'NEO/USDC']
                    
                    for pair in pairs:
                        pair_data = {
                            "symbol": pair,
                            "enabled": True,
                            "exchange": "binance",
                            "lastUpdate": datetime.utcnow().isoformat()
                        }
                        pairs_list.append(pair_data)
                    
                    conn.close()
                    
            except Exception as e:
                # Import or other error - use default pairs
                default_pairs = ['BNB/USDC', 'BTC/USDC', 'ETH/USDC', 'XRP/USDC', 'XLM/USDC', 
                                'LINK/USDC', 'LTC/USDC', 'TRX/USDC', 'ADA/USDC', 'NEO/USDC']
                for pair in default_pairs:
                    pair_data = {
                        "symbol": pair,
                        "enabled": True,
                        "exchange": "binance",
                        "lastUpdate": datetime.utcnow().isoformat()
                    }
                    pairs_list.append(pair_data)
            
            # Publish to Redis
            if pairs_list:
                self.redis_client.set('trading_pairs', json.dumps(pairs_list))
                self.redis_client.publish('trading_pairs', json.dumps(pairs_list))
                
        except Exception as e:
            logger.error(f"Error publishing trading pairs: {e}")
    
    def _publish_fee_data(self):
        """Publish trading fee data"""
        try:
            # Import exchange manager if available
            try:
                from backend.services.exchange_manager import exchange_manager
                
                fee_data = {}
                
                # Get fees for each connected exchange
                for exchange_id in exchange_manager.exchanges:
                    # Use sync Redis client to check if we need to fetch
                    fees = {
                        "spot": {
                            "maker": 0.001,  # 0.1%
                            "taker": 0.001   # 0.1%
                        },
                        "futures": {
                            "maker": 0.0002,  # 0.02%
                            "taker": 0.0004   # 0.04%
                        }
                    }
                    
                    # For Binance, use specific fees
                    if exchange_id == 'binance':
                        fees = {
                            "spot": {
                                "maker": 0.001,  # 0.1%
                                "taker": 0.001   # 0.1%
                            },
                            "futures": {
                                "maker": 0.0002,  # 0.02%
                                "taker": 0.0004   # 0.04%
                            }
                        }
                    
                    fee_data[exchange_id] = fees
                
                # Store in Redis
                self.redis_client.set('trading_fees', json.dumps(fee_data))
                
            except Exception as e:
                # Use default fees
                default_fees = {
                    "binance": {
                        "spot": {
                            "maker": 0.001,  # 0.1%
                            "taker": 0.001   # 0.1%
                        },
                        "futures": {
                            "maker": 0.0002,  # 0.02%
                            "taker": 0.0004   # 0.04%
                        }
                    }
                }
                self.redis_client.set('trading_fees', json.dumps(default_fees))
                
        except Exception as e:
            logger.error(f"Error publishing fee data: {e}")

logger = logging.getLogger(__name__)


# Global instance
_monitoring_bridge = None

def get_monitoring_bridge():
    """Get or create the monitoring bridge instance"""
    global _monitoring_bridge
    if _monitoring_bridge is None:
        _monitoring_bridge = MonitoringBridge()
    return _monitoring_bridge

def start_monitoring():
    """Start the monitoring bridge"""
    bridge = get_monitoring_bridge()
    bridge.start()

def stop_monitoring():
    """Stop the monitoring bridge"""
    bridge = get_monitoring_bridge()
    bridge.stop()