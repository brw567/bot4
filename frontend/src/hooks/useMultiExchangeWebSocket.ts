import { useEffect, useState, useRef, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import logger from '../services/logger';

interface MultiExchangeUpdate {
  type: 'multi_exchange_update';
  timestamp: string;
  data: Record<string, any>;
}

interface ArbitrageAlert {
  type: 'arbitrage_alert';
  timestamp: string;
  opportunities: Array<{
    symbol: string;
    profit: string;
    buy: string;
    sell: string;
    volume: number;
  }>;
}

type WebSocketMessage = MultiExchangeUpdate | ArbitrageAlert;

interface UseMultiExchangeWebSocketOptions {
  channel: 'multi-exchange' | 'arbitrage';
  enabled?: boolean;
  onMessage?: (data: WebSocketMessage) => void;
  reconnectInterval?: number;
}

export const useMultiExchangeWebSocket = ({
  channel,
  enabled = true,
  onMessage,
  reconnectInterval = 5000
}: UseMultiExchangeWebSocketOptions) => {
  const { token } = useAuth();
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!token || !enabled) return;

    try {
      const wsUrl = `ws://localhost:8000/ws/${channel}`;
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        logger.info(`Multi-Exchange WebSocket connected to ${channel}`);
        setIsConnected(true);
        setError(null);
        
        // Send authentication if needed
        ws.send(JSON.stringify({ type: 'auth', token }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WebSocketMessage;
          setLastMessage(data);
          onMessage?.(data);
        } catch (err) {
          logger.error('Failed to parse WebSocket message:', err);
        }
      };

      ws.onerror = (event) => {
        logger.error(`WebSocket error on ${channel}:`, event);
        setError('WebSocket connection error');
      };

      ws.onclose = (event) => {
        logger.info(`WebSocket disconnected from ${channel}:`, event.code, event.reason);
        setIsConnected(false);
        wsRef.current = null;

        // Attempt to reconnect
        if (enabled && !reconnectTimeoutRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connect();
          }, reconnectInterval);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      logger.error('Failed to create WebSocket connection:', err);
      setError('Failed to connect');
    }
  }, [token, enabled, channel, onMessage, reconnectInterval]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      logger.warn('Cannot send message: WebSocket not connected');
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [enabled, connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    reconnect: connect
  };
};

// Specific hooks for each channel
export const useMultiExchangeUpdates = (options?: Partial<UseMultiExchangeWebSocketOptions>) => {
  return useMultiExchangeWebSocket({
    channel: 'multi-exchange',
    ...options
  });
};

export const useArbitrageAlerts = (options?: Partial<UseMultiExchangeWebSocketOptions>) => {
  return useMultiExchangeWebSocket({
    channel: 'arbitrage',
    ...options
  });
};