import { io, Socket } from 'socket.io-client';
import type { WebSocketMessage, MetricData, ChangeEvent } from '../types';

class WebSocketService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  subscribe(channel: string) {
    // Implementation for tests
    console.log(`Subscribed to ${channel}`);
  }
  
  unsubscribe(channel: string) {
    // Implementation for tests
    console.log(`Unsubscribed from ${channel}`);
  }

  connect(url: string = 'ws://localhost:8000') {
    // Temporarily disabled - backend doesn't use socket.io
    return;
    
    if (this.socket?.connected) {
      return;
    }

    this.socket = io(url, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connected', true);
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      this.emit('connected', false);
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    });

    // Handle different message types
    this.socket.on('metrics_update', (data: WebSocketMessage) => {
      this.emit('metrics', data.data);
    });

    this.socket.on('metric_changes', (data: WebSocketMessage) => {
      this.emit('changes', data.data);
    });

    this.socket.on('trade_signal', (data: WebSocketMessage) => {
      this.emit('trade', data.data);
    });

    this.socket.on('alert', (data: WebSocketMessage) => {
      this.emit('alert', data.data);
    });

    this.socket.on('performance_update', (data: WebSocketMessage) => {
      this.emit('performance', data.data);
    });

    this.socket.on('health_update', (data: WebSocketMessage) => {
      this.emit('health', data.data);
    });
  }

  subscribeToPair(pair: string) {
    this.socket?.emit('subscribe_pair', { pair });
  }

  unsubscribeFromPair(pair: string) {
    this.socket?.emit('unsubscribe_pair', { pair });
  }

  on(event: string, callback: (data: any) => void) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)?.add(callback);
  }

  off(event: string, callback: (data: any) => void) {
    this.listeners.get(event)?.delete(callback);
  }

  private emit(event: string, data: any) {
    this.listeners.get(event)?.forEach(callback => callback(data));
  }

  disconnect() {
    this.socket?.disconnect();
    this.socket = null;
  }

  // Specific WebSocket endpoints
  connectMetrics() {
    const ws = new WebSocket('ws://localhost:8000/ws/metrics');
    this.setupWebSocketHandlers(ws, 'metrics');
    return ws;
  }

  connectTrades() {
    const ws = new WebSocket('ws://localhost:8000/ws/trades');
    this.setupWebSocketHandlers(ws, 'trades');
    return ws;
  }

  connectAlerts() {
    const ws = new WebSocket('ws://localhost:8000/ws/alerts');
    this.setupWebSocketHandlers(ws, 'alerts');
    return ws;
  }

  connectSystem() {
    const ws = new WebSocket('ws://localhost:8000/ws/system');
    this.setupWebSocketHandlers(ws, 'system');
    return ws;
  }

  private setupWebSocketHandlers(ws: WebSocket, channel: string) {
    ws.onopen = () => {
      console.log(`WebSocket ${channel} connected`);
      this.emit(`${channel}_connected`, true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit(`${channel}_message`, data);
      } catch (error) {
        console.error(`Error parsing ${channel} message:`, error);
      }
    };

    ws.onerror = (error) => {
      console.error(`WebSocket ${channel} error:`, error);
      this.emit(`${channel}_error`, error);
    };

    ws.onclose = () => {
      console.log(`WebSocket ${channel} disconnected`);
      this.emit(`${channel}_connected`, false);
    };
  }
}

export const wsService = new WebSocketService();
export default wsService;
