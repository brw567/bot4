/**
 * Frontend logging service that sends logs to backend
 */

interface LogEntry {
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  timestamp: string;
  context?: any;
  userAgent?: string;
  url?: string;
}

class FrontendLogger {
  private logQueue: LogEntry[] = [];
  private isOnline = true;
  private flushInterval: number;
  private maxQueueSize = 100;

  constructor() {
    // Check online status
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.flushLogs();
    });
    
    window.addEventListener('offline', () => {
      this.isOnline = false;
    });

    // Flush logs every 5 seconds
    this.flushInterval = window.setInterval(() => {
      this.flushLogs();
    }, 5000);

    // Capture unhandled errors
    window.addEventListener('error', (event) => {
      this.error('Unhandled error', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        error: event.error?.stack
      });
    });

    // Capture unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.error('Unhandled promise rejection', {
        reason: event.reason,
        promise: event.promise
      });
    });
  }

  private createLogEntry(
    level: LogEntry['level'], 
    message: string, 
    context?: any
  ): LogEntry {
    return {
      level,
      message,
      timestamp: new Date().toISOString(),
      context,
      userAgent: navigator.userAgent,
      url: window.location.href
    };
  }

  private addToQueue(entry: LogEntry) {
    this.logQueue.push(entry);
    
    // Prevent queue from growing too large
    if (this.logQueue.length > this.maxQueueSize) {
      this.logQueue = this.logQueue.slice(-this.maxQueueSize);
    }

    // Also log to console in development
    if (process.env.NODE_ENV === 'development') {
      const consoleMethod = entry.level === 'error' ? 'error' : 
                          entry.level === 'warn' ? 'warn' : 
                          entry.level === 'debug' ? 'debug' : 'log';
      console[consoleMethod](`[${entry.level.toUpperCase()}] ${entry.message}`, entry.context);
    }
  }

  private async flushLogs() {
    if (!this.isOnline || this.logQueue.length === 0) {
      return;
    }

    const logsToSend = [...this.logQueue];
    this.logQueue = [];

    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/api/logs/frontend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        },
        body: JSON.stringify({ logs: logsToSend })
      });

      if (!response.ok) {
        // Put logs back in queue if sending failed
        this.logQueue = [...logsToSend, ...this.logQueue];
      }
    } catch (error) {
      // Put logs back in queue if sending failed
      this.logQueue = [...logsToSend, ...this.logQueue];
    }
  }

  debug(message: string, context?: any) {
    this.addToQueue(this.createLogEntry('debug', message, context));
  }

  info(message: string, context?: any) {
    this.addToQueue(this.createLogEntry('info', message, context));
  }

  warn(message: string, context?: any) {
    this.addToQueue(this.createLogEntry('warn', message, context));
  }

  error(message: string, context?: any) {
    this.addToQueue(this.createLogEntry('error', message, context));
  }

  // Log authentication attempts
  logAuth(action: 'login' | 'logout' | 'token_refresh', success: boolean, details?: any) {
    this.info(`Auth ${action}`, {
      action,
      success,
      ...details
    });
  }

  // Log API calls
  logApiCall(method: string, url: string, status: number, duration: number) {
    const level = status >= 400 ? 'error' : 'info';
    this.addToQueue(this.createLogEntry(level, `API ${method} ${url}`, {
      method,
      url,
      status,
      duration
    }));
  }

  // Clean up on destroy
  destroy() {
    if (this.flushInterval) {
      clearInterval(this.flushInterval);
    }
    this.flushLogs();
  }
}

// Create singleton instance
export const logger = new FrontendLogger();

// Export for use in other modules
export default logger;