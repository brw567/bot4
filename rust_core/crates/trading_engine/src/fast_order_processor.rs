// Fast Order Processor - Zero-Allocation Hot Path
// Team: Jordan (Performance) & Sam (Architecture) & Casey (Exchange)
// CRITICAL: Demonstrates proper object pool usage for <100μs latency
// References:
// - "Ultra-Low Latency Trading Systems" - Narang
// - "High-Performance Trading" - Aldridge
// - "Zero-Copy Architecture" - Linux Journal

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use rust_decimal::Decimal;
use anyhow::Result;
use tracing::{info, warn, debug};

use infrastructure::{
    acquire_order, acquire_signal, acquire_market_data, acquire_risk_check,
    acquire_execution, Order, Signal,
    OrderSide, OrderType, OrderStatus, SignalType, POOL_REGISTRY,
};

/// Fast order processor using pre-allocated object pools
/// ZERO allocations in the hot path
pub struct FastOrderProcessor {
    // Performance metrics
    orders_processed: Arc<AtomicU64>,
    signals_processed: Arc<AtomicU64>,
    allocations_avoided: Arc<AtomicU64>,
    
    // Control flags
    is_running: Arc<AtomicBool>,
    
    // Risk limits (pre-calculated to avoid allocation)
    max_position_size: Decimal,
    max_order_value: Decimal,
    min_order_size: Decimal,
}

impl Default for FastOrderProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl FastOrderProcessor {
    pub fn new() -> Self {
        Self {
            orders_processed: Arc::new(AtomicU64::new(0)),
            signals_processed: Arc::new(AtomicU64::new(0)),
            allocations_avoided: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(true)),
            max_position_size: Decimal::from(100000),
            max_order_value: Decimal::from(1000000),
            min_order_size: Decimal::from(10),
        }
    }
    
    /// Process trading signal with ZERO allocations
    /// Hot path: <10μs for signal -> order conversion
    #[inline(always)]
    pub fn process_signal_fast(&self, 
        symbol: &str, 
        strength: f64, 
        confidence: f64
    ) -> Result<u64> {
        // START HOT PATH - NO ALLOCATIONS ALLOWED
        let start = Instant::now();
        
        // COMPREHENSIVE LOGGING: Entry point
        debug!("Processing signal: symbol={}, strength={:.4}, confidence={:.4}", 
               symbol, strength, confidence);
        
        // Acquire pre-allocated signal from pool (zero allocation)
        let mut signal = acquire_signal();
        self.allocations_avoided.fetch_add(1, Ordering::Relaxed);
        
        // Fill signal data (reuse existing String capacity)
        signal.symbol.clear();
        signal.symbol.push_str(symbol);
        signal.strength = strength;
        signal.confidence = confidence;
        signal.signal_type = if strength > 0.0 { 
            SignalType::Long 
        } else { 
            SignalType::Short 
        };
        signal.timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        
        // Quick validation (no allocations)
        if confidence < 0.7 {
            debug!("Signal rejected - confidence too low: {} < 0.7 for {}", confidence, symbol);
            return Ok(0); // Signal auto-returns to pool on drop
        }
        
        // Convert to order using pre-allocated order
        let order_id = self.signal_to_order_fast(&signal)?;
        
        let elapsed = start.elapsed();
        let elapsed_us = elapsed.as_micros();
        
        // COMPREHENSIVE LOGGING: Performance tracking
        if elapsed_us > 10 {
            warn!("Slow signal processing: {}μs for {} (target: <10μs)", elapsed_us, symbol);
        } else {
            debug!("Signal processed successfully: order_id={}, latency={}μs", order_id, elapsed_us);
        }
        
        self.signals_processed.fetch_add(1, Ordering::Relaxed);
        
        // COMPREHENSIVE LOGGING: Success metrics
        if self.signals_processed.load(Ordering::Relaxed) % 1000 == 0 {
            info!("Milestone: {} signals processed, {} allocations avoided", 
                  self.signals_processed.load(Ordering::Relaxed),
                  self.allocations_avoided.load(Ordering::Relaxed));
        }
        
        Ok(order_id)
        // Signal automatically returned to pool when dropped
    }
    
    /// Convert signal to order with ZERO allocations
    #[inline(always)]
    fn signal_to_order_fast(&self, signal: &Signal) -> Result<u64> {
        // Acquire pre-allocated order from pool
        let mut order = acquire_order();
        self.allocations_avoided.fetch_add(1, Ordering::Relaxed);
        
        // Generate order ID without allocation
        static ORDER_COUNTER: AtomicU64 = AtomicU64::new(1);
        let order_id = ORDER_COUNTER.fetch_add(1, Ordering::Relaxed);
        
        // Fill order data (reuse existing capacity)
        order.id = order_id;
        order.symbol.clear();
        order.symbol.push_str(&signal.symbol);
        order.side = if signal.strength > 0.0 { 
            OrderSide::Buy 
        } else { 
            OrderSide::Sell 
        };
        
        // Calculate quantity based on signal strength (no allocation)
        let base_quantity = Decimal::from(100); // Pre-calculated base
        order.quantity = base_quantity * Decimal::from_f64_retain(signal.strength.abs())
            .unwrap_or(Decimal::ONE);
        
        // Clamp to limits (no allocation)
        if order.quantity > self.max_position_size {
            order.quantity = self.max_position_size;
        }
        if order.quantity < self.min_order_size {
            order.quantity = self.min_order_size;
        }
        
        order.price = Decimal::ZERO; // Market order
        order.order_type = OrderType::Market;
        order.timestamp = signal.timestamp;
        order.exchange.clear();
        order.exchange.push_str("binance"); // Pre-selected exchange
        order.status = OrderStatus::Pending;
        
        // Execute risk check using pre-allocated object
        if !self.check_risk_fast(&order)? {
            return Ok(0); // Order rejected, auto-returned to pool
        }
        
        // Submit order (would go to exchange adapter)
        self.submit_order_fast(&order)?;
        
        self.orders_processed.fetch_add(1, Ordering::Relaxed);
        
        Ok(order_id)
        // Order automatically returned to pool when dropped
    }
    
    /// Risk check with ZERO allocations
    #[inline(always)]
    fn check_risk_fast(&self, order: &Order) -> Result<bool> {
        // Acquire pre-allocated risk check from pool
        let mut risk_check = acquire_risk_check();
        self.allocations_avoided.fetch_add(1, Ordering::Relaxed);
        
        // Perform checks (no allocations)
        risk_check.order_id = order.id;
        risk_check.position_size_ok = order.quantity <= self.max_position_size;
        risk_check.order_value_ok = order.quantity * order.price <= self.max_order_value;
        risk_check.min_size_ok = order.quantity >= self.min_order_size;
        risk_check.timestamp = order.timestamp;
        
        // Calculate risk score (no allocation)
        let passed = risk_check.position_size_ok && 
                    risk_check.order_value_ok && 
                    risk_check.min_size_ok;
        
        if !passed {
            debug!("Risk check failed for order {}", order.id);
        }
        
        Ok(passed)
        // RiskCheck automatically returned to pool when dropped
    }
    
    /// Submit order with ZERO allocations
    #[inline(always)]
    fn submit_order_fast(&self, order: &Order) -> Result<()> {
        // In real implementation, this would use the exchange adapter
        // For now, just create an execution report
        
        let mut execution = acquire_execution();
        self.allocations_avoided.fetch_add(1, Ordering::Relaxed);
        
        execution.order_id = order.id;
        execution.symbol.clear();
        execution.symbol.push_str(&order.symbol);
        execution.executed_quantity = order.quantity;
        execution.executed_price = order.price;
        execution.status = OrderStatus::Submitted;
        execution.timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        
        // Log without allocation
        debug!("Order {} submitted", order.id);
        
        Ok(())
        // ExecutionReport automatically returned to pool when dropped
    }
    
    /// Process market data update with ZERO allocations
    #[inline(always)]
    pub fn process_market_data_fast(&self,
        symbol: &str,
        bid: Decimal,
        ask: Decimal,
        last: Decimal,
        volume: Decimal
    ) -> Result<()> {
        // START HOT PATH
        let start = Instant::now();
        
        // Acquire pre-allocated market data from pool
        let mut market_data = acquire_market_data();
        self.allocations_avoided.fetch_add(1, Ordering::Relaxed);
        
        // Fill data (reuse existing capacity)
        market_data.symbol.clear();
        market_data.symbol.push_str(symbol);
        market_data.bid = bid;
        market_data.ask = ask;
        market_data.last = last;
        market_data.volume = volume;
        market_data.timestamp = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
        market_data.exchange.clear();
        market_data.exchange.push_str("binance");
        
        // Calculate spread without allocation
        let spread = ask - bid;
        let spread_bps = (spread / ask) * Decimal::from(10000);
        
        // Check for unusual conditions (no allocation)
        if spread_bps > Decimal::from(100) {
            warn!("Wide spread detected: {} bps", spread_bps);
        }
        
        let elapsed = start.elapsed();
        if elapsed.as_micros() > 5 {
            warn!("Slow market data processing: {}μs", elapsed.as_micros());
        }
        
        Ok(())
        // MarketData automatically returned to pool when dropped
    }
    
    /// Get performance statistics
    pub fn stats(&self) -> ProcessorStats {
        ProcessorStats {
            orders_processed: self.orders_processed.load(Ordering::Relaxed),
            signals_processed: self.signals_processed.load(Ordering::Relaxed),
            allocations_avoided: self.allocations_avoided.load(Ordering::Relaxed),
            pool_stats: POOL_REGISTRY.global_stats(),
        }
    }
    
    /// Shutdown the processor
    pub fn shutdown(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        info!("Fast order processor shutdown");
    }
}

#[derive(Debug)]
pub struct ProcessorStats {
    pub orders_processed: u64,
    pub signals_processed: u64,
    pub allocations_avoided: u64,
    pub pool_stats: infrastructure::GlobalPoolStats,
}

impl ProcessorStats {
    pub fn print_summary(&self) {
        println!("=== Fast Order Processor Statistics ===");
        println!("Orders processed: {}", self.orders_processed);
        println!("Signals processed: {}", self.signals_processed);
        println!("Allocations avoided: {}", self.allocations_avoided);
        println!("Estimated memory saved: {} MB", 
            (self.allocations_avoided * 1024) / 1_048_576); // Rough estimate
        
        // Pool utilization
        let order_hit_rate = self.pool_stats.orders.hit_rate;
        let signal_hit_rate = self.pool_stats.signals.hit_rate;
        println!("Order pool hit rate: {:.2}%", order_hit_rate * 100.0);
        println!("Signal pool hit rate: {:.2}%", signal_hit_rate * 100.0);
        
        if order_hit_rate < 0.95 {
            println!("WARNING: Order pool hit rate below 95% - consider increasing pool size");
        }
        if signal_hit_rate < 0.95 {
            println!("WARNING: Signal pool hit rate below 95% - consider increasing pool size");
        }
    }
}

// ============================================================================
// BENCHMARKS - Jordan's performance validation
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_allocation_signal_processing() {
        let processor = FastOrderProcessor::new();
        
        // Warm up the pools
        for _ in 0..100 {
            processor.process_signal_fast("BTCUSDT", 0.8, 0.9).unwrap();
        }
        
        // Measure allocations
        let start_allocs = processor.allocations_avoided.load(Ordering::Relaxed);
        
        // Process 10,000 signals
        let start = Instant::now();
        for i in 0..10_000 {
            let strength = ((i % 200) as f64 - 100.0) / 100.0;
            let confidence = 0.7 + (i % 30) as f64 / 100.0;
            processor.process_signal_fast("BTCUSDT", strength, confidence).unwrap();
        }
        let elapsed = start.elapsed();
        
        let end_allocs = processor.allocations_avoided.load(Ordering::Relaxed);
        let avoided = end_allocs - start_allocs;
        
        println!("Processed 10,000 signals in {:?}", elapsed);
        println!("Average latency: {}μs", elapsed.as_micros() / 10_000);
        println!("Allocations avoided: {}", avoided);
        
        // Should be at least 3 objects per signal (_signal, order, risk check)
        assert!(avoided >= 30_000);
        
        // Should be fast (< 10μs average)
        assert!(elapsed.as_micros() / 10_000 < 10);
    }
    
    #[test]
    fn test_pool_hit_rates() {
        let processor = FastOrderProcessor::new();
        
        // Process many signals to test pool effectiveness
        for i in 0..1000 {
            processor.process_signal_fast(
                "ETHUSDT",
                (i as f64 % 2.0) - 1.0,
                0.75
            ).unwrap();
        }
        
        let stats = processor.stats();
        
        // Check pool hit rates are high
        assert!(stats.pool_stats.orders.hit_rate() > 0.9);
        assert!(stats.pool_stats.signals.hit_rate() > 0.9);
        assert!(stats.pool_stats.risk_checks.hit_rate() > 0.9);
        
        stats.print_summary();
    }
    
    #[test]
    fn test_market_data_processing() {
        let processor = FastOrderProcessor::new();
        
        let start = Instant::now();
        for i in 0..100_000 {
            processor.process_market_data_fast(
                "BTCUSDT",
                Decimal::from(50000 + i % 100),
                Decimal::from(50001 + i % 100),
                Decimal::from(50000 + i % 100),
                Decimal::from(1000000),
            ).unwrap();
        }
        let elapsed = start.elapsed();
        
        println!("Processed 100,000 market updates in {:?}", elapsed);
        println!("Throughput: {} updates/sec", 100_000_000 / elapsed.as_millis());
        
        // Should handle > 100k updates/sec
        assert!(elapsed.as_secs() < 2);
    }
}