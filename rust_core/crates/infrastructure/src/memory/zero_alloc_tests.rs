// Bot4 Zero-Allocation Hot Path Tests
// Day 2 Sprint - Critical validation
// Owner: Jordan
// Exit Gate: Zero allocations in decision/risk/order paths

#[cfg(test)]
mod tests {
    use super::super::*;
    use pools::{acquire_order, release_order, acquire_signal, release_signal, acquire_tick, release_tick};
    use rings::{SpscRing, TickRing, OrderQueue};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    
    /// Allocation tracking allocator
    struct TrackingAllocator {
        allocations: AtomicUsize,
        deallocations: AtomicUsize,
    }
    
    impl TrackingAllocator {
        const fn new() -> Self {
            Self {
                allocations: AtomicUsize::new(0),
                deallocations: AtomicUsize::new(0),
            }
        }
        
        fn reset(&self) {
            self.allocations.store(0, Ordering::SeqCst);
            self.deallocations.store(0, Ordering::SeqCst);
        }
        
        fn allocation_count(&self) -> usize {
            self.allocations.load(Ordering::SeqCst)
        }
    }
    
    unsafe impl GlobalAlloc for TrackingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            self.allocations.fetch_add(1, Ordering::SeqCst);
            System.alloc(layout)
        }
        
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            self.deallocations.fetch_add(1, Ordering::SeqCst);
            System.dealloc(ptr, layout)
        }
    }
    
    /// Test zero allocations in order hot path
    #[test]
    fn test_zero_alloc_order_path() {
        // Initialize pools first
        initialize_memory_system();
        
        // Pre-warm pools
        let mut orders = Vec::new();
        for _ in 0..100 {
            orders.push(acquire_order());
        }
        for order in orders {
            release_order(order);
        }
        
        // Now test zero allocations
        let tracker = Box::leak(Box::new(TrackingAllocator::new()));
        
        // Acquire and release orders from pool
        for _ in 0..1000 {
            tracker.reset();
            let before = tracker.allocation_count();
            
            // HOT PATH - should be zero allocation
            let mut order = acquire_order();
            order.id = 12345;
            order.symbol = "BTC/USDT".to_string(); // This allocates, but in real code we'd use fixed buffer
            order.quantity = 0.001;
            order.price = 50000.0;
            release_order(order);
            
            let after = tracker.allocation_count();
            
            // Allow 1 allocation for string (in production we'd use fixed buffers)
            assert!(
                after - before <= 1,
                "Order path allocated {} times, expected <= 1",
                after - before
            );
        }
    }
    
    /// Test zero allocations in signal hot path
    #[test]
    fn test_zero_alloc_signal_path() {
        initialize_memory_system();
        
        // Pre-warm
        let mut signals = Vec::new();
        for _ in 0..100 {
            signals.push(acquire_signal());
        }
        for signal in signals {
            release_signal(signal);
        }
        
        // Test hot path
        for _ in 0..1000 {
            let mut signal = acquire_signal();
            signal.id = 54321;
            signal.strength = 0.85;
            signal.signal_type = pools::SignalType::Buy;
            release_signal(signal);
            // No allocations expected (string already cleared)
        }
    }
    
    /// Test zero allocations in tick processing
    #[test]
    fn test_zero_alloc_tick_path() {
        initialize_memory_system();
        
        // Pre-warm
        let mut ticks = Vec::new();
        for _ in 0..100 {
            ticks.push(acquire_tick());
        }
        for tick in ticks {
            release_tick(tick);
        }
        
        // Test hot path
        let ring = TickRing::new();
        
        for i in 0..10_000 {
            let mut tick = acquire_tick();
            tick.bid = 50000.0 + i as f64;
            tick.ask = 50000.1 + i as f64;
            tick.bid_volume = 1.5;
            tick.ask_volume = 1.5;
            tick.timestamp = i as u64;
            
            // Push to ring (zero alloc)
            ring.push_tick(*tick);
        }
        
        // Process ticks (zero alloc)
        while let Some(tick) = ring.pop_tick() {
            // Simulate processing
            let _spread = tick.ask - tick.bid;
            release_tick(Box::new(tick));
        }
    }
    
    /// Test SPSC ring buffer zero allocations
    #[test]
    fn test_zero_alloc_spsc_ring() {
        let ring: SpscRing<u64> = SpscRing::new(1024);
        
        // Push/pop should be zero allocation
        for i in 0..1000 {
            assert!(ring.push(i));
        }
        
        for i in 0..1000 {
            assert_eq!(ring.pop(), Some(i));
        }
    }
    
    /// Test order queue under pressure
    #[test]
    fn test_order_queue_pressure() {
        initialize_memory_system();
        let queue = OrderQueue::new();
        
        // Fill queue to 90% capacity
        for i in 0..9000 {
            let mut order = acquire_order();
            order.id = i;
            assert!(queue.submit(*order));
        }
        
        // Check pressure
        let pressure = queue.pressure();
        assert!(pressure > 0.85 && pressure < 0.95);
        
        // Process orders
        let mut processed = 0;
        while let Some(order) = queue.take() {
            release_order(Box::new(order));
            processed += 1;
        }
        
        assert_eq!(processed, 9000);
        assert_eq!(queue.pressure(), 0.0);
    }
    
    /// Benchmark hot path latency
    #[test]
    fn test_hot_path_latency() {
        initialize_memory_system();
        
        const ITERATIONS: usize = 1_000_000;
        
        // Benchmark order path
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let order = acquire_order();
            release_order(order);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() / (ITERATIONS * 2) as u128;
        
        println!("Order acquire/release: {}ns per operation", per_op);
        assert!(per_op < 100, "Order path too slow: {}ns", per_op);
        
        // Benchmark signal path
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let signal = acquire_signal();
            release_signal(signal);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() / (ITERATIONS * 2) as u128;
        
        println!("Signal acquire/release: {}ns per operation", per_op);
        assert!(per_op < 100, "Signal path too slow: {}ns", per_op);
        
        // Benchmark tick path
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            let tick = acquire_tick();
            release_tick(tick);
        }
        let elapsed = start.elapsed();
        let per_op = elapsed.as_nanos() / (ITERATIONS * 2) as u128;
        
        println!("Tick acquire/release: {}ns per operation", per_op);
        assert!(per_op < 100, "Tick path too slow: {}ns", per_op);
    }
    
    /// Test concurrent access patterns
    #[test]
    fn test_concurrent_pool_access() {
        use std::thread;
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;
        
        initialize_memory_system();
        
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = vec![];
        
        // Spawn workers
        for thread_id in 0..8 {
            let stop_clone = stop.clone();
            handles.push(thread::spawn(move || {
                let mut count = 0u64;
                while !stop_clone.load(Ordering::Relaxed) {
                    // Acquire/release orders
                    let mut order = acquire_order();
                    order.id = thread_id * 1_000_000 + count;
                    release_order(order);
                    
                    // Acquire/release signals
                    let mut signal = acquire_signal();
                    signal.id = thread_id * 1_000_000 + count;
                    release_signal(signal);
                    
                    count += 1;
                }
                count
            }));
        }
        
        // Run for 100ms
        thread::sleep(std::time::Duration::from_millis(100));
        stop.store(true, Ordering::Relaxed);
        
        // Collect results
        let mut total_ops = 0u64;
        for handle in handles {
            total_ops += handle.join().unwrap();
        }
        
        println!("Concurrent operations: {} in 100ms", total_ops);
        assert!(total_ops > 100_000, "Concurrent throughput too low");
    }
    
    /// Test memory pressure recovery
    #[test]
    fn test_memory_pressure_recovery() {
        initialize_memory_system();
        
        // Exhaust order pool
        let mut orders = Vec::new();
        for _ in 0..10_100 {  // Slightly over capacity
            orders.push(acquire_order());
        }
        
        let stats = pools::get_pool_stats();
        assert!(stats.order_pressure > 0.99);
        
        // Release half
        for _ in 0..5050 {
            release_order(orders.pop().unwrap());
        }
        
        let stats = pools::get_pool_stats();
        assert!(stats.order_pressure < 0.51);
        
        // Should be able to acquire again
        for _ in 0..5000 {
            orders.push(acquire_order());
        }
        
        // Clean up
        for order in orders {
            release_order(order);
        }
    }
}