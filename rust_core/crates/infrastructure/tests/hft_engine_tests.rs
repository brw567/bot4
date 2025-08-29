//! HFT Engine Comprehensive Tests
//! Team: InfraEngineer + ExchangeSpec
//! Coverage Target: 100%
//! Research: DPDK, kernel bypass, lock-free structures

use infrastructure::hft_optimizations::*;
use infrastructure::extreme_performance::*;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg(test)]
mod hft_engine_tests {
    use super::*;
    
    #[test]
    fn test_hardware_timestamp_precision() {
        let ts1 = HFTEngine::hardware_timestamp();
        std::thread::sleep(Duration::from_micros(1));
        let ts2 = HFTEngine::hardware_timestamp();
        
        // Should detect microsecond differences
        assert!(ts2 > ts1);
        
        // Test nanosecond precision
        let mut timestamps = Vec::new();
        for _ in 0..1000 {
            timestamps.push(HFTEngine::hardware_timestamp());
        }
        
        // Should have unique timestamps
        timestamps.dedup();
        assert!(timestamps.len() > 900);
    }
    
    #[test]
    fn test_zero_copy_tick_processing() {
        let engine = HFTEngine::new_colocated();
        
        let tick = MarketTick {
            symbol_id: 1,
            exchange_id: 1,
            _padding1: [0; 3],
            bid_price: 100_000_000, // $100.00 in fixed point
            bid_size: 1000,
            ask_price: 100_010_000, // $100.01
            ask_size: 1000,
            timestamp_ns: 1_000_000_000,
            sequence: 1,
            _padding2: [0; 8],
        };
        
        let start = Instant::now();
        for _ in 0..1_000_000 {
            let decision = engine.process_tick_zero_copy(&tick);
            match decision {
                Decision::Trade | Decision::Wait | Decision::Halt => {}
            }
        }
        let elapsed = start.elapsed();
        
        // Should process 1M ticks in <100ms (>10M/sec)
        assert!(elapsed < Duration::from_millis(100));
    }
    
    #[test]
    fn test_cache_line_alignment() {
        // Verify structures are cache-line aligned (64 bytes)
        assert_eq!(std::mem::size_of::<MarketTick>(), 64);
        assert_eq!(std::mem::size_of::<Order>(), 64);
        
        // Verify alignment
        let tick = MarketTick {
            symbol_id: 1,
            exchange_id: 1,
            _padding1: [0; 3],
            bid_price: 100_000_000,
            bid_size: 1000,
            ask_price: 100_010_000,
            ask_size: 1000,
            timestamp_ns: 0,
            sequence: 0,
            _padding2: [0; 8],
        };
        
        let addr = &tick as *const _ as usize;
        assert_eq!(addr % 64, 0, "MarketTick not cache-line aligned");
    }
    
    #[test]
    fn test_emergency_stop() {
        let engine = HFTEngine::new_colocated();
        
        // Normal operation
        let tick = create_test_tick();
        let decision = engine.process_tick_zero_copy(&tick);
        assert!(matches!(decision, Decision::Trade | Decision::Wait));
        
        // Trigger emergency stop
        engine.emergency_stop.store(true, Ordering::Release);
        
        // Should halt immediately
        let decision = engine.process_tick_zero_copy(&tick);
        assert_eq!(decision, Decision::Halt);
    }
}

#[cfg(test)]
mod lock_free_tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_lock_free_ring_buffer() {
        let buffer = Arc::new(LockFreeRingBuffer::<u64>::new(1024));
        let buffer_clone = buffer.clone();
        
        // Producer thread
        let producer = thread::spawn(move || {
            for i in 0..10000 {
                buffer_clone.push(i);
            }
        });
        
        // Consumer thread
        let consumer = thread::spawn(move || {
            let mut count = 0;
            while count < 10000 {
                if let Some(_) = buffer.pop() {
                    count += 1;
                }
            }
            count
        });
        
        producer.join().unwrap();
        let consumed = consumer.join().unwrap();
        
        assert_eq!(consumed, 10000);
    }
    
    #[test]
    fn test_concurrent_ring_buffer() {
        let buffer = Arc::new(LockFreeRingBuffer::<u64>::new(1024));
        let mut handles = vec![];
        
        // Multiple producers
        for t in 0..4 {
            let buffer_clone = buffer.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    buffer_clone.push(t * 1000 + i);
                }
            }));
        }
        
        // Multiple consumers
        let consumed = Arc::new(AtomicU64::new(0));
        for _ in 0..4 {
            let buffer_clone = buffer.clone();
            let consumed_clone = consumed.clone();
            handles.push(thread::spawn(move || {
                loop {
                    if let Some(_) = buffer_clone.pop() {
                        consumed_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    if consumed_clone.load(Ordering::Relaxed) >= 4000 {
                        break;
                    }
                }
            }));
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        assert_eq!(consumed.load(Ordering::Relaxed), 4000);
    }
}

#[cfg(test)]
mod cpu_optimization_tests {
    use super::*;
    
    #[test]
    #[ignore] // Requires specific hardware
    fn test_cpu_pinning() {
        CpuPinning::pin_to_core(0);
        
        // Verify we're pinned to core 0
        let cpu = unsafe { libc::sched_getcpu() };
        assert_eq!(cpu, 0);
    }
    
    #[test]
    #[ignore] // Requires huge pages enabled
    fn test_huge_pages_allocation() {
        let size = 2 * 1024 * 1024; // 2MB
        let ptr = HugePages::alloc_huge(size);
        
        assert!(!ptr.is_null());
        
        // Write and read test
        unsafe {
            for i in 0..size {
                *ptr.add(i) = (i % 256) as u8;
            }
            for i in 0..size {
                assert_eq!(*ptr.add(i), (i % 256) as u8);
            }
        }
        
        // Cleanup
        unsafe {
            libc::munmap(ptr as *mut libc::c_void, size);
        }
    }
}

#[cfg(test)]
mod adaptive_tuner_tests {
    use super::*;
    
    #[test]
    fn test_thompson_sampling() {
        let mut tuner = AdaptiveAutoTuner::new();
        
        // Simulate different arm rewards
        tuner.arm_rewards = vec![10.0, 5.0, 15.0, 8.0];
        tuner.arm_counts = vec![20, 20, 20, 20];
        
        // Thompson sampling should favor arm 2 (highest reward)
        let mut selections = vec![0; 4];
        for _ in 0..1000 {
            let params = tuner.select_parameters();
            let idx = ((params.position_size_pct - 0.01) / 0.005) as usize;
            selections[idx.min(3)] += 1;
        }
        
        // Arm 2 should be selected most often
        assert!(selections[2] > selections[0]);
        assert!(selections[2] > selections[1]);
        assert!(selections[2] > selections[3]);
    }
    
    #[test]
    fn test_parameter_adaptation() {
        let mut tuner = AdaptiveAutoTuner::new();
        
        // Simulate learning over time
        for _ in 0..100 {
            let params = tuner.select_parameters();
            
            // Simulate PnL based on parameters
            let pnl = if params.position_size_pct > 0.02 {
                100.0 // Higher position size = higher reward
            } else {
                50.0
            };
            
            tuner.update_reward(params, pnl);
        }
        
        // Should learn to prefer higher position sizes
        let final_params = tuner.select_parameters();
        assert!(final_params.position_size_pct > 0.015);
    }
}

// Helper functions
fn create_test_tick() -> MarketTick {
    MarketTick {
        symbol_id: 1,
        exchange_id: 1,
        _padding1: [0; 3],
        bid_price: 100_000_000,
        bid_size: 1000,
        ask_price: 100_010_000,
        ask_size: 1000,
        timestamp_ns: 1_000_000_000,
        sequence: 1,
        _padding2: [0; 8],
    }
}
