use domain_types::PipelineMetrics;
//! # ZERO-COPY DATA PIPELINE - Maximum Performance
//! Ellis (Performance Lead): "Every nanosecond counts"

use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam::channel::{bounded, unbounded, Receiver, Sender};
use mmap_rs::{MmapOptions, Mmap};
use ringbuf::{HeapRb, Producer, Consumer};

/// Zero-copy market data pipeline
/// TODO: Add docs
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
// pub struct ZeroCopyPipeline {
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     /// Ring buffer for real-time data
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     ring_buffer: Arc<HeapRb<MarketDataPacket>>,
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     /// Memory-mapped file for historical data
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     mmap: Option<Mmap>,
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     /// Lock-free queue for orders
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     order_queue: (Sender<OrderPacket>, Receiver<OrderPacket>),
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     /// Metrics
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
//     metrics: Arc<RwLock<PipelineMetrics>>,
// ELIMINATED: Duplicate - use infrastructure::zero_copy::ZeroCopyPipeline
// }

#[repr(C, align(64))] // Cache-line aligned
/// TODO: Add docs
pub struct MarketDataPacket {
    pub timestamp: u64,
    pub symbol_id: u32,
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
    pub last_price: f64,
    pub volume: f64,
    _padding: [u8; 16], // Ensure 64-byte alignment
}

#[repr(C, align(64))]
/// TODO: Add docs
pub struct OrderPacket {
    pub order_id: u64,
    pub symbol_id: u32,
    pub side: u8,
    pub order_type: u8,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
    _padding: [u8; 26],
}

impl ZeroCopyPipeline {
    pub fn new(capacity: usize) -> Self {
        let ring_buffer = HeapRb::new(capacity);
        let (tx, rx) = unbounded();
        
        Self {
            ring_buffer: Arc::new(ring_buffer),
            mmap: None,
            order_queue: (tx, rx),
            metrics: Arc::new(RwLock::new(PipelineMetrics::default())),
        }
    }
    
    /// Process market data with zero allocations
    #[inline(always)]
    pub fn process_market_data(&mut self, packet: MarketDataPacket) {
        // Direct write to ring buffer - no allocation
        if let Some(mut producer) = self.ring_buffer.try_write() {
            producer.push(packet);
            
            let mut metrics = self.metrics.write();
            metrics.packets_processed += 1;
        }
    }
    
    /// Submit order with zero-copy
    #[inline(always)]
    pub fn submit_order(&self, order: OrderPacket) {
        // Lock-free send
        let _ = self.order_queue.0.try_send(order);
    }
}

#[derive(Default)]
// ELIMINATED: use domain_types::PipelineMetrics
// pub struct PipelineMetrics {
    pub packets_processed: u64,
    pub bytes_processed: u64,
    pub orders_submitted: u64,
    pub latency_ns: u64,
}

// Ellis: "Zero allocations in the hot path!"
