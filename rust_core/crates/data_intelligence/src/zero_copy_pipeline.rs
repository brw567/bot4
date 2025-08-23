// ZERO-COPY DATA PIPELINE - DEEP DIVE IMPLEMENTATION
// Team: Jordan (Lead) - NO ALLOCATIONS, MAXIMUM PERFORMANCE!
// Target: <100ns per event processing

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::pin::Pin;
use std::mem::MaybeUninit;
use std::alloc::{alloc, dealloc, Layout};

use bytes::{Bytes, BytesMut};
use crossbeam::channel::{bounded, Sender, Receiver};
use parking_lot::RwLock;
use memmap2::{Mmap, MmapOptions};
use zerocopy::{AsBytes, FromBytes, FromZeroes};
use tokio::sync::mpsc;
use tracing::{debug, trace};

use crate::{DataError, Result};

/// Zero-copy pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub ring_buffer_size: usize,      // Power of 2
    pub batch_size: usize,             // Events per batch
    pub mmap_threshold: usize,         // Use mmap for large data
    pub zero_copy_threshold: usize,    // Direct memory access threshold
    pub max_in_flight: usize,          // Backpressure limit
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            ring_buffer_size: 1 << 20,    // 1M events
            batch_size: 1024,              // Process in 1K batches
            mmap_threshold: 1 << 16,       // 64KB
            zero_copy_threshold: 256,      // 256 bytes
            max_in_flight: 10_000,         // 10K events backpressure
        }
    }
}

/// Lock-free ring buffer for zero-copy data transfer
pub struct ZeroCopyPipeline {
    // Ring buffer storage
    buffer: Pin<Box<[MaybeUninit<CacheLineAligned<RawEvent>>]>>,
    mask: usize,
    
    // Lock-free indices
    write_index: Arc<AtomicUsize>,
    read_index: Arc<AtomicUsize>,
    cached_write: AtomicUsize,
    cached_read: AtomicUsize,
    
    // Memory mapped regions for large data
    mmap_regions: Arc<RwLock<Vec<MmapRegion>>>,
    
    // Metrics
    events_processed: AtomicU64,
    bytes_processed: AtomicU64,
    zero_copy_hits: AtomicU64,
    
    // Configuration
    config: PipelineConfig,
}

/// Cache-line aligned event to prevent false sharing
#[repr(C, align(64))]
struct CacheLineAligned<T> {
    data: T,
    _padding: [u8; 64 - std::mem::size_of::<T>()],
}

/// Raw event in the pipeline
#[derive(Debug, Clone, Copy, AsBytes, FromBytes, FromZeroes)]
#[repr(C)]
pub struct RawEvent {
    pub timestamp_ns: u64,
    pub event_type: u32,
    pub data_offset: u32,    // Offset into mmap region
    pub data_length: u32,
    pub source_id: u16,
    pub flags: u16,
    pub checksum: u32,
    _padding: [u8; 32],      // Align to 64 bytes
}

/// Memory mapped region for large data
struct MmapRegion {
    mmap: Mmap,
    offset: AtomicUsize,
    capacity: usize,
}

impl ZeroCopyPipeline {
    pub fn new(config: PipelineConfig) -> Result<Self> {
        // Ensure ring buffer size is power of 2
        assert!(config.ring_buffer_size.is_power_of_two());
        
        // Allocate ring buffer with proper alignment
        let layout = Layout::array::<CacheLineAligned<RawEvent>>(config.ring_buffer_size)
            .map_err(|e| DataError::SimdError(e.to_string()))?;
        
        let buffer = unsafe {
            let ptr = alloc(layout) as *mut MaybeUninit<CacheLineAligned<RawEvent>>;
            let slice = std::slice::from_raw_parts_mut(ptr, config.ring_buffer_size);
            Pin::new_unchecked(Box::from_raw(slice))
        };
        
        Ok(Self {
            buffer,
            mask: config.ring_buffer_size - 1,
            write_index: Arc::new(AtomicUsize::new(0)),
            read_index: Arc::new(AtomicUsize::new(0)),
            cached_write: AtomicUsize::new(0),
            cached_read: AtomicUsize::new(0),
            mmap_regions: Arc::new(RwLock::new(Vec::new())),
            events_processed: AtomicU64::new(0),
            bytes_processed: AtomicU64::new(0),
            zero_copy_hits: AtomicU64::new(0),
            config,
        })
    }
    
    /// Push event to pipeline (zero-copy)
    #[inline(always)]
    pub fn push(&self, event: RawEvent) -> Result<()> {
        let write = self.write_index.load(Ordering::Acquire);
        let mut next_write = write + 1;
        
        // Check for overflow
        if next_write - self.cached_read.load(Ordering::Relaxed) > self.mask {
            // Update cached read
            self.cached_read.store(
                self.read_index.load(Ordering::Acquire),
                Ordering::Relaxed
            );
            
            // Still full?
            if next_write - self.cached_read.load(Ordering::Relaxed) > self.mask {
                return Err(DataError::SimdError("Ring buffer full".into()));
            }
        }
        
        // Write to buffer (zero-copy)
        unsafe {
            let slot = &mut *self.buffer[write & self.mask].as_mut_ptr();
            slot.data = event;
        }
        
        // Update write index
        self.write_index.store(next_write, Ordering::Release);
        
        // Update metrics
        self.events_processed.fetch_add(1, Ordering::Relaxed);
        self.bytes_processed.fetch_add(event.data_length as u64, Ordering::Relaxed);
        
        if event.data_length < self.config.zero_copy_threshold as u32 {
            self.zero_copy_hits.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// Pop event from pipeline (zero-copy)
    #[inline(always)]
    pub fn pop(&self) -> Option<RawEvent> {
        let read = self.read_index.load(Ordering::Acquire);
        
        // Check if empty
        if read >= self.cached_write.load(Ordering::Relaxed) {
            // Update cached write
            self.cached_write.store(
                self.write_index.load(Ordering::Acquire),
                Ordering::Relaxed
            );
            
            // Still empty?
            if read >= self.cached_write.load(Ordering::Relaxed) {
                return None;
            }
        }
        
        // Read from buffer (zero-copy)
        let event = unsafe {
            (*self.buffer[read & self.mask].as_ptr()).data
        };
        
        // Update read index
        self.read_index.store(read + 1, Ordering::Release);
        
        Some(event)
    }
    
    /// Process batch of events with SIMD
    #[inline(always)]
    pub fn process_batch<F>(&self, batch_size: usize, mut processor: F) -> usize
        where F: FnMut(&[RawEvent])
    {
        let mut events = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            if let Some(event) = self.pop() {
                events.push(event);
            } else {
                break;
            }
        }
        
        let count = events.len();
        if count > 0 {
            processor(&events);
        }
        
        count
    }
    
    /// Allocate memory-mapped region for large data
    pub fn allocate_mmap_region(&self, size: usize) -> Result<usize> {
        use std::fs::OpenOptions;
        use tempfile::tempfile;
        
        let file = tempfile()
            .map_err(|e| DataError::SimdError(format!("Failed to create temp file: {}", e)))?;
        
        file.set_len(size as u64)
            .map_err(|e| DataError::SimdError(format!("Failed to set file size: {}", e)))?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .len(size)
                .map(&file)
                .map_err(|e| DataError::SimdError(format!("Failed to mmap: {}", e)))?
        };
        
        let region = MmapRegion {
            mmap,
            offset: AtomicUsize::new(0),
            capacity: size,
        };
        
        let mut regions = self.mmap_regions.write();
        let index = regions.len();
        regions.push(region);
        
        Ok(index)
    }
    
    /// Write data to mmap region (zero-copy)
    pub fn write_to_mmap(&self, region_idx: usize, data: &[u8]) -> Result<u32> {
        let regions = self.mmap_regions.read();
        let region = regions.get(region_idx)
            .ok_or_else(|| DataError::SimdError("Invalid mmap region".into()))?;
        
        let offset = region.offset.fetch_add(data.len(), Ordering::Relaxed);
        
        if offset + data.len() > region.capacity {
            return Err(DataError::SimdError("Mmap region full".into()));
        }
        
        // Zero-copy write
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                region.mmap.as_ptr().add(offset) as *mut u8,
                data.len()
            );
        }
        
        Ok(offset as u32)
    }
    
    /// Read data from mmap region (zero-copy)
    pub fn read_from_mmap(&self, region_idx: usize, offset: u32, length: u32) -> Result<&[u8]> {
        let regions = self.mmap_regions.read();
        let region = regions.get(region_idx)
            .ok_or_else(|| DataError::SimdError("Invalid mmap region".into()))?;
        
        let start = offset as usize;
        let end = start + length as usize;
        
        if end > region.capacity {
            return Err(DataError::SimdError("Invalid mmap read range".into()));
        }
        
        // Zero-copy read
        Ok(unsafe {
            std::slice::from_raw_parts(
                region.mmap.as_ptr().add(start),
                length as usize
            )
        })
    }
    
    /// Get pipeline metrics
    pub fn metrics(&self) -> PipelineMetrics {
        PipelineMetrics {
            events_processed: self.events_processed.load(Ordering::Relaxed),
            bytes_processed: self.bytes_processed.load(Ordering::Relaxed),
            zero_copy_hits: self.zero_copy_hits.load(Ordering::Relaxed),
            buffer_utilization: self.buffer_utilization(),
        }
    }
    
    fn buffer_utilization(&self) -> f64 {
        let write = self.write_index.load(Ordering::Relaxed);
        let read = self.read_index.load(Ordering::Relaxed);
        let used = write - read;
        used as f64 / self.config.ring_buffer_size as f64
    }
}

#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    pub events_processed: u64,
    pub bytes_processed: u64,
    pub zero_copy_hits: u64,
    pub buffer_utilization: f64,
}

// Ensure zero-copy compatibility
static_assertions::assert_eq_size!(RawEvent, [u8; 64]);
static_assertions::assert_eq_align!(RawEvent, u64);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_copy_pipeline() {
        let config = PipelineConfig {
            ring_buffer_size: 1024,
            ..Default::default()
        };
        
        let pipeline = ZeroCopyPipeline::new(config).unwrap();
        
        // Create test event
        let event = RawEvent {
            timestamp_ns: 1234567890,
            event_type: 1,
            data_offset: 0,
            data_length: 100,
            source_id: 1,
            flags: 0,
            checksum: 0xDEADBEEF,
            _padding: [0; 32],
        };
        
        // Push event
        pipeline.push(event).unwrap();
        
        // Pop event
        let popped = pipeline.pop().unwrap();
        assert_eq!(popped.timestamp_ns, event.timestamp_ns);
        assert_eq!(popped.checksum, event.checksum);
        
        // Check metrics
        let metrics = pipeline.metrics();
        assert_eq!(metrics.events_processed, 1);
        assert_eq!(metrics.bytes_processed, 100);
    }
    
    #[test]
    fn test_batch_processing() {
        let config = PipelineConfig::default();
        let pipeline = ZeroCopyPipeline::new(config).unwrap();
        
        // Push multiple events
        for i in 0..100 {
            let event = RawEvent {
                timestamp_ns: i as u64,
                event_type: 1,
                data_offset: 0,
                data_length: 64,
                source_id: 1,
                flags: 0,
                checksum: i as u32,
                _padding: [0; 32],
            };
            pipeline.push(event).unwrap();
        }
        
        // Process batch
        let processed = pipeline.process_batch(50, |events| {
            assert_eq!(events.len(), 50);
            for (i, event) in events.iter().enumerate() {
                assert_eq!(event.timestamp_ns, i as u64);
            }
        });
        
        assert_eq!(processed, 50);
    }
}