// Layer 1.3: Event-Driven Processing with Adaptive Sampling
// DEEP DIVE Implementation - NO SHORTCUTS, NO FAKES, NO PLACEHOLDERS
// Full Team: Alex (Architecture) + Morgan (Math) + Sam (Code) + Quinn (Risk) + 
//           Jordan (Performance) + Casey (Exchange) + Riley (Testing) + Avery (Data)
//
// External Research Applied:
// - "Event Processing in Action" - Etzion & Niblett (2010)
// - "Streaming Systems" - Akidau, Chernyak, Lax (2018)  
// - "High Performance Browser Networking" - Grigorik (2024)
// - Chronicle Software's microsecond architectures
// - LMAX Disruptor pattern
// - Aeron messaging for finance
// - TimeMixer volatility forecasting (2024)
// - DeepVol adaptive sampling (2024)

pub mod processor;
pub mod adaptive_sampler;
pub mod bucketed_aggregator;
pub mod volatility_tracker;
pub mod event_dispatcher;
pub mod time_window;
pub mod performance_monitor;

// Re-export main types
pub use processor::{
    EventProcessor,
    ProcessorConfig,
    EventPriority,
    ProcessingResult,
    ProcessorMetrics,
};

pub use adaptive_sampler::{
    AdaptiveSampler,
    SamplerConfig,
    SamplingStrategy,
    VolatilityRegime,
    SamplingRate,
};

pub use bucketed_aggregator::{
    BucketedAggregator,
    BucketConfig,
    AggregateWindow,
    BucketStats,
    WindowType,
};

pub use volatility_tracker::{
    VolatilityTracker,
    VolatilityConfig,
    VolatilityMetrics,
    RegimeChange,
    VolatilityModel,
};

pub use event_dispatcher::{
    EventDispatcher,
    DispatchStrategy,
    EventRoute,
    DispatchMetrics,
    Priority,
};

pub use time_window::{
    TimeWindow,
    WindowManager,
    WindowConfig,
    TumblingWindow,
    SlidingWindow,
    SessionWindow,
};

pub use performance_monitor::{
    PerformanceMonitor,
    LatencyTracker,
    ThroughputMetrics,
    SystemLoad,
};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_structure() {
        // Verify all modules are properly organized
        let _ = ProcessorConfig::default();
        let _ = SamplerConfig::default();
        let _ = BucketConfig::default();
    }
}