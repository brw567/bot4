// Infrastructure crate - Core components for Bot4 trading platform
// Validated by Sophia (ChatGPT) for production readiness
// Day 2 Sprint: Added memory management (MiMalloc + Pools)
// Phase 1: Added parallelization and runtime optimization
// Phase 3: Added stream processing for ML integration
// OPTIMIZATION SPRINT Day 2: Zero-copy architecture (10x speedup)

pub mod circuit_breaker;
pub mod memory;
pub mod parallelization;
pub mod runtime_optimization;
pub mod hot_path_verification;
pub mod stream_processing;
pub mod zero_copy;  // Zero-copy architecture - FULL TEAM implementation
pub mod object_pools;  // Comprehensive object pools - Nexus Priority 1
pub mod rayon_enhanced;  // Enhanced Rayon parallelization - Nexus Priority 1

// Re-export main types
pub use circuit_breaker::{
    GlobalCircuitBreaker,
    ComponentBreaker as CircuitBreaker,
    CircuitConfig,
    CircuitState,
    CircuitError,
    Permit,
    CallGuard,
    Outcome,
    Clock,
    SystemClock,
};

// Re-export memory management (legacy)
pub use memory::{
    initialize_memory_system,
    MemoryStats,
    pools::{acquire_order as legacy_acquire_order, release_order, 
            acquire_signal as legacy_acquire_signal, release_signal, 
            acquire_tick, release_tick, SignalType as LegacySignalType},
    rings::{SpscRing, MpmcRing, TickRing, OrderQueue},
};

// Re-export parallelization components
pub use parallelization::{
    ParallelizationConfig,
    InstrumentSharding,
    LockFreeStats,
    CpuAffinityManager,
    ParallelProcessor,
    memory_ordering,
};

// Re-export runtime optimization
pub use runtime_optimization::{
    OptimizedRuntime,
    RuntimeStats,
    ZeroAllocTask,
    HotPathVerifier,
    async_primitives,
};

// Re-export stream processing components
pub use stream_processing::{
    StreamProcessor,
    StreamConfig,
    StreamMessage,
    SignalAction,
    RiskEventType,
    RiskSeverity,
    consumer::{StreamConsumer, ConsumerConfig, MessageHandler},
    producer::{BatchProducer, PriorityProducer},
    processor::{ProcessingPipeline, ProcessorStage, PipelineBuilder},
    router::{MessageRouter, RoutingRule, LoadBalancedRouter, FanoutRouter},
};

// Re-export zero-copy components - Day 2 optimization
pub use zero_copy::{
    ObjectPool,
    PoolGuard,
    Arena,
    LockFreeMetrics,
    ZeroCopyPipeline,
    RingBuffer,
    ZeroCopyMatrix,
    MemoryPoolManager,
};

// Re-export comprehensive object pools - Nexus Priority 1
pub use object_pools::{
    POOL_REGISTRY,
    acquire_order,
    acquire_signal,
    acquire_market_data,
    acquire_position,
    acquire_risk_check,
    acquire_execution,
    acquire_feature,
    acquire_ml_inference,
    Order,
    Signal,
    MarketData,
    Position,
    RiskCheck,
    ExecutionReport,
    Feature,
    MLInference,
};

// Re-export enhanced Rayon parallelization - Nexus Priority 1
pub use rayon_enhanced::{
    ParallelTradingEngine,
    EngineMetrics,
    ParallelPipeline,
    PipelineStage,
};