// Infrastructure crate - Core components for Bot4 trading platform
// Validated by Sophia (ChatGPT) for production readiness
// Day 2 Sprint: Added memory management (MiMalloc + Pools)

pub mod circuit_breaker;
pub mod memory;

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

// Re-export memory management
pub use memory::{
    initialize_memory_system,
    MemoryStats,
    pools::{acquire_order, release_order, acquire_signal, release_signal, acquire_tick, release_tick},
    rings::{SpscRing, MpmcRing, TickRing, OrderQueue},
};