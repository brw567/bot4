// Cache padding for hot atomics to prevent false sharing
// Addresses Sophia's minor nit #1

use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicU64, AtomicU32, AtomicU8};

/// Cache-padded atomic types to prevent false sharing on hot paths
pub type PaddedAtomicU64 = CachePadded<AtomicU64>;
pub type PaddedAtomicU32 = CachePadded<AtomicU32>;
pub type PaddedAtomicU8 = CachePadded<AtomicU8>;

/// Helper macro to mark transition functions as cold
#[macro_export]
macro_rules! cold_transition {
    ($fn:item) => {
        #[cold]
        $fn
    };
}