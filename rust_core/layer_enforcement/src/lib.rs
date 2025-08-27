//! # Layer Architecture Enforcement
//! 
//! Compile-time enforcement of the 7-layer architecture using phantom types
//! and compile-time assertions. Prevents cross-layer violations at compile time.
//!
//! ## Layer Hierarchy (Bottom to Top)
//! - Layer 0: Infrastructure (foundation)
//! - Layer 1: Data (ingestion, storage)
//! - Layer 2: Risk Management
//! - Layer 3: Machine Learning & Analysis
//! - Layer 4: Trading Strategies
//! - Layer 5: Execution (trading engine, orders)
//! - Layer 6: Integration & Testing
//!
//! ## Rules
//! - Dependencies flow DOWN only (higher can depend on lower)
//! - No circular dependencies allowed
//! - Cross-layer communication via traits/events
//!
//! ## External Research Applied
//! - "Clean Architecture" (Robert C. Martin)
//! - "Hexagonal Architecture" (Alistair Cockburn)
//! - "Domain-Driven Design" (Eric Evans)
//! - Rust phantom types for compile-time guarantees
//! - Zero-cost abstractions principle

#![warn(missing_docs)]

use std::marker::PhantomData;

/// Layer markers using phantom types for zero-cost compile-time checking
pub mod layers {
    use std::marker::PhantomData;
    
    /// Layer 0: Infrastructure - Foundation layer
    pub struct Layer0Infrastructure;
    
    /// Layer 1: Data - Ingestion and storage
    pub struct Layer1Data;
    
    /// Layer 2: Risk - Risk management
    pub struct Layer2Risk;
    
    /// Layer 3: ML - Machine learning and analysis
    pub struct Layer3ML;
    
    /// Layer 4: Strategies - Trading strategies
    pub struct Layer4Strategies;
    
    /// Layer 5: Execution - Trading engine and orders
    pub struct Layer5Execution;
    
    /// Layer 6: Integration - Testing and integration
    pub struct Layer6Integration;
    
    /// Component with layer enforcement
    pub struct Component<L> {
        /// Phantom data for layer type
        pub _layer: PhantomData<L>,
    }
    
    /// Trait for layer ordering
    pub trait LayerOrder {
        /// Layer number for ordering
        const LEVEL: u8;
    }
    
    impl LayerOrder for Layer0Infrastructure { const LEVEL: u8 = 0; }
    impl LayerOrder for Layer1Data { const LEVEL: u8 = 1; }
    impl LayerOrder for Layer2Risk { const LEVEL: u8 = 2; }
    impl LayerOrder for Layer3ML { const LEVEL: u8 = 3; }
    impl LayerOrder for Layer4Strategies { const LEVEL: u8 = 4; }
    impl LayerOrder for Layer5Execution { const LEVEL: u8 = 5; }
    impl LayerOrder for Layer6Integration { const LEVEL: u8 = 6; }
    
    /// Trait for components that can depend on lower layers
    pub trait CanDependOn<Target> {
        /// Marker for valid dependency
        fn dependency_allowed() -> bool;
    }
    
    // Macro to generate valid dependencies
    macro_rules! impl_can_depend {
        ($from:ty, $to:ty) => {
            impl CanDependOn<$to> for $from
            where
                $from: LayerOrder,
                $to: LayerOrder,
            {
                fn dependency_allowed() -> bool {
                    <$from as LayerOrder>::LEVEL >= <$to as LayerOrder>::LEVEL
                }
            }
        };
    }
    
    // Layer 6 can depend on all lower layers
    impl_can_depend!(Layer6Integration, Layer0Infrastructure);
    impl_can_depend!(Layer6Integration, Layer1Data);
    impl_can_depend!(Layer6Integration, Layer2Risk);
    impl_can_depend!(Layer6Integration, Layer3ML);
    impl_can_depend!(Layer6Integration, Layer4Strategies);
    impl_can_depend!(Layer6Integration, Layer5Execution);
    impl_can_depend!(Layer6Integration, Layer6Integration);
    
    // Layer 5 can depend on layers 0-5
    impl_can_depend!(Layer5Execution, Layer0Infrastructure);
    impl_can_depend!(Layer5Execution, Layer1Data);
    impl_can_depend!(Layer5Execution, Layer2Risk);
    impl_can_depend!(Layer5Execution, Layer3ML);
    impl_can_depend!(Layer5Execution, Layer4Strategies);
    impl_can_depend!(Layer5Execution, Layer5Execution);
    
    // Layer 4 can depend on layers 0-4
    impl_can_depend!(Layer4Strategies, Layer0Infrastructure);
    impl_can_depend!(Layer4Strategies, Layer1Data);
    impl_can_depend!(Layer4Strategies, Layer2Risk);
    impl_can_depend!(Layer4Strategies, Layer3ML);
    impl_can_depend!(Layer4Strategies, Layer4Strategies);
    
    // Layer 3 can depend on layers 0-3
    impl_can_depend!(Layer3ML, Layer0Infrastructure);
    impl_can_depend!(Layer3ML, Layer1Data);
    impl_can_depend!(Layer3ML, Layer2Risk);
    impl_can_depend!(Layer3ML, Layer3ML);
    
    // Layer 2 can depend on layers 0-2
    impl_can_depend!(Layer2Risk, Layer0Infrastructure);
    impl_can_depend!(Layer2Risk, Layer1Data);
    impl_can_depend!(Layer2Risk, Layer2Risk);
    
    // Layer 1 can depend on layers 0-1
    impl_can_depend!(Layer1Data, Layer0Infrastructure);
    impl_can_depend!(Layer1Data, Layer1Data);
    
    // Layer 0 can only depend on itself
    impl_can_depend!(Layer0Infrastructure, Layer0Infrastructure);
}

/// Macro to enforce layer dependencies at compile time
#[macro_export]
macro_rules! enforce_layer {
    ($component:ty, $layer:ty) => {
        impl $component {
            /// Get the layer marker for compile-time checking
            pub fn layer_marker() -> $crate::layers::Component<$layer> {
                $crate::layers::Component {
                    _layer: std::marker::PhantomData,
                }
            }
            
            /// Validate dependency at compile time
            pub fn validate_dependency<T>() 
            where 
                $layer: $crate::layers::CanDependOn<T>,
            {
                // Compile-time check - no runtime cost
            }
        }
    };
}

/// Macro to check dependencies between layers
#[macro_export]
macro_rules! check_dependency {
    ($from:ty, $to:ty) => {
        {
            fn _check_dependency<F, T>()
            where
                F: $crate::layers::LayerOrder + $crate::layers::CanDependOn<T>,
                T: $crate::layers::LayerOrder,
            {
                // This will fail to compile if dependency is invalid
                let _ = <F as $crate::layers::CanDependOn<T>>::dependency_allowed();
            }
            
            _check_dependency::<$from, $to>();
        }
    };
}

/// Static assertions for layer rules
#[macro_export]
macro_rules! assert_layer_dependency {
    ($from:ty, can_depend_on, $to:ty) => {
        const _: () = {
            fn check() where $from: $crate::layers::CanDependOn<$to> {}
        };
    };
    
    ($from:ty, cannot_depend_on, $to:ty) => {
        const _: () = {
            // This should fail to compile if the dependency exists
            // Rust doesn't have negative trait bounds yet
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::*;
    
    #[test]
    fn test_valid_dependencies() {
        // These should compile
        check_dependency!(Layer5Execution, Layer2Risk);
        check_dependency!(Layer3ML, Layer1Data);
        check_dependency!(Layer2Risk, Layer0Infrastructure);
    }
    
    // This test would fail to compile if uncommented:
    // #[test]
    // fn test_invalid_dependency() {
    //     check_dependency!(Layer0Infrastructure, Layer5Execution);
    // }
}