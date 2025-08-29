//! STRATEGIES CRATE - Advanced Trading Strategies with Game Theory & SIMD
//! Full team collaboration: All 8 agents

pub mod game_theory_router;
pub mod simd_indicators;

// Re-exports for convenience
pub use game_theory_router::{
    GameTheoryRouter,
    NashEquilibriumSolver,
    ShapleyValueAllocator,
    PrisonersDilemmaDetector,
    ColonelBlottoStrategy,
    ChickenGameAnalyzer,
    AggressionStrategy,
};

pub use simd_indicators::{
    SimdBollingerBands,
    SimdRSI,
    SimdMACD,
    BollingerResult,
    MACDResult,
};