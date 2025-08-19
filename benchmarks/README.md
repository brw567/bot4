# Bot4 Performance Benchmarks - Reproducibility Guide

## Hardware Manifest
```yaml
cpu_model: Intel Core i7-12700K
cpu_flags: avx512f avx512dq avx512cd avx512bw avx512vl
cores: 12 (8 P-cores + 4 E-cores)
ram: 32GB DDR5-5600
kernel: Linux 5.15.0-151-generic
mitigations: OFF (for benchmarking)
governor: performance
smt: enabled
microcode: 0xa4
```

## Build Configuration
```bash
# Compiler flags
export RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1"
export CARGO_PROFILE_RELEASE_LTO=true
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

# CPU affinity (P-cores only)
taskset -c 0-7 cargo bench --features avx512
```

## Benchmark Harness
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::time::Duration;

pub fn bench_hot_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_path");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    
    // Prevent over-optimization
    group.bench_function("decision_latency", |b| {
        let input = black_box(generate_market_data());
        b.iter(|| {
            let decision = make_trading_decision(black_box(&input));
            black_box(decision)
        });
    });
}

pub fn bench_ml_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("ml_inference");
    
    for batch_size in [1, 8, 32, 128].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, &size| {
                let features = black_box(generate_features(size));
                let model = load_model();
                b.iter(|| {
                    let pred = model.predict(black_box(&features));
                    black_box(pred)
                });
            },
        );
    }
}
```

## Reproducibility Protocol
1. Clean build: `cargo clean && cargo build --release`
2. Disable CPU throttling: `sudo cpupower frequency-set -g performance`
3. Pin to P-cores: `taskset -c 0-7`
4. Run 3 times, report median: `./scripts/bench_reproducible.sh`
5. Include perf counters: `perf stat -e cycles,instructions,cache-misses`

## Anti-Gaming Measures
- `black_box()` on all inputs/outputs
- Random input generation per iteration
- Cold cache runs included
- No inline assembly in hot path
- Verification of actual computation via output hash