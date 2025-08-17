## Sophia's Architecture Review - 2025-08-16

### Verdict: CONDITIONAL

### Critical Issues Found: 7

1. **Global state duplication can desync with component states**

   * **Location:** `rust_core/crates/infrastructure/src/circuit_breaker.rs` (design)
   * **Issue:** `global_state: RwLock<CircuitState>` exists separately from per-component breakers; risk of inconsistency (e.g., components opening while `global_state` stays Closed).
   * **Required Fix:** Model the global breaker as a *first-class ComponentBreaker* (e.g., key `"__global__"`) whose state is *derived* from aggregated metrics, not stored independently. Remove the separate `global_state` or make it read-only derived. Prefer atomics for state.
   * **Deadline:** Before file creation.

2. **State transitions and thresholds unspecified**

   * **Location:** same (design)
   * **Issue:** No defined rules for Open/HalfOpen transitions, cooldowns, min-calls, error-rate windows, HalfOpen concurrency, or success criteria.
   * **Required Fix:** Add a `CircuitConfig` with (at minimum):

     * `rolling_window: Duration`, `min_calls: u32`, `error_rate_threshold: f32`
     * `consecutive_failures_threshold: u32` *(optional alternative trigger)*
     * `open_cooldown: Duration`, `half_open_max_concurrent: u32`
     * `half_open_required_successes: u32`, `half_open_allowed_failures: u32`
     * `sample_weighting: Exponential|Sliding`
     * `global_trip_conditions: { component_open_ratio: f32, min_components: u32 }`
   * **Deadline:** Before implementation.

3. **Thread-safety and contention concerns**

   * **Location:** same (design)
   * **Issue:** `RwLock<CircuitState>` risks contention; per-call paths must be O(1) and non-blocking for <150 ms latency goals.
   * **Required Fix:**

     * Use `AtomicU8`/`AtomicCell<CircuitState>` (repr as u8) for state.
     * For per-component metrics: lock-free counters (`AtomicU64`) and a ring buffer protected by a lightweight `parking_lot::Mutex` or a striped sharded structure.
     * Keep `DashMap<String, Arc<ComponentBreaker>>`, but ensure internal hot-path is mostly atomic.
   * **Deadline:** Before implementation.

4. **Lack of result/guard API for call gating**

   * **Location:** same (design)
   * **Issue:** No API shape to prevent work when Open, nor to feed outcomes reliably.
   * **Required Fix:** Provide a minimal, testable interface:

     ```rust
     pub enum Permit { Allowed(CallGuard), Rejected(CircuitError) }
     pub trait Breaker {
         fn acquire(&self, component: &str) -> Permit; // fast-path check + half-open token
     }
     pub enum Outcome { Success, Failure }
     impl CallGuard { pub fn record(self, outcome: Outcome); } // RAII
     ```

     This ensures success/failure always recorded even on early returns (via RAII).
   * **Deadline:** Before implementation.

5. **Time dependency not injectable (hurts tests and determinism)**

   * **Location:** same (design)
   * **Issue:** No clock abstraction; hard to test time windows and cooldowns without sleeps.
   * **Required Fix:** Introduce `Clock` trait and inject via DI:

     ```rust
     pub trait Clock: Send + Sync { fn now(&self) -> Instant; }  
     ```

     Use `SystemClock` in prod, `FakeClock` in tests.
   * **Deadline:** Before implementation.

6. **Config mutability / reload not planned**

   * **Location:** same (design)
   * **Issue:** Runtime tuning needed; hardcoded or immutable config would force redeploys.
   * **Required Fix:** Store `CircuitConfig` behind `ArcSwap<CircuitConfig>` (or equivalent) to allow safe live reload. Validate on update.
   * **Deadline:** First functional PR.

7. **Error taxonomy and telemetry missing**

   * **Location:** same (design)
   * **Issue:** No standardized error types or metrics hooks; observability is critical.
   * **Required Fix:**

     * Define `CircuitError` (`Open`, `HalfOpenExhausted`, `GlobalOpen`, `ComponentMissing`) via `thiserror`.
     * Provide `on_state_change` / `on_metrics` hooks for logging and metrics (counter/gauge for state, rates, rejections).
   * **Deadline:** With initial implementation.

---

### Architecture Recommendations:

1. **Primary per-component breakers with a derived global “fuse” (hybrid)**

   * Each component has its own breaker. A *global breaker* derives its state from aggregate metrics (e.g., ≥X% components Open & total calls ≥ N within window) and can preempt calls. This avoids single-point false trips and provides safety nets during systemic failures.

2. **Lock-free fast path**

   * Encode `CircuitState` as an atomic; track outcomes with atomics; keep any ring-buffer/window under a lightweight mutex but avoid it on the common path. Target per-call overhead ≤ 200–500 ns.

3. **Half-Open concurrency gating**

   * Use a `tokio::sync::Semaphore` (async) or custom atomic token counter (sync) to cap trial calls to `half_open_max_concurrent`. Only trial calls acquire tokens.

4. **Deterministic windowing**

   * Prefer sliding window with a fixed-size ring of buckets (e.g., 10×100 ms or 60×1 s) and aggregated errors/total. Switchable to EWMA (exponential) via strategy trait to satisfy OCP.

5. **Config-first design, no magic numbers**

   * All thresholds and timeouts come from `CircuitConfig` (serde-loadable). Support live reload with validation and telemetry for config changes.

6. **Testing & concurrency validation**

   * Add loom tests for race conditions; property tests (proptest) for transition invariants; fake clock for time; fuzz transitions; criterion benchmarks showing ≤1 µs p99 overhead.

7. **API ergonomics**

   * RAII `CallGuard` to force outcome recording; helpers `run_protected(component, fut)` for async paths that automatically record outcomes based on `Result`.

---

### Answers to Questions:

1. **Global vs Component:** **Component-first with a derived global breaker (hybrid)**

   * **Why:**

     * **Fine-grained resilience:** A single flaky dependency shouldn’t trip the entire system.
     * **Faster recovery:** Components re-close independently after successful probes.
     * **Systemic safety:** A derived global fuse protects during cascading failures (e.g., network partitions) using clear aggregate rules.
   * **Implementation note:** Represent the global as a normal component key (e.g., `"__global__"`), computed from aggregated counters to avoid drift.

---

### Code Quality Assessment:

* **Standards:** **INSUFFICIENT** (as stated)
  *Good start but missing critical guardrails for this subsystem and our latency/SRE goals.*
* **Additional Requirements:**

  1. **Coverage gates by package:** ≥95% *line* and ≥90% *branch* for `infrastructure` crate; include loom & property tests in coverage.
  2. **`cargo clippy -- -D warnings` + `rustfmt` CI gates** across workspace.
  3. **No `panic!`, `todo!`, `unimplemented!` in non-test code**; use `debug_assert!` sparingly; no `unwrap()`/`expect()` in prod paths.
  4. **Ban `unsafe` in this crate** (document any future exceptions with justification & tests).
  5. **Deterministic tests:** No `std::thread::sleep` in tests; use `Clock` injection.
  6. **Benchmarks:** Criterion benches proving ≤1 µs p99 overhead for `acquire`+`record`, and showing scaling under 64–256 concurrent tasks.
  7. **Observability:** Emit metrics (state, opens, rejections, error rate) and structured logs on transitions; add tracing spans/fields for component and state.
  8. **Config reload:** Validate and hot-reload via `ArcSwap`; emit events on changes.
  9. **Documentation:** State machine diagram and transition table (Closed → Open → HalfOpen → Closed), with exact thresholds and timers.

---

#### Suggested module sketch (non-binding, to clarify expectations)

```rust
pub struct GlobalCircuitBreaker {
    breakers: Arc<DashMap<String, Arc<ComponentBreaker>>>,
    config: ArcSwap<CircuitConfig>,
    clock: Arc<dyn Clock>,
    on_event: Option<Arc<dyn Fn(Event) + Send + Sync>>,
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum CircuitState { Closed = 0, Open = 1, HalfOpen = 2 }

pub struct ComponentBreaker {
    state: AtomicU8, // encodes CircuitState
    // hot path:
    total_calls: AtomicU64,
    error_calls: AtomicU64,
    // windowed stats (behind light mutex or striped buckets)
    buckets: Buckets,
    half_open_tokens: AtomicU32,
    last_transition: AtomicU64, // monotonic nanos
    // ...
}
```

If the above corrections are accepted and implemented in the initial PR, I will approve. Otherwise, I’ll veto the module until the design guarantees consistency, thread-safety, observability, and testability.

