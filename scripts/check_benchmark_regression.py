#!/usr/bin/env python3
"""
Benchmark Regression Detection Script
Riley: Analyzes Criterion benchmark results to detect performance regressions
Part of CI/CD Quality Gates - 48hr deadline requirement
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Performance regression thresholds
THRESHOLDS = {
    'critical': 0.10,   # 10% regression fails the build
    'warning': 0.05,    # 5% regression produces warning
    'improvement': -0.05 # 5% improvement is celebrated
}

# Critical benchmarks that must not regress
CRITICAL_BENCHMARKS = [
    'circuit_breaker',
    'risk_engine',
    'order_processing',
    'market_data_parse',
    'dcc_garch_forecast',
    'memory_allocation'
]

def parse_criterion_output(base_path: Path) -> Dict[str, float]:
    """Parse Criterion benchmark results from JSON files"""
    results = {}
    
    criterion_dir = base_path / "target" / "criterion"
    if not criterion_dir.exists():
        print(f"❌ No benchmark results found at {criterion_dir}")
        return results
    
    # Parse each benchmark group
    for group_dir in criterion_dir.iterdir():
        if not group_dir.is_dir() or group_dir.name in ['baseline', 'current']:
            continue
            
        # Look for estimates.json
        estimates_file = group_dir / "current" / "estimates.json"
        if estimates_file.exists():
            with open(estimates_file, 'r') as f:
                data = json.load(f)
                # Extract mean execution time in nanoseconds
                if 'mean' in data:
                    results[group_dir.name] = data['mean']['point_estimate']
    
    return results

def compare_benchmarks(current: Dict[str, float], baseline: Dict[str, float]) -> List[Tuple[str, float, float, float]]:
    """Compare current benchmarks against baseline"""
    comparisons = []
    
    for bench_name, current_time in current.items():
        if bench_name in baseline:
            baseline_time = baseline[bench_name]
            # Calculate percentage change
            change = (current_time - baseline_time) / baseline_time
            comparisons.append((bench_name, current_time, baseline_time, change))
    
    return comparisons

def check_regressions(comparisons: List[Tuple[str, float, float, float]]) -> Tuple[bool, List[str], List[str]]:
    """Check for performance regressions"""
    has_critical = False
    warnings = []
    improvements = []
    
    for bench_name, current, baseline, change in comparisons:
        # Convert to human-readable format
        current_ms = current / 1_000_000
        baseline_ms = baseline / 1_000_000
        change_pct = change * 100
        
        # Check if this is a critical benchmark
        is_critical = any(critical in bench_name for critical in CRITICAL_BENCHMARKS)
        
        if change > THRESHOLDS['critical']:
            if is_critical:
                has_critical = True
                warnings.append(
                    f"❌ CRITICAL REGRESSION: {bench_name}\n"
                    f"   Current: {current_ms:.3f}ms, Baseline: {baseline_ms:.3f}ms\n"
                    f"   Regression: {change_pct:.1f}%"
                )
            else:
                warnings.append(
                    f"⚠️  REGRESSION: {bench_name}\n"
                    f"   Current: {current_ms:.3f}ms, Baseline: {baseline_ms:.3f}ms\n"
                    f"   Regression: {change_pct:.1f}%"
                )
        elif change > THRESHOLDS['warning']:
            warnings.append(
                f"⚠️  Warning: {bench_name} regressed by {change_pct:.1f}%\n"
                f"   Current: {current_ms:.3f}ms, Baseline: {baseline_ms:.3f}ms"
            )
        elif change < THRESHOLDS['improvement']:
            improvements.append(
                f"✅ Improvement: {bench_name} improved by {abs(change_pct):.1f}%\n"
                f"   Current: {current_ms:.3f}ms, Baseline: {baseline_ms:.3f}ms"
            )
    
    return has_critical, warnings, improvements

def validate_performance_targets(current: Dict[str, float]) -> List[str]:
    """Validate that benchmarks meet absolute performance targets"""
    violations = []
    
    # Define absolute performance targets (in nanoseconds)
    TARGETS = {
        'decision_latency': 1_000,        # 1μs
        'risk_check': 10_000,             # 10μs  
        'circuit_breaker': 100,           # 100ns
        'order_internal': 100_000,        # 100μs
        'memory_allocation': 10,          # 10ns
        'dcc_garch_forecast': 1_000_000,  # 1ms
    }
    
    for target_name, max_ns in TARGETS.items():
        # Find matching benchmarks
        for bench_name, time_ns in current.items():
            if target_name in bench_name.lower():
                if time_ns > max_ns:
                    violations.append(
                        f"❌ Performance target violation: {bench_name}\n"
                        f"   Current: {time_ns/1000:.1f}μs, Target: ≤{max_ns/1000:.1f}μs"
                    )
    
    return violations

def main():
    """Main entry point for regression detection"""
    print("=" * 60)
    print("BENCHMARK REGRESSION DETECTION")
    print("=" * 60)
    
    # Find project root
    project_root = Path(os.getcwd())
    if project_root.name != 'rust_core':
        project_root = project_root / 'rust_core'
    
    # Parse current and baseline results
    current = parse_criterion_output(project_root)
    
    baseline_path = project_root / "target" / "criterion" / "baseline"
    baseline = {}
    if baseline_path.exists():
        # Copy baseline structure and parse
        baseline = parse_criterion_output(Path(str(project_root).replace('/current/', '/baseline/')))
    
    if not current:
        print("❌ No current benchmark results found")
        sys.exit(1)
    
    print(f"Found {len(current)} benchmark results")
    
    # Check absolute performance targets
    print("\n📊 Checking Performance Targets...")
    violations = validate_performance_targets(current)
    if violations:
        for violation in violations:
            print(violation)
    else:
        print("✅ All performance targets met!")
    
    # Compare against baseline if available
    if baseline:
        print(f"\n📊 Comparing against baseline ({len(baseline)} benchmarks)...")
        comparisons = compare_benchmarks(current, baseline)
        
        has_critical, warnings, improvements = check_regressions(comparisons)
        
        # Print improvements
        if improvements:
            print("\n🎉 Performance Improvements:")
            for improvement in improvements:
                print(improvement)
        
        # Print warnings
        if warnings:
            print("\n⚠️  Performance Warnings:")
            for warning in warnings:
                print(warning)
        
        # Exit with error if critical regression
        if has_critical:
            print("\n❌ CRITICAL PERFORMANCE REGRESSION DETECTED!")
            print("Build failed due to performance regression in critical path")
            sys.exit(1)
        
        # Exit with error if absolute targets violated
        if violations:
            print("\n❌ PERFORMANCE TARGETS NOT MET!")
            sys.exit(1)
            
        print("\n✅ No critical regressions detected")
    else:
        print("\n📝 No baseline found - current results will become the baseline")
        
        # Still check absolute targets
        if violations:
            print("\n❌ PERFORMANCE TARGETS NOT MET!")
            sys.exit(1)
    
    print("\n✅ Benchmark validation passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()