#!/usr/bin/env python3
"""
P99.9 Performance Gate Checker
Owner: Jordan | Pre-Production Requirement #6
Parses benchmark results and validates P99.9 latencies
"""

import json
import sys
import statistics
from typing import Dict, List, Tuple

# P99.9 performance targets (in microseconds)
PERFORMANCE_GATES = {
    "feature_vector": 15.0,      # 15μs (3x P99 target of 5μs)
    "order_submission": 300.0,    # 300μs (3x P99 target of 100μs)
    "risk_check": 30.0,          # 30μs (3x P99 target of 10μs)
    "sma_indicator": 0.6,        # 600ns (3x P99 target of 200ns)
    "rsi_indicator": 1.5,        # 1.5μs (3x P99 target of 500ns)
    "inference": 0.15,           # 150ns (3x P99 target of 50ns)
}

def parse_criterion_output(filepath: str) -> Dict[str, float]:
    """Parse Criterion benchmark JSON output"""
    results = {}
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    for benchmark in data.get('benchmarks', []):
        name = benchmark['name']
        
        # Extract percentiles
        if 'percentiles' in benchmark:
            p99_9 = benchmark['percentiles'].get('99.9', 0)
            # Convert to microseconds
            p99_9_us = p99_9 / 1000.0 if benchmark['unit'] == 'ns' else p99_9
            results[name] = p99_9_us
    
    return results

def check_contention_results(filepath: str) -> Tuple[float, float, float]:
    """Parse contention test results"""
    throughput = 0
    p99_9_latency = 0
    fairness_ratio = 0
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if 'Throughput:' in line:
            throughput = float(line.split(':')[1].split()[0])
        elif 'P99.9 latency:' in line:
            p99_9_latency = float(line.split(':')[1].split()[0])
        elif 'Fairness ratio:' in line:
            fairness_ratio = float(line.split(':')[1].split()[0])
    
    return throughput, p99_9_latency, fairness_ratio

def calculate_p99_9(values: List[float]) -> float:
    """Calculate P99.9 percentile"""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = int(len(sorted_values) * 0.999)
    return sorted_values[min(index, len(sorted_values) - 1)]

def main():
    if len(sys.argv) < 2:
        print("Usage: check_p99_9_gates.py <latency_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    # Parse benchmark results
    results = parse_criterion_output(results_file)
    
    # Check each gate
    all_pass = True
    output = {
        "all_pass": True,
        "gates": {}
    }
    
    for gate_name, target in PERFORMANCE_GATES.items():
        actual = results.get(gate_name, float('inf'))
        passed = actual <= target
        
        if not passed:
            all_pass = False
            print(f"❌ {gate_name}: {actual:.3f}μs (target: {target:.3f}μs)")
        else:
            print(f"✅ {gate_name}: {actual:.3f}μs (target: {target:.3f}μs)")
        
        output["gates"][gate_name] = {
            "target": target,
            "actual": actual,
            "passed": passed
        }
    
    output["all_pass"] = all_pass
    
    # Check for performance regression
    if "baseline.json" in sys.argv:
        baseline_file = sys.argv[sys.argv.index("baseline.json")]
        baseline = parse_criterion_output(baseline_file)
        
        for name, current in results.items():
            if name in baseline:
                prev = baseline[name]
                regression = ((current - prev) / prev) * 100
                
                if regression > 10:  # More than 10% regression
                    print(f"⚠️  {name}: {regression:.1f}% regression detected")
                    output["regressions"].append({
                        "name": name,
                        "regression_percent": regression
                    })
    
    # Write output for GitHub Actions
    with open("performance_gate_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Set GitHub Actions output
    print(f"::set-output name=gates_passed::{str(all_pass).lower()}")
    
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()