#!/usr/bin/env python3
"""
Performance Gates Checker - Enforces latency/throughput requirements
Required by Sophia's review for CI gates
Owner: Jordan
"""

import re
import sys
from typing import Dict, List, Optional

class PerformanceGates:
    def __init__(self):
        # Performance targets from Sophia's review
        self.gates = {
            'decision_latency_p99': {'target': 1.0, 'unit': 'µs'},
            'risk_check_p99': {'target': 10.0, 'unit': 'µs'},
            'order_internal_p99': {'target': 100.0, 'unit': 'µs'},
            'cb_check_p99': {'target': 0.1, 'unit': 'µs'},  # 100ns
            'allocation_p99': {'target': 0.01, 'unit': 'µs'},  # 10ns
            'throughput': {'target': 500000, 'unit': 'ops/s'},
            'orders_per_sec': {'target': 5000, 'unit': 'orders/s'},
        }
        
        self.results = {}
        self.failures = []
        
    def parse_benchmark_output(self, content: str) -> None:
        """Parse benchmark results from cargo bench output."""
        
        # Pattern for criterion benchmark results
        # Example: test bench_risk_check ... bench:         8.2 µs/iter (+/- 0.5)
        bench_pattern = r'(\w+).*?bench:\s+([0-9.]+)\s*([nµm]s)'
        
        for match in re.finditer(bench_pattern, content):
            name = match.group(1)
            value = float(match.group(2))
            unit = match.group(3)
            
            # Convert to microseconds
            if unit == 'ns':
                value = value / 1000
            elif unit == 'ms':
                value = value * 1000
                
            self.results[name] = {'value': value, 'unit': 'µs'}
        
        # Pattern for throughput results
        # Example: throughput: 523,456 ops/sec
        throughput_pattern = r'throughput[:\s]+([0-9,]+)\s*ops'
        match = re.search(throughput_pattern, content, re.IGNORECASE)
        if match:
            value = int(match.group(1).replace(',', ''))
            self.results['throughput'] = {'value': value, 'unit': 'ops/s'}
    
    def check_gates(self) -> bool:
        """Check if all performance gates pass."""
        
        for gate_name, gate_spec in self.gates.items():
            # Find matching result
            result = None
            for result_name, result_data in self.results.items():
                if gate_name in result_name.lower() or result_name in gate_name:
                    result = result_data
                    break
            
            if not result:
                self.failures.append(f"Missing benchmark for {gate_name}")
                continue
            
            # Check if gate passes
            if gate_spec['unit'] == result.get('unit'):
                if 'throughput' in gate_name or 'orders' in gate_name:
                    # For throughput, higher is better
                    if result['value'] < gate_spec['target']:
                        self.failures.append(
                            f"{gate_name}: {result['value']:.1f} < {gate_spec['target']} {gate_spec['unit']}"
                        )
                else:
                    # For latency, lower is better
                    if result['value'] > gate_spec['target']:
                        self.failures.append(
                            f"{gate_name}: {result['value']:.2f} > {gate_spec['target']} {gate_spec['unit']}"
                        )
        
        return len(self.failures) == 0
    
    def generate_report(self) -> str:
        """Generate a performance report."""
        report = ["=" * 60]
        report.append("Bot4 Performance Gates Report")
        report.append("=" * 60)
        report.append("")
        
        # Show results
        report.append("📊 Benchmark Results:")
        for name, data in sorted(self.results.items()):
            report.append(f"  • {name}: {data['value']:.2f} {data['unit']}")
        
        report.append("")
        report.append("🎯 Performance Gates:")
        
        for gate_name, gate_spec in self.gates.items():
            result = None
            for result_name, result_data in self.results.items():
                if gate_name in result_name.lower() or result_name in gate_name:
                    result = result_data
                    break
            
            if result:
                if 'throughput' in gate_name or 'orders' in gate_name:
                    passed = result['value'] >= gate_spec['target']
                else:
                    passed = result['value'] <= gate_spec['target']
                
                status = "✅ PASS" if passed else "❌ FAIL"
                report.append(
                    f"  • {gate_name}: {result['value']:.2f} vs {gate_spec['target']} {gate_spec['unit']} - {status}"
                )
            else:
                report.append(f"  • {gate_name}: NO DATA - ❌ FAIL")
        
        if self.failures:
            report.append("")
            report.append("❌ FAILURES:")
            for failure in self.failures:
                report.append(f"  • {failure}")
            report.append("")
            report.append("⚠️  Performance gates FAILED - merge blocked!")
        else:
            report.append("")
            report.append("✅ All performance gates PASSED!")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def run(self, benchmark_file: str) -> int:
        """Run performance gate checks."""
        
        try:
            with open(benchmark_file, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"❌ Benchmark file not found: {benchmark_file}")
            return 1
        
        self.parse_benchmark_output(content)
        self.check_gates()
        
        print(self.generate_report())
        
        return 0 if not self.failures else 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: check_performance_gates.py <benchmark_results.txt>")
        sys.exit(1)
    
    checker = PerformanceGates()
    sys.exit(checker.run(sys.argv[1]))