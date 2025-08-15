#!/usr/bin/env python3
"""
Comprehensive Mock Data and Fake Implementation Audit
Sam's domain - Zero tolerance for fake implementations
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MockDataIssue:
    file_path: str
    line_number: int
    issue_type: str
    severity: str  # critical, high, medium, low
    code_snippet: str
    description: str

class MockDataAuditor:
    def __init__(self):
        self.issues: List[MockDataIssue] = []
        self.stats = {
            'files_checked': 0,
            'lines_checked': 0,
            'critical_issues': 0,
            'high_issues': 0,
            'medium_issues': 0,
            'low_issues': 0
        }
        
        # Patterns to detect
        self.patterns = {
            'random_generators': {
                'pattern': r'(np\.random\.|random\.)(uniform|choice|randint|random|gauss|normal|randn)',
                'severity': 'high',
                'description': 'Random data generator found'
            },
            'placeholder_values': {
                'pattern': r'#\s*(placeholder|mock|fake|dummy|todo|fixme|xxx|hack)',
                'severity': 'medium',
                'description': 'Placeholder comment found'
            },
            'fake_calculations': {
                'pattern': r'(price|value|amount)\s*\*\s*0\.\d+(?!\d)',
                'severity': 'critical',
                'description': 'Fake calculation using fixed percentage'
            },
            'hardcoded_test_values': {
                'pattern': r'(return|=)\s*(50000|42|123|999|0\.5|"test"|\'test\'|"mock"|\'mock\')',
                'severity': 'medium',
                'description': 'Hardcoded test value'
            },
            'mock_classes': {
                'pattern': r'class\s+(Mock|Fake|Dummy|Test)\w+',
                'severity': 'high',
                'description': 'Mock class definition'
            },
            'not_implemented': {
                'pattern': r'(raise\s+)?NotImplemented(Error)?|pass\s*#.*implement',
                'severity': 'critical',
                'description': 'Not implemented functionality'
            },
            'simulated_data': {
                'pattern': r'(simulate|generate)_(data|prices|returns|signals)',
                'severity': 'high',
                'description': 'Data simulation function'
            },
            'example_usage': {
                'pattern': r'example_usage|sample_data|test_data',
                'severity': 'low',
                'description': 'Example or test data'
            }
        }
        
    def audit_file(self, file_path: str) -> None:
        """Audit a single file for mock data and fake implementations"""
        self.stats['files_checked'] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, 1):
                self.stats['lines_checked'] += 1
                
                # Check each pattern
                for pattern_name, pattern_info in self.patterns.items():
                    if re.search(pattern_info['pattern'], line, re.IGNORECASE):
                        issue = MockDataIssue(
                            file_path=file_path,
                            line_number=i,
                            issue_type=pattern_name,
                            severity=pattern_info['severity'],
                            code_snippet=line.strip(),
                            description=pattern_info['description']
                        )
                        self.issues.append(issue)
                        self.stats[f"{pattern_info['severity']}_issues"] += 1
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    def audit_directory(self, directory: str) -> None:
        """Recursively audit all Python files in directory"""
        for root, dirs, files in os.walk(directory):
            # Skip __pycache__ and .git directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', '.venv']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.audit_file(file_path)
                    
    def generate_report(self) -> str:
        """Generate comprehensive audit report"""
        report = []
        report.append("=" * 80)
        report.append("MOCK DATA AND FAKE IMPLEMENTATION AUDIT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Statistics
        report.append("STATISTICS:")
        report.append("-" * 40)
        report.append(f"Files checked: {self.stats['files_checked']}")
        report.append(f"Lines checked: {self.stats['lines_checked']}")
        report.append(f"Critical issues: {self.stats['critical_issues']}")
        report.append(f"High issues: {self.stats['high_issues']}")
        report.append(f"Medium issues: {self.stats['medium_issues']}")
        report.append(f"Low issues: {self.stats['low_issues']}")
        report.append(f"Total issues: {len(self.issues)}")
        report.append("")
        
        # Group issues by severity
        critical_issues = [i for i in self.issues if i.severity == 'critical']
        high_issues = [i for i in self.issues if i.severity == 'high']
        medium_issues = [i for i in self.issues if i.severity == 'medium']
        low_issues = [i for i in self.issues if i.severity == 'low']
        
        # Critical Issues (Must Fix)
        if critical_issues:
            report.append("🚨 CRITICAL ISSUES (MUST FIX IMMEDIATELY):")
            report.append("=" * 40)
            for issue in critical_issues[:20]:  # Limit to first 20
                report.append(f"\n📁 {issue.file_path}")
                report.append(f"  Line {issue.line_number}: {issue.description}")
                report.append(f"  > {issue.code_snippet}")
                
        # High Priority Issues
        if high_issues:
            report.append("\n⚠️  HIGH PRIORITY ISSUES:")
            report.append("=" * 40)
            for issue in high_issues[:20]:  # Limit to first 20
                report.append(f"\n📁 {issue.file_path}")
                report.append(f"  Line {issue.line_number}: {issue.description}")
                report.append(f"  > {issue.code_snippet}")
                
        # Summary by file
        report.append("\n📊 ISSUES BY FILE:")
        report.append("=" * 40)
        
        file_issues = {}
        for issue in self.issues:
            if issue.file_path not in file_issues:
                file_issues[issue.file_path] = []
            file_issues[issue.file_path].append(issue)
            
        # Sort by number of issues
        sorted_files = sorted(file_issues.items(), key=lambda x: len(x[1]), reverse=True)
        
        for file_path, issues in sorted_files[:20]:  # Top 20 files
            critical_count = sum(1 for i in issues if i.severity == 'critical')
            high_count = sum(1 for i in issues if i.severity == 'high')
            medium_count = sum(1 for i in issues if i.severity == 'medium')
            low_count = sum(1 for i in issues if i.severity == 'low')
            
            report.append(f"\n{file_path}")
            report.append(f"  Critical: {critical_count}, High: {high_count}, Medium: {medium_count}, Low: {low_count}")
            
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        report.append("=" * 40)
        
        if critical_issues:
            report.append("1. ❌ FIX ALL CRITICAL ISSUES IMMEDIATELY")
            report.append("   - Remove all fake calculations")
            report.append("   - Implement all NotImplemented methods")
            
        if high_issues:
            report.append("2. ⚠️  REPLACE MOCK DATA GENERATORS")
            report.append("   - Connect to real data sources")
            report.append("   - Use historical data for backtesting")
            report.append("   - Remove all random generators from production code")
            
        report.append("3. 📝 COMPLETE ALL TODOs")
        report.append("   - Implement pending functionality")
        report.append("   - Remove placeholder values")
        
        report.append("4. 🔄 REFACTOR EXAMPLE CODE")
        report.append("   - Move example_usage files to tests/")
        report.append("   - Separate mock implementations from production")
        
        # Final verdict
        report.append("\n" + "=" * 80)
        if critical_issues or self.stats['critical_issues'] > 0:
            report.append("❌ SAM'S VERDICT: REJECTED - CRITICAL ISSUES FOUND!")
            report.append("Fix all critical issues before going to production.")
        elif high_issues:
            report.append("⚠️  SAM'S VERDICT: NEEDS WORK - HIGH PRIORITY ISSUES FOUND")
            report.append("Address high priority issues before production deployment.")
        else:
            report.append("✅ SAM'S VERDICT: ACCEPTABLE - Only minor issues found")
            
        report.append("=" * 80)
        
        return "\n".join(report)
        
    def save_json_report(self, output_file: str) -> None:
        """Save detailed JSON report for further analysis"""
        issues_data = []
        for issue in self.issues:
            issues_data.append({
                'file': issue.file_path,
                'line': issue.line_number,
                'type': issue.issue_type,
                'severity': issue.severity,
                'code': issue.code_snippet,
                'description': issue.description
            })
            
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats,
            'issues': issues_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
def main():
    auditor = MockDataAuditor()
    
    # Audit source directory
    src_dir = '/home/hamster/bot4/src'
    print(f"Auditing {src_dir}...")
    auditor.audit_directory(src_dir)
    
    # Generate and print report
    report = auditor.generate_report()
    print(report)
    
    # Save JSON report
    json_file = '/home/hamster/bot4/mock_data_audit.json'
    auditor.save_json_report(json_file)
    print(f"\nDetailed JSON report saved to: {json_file}")
    
    # Exit with error if critical issues found
    if auditor.stats['critical_issues'] > 0:
        exit(1)
    elif auditor.stats['high_issues'] > 0:
        exit(2)
    else:
        exit(0)

if __name__ == "__main__":
    main()