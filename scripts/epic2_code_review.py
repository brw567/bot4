#!/usr/bin/env python3
"""
EPIC 2: Comprehensive Code Review Script
Each agent reviews their domain for issues
"""

import os
import re
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class CodeIssue:
    """Represents a code issue found during review"""
    file: str
    line: int
    category: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    description: str
    agent: str
    code_snippet: str

class CodeReviewSystem:
    """Multi-agent code review system"""
    
    def __init__(self):
        self.issues = []
        self.stats = {
            'files_reviewed': 0,
            'issues_found': 0,
            'critical_issues': 0,
            'by_agent': {}
        }
    
    def sam_review_calculations(self) -> List[CodeIssue]:
        """Sam reviews all mathematical calculations"""
        issues = []
        
        # Patterns Sam looks for
        patterns = {
            'hardcoded_percent': (r'0\.\d{2,}', 'medium'),  # 0.02, 0.015, etc
            'magic_number': (r'(?<!\.)(?<!\d)[2-9]\d{2,}(?!\d)', 'low'),  # Large magic numbers
            'simple_mult': (r'price\s*\*\s*0\.\d+', 'high'),  # price * 0.02
            'fixed_threshold': (r'if.*[<>]=?\s*0\.\d+', 'medium'),  # if x > 0.5
        }
        
        # Review strategy files
        for root, dirs, files in os.walk('src/strategies'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.stats['files_reviewed'] += 1
                    
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        # Skip comments and strings
                        if line.strip().startswith('#') or '"""' in line or "'''" in line:
                            continue
                        
                        for pattern_name, (pattern, severity) in patterns.items():
                            if re.search(pattern, line):
                                # Check if it's in config or constants
                                if 'config' in line.lower() or 'const' in line.lower():
                                    continue
                                
                                issues.append(CodeIssue(
                                    file=filepath,
                                    line=i,
                                    category='mathematical',
                                    severity=severity,
                                    description=f'Hardcoded value found: {pattern_name}',
                                    agent='Sam',
                                    code_snippet=line.strip()[:80]
                                ))
        
        print(f"Sam reviewed {self.stats['files_reviewed']} files")
        return issues
    
    def morgan_review_ml(self) -> List[CodeIssue]:
        """Morgan reviews ML implementations"""
        issues = []
        
        ml_files = [
            'src/core/ml_engine.py',
            'src/core/feature_store.py',
            'src/core/online_learning_system.py'
        ]
        
        for filepath in ml_files:
            if not os.path.exists(filepath):
                continue
            
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for ML best practices
            if 'train_test_split' not in content and 'ML' in filepath:
                issues.append(CodeIssue(
                    file=filepath,
                    line=0,
                    category='ml_validation',
                    severity='high',
                    description='Missing train_test_split',
                    agent='Morgan',
                    code_snippet='File missing train/test split'
                ))
            
            if 'cross_val' not in content and 'ML' in filepath:
                issues.append(CodeIssue(
                    file=filepath,
                    line=0,
                    category='ml_validation',
                    severity='medium',
                    description='No cross-validation found',
                    agent='Morgan',
                    code_snippet='Consider adding cross-validation'
                ))
            
            # Check for overfitting indicators
            for i, line in enumerate(lines, 1):
                if 'accuracy' in line and '1.0' in line:
                    issues.append(CodeIssue(
                        file=filepath,
                        line=i,
                        category='ml_overfitting',
                        severity='critical',
                        description='Perfect accuracy - possible overfitting',
                        agent='Morgan',
                        code_snippet=line.strip()[:80]
                    ))
        
        print(f"Morgan reviewed {len(ml_files)} ML files")
        return issues
    
    def quinn_review_risk(self) -> List[CodeIssue]:
        """Quinn reviews risk management"""
        issues = []
        
        risk_files = [
            'src/core/risk_engine.py',
            'src/core/monte_carlo_engine.py'
        ]
        
        for filepath in risk_files:
            if not os.path.exists(filepath):
                continue
            
            with open(filepath, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for risk controls
            if 'stop_loss' not in content.lower():
                issues.append(CodeIssue(
                    file=filepath,
                    line=0,
                    category='risk_control',
                    severity='critical',
                    description='No stop loss implementation found',
                    agent='Quinn',
                    code_snippet='Missing stop loss logic'
                ))
            
            # Check for position limits
            for i, line in enumerate(lines, 1):
                if 'position' in line.lower() and 'limit' not in line.lower():
                    if 'size' in line.lower() and 'max' not in line.lower():
                        issues.append(CodeIssue(
                            file=filepath,
                            line=i,
                            category='risk_limit',
                            severity='high',
                            description='Position without limit check',
                            agent='Quinn',
                            code_snippet=line.strip()[:80]
                        ))
        
        print(f"Quinn reviewed {len(risk_files)} risk files")
        return issues
    
    def jordan_review_performance(self) -> List[CodeIssue]:
        """Jordan reviews performance and scalability"""
        issues = []
        
        # Check for performance issues
        for root, dirs, files in os.walk('src'):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    
                    with open(filepath, 'r') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        # Check for sync operations in async context
                        if 'async def' in lines[max(0, i-5):i]:
                            if 'time.sleep' in line:
                                issues.append(CodeIssue(
                                    file=filepath,
                                    line=i,
                                    category='performance',
                                    severity='high',
                                    description='Sync sleep in async function',
                                    agent='Jordan',
                                    code_snippet=line.strip()[:80]
                                ))
                        
                        # Check for missing connection pooling
                        if 'connect(' in line and 'pool' not in line.lower():
                            issues.append(CodeIssue(
                                file=filepath,
                                line=i,
                                category='scalability',
                                severity='medium',
                                description='Direct connection without pooling',
                                agent='Jordan',
                                code_snippet=line.strip()[:80]
                            ))
        
        print(f"Jordan reviewed performance aspects")
        return issues[:10]  # Limit to first 10 issues
    
    def generate_report(self):
        """Generate comprehensive review report"""
        all_issues = []
        
        # Run all reviews
        print("\nEPIC 2: System Code Review")
        print("="*60)
        
        all_issues.extend(self.sam_review_calculations())
        all_issues.extend(self.morgan_review_ml())
        all_issues.extend(self.quinn_review_risk())
        all_issues.extend(self.jordan_review_performance())
        
        # Categorize issues
        critical = [i for i in all_issues if i.severity == 'critical']
        high = [i for i in all_issues if i.severity == 'high']
        medium = [i for i in all_issues if i.severity == 'medium']
        low = [i for i in all_issues if i.severity == 'low']
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_issues': len(all_issues),
                'critical': len(critical),
                'high': len(high),
                'medium': len(medium),
                'low': len(low),
                'files_reviewed': self.stats['files_reviewed']
            },
            'by_agent': {},
            'issues': []
        }
        
        # Group by agent
        for issue in all_issues:
            if issue.agent not in report['by_agent']:
                report['by_agent'][issue.agent] = 0
            report['by_agent'][issue.agent] += 1
            
            report['issues'].append({
                'file': issue.file,
                'line': issue.line,
                'category': issue.category,
                'severity': issue.severity,
                'description': issue.description,
                'agent': issue.agent,
                'snippet': issue.code_snippet
            })
        
        # Save report
        with open('EPIC2_REVIEW_REPORT.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nReview Summary")
        print("-"*60)
        print(f"Total Issues Found: {len(all_issues)}")
        print(f"  Critical: {len(critical)}")
        print(f"  High: {len(high)}")
        print(f"  Medium: {len(medium)}")
        print(f"  Low: {len(low)}")
        print("\nBy Agent:")
        for agent, count in report['by_agent'].items():
            print(f"  {agent}: {count} issues")
        
        print("\nTop Critical Issues:")
        for issue in critical[:5]:
            print(f"  • {issue.file}:{issue.line} - {issue.description}")
        
        print("\nReport saved to EPIC2_REVIEW_REPORT.json")
        
        return report

if __name__ == "__main__":
    reviewer = CodeReviewSystem()
    report = reviewer.generate_report()
    
    # Determine exit code
    if report['summary']['critical'] > 0:
        print("\n⚠️  Critical issues found - immediate attention required")
        exit(1)
    else:
        print("\n✅ No critical issues found")
        exit(0)