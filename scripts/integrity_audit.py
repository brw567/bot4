#!/usr/bin/env python3
"""
Comprehensive Integrity Audit Script
Finds all temporary implementations, TODOs, and incomplete code
Zero tolerance policy enforcement
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class IntegrityIssue:
    file_path: str
    line_number: int
    category: str
    severity: str
    code_snippet: str
    description: str
    owner: str  # Which agent should handle this

class IntegrityAuditor:
    def __init__(self):
        self.issues = []
        self.statistics = {
            'total_files': 0,
            'total_lines': 0,
            'todos': 0,
            'temporary_implementations': 0,
            'mock_data': 0,
            'placeholders': 0,
            'incomplete_apis': 0,
            'hardcoded_values': 0,
            'missing_implementations': 0
        }
        
        # Comprehensive patterns to detect issues
        self.patterns = {
            'todos': {
                'regex': r'#\s*(TODO|FIXME|XXX|HACK|BUG|REFACTOR|OPTIMIZE):?\s*(.*)$',
                'severity': 'high',
                'owner': 'alex',
                'description': 'Unfinished task'
            },
            'temporary_implementations': {
                'regex': r'(for now|will be replaced|would use|production would|simplified|temporary|quick fix)',
                'severity': 'critical',
                'owner': 'sam',
                'description': 'Temporary implementation'
            },
            'placeholder_returns': {
                'regex': r'(return\s+({}|\[\]|None|0|""|\'\'|pass))\s*#.*placeholder',
                'severity': 'critical',
                'owner': 'sam',
                'description': 'Placeholder return value'
            },
            'mock_data_generators': {
                'regex': r'(mock|fake|dummy|sample|test)[\s_]*(data|value|response|result)',
                'severity': 'critical',
                'owner': 'avery',
                'description': 'Mock data usage'
            },
            'hardcoded_values': {
                'regex': r'(=\s*(0\.01|0\.02|0\.03|0\.05|0\.1|0\.98|0\.95|0\.99)\s*[,\)])',
                'severity': 'high',
                'owner': 'sam',
                'description': 'Hardcoded percentage value'
            },
            'not_implemented': {
                'regex': r'(raise\s+)?NotImplemented(Error)?|pass\s*$',
                'severity': 'critical',
                'owner': 'alex',
                'description': 'Missing implementation'
            },
            'simple_calculations': {
                'regex': r'#.*[Ss]imple|#.*[Ss]implified|#.*[Bb]asic',
                'severity': 'high',
                'owner': 'sam',
                'description': 'Oversimplified calculation'
            },
            'missing_error_handling': {
                'regex': r'except:\s*(pass|continue)',
                'severity': 'high',
                'owner': 'jordan',
                'description': 'Missing error handling'
            },
            'console_output': {
                'regex': r'(print\(|console\.log)',
                'severity': 'medium',
                'owner': 'jordan',
                'description': 'Debug output in code'
            },
            'incomplete_api_endpoints': {
                'regex': r'#\s*TODO:\s*(Get|Implement|Connect|Integrate)',
                'severity': 'critical',
                'owner': 'casey',
                'description': 'Incomplete API endpoint'
            }
        }
        
        # Agent responsibility mapping
        self.agent_owners = {
            'alex': 'Alex (Team Lead) - Architecture & Strategy',
            'sam': 'Sam (Quant) - Mathematical & Trading Logic',
            'morgan': 'Morgan (ML) - Machine Learning & Models',
            'quinn': 'Quinn (Risk) - Risk Management',
            'jordan': 'Jordan (DevOps) - Infrastructure & Performance',
            'casey': 'Casey (Exchange) - Exchange Integration',
            'riley': 'Riley (Frontend) - UI/UX',
            'avery': 'Avery (Data) - Data Pipeline & Integrity'
        }
    
    def audit_file(self, file_path: str) -> None:
        """Audit a single file for integrity issues"""
        self.statistics['total_files'] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                self.statistics['total_lines'] += 1
                self.check_line_for_issues(file_path, line_num, line)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    def check_line_for_issues(self, file_path: str, line_num: int, line: str) -> None:
        """Check a single line for all types of issues"""
        
        for pattern_name, pattern_info in self.patterns.items():
            if re.search(pattern_info['regex'], line, re.IGNORECASE):
                # Determine the responsible agent based on file content
                owner = self.determine_owner(file_path, pattern_info['owner'])
                
                issue = IntegrityIssue(
                    file_path=file_path,
                    line_number=line_num,
                    category=pattern_name,
                    severity=pattern_info['severity'],
                    code_snippet=line.strip(),
                    description=pattern_info['description'],
                    owner=owner
                )
                
                self.issues.append(issue)
                
                # Update statistics
                if 'todo' in pattern_name.lower():
                    self.statistics['todos'] += 1
                elif 'temporary' in pattern_name.lower():
                    self.statistics['temporary_implementations'] += 1
                elif 'mock' in pattern_name.lower():
                    self.statistics['mock_data'] += 1
                elif 'placeholder' in pattern_name.lower():
                    self.statistics['placeholders'] += 1
                elif 'api' in pattern_name.lower():
                    self.statistics['incomplete_apis'] += 1
                elif 'hardcoded' in pattern_name.lower():
                    self.statistics['hardcoded_values'] += 1
                elif 'not_implemented' in pattern_name.lower():
                    self.statistics['missing_implementations'] += 1
    
    def determine_owner(self, file_path: str, default_owner: str) -> str:
        """Determine which agent should own fixing this issue"""
        
        # File-based ownership rules
        if 'ml_' in file_path or 'model' in file_path:
            return 'morgan'
        elif 'risk' in file_path:
            return 'quinn'
        elif 'exchange' in file_path or 'order' in file_path:
            return 'casey'
        elif 'api/routers' in file_path:
            return 'riley'
        elif 'strategies' in file_path:
            return 'sam'
        elif 'data' in file_path or 'feature' in file_path:
            return 'avery'
        elif 'monitoring' in file_path or 'performance' in file_path:
            return 'jordan'
        else:
            return default_owner
    
    def audit_directory(self, directory: str) -> None:
        """Recursively audit all Python files in directory"""
        for root, dirs, files in os.walk(directory):
            # Skip directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'venv', '.venv']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.audit_file(file_path)
    
    def generate_report(self) -> str:
        """Generate comprehensive integrity audit report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE INTEGRITY AUDIT REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall Statistics
        report.append("📊 OVERALL STATISTICS:")
        report.append("-" * 40)
        report.append(f"Files audited: {self.statistics['total_files']}")
        report.append(f"Lines analyzed: {self.statistics['total_lines']}")
        report.append(f"Total issues found: {len(self.issues)}")
        report.append("")
        report.append("Issue Breakdown:")
        report.append(f"  TODOs: {self.statistics['todos']}")
        report.append(f"  Temporary implementations: {self.statistics['temporary_implementations']}")
        report.append(f"  Mock data: {self.statistics['mock_data']}")
        report.append(f"  Placeholders: {self.statistics['placeholders']}")
        report.append(f"  Incomplete APIs: {self.statistics['incomplete_apis']}")
        report.append(f"  Hardcoded values: {self.statistics['hardcoded_values']}")
        report.append(f"  Missing implementations: {self.statistics['missing_implementations']}")
        report.append("")
        
        # Critical Issues
        critical_issues = [i for i in self.issues if i.severity == 'critical']
        if critical_issues:
            report.append("🚨 CRITICAL ISSUES (Must fix immediately):")
            report.append("=" * 40)
            for issue in critical_issues[:30]:  # Limit to first 30
                report.append(f"\n📁 {issue.file_path}:{issue.line_number}")
                report.append(f"  Category: {issue.category}")
                report.append(f"  Owner: {self.agent_owners.get(issue.owner, issue.owner)}")
                report.append(f"  Issue: {issue.description}")
                report.append(f"  Code: {issue.code_snippet[:100]}")
        
        # Group by Owner
        report.append("\n" + "=" * 80)
        report.append("📋 ISSUES BY OWNER:")
        report.append("=" * 40)
        
        owner_issues = {}
        for issue in self.issues:
            if issue.owner not in owner_issues:
                owner_issues[issue.owner] = []
            owner_issues[issue.owner].append(issue)
        
        for owner, issues in sorted(owner_issues.items(), key=lambda x: len(x[1]), reverse=True):
            agent_name = self.agent_owners.get(owner, owner)
            report.append(f"\n{agent_name}: {len(issues)} issues")
            
            # Show top 5 issues for each owner
            for issue in issues[:5]:
                report.append(f"  - {issue.file_path}:{issue.line_number} - {issue.description}")
        
        # Files with Most Issues
        report.append("\n" + "=" * 80)
        report.append("📂 FILES WITH MOST ISSUES:")
        report.append("=" * 40)
        
        file_issues = {}
        for issue in self.issues:
            if issue.file_path not in file_issues:
                file_issues[issue.file_path] = []
            file_issues[issue.file_path].append(issue)
        
        sorted_files = sorted(file_issues.items(), key=lambda x: len(x[1]), reverse=True)
        for file_path, issues in sorted_files[:20]:
            critical_count = sum(1 for i in issues if i.severity == 'critical')
            high_count = sum(1 for i in issues if i.severity == 'high')
            medium_count = sum(1 for i in issues if i.severity == 'medium')
            
            report.append(f"\n{file_path}")
            report.append(f"  Total: {len(issues)} | Critical: {critical_count} | High: {high_count} | Medium: {medium_count}")
        
        # Action Items
        report.append("\n" + "=" * 80)
        report.append("🎯 ACTION ITEMS:")
        report.append("=" * 40)
        
        report.append("\n1. IMMEDIATE (Today):")
        report.append("   - Fix all critical issues")
        report.append("   - Remove all temporary implementations")
        report.append("   - Replace all mock data")
        
        report.append("\n2. HIGH PRIORITY (This Week):")
        report.append("   - Complete all TODOs")
        report.append("   - Fix hardcoded values")
        report.append("   - Implement missing functions")
        
        report.append("\n3. ONGOING:")
        report.append("   - Remove debug prints")
        report.append("   - Add proper error handling")
        report.append("   - Complete API implementations")
        
        # Final Verdict
        report.append("\n" + "=" * 80)
        if critical_issues:
            report.append("❌ INTEGRITY CHECK: FAILED")
            report.append(f"{len(critical_issues)} critical issues must be resolved before production.")
        elif self.statistics['temporary_implementations'] > 0:
            report.append("⚠️  INTEGRITY CHECK: NEEDS WORK")
            report.append("Temporary implementations must be replaced.")
        else:
            report.append("✅ INTEGRITY CHECK: PASSED")
            report.append("No critical integrity issues found.")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_json_report(self, output_file: str) -> None:
        """Save detailed JSON report for tracking"""
        issues_data = []
        for issue in self.issues:
            issues_data.append({
                'file': issue.file_path,
                'line': issue.line_number,
                'category': issue.category,
                'severity': issue.severity,
                'owner': issue.owner,
                'owner_name': self.agent_owners.get(issue.owner, issue.owner),
                'description': issue.description,
                'code': issue.code_snippet
            })
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.statistics,
            'total_issues': len(self.issues),
            'critical_count': sum(1 for i in self.issues if i.severity == 'critical'),
            'high_count': sum(1 for i in self.issues if i.severity == 'high'),
            'medium_count': sum(1 for i in self.issues if i.severity == 'medium'),
            'issues': issues_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    def generate_backlog_items(self) -> List[Dict]:
        """Generate JIRA-style backlog items from issues"""
        backlog = []
        
        # Group issues by file and type for better task organization
        file_issues = {}
        for issue in self.issues:
            key = (issue.file_path, issue.category, issue.owner)
            if key not in file_issues:
                file_issues[key] = []
            file_issues[key].append(issue)
        
        # Create backlog items
        for (file_path, category, owner), issues in file_issues.items():
            backlog_item = {
                'title': f"Fix {category} issues in {Path(file_path).name}",
                'description': f"Address {len(issues)} {category} issues in {file_path}",
                'owner': self.agent_owners.get(owner, owner),
                'priority': issues[0].severity,
                'story_points': min(8, len(issues)),  # Cap at 8 story points
                'acceptance_criteria': [
                    f"Fix issue at line {i.line_number}: {i.description}" for i in issues[:5]
                ],
                'file': file_path,
                'issue_count': len(issues)
            }
            backlog.append(backlog_item)
        
        # Sort by priority and issue count
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        backlog.sort(key=lambda x: (priority_order.get(x['priority'], 3), -x['issue_count']))
        
        return backlog

def main():
    auditor = IntegrityAuditor()
    
    # Audit source directory
    src_dir = '/home/hamster/bot4/src'
    print(f"🔍 Auditing {src_dir} for integrity issues...")
    auditor.audit_directory(src_dir)
    
    # Generate and print report
    report = auditor.generate_report()
    print(report)
    
    # Save JSON report
    json_file = '/home/hamster/bot4/integrity_audit.json'
    auditor.save_json_report(json_file)
    print(f"\n📄 Detailed JSON report saved to: {json_file}")
    
    # Generate backlog items
    backlog = auditor.generate_backlog_items()
    backlog_file = '/home/hamster/bot4/backlog_items.json'
    with open(backlog_file, 'w') as f:
        json.dump(backlog, f, indent=2)
    print(f"📋 Backlog items saved to: {backlog_file}")
    print(f"   Total backlog items: {len(backlog)}")
    
    # Exit code based on critical issues
    critical_count = sum(1 for i in auditor.issues if i.severity == 'critical')
    if critical_count > 0:
        print(f"\n❌ {critical_count} critical issues found. Fix before proceeding!")
        exit(1)
    else:
        print("\n✅ No critical issues found.")
        exit(0)

if __name__ == "__main__":
    main()