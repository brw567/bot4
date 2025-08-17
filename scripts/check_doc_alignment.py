#!/usr/bin/env python3
"""
Doc Alignment Checker - Ensures all project documents are synchronized
Required by Sophia's review - CRITICAL for CI gates
Owner: Alex
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

class DocAlignmentChecker:
    def __init__(self):
        self.base_path = Path("/home/hamster/bot4")
        self.errors = []
        self.warnings = []
        
    def extract_phases(self, file_path: Path) -> Dict[str, Dict]:
        """Extract phase information from a document."""
        phases = {}
        
        if not file_path.exists():
            self.errors.append(f"File not found: {file_path}")
            return phases
            
        content = file_path.read_text()
        
        # Pattern for phases (handles Phase X, Phase X.Y format)
        phase_pattern = r'### Phase (\d+(?:\.\d+)?):?\s+([^-\n]+?)(?:\s*[-–]\s*(.+?))?(?:\n|$)'
        
        for match in re.finditer(phase_pattern, content, re.MULTILINE):
            phase_num = match.group(1)
            phase_name = match.group(2).strip()
            status = match.group(3).strip() if match.group(3) else ""
            
            phases[phase_num] = {
                'name': phase_name,
                'status': status,
                'line': content[:match.start()].count('\n') + 1
            }
            
        return phases
    
    def extract_tasks(self, file_path: Path) -> Set[str]:
        """Extract task IDs from a document."""
        tasks = set()
        
        if not file_path.exists():
            return tasks
            
        content = file_path.read_text()
        
        # Pattern for task IDs (e.g., Task 1.2.3, Task-1.2.3)
        task_pattern = r'Task[\s-](\d+(?:\.\d+)*)'
        
        for match in re.finditer(task_pattern, content, re.IGNORECASE):
            tasks.add(match.group(1))
            
        return tasks
    
    def check_phase_alignment(self) -> bool:
        """Check if phases are aligned across all documents."""
        docs = {
            'master': self.base_path / 'PROJECT_MANAGEMENT_MASTER.md',
            'readme': self.base_path / 'README.md',
            'architecture': self.base_path / 'docs' / 'LLM_OPTIMIZED_ARCHITECTURE.md',
            'tasks': self.base_path / 'docs' / 'LLM_TASK_SPECIFICATIONS.md'
        }
        
        all_phases = {}
        for doc_name, doc_path in docs.items():
            all_phases[doc_name] = self.extract_phases(doc_path)
        
        # Master is source of truth
        master_phases = all_phases.get('master', {})
        
        if not master_phases:
            self.errors.append("No phases found in PROJECT_MANAGEMENT_MASTER.md")
            return False
        
        # Check critical Phase 3.5 exists
        if '3.5' not in master_phases:
            self.warnings.append("Phase 3.5 (Emotion-Free Trading) missing from master!")
        
        # Compare other docs to master
        for doc_name, phases in all_phases.items():
            if doc_name == 'master':
                continue
                
            # Check for missing phases
            for phase_num, phase_info in master_phases.items():
                if phase_num not in phases:
                    self.errors.append(
                        f"{doc_name}: Missing Phase {phase_num} ({phase_info['name']})"
                    )
                elif phases[phase_num]['name'] != phase_info['name']:
                    self.warnings.append(
                        f"{doc_name}: Phase {phase_num} name mismatch: "
                        f"'{phases[phase_num]['name']}' vs '{phase_info['name']}'"
                    )
            
            # Check for extra phases
            for phase_num in phases:
                if phase_num not in master_phases:
                    self.warnings.append(
                        f"{doc_name}: Extra Phase {phase_num} not in master"
                    )
        
        return len(self.errors) == 0
    
    def check_performance_targets(self) -> bool:
        """Verify performance targets are consistent."""
        targets = {
            'decision_latency': r'[≤<]\s*1\s*[μµ]s',
            'risk_latency': r'[≤<]\s*10\s*[μµ]s',
            'order_latency': r'[≤<]\s*100\s*[μµ]s',
        }
        
        docs = [
            self.base_path / 'PROJECT_MANAGEMENT_MASTER.md',
            self.base_path / 'README.md',
        ]
        
        for doc_path in docs:
            if not doc_path.exists():
                continue
                
            content = doc_path.read_text()
            for target_name, pattern in targets.items():
                if not re.search(pattern, content, re.IGNORECASE):
                    self.warnings.append(
                        f"{doc_path.name}: Missing or incorrect {target_name} target"
                    )
        
        return True
    
    def check_critical_requirements(self) -> bool:
        """Ensure critical requirements are documented."""
        critical = [
            'Circuit Breaker',
            'Risk Management',
            'Stop[- ]Loss',
            'Position.*Limit',
            'Drawdown',
            'Emotion[- ]Free',
        ]
        
        master = self.base_path / 'PROJECT_MANAGEMENT_MASTER.md'
        if not master.exists():
            self.errors.append("PROJECT_MANAGEMENT_MASTER.md not found!")
            return False
            
        content = master.read_text()
        
        for requirement in critical:
            if not re.search(requirement, content, re.IGNORECASE):
                self.errors.append(f"Critical requirement missing: {requirement}")
        
        return len(self.errors) == 0
    
    def run(self) -> int:
        """Run all alignment checks."""
        print("🔍 Bot4 Document Alignment Checker")
        print("=" * 50)
        
        # Run checks
        phase_ok = self.check_phase_alignment()
        perf_ok = self.check_performance_targets()
        req_ok = self.check_critical_requirements()
        
        # Report results
        if self.errors:
            print("\n❌ ERRORS (must fix):")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS (should fix):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            print("\n✅ All documents are aligned!")
        
        print("\n" + "=" * 50)
        print(f"Summary: {len(self.errors)} errors, {len(self.warnings)} warnings")
        
        # Return non-zero exit code if errors
        return 1 if self.errors else 0

if __name__ == "__main__":
    checker = DocAlignmentChecker()
    sys.exit(checker.run())