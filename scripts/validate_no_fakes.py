#!/usr/bin/env python3
"""
Validate No Fake Implementations Script
Sam's enforcement tool - zero tolerance for fake code
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Fake implementation patterns to detect
FAKE_PATTERNS = [
    (r'price\s*\*\s*0\.0[0-9]+', 'Fake calculation using price percentage'),
    (r'return\s+0\s*#\s*TODO', 'Placeholder return statement'),
    (r'pass\s*#\s*implement\s*later', 'Unimplemented function'),
    (r'random\.choice', 'Random selection instead of logic'),
    (r'random\.randint', 'Random number instead of calculation'),
    (r'return\s+\"?fake\"?', 'Literal fake return'),
    (r'#\s*FAKE', 'Fake marker comment'),
    (r'mock_[a-z_]+\s*=', 'Mock data in production code'),
    (r'sleep\(\d+\)\s*#\s*simulate', 'Fake delay simulation'),
    (r'raise\s+NotImplementedError', 'Unimplemented functionality'),
]

# Specific fake implementations we've seen
KNOWN_FAKES = [
    ('atr = price * 0.02', 'Fake ATR calculation'),
    ('atr = market_data.close * 0.02', 'Fake ATR from RC5'),
    ('return random.choice(["up", "down"])', 'Fake ML prediction'),
    ('confidence = 0.5', 'Hardcoded confidence'),
    ('sharpe_ratio = 2.0  # good enough', 'Fake Sharpe ratio'),
]

# Directories to check
CHECK_DIRS = ['src', 'strategies', 'indicators', 'ml']

# Files to exclude
EXCLUDE_PATTERNS = [
    '*.pyc',
    '__pycache__',
    '*.log',
    'test_*.py',
    '*_test.py',
]

class FakeDetector:
    def __init__(self):
        self.violations = []
        self.files_checked = 0
        self.lines_checked = 0
        
    def check_file(self, filepath: Path) -> List[Tuple[int, str, str]]:
        """Check a single file for fake implementations"""
        violations = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                self.files_checked += 1
                self.lines_checked += len(lines)
                
                for line_num, line in enumerate(lines, 1):
                    # Check regex patterns
                    for pattern, description in FAKE_PATTERNS:
                        if re.search(pattern, line, re.IGNORECASE):
                            violations.append((line_num, line.strip(), description))
                    
                    # Check known fakes
                    for fake_code, description in KNOWN_FAKES:
                        if fake_code in line:
                            violations.append((line_num, line.strip(), f"Known fake: {description}"))
                            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
        return violations
    
    def should_check_file(self, filepath: Path) -> bool:
        """Determine if file should be checked"""
        # Only check Python files
        if not filepath.suffix == '.py':
            return False
            
        # Skip excluded patterns
        for pattern in EXCLUDE_PATTERNS:
            if filepath.match(pattern):
                return False
                
        return True
    
    def check_directory(self, directory: str) -> None:
        """Recursively check directory for fake implementations"""
        path = Path(directory)
        
        if not path.exists():
            return
            
        for filepath in path.rglob('*.py'):
            if self.should_check_file(filepath):
                violations = self.check_file(filepath)
                if violations:
                    self.violations.append((str(filepath), violations))
    
    def print_report(self) -> None:
        """Print detection report"""
        print("=" * 80)
        print("FAKE IMPLEMENTATION DETECTION REPORT")
        print("=" * 80)
        print(f"Files checked: {self.files_checked}")
        print(f"Lines checked: {self.lines_checked}")
        print()
        
        if not self.violations:
            print("✅ NO FAKE IMPLEMENTATIONS DETECTED!")
            print("Sam approves: All implementations are real.")
        else:
            print("❌ FAKE IMPLEMENTATIONS FOUND!")
            print()
            
            for filepath, file_violations in self.violations:
                print(f"\n📁 {filepath}")
                print("-" * 40)
                for line_num, line, description in file_violations:
                    print(f"  Line {line_num}: {description}")
                    print(f"    > {line}")
            
            print("\n" + "=" * 80)
            print("❌ SAM'S VERDICT: REJECTED!")
            print("Fix all fake implementations before proceeding.")
            print("=" * 80)
    
    def run(self) -> bool:
        """Run validation and return True if no fakes found"""
        for directory in CHECK_DIRS:
            self.check_directory(directory)
        
        self.print_report()
        return len(self.violations) == 0

def main():
    """Main execution"""
    detector = FakeDetector()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Check specific files/directories
        for path in sys.argv[1:]:
            if os.path.isdir(path):
                detector.check_directory(path)
            elif os.path.isfile(path):
                if detector.should_check_file(Path(path)):
                    violations = detector.check_file(Path(path))
                    if violations:
                        detector.violations.append((path, violations))
    else:
        # Check default directories
        success = detector.run()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()