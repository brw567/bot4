#!/usr/bin/env python3
"""
Validates that no fake implementations exist in Rust codebase
Task: Quality enforcement for Rust code
Author: Sam
Zero tolerance for fake implementations
"""

import os
import re
import sys
from pathlib import Path

# Patterns that indicate fake implementations
FAKE_PATTERNS = [
    r'todo!\(\)',                    # Rust todo macro
    r'unimplemented!\(\)',          # Rust unimplemented macro  
    r'panic!\("not implemented',    # Panic with not implemented
    r'unreachable!\(\)',            # Unreachable code
    r'// FAKE',                     # Marked as fake
    r'// TODO(?!.*Task-\d+)',       # TODO without task ID
    r'return 0\.0[0-9]',            # Returning constants
    r'price \* 0\.0[0-9]',          # Hardcoded multipliers
    r'rand::random',                # Random values in production
    r'thread::sleep',               # Sleep in production code
    r'println!',                    # Debug prints in production
    r'dbg!',                        # Debug macro in production
    r'mock_',                       # Mock implementations
    r'dummy_',                      # Dummy implementations
    r'fake_',                       # Fake implementations
    r'test_data',                   # Test data in production
    r'hardcoded',                   # Hardcoded values
    r'magic_number',                # Magic numbers
    r'placeholder',                 # Placeholder code
    r'stub_',                       # Stub implementations
]

# Additional Rust-specific checks
RUST_QUALITY_PATTERNS = [
    r'unsafe\s*\{',                 # Unsafe code blocks
    r'#\[allow\(.*\)\]',           # Allowing lints
    r'as\s+\*mut',                  # Raw pointer casts
    r'Box::leak',                   # Memory leaks
    r'std::mem::forget',            # Forgetting values
    r'\.unwrap\(\)',                # Unwrap without error handling
    r'\.expect\("[^"]*"\)',        # Expect without proper message
]

EXCLUDE_DIRS = [
    'target',
    '.git',
    'node_modules',
    'legacy',  # Quarantined Python code
    'tests',   # Test directories can have mocks
    'benches', # Benchmarks
    'examples' # Example code
]

EXCLUDE_FILES = [
    'validate_no_fakes',
    'test.rs',
    'tests.rs',
    'bench.rs',
    'mock.rs',  # Explicitly mock files for testing
]

def check_file(filepath):
    """Check a single Rust file for fake implementations"""
    
    # Skip excluded files
    for exclude in EXCLUDE_FILES:
        if exclude in str(filepath):
            return []
    
    # Only check Rust files
    if not filepath.suffix == '.rs':
        return []
    
    violations = []
    quality_issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Skip comments and test code
                if line.strip().startswith('//'):
                    continue
                if '#[test]' in line or '#[cfg(test)]' in line:
                    continue
                    
                # Check for fake patterns
                for pattern in FAKE_PATTERNS:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append(f"{filepath}:{i}: FAKE: {line.strip()}")
                
                # Check for quality issues
                for pattern in RUST_QUALITY_PATTERNS:
                    if re.search(pattern, line):
                        quality_issues.append(f"{filepath}:{i}: QUALITY: {line.strip()}")
                        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return violations, quality_issues

def validate_rust_structure(root_path):
    """Validate Rust project structure"""
    issues = []
    
    # Check for Cargo.toml
    if not (root_path / 'Cargo.toml').exists():
        if not (root_path / 'rust_core' / 'Cargo.toml').exists():
            issues.append("ERROR: No Cargo.toml found")
    
    # Check for proper module structure
    src_path = root_path / 'src'
    rust_core_path = root_path / 'rust_core'
    
    if not src_path.exists() and not rust_core_path.exists():
        issues.append("ERROR: No src/ or rust_core/ directory found")
    
    return issues

def main():
    root_path = Path('.')
    violations = []
    quality_issues = []
    structure_issues = validate_rust_structure(root_path)
    
    if structure_issues:
        for issue in structure_issues:
            print(f"❌ {issue}")
    
    # Check all Rust files
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if file.endswith('.rs'):
                filepath = Path(root) / file
                file_violations, file_quality = check_file(filepath)
                violations.extend(file_violations)
                quality_issues.extend(file_quality)
    
    # Report results
    if violations:
        print("❌ FAKE IMPLEMENTATIONS FOUND IN RUST!")
        print("=" * 50)
        for v in violations:
            print(f"  {v}")
        print("\n" + "=" * 50)
        print(f"Total violations: {len(violations)}")
        print("\nSam says: REJECTED! No fake implementations allowed in Rust!")
        return 1
    
    if quality_issues:
        print("⚠️  Quality issues found (not blocking):")
        for q in quality_issues[:10]:  # Show first 10
            print(f"  {q}")
        if len(quality_issues) > 10:
            print(f"  ... and {len(quality_issues) - 10} more")
        print("\nConsider addressing these for better code quality")
    
    print("✅ No fake implementations found in Rust code!")
    print("✅ Sam approves: All Rust implementations are real!")
    
    # Additional stats
    rust_files = list(Path('.').rglob('*.rs'))
    rust_files = [f for f in rust_files if 'target' not in str(f)]
    print(f"\n📊 Stats:")
    print(f"  Rust files checked: {len(rust_files)}")
    
    if rust_files:
        total_lines = sum(len(open(f).readlines()) for f in rust_files if f.is_file())
        print(f"  Total lines of Rust: {total_lines}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())