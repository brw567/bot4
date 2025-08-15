#!/usr/bin/env python3
"""
Script to automatically add architecture headers to source files
Task 3.1.2: Add proper headers to all source files
Author: Alex (Team Lead)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

def determine_component_info(file_path: Path) -> Tuple[str, str, str]:
    """Determine component type, author, and task based on file path."""
    
    path_str = str(file_path)
    name = file_path.stem
    
    # Component mapping based on directory
    if 'strategies' in path_str:
        component = "Trading Strategy"
        author = "Sam (Quant Developer)"
        task = "EPIC-3: Trading Logic"
    elif 'core' in path_str:
        if 'risk' in name.lower() or 'stop_loss' in name.lower() or 'emergency' in name.lower():
            component = "Risk Management"
            author = "Quinn (Risk Manager)"
            task = "EPIC-3: Risk Controls"
        elif 'ml' in name.lower() or 'model' in name.lower() or 'feature' in name.lower():
            component = "ML Infrastructure"
            author = "Morgan (ML Engineer)"
            task = "EPIC-5: ML Enhancement"
        elif 'exchange' in name.lower() or 'websocket' in name.lower() or 'order' in name.lower():
            component = "Exchange Integration"
            author = "Casey (Exchange Specialist)"
            task = "EPIC-3: Trading Infrastructure"
        elif 'monitor' in name.lower() or 'health' in name.lower() or 'performance' in name.lower():
            component = "Monitoring"
            author = "Jordan (DevOps)"
            task = "EPIC-6: Monitoring"
        elif 'backtest' in name.lower() or 'analytics' in name.lower():
            component = "Analytics"
            author = "Avery (Data Engineer)"
            task = "EPIC-4: Analytics"
        else:
            component = "Core Infrastructure"
            author = "Alex (Team Lead)"
            task = "EPIC-2: System Architecture"
    elif 'api' in path_str:
        component = "REST API"
        author = "Casey (Exchange Specialist)"
        task = "EPIC-3: API Integration"
    elif 'frontend' in path_str:
        component = "Frontend UI"
        author = "Riley (Frontend Developer)"
        task = "EPIC-3: User Interface"
    elif 'database' in path_str or 'alembic' in path_str:
        component = "Database"
        author = "Avery (Data Engineer)"
        task = "EPIC-9: Data Integrity"
    elif 'tests' in path_str:
        component = "Testing"
        author = "Riley (QA Engineer)"
        task = "EPIC-10: Testing Suite"
    elif 'utils' in path_str:
        component = "Utilities"
        author = "Alex (Team Lead)"
        task = "EPIC-2: System Architecture"
    else:
        component = "System Component"
        author = "Alex (Team Lead)"
        task = "EPIC-2: System Architecture"
    
    return component, author, task

def generate_python_header(file_path: Path) -> str:
    """Generate appropriate header for Python file."""
    
    component, author, task = determine_component_info(file_path)
    name = file_path.stem.replace('_', ' ').title()
    
    # Map to architecture sections
    arch_sections = {
        "Trading Strategy": "Trading Strategies",
        "Risk Management": "Risk Management",
        "ML Infrastructure": "ML/AI Components",
        "Exchange Integration": "Exchange Integration",
        "Monitoring": "Monitoring & Alerts",
        "Analytics": "Analytics System",
        "Core Infrastructure": "Core Components",
        "REST API": "API Layer",
        "Frontend UI": "User Interface",
        "Database": "Database Layer",
        "Testing": "Testing Infrastructure",
        "Utilities": "Utility Components"
    }
    
    arch_ref = arch_sections.get(component, "System Architecture")
    
    header = f'''"""
Component: {component} - {name}
Task: {task}
Author: {author}
Created: 2025-01-10
Modified: 2025-01-10

Description:
{name} implementation for the Bot3 trading platform.

Architecture Reference:
See ARCHITECTURE.md > {arch_ref} > {name}

Dependencies:
- See imports below for dependencies

Performance Characteristics:
- Latency: <100ms requirement (Jordan)
- Reliability: 99.9% uptime target

Tests:
- Unit tests: tests/test_{file_path.stem}.py
- Integration tests: tests/integration/

Enhancement Opportunities:
- See TASK_LIST.md for planned improvements
"""

'''
    return header

def generate_typescript_header(file_path: Path) -> str:
    """Generate appropriate header for TypeScript/React file."""
    
    component, author, task = determine_component_info(file_path)
    name = file_path.stem.replace('_', ' ').title()
    
    header = f'''/**
 * Component: {component} - {name}
 * Task: {task}
 * Author: {author}
 * Created: 2025-01-10
 * Modified: 2025-01-10
 * 
 * Description:
 * {name} component for the Bot3 trading platform frontend.
 * 
 * Architecture Reference:
 * See ARCHITECTURE.md > User Interface > {name}
 * 
 * Tests:
 * - Component tests: {file_path.stem}.test.tsx
 * - Integration tests: tests/integration/
 */

'''
    return header

def add_header_to_file(file_path: Path, dry_run: bool = False) -> bool:
    """Add header to file if missing."""
    
    try:
        content = file_path.read_text()
        
        # Check if header already exists
        if 'Component:' in content[:500] and 'Architecture Reference:' in content[:1000]:
            print(f"  ✓ {file_path.name} - already has header")
            return False
        
        # Generate appropriate header
        if file_path.suffix == '.py':
            header = generate_python_header(file_path)
        elif file_path.suffix in ['.ts', '.tsx', '.js', '.jsx']:
            header = generate_typescript_header(file_path)
        else:
            print(f"  ⚠ {file_path.name} - unsupported file type")
            return False
        
        # Handle shebang line for scripts
        if content.startswith('#!/'):
            lines = content.split('\n', 1)
            new_content = lines[0] + '\n' + header + (lines[1] if len(lines) > 1 else '')
        else:
            new_content = header + content
        
        if not dry_run:
            file_path.write_text(new_content)
            print(f"  ✅ {file_path.name} - header added")
        else:
            print(f"  🔍 {file_path.name} - would add header (dry run)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ {file_path.name} - error: {e}")
        return False

def main():
    """Main function to add headers to all source files."""
    
    # Parse arguments
    dry_run = '--dry-run' in sys.argv
    specific_dir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != '--dry-run' else None
    
    print("="*60)
    print("ADDING ARCHITECTURE HEADERS TO SOURCE FILES")
    print("="*60)
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
    
    # Determine directories to process
    if specific_dir:
        dirs = [Path(specific_dir)]
    else:
        dirs = [
            Path('src'),
            Path('tests'),
            Path('frontend/src'),
            Path('scripts'),
            Path('alembic')
        ]
    
    total_files = 0
    files_updated = 0
    
    for directory in dirs:
        if not directory.exists():
            continue
        
        print(f"\n📁 Processing {directory}...")
        
        # Find all Python and TypeScript files
        patterns = ['*.py', '*.ts', '*.tsx']
        files = []
        for pattern in patterns:
            files.extend(directory.rglob(pattern))
        
        # Exclude some directories
        files = [f for f in files if not any(part in str(f) for part in [
            '__pycache__', 'node_modules', 'venv', '.git', 'build', 'dist'
        ])]
        
        for file_path in sorted(files):
            total_files += 1
            if add_header_to_file(file_path, dry_run):
                files_updated += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files processed: {total_files}")
    print(f"Files updated: {files_updated}")
    print(f"Files already had headers: {total_files - files_updated}")
    
    if dry_run:
        print("\n💡 Run without --dry-run to actually add headers")
    else:
        print("\n✅ Headers added successfully!")
        print("Next: Run scripts/verify_architecture_links.py to verify")

if __name__ == "__main__":
    main()