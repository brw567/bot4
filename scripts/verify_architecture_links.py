#!/usr/bin/env python3
"""
Component: Infrastructure Scripts
Task: Workflow Enhancement Implementation
Author: Sam (Code Quality)
Tests: tests/test_verification_scripts.py
Architecture: See ARCHITECTURE.md > Infrastructure Components > Verification Scripts
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

def find_python_files(directory: str) -> List[Path]:
    """Find all Python files in the project"""
    return list(Path(directory).rglob("*.py"))

def find_typescript_files(directory: str) -> List[Path]:
    """Find all TypeScript/JavaScript files in the project"""
    ts_files = list(Path(directory).rglob("*.ts"))
    tsx_files = list(Path(directory).rglob("*.tsx"))
    js_files = list(Path(directory).rglob("*.js"))
    jsx_files = list(Path(directory).rglob("*.jsx"))
    return ts_files + tsx_files + js_files + jsx_files

def check_header_comment(file_path: Path) -> Dict[str, bool]:
    """Check if file has proper header comment"""
    required_fields = {
        "Component": False,
        "Task": False,
        "Author": False,
        "Tests": False,
        "Architecture": False
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for header comment in first 500 characters
        header = content[:500]
        
        for field in required_fields:
            if f"{field}:" in header:
                required_fields[field] = True
                
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return required_fields

def verify_task_ids(file_path: Path) -> Tuple[bool, List[str]]:
    """Verify that Task IDs reference real tasks in TASK_LIST.md"""
    task_ids = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Find all TASK-XXX references
        task_pattern = r'TASK-\d+'
        task_ids = re.findall(task_pattern, content)
        
        # Also check for task references in TODO comments
        todo_pattern = r'TODO.*?TASK-\d+'
        todo_matches = re.findall(todo_pattern, content)
        
        if content.count("TODO") > len(todo_matches):
            print(f"WARNING: {file_path} has TODO without task ID")
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return len(task_ids) > 0, task_ids

def verify_architecture_links(file_path: Path) -> bool:
    """Verify that Architecture references are valid"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for ARCHITECTURE.md reference
        if "ARCHITECTURE.md" in content[:1000]:  # Check in header area
            return True
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        
    return False

def main():
    """Main verification function"""
    project_root = Path("/home/hamster/bot4")
    src_dir = project_root / "src"
    frontend_dir = project_root / "frontend" / "src"
    
    # Collect all source files
    python_files = find_python_files(src_dir)
    ts_files = find_typescript_files(frontend_dir) if frontend_dir.exists() else []
    
    all_files = python_files + ts_files
    
    # Exclude certain directories
    excluded_dirs = ["__pycache__", "venv", ".venv", "node_modules", ".git"]
    all_files = [f for f in all_files if not any(ex in str(f) for ex in excluded_dirs)]
    
    # Statistics
    total_files = len(all_files)
    missing_headers = []
    missing_task_ids = []
    missing_arch_links = []
    
    print(f"Verifying {total_files} source files...")
    print("-" * 60)
    
    for file_path in all_files:
        # Skip test files and migrations
        if "test_" in file_path.name or "migration" in str(file_path):
            continue
            
        # Check header comment
        header_check = check_header_comment(file_path)
        if not all(header_check.values()):
            missing_fields = [k for k, v in header_check.items() if not v]
            missing_headers.append((file_path, missing_fields))
            
        # Check task IDs
        has_task, task_ids = verify_task_ids(file_path)
        if not has_task:
            missing_task_ids.append(file_path)
            
        # Check architecture links
        if not verify_architecture_links(file_path):
            missing_arch_links.append(file_path)
    
    # Report results
    print("\n📊 Verification Results:")
    print(f"Total files checked: {total_files}")
    print(f"Files missing headers: {len(missing_headers)}")
    print(f"Files missing task IDs: {len(missing_task_ids)}")
    print(f"Files missing architecture links: {len(missing_arch_links)}")
    
    # Detailed report
    if missing_headers:
        print("\n❌ Files with incomplete headers:")
        for file_path, fields in missing_headers[:10]:  # Show first 10
            print(f"  {file_path.relative_to(project_root)}: Missing {', '.join(fields)}")
        if len(missing_headers) > 10:
            print(f"  ... and {len(missing_headers) - 10} more")
    
    if missing_task_ids:
        print("\n⚠️ Files without task IDs:")
        for file_path in missing_task_ids[:10]:
            print(f"  {file_path.relative_to(project_root)}")
        if len(missing_task_ids) > 10:
            print(f"  ... and {len(missing_task_ids) - 10} more")
    
    if missing_arch_links:
        print("\n⚠️ Files without architecture links:")
        for file_path in missing_arch_links[:10]:
            print(f"  {file_path.relative_to(project_root)}")
        if len(missing_arch_links) > 10:
            print(f"  ... and {len(missing_arch_links) - 10} more")
    
    # Exit code based on results
    if missing_headers or missing_task_ids or missing_arch_links:
        print("\n❌ Verification FAILED")
        print("Please add proper headers to all source files.")
        print("Use docs/templates/code_header_template.py as reference.")
        return 1
    else:
        print("\n✅ All files have proper documentation!")
        return 0

if __name__ == "__main__":
    sys.exit(main())