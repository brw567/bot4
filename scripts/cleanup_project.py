#!/usr/bin/env python3
"""
Project Cleanup Script for Bot3
Safely archives unused modules and reorganizes the project structure
"""

import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import ast

class ProjectCleaner:
    def __init__(self, project_root: str = "/home/hamster/bot4"):
        self.project_root = Path(project_root)
        self.archive_dir = self.project_root / "archive" / "legacy_modules"
        self.stats = {
            'files_archived': 0,
            'files_deleted': 0,
            'lines_removed': 0,
            'space_saved_mb': 0
        }
        
    def create_archive_structure(self):
        """Create archive directory structure"""
        print("📁 Creating archive structure...")
        
        dirs = [
            self.archive_dir / "strategies",
            self.archive_dir / "ml",
            self.archive_dir / "utils",
            self.archive_dir / "tests",
            self.archive_dir / "deprecated"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Archive structure created at {self.archive_dir}")
        
    def find_unused_files(self) -> List[Path]:
        """Find potentially unused Python files"""
        print("\n🔍 Searching for unused files...")
        
        unused_patterns = [
            "*mock*.py",
            "*fake*.py",
            "*test_*.py",  # Old test files not in tests/
            "*_old.py",
            "*_backup.py",
            "*_deprecated.py",
            "template_*.py",
            "*_example.py"
        ]
        
        unused_files = []
        src_dir = self.project_root / "src"
        
        for pattern in unused_patterns:
            for file_path in src_dir.rglob(pattern):
                # Don't archive files in tests/ directory
                if "tests" not in str(file_path):
                    unused_files.append(file_path)
                    
        print(f"📊 Found {len(unused_files)} potentially unused files")
        return unused_files
        
    def find_duplicate_functions(self) -> Dict[str, List[str]]:
        """Find duplicate function implementations"""
        print("\n🔍 Searching for duplicate functions...")
        
        function_hashes = {}
        duplicates = {}
        
        for py_file in self.project_root.rglob("*.py"):
            if "archive" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Hash the function body
                        func_str = ast.dump(node)
                        func_hash = hashlib.md5(func_str.encode()).hexdigest()
                        
                        if func_hash in function_hashes:
                            if func_hash not in duplicates:
                                duplicates[func_hash] = [function_hashes[func_hash]]
                            duplicates[func_hash].append(f"{py_file}:{node.name}")
                        else:
                            function_hashes[func_hash] = f"{py_file}:{node.name}"
            except:
                pass
                
        print(f"📊 Found {len(duplicates)} duplicate function implementations")
        return duplicates
        
    def archive_file(self, file_path: Path, reason: str):
        """Archive a single file"""
        relative_path = file_path.relative_to(self.project_root)
        
        # Determine archive location
        if "strategies" in str(relative_path):
            archive_path = self.archive_dir / "strategies" / file_path.name
        elif "ml" in str(relative_path):
            archive_path = self.archive_dir / "ml" / file_path.name
        elif "utils" in str(relative_path):
            archive_path = self.archive_dir / "utils" / file_path.name
        elif "test" in str(relative_path):
            archive_path = self.archive_dir / "tests" / file_path.name
        else:
            archive_path = self.archive_dir / "deprecated" / file_path.name
            
        # Move file
        try:
            shutil.move(str(file_path), str(archive_path))
            self.stats['files_archived'] += 1
            
            # Calculate size
            size_mb = archive_path.stat().st_size / 1024 / 1024
            self.stats['space_saved_mb'] += size_mb
            
            # Count lines
            with open(archive_path, 'r') as f:
                lines = len(f.readlines())
                self.stats['lines_removed'] += lines
                
            print(f"  📦 Archived: {relative_path} -> {archive_path.name} ({reason})")
            return True
        except Exception as e:
            print(f"  ❌ Failed to archive {file_path}: {e}")
            return False
            
    def remove_commented_code(self):
        """Remove commented-out code from all Python files"""
        print("\n🧹 Removing commented-out code...")
        
        removed_lines = 0
        
        for py_file in self.project_root.rglob("*.py"):
            if "archive" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r') as f:
                    lines = f.readlines()
                    
                new_lines = []
                for line in lines:
                    # Keep important comments (docstrings, inline explanations)
                    if line.strip().startswith('#'):
                        # Skip if it's commented-out code
                        if any(keyword in line for keyword in ['import', 'def ', 'class ', 'return', '=', '(', ')']):
                            removed_lines += 1
                            continue
                    new_lines.append(line)
                    
                # Write back if changes were made
                if len(new_lines) < len(lines):
                    with open(py_file, 'w') as f:
                        f.writelines(new_lines)
                        
            except Exception as e:
                print(f"  ⚠️ Error processing {py_file}: {e}")
                
        print(f"✅ Removed {removed_lines} lines of commented code")
        self.stats['lines_removed'] += removed_lines
        
    def optimize_imports(self):
        """Optimize and organize imports in all Python files"""
        print("\n📦 Optimizing imports...")
        
        try:
            import isort
            
            for py_file in self.project_root.rglob("*.py"):
                if "archive" in str(py_file) or "__pycache__" in str(py_file):
                    continue
                    
                isort.file(str(py_file), profile="black", force_single_line=True)
                
            print("✅ All imports optimized")
        except ImportError:
            print("⚠️ isort not installed. Run: pip install isort")
            
    def create_manifest(self):
        """Create archive manifest"""
        manifest_path = self.archive_dir / "MANIFEST.json"
        
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'archived_files': []
        }
        
        for file_path in self.archive_dir.rglob("*.py"):
            relative_path = file_path.relative_to(self.archive_dir)
            manifest['archived_files'].append({
                'path': str(relative_path),
                'size_kb': file_path.stat().st_size / 1024,
                'reason': 'unused/deprecated'
            })
            
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        print(f"\n📄 Manifest created: {manifest_path}")
        
    def generate_report(self):
        """Generate cleanup report"""
        print("\n" + "="*60)
        print("📊 CLEANUP REPORT")
        print("="*60)
        print(f"  Files Archived:     {self.stats['files_archived']}")
        print(f"  Files Deleted:      {self.stats['files_deleted']}")
        print(f"  Lines Removed:      {self.stats['lines_removed']:,}")
        print(f"  Space Saved:        {self.stats['space_saved_mb']:.2f} MB")
        print("="*60)
        
    def run_cleanup(self, dry_run: bool = True):
        """Run the complete cleanup process"""
        print(f"\n{'='*60}")
        print(f"🚀 BOT3 PROJECT CLEANUP")
        print(f"{'='*60}")
        print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        print(f"Project Root: {self.project_root}")
        
        # Step 1: Create archive structure
        if not dry_run:
            self.create_archive_structure()
        
        # Step 2: Find and archive unused files
        unused_files = self.find_unused_files()
        
        if unused_files:
            print(f"\n📦 Files to archive:")
            for file_path in unused_files[:10]:  # Show first 10
                print(f"  - {file_path.relative_to(self.project_root)}")
            if len(unused_files) > 10:
                print(f"  ... and {len(unused_files) - 10} more")
                
            if not dry_run:
                response = input("\n❓ Archive these files? (y/n): ")
                if response.lower() == 'y':
                    for file_path in unused_files:
                        self.archive_file(file_path, "unused pattern")
                        
        # Step 3: Find duplicate functions
        duplicates = self.find_duplicate_functions()
        if duplicates:
            print(f"\n🔄 Duplicate functions found:")
            for i, (hash_val, functions) in enumerate(list(duplicates.items())[:5]):
                print(f"  Duplicate set {i+1}:")
                for func in functions:
                    print(f"    - {func}")
                    
        # Step 4: Remove commented code
        if not dry_run:
            response = input("\n❓ Remove commented-out code? (y/n): ")
            if response.lower() == 'y':
                self.remove_commented_code()
                
        # Step 5: Optimize imports
        if not dry_run:
            response = input("\n❓ Optimize imports? (y/n): ")
            if response.lower() == 'y':
                self.optimize_imports()
                
        # Step 6: Create manifest
        if not dry_run:
            self.create_manifest()
            
        # Step 7: Generate report
        self.generate_report()
        
        print("\n✅ Cleanup complete!")
        
        if dry_run:
            print("\n💡 Run with --live to actually perform cleanup")
            

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up Bot3 project")
    parser.add_argument('--live', action='store_true', 
                       help='Actually perform cleanup (default is dry run)')
    parser.add_argument('--path', default='/home/hamster/bot4',
                       help='Project root path')
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner(args.path)
    cleaner.run_cleanup(dry_run=not args.live)
    

if __name__ == "__main__":
    main()