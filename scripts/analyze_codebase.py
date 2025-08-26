#!/usr/bin/env python3
"""
Complete Codebase Analysis Tool
Extracts all functions, methods, structs, traits, and dependencies
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

class CodebaseAnalyzer:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.components = defaultdict(list)
        self.dependencies = defaultdict(set)
        self.trait_impls = defaultdict(list)
        self.data_flows = []
        
    def analyze(self):
        """Analyze entire codebase"""
        rust_files = list(self.root_path.rglob("*.rs"))
        print(f"Analyzing {len(rust_files)} Rust files...")
        
        for file_path in rust_files:
            if "target" in str(file_path) or ".git" in str(file_path):
                continue
            self.analyze_file(file_path)
            
        return self.generate_report()
    
    def analyze_file(self, file_path: Path):
        """Analyze a single Rust file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                rel_path = str(file_path.relative_to(self.root_path))
                
                # Extract components
                self.extract_structs(content, rel_path)
                self.extract_enums(content, rel_path)
                self.extract_traits(content, rel_path)
                self.extract_functions(content, rel_path)
                self.extract_impls(content, rel_path)
                self.extract_dependencies(content, rel_path)
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def extract_structs(self, content: str, file_path: str):
        """Extract all struct definitions"""
        # Match pub struct and struct
        pattern = r'(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?\s*(?:\{[^}]*\}|\([^)]*\)|;)'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            struct_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            # Extract fields if it's a regular struct
            fields = []
            if '{' in match.group(0):
                field_pattern = r'(?:pub\s+)?(\w+):\s+([^,}]+)'
                field_matches = re.finditer(field_pattern, match.group(0))
                fields = [(fm.group(1), fm.group(2).strip()) for fm in field_matches]
            
            self.components['structs'].append({
                'name': struct_name,
                'file': file_path,
                'line': line_num,
                'fields': fields,
                'visibility': 'pub' if match.group(0).startswith('pub') else 'private'
            })
    
    def extract_enums(self, content: str, file_path: str):
        """Extract all enum definitions"""
        pattern = r'(?:pub\s+)?enum\s+(\w+)(?:<[^>]+>)?\s*\{([^}]*)\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            enum_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            # Extract variants
            variants = []
            variant_pattern = r'(\w+)(?:\([^)]*\)|\{[^}]*\})?'
            variant_matches = re.finditer(variant_pattern, match.group(2))
            variants = [vm.group(1) for vm in variant_matches if vm.group(1)]
            
            self.components['enums'].append({
                'name': enum_name,
                'file': file_path,
                'line': line_num,
                'variants': variants,
                'visibility': 'pub' if match.group(0).startswith('pub') else 'private'
            })
    
    def extract_traits(self, content: str, file_path: str):
        """Extract all trait definitions"""
        pattern = r'(?:pub\s+)?trait\s+(\w+)(?:<[^>]+>)?\s*(?::\s*[^{]+)?\s*\{([^}]*)\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            trait_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            
            # Extract trait methods
            methods = []
            method_pattern = r'fn\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^;{]+)?'
            method_matches = re.finditer(method_pattern, match.group(2))
            methods = [mm.group(1) for mm in method_matches]
            
            self.components['traits'].append({
                'name': trait_name,
                'file': file_path,
                'line': line_num,
                'methods': methods,
                'visibility': 'pub' if match.group(0).startswith('pub') else 'private'
            })
    
    def extract_functions(self, content: str, file_path: str):
        """Extract all function definitions"""
        # Match functions outside of impl blocks
        pattern = r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*([^{]+))?\s*\{'
        
        # Split content into lines to check for impl context
        lines = content.split('\n')
        impl_depth = 0
        
        for i, line in enumerate(lines):
            # Track impl blocks
            if 'impl' in line and '{' in line:
                impl_depth += 1
            if impl_depth > 0 and '}' in line:
                impl_depth -= 1
                
            # Only match functions outside impl blocks
            if impl_depth == 0:
                match = re.match(pattern, line)
                if match:
                    func_name = match.group(1)
                    return_type = match.group(2).strip() if match.group(2) else "()"
                    
                    self.components['functions'].append({
                        'name': func_name,
                        'file': file_path,
                        'line': i + 1,
                        'return_type': return_type,
                        'is_async': 'async' in line,
                        'visibility': 'pub' if line.strip().startswith('pub') else 'private'
                    })
    
    def extract_impls(self, content: str, file_path: str):
        """Extract all impl blocks and their methods"""
        # Match impl blocks
        impl_pattern = r'impl(?:<[^>]+>)?\s+(?:(\w+)\s+for\s+)?(\w+)(?:<[^>]+>)?\s*\{'
        matches = list(re.finditer(impl_pattern, content))
        
        for match in matches:
            trait_name = match.group(1) if match.group(1) else None
            struct_name = match.group(2)
            impl_start = match.end()
            
            # Find the closing brace
            brace_count = 1
            impl_end = impl_start
            while brace_count > 0 and impl_end < len(content):
                if content[impl_end] == '{':
                    brace_count += 1
                elif content[impl_end] == '}':
                    brace_count -= 1
                impl_end += 1
            
            impl_content = content[impl_start:impl_end-1]
            
            # Extract methods from impl block
            method_pattern = r'(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*([^{]+))?'
            method_matches = re.finditer(method_pattern, impl_content)
            
            methods = []
            for mm in method_matches:
                method_name = mm.group(1)
                return_type = mm.group(2).strip() if mm.group(2) else "()"
                line_num = content[:match.start() + mm.start()].count('\n') + 1
                
                methods.append({
                    'name': method_name,
                    'line': line_num,
                    'return_type': return_type,
                    'is_async': 'async' in mm.group(0)
                })
            
            impl_data = {
                'struct': struct_name,
                'trait': trait_name,
                'file': file_path,
                'line': content[:match.start()].count('\n') + 1,
                'methods': methods
            }
            
            if trait_name:
                self.trait_impls[struct_name].append(impl_data)
            else:
                self.components['impls'].append(impl_data)
    
    def extract_dependencies(self, content: str, file_path: str):
        """Extract use statements and dependencies"""
        # Extract use statements
        use_pattern = r'^use\s+([^;]+);'
        matches = re.finditer(use_pattern, content, re.MULTILINE)
        
        for match in matches:
            dep = match.group(1).strip()
            self.dependencies[file_path].add(dep)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive report"""
        return {
            'summary': {
                'total_files': len(set(
                    item['file'] for category in self.components.values() 
                    for item in category
                )),
                'total_structs': len(self.components['structs']),
                'total_enums': len(self.components['enums']),
                'total_traits': len(self.components['traits']),
                'total_functions': len(self.components['functions']),
                'total_impls': len(self.components['impls']),
                'total_trait_impls': sum(len(v) for v in self.trait_impls.values()),
            },
            'components': dict(self.components),
            'trait_implementations': dict(self.trait_impls),
            'dependencies': {k: list(v) for k, v in self.dependencies.items()},
        }

def main():
    analyzer = CodebaseAnalyzer("/home/hamster/bot4/rust_core")
    report = analyzer.analyze()
    
    # Save detailed report
    with open("/home/hamster/bot4/CODEBASE_ANALYSIS.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n=== CODEBASE ANALYSIS COMPLETE ===")
    print(f"Total Structs: {report['summary']['total_structs']}")
    print(f"Total Enums: {report['summary']['total_enums']}")
    print(f"Total Traits: {report['summary']['total_traits']}")
    print(f"Total Functions: {report['summary']['total_functions']}")
    print(f"Total Impl Blocks: {report['summary']['total_impls']}")
    print(f"Total Trait Impls: {report['summary']['total_trait_impls']}")
    print("\nDetailed report saved to CODEBASE_ANALYSIS.json")

if __name__ == "__main__":
    main()