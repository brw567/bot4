#!/usr/bin/env python3
"""
Link all functionality to ARCHITECTURE.md sections.
Task 3.1.3: Create bidirectional links between code and architecture documentation.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureSection:
    """Represents a section in ARCHITECTURE.md"""
    title: str
    level: int
    line_number: int
    file_references: List[str] = field(default_factory=list)
    related_components: List[str] = field(default_factory=list)
    
    
@dataclass
class CodeComponent:
    """Represents a code component"""
    file_path: str
    class_names: List[str] = field(default_factory=list)
    function_names: List[str] = field(default_factory=list)
    architecture_refs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)


class ArchitectureLinkingSystem:
    """System to link code functionality to architecture documentation."""
    
    def __init__(self):
        self.architecture_sections: Dict[str, ArchitectureSection] = {}
        self.code_components: Dict[str, CodeComponent] = {}
        self.section_to_files: Dict[str, List[str]] = defaultdict(list)
        self.file_to_sections: Dict[str, List[str]] = defaultdict(list)
        
    def parse_architecture(self, arch_file: Path) -> None:
        """Parse ARCHITECTURE.md and extract sections."""
        logger.info(f"Parsing {arch_file}...")
        
        content = arch_file.read_text()
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Match section headers
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Extract file references from the title if present
                file_refs = re.findall(r'`([^`]+\.py)`', title)
                
                section = ArchitectureSection(
                    title=title,
                    level=level,
                    line_number=i,
                    file_references=file_refs
                )
                
                # Use clean title as key
                clean_title = re.sub(r'`[^`]+`', '', title).strip()
                clean_title = re.sub(r'\([^)]+\)', '', clean_title).strip()
                self.architecture_sections[clean_title] = section
                
                # Map file references to sections
                for file_ref in file_refs:
                    self.section_to_files[clean_title].append(file_ref)
                    self.file_to_sections[file_ref].append(clean_title)
        
        logger.info(f"Found {len(self.architecture_sections)} architecture sections")
    
    def scan_codebase(self, src_path: Path) -> None:
        """Scan codebase and extract component information."""
        logger.info(f"Scanning codebase at {src_path}...")
        
        for py_file in src_path.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue
                
            relative_path = str(py_file.relative_to(src_path.parent))
            component = CodeComponent(file_path=relative_path)
            
            try:
                content = py_file.read_text()
                
                # Extract classes
                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                component.class_names = classes
                
                # Extract functions
                functions = re.findall(r'^(?:async\s+)?def\s+(\w+)', content, re.MULTILINE)
                component.function_names = functions
                
                # Look for architecture references in comments
                arch_refs = re.findall(r'Architecture Reference:.*?([^\n]+)', content)
                component.architecture_refs = arch_refs
                
                # Extract imports to find dependencies
                imports = re.findall(r'^(?:from|import)\s+([^\s]+)', content, re.MULTILINE)
                component.dependencies = imports
                
                self.code_components[relative_path] = component
                
            except Exception as e:
                logger.error(f"Error processing {py_file}: {e}")
        
        logger.info(f"Scanned {len(self.code_components)} code files")
    
    def create_mapping(self) -> Dict[str, Any]:
        """Create comprehensive mapping between architecture and code."""
        mapping = {
            'architecture_to_code': {},
            'code_to_architecture': {},
            'orphaned_components': [],
            'undocumented_sections': []
        }
        
        # Map architecture sections to code components
        for section_title, section in self.architecture_sections.items():
            related_files = []
            
            # Direct file references in section title
            related_files.extend(section.file_references)
            
            # Find files based on naming patterns
            for file_path, component in self.code_components.items():
                # Check if section title matches file or class names
                file_name = Path(file_path).stem
                
                # Various matching strategies
                if any([
                    file_name.lower() in section_title.lower(),
                    section_title.lower() in file_name.lower(),
                    any(cls.lower() in section_title.lower() for cls in component.class_names),
                    any(section_title.lower() in cls.lower() for cls in component.class_names)
                ]):
                    if file_path not in related_files:
                        related_files.append(file_path)
            
            mapping['architecture_to_code'][section_title] = {
                'line_number': section.line_number,
                'level': section.level,
                'related_files': related_files
            }
            
            if not related_files:
                mapping['undocumented_sections'].append(section_title)
        
        # Map code components to architecture sections
        for file_path, component in self.code_components.items():
            related_sections = []
            
            # Check direct references
            related_sections.extend(self.file_to_sections.get(file_path, []))
            
            # Find sections based on content matching
            file_name = Path(file_path).stem
            for section_title in self.architecture_sections.keys():
                if any([
                    file_name.lower() in section_title.lower(),
                    section_title.lower() in file_name.lower(),
                    any(cls.lower() in section_title.lower() for cls in component.class_names),
                    any(section_title.lower() in cls.lower() for cls in component.class_names)
                ]):
                    if section_title not in related_sections:
                        related_sections.append(section_title)
            
            mapping['code_to_architecture'][file_path] = {
                'classes': component.class_names,
                'functions': component.function_names[:5],  # First 5 functions
                'related_sections': related_sections
            }
            
            if not related_sections:
                mapping['orphaned_components'].append(file_path)
        
        return mapping
    
    def update_architecture_file(self, arch_file: Path, mapping: Dict) -> None:
        """Update ARCHITECTURE.md with code references."""
        logger.info("Updating ARCHITECTURE.md with code references...")
        
        content = arch_file.read_text()
        lines = content.split('\n')
        updated_lines = []
        
        for i, line in enumerate(lines):
            updated_lines.append(line)
            
            # Check if this is a section header
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if header_match:
                title = header_match.group(2).strip()
                clean_title = re.sub(r'`[^`]+`', '', title).strip()
                clean_title = re.sub(r'\([^)]+\)', '', clean_title).strip()
                
                # Get related files for this section
                section_data = mapping['architecture_to_code'].get(clean_title, {})
                related_files = section_data.get('related_files', [])
                
                if related_files and i + 1 < len(lines):
                    # Check if next line already has file references
                    next_line = lines[i + 1] if i + 1 < len(lines) else ""
                    
                    # Don't add if already has references
                    if not next_line.startswith('**Related Files:**'):
                        # Add related files as a comment
                        file_list = ', '.join([f'`{f}`' for f in related_files[:5]])
                        if len(related_files) > 5:
                            file_list += f' (+{len(related_files) - 5} more)'
                        updated_lines.append(f"**Related Files:** {file_list}")
                        updated_lines.append("")
        
        # Create backup
        backup_path = arch_file.with_suffix('.md.backup')
        arch_file.rename(backup_path)
        logger.info(f"Created backup at {backup_path}")
        
        # Write updated content
        arch_file.write_text('\n'.join(updated_lines))
        logger.info("Updated ARCHITECTURE.md with file references")
    
    def generate_architecture_index(self, mapping: Dict) -> str:
        """Generate an index document linking architecture to code."""
        output = []
        output.append("# Architecture to Code Index")
        output.append(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        output.append("## Overview")
        output.append(f"- Total Architecture Sections: {len(mapping['architecture_to_code'])}")
        output.append(f"- Total Code Components: {len(mapping['code_to_architecture'])}")
        output.append(f"- Orphaned Components: {len(mapping['orphaned_components'])}")
        output.append(f"- Undocumented Sections: {len(mapping['undocumented_sections'])}")
        output.append("")
        
        # Architecture to Code mapping
        output.append("## Architecture Sections → Code Files")
        output.append("")
        
        for section, data in sorted(mapping['architecture_to_code'].items()):
            if data['related_files']:
                indent = "  " * (data['level'] - 1)
                output.append(f"{indent}### {section}")
                for file in data['related_files'][:10]:  # Limit to 10 files
                    output.append(f"{indent}  - `{file}`")
                if len(data['related_files']) > 10:
                    output.append(f"{indent}  - _(+{len(data['related_files']) - 10} more files)_")
                output.append("")
        
        # Code to Architecture mapping
        output.append("## Code Files → Architecture Sections")
        output.append("")
        
        # Group by directory
        files_by_dir = defaultdict(list)
        for file_path in mapping['code_to_architecture'].keys():
            dir_path = str(Path(file_path).parent)
            files_by_dir[dir_path].append(file_path)
        
        for dir_path in sorted(files_by_dir.keys()):
            output.append(f"### {dir_path}/")
            for file_path in sorted(files_by_dir[dir_path]):
                data = mapping['code_to_architecture'][file_path]
                output.append(f"- **{Path(file_path).name}**")
                if data['related_sections']:
                    for section in data['related_sections'][:3]:
                        output.append(f"  → {section}")
                else:
                    output.append("  → _(No architecture reference)_")
            output.append("")
        
        # Orphaned components
        if mapping['orphaned_components']:
            output.append("## Orphaned Components (No Architecture Reference)")
            output.append("")
            for file in sorted(mapping['orphaned_components'])[:20]:
                output.append(f"- `{file}`")
            if len(mapping['orphaned_components']) > 20:
                output.append(f"- _(+{len(mapping['orphaned_components']) - 20} more)_")
            output.append("")
        
        # Undocumented sections
        if mapping['undocumented_sections']:
            output.append("## Architecture Sections Without Code")
            output.append("")
            for section in sorted(mapping['undocumented_sections'])[:20]:
                output.append(f"- {section}")
            if len(mapping['undocumented_sections']) > 20:
                output.append(f"- _(+{len(mapping['undocumented_sections']) - 20} more)_")
        
        return '\n'.join(output)
    
    def update_code_headers(self, mapping: Dict) -> int:
        """Update code files with architecture references."""
        logger.info("Updating code files with architecture references...")
        
        updated_count = 0
        src_path = Path('src')
        
        for file_path, data in mapping['code_to_architecture'].items():
            if not data['related_sections']:
                continue
            
            full_path = Path(file_path)
            if not full_path.exists():
                continue
            
            try:
                content = full_path.read_text()
                lines = content.split('\n')
                
                # Check if already has architecture reference
                if 'Architecture Reference:' in content:
                    continue
                
                # Find the right place to insert architecture reference
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#') and not line.startswith('"""'):
                        # Found first code line, insert before it
                        insert_index = i
                        break
                    elif line.startswith('"""') and i > 0:
                        # Found docstring, find its end
                        for j in range(i + 1, len(lines)):
                            if '"""' in lines[j]:
                                insert_index = j + 1
                                break
                        break
                
                # Insert architecture reference
                section_refs = ', '.join([f'"{s}"' for s in data['related_sections'][:3]])
                reference_comment = [
                    "",
                    "# Architecture Reference:",
                    f"# See ARCHITECTURE.md sections: {section_refs}",
                    ""
                ]
                
                for ref_line in reversed(reference_comment):
                    lines.insert(insert_index, ref_line)
                
                # Write back
                full_path.write_text('\n'.join(lines))
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Error updating {file_path}: {e}")
        
        logger.info(f"Updated {updated_count} code files with architecture references")
        return updated_count


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("ARCHITECTURE LINKING SYSTEM")
    logger.info("Task 3.1.3: Link functionality to ARCHITECTURE.md")
    logger.info("=" * 60)
    
    # Initialize system
    linker = ArchitectureLinkingSystem()
    
    # Parse architecture
    arch_file = Path('ARCHITECTURE.md')
    if not arch_file.exists():
        logger.error("ARCHITECTURE.md not found!")
        return
    
    linker.parse_architecture(arch_file)
    
    # Scan codebase
    src_path = Path('src')
    if not src_path.exists():
        logger.error("src/ directory not found!")
        return
    
    linker.scan_codebase(src_path)
    
    # Create mapping
    mapping = linker.create_mapping()
    
    # Save mapping as JSON
    mapping_file = Path('architecture_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved mapping to {mapping_file}")
    
    # Generate index document
    index_content = linker.generate_architecture_index(mapping)
    index_file = Path('ARCHITECTURE_INDEX.md')
    index_file.write_text(index_content)
    logger.info(f"Generated {index_file}")
    
    # Update ARCHITECTURE.md with references
    # linker.update_architecture_file(arch_file, mapping)
    
    # Update code files with architecture references
    # updated_files = linker.update_code_headers(mapping)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("LINKING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Parsed {len(linker.architecture_sections)} architecture sections")
    logger.info(f"✅ Scanned {len(linker.code_components)} code files")
    logger.info(f"✅ Found {len(mapping['orphaned_components'])} orphaned components")
    logger.info(f"✅ Found {len(mapping['undocumented_sections'])} undocumented sections")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - architecture_mapping.json: Complete mapping data")
    logger.info("  - ARCHITECTURE_INDEX.md: Human-readable index")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review ARCHITECTURE_INDEX.md")
    logger.info("  2. Update orphaned components with architecture refs")
    logger.info("  3. Document undocumented architecture sections")
    logger.info("  4. Run with --update flag to modify files")


if __name__ == "__main__":
    main()