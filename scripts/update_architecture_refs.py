#!/usr/bin/env python3
"""
Update code files with architecture references based on the mapping.
Part of Task 3.1.3: Link functionality to ARCHITECTURE.md sections.
"""

import json
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_mapping(mapping_file: Path) -> dict:
    """Load the architecture mapping from JSON."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def update_file_with_reference(file_path: Path, sections: list) -> bool:
    """Update a single file with architecture reference."""
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return False
    
    try:
        content = file_path.read_text()
        
        # Check if already has architecture reference
        if 'Architecture Reference:' in content:
            logger.debug(f"File already has reference: {file_path}")
            return False
        
        lines = content.split('\n')
        
        # Find insertion point (after docstring and imports)
        insert_index = 0
        in_docstring = False
        docstring_count = 0
        
        for i, line in enumerate(lines):
            # Track docstrings
            if '"""' in line:
                docstring_count += line.count('"""')
                if docstring_count % 2 == 1:
                    in_docstring = True
                else:
                    in_docstring = False
                    if docstring_count >= 2:  # After first docstring
                        insert_index = i + 1
                        break
            
            # If no docstring, insert after imports
            if not in_docstring and line.strip() and not line.startswith('import') and not line.startswith('from'):
                if not line.startswith('#'):
                    insert_index = i
                    break
        
        # Format sections for reference
        if len(sections) > 3:
            section_list = ', '.join(sections[:3]) + f' (+{len(sections) - 3} more)'
        else:
            section_list = ', '.join(sections)
        
        # Create the architecture reference block
        arch_ref = [
            "",
            "# Architecture Reference:",
            f"# See ARCHITECTURE.md > {section_list}",
            ""
        ]
        
        # Insert the reference
        for ref_line in reversed(arch_ref):
            lines.insert(insert_index, ref_line)
        
        # Write back
        file_path.write_text('\n'.join(lines))
        logger.info(f"Updated: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating {file_path}: {e}")
        return False


def update_all_files(mapping: dict) -> tuple:
    """Update all files with architecture references."""
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for file_path_str, data in mapping['code_to_architecture'].items():
        file_path = Path(file_path_str)
        sections = data.get('related_sections', [])
        
        if not sections:
            logger.debug(f"No sections for: {file_path}")
            skipped_count += 1
            continue
        
        if update_file_with_reference(file_path, sections):
            updated_count += 1
        else:
            skipped_count += 1
    
    return updated_count, skipped_count, error_count


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("UPDATE ARCHITECTURE REFERENCES")
    logger.info("=" * 60)
    
    # Load mapping
    mapping_file = Path('architecture_mapping.json')
    if not mapping_file.exists():
        logger.error("architecture_mapping.json not found! Run link_architecture.py first.")
        return
    
    mapping = load_mapping(mapping_file)
    logger.info(f"Loaded mapping with {len(mapping['code_to_architecture'])} files")
    
    # Statistics
    files_with_refs = sum(1 for data in mapping['code_to_architecture'].values() 
                         if data.get('related_sections'))
    files_without_refs = len(mapping['code_to_architecture']) - files_with_refs
    
    logger.info(f"Files with architecture sections: {files_with_refs}")
    logger.info(f"Files without architecture sections: {files_without_refs}")
    
    # Update files
    logger.info("")
    logger.info("Updating files...")
    updated, skipped, errors = update_all_files(mapping)
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("UPDATE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Updated: {updated} files")
    logger.info(f"⏭️  Skipped: {skipped} files (already have refs or no sections)")
    logger.info(f"❌ Errors: {errors} files")
    
    # List orphaned components
    if mapping.get('orphaned_components'):
        logger.info("")
        logger.info("Orphaned components (need architecture documentation):")
        for comp in mapping['orphaned_components'][:10]:
            logger.info(f"  - {comp}")
        if len(mapping['orphaned_components']) > 10:
            logger.info(f"  - (+{len(mapping['orphaned_components']) - 10} more)")


if __name__ == "__main__":
    main()