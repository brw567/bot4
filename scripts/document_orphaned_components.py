#!/usr/bin/env python3
"""
Document orphaned components in ARCHITECTURE.md
Part of Task 3.1.4: Document all features in architecture
"""

import json
from pathlib import Path
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_mapping(mapping_file: Path) -> dict:
    """Load the architecture mapping from JSON."""
    with open(mapping_file, 'r') as f:
        return json.load(f)


def analyze_component(file_path: Path) -> dict:
    """Analyze a Python file to extract its purpose and features."""
    try:
        content = file_path.read_text()
        tree = ast.parse(content)
        
        info = {
            'path': str(file_path),
            'name': file_path.stem,
            'docstring': None,
            'classes': [],
            'functions': [],
            'imports': [],
        }
        
        # Extract module docstring
        if isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            info['docstring'] = tree.body[0].value.value
        
        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {'name': node.name, 'docstring': ast.get_docstring(node)}
                info['classes'].append(class_info)
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                func_info = {'name': node.name, 'docstring': ast.get_docstring(node)}
                info['functions'].append(func_info)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                info['imports'].append(node.module)
        
        return info
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return None


def categorize_orphaned_components(orphaned: list) -> dict:
    """Categorize orphaned components by type and importance."""
    categories = {
        'core_infrastructure': [],
        'trading_strategies': [],
        'risk_management': [],
        'data_management': [],
        'api_endpoints': [],
        'utilities': [],
        'testing': [],
    }
    
    for comp_path in orphaned:
        path = Path(comp_path)
        
        # Categorize based on path and name
        if 'core' in path.parts:
            if any(x in path.name for x in ['trading', 'order', 'exchange', 'smart']):
                categories['core_infrastructure'].append(comp_path)
            elif any(x in path.name for x in ['risk', 'stop', 'position', 'validator']):
                categories['risk_management'].append(comp_path)
            elif any(x in path.name for x in ['data', 'cache', 'correlation', 'calculator']):
                categories['data_management'].append(comp_path)
            else:
                categories['core_infrastructure'].append(comp_path)
        elif 'strategies' in path.parts:
            categories['trading_strategies'].append(comp_path)
        elif 'api' in path.parts:
            categories['api_endpoints'].append(comp_path)
        elif 'utils' in path.parts:
            categories['utilities'].append(comp_path)
        elif 'test' in path.name:
            categories['testing'].append(comp_path)
        else:
            categories['utilities'].append(comp_path)
    
    return categories


def generate_documentation_sections(categories: dict) -> str:
    """Generate documentation sections for orphaned components."""
    sections = []
    
    # Core Infrastructure Components
    if categories['core_infrastructure']:
        sections.append("\n### Additional Core Infrastructure Components\n")
        
        for comp_path in categories['core_infrastructure']:
            path = Path(comp_path)
            info = analyze_component(path)
            if info:
                comp_name = path.stem.replace('_', ' ').title()
                sections.append(f"\n#### {comp_name} (`{comp_path}`)")
                
                if info['docstring']:
                    sections.append(f"**Purpose:** {info['docstring'].strip().split('.')[0]}.")
                
                if info['classes']:
                    sections.append("\n**Key Classes:**")
                    for cls in info['classes'][:4]:  # Limit to 4 classes
                        if cls['docstring']:
                            sections.append(f"- `{cls['name']}`: {cls['docstring'].strip().split('.')[0]}")
                        else:
                            sections.append(f"- `{cls['name']}`")
                
                if info['functions'] and len(info['functions']) > 2:
                    sections.append("\n**Key Functions:**")
                    for func in info['functions'][:4]:  # Limit to 4 functions
                        if func['docstring']:
                            sections.append(f"- `{func['name']}()`: {func['docstring'].strip().split('.')[0]}")
                        else:
                            sections.append(f"- `{func['name']}()`")
    
    # Trading Strategies
    if categories['trading_strategies']:
        sections.append("\n### Strategy Components\n")
        
        for comp_path in categories['trading_strategies']:
            path = Path(comp_path)
            info = analyze_component(path)
            if info:
                strategy_name = path.stem.replace('_', ' ').title()
                sections.append(f"\n#### {strategy_name} Strategy (`{comp_path}`)")
                
                if info['docstring']:
                    sections.append(f"**Description:** {info['docstring'].strip()}")
                
                if info['classes']:
                    sections.append("\n**Implementation:**")
                    for cls in info['classes'][:2]:  # Limit to 2 classes
                        if cls['docstring']:
                            sections.append(f"- `{cls['name']}`: {cls['docstring'].strip().split('.')[0]}")
                        else:
                            sections.append(f"- `{cls['name']}`")
    
    # Risk Management Components
    if categories['risk_management']:
        sections.append("\n### Risk Management Components\n")
        
        for comp_path in categories['risk_management']:
            path = Path(comp_path)
            info = analyze_component(path)
            if info:
                comp_name = path.stem.replace('_', ' ').title()
                sections.append(f"\n#### {comp_name} (`{comp_path}`)")
                
                if info['docstring']:
                    sections.append(f"**Purpose:** {info['docstring'].strip()}")
                
                if info['classes']:
                    sections.append("\n**Components:**")
                    for cls in info['classes'][:3]:
                        sections.append(f"- `{cls['name']}`")
    
    # Data Management
    if categories['data_management']:
        sections.append("\n### Data Management Components\n")
        
        for comp_path in categories['data_management']:
            path = Path(comp_path)
            comp_name = path.stem.replace('_', ' ').title()
            sections.append(f"\n#### {comp_name} (`{comp_path}`)")
            sections.append(f"**Purpose:** Data processing and management")
    
    return '\n'.join(sections)


def update_architecture_file(new_sections: str) -> bool:
    """Append new sections to ARCHITECTURE.md."""
    arch_file = Path('ARCHITECTURE.md')
    
    try:
        content = arch_file.read_text()
        
        # Find insertion point (before Technology Stack or at end)
        lines = content.split('\n')
        insert_index = len(lines)
        
        for i, line in enumerate(lines):
            if '## Technology Stack' in line:
                insert_index = i
                break
        
        # Insert new sections
        lines.insert(insert_index, new_sections)
        
        # Write back
        arch_file.write_text('\n'.join(lines))
        logger.info(f"Updated ARCHITECTURE.md with new sections")
        return True
        
    except Exception as e:
        logger.error(f"Error updating ARCHITECTURE.md: {e}")
        return False


def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("DOCUMENT ORPHANED COMPONENTS")
    logger.info("=" * 60)
    
    # Load mapping
    mapping_file = Path('architecture_mapping.json')
    if not mapping_file.exists():
        logger.error("architecture_mapping.json not found! Run link_architecture.py first.")
        return
    
    mapping = load_mapping(mapping_file)
    orphaned = mapping.get('orphaned_components', [])
    
    logger.info(f"Found {len(orphaned)} orphaned components to document")
    
    # Categorize components
    categories = categorize_orphaned_components(orphaned)
    
    # Log category counts
    for category, components in categories.items():
        if components:
            logger.info(f"{category}: {len(components)} components")
    
    # Generate documentation
    logger.info("\nGenerating documentation sections...")
    new_sections = generate_documentation_sections(categories)
    
    # Preview
    logger.info("\nGenerated documentation preview:")
    logger.info("-" * 40)
    preview_lines = new_sections.split('\n')[:20]
    for line in preview_lines:
        logger.info(line)
    logger.info(f"... ({len(new_sections.split(chr(10)))} total lines)")
    
    # Update ARCHITECTURE.md
    if update_architecture_file(new_sections):
        logger.info("\n✅ Successfully documented orphaned components in ARCHITECTURE.md")
    else:
        logger.error("\n❌ Failed to update ARCHITECTURE.md")
    
    # Create summary report
    report = [
        "# Orphaned Components Documentation Report",
        f"**Date**: 2025-08-10",
        f"**Total Components Documented**: {len(orphaned)}",
        "",
        "## Categories:",
    ]
    
    for category, components in categories.items():
        if components:
            report.append(f"- **{category.replace('_', ' ').title()}**: {len(components)} components")
    
    report.append("")
    report.append("## Key Components Added:")
    
    # List key components
    key_components = [
        'src/core/trading_integrity.py',
        'src/core/smart_order_router.py',
        'src/strategies/delta_neutral_enhanced.py',
        'src/strategies/triangular_arb_enhanced.py',
        'src/core/monte_carlo_engine.py',
        'src/core/online_learning_system.py',
    ]
    
    for comp in key_components:
        if comp in orphaned:
            report.append(f"- ✅ {comp}")
    
    # Save report
    report_file = Path('ORPHANED_COMPONENTS_REPORT.md')
    report_file.write_text('\n'.join(report))
    logger.info(f"\n📝 Report saved to {report_file}")


if __name__ == "__main__":
    main()