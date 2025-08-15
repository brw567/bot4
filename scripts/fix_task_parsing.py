#!/usr/bin/env python3
"""
Fix task parsing in the task numbering system.
This version properly handles various task formats including bold text.
"""

import re
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_task_line(line: str) -> Dict[str, any]:
    """Parse a task line and extract all components."""
    
    # First, check if this is a task line
    if not re.match(r'^\s*- \[[ x]\]', line):
        return None
    
    # Extract indentation
    indent_match = re.match(r'^(\s*)', line)
    indent = len(indent_match.group(1)) if indent_match else 0
    
    # Extract checkbox status
    status_match = re.search(r'- \[([ x])\]', line)
    status = 'completed' if status_match and status_match.group(1) == 'x' else 'pending'
    
    # Remove the checkbox part to make parsing easier
    text_after_checkbox = line[line.index(']') + 1:].strip()
    
    # Extract model and complexity tags
    model = None
    complexity = None
    model_match = re.search(r'\[([SO])4\]', text_after_checkbox)
    if model_match:
        model = model_match.group(1) + '4'
        text_after_checkbox = text_after_checkbox.replace(model_match.group(0), '')
    
    complexity_match = re.search(r'\[(LOW|MED|HIGH|CRIT)\]', text_after_checkbox)
    if complexity_match:
        complexity = complexity_match.group(1)
        text_after_checkbox = text_after_checkbox.replace(complexity_match.group(0), '')
    
    # Extract title - handle bold text
    # Case 1: **EPIC-N**: Title
    # Case 2: **N.N** Title
    # Case 3: Plain text title
    title = text_after_checkbox.strip()
    
    # Remove surrounding ** if present
    title = re.sub(r'^\*\*(.*?)\*\*$', r'\1', title)
    title = re.sub(r'\*\*$', '', title)  # Remove trailing **
    
    # Extract task ID if present in title
    task_id = None
    id_match = re.match(r'^(EPIC-\d+|\d+(?:\.\d+)*)[:\s]+(.+)', title)
    if id_match:
        task_id = id_match.group(1)
        title = id_match.group(2)
    
    return {
        'indent': indent,
        'level': indent // 2,
        'status': status,
        'task_id': task_id,
        'title': title.strip(),
        'model': model,
        'complexity': complexity,
        'original_line': line
    }


def test_parsing():
    """Test the parsing function with various formats."""
    test_lines = [
        '- [x] **EPIC-2**: Full System Code Review',
        '- [x] **EPIC-3**: Trading Logic Integrity Review',
        '- [x] Start application and perform comprehensive testing',
        '- [ ] **5.4** Add A/B testing for ML parameters **[S4][MED]**',
        '  - [x] Create missing tables (system_metrics, alerts, routing_decisions, etc.)',
        '- [ ] Implement data validation at ingestion **[S4][MED]**',
        '- [x] Fix test fixtures to use @pytest_asyncio.fixture'
    ]
    
    for line in test_lines:
        result = parse_task_line(line)
        if result:
            print(f"\nLine: {line[:60]}...")
            print(f"  Title: [{result['title']}]")
            print(f"  Task ID: {result['task_id']}")
            print(f"  Level: {result['level']}")
            print(f"  Status: {result['status']}")
            print(f"  Model: {result['model']}")
            print(f"  Complexity: {result['complexity']}")


def fix_task_numbering_file():
    """Fix the task numbering implementation file."""
    
    file_path = Path('scripts/implement_task_numbering.py')
    if not file_path.exists():
        logger.error(f"File {file_path} not found")
        return
    
    # Read the file
    content = file_path.read_text()
    
    # Replace the problematic regex section
    old_pattern = r"task_match = re\.match\(r'\^\(\\s\*\)- \\\[\(\[ x\]\)\\\].*?', line\)"
    
    # Create the new parsing code
    new_code = '''# Parse task line using improved parsing
            parsed = self._parse_task_line(line)
            if parsed:
                indent = parsed['indent']
                status = parsed['status']
                title = parsed['title']
                model = parsed['model']
                complexity = parsed['complexity']
                task_id = parsed['task_id']
                level = parsed['level']'''
    
    # We'll need to add the parsing method too
    parse_method = '''
    def _parse_task_line(self, line: str) -> Dict[str, any]:
        """Parse a task line and extract all components."""
        
        # First, check if this is a task line
        if not re.match(r'^\\s*- \\[[ x]\\]', line):
            return None
        
        # Extract indentation
        indent_match = re.match(r'^(\\s*)', line)
        indent = len(indent_match.group(1)) if indent_match else 0
        
        # Extract checkbox status
        status_match = re.search(r'- \\[([ x])\\]', line)
        status = 'completed' if status_match and status_match.group(1) == 'x' else 'pending'
        
        # Remove the checkbox part to make parsing easier
        text_after_checkbox = line[line.index(']') + 1:].strip()
        
        # Extract model and complexity tags
        model = None
        complexity = None
        model_match = re.search(r'\\[([SO])4\\]', text_after_checkbox)
        if model_match:
            model = model_match.group(1) + '4'
            text_after_checkbox = text_after_checkbox.replace(model_match.group(0), '')
        
        complexity_match = re.search(r'\\[(LOW|MED|HIGH|CRIT)\\]', text_after_checkbox)
        if complexity_match:
            complexity = complexity_match.group(1)
            text_after_checkbox = text_after_checkbox.replace(complexity_match.group(0), '')
        
        # Extract title - handle bold text
        title = text_after_checkbox.strip()
        
        # Remove surrounding ** if present
        title = re.sub(r'^\\*\\*(.*?)\\*\\*$', r'\\1', title)
        title = re.sub(r'\\*\\*$', '', title)  # Remove trailing **
        
        # Extract task ID if present in title
        task_id = None
        id_match = re.match(r'^(EPIC-\\d+|\\d+(?:\\.\\d+)*)[:\\s]+(.+)', title)
        if id_match:
            task_id = id_match.group(1)
            title = id_match.group(2)
        
        return {
            'indent': indent,
            'level': indent // 2,
            'status': status,
            'task_id': task_id,
            'title': title.strip(),
            'model': model,
            'complexity': complexity
        }
'''
    
    logger.info("Parsing method would be added to implement_task_numbering.py")
    logger.info("Run test_parsing() to verify the parsing works correctly")


if __name__ == "__main__":
    print("Testing task line parsing...")
    print("=" * 60)
    test_parsing()
    
    print("\n" + "=" * 60)
    print("To fix the implement_task_numbering.py file, the parsing")
    print("logic needs to be replaced with the improved version above.")