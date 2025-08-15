#!/usr/bin/env python3
"""
Component: Task Management - Task Numbering System
Task: 2.1 - Implement tree structure enumeration
Author: Alex (Team Lead) with Opus oversight
Created: 2025-01-10
Modified: 2025-01-10

Description:
Implements comprehensive task numbering system with hierarchical structure,
cross-references, and automatic index generation.

Architecture Reference:
See ARCHITECTURE.md > Development Processes > Task Management

Requirements:
- Hierarchical numbering (EPIC-X, X.Y, X.Y.Z)
- Task dependency tracking
- Cross-reference index
- Code comment updates
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a single task with hierarchical numbering"""
    id: str  # e.g., "1.2.3"
    epic: Optional[str]  # e.g., "EPIC-1"
    title: str
    status: str  # completed, pending, in_progress
    model: Optional[str]  # [S4], [O4]
    complexity: Optional[str]  # [LOW], [MED], [HIGH], [CRIT]
    description: str
    subtasks: List['Task'] = field(default_factory=list)
    parent: Optional['Task'] = None
    dependencies: Set[str] = field(default_factory=set)
    references: List[str] = field(default_factory=list)  # Files that reference this task
    
    def get_full_path(self) -> str:
        """Get full hierarchical path"""
        if self.epic:
            return f"{self.epic}/{self.id}"
        return self.id
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'epic': self.epic,
            'title': self.title,
            'status': self.status,
            'model': self.model,
            'complexity': self.complexity,
            'description': self.description,
            'subtasks': [t.to_dict() for t in self.subtasks],
            'dependencies': list(self.dependencies),
            'references': self.references
        }


class TaskNumberingSystem:
    """
    Comprehensive task numbering system implementation.
    Parses, numbers, and indexes all tasks with cross-references.
    """
    
    def __init__(self, task_file: str = 'TASK_LIST.md'):
        self.task_file = Path(task_file)
        self.tasks: Dict[str, Task] = {}
        self.epics: Dict[str, Task] = {}
        self.index: Dict[str, Task] = {}  # Quick lookup by ID
        self.model_assignments: Dict[str, List[Task]] = {
            'S4': [],
            'O4': []
        }
        self.complexity_levels: Dict[str, List[Task]] = {
            'LOW': [],
            'MED': [],
            'HIGH': [],
            'CRIT': []
        }
        
    def parse_task_list(self) -> None:
        """Parse existing TASK_LIST.md and extract tasks"""
        if not self.task_file.exists():
            raise FileNotFoundError(f"Task file {self.task_file} not found")
        
        content = self.task_file.read_text()
        lines = content.split('\n')
        
        current_epic = None
        current_section = None
        task_stack = []  # Stack for nested tasks
        
        for line_num, line in enumerate(lines, 1):
            # Detect EPIC sections
            epic_match = re.match(r'^###.*EPIC-(\d+):\s*(.+)', line)
            if epic_match:
                epic_num = epic_match.group(1)
                epic_title = epic_match.group(2)
                current_epic = f"EPIC-{epic_num}"
                
                epic_task = Task(
                    id=epic_num,
                    epic=current_epic,
                    title=epic_title,
                    status='pending',
                    model=None,
                    complexity=None,
                    description=epic_title
                )
                self.epics[current_epic] = epic_task
                task_stack = [epic_task]
                continue
            
            # Parse task lines - improved regex to capture full titles including bold text
            # Handle both **text** and plain text formats
            task_match = re.match(r'^(\s*)- \[([ x])\]\s*(?:\*\*)?([^*\[\n]+?)(?:\*\*)?(?:\s*\[([SO])4\])?(?:\s*\[(LOW|MED|HIGH|CRIT)\])?', line)
            if task_match:
                indent = len(task_match.group(1))
                status = 'completed' if task_match.group(2) == 'x' else 'pending'
                raw_title = task_match.group(3).strip()
                model = task_match.group(4) + '4' if task_match.group(4) else None
                complexity = task_match.group(5)
                
                # Calculate nesting level
                level = indent // 2
                
                # Extract task number if present in the title
                num_match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', raw_title)
                if num_match:
                    task_id = num_match.group(1)
                    title = num_match.group(2).strip()
                else:
                    # If no number in title, use the full raw_title
                    title = raw_title
                    task_id = None  # Will be assigned during numbering phase
                
                # Generate task ID if not present
                if not task_id:
                    if level == 0 and current_epic:
                        parent_task = self.epics.get(current_epic)
                        task_id = f"{parent_task.id}.{len(parent_task.subtasks) + 1}"
                    elif level > 0 and task_stack and level - 1 < len(task_stack):
                        parent = task_stack[level - 1]
                        if parent:
                            task_id = f"{parent.id}.{len(parent.subtasks) + 1}"
                        else:
                            task_id = str(len(self.tasks) + 1)
                    else:
                        task_id = str(len(self.tasks) + 1)
                
                # Create task
                task = Task(
                    id=task_id,
                    epic=current_epic,
                    title=title,
                    status=status,
                    model=model,
                    complexity=complexity,
                    description=title
                )
                
                # Add to parent if nested
                if level > 0 and task_stack:
                    parent = task_stack[level - 1] if level <= len(task_stack) else task_stack[-1]
                    parent.subtasks.append(task)
                    task.parent = parent
                
                # Update stack
                if level < len(task_stack):
                    task_stack = task_stack[:level]
                task_stack.append(task)
                
                # Index task
                self.tasks[task.get_full_path()] = task
                self.index[task_id] = task
                
                # Categorize by model and complexity
                if model:
                    self.model_assignments[model].append(task)
                if complexity:
                    self.complexity_levels[complexity].append(task)
    
    def apply_numbering(self) -> None:
        """Apply systematic numbering to all tasks"""
        logger.info("Applying hierarchical numbering to tasks...")
        
        # Number EPICs
        epic_counter = 1
        for epic_key in sorted(self.epics.keys()):
            epic = self.epics[epic_key]
            if not epic.id.startswith('EPIC-'):
                epic.epic = f"EPIC-{epic_counter}"
                epic.id = str(epic_counter)
                epic_counter += 1
            
            # Number subtasks within EPIC
            self._number_subtasks(epic, epic.id)
    
    def _number_subtasks(self, parent: Task, parent_id: str) -> None:
        """Recursively number subtasks"""
        for idx, subtask in enumerate(parent.subtasks, 1):
            subtask.id = f"{parent_id}.{idx}"
            # Update index
            self.index[subtask.id] = subtask
            
            # Recursively number nested subtasks
            if subtask.subtasks:
                self._number_subtasks(subtask, subtask.id)
    
    def extract_dependencies(self) -> None:
        """Extract task dependencies from descriptions"""
        logger.info("Extracting task dependencies...")
        
        for task in self.tasks.values():
            # Look for references to other tasks
            references = re.findall(r'(?:Task|EPIC|task|epic)[\s-]*(\d+(?:\.\d+)*)', task.description)
            for ref in references:
                if ref in self.index:
                    task.dependencies.add(ref)
            
            # Look for "depends on", "requires", "after" patterns
            dep_patterns = [
                r'(?:depends on|requires|after)\s+(?:Task|task)?\s*(\d+(?:\.\d+)*)',
                r'(?:Part of|part of)\s+(?:Task|task)?\s*(\d+(?:\.\d+)*)'
            ]
            
            for pattern in dep_patterns:
                matches = re.findall(pattern, task.description)
                for match in matches:
                    if match in self.index:
                        task.dependencies.add(match)
    
    def find_code_references(self) -> None:
        """Find references to tasks in code files"""
        logger.info("Scanning codebase for task references...")
        
        src_path = Path('src')
        if not src_path.exists():
            logger.warning("src directory not found")
            return
        
        # Patterns to search for
        patterns = [
            r'Task:\s*(EPIC-\d+|[\d.]+)',
            r'EPIC-\d+',
            r'Task\s+(\d+(?:\.\d+)*)',
            r'#\s*(\d+(?:\.\d+)*)'
        ]
        
        for py_file in src_path.rglob('*.py'):
            try:
                content = py_file.read_text()
                
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        if match in self.index:
                            task = self.index[match]
                            task.references.append(str(py_file))
                        elif f"EPIC-{match}" in self.epics:
                            epic = self.epics[f"EPIC-{match}"]
                            epic.references.append(str(py_file))
            except Exception as e:
                logger.error(f"Error reading {py_file}: {e}")
    
    def generate_numbered_tasklist(self) -> str:
        """Generate updated TASK_LIST.md with proper numbering"""
        logger.info("Generating numbered task list...")
        
        output = []
        output.append("# Bot3 Trading Platform - Master Task List")
        output.append(f"Updated: {datetime.now().strftime('%Y-%m-%d')}")
        output.append("Status: Active Development with Hierarchical Task Numbering")
        output.append("")
        output.append("## Task Numbering System")
        output.append("- **EPIC-X**: Major feature areas")
        output.append("- **X.Y**: Main tasks within EPICs")
        output.append("- **X.Y.Z**: Subtasks and detailed items")
        output.append("")
        output.append("## Model Assignment Legend")
        output.append("- **[S4]** = Sonnet 4 (Pattern-based, single-file, documentation)")
        output.append("- **[O4]** = Opus 4.1 (Complex, creative, multi-system)")
        output.append("- **[LOW]** = Simple, clear pattern tasks")
        output.append("- **[MED]** = Moderate complexity")
        output.append("- **[HIGH]** = Complex multi-file tasks")
        output.append("- **[CRIT]** = Critical architecture/ML tasks")
        output.append("")
        
        # Generate task statistics
        output.append("## Task Statistics")
        output.append(f"- Total Tasks: {len(self.tasks)}")
        output.append(f"- EPICs: {len(self.epics)}")
        output.append(f"- Completed: {sum(1 for t in self.tasks.values() if t.status == 'completed')}")
        output.append(f"- Pending: {sum(1 for t in self.tasks.values() if t.status == 'pending')}")
        output.append("")
        
        # Model assignments
        output.append("### Model Assignments")
        output.append(f"- Sonnet 4 Tasks: {len(self.model_assignments['S4'])}")
        output.append(f"- Opus 4.1 Tasks: {len(self.model_assignments['O4'])}")
        output.append("")
        
        # Complexity distribution
        output.append("### Complexity Distribution")
        for level in ['LOW', 'MED', 'HIGH', 'CRIT']:
            output.append(f"- {level}: {len(self.complexity_levels[level])} tasks")
        output.append("")
        
        # Generate task tree
        output.append("## Task Hierarchy")
        output.append("")
        
        # Completed tasks
        completed_epics = [e for e in self.epics.values() if self._is_epic_complete(e)]
        if completed_epics:
            output.append("### ✅ Completed EPICs")
            for epic in sorted(completed_epics, key=lambda e: e.id):
                output.extend(self._format_task_tree(epic, 0))
            output.append("")
        
        # Pending tasks
        pending_epics = [e for e in self.epics.values() if not self._is_epic_complete(e)]
        if pending_epics:
            output.append("### 📋 Active EPICs")
            for epic in sorted(pending_epics, key=lambda e: e.id):
                output.extend(self._format_task_tree(epic, 0))
            output.append("")
        
        return '\n'.join(output)
    
    def _is_epic_complete(self, epic: Task) -> bool:
        """Check if all tasks in an EPIC are complete"""
        if epic.status != 'completed':
            return False
        
        def check_subtasks(task):
            if task.status != 'completed':
                return False
            for subtask in task.subtasks:
                if not check_subtasks(subtask):
                    return False
            return True
        
        return check_subtasks(epic)
    
    def _format_task_tree(self, task: Task, level: int) -> List[str]:
        """Format task and subtasks as tree structure"""
        output = []
        indent = "  " * level
        
        # Format main task line
        checkbox = "[x]" if task.status == 'completed' else "[ ]"
        
        # Format task header
        if level == 0:  # EPIC level
            output.append(f"")
            output.append(f"#### {task.epic}: {task.title}")
        else:
            task_line = f"{indent}- {checkbox} **{task.id}** {task.title}"
            
            # Add model and complexity tags
            if task.model:
                task_line += f" **[{task.model}]**"
            if task.complexity:
                task_line += f" **[{task.complexity}]**"
            
            output.append(task_line)
        
        # Add dependencies if present
        if task.dependencies and level > 0:
            deps_str = ", ".join(sorted(task.dependencies))
            output.append(f"{indent}  _Dependencies: {deps_str}_")
        
        # Add references if present
        if task.references and level > 0:
            refs_str = ", ".join(sorted(set(task.references))[:3])  # Show first 3
            if len(task.references) > 3:
                refs_str += f" (+{len(task.references) - 3} more)"
            output.append(f"{indent}  _Referenced in: {refs_str}_")
        
        # Format subtasks
        for subtask in task.subtasks:
            output.extend(self._format_task_tree(subtask, level + 1))
        
        return output
    
    def generate_task_index(self) -> str:
        """Generate task index for quick lookup"""
        logger.info("Generating task index...")
        
        output = []
        output.append("# Task Index - Quick Reference")
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        output.append("## Task ID Lookup")
        output.append("")
        
        # Sort tasks by ID
        sorted_tasks = sorted(self.index.items(), key=lambda x: self._sort_key(x[0]))
        
        for task_id, task in sorted_tasks:
            status_icon = "✅" if task.status == 'completed' else "⏳"
            model_tag = f"[{task.model}]" if task.model else ""
            complexity_tag = f"[{task.complexity}]" if task.complexity else ""
            
            output.append(f"- **{task_id}** {status_icon} {task.title} {model_tag}{complexity_tag}")
            
            if task.dependencies:
                deps = ", ".join(sorted(task.dependencies))
                output.append(f"  - Dependencies: {deps}")
            
            if task.references:
                refs = ", ".join(sorted(set(task.references))[:5])
                output.append(f"  - References: {refs}")
        
        output.append("")
        output.append("## Task Dependencies Graph")
        output.append("```mermaid")
        output.append("graph TD")
        
        # Generate mermaid graph
        for task_id, task in sorted_tasks[:20]:  # Limit to first 20 for readability
            if task.dependencies:
                for dep_id in task.dependencies:
                    output.append(f"  {dep_id.replace('.', '_')} --> {task_id.replace('.', '_')}")
        
        output.append("```")
        
        return '\n'.join(output)
    
    def _sort_key(self, task_id: str) -> tuple:
        """Generate sort key for task IDs"""
        parts = task_id.split('.')
        return tuple(int(p) if p.isdigit() else p for p in parts)
    
    def update_code_references(self) -> Dict[str, List[str]]:
        """Update task references in code files"""
        logger.info("Updating task references in code...")
        
        updates = {}
        src_path = Path('src')
        
        if not src_path.exists():
            logger.warning("src directory not found")
            return updates
        
        for py_file in src_path.rglob('*.py'):
            try:
                content = py_file.read_text()
                original_content = content
                
                # Update EPIC references
                for epic_id, epic in self.epics.items():
                    # Update various formats
                    patterns = [
                        (r'Task:\s*EPIC[- ]?\d+', f'Task: {epic_id}'),
                        (r'#\s*EPIC[- ]?\d+', f'# {epic_id}'),
                    ]
                    
                    for pattern, replacement in patterns:
                        content = re.sub(pattern, replacement, content)
                
                # Update task references to use proper numbering
                for task_id, task in self.index.items():
                    if task.epic:
                        # Update references to use full path
                        patterns = [
                            (rf'Task:\s*{re.escape(task_id)}(?!\.\d)', f'Task: {task.epic}/{task_id}'),
                            (rf'#\s*{re.escape(task_id)}(?!\.\d)', f'# {task.epic}/{task_id}'),
                        ]
                        
                        for pattern, replacement in patterns:
                            content = re.sub(pattern, replacement, content)
                
                if content != original_content:
                    py_file.write_text(content)
                    updates[str(py_file)] = ["Updated task references to use hierarchical numbering"]
                    
            except Exception as e:
                logger.error(f"Error updating {py_file}: {e}")
        
        return updates
    
    def save_index_json(self) -> None:
        """Save task index as JSON for programmatic access"""
        logger.info("Saving task index as JSON...")
        
        index_data = {
            'generated': datetime.now().isoformat(),
            'statistics': {
                'total_tasks': len(self.tasks),
                'epics': len(self.epics),
                'completed': sum(1 for t in self.tasks.values() if t.status == 'completed'),
                'pending': sum(1 for t in self.tasks.values() if t.status == 'pending'),
            },
            'tasks': {task_id: task.to_dict() for task_id, task in self.index.items()},
            'epics': {epic_id: epic.to_dict() for epic_id, epic in self.epics.items()},
            'model_assignments': {
                model: [t.id for t in tasks]
                for model, tasks in self.model_assignments.items()
            },
            'complexity_distribution': {
                level: [t.id for t in tasks]
                for level, tasks in self.complexity_levels.items()
            }
        }
        
        with open('task_index.json', 'w') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"Task index saved to task_index.json")


def main():
    """Main execution function"""
    logger.info("=" * 60)
    logger.info("TASK NUMBERING SYSTEM IMPLEMENTATION")
    logger.info("=" * 60)
    
    # Initialize system
    numbering_system = TaskNumberingSystem()
    
    # Parse existing tasks
    logger.info("Step 1: Parsing existing task list...")
    numbering_system.parse_task_list()
    logger.info(f"  Found {len(numbering_system.tasks)} tasks in {len(numbering_system.epics)} EPICs")
    
    # Apply systematic numbering
    logger.info("Step 2: Applying hierarchical numbering...")
    numbering_system.apply_numbering()
    
    # Extract dependencies
    logger.info("Step 3: Extracting task dependencies...")
    numbering_system.extract_dependencies()
    
    # Find code references
    logger.info("Step 4: Finding task references in code...")
    numbering_system.find_code_references()
    
    # Generate numbered task list
    logger.info("Step 5: Generating numbered task list...")
    numbered_tasklist = numbering_system.generate_numbered_tasklist()
    
    # Save backup of original
    if Path('TASK_LIST.md').exists():
        backup_path = f"TASK_LIST.md.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        Path('TASK_LIST.md').rename(backup_path)
        logger.info(f"  Original backed up to {backup_path}")
    
    # Save new numbered task list
    with open('TASK_LIST_NUMBERED.md', 'w') as f:
        f.write(numbered_tasklist)
    logger.info("  Numbered task list saved to TASK_LIST_NUMBERED.md")
    
    # Generate task index
    logger.info("Step 6: Generating task index...")
    task_index = numbering_system.generate_task_index()
    with open('TASK_INDEX.md', 'w') as f:
        f.write(task_index)
    logger.info("  Task index saved to TASK_INDEX.md")
    
    # Save JSON index
    logger.info("Step 7: Saving JSON index...")
    numbering_system.save_index_json()
    
    # Update code references
    logger.info("Step 8: Updating code references...")
    updates = numbering_system.update_code_references()
    logger.info(f"  Updated {len(updates)} files with new task numbering")
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("TASK NUMBERING IMPLEMENTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Processed {len(numbering_system.tasks)} tasks")
    logger.info(f"✅ Organized into {len(numbering_system.epics)} EPICs")
    logger.info(f"✅ Found {sum(len(t.dependencies) for t in numbering_system.tasks.values())} dependencies")
    logger.info(f"✅ Updated {len(updates)} code files")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - TASK_LIST_NUMBERED.md: Updated task list with numbering")
    logger.info("  - TASK_INDEX.md: Quick reference index")
    logger.info("  - task_index.json: Programmatic access to task data")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review TASK_LIST_NUMBERED.md")
    logger.info("  2. If satisfied, rename to TASK_LIST.md")
    logger.info("  3. Commit changes to version control")


if __name__ == "__main__":
    main()