# MANDATORY WORKFLOW PROTOCOL - EFFECTIVE IMMEDIATELY
**Date**: January 14, 2025
**Status**: ENFORCED
**Owner**: Alex (Team Lead)

## 🎯 CRITICAL DIRECTIVE: Keep Documentation in Context Window

### MANDATORY WORKFLOW (NO EXCEPTIONS)

#### 1. EVERY Grooming Session MUST START WITH:
```
1. Read ARCHITECTURE.md (current state)
2. Read PROJECT_MANAGEMENT_TASK_LIST_V4.md (current progress) 
3. Announce current state to team
4. Verify everyone understands context
5. THEN begin grooming
```

#### 2. EVERY Completed Subtask MUST END WITH:
```
1. Update ARCHITECTURE.md immediately
2. Update PROJECT_MANAGEMENT_TASK_LIST_V4.md immediately
3. Verify updates are in context window
4. Announce completion to team
5. THEN move to next task
```

### WHY THIS MATTERS
- **Context Window Management**: Keeps crucial info always accessible
- **No Lost Work**: Documentation stays synchronized
- **Team Alignment**: Everyone knows current state
- **Audit Trail**: Complete record of progress

### ENFORCEMENT
- **Blocker**: Cannot start grooming without reading docs
- **Blocker**: Cannot mark complete without updating docs
- **Verification**: Alex checks every session
- **Violation**: Work doesn't count if docs not updated

### WORKFLOW EXAMPLE

```python
def grooming_session(task):
    # MANDATORY START
    architecture = read("ARCHITECTURE.md")
    project_plan = read("PROJECT_MANAGEMENT_TASK_LIST_V4.md")
    
    announce(f"Current state: {architecture.summary}")
    announce(f"Progress: {project_plan.progress}")
    
    # NOW we can groom
    subtasks = groom_task(task)
    
    return subtasks

def complete_subtask(subtask):
    # Do the work
    result = implement(subtask)
    
    # MANDATORY END
    update("ARCHITECTURE.md", result.components)
    update("PROJECT_MANAGEMENT_TASK_LIST_V4.md", result.progress)
    
    verify_in_context_window()
    announce(f"Completed: {subtask} - Docs updated")
    
    return result
```

### CHECKLIST FOR EVERY SESSION

#### Grooming Start:
- [ ] Read ARCHITECTURE.md
- [ ] Read PROJECT_MANAGEMENT_TASK_LIST_V4.md  
- [ ] Announce current state
- [ ] Verify team understanding
- [ ] Begin grooming

#### Subtask Completion:
- [ ] Complete implementation
- [ ] Update ARCHITECTURE.md
- [ ] Update PROJECT_MANAGEMENT_TASK_LIST_V4.md
- [ ] Verify in context window
- [ ] Announce completion

### IMMEDIATE ACTIONS
1. This protocol is effective NOW
2. All team members must acknowledge
3. Next grooming session follows this format
4. All completions follow this format

---

**Remember**: Documentation in context window = Never lose progress!