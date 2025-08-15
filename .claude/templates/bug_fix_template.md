# Bug Fix Template for Sonnet 4

## Input Format
```yaml
file: [path/to/file.py]
line_range: [start-end]
issue_type: [fake_implementation|logic_error|syntax|performance]
description: [brief description]
```

## Required Analysis
1. **Identify Issue** (10 tokens)
   - Exact line number
   - Root cause
   - Impact assessment

2. **Proposed Fix** (50 tokens)
   ```python
   # Show only the changed code
   ```

3. **Validation** (20 tokens)
   - Test to verify fix
   - Expected outcome

## Response Constraints
- Max 100 tokens
- Single file focus
- No explanatory text unless critical
- Code-first approach

## Example Response
```markdown
Issue: Line 443 - Fake ATR calculation
Fix:
```python
# Replace line 443
atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range().iloc[-1]
```
Test: `assert atr > 0 and atr != price * 0.02`
```