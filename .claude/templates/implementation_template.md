# Implementation Template for Sonnet 4

## Input Format
```yaml
task: [specific task description]
location: [file:line_number]
dependencies: [required imports]
pattern: [singleton|factory|observer|strategy]
constraints: [performance|memory|security]
```

## Implementation Structure

### 1. Imports (Minimal)
```python
from typing import [only needed types]
import [only required modules]
```

### 2. Implementation (Focused)
```python
def function_name(params) -> return_type:
    """One-line docstring"""
    # Direct implementation
    # No verbose comments
    # Use established patterns
    return result
```

### 3. Error Handling (If Required)
```python
try:
    # Main logic
except SpecificError as e:
    logger.error(f"Context: {e}")
    raise
```

## Code Patterns Library

### Pattern 1: Data Validation
```python
def validate_input(data: Dict) -> bool:
    required = ['field1', 'field2']
    return all(k in data for k in required)
```

### Pattern 2: Caching
```python
@cached(ttl=300)
async def expensive_operation(param):
    return await compute(param)
```

### Pattern 3: Rate Limiting
```python
@rate_limit(calls=10, period=60)
async def api_call(endpoint):
    return await fetch(endpoint)
```

## Response Constraints
- Max 50 lines of code
- No explanatory paragraphs
- Use type hints always
- Follow existing patterns
- Single responsibility

## Example Response
```python
# File: src/core/calculator.py, Line: 150
async def calculate_atr(ohlcv: pd.DataFrame) -> float:
    """Calculate ATR using ta library"""
    if len(ohlcv) < 14:
        raise ValueError("Insufficient data")
    
    atr = ta.volatility.AverageTrueRange(
        ohlcv['high'], 
        ohlcv['low'], 
        ohlcv['close']
    )
    return atr.average_true_range().iloc[-1]
```