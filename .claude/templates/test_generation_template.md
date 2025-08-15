# Test Generation Template for Sonnet 4

## Input Format
```yaml
function: [function_name]
file: [path/to/file.py]
parameters: [list of params with types]
return_type: [type]
dependencies: [external dependencies]
```

## Test Structure
```python
import pytest
from [module] import [function]

class Test[FunctionName]:
    """Tests for {function_name}"""
    
    def test_normal_case(self):
        # Arrange (5 lines max)
        # Act (1 line)
        # Assert (2 lines max)
        pass
    
    def test_edge_case(self):
        # Edge case (5 lines max)
        pass
    
    def test_error_case(self):
        # Error handling (5 lines max)
        with pytest.raises(ExpectedException):
            # Trigger error
            pass
```

## Constraints
- Max 30 lines per test class
- Use pytest fixtures
- Mock external dependencies
- Cover: normal, edge, error cases
- No explanatory comments

## Quick Patterns
```python
# For async functions
@pytest.mark.asyncio
async def test_async_function():
    result = await function()
    assert result == expected

# For parameterized tests
@pytest.mark.parametrize("input,expected", [
    (val1, result1),
    (val2, result2),
])
def test_parametrized(input, expected):
    assert function(input) == expected
```