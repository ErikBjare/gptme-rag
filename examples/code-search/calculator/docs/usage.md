# Calculator Library Usage Guide

The Calculator library provides a simple yet powerful calculator implementation with memory functionality.

## Basic Usage

```python
from calculator import Calculator

# Create a calculator instance
calc = Calculator()

# Basic arithmetic
result = calc.add(5, 3)      # 8
result = calc.subtract(5, 3)  # 2
result = calc.multiply(4, 3)  # 12
result = calc.divide(6, 2)    # 3.0
```

## Memory Functions

The calculator includes memory functionality for storing and recalling values:

```python
# Store a value in memory
calc.store_memory(42)

# Recall the stored value
value = calc.recall_memory()  # 42.0

# Add a value to memory
result = calc.add_from_memory(8)  # 50.0
```

## Error Handling

The calculator handles errors gracefully:

```python
try:
    result = calc.divide(5, 0)
except ZeroDivisionError as e:
    print(f"Error: {e}")  # Error: Cannot divide by zero
```

## Best Practices

1. Create a new calculator instance for each independent calculation session
2. Use memory functions to store intermediate results
3. Always handle potential errors, especially for division
