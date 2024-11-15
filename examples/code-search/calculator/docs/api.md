# Calculator API Reference

## Calculator Class

The `Calculator` class provides basic arithmetic operations and memory functionality.

### Basic Operations

#### `add(a: float, b: float) -> float`

Add two numbers together.

**Parameters:**
- `a`: First number
- `b`: Second number

**Returns:**
- Sum of the two numbers

**Example:**
```python
calc = Calculator()
result = calc.add(5, 3)  # Returns 8
```

#### `subtract(a: float, b: float) -> float`

Subtract the second number from the first.

**Parameters:**
- `a`: Number to subtract from
- `b`: Number to subtract

**Returns:**
- Difference between the numbers

**Example:**
```python
result = calc.subtract(5, 3)  # Returns 2
```

#### `multiply(a: float, b: float) -> float`

Multiply two numbers.

**Parameters:**
- `a`: First number
- `b`: Second number

**Returns:**
- Product of the numbers

**Example:**
```python
result = calc.multiply(4, 3)  # Returns 12
```

#### `divide(a: float, b: float) -> float`

Divide the first number by the second.

**Parameters:**
- `a`: Number to divide
- `b`: Number to divide by

**Returns:**
- Quotient of the division

**Raises:**
- `ZeroDivisionError`: If `b` is zero

**Example:**
```python
result = calc.divide(6, 2)  # Returns 3.0
```

### Memory Operations

#### `store_memory(value: float) -> None`

Store a value in calculator memory.

**Parameters:**
- `value`: Value to store

**Example:**
```python
calc.store_memory(42)
```

#### `recall_memory() -> float`

Recall the value from calculator memory.

**Returns:**
- Value stored in memory

**Example:**
```python
value = calc.recall_memory()  # Returns stored value
```

#### `add_from_memory(value: float) -> float`

Add a value to the stored memory value.

**Parameters:**
- `value`: Value to add to memory

**Returns:**
- Sum of memory and value

**Example:**
```python
calc.store_memory(10)
result = calc.add_from_memory(5)  # Returns 15.0
```

## Error Handling

The calculator methods may raise the following exceptions:

- `ZeroDivisionError`: Raised when attempting to divide by zero
