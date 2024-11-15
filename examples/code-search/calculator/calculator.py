"""A simple calculator library demonstrating documentation search."""

class Calculator:
    """A basic calculator with memory functionality.
    
    This calculator supports basic arithmetic operations and includes
    a memory feature for storing and recalling values.
    
    Examples:
        >>> calc = Calculator()
        >>> calc.add(5, 3)
        8
        >>> calc.store_memory(10)
        >>> calc.add_from_memory(5)
        15
    """
    
    def __init__(self):
        """Initialize the calculator with empty memory."""
        self._memory = 0.0
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of the numbers
            
        Examples:
            >>> calc = Calculator()
            >>> calc.add(2, 3)
            5
        """
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract second number from first.
        
        Args:
            a: Number to subtract from
            b: Number to subtract
            
        Returns:
            Difference between the numbers
            
        Examples:
            >>> calc = Calculator()
            >>> calc.subtract(5, 3)
            2
        """
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of the numbers
            
        Examples:
            >>> calc = Calculator()
            >>> calc.multiply(4, 3)
            12
        """
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """Divide first number by second.
        
        Args:
            a: Number to divide
            b: Number to divide by
            
        Returns:
            Quotient of the division
            
        Raises:
            ZeroDivisionError: If b is zero
            
        Examples:
            >>> calc = Calculator()
            >>> calc.divide(6, 2)
            3.0
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
    
    def store_memory(self, value: float) -> None:
        """Store a value in calculator memory.
        
        Args:
            value: Value to store
            
        Examples:
            >>> calc = Calculator()
            >>> calc.store_memory(42)
            >>> calc.recall_memory()
            42.0
        """
        self._memory = value
    
    def recall_memory(self) -> float:
        """Recall the value from calculator memory.
        
        Returns:
            Value stored in memory
            
        Examples:
            >>> calc = Calculator()
            >>> calc.store_memory(42)
            >>> calc.recall_memory()
            42.0
        """
        return self._memory
    
    def add_from_memory(self, value: float) -> float:
        """Add a value to the stored memory value.
        
        Args:
            value: Value to add to memory
            
        Returns:
            Sum of memory and value
            
        Examples:
            >>> calc = Calculator()
            >>> calc.store_memory(10)
            >>> calc.add_from_memory(5)
            15.0
        """
        return self.add(self._memory, value)
