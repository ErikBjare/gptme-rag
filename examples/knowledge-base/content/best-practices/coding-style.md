# Coding Style Guide

This guide outlines our coding standards and style conventions.

## Python Style Guide

### Code Formatting

We use Black for code formatting with these guidelines:

- Line length: 88 characters
- Use double quotes for strings
- Include trailing commas in multi-line structures

```python
# Good
def long_function_name(
    arg1: str,
    arg2: int,
    arg3: List[str],
) -> Dict[str, Any]:
    return {
        "arg1": arg1,
        "arg2": arg2,
        "items": arg3,
    }

# Bad
def long_function_name(arg1: str, arg2: int, arg3: List[str]) -> Dict[str, Any]:
    return {'arg1':arg1,'arg2':arg2,'items':arg3}
```

### Naming Conventions

1. **Variables and Functions**
   - Use snake_case
   - Be descriptive
   ```python
   # Good
   user_count = get_active_users()

   # Bad
   n = get_users()
   ```

2. **Classes**
   - Use PascalCase
   - Noun phrases
   ```python
   # Good
   class UserAuthentication:

   # Bad
   class authenticate_user:
   ```

3. **Constants**
   - Use UPPER_CASE
   ```python
   # Good
   MAX_CONNECTIONS = 100

   # Bad
   maximumConnections = 100
   ```

### Type Hints

Use type hints consistently:

```python
from typing import List, Optional, Dict

def process_items(
    items: List[str],
    config: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Process a list of items with optional configuration.

    Args:
        items: Items to process
        config: Optional configuration

    Returns:
        Processed items
    """
    return [item.upper() for item in items]
```

### Docstrings

Use Google-style docstrings:

```python
def complex_function(
    param1: str,
    param2: int,
) -> bool:
    """Short description of function.

    Longer description if needed, explaining complex
    behavior or important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: Description of when this is raised

    Example:
        >>> complex_function("test", 42)
        True
    """
    pass
```

## Git Conventions

### Commit Messages

Follow Conventional Commits format:

```plaintext
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- style: Code style changes
- refactor: Code refactoring
- test: Adding tests
- chore: Maintenance tasks

Examples:
```plaintext
feat(auth): add OAuth2 support
fix(api): handle null response from service
docs(readme): update installation instructions
```

### Branch Names

Format: `<type>/<description>`

Examples:
```plaintext
feature/user-authentication
bugfix/login-error
docs/api-documentation
```

## Project Structure

### Python Projects

```plaintext
project_name/
├── src/
│   └── project_name/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       └── utils/
├── tests/
│   ├── __init__.py
│   ├── test_main.py
│   └── conftest.py
├── docs/
├── README.md
├── pyproject.toml
└── .gitignore
```

### Module Organization

- One class per file (usually)
- Related utilities in modules
- Clear separation of concerns

## Code Quality Tools

### Required Tools

1. **Black**
   ```bash
   black .
   ```

2. **Flake8**
   ```bash
   flake8 src tests
   ```

3. **MyPy**
   ```bash
   mypy src
   ```

4. **isort**
   ```bash
   isort .
   ```

### Pre-commit Hooks

Use `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
```

## Best Practices

1. **DRY (Don't Repeat Yourself)**
   ```python
   # Good
   def get_user_data(user_id: int) -> Dict[str, Any]:
       return fetch_user(user_id)

   # Bad
   def get_user_name(user_id: int) -> str:
       return fetch_user(user_id)["name"]

   def get_user_email(user_id: int) -> str:
       return fetch_user(user_id)["email"]
   ```

2. **SOLID Principles**
   - Single Responsibility
   - Open/Closed
   - Liskov Substitution
   - Interface Segregation
   - Dependency Inversion

3. **Error Handling**
   ```python
   # Good
   try:
       process_data(data)
   except ValidationError as e:
       logger.error(f"Invalid data: {e}")
       raise
   except ProcessingError as e:
       logger.error(f"Processing failed: {e}")
       return None
   ```

## Related Guides

- [Testing Guide](testing.md)
- [Development Workflow](../development/workflow.md)
- [Documentation Guide](documentation.md)
