# Testing Best Practices

This guide outlines our testing standards and best practices.

## Testing Philosophy

### Key Principles

1. **Test Early, Test Often**
   - Write tests during development, not after
   - Run tests frequently during development
   - Fix failing tests immediately

2. **Test Pyramid**
   - Many unit tests (base)
   - Fewer integration tests (middle)
   - Minimal end-to-end tests (top)

3. **Test Coverage**
   - Aim for 80%+ coverage
   - Focus on critical paths
   - Don't chase 100% blindly

## Types of Tests

### Unit Tests

Test individual components in isolation.

```python
def test_calculator_add():
    calc = Calculator()
    result = calc.add(2, 3)
    assert result == 5
```

Best practices:
- Test one thing per test
- Use descriptive test names
- Follow Arrange-Act-Assert pattern
- Mock external dependencies

### Integration Tests

Test component interactions.

```python
def test_user_registration_flow():
    # Test user registration end-to-end
    user_service = UserService(db_session)
    email_service = EmailService()
    
    user = user_service.register("test@example.com", "password123")
    assert user.is_active == False
    assert email_service.was_verification_sent(user.email)
```

Best practices:
- Use test databases
- Clean up test data
- Test realistic scenarios
- Consider error cases

### End-to-End Tests

Test complete user workflows.

```python
def test_user_login_workflow():
    driver = webdriver.Chrome()
    driver.get(BASE_URL)
    
    # Login flow
    driver.find_element_by_id("email").send_keys("user@example.com")
    driver.find_element_by_id("password").send_keys("password123")
    driver.find_element_by_id("login-button").click()
    
    assert "Dashboard" in driver.title
```

Best practices:
- Test critical user paths
- Use stable selectors
- Handle timing issues
- Clean up test data

## Test Organization

### Directory Structure

```plaintext
tests/
├── unit/
│   ├── test_models.py
│   └── test_services.py
├── integration/
│   └── test_workflows.py
├── e2e/
│   └── test_user_journeys.py
└── conftest.py
```

### Fixtures

Use fixtures for common setup:

```python
@pytest.fixture
def test_database():
    """Provide clean test database."""
    db = Database("test")
    db.migrate()
    yield db
    db.cleanup()

@pytest.fixture
def authenticated_client(test_database):
    """Provide authenticated client."""
    client = TestClient()
    client.login(user="test", password="test")
    return client
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific tests
pytest tests/unit/
pytest tests/test_file.py::test_function

# Run marked tests
pytest -m "slow"
```

### CI Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest --cov=app
```

## Common Patterns

### Testing Exceptions

```python
def test_division_by_zero():
    calc = Calculator()
    with pytest.raises(ZeroDivisionError):
        calc.divide(1, 0)
```

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result == expected_value
```

### Parameterized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert square(input) == expected
```

## Best Practices Checklist

- [ ] Tests are independent and isolated
- [ ] Tests are deterministic
- [ ] Test names are descriptive
- [ ] Tests focus on behavior, not implementation
- [ ] Tests are maintainable
- [ ] Coverage is monitored
- [ ] CI pipeline includes tests
- [ ] Test data is cleaned up

## Related Guides

- [Development Workflow](../development/workflow.md)
- [Debugging Guide](../troubleshooting/debugging.md)
- [Coding Standards](coding-style.md)
