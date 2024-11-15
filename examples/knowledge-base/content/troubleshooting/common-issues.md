# Common Issues and Solutions

This guide covers frequently encountered issues and their solutions.

## Development Environment

### Virtual Environment Issues

#### "Command not found: python"
- **Cause**: Python not installed or not in PATH
- **Solution**:
  1. Install Python from python.org
  2. Add Python to system PATH
  3. Verify with `python --version`

#### "pip not found"
- **Cause**: pip not installed or wrong environment active
- **Solution**:
  1. Ensure virtual environment is activated
  2. Run `python -m ensurepip`
  3. Upgrade pip: `python -m pip install --upgrade pip`

#### "Requirements installation fails"
- **Cause**: Missing system dependencies or conflicts
- **Solution**:
  1. Update pip: `pip install --upgrade pip`
  2. Install build tools:
     - Windows: Install Visual C++ Build Tools
     - Linux: `sudo apt install python3-dev build-essential`
  3. Try installing requirements one by one

## Git Issues

### "Cannot push to repository"

#### Permission Denied
- **Cause**: SSH key not set up or wrong permissions
- **Solution**:
  1. Check SSH key: `ssh -T git@github.com`
  2. Generate new key if needed:
     ```bash
     ssh-keygen -t ed25519
     cat ~/.ssh/id_ed25519.pub  # Add to GitHub
     ```

#### Branch Protection
- **Cause**: Direct push to protected branch
- **Solution**:
  1. Create feature branch
  2. Submit pull request
  3. Get required approvals

## Testing Issues

### "Tests failing unexpectedly"

#### Database Tests Fail
- **Cause**: Test database not set up correctly
- **Solution**:
  1. Check test database configuration
  2. Reset test database:
     ```bash
     python manage.py flush --database=test
     python manage.py migrate --database=test
     ```

#### Random Test Failures
- **Cause**: Race conditions or resource cleanup
- **Solution**:
  1. Add proper test isolation
  2. Clean up resources in tearDown
  3. Run tests with `pytest -v` for details

## Deployment Issues

### "Application not starting"

#### Port Already in Use
- **Cause**: Another process using required port
- **Solution**:
  1. Find process: `lsof -i :8000`
  2. Kill process or use different port
  3. Restart application

#### Missing Environment Variables
- **Cause**: Environment not configured
- **Solution**:
  1. Check `.env` file exists
  2. Verify all required variables set
  3. Restart application to load new variables

## Performance Issues

### "Application running slowly"

#### High Memory Usage
- **Cause**: Memory leaks or inefficient caching
- **Solution**:
  1. Monitor with `top` or Task Manager
  2. Profile application
  3. Review and optimize caching

#### Slow Database Queries
- **Cause**: Missing indexes or N+1 queries
- **Solution**:
  1. Enable query logging
  2. Add necessary indexes
  3. Use select_related/prefetch_related

## Related Guides

- [Development Setup](../development/setup.md)
- [Debugging Guide](debugging.md)
- [Error Reference](errors.md)
