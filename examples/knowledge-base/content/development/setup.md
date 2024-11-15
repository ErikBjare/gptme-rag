# Development Environment Setup

This guide covers setting up your development environment for the project.

## Prerequisites

- Python 3.10 or later
- Git
- A code editor (VS Code recommended)

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/example/project.git
   cd project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Development Settings**
   - Copy `.env.example` to `.env`
   - Update settings for your environment
   - Set `DEBUG=True` for development

## Editor Setup

### VS Code

1. Install recommended extensions:
   - Python
   - GitLens
   - Python Test Explorer

2. Configure settings:
   ```json
   {
     "python.linting.enabled": true,
     "python.formatting.provider": "black"
   }
   ```

### PyCharm

1. Open project directory
2. Select the virtual environment as the project interpreter
3. Enable auto-formatting with Black

## Verification

Run the following to verify your setup:

```bash
# Run tests
pytest

# Start development server
python manage.py runserver

# Run linting
flake8
```

## Troubleshooting

### Common Issues

1. **Virtual Environment Not Activated**
   - Check for `(venv)` in your terminal prompt
   - Run activation command again

2. **Dependencies Missing**
   - Ensure venv is activated
   - Run `pip install -r requirements.txt` again

3. **Port Already in Use**
   - Check for other running servers
   - Use different port: `python manage.py runserver 8001`

## Next Steps

- Read the [Development Workflow](workflow.md) guide
- Review [Coding Standards](../best-practices/coding-style.md)
- Set up [Testing Environment](../best-practices/testing.md)
