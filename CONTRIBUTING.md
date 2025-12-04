# Contributing to qlass

Thanks for your interest in contributing to qlass! This guide will help you get started.

## Quick Start
While you can use qlass distribution from PyPI, in order to contribute to qlass, you need to install the development version from Github to be sure that there are no breaking changes with respect to the current `main` version.  
1. **Fork and clone** the repository
2. **Install dependencies** using [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync --all-groups
   ```

## Development Workflow
We recommend the following development workflow, in order to do some checks locally and facilitate the code review. 
### Running Tests
```bash
uv run pytest
```

For coverage report:
```bash
uv run --group test pytest --cov
```

### Code Quality Checks

**Linting:**
```bash
uv run --group lint ruff check .
uv run --group lint ruff format .
```

**Type Checking:**
```bash
uv run --group type-check mypy src/qlass
```

### Before Submitting a PR

Make sure all checks pass:
```bash
# Format code
uv run --group lint ruff format .

# Check linting
uv run --group lint ruff check .

# Type check
uv run --group type-check mypy src/qlass

# Run tests
uv run --group test pytest --cov

# Run all example scripts
make check-examples
```
Running make check-examples might take some time as we are running all the example scripts.
## Code Standards

- **Python version**: 3.10+
- **Type hints**: Required for all functions (enforced by mypy)
- **Formatting**: Handled by ruff (100 character line length, double quotes)
- **Docstrings**: Recommended for public APIs

## Pull Request Process

1. Create a new branch for your changes
2. Make your changes and add tests if applicable
3. Ensure all checks pass locally
4. Push to your fork and submit a pull request
5. Wait for CI to pass and address any review feedback

## Questions?

Feel free to open [an issue](https://github.com/unitaryfoundation/qlass/issues/new) for questions or discussions.