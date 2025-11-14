# Contributing to XPLIA

First off, thank you for considering contributing to XPLIA! It's people like you that make XPLIA such a great tool for the AI explainability community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by the [XPLIA Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/xplia.git
   cd xplia
   ```

3. **Create a development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,full]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples**
- **Describe the behavior you observed and what you expected**
- **Include screenshots if relevant**
- **Include your environment details** (OS, Python version, XPLIA version)

**Bug Report Template**:
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g., Ubuntu 20.04]
 - Python: [e.g., 3.10]
 - XPLIA: [e.g., 1.0.0]

**Additional context**
Any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List examples** of how the enhancement would be used

### Your First Code Contribution

Unsure where to begin? Look for issues tagged with:
- `good first issue` - Simple issues for beginners
- `help wanted` - Issues where we need community help

### Pull Requests

1. **Create a branch** from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the [coding standards](#coding-standards)
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: Add amazing feature"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `test:` - Adding or updating tests
   - `refactor:` - Code refactoring
   - `perf:` - Performance improvements
   - `chore:` - Maintenance tasks

4. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/nicolasseverino/xplia.git
cd xplia

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev,full]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/ -v
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=xplia --cov-report=html

# Run specific test file
pytest tests/test_explainers.py -v

# Run tests matching pattern
pytest tests/ -k "test_shap" -v

# Run benchmarks
pytest tests/benchmarks/ -m benchmark
```

### Code Quality Checks

```bash
# Linting
flake8 xplia

# Type checking
mypy xplia --ignore-missing-imports

# Security scanning
bandit -r xplia

# Code formatting
black xplia tests
isort xplia tests
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use Black for code formatting (line length: 100)
- Use isort for import sorting
- Maximum function complexity: 10 (enforced by flake8)

### Naming Conventions

```python
# Classes: PascalCase
class MyExplainer(ExplainerBase):
    pass

# Functions and variables: snake_case
def calculate_feature_importance():
    feature_names = []

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 100

# Private methods: _leading_underscore
def _internal_helper():
    pass
```

### Docstring Format

Use NumPy-style docstrings:

```python
def explain(self, X, method='shap'):
    """
    Generate explanations for model predictions.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input instances to explain.

    method : str, default='shap'
        Explanation method to use.

    Returns
    -------
    ExplanationResult
        Structured explanation with metadata and quality metrics.

    Raises
    ------
    ValueError
        If X has invalid shape or contains NaN values.

    Examples
    --------
    >>> explainer = MyExplainer(model)
    >>> result = explainer.explain(X_test[:5])
    >>> print(result.feature_importance)
    """
```

## Testing Guidelines

### Writing Tests

- **One test per functionality**
- **Clear test names**: `test_shap_returns_correct_shape`
- **Use fixtures** for common setup
- **Test edge cases**: empty input, single sample, NaN values, etc.

```python
import pytest
import numpy as np

class TestMyExplainer:
    @pytest.fixture
    def sample_data(self):
        return np.random.randn(100, 10)

    def test_explain_returns_correct_shape(self, sample_data):
        explainer = MyExplainer(model)
        result = explainer.explain(sample_data[:5])
        assert result.feature_importance.shape == (10,)

    def test_explain_handles_empty_input(self):
        explainer = MyExplainer(model)
        with pytest.raises(ValueError):
            explainer.explain(np.array([]))
```

### Test Coverage Requirements

- **Minimum coverage**: 50% for new code
- **Target coverage**: 80% overall
- All public APIs must have tests
- Critical paths must have edge case tests

## Documentation

### Types of Documentation

1. **Code Comments**: Explain *why*, not *what*
2. **Docstrings**: API documentation (see above)
3. **User Guides**: `docs/` directory
4. **Examples**: `examples/` directory

### Documentation Updates

When adding features:
- Update relevant markdown files in `docs/`
- Add examples to `examples/`
- Update `README.md` if it affects main features
- Update `CHANGELOG.md`

### Building Documentation

```bash
cd docs
make html
open _build/html/index.html
```

## Pull Request Process

1. **Update documentation** for any new features
2. **Add tests** with minimum 50% coverage for new code
3. **Update CHANGELOG.md** under "Unreleased" section
4. **Ensure all tests pass**: `pytest tests/ -v`
5. **Ensure code quality**: `flake8 xplia` and `mypy xplia`
6. **Request review** from maintainers

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Updated CHANGELOG.md
- [ ] Code follows style guidelines
- [ ] No new warnings

## Related Issues
Fixes #(issue number)
```

## Release Process

(For maintainers only)

1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create release branch: `git checkout -b release/vX.Y.Z`
4. Run full test suite
5. Create GitHub release
6. CI/CD will publish to PyPI automatically

## Questions?

- **GitHub Discussions**: https://github.com/nicolasseverino/xplia/discussions
- **Discord**: [Join our server]
- **Email**: contact@xplia.com

## Recognition

Contributors will be:
- Listed in `CONTRIBUTORS.md`
- Mentioned in release notes
- Credited in relevant documentation

Thank you for contributing to XPLIA! 🎉
