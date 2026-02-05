# Contributing to PhyNEO-Electrolyte

Thank you for your interest in contributing to PhyNEO-Electrolyte! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/phyneo-electrolyte.git
   cd phyneo-electrolyte
   ```
3. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

### Formatting

We use `black` for code formatting:

```bash
black phyneo/
```

### Linting

We use `flake8` for linting:

```bash
flake8 phyneo/ --max-line-length=100
```

## Testing

### Running Tests

```bash
pytest tests/
```

### Writing Tests

- Add tests for new features in the `tests/` directory
- Ensure all tests pass before submitting a PR
- Aim for high test coverage

Example test structure:

```python
import pytest
from phyneo.models import SlaterTypeFunction

def test_slater_function_forward():
    model = SlaterTypeFunction(n_orbitals=16)
    distances = torch.randn(2, 10).abs()
    output = model(distances)
    assert output.shape == (2, 10, 16)
```

## Documentation

- Update documentation for any new features
- Use clear, concise language
- Include code examples where appropriate
- Update the API reference if adding new functions/classes

## Commit Messages

Follow these conventions:

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests where appropriate

Examples:
```
Add Slater-type orbital cutoff function

Fix normalization in pairwise correction model

Update training documentation with new examples
```

## Pull Request Process

1. Update documentation to reflect changes
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Submit a pull request with a clear description

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

## Areas for Contribution

### High Priority

- Additional force field models
- Performance optimizations
- More comprehensive tests
- Documentation improvements
- Example notebooks

### New Features

- Additional training utilities
- Visualization tools
- Analysis functions
- Data preprocessing tools
- Integration with other MD packages

### Bug Fixes

- Check the Issues page for open bugs
- Report new bugs with detailed descriptions

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Maintain a professional environment

## Questions?

- Open an issue for questions
- Check existing documentation first
- Be clear and specific in your questions

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to PhyNEO-Electrolyte!
