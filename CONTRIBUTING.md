# Contributing to Titans

Thank you for your interest in contributing to the Titans implementation! This guide will help you get started with contributing to this project.

## 🎯 Ways to Contribute

- 🐛 **Bug Reports**: Report issues or unexpected behavior
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve documentation, tutorials, or examples
- 🔧 **Code Contributions**: Fix bugs, implement features, optimize performance
- 🧪 **Testing**: Add tests, improve test coverage, or test on new platforms
- 📊 **Benchmarks**: Contribute performance benchmarks or analysis
- 🔬 **Research**: Reproduce experiments, validate results, or extend research

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Git
- Basic understanding of JAX/Flax
- Familiarity with the Titans paper (recommended)

### Development Setup

1. **Fork and Clone**

   ```bash
   git clone https://github.com/mahdi-shafiei/titans-flax.git
   cd titans-flax
   ```

2. **Set up Development Environment**

   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows

   # Install in development mode
   pip install -e ".[dev,research]"

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**

   ```bash
   # Run quick tests
   pytest tests/test_titans.py -v

   # Run quick example
   python -c "import titans; titans.quick_start_example()"
   ```

## 📋 Development Workflow

### 1. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 2. Make Changes

Follow our coding standards:

- **Code Style**: Use Black for formatting, isort for imports
- **Type Hints**: Add type annotations for all functions
- **Documentation**: Write clear docstrings using Google style
- **Tests**: Add tests for new functionality

### 3. Test Your Changes

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run specific tests
pytest tests/test_specific_module.py -v

# Test with coverage
make test-coverage
```

### 4. Commit Changes

Use conventional commit messages:

```bash
# Examples of good commit messages
git commit -m "feat: add support for custom memory architectures"
git commit -m "fix: resolve memory leak in long sequence processing"
git commit -m "docs: update README with new examples"
git commit -m "test: add integration tests for MAC variant"
git commit -m "perf: optimize memory update mechanism"
```

### 5. Submit Pull Request

1. Push your branch to your fork
2. Create a Pull Request on GitHub
3. Fill out the PR template
4. Wait for review and address feedback

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── test_neural_memory.py      # Core memory tests
├── test_titans_variants.py    # Architecture tests
├── test_training.py          # Training utilities tests
├── test_integration.py       # Integration tests
├── conftest.py              # Pytest configuration
└── fixtures/                # Test data and fixtures
```

### Writing Tests

1. **Unit Tests**: Test individual components

   ```python
   def test_neural_memory_forward_pass(rngs, sample_input):
       memory = NeuralLongTermMemory(d_model=128, rngs=rngs)
       output, state = memory(sample_input)
       assert output.shape == sample_input.shape
       assert isinstance(state, MemoryState)
   ```

2. **Integration Tests**: Test complete workflows

   ```python
   @pytest.mark.integration
   def test_complete_training_pipeline():
       # Test full training workflow
       pass
   ```

3. **Property Tests**: Test mathematical properties

   ```python
   def test_memory_magnitude_increases_with_updates():
       # Test that memory magnitude grows with adaptation
       pass
   ```

### Test Guidelines

- Use descriptive test names
- Include edge cases and error conditions
- Mock external dependencies
- Use parametrized tests for multiple scenarios
- Add performance tests for critical paths

```python
@pytest.mark.parametrize("variant", ["mac", "mag", "mal", "lmm"])
def test_all_variants_produce_finite_output(variant, rngs):
    model = create_titans_model(variant, d_model=64, rngs=rngs)
    x = jnp.ones((2, 32, 64))
    output = model(x)
    assert jnp.isfinite(output).all()
```

## 📚 Documentation Guidelines

### Code Documentation

Use Google-style docstrings:

```python
def memory_update_step(
    self,
    keys: jax.Array,
    values: jax.Array,
    memory_state: MemoryState,
) -> MemoryState:
    """Update memory using gradient descent with momentum.

    This implements the core memory update mechanism from the Titans paper,
    using gradient descent with momentum and weight decay.

    Args:
        keys: Key tensor of shape (batch, seq_len, d_model)
        values: Value tensor of shape (batch, seq_len, d_model)
        memory_state: Current memory state

    Returns:
        Updated memory state

    Raises:
        ValueError: If input shapes are incompatible

    Example:
        >>> memory = NeuralLongTermMemory(d_model=128, rngs=rngs)
        >>> keys = jnp.ones((2, 16, 128))
        >>> values = jnp.ones((2, 16, 128))
        >>> new_state = memory.memory_update_step(keys, values, old_state)
    """
```

### README and Tutorials

- Use clear, concise language
- Include runnable code examples
- Add diagrams where helpful
- Keep examples up-to-date with API changes

## 🎨 Code Style

### Formatting

We use automated formatting tools:

```bash
# Format Python code
black titans/

# Sort imports
isort titans/

# Lint code
ruff titans/

# Type checking
mypy titans/
```

### Naming Conventions

- **Classes**: PascalCase (`TitansMAC`, `NeuralLongTermMemory`)
- **Functions/Variables**: snake_case (`create_titans_model`, `memory_state`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_MEMORY_DEPTH`)
- **Private**: Leading underscore (`_internal_function`)

### Code Organization

```python
# Standard library imports
import time
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import jax
import jax.numpy as jnp
from flax import nnx

# Local imports
from titans.neural_memory import MemoryState
from titans.config import TitansConfig
from titans.core import create_titans_model
from titans.benchmark import TitansBenchmark
```

## 🏗️ Architecture Guidelines

### Adding New Features

1. **Design Discussion**: Open an issue to discuss design
2. **API Consistency**: Follow existing patterns
3. **Documentation**: Document design decisions
4. **Testing**: Full test coverage
5. **Examples**: Provide usage examples

### Performance Considerations

- Use JAX transformations (jit, vmap, etc.) appropriately
- Profile memory usage for large models
- Consider numerical stability
- Benchmark performance impact

### Memory Management

- Follow JAX best practices for memory
- Use gradient checkpointing for large models
- Implement memory-efficient versions when needed
- Document memory requirements

## 🐛 Bug Reports

### Before Reporting

1. Search existing issues
2. Try latest version
3. Create minimal reproduction case
4. Check documentation

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce the behavior:
1. Create model with config X
2. Run code snippet Y
3. See error Z

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.16]
- JAX version: [e.g., 0.4.20]
- Titans version: [e.g., 0.1.0]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Implementation**
High-level description of how this could be implemented

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 🔬 Research Contributions

### Paper Reproduction

- Validate experimental results
- Test on different hardware/settings
- Document any discrepancies
- Contribute fixes for reproduction issues

### Extensions

- Implement paper variations or improvements
- Add support for new tasks/domains
- Optimize existing implementations
- Contribute novel architectures

### Benchmarking

- Add new benchmark tasks
- Compare with other architectures
- Optimize performance
- Document best practices

## 📊 Performance Guidelines

### Benchmarking

Always benchmark performance changes:

```python
# Before your changes
from titans.benchmark import TitansBenchmark
benchmark = TitansBenchmark()
baseline_results = benchmark.compare_variants()

# After your changes
optimized_results = benchmark.compare_variants()

# Compare and document improvements
```

### Memory Usage

Monitor memory usage for large models:

```python
from titans.optimization_utilities import MemoryOptimizer

profile = MemoryOptimizer.estimate_memory_usage(
    model, batch_size=4, seq_len=1024
)
print(profile)
```

## 📅 Release Process

### Version Numbering

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Run full test suite
4. Update documentation
5. Create release PR
6. Tag release
7. Deploy to PyPI

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Acknowledge contributions

### Communication

- **Issues**: Bug reports, feature requests
- **Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions
- **Email**: <contact@titans-implementation.com> for private matters

### Recognition

We recognize contributors in:

- README acknowledgments
- Release notes
- Contributor hall of fame
- Conference presentations (when appropriate)

## 📖 Resources

### Learning Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Guide](https://flax.readthedocs.io/en/latest/guides/index.html)
- [Original Titans Paper](https://arxiv.org/abs/2501.00663)
- [Project Documentation](https://titans-flax.readthedocs.io/)

### Development Tools

- **IDE Setup**: VS Code with Python/JAX extensions
- **Debugging**: JAX debugging guide
- **Profiling**: JAX profiling tools
- **Testing**: pytest documentation

## 🎉 Getting Help

### Where to Ask Questions

1. **GitHub Discussions**: General questions and ideas
2. **Issues**: Bug reports and feature requests
3. **Email**: <contact@titans-implementation.com>
4. **Community**: Join our community channels

### What to Include

- Clear problem description
- Minimal code example
- Environment details
- What you've already tried

## 🏆 Contributor Recognition

### Types of Recognition

- **Code Contributors**: Listed in README and releases
- **Issue Reporters**: Acknowledged in bug fix commits
- **Documentation**: Recognized in docs and README
- **Research**: Co-authorship consideration for significant contributions

### Levels of Contribution

- **First-time Contributors**: Welcome package and guidance
- **Regular Contributors**: Added to contributors team
- **Core Contributors**: Repository access and decision input
- **Maintainers**: Full project responsibilities

Thank you for contributing to Titans! Your contributions help advance the state of neural memory research and make these powerful models accessible to everyone.

---

## Quick Commands

```bash
# Setup development environment
make setup

# Run all checks
make check

# Run tests
make test

# Format code
make format

# Build documentation
make docs

# Clean up
make clean
```

For more details, see our [Development Guide](docs/development.md) and [API Reference](docs/api/).

## Shell Command Best Practices

When running commands through AI tools or automated systems, follow these practices:

### Package Installation Commands

Always quote package version specifiers to prevent shell parsing issues:

```bash
# ❌ Avoid - may cause redirection artifacts in some environments
pip install package>=1.0.0

# ✅ Correct - prevents version specifier parsing issues
pip install 'package>=1.0.0'
pip install "package>=1.0.0"
```

### Why This Matters

Some AI tools and automated command execution systems may incorrectly parse version specifiers like `>=1.0.0` as shell redirection operators (`> =1.0.0`), creating unwanted artifact files. Proper quoting prevents this issue.
