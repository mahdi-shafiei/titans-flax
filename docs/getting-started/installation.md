# Installation

This guide will help you install Titans and its dependencies for various use cases.

## Requirements

- Python 3.9 or higher
- JAX 0.4.20 or higher
- GPU support (recommended for large models)

## Basic Installation

### From PyPI (Recommended)

```bash
pip install titans-flax
```

### From Source

```bash
git clone https://github.com/mahdi-shafiei/titans-flax.git
cd titans-flax
pip install -e .
```

## Installation Options

### Development Installation

For contributing to the project or running tests:

```bash
git clone https://github.com/mahdi-shafiei/titans-flax.git
cd titans-flax
pip install -e ".[dev]"
pre-commit install  # Optional: for code quality checks
```

### Research Installation

For research and experimentation with additional tools:

```bash
pip install "titans-flax[research]"
```

Includes:
- Jupyter notebooks
- TensorBoard
- Additional visualization tools

### Documentation Installation

To build and serve documentation locally:

```bash
pip install "titans-flax[docs]"
```

### GPU Support

#### CUDA (NVIDIA GPUs)

```bash
pip install "titans-flax[gpu]"
# or
pip install jax[cuda12_pip] titans-flax
```

#### TPU Support

```bash
pip install "titans-flax[tpu]"
# or
pip install jax[tpu] titans-flax
```

### Complete Installation

Install all optional dependencies:

```bash
pip install "titans-flax[all]"
```

## Verify Installation

After installation, verify everything works:

```python
import jax
import titans

# Check JAX configuration
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

# Check Titans installation
from titans import create_titans_model
model = create_titans_model(variant="mac", d_model=64, seed=42)
print("✅ Titans installation successful!")
```

## Common Issues

### JAX Installation Issues

If you encounter JAX installation problems:

```bash
# Uninstall existing JAX
pip uninstall jax jaxlib

# Reinstall with specific CUDA version
pip install jax[cuda12_pip]  # For CUDA 12
pip install jax[cuda11_pip]  # For CUDA 11
```

### Memory Issues

For large models, you may need to:

1. **Enable memory preallocation**:
   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```

2. **Set memory fraction**:
   ```bash
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
   ```

3. **Use gradient checkpointing**:
   ```python
   from titans import create_titans_model

   model = create_titans_model(
       variant="mac",
       d_model=512,
       gradient_checkpointing=True
   )
   ```

### Performance Optimization

For optimal performance:

1. **Enable XLA compilation**:
   ```python
   import jax

   # Compile model functions
   @jax.jit
   def forward_pass(model, x):
       return model(x)
   ```

2. **Use mixed precision**:
   ```python
   from titans import create_titans_model

   model = create_titans_model(
       variant="mac",
       d_model=512,
       dtype=jax.numpy.bfloat16  # Use bfloat16 for training
   )
   ```

## Environment Setup

### Using Conda

```bash
conda create -n titans python=3.11
conda activate titans
pip install titans-flax[all]
```

### Using Poetry

```bash
poetry new my-titans-project
cd my-titans-project
poetry add titans-flax
```

### Using Docker

```dockerfile
FROM python:3.11-slim

RUN pip install titans-flax[gpu]

WORKDIR /app
COPY . .

CMD ["python", "your_titans_script.py"]
```

## Next Steps

Once installed, check out:

- [Quick Start Guide](quick-start.md) - Get up and running quickly
- [Configuration Guide](configuration.md) - Learn about configuration options
- [Basic Usage Tutorial](../tutorials/basic-usage.md) - Detailed usage examples

## Getting Help

If you encounter issues:

1. Check the [FAQ](../faq.md)
2. Search [existing issues](https://github.com/mahdi-shafiei/titans-flax/issues)
3. Create a [new issue](https://github.com/mahdi-shafiei/titans-flax/issues/new) with details about your environment and the problem
