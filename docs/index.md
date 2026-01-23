# Titans: Learning to Memorize at Test Time

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-compatible-green.svg)](https://jax.readthedocs.io/)
[![Flax NNX](https://img.shields.io/badge/Flax-NNX-orange.svg)](https://flax.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete implementation of the **Titans** architecture from the paper *"Titans: Learning to Memorize at Test Time"* using the Flax NNX framework. This implementation provides neural long-term memory that learns and adapts during inference, enabling models to handle extremely long sequences and perform test-time adaptation.

## 🚀 Key Features

- **🧠 Neural Long-term Memory**: Deep neural networks that adapt during inference using gradient descent with momentum and weight decay
- **🏗️ Four Architecture Variants**: MAC, MAG, MAL, and LMM variants for different use cases
- **📈 Test-time Learning**: Models continue learning and adapting during inference
- **🔧 Production Ready**: Complete training, optimization, and deployment utilities
- **📊 Research Tools**: Complete suite for reproducing paper results and conducting experiments
- **⚡ Performance Optimized**: Memory-efficient implementations with gradient checkpointing and quantization support

## 📊 Architecture Variants

| Variant | Description | Best For |
|---------|-------------|----------|
| **MAC** | Memory as Context | Very long sequences (2M+ tokens) |
| **MAG** | Memory as Gate | Balanced efficiency and effectiveness |
| **MAL** | Memory as Layer | Fast training and simple architecture |
| **LMM** | Long-term Memory Module | Pure memory without attention |

## 🛠️ Quick Installation

```bash
pip install titans-flax
```

For development:
```bash
git clone https://github.com/mahdi-shafiei/titans-flax.git
cd titans-flax
pip install -e ".[dev,docs]"
```

## ⚡ Quick Start

```python
import jax.numpy as jnp
from titans.core import create_titans_model

# Create a Titans model
model = create_titans_model(
    variant="mac",      # Choose: "mac", "mag", "mal", "lmm"
    d_model=128,
    seed=42
)

# Process input with test-time adaptation
x = jnp.ones((2, 64, 128))  # (batch, seq_len, d_model)
output = model(x, deterministic=False)  # Enable memory updates

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
```

## 📚 Documentation Structure

### [Getting Started](getting-started/installation.md)
Learn how to install and configure Titans for your use case.

### [Tutorials](tutorials/index.md)
Step-by-step guides covering:

- [Understanding Titans Architecture](tutorials/understanding-titans.md)
- [Basic Usage Patterns](tutorials/basic-usage.md)
- [Memory Mechanisms](tutorials/memory-mechanisms.md)
- [Architecture Variants](tutorials/architecture-variants.md)
- [Training Models](tutorials/training-models.md)
- [Test-Time Adaptation](tutorials/test-time-adaptation.md)
- [Performance Optimization](tutorials/performance-optimization.md)

### [Examples](examples/index.md)
Practical implementations and use cases:

- [Language Modeling](examples/language-modeling.md)
- [Long Context Tasks](examples/long-context.md)
- [Memory Analysis](examples/memory-analysis.md)
- [Benchmarking](examples/benchmarking.md)
- [Research Reproduction](examples/research-reproduction.md)

### [API Reference](api/index.md)
Complete API documentation with automatically generated references for all modules.

## 🎯 Common Use Cases

=== "Language Modeling"
    ```python
    from titans import get_config
    from titans.core import create_titans_model, create_config_for_task

    config = get_config("language_modeling")
    model = create_titans_model(
        variant=config.variant,
        d_model=config.architecture.d_model,
        num_heads=config.attention.num_heads,
        memory_depth=config.memory.memory_depth,
        seed=42
    )
    # For a complete language model implementation, see examples/language_modeling_app.py
    ```

=== "Long Context Processing"
    ```python
    from titans import get_config
    from titans.core import create_titans_model

    config = get_config("long_context")
    model = create_titans_model(
        variant=config.variant,
        d_model=config.architecture.d_model,
        memory_depth=config.memory.memory_depth,
        seed=42
    )
    ```

=== "Research & Experimentation"
    ```python
    from titans.research_utilities import PaperExperiments

    experiments = PaperExperiments()
    results = experiments.run_all_experiments()
    ```

## 📈 Performance Highlights

- **Memory Efficiency**: Process sequences up to 2M tokens
- **Test-Time Adaptation**: Continuous learning during inference
- **GPU Optimized**: Efficient JAX/XLA compilation
- **Research Reproducible**: Exact paper implementations

## 🔬 Research & Paper

This implementation is based on the paper "Titans: Learning to Memorize at Test Time". The architecture introduces neural long-term memory that adapts during inference, enabling:

- **Adaptive Memory**: Neural networks that learn during inference
- **Extreme Length Handling**: Processing of very long sequences
- **Test-Time Learning**: Continuous adaptation to new data
- **Memory Variants**: Multiple architectures for different use cases

## 🤝 Community & Support

- **Issues**: [GitHub Issues](https://github.com/mahdi-shafiei/titans-flax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mahdi-shafiei/titans-flax/discussions)
- **Contributing**: See our [Contributing Guide](CONTRIBUTING.md)

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

!!! tip "Getting Started"
    New to Titans? Start with our [Installation Guide](getting-started/installation.md) and then try the [Quick Start Tutorial](getting-started/quick-start.md).

## Training Example

```python
from titans.training import TitansTrainer
from titans.config import get_config

# Set up configuration
config = get_config("small")

# Create trainer
trainer = TitansTrainer(model, config)

# Train model (with your data)
# trainer.train(train_dataset, eval_dataset)
```

## Advanced Usage

```python
from titans import get_config
from titans.core import create_titans_model, create_config_for_task

# Use predefined configurations
config = get_config("long_context")
model = create_titans_model(
    variant=config.variant,
    d_model=config.architecture.d_model,
    seed=42
)

# Or create task-specific configuration
config = create_config_for_task("language_modeling", "medium")

# Create model with configuration
from titans.core import create_titans_model
```
