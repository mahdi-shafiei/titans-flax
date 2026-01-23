# API Reference

Welcome to the Titans API reference documentation. This section provides detailed documentation for all public APIs, classes, and functions.

## Quick Navigation

### Core Components
- [Main Module](core.md) - Model factory functions and core classes
- [Memory Module](neural-memory.md) - Memory mechanisms and state management
- [Attention Modules](attention-modules.md) - Attention mechanisms
- [Architecture Variants](variants.md) - Four Titans variants (MAC, MAG, MAL, LMM)

### Configuration & Training
- [Configuration](config.md) - Model configuration and presets
- [Training](training.md) - Training utilities and metrics

### Utilities
- [Optimization](optimization-utilities.md) - Performance optimization tools
- [Benchmarking](benchmark.md) - Benchmarking and evaluation suite
- [Serialization](serialization.md) - Model persistence and checkpointing

## Getting Started

The quickest way to get started is with the main module:

```python
from titans.core import create_titans_model
from titans import TitansModel, TitansConfig

# Create a model
model = create_titans_model(variant="mac", d_model=128)

# Use configuration
config = TitansConfig(variant="mag", d_model=256)
```

For more examples, see the [Examples](../examples/) section.

## Main Module

::: titans.core
    options:
      show_root_heading: true
      show_source: false

- `create_titans_model()` - Factory function for creating models
- `TitansModel` - Main model wrapper class
- `TitansVariant` - Type hint for variant names

## Architecture Variants

The four Titans variants with different memory integration strategies:

- **MAC (Memory as Context)**: Best for very long sequences (2M+ tokens)
- **MAG (Memory as Gate)**: Balanced efficiency and effectiveness
- **MAL (Memory as Layer)**: Fast training and simple architecture
- **LMM (Long-term Memory Module)**: Pure memory without attention

### Implementation

::: titans.variants
    options:
      show_root_heading: true
      show_source: false

- `TitansMAC` - Memory as Context implementation
- `TitansMAG` - Memory as Gate implementation
- `TitansMAL` - Memory as Layer implementation
- `TitansLMM` - Long-term Memory Module implementation

## Configuration

::: titans.config
    options:
      show_root_heading: true
      show_source: false

- `TitansConfig` - Complete configuration class
- `get_config()` - Get predefined configurations
- `create_config_for_task()` - Task-specific configurations

## Training

::: titans.training
    options:
      show_root_heading: true
      show_source: false

- `TitansTrainer` - Advanced trainer with checkpointing
- `TrainingMetrics` - Training metrics container
- `LearningRateScheduler` - Learning rate scheduling utilities

## Memory Components

### Neural Long-term Memory

::: titans.neural_memory
    options:
      show_root_heading: true
      show_source: false

- `NeuralLongTermMemory` - Main memory module with gradient descent
- `PersistentMemory` - Persistent memory tokens
- `MemoryState` - Memory state container

### Attention Mechanisms

::: titans.attention_modules
    options:
      show_root_heading: true
      show_source: false

- `CausalAttention` - Standard causal attention
- `SlidingWindowAttention` - Sliding window attention for efficiency
- `MultiHeadAttention` - Multi-head attention wrapper

## Benchmarking

::: titans.benchmark
    options:
      show_root_heading: true
      show_source: false

- `TitansBenchmark` - Full benchmarking suite
- `BenchmarkResult` - Individual benchmark results
- `PerformanceMetrics` - Performance metrics container

## Serialization

::: titans.serialization
    options:
      show_root_heading: true
      show_source: false

- `TitansCheckpoint` - Checkpoint management
- `ModelExporter` - Model export utilities
- `MemoryStateManager` - Memory state persistence

## Quick Reference

### Creating Models

```python
from titans import create_titans_model

# Create a MAC variant for long sequences
model = create_titans_model(
    variant="mac",
    d_model=512,
    num_heads=8,
    memory_depth=4,
    seed=42
)
```

### Configuration

```python
from titans import get_config, create_config_for_task

# Use predefined config
config = get_config("medium")

# Create task-specific config
config = create_config_for_task("language_modeling", "large")
```

### Training

```python
from titans import TitansTrainer

trainer = TitansTrainer(model, config)
trainer.train(train_dataset, eval_dataset)
```

### Memory Management

```python
# Reset memory
model.reset_memory()

# Get memory state
memory_state = model.get_memory_state()

# Check memory norm
memory_norm = model.get_memory_norm()
```

## API Conventions

### Function Signatures
All major functions follow consistent patterns:

- **Models**: `__call__(x, deterministic=True, **kwargs)`
- **Trainers**: `train(train_data, eval_data=None, **kwargs)`
- **Configs**: Pydantic models with validation

### Memory Behavior
- **`deterministic=False`**: Enables memory updates during inference
- **`deterministic=True`**: Disables memory updates (for evaluation)
- **Memory persistence**: Maintains state across calls until reset

### Error Handling
- Clear error messages with suggested fixes
- Type checking with informative warnings
- Graceful degradation when possible

## Type Annotations

The library uses full type annotations:

```python
from typing import Optional, Union, Dict, Any
import jax.numpy as jnp
from flax import nnx

def create_titans_model(
    variant: str,
    d_model: int,
    num_heads: int = 8,
    memory_depth: int = 4,
    seed: int = 42,
    **kwargs: Any
) -> nnx.Module:
    """Create a Titans model with specified configuration."""
    pass
```

## Development Guidelines

### Adding New APIs

When contributing new APIs:

1. **Use Google-style docstrings**
2. **Include type annotations**
3. **Provide usage examples**
4. **Add thorough tests**
5. **Update this documentation**

### Docstring Format

```python
def example_function(x: jax.Array, y: Optional[int] = None) -> jax.Array:
    """Brief description of the function.

    Longer description if needed. Can span multiple lines and include
    mathematical notation using LaTeX: $f(x) = x^2$.

    Args:
        x: Input array of shape (batch, seq_len, d_model).
        y: Optional parameter for controlling behavior.

    Returns:
        Output array of same shape as input.

    Raises:
        ValueError: If input dimensions are invalid.

    Example:
        ```python
        import jax.numpy as jnp
        result = example_function(jnp.ones((2, 10, 128)))
        ```
    """
    pass
```

## Dependencies

The API documentation assumes you have these imports:

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import titans
```

For specific modules:

```python
from titans import (
    create_titans_model,
    TitansConfig,
    get_config,
    TitansTrainer
)
```

---

!!! tip "Interactive Examples"
    Many API examples can be run directly. Copy the code and try it in your Python environment!
