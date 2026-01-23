# Parallel Training Algorithm (Section 3.2)

This implementation adds the parallel training algorithm described in Section 3.2 of the Titans paper as an optional feature. The algorithm uses tensorized mini-batch gradient descent and parallel associative scan to accelerate neural memory training on long sequences.

## 🔬 Algorithm Overview

The paper shows how the neural memory training can be parallelized by:

1. **Chunking**: Split sequences into chunks of size `b`
2. **Tensorized Gradients**: Compute gradients using matrix operations (Equation 17)
3. **Parallel Momentum**: Use associative scan for momentum updates (Equation 18)
4. **Batch Updates**: Update memory weights in batches rather than sequentially

### Mathematical Foundation

The key insight is reformulating the sequential updates:
```
M_t = (1 - α_t)M_{t-1} + S_t
S_t = η_t S_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t)
```

Into chunked matrix operations:
```
∇ℓ = Θ_b B_b (W_0 X - X) X^T  (Equation 17)
```

Where `Θ_b` and `B_b` are diagonal matrices containing learning rates and decay factors.

## 🚀 Quick Start

### Basic Usage

```python
from titans.enhanced_neural_memory import EnhancedNeuralLongTermMemory, create_parallel_config
from flax import nnx
import jax.numpy as jnp

# Create parallel training configuration
parallel_config = create_parallel_config(
    enabled=True,
    chunk_size=64,
    min_sequence_length=256
)

# Create enhanced neural memory with parallel training
memory = EnhancedNeuralLongTermMemory(
    d_model=256,
    memory_depth=3,
    parallel_config=parallel_config,
    rngs=nnx.Rngs(42)
)

# Use normally - parallel training is automatic
x = jnp.ones((2, 512, 256))  # Long sequence triggers parallel training
output, memory_state = memory(x, deterministic=False)
```

### Configuration Options

```python
from titans.enhanced_config import EnhancedTitansConfig, create_config_for_task_with_parallel

# Create configuration with parallel training for long sequences
config = create_config_for_task_with_parallel(
    task_type="long_sequence",
    model_size="medium",
    enable_parallel=True
)

print(config.get_parallel_training_summary())
# Output:
# Parallel training: Enabled
#   Chunk size: 128
#   Chunk-level params: True
#   Min sequence length: 512
#   Auto-enable threshold: 1024
```

## ⚙️ Configuration Parameters

### `ParallelTrainingConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable/disable parallel training |
| `chunk_size` | `int` | `64` | Size of chunks for tensorized operations |
| `use_chunk_level_params` | `bool` | `False` | Use chunk-level vs token-level parameters |
| `min_sequence_length` | `int` | `256` | Minimum length to trigger parallel mode |
| `auto_enable_threshold` | `int` | `1024` | Auto-enable for sequences longer than this |

### Recommended Settings

| Use Case | `chunk_size` | `chunk_level_params` | Notes |
|----------|--------------|---------------------|--------|
| **Short sequences (< 512)** | N/A | N/A | Use sequential training |
| **Medium sequences (512-2048)** | `64` | `False` | Balanced performance |
| **Long sequences (> 2048)** | `128` | `True` | Maximum efficiency |
| **Very long sequences (> 8192)** | `256` | `True` | Reduce memory overhead |

## 🔧 Integration with Existing Code

### Drop-in Replacement

The enhanced neural memory is designed as a drop-in replacement:

```python
# Before: Original neural memory
from titans.neural_memory import NeuralLongTermMemory

# After: Enhanced version with parallel training
from titans.enhanced_neural_memory import EnhancedNeuralLongTermMemory as NeuralLongTermMemory

# The API is identical, just add parallel_config
memory = NeuralLongTermMemory(
    d_model=256,
    parallel_config=create_parallel_config(enabled=True),
    rngs=nngs
)
```

### Configuration Migration

```python
# Migrate existing configurations
from titans.config import get_config
from titans.enhanced_config import EnhancedTitansConfig, ParallelTrainingConfig

# Load existing config
old_config = get_config("medium")

# Convert to enhanced config with parallel training
new_config = EnhancedTitansConfig(
    variant=old_config.variant,
    # ... copy other fields ...
    memory=MemoryConfig(
        # ... copy memory fields ...
        parallel_training=ParallelTrainingConfig(
            enabled=True,
            chunk_size=64
        )
    )
)
```

## 📊 Performance Characteristics

### When to Use Parallel Training

✅ **Recommended for:**
- Long sequences (> 256 tokens)
- Batch processing of multiple sequences
- Memory-intensive tasks requiring deep neural memory
- Training on TPUs/multi-GPU setups

❌ **Not recommended for:**
- Short sequences (< 128 tokens)
- Single-token inference
- Memory-constrained environments
- Simple memory depths (< 2 layers)

### Expected Performance Gains

| Sequence Length | Expected Speedup | Memory Usage |
|----------------|------------------|--------------|
| 128-256 | 0.8-1.2x | +10% |
| 512-1024 | 1.2-2.0x | +15% |
| 2048-4096 | 1.5-3.0x | +20% |
| > 4096 | 2.0-4.0x | +25% |

*Note: Actual performance depends on hardware, memory depth, and chunk size.*

## 🧪 Advanced Usage

### Custom Training Loop

```python
def custom_training_with_parallel(model, data_loader):
    for batch in data_loader:
        x, targets = batch

        # Force parallel training for this batch
        output, memory_state = model.neural_memory(
            x,
            deterministic=False,
            force_parallel=True  # Override automatic detection
        )

        # Compute loss and update
        loss = compute_loss(output, targets)
        # ... rest of training loop
```

### Dynamic Configuration

```python
def adaptive_parallel_training(memory, sequence_length):
    """Dynamically adjust parallel training based on sequence length."""

    if sequence_length < 256:
        # Use sequential training
        return memory(x, force_parallel=False)
    elif sequence_length < 1024:
        # Use small chunks
        memory.parallel_config.chunk_size = 32
        return memory(x, force_parallel=True)
    else:
        # Use large chunks with chunk-level params
        memory.parallel_config.chunk_size = 128
        memory.parallel_config.use_chunk_level_params = True
        return memory(x, force_parallel=True)
```

### Memory Evolution Monitoring

```python
def monitor_memory_evolution(memory, sequences):
    """Monitor how memory evolves with parallel training."""

    memory_norms = []
    memory_state = None

    for i, seq in enumerate(sequences):
        output, memory_state = memory(
            seq,
            memory_state=memory_state,
            deterministic=False
        )

        # Track memory magnitude
        norm = sum(float(jnp.sum(jnp.abs(w))) for w in memory_state.weights)
        memory_norms.append(norm)

        print(f"Sequence {i}: Memory norm = {norm:.4f}")

    return memory_norms
```

## 🔍 Debugging and Troubleshooting

### Common Issues

1. **Slower than sequential training**
   - Try larger `chunk_size` (128 or 256)
   - Enable `use_chunk_level_params=True`
   - Ensure sequence length > `min_sequence_length`

2. **Memory overflow errors**
   - Reduce `chunk_size`
   - Decrease `memory_depth`
   - Enable gradient checkpointing

3. **Numerical instability**
   - Reduce `learning_rate` in memory config
   - Increase `forget_rate` for faster decay
   - Use smaller `momentum_decay`

### Diagnostic Tools

```python
def diagnose_parallel_training(config):
    """Diagnose parallel training configuration."""

    print(f"Parallel training: {config.memory.parallel_training.enabled}")
    print(f"Chunk size: {config.memory.parallel_training.chunk_size}")
    print(f"Max sequence length: {config.training.max_seq_length}")

    # Check if configuration makes sense
    if config.memory.parallel_training.chunk_size > config.training.max_seq_length:
        print("⚠️  WARNING: chunk_size > max_seq_length")

    if config.memory.parallel_training.min_sequence_length > config.training.max_seq_length:
        print("⚠️  WARNING: min_sequence_length > max_seq_length")

    # Recommend optimal settings
    optimal_chunk = min(64, config.training.max_seq_length // 4)
    print(f"💡 Recommended chunk_size: {optimal_chunk}")
```

## 🔬 Implementation Details

### Tensorized Gradient Computation

The implementation follows Equation 17 from the paper:

```python
def _compute_chunk_gradients_tensorized(self, chunk_keys, chunk_values, memory_weights, theta_vals):
    """Implements: ∇ℓ = Θ_b B_b (W_0 X - X) X^T"""

    # Apply learning rate weighting per token
    weighted_errors = theta_vals[:, None] * errors

    # Compute gradient using matrix operations
    gradient = jnp.dot(layer_input.T, weighted_errors) / chunk_len

    return gradient
```

### Parallel Associative Scan

For momentum computation following Equation 18:

```python
def parallel_associative_scan(transitions, elements):
    """Parallel scan for: h_t = a_t * h_{t-1} + b_t"""

    def combine_fn(a, b):
        (a1, b1), (a2, b2) = a, b
        return (a1 * a2, a1 * b2 + b1)

    scan_inputs = (transitions, elements)
    _, outputs = lax.associative_scan(combine_fn, scan_inputs)

    return outputs
```

## 📚 References

- **Section 3.2** of "Titans: Learning to Memorize at Test Time"
- **Equation 17**: Tensorized mini-batch gradient descent
- **Equation 18**: Parallel momentum computation
- [JAX Associative Scan Documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html)

## 🤝 Contributing

To extend the parallel training algorithm:

1. **Add new chunking strategies** in `_chunk_sequence()`
2. **Implement custom parameter schedules** in chunk processing
3. **Add memory-efficient variants** for very long sequences
4. **Optimize for specific hardware** (TPU, multi-GPU)

The modular design makes it easy to experiment with different parallelization approaches while maintaining compatibility with the existing Titans architecture.
