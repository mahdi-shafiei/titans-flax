# Understanding Titans

This tutorial introduces the core concepts behind the Titans architecture and explains how neural long-term memory enables test-time adaptation.

## Learning Objectives

After completing this tutorial, you will understand:

- The motivation for neural long-term memory
- How test-time adaptation works
- The differences between the four Titans variants
- When to use Titans vs traditional transformers

## The Problem: Long Sequences and Adaptation

Traditional transformer models face two key limitations:

1. **Quadratic complexity** with sequence length
2. **Fixed parameters** during inference (no adaptation)

Titans addresses both issues through **neural long-term memory** that adapts during inference.

## What is Neural Long-term Memory?

Neural long-term memory is a learnable component that:

- **Stores information** from previous inputs
- **Adapts during inference** using gradient descent
- **Maintains state** across sequence processing steps
- **Scales efficiently** to very long sequences

### Mathematical Foundation

The memory update follows this pattern:

\[
\memory_{t+1} = \memory_t - \alpha \grad_{\memory_t} \loss(\memory_t, \vec{x}_t)
\]

Where:
- \(\memory_t\) is the memory state at step \(t\)
- \(\alpha\) is the learning rate
- \(\grad_{\memory_t} \loss\) is the gradient of the loss with respect to memory
- \(\vec{x}_t\) is the current input

## Core Components

### 1. Memory Module

The memory module contains learnable parameters that adapt during inference:

```python
import jax
import jax.numpy as jnp
from titans.neural_memory import NeuralLongTermMemory
from flax import nnx

# Create memory module
rngs = nnx.Rngs(42)
memory = NeuralLongTermMemory(
    d_model=512,
    memory_depth=4,  # Number of memory layers
    learning_rate=0.01,
    momentum_decay=0.9,
    rngs=rngs
)

# Process input (memory adapts automatically)
x = jax.random.normal(jax.random.key(42), (1, 128, 512))
output, memory_state = memory(x, deterministic=False)  # Enable adaptation

print(f"Output shape: {output.shape}")
print(f"Memory state weights: {len(memory_state.weights)} layers")
```

### 2. Attention Integration

Memory is integrated with attention in different ways depending on the variant:

```python
from titans.core import create_titans_model

# Different integration strategies
mac_model = create_titans_model(variant="mac", d_model=512, seed=42)  # Memory as Context
mag_model = create_titans_model(variant="mag", d_model=512, seed=42)  # Memory as Gate
mal_model = create_titans_model(variant="mal", d_model=512, seed=42)  # Memory as Layer
lmm_model = create_titans_model(variant="lmm", d_model=512, seed=42)  # Long-term Memory Module
```

## The Four Variants

### MAC: Memory as Context

**Use case**: Very long sequences (1M+ tokens)

The MAC variant treats memory as additional context for attention:

\[
\attn(\query, \key, \value) = \attn(\query, [\key; \memory], [\value; \memory])
\]

```python
# Best for extremely long documents
mac_model = create_titans_model(
    variant="mac",
    d_model=512,
    memory_depth=8,  # More memory for longer sequences
    seed=42
)

# Process very long sequence (reduced size for demo)
long_sequence = jax.random.normal(jax.random.key(42), (1, 10000, 512))
output = mac_model(long_sequence, deterministic=False)
```

**Advantages**:
- Handles extremely long sequences efficiently
- Memory provides global context
- Scales sub-quadratically

**Trade-offs**:
- More complex attention computation
- Higher memory usage

### MAG: Memory as Gate

**Use case**: Balanced efficiency and effectiveness

The MAG variant uses memory to gate attention computations:

\[
\text{output} = \sigma(\memory) \odot \attn(\query, \key, \value) + (1 - \sigma(\memory)) \odot \vec{x}
\]

```python
# Balanced approach
mag_model = create_titans_model(
    variant="mag",
    d_model=512,
    memory_depth=4,
    seed=42
)

# Good for most applications
x = jax.random.normal(jax.random.key(42), (1, 1024, 512))
output = mag_model(x, deterministic=False)
```

**Advantages**:
- Good balance of performance and efficiency
- Memory controls information flow
- Easier to train than MAC

**Trade-offs**:
- Moderate complexity
- May not scale to extremely long sequences

### MAL: Memory as Layer

**Use case**: Simple and fast training

The MAL variant adds memory as an additional layer:

\[
\vec{h} = \text{LayerNorm}(\vec{x} + \text{Attention}(\vec{x}))
\]
\[
\text{output} = \text{LayerNorm}(\vec{h} + \memory(\vec{h}))
\]

```python
# Simplest variant
mal_model = create_titans_model(
    variant="mal",
    d_model=512,
    memory_depth=2,  # Fewer layers for simplicity
    seed=42
)

# Fast training and inference
x = jax.random.normal(jax.random.key(42), (1, 512, 512))
output = mal_model(x, deterministic=False)
```

**Advantages**:
- Simplest architecture
- Fast training and inference
- Easy to understand and debug

**Trade-offs**:
- Limited scalability
- Less sophisticated memory integration

### LMM: Long-term Memory Module

**Use case**: Pure memory without attention overhead

The LMM variant focuses purely on memory, bypassing traditional attention:

\[
\text{output} = \memory(\vec{x})
\]

```python
# Pure memory approach
lmm_model = create_titans_model(
    variant="lmm",
    d_model=512,
    memory_depth=6,  # Focus on memory capacity
    seed=42
)

# Efficient for memory-focused tasks
x = jax.random.normal(jax.random.key(42), (1, 2048, 512))
output = lmm_model(x, deterministic=False)
```

**Advantages**:
- Maximum efficiency
- Pure focus on memory adaptation
- Lowest computational overhead

**Trade-offs**:
- No attention mechanism
- May miss some dependencies

## Test-Time Adaptation in Action

Let's see how memory adapts during inference:

```python
def demonstrate_adaptation():
    model = create_titans_model(variant="mag", d_model=128, memory_depth=2, seed=42)

    # Generate related sequences
    base_pattern = jnp.array([1.0, 0.5, -0.5, -1.0])
    sequences = []

    for i in range(10):
        # Add noise to base pattern
        noise = 0.1 * jax.random.normal(jax.random.key(i), base_pattern.shape)
        pattern = base_pattern + noise

        # Repeat pattern to create sequence
        seq = jnp.tile(pattern, (1, 32, 32))  # (1, 128, 128)
        sequences.append(seq)

    # Track memory evolution
    model.reset_memory()
    memory_norms = []
    losses = []

    for i, seq in enumerate(sequences):
        # Forward pass (memory adapts)
        output = model(seq, deterministic=False)

        # Compute simple reconstruction loss
        loss = jnp.mean((output - seq) ** 2)
        losses.append(float(loss))

        # Track memory norm using the model's method
        memory_norm = model.get_memory_norm()
        memory_norms.append(float(memory_norm))

        print(f"Step {i+1}: Loss = {loss:.4f}, Memory norm = {memory_norm:.4f}")

    return losses, memory_norms

# Run the demonstration
losses, memory_norms = demonstrate_adaptation()

# Print summary
print(f"\nAdaptation Summary:")
print(f"Initial loss: {losses[0]:.4f}")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
print(f"Memory evolution: {memory_norms[0]:.4f} → {memory_norms[-1]:.4f}")
```

## When to Use Titans

### Choose Titans when:

1. **Long sequences**: Working with sequences longer than typical transformer limits
2. **Domain adaptation**: Need models to adapt to new domains during inference
3. **Streaming data**: Processing continuous streams where adaptation is beneficial
4. **Memory-critical tasks**: Tasks requiring long-term memory of patterns

### Choose traditional transformers when:

1. **Short sequences**: Working with sequences under 2K tokens
2. **Static domains**: Domain is fixed and well-defined
3. **Maximum speed**: Need absolute fastest inference
4. **Simplicity**: Want the simplest possible architecture

## Variant Selection Guide

| Sequence Length | Adaptation Needs | Recommended Variant |
|----------------|------------------|-------------------|
| < 1K tokens | Low | MAL |
| 1K - 10K tokens | Medium | MAG |
| 10K - 100K tokens | High | MAC |
| > 100K tokens | High | MAC |
| Any length | Pure memory | LMM |

## Memory Configuration

### Key parameters to tune:

```python
config = {
    "memory_depth": 4,          # Number of memory layers (2-8)
    "memory_learning_rate": 0.01, # Adaptation rate (0.001-0.1)
    "memory_momentum": 0.9,     # Momentum for adaptation (0.0-0.99)
    "memory_weight_decay": 1e-4, # Regularization (0-0.01)
    "memory_init_scale": 0.02   # Initial memory scale (0.001-0.1)
}
```

### Rule of thumb:
- **Higher memory_depth**: Better for complex patterns, slower adaptation
- **Higher learning_rate**: Faster adaptation, less stability
- **Higher momentum**: Smoother adaptation, slower convergence

## Understanding Memory Behavior

### Memory initialization:
```python
# Memory starts with small random values
initial_norm = model.get_memory_norm()
print(f"Initial memory norm: {initial_norm:.4f}")  # Usually ~0.02
```

### Memory adaptation:
```python
# Memory adapts based on gradients
for step in range(10):
    output = model(x, deterministic=False)
    norm = model.get_memory_norm()
    print(f"Step {step}: Memory norm = {norm:.4f}")
```

### Memory reset:
```python
# Reset memory for new task/domain
model.reset_memory()
print(f"After reset: {model.get_memory_norm():.4f}")  # Back to ~0.02
```

## Common Patterns

### 1. Domain Adaptation
```python
# Train on source domain
source_data = load_source_domain()
model = train_titans_model(source_data)

# Adapt to target domain during inference
target_data = load_target_domain()
model.reset_memory()

for batch in target_data:
    output = model(batch, deterministic=False)  # Memory adapts
    # Use output for predictions
```

### 2. Incremental Learning
```python
# Process documents sequentially
documents = load_long_documents()
model.reset_memory()

for doc in documents:
    output = model(doc, deterministic=False)
    # Memory retains information from previous documents
```

### 3. Few-shot Adaptation
```python
# Few examples to adapt to new task
examples = load_few_shot_examples()
model.reset_memory()

# Adapt memory using examples
for example in examples:
    _ = model(example, deterministic=False)

# Now use adapted model
result = model(test_input, deterministic=False)
```

## Next Steps

Now that you understand the core concepts:

1. Try the [Basic Usage](basic-usage.md) tutorial for hands-on experience
2. Explore [Memory Mechanisms](memory-mechanisms.md) for deeper technical details
3. Learn about [Architecture Variants](architecture-variants.md) to choose the right one
4. Practice with the [Language Modeling](../examples/language-modeling.md) example

## Key Takeaways

- **Neural long-term memory** enables adaptation during inference
- **Four variants** offer different trade-offs between efficiency and capability
- **Test-time adaptation** allows models to improve on new data without retraining
- **Memory configuration** significantly affects behavior and performance
- **Variant selection** depends on sequence length and adaptation requirements

!!! tip "Experimentation"
    The best way to understand Titans is to experiment with different variants and configurations on your specific use case!

## Quick Start

Here's how to create and use a Titans model:

```python
import jax.numpy as jnp
from titans.core import create_titans_model

# Create a MAC model (best for very long sequences)
model = create_titans_model(
    variant="mac",
    d_model=256,
    memory_depth=3,
    seed=42
)

# Process a sequence with test-time adaptation
x = jnp.ones((1, 512, 256))
output = model(x, deterministic=False)

print(f"Input: {x.shape} → Output: {output.shape}")
print(f"Memory state norm: {model.get_memory_norm():.4f}")
```
