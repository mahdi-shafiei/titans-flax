# Quick Start

Get up and running with Titans in under 5 minutes! This guide will walk you through the basic steps to create and use a Titans model.

## Installation

First, install Titans:

```bash
pip install titans-flax
```

For development or to access all features:

```bash
pip install "titans-flax[all]"
```

## Your First Titans Model

Let's create a simple Titans model and see it in action:

```python
import jax.numpy as jnp
from titans.core import create_titans_model

# Create a Titans model
model = create_titans_model(
    variant="mac",      # Memory as Context - best for long sequences
    d_model=256,        # Model dimension
    num_heads=8,        # Number of attention heads
    memory_depth=4,     # Memory update depth
    seed=42            # Random seed for reproducibility
)

# Generate some test data
batch_size, seq_len = 2, 128
x = jnp.ones((batch_size, seq_len, 256))

# Process the sequence with memory updates enabled
output = model(x, deterministic=False)  # deterministic=False enables memory updates

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Memory magnitude: {model.get_memory_norm()}")
```

## Processing Your First Sequence

Now let's process a sequence and see the memory adaptation in action:

```python
# Create some sample data
batch_size, seq_len, d_model = 2, 64, 128
x = jax.random.normal(jax.random.key(42), (batch_size, seq_len, d_model))

print(f"Input shape: {x.shape}")

# Process the sequence (memory will adapt automatically)
output = model(x, deterministic=False)  # deterministic=False enables memory updates

print(f"Output shape: {output.shape}")
print(f"Memory norm after processing: {model.get_memory_norm():.4f}")
```

## Test-Time Adaptation Demo

Here's a simple demonstration of how Titans adapts during inference:

```python
def quick_adaptation_demo():
    """Demonstrate memory adaptation with a simple pattern."""

    # Reset memory for clean demo
    model.reset_memory()
    print("🔄 Reset memory")

    # Create a simple repeating pattern
    pattern = jnp.array([1.0, 0.0, -1.0, 0.0])

    print("\n📊 Processing sequences with similar patterns...")

    for step in range(5):
        # Create sequence with slight variations of the pattern
        noise = 0.1 * jax.random.normal(jax.random.key(step), pattern.shape)
        noisy_pattern = pattern + noise

        # Repeat pattern to create a sequence
        seq = jnp.tile(noisy_pattern, (1, 16, 32))  # (1, 64, 128)

        # Process sequence (memory adapts)
        output = model(seq, deterministic=False)

        # Check memory evolution
        memory_norm = model.get_memory_norm()

        # Simple reconstruction error
        error = jnp.mean((output - seq) ** 2)

        print(f"  Step {step+1}: Memory norm = {memory_norm:.4f}, Error = {error:.4f}")

    print("\n✨ Notice how memory norm increases and error typically decreases!")

quick_adaptation_demo()
```

## Language Modeling Example

Let's try a more practical example with text processing:

```python
from transformers import AutoTokenizer

# Setup a simple tokenizer (you can use any tokenizer)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def process_text(text, max_length=64):
    """Process text through Titans model."""

    # Tokenize text
    tokens = tokenizer.encode(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="np"
    )

    # Convert to JAX array and add batch dimension
    tokens = jnp.array(tokens).reshape(1, -1)  # (1, seq_len)

    # Create embeddings (simplified - normally you'd use a proper embedding layer)
    vocab_size = tokenizer.vocab_size
    embed_matrix = jax.random.normal(jax.random.key(123), (vocab_size, 128))
    embeddings = embed_matrix[tokens]  # (1, seq_len, 128)

    # Process through Titans
    output = model(embeddings, deterministic=False)

    return output, tokens

# Example texts
texts = [
    "The future of artificial intelligence is bright.",
    "Machine learning models are becoming more sophisticated.",
    "Deep learning has revolutionized computer science."
]

print("🔤 Processing text examples:")
model.reset_memory()

for i, text in enumerate(texts):
    output, tokens = process_text(text)
    memory_norm = model.get_memory_norm()

    print(f"  Text {i+1}: '{text[:30]}...'")
    print(f"    Memory norm: {memory_norm:.4f}")
    print(f"    Output shape: {output.shape}")
```

## Exploring Different Variants

Try different Titans variants to see their characteristics:

```python
def compare_variants():
    """Compare different Titans variants."""

    variants = ["mac", "mag", "mal", "lmm"]
    x = jax.random.normal(jax.random.key(42), (1, 128, 128))

    print("🔬 Comparing Titans variants:")

    for variant in variants:
        # Create model
        model_variant = create_titans_model(
            variant=variant,
            d_model=128,
            memory_depth=2,  # Smaller for quick demo
            seed=42
        )

        # Process sequence
        output = model_variant(x, deterministic=False)
        memory_norm = model_variant.get_memory_norm()
        param_count = model_variant.count_params()

        print(f"  {variant.upper()}: {param_count:,} params, memory norm = {memory_norm:.4f}")

compare_variants()
```

## Memory Analysis

Explore how memory behaves:

```python
def analyze_memory():
    """Analyze memory behavior patterns."""

    model.reset_memory()

    # Different types of inputs
    inputs = {
        "random": jax.random.normal(jax.random.key(1), (1, 64, 128)),
        "structured": jnp.ones((1, 64, 128)),
        "zeros": jnp.zeros((1, 64, 128)),
        "alternating": jnp.tile(jnp.array([1, -1]), (1, 32, 64))
    }

    print("🧠 Memory response to different inputs:")

    for name, input_data in inputs.items():
        model.reset_memory()

        # Process multiple times to see adaptation
        for step in range(3):
            output = model(input_data, deterministic=False)
            memory_norm = model.get_memory_norm()

            if step == 0:
                print(f"  {name.capitalize()} input:")
            print(f"    Step {step+1}: Memory norm = {memory_norm:.4f}")

analyze_memory()
```

## Configuration Options

Explore different configuration options:

```python
from titans import get_config, create_config_for_task

# Try predefined configurations
configs = ["small", "medium", "large"]

print("⚙️  Available configurations:")
for config_name in configs:
    config = get_config(config_name)
    print(f"  {config_name}: d_model={config.architecture.d_model}, "
          f"memory_depth={config.memory.memory_depth}")

# Create task-specific configuration
lm_config = create_config_for_task("language_modeling", "medium")
print(f"\n📝 Language modeling config: d_model={lm_config.architecture.d_model}")
```

## Performance Tips

Here are some quick performance tips:

```python
# 1. Use JIT compilation for faster execution
import jax

@jax.jit
def fast_forward(model, x):
    return model(x, deterministic=False)

# 2. Use the right variant for your use case
def choose_variant(sequence_length):
    if sequence_length < 1000:
        return "mal"  # Fastest for short sequences
    elif sequence_length < 10000:
        return "mag"  # Balanced
    else:
        return "mac"  # Best for very long sequences

# 3. Enable gradient checkpointing for large models
efficient_model = create_titans_model(
    variant="mac",
    d_model=512,
    gradient_checkpointing=True  # Saves memory
)

print("🚀 Performance tips applied!")
```

## What's Next?

Now that you've seen the basics, here are the next steps:

1. **Deep Dive**: Read [Understanding Titans](../tutorials/understanding-titans.md) for detailed concepts
2. **Practical Examples**: Try the [Language Modeling](../examples/language-modeling.md) example
3. **Architecture Guide**: Learn about [Architecture Variants](../tutorials/architecture-variants.md)
4. **Training**: Explore [Training Models](../tutorials/training-models.md)
5. **API Reference**: Check the complete [API Documentation](../api/index.md)

## Common Issues

### Memory Not Adapting?
Make sure you're using `deterministic=False`:

```python
# ✅ Correct - enables memory updates
output = model(x, deterministic=False)

# ❌ Incorrect - no memory updates
output = model(x, deterministic=True)
```

### GPU Memory Issues?
Try gradient checkpointing:

```python
model = create_titans_model(
    variant="mag",
    d_model=512,
    gradient_checkpointing=True
)
```

### Slow Performance?
Use JIT compilation:

```python
@jax.jit
def compiled_forward(model, x):
    return model(x, deterministic=False)
```

## Getting Help

- 📖 [Full Documentation](../index.md)
- 🐛 [GitHub Issues](https://github.com/mahdi-shafiei/titans-flax/issues)
- 💬 [Discussions](https://github.com/mahdi-shafiei/titans-flax/discussions)

---

🎉 **Congratulations!** You've successfully started using Titans. The neural long-term memory is now adapting during inference, enabling your model to handle long sequences and adapt to new patterns automatically.
