# Language Modeling with Titans

This example demonstrates how to use Titans for language modeling tasks, including training from scratch, test-time adaptation, and text generation.

## Overview

Language modeling is a natural fit for Titans because:

- **Long-range dependencies**: Language often requires understanding long contexts
- **Test-time adaptation**: Models can adapt to new domains or styles during inference
- **Memory efficiency**: Titans can handle very long documents efficiently

## Quick Start

```python
from titans import TitansLanguageModel, create_config_for_task
import jax.numpy as jnp

# Create a language model
config = create_config_for_task("language_modeling", "medium")
model = TitansLanguageModel(
    vocab_size=50000,
    d_model=config.architecture.d_model,
    titans_config=config,
    rngs=nnx.Rngs(42)
)

# Tokenize and process text
tokens = jnp.array([[1, 15, 23, 42, 8, 2]])  # Example token sequence
logits = model(tokens, deterministic=False)
print(f"Output logits shape: {logits.shape}")
```

## Complete Training Example

### 1. Setup and Configuration

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from titans import (
    TitansLanguageModel,
    LanguageModelTrainer,
    create_config_for_task,
    get_config
)
from datasets import load_dataset
import transformers
from titans.training import TitansTrainer

# Configuration
config = create_config_for_task("language_modeling", model_size="medium")
config.training.learning_rate = 1e-4
config.training.batch_size = 8
config.training.sequence_length = 512
config.training.num_epochs = 5

# Model setup
vocab_size = 50000
model = TitansLanguageModel(
    vocab_size=vocab_size,
    d_model=config.architecture.d_model,
    titans_config=config,
    rngs=nnx.Rngs(42)
)

print(f"Model created with {model.count_params():,} parameters")
```

### 2. Data Preparation

```python
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load and prepare dataset
def prepare_dataset():
    # Load a small dataset for demonstration
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.training.sequence_length,
            return_tensors="np"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    return tokenized_dataset

train_dataset = prepare_dataset()

# Create data loader
def collate_fn(batch):
    input_ids = jnp.array([item["input_ids"] for item in batch])
    attention_mask = jnp.array([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": input_ids  # For language modeling, labels = input_ids
    }

train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)
```

### 3. Training Loop

```python
# Setup optimizer
optimizer = optax.adamw(
    learning_rate=config.training.learning_rate,
    weight_decay=config.training.weight_decay
)

# Training state
opt_state = optimizer.init(nnx.state(model))

def compute_loss(model, batch):
    """Compute language modeling loss."""
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    # Forward pass
    logits = model(input_ids, deterministic=False)

    # Compute cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, :-1], labels[:, 1:]
    )

    # Apply attention mask
    mask = attention_mask[:, 1:]
    loss = jnp.sum(loss * mask) / jnp.sum(mask)

    return loss

@jax.jit
def train_step(model, opt_state, batch):
    """Single training step."""

    def loss_fn(model):
        return compute_loss(model, batch)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state)
    nnx.update(model, updates)

    return loss, opt_state

# Training loop
def train_epoch(model, opt_state, data_loader, epoch):
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(data_loader):
        loss, opt_state = train_step(model, opt_state, batch)
        total_loss += float(loss)
        num_batches += 1

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}")

            # Check memory adaptation
            memory_norm = model.get_memory_norm()
            print(f"Memory norm: {memory_norm:.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss, opt_state

# Full training
print("Starting training...")
for epoch in range(config.training.num_epochs):
    avg_loss, opt_state = train_epoch(model, opt_state, train_loader, epoch)
    print(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

    # Reset memory between epochs for consistency
    model.reset_memory()

print("Training completed!")
```

### 4. Text Generation

```python
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text using the trained model."""

    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="np")
    input_ids = jnp.array(input_ids)

    # Reset memory for generation
    model.reset_memory()

    generated = input_ids

    for _ in range(max_length):
        # Forward pass
        logits = model(generated, deterministic=False)

        # Get next token probabilities
        next_token_logits = logits[0, -1, :] / temperature
        probs = jax.nn.softmax(next_token_logits)

        # Sample next token
        key = jax.random.PRNGKey(jnp.sum(generated))
        next_token = jax.random.categorical(key, next_token_logits, shape=(1,))

        # Append to sequence
        generated = jnp.concatenate([generated, next_token.reshape(1, 1)], axis=1)

        # Check for end token
        if next_token[0] == tokenizer.eos_token_id:
            break

    # Decode generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

# Generate some text
prompts = [
    "The future of artificial intelligence",
    "In a world where technology",
    "The most important discovery"
]

for prompt in prompts:
    generated = generate_text(model, tokenizer, prompt, max_length=50)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")
```

## Test-Time Adaptation

One of Titans' key features is test-time adaptation. Here's how to use it:

```python
def demonstrate_adaptation():
    """Show how memory adapts during inference."""

    # Prepare a sequence from a specific domain
    domain_texts = [
        "The neural network architecture consists of multiple layers.",
        "Deep learning models require substantial computational resources.",
        "Backpropagation is the primary training algorithm.",
        "Gradient descent optimizes the model parameters.",
        "Overfitting occurs when models memorize training data."
    ]

    # Reset memory
    model.reset_memory()

    print("Demonstrating test-time adaptation:")

    for i, text in enumerate(domain_texts):
        # Tokenize
        tokens = tokenizer.encode(text, return_tensors="np", max_length=128, truncation=True)
        tokens = jnp.array(tokens)

        # Forward pass (memory updates automatically)
        logits = model(tokens, deterministic=False)

        # Compute perplexity
        loss = compute_loss(model, {"input_ids": tokens, "labels": tokens, "attention_mask": jnp.ones_like(tokens)})
        perplexity = jnp.exp(loss)

        # Check memory state
        memory_norm = model.get_memory_norm()

        print(f"Step {i+1}: Perplexity = {perplexity:.2f}, Memory norm = {memory_norm:.4f}")
        print(f"  Text: {text[:50]}...")

    print("\nMemory has adapted to the technical domain!")

demonstrate_adaptation()
```

## Memory Analysis

Understanding how memory evolves during processing:

```python
import matplotlib.pyplot as plt

def analyze_memory_evolution():
    """Analyze how memory changes during processing."""

    # Prepare test sequences
    sequences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning algorithms learn patterns from data.",
        "Deep neural networks have multiple hidden layers.",
        "Artificial intelligence will transform many industries."
    ]

    memory_norms = []
    memory_weights = []

    # Reset memory
    model.reset_memory()

    for i, text in enumerate(sequences):
        # Process sequence
        tokens = tokenizer.encode(text, return_tensors="np", max_length=64, truncation=True)
        tokens = jnp.array(tokens)

        _ = model(tokens, deterministic=False)

        # Record memory state
        memory_norm = model.get_memory_norm()
        memory_norms.append(float(memory_norm))

        # Get memory weights (example - implementation depends on model structure)
        memory_state = model.get_memory_state()
        if memory_state:
            avg_weight = float(jnp.mean(jnp.abs(memory_state.weights[0])))
            memory_weights.append(avg_weight)
        else:
            memory_weights.append(0.0)

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(memory_norms, 'b-o')
    ax1.set_title('Memory Norm Evolution')
    ax1.set_xlabel('Processing Step')
    ax1.set_ylabel('Memory Norm')
    ax1.grid(True)

    ax2.plot(memory_weights, 'r-o')
    ax2.set_title('Average Memory Weight')
    ax2.set_xlabel('Processing Step')
    ax2.set_ylabel('Average Weight Magnitude')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('memory_analysis.png')
    plt.show()

    return memory_norms, memory_weights

memory_norms, memory_weights = analyze_memory_evolution()
```

## Performance Optimization

### Memory Efficiency

```python
# Enable gradient checkpointing for large models
config_efficient = get_config("efficient")
config_efficient.architecture.gradient_checkpointing = True

efficient_model = TitansLanguageModel(
    vocab_size=vocab_size,
    titans_config=config_efficient,
    rngs=nnx.Rngs(42)
)

# Use mixed precision for faster training
@jax.jit
def efficient_train_step(model, opt_state, batch):
    """Training step with mixed precision."""

    def loss_fn(model):
        # Cast to bfloat16 for computation
        with jax.default_dtype(jnp.bfloat16):
            return compute_loss(model, batch)

    # Compute gradients in full precision
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Update in full precision
    updates, opt_state = optimizer.update(grads, opt_state)
    nnx.update(model, updates)

    return loss, opt_state
```

### Batch Processing

```python
def efficient_batch_processing(texts, batch_size=8):
    """Process multiple texts efficiently."""

    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        batch_tokens = []
        max_len = 0

        for text in batch_texts:
            tokens = tokenizer.encode(text, max_length=512, truncation=True)
            batch_tokens.append(tokens)
            max_len = max(max_len, len(tokens))

        # Pad to same length
        padded_batch = []
        attention_masks = []

        for tokens in batch_tokens:
            padded = tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
            mask = [1] * len(tokens) + [0] * (max_len - len(tokens))

            padded_batch.append(padded)
            attention_masks.append(mask)

        # Convert to JAX arrays
        input_ids = jnp.array(padded_batch)
        attention_mask = jnp.array(attention_masks)

        # Process batch
        logits = model(input_ids, deterministic=False)
        results.append(logits)

    return results
```

## Evaluation

### Perplexity Calculation

```python
def evaluate_perplexity(model, eval_dataset):
    """Calculate perplexity on evaluation dataset."""

    total_loss = 0.0
    total_tokens = 0

    model.reset_memory()  # Reset for consistent evaluation

    for batch in eval_dataset:
        input_ids = jnp.array(batch["input_ids"])
        attention_mask = jnp.array(batch["attention_mask"])

        # Forward pass
        logits = model(input_ids, deterministic=True)  # Deterministic for evaluation

        # Compute loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1], input_ids[:, 1:]
        )

        # Apply attention mask
        mask = attention_mask[:, 1:]
        masked_loss = loss * mask

        total_loss += float(jnp.sum(masked_loss))
        total_tokens += float(jnp.sum(mask))

    avg_loss = total_loss / total_tokens
    perplexity = jnp.exp(avg_loss)

    return float(perplexity)

# Evaluate model
eval_perplexity = evaluate_perplexity(model, eval_loader)
print(f"Evaluation perplexity: {eval_perplexity:.2f}")
```

## Advanced Features

### Domain Adaptation

```python
def domain_adaptation_example():
    """Demonstrate domain adaptation with test-time learning."""

    # Source domain: General text
    general_texts = [
        "The weather is nice today.",
        "I enjoy reading books in my free time.",
        "Travel broadens the mind and soul."
    ]

    # Target domain: Technical text
    technical_texts = [
        "The algorithm converges to the optimal solution.",
        "Neural networks approximate complex functions.",
        "Gradient descent minimizes the loss function."
    ]

    # Process general domain first
    model.reset_memory()
    print("Processing general domain:")

    for text in general_texts:
        tokens = tokenizer.encode(text, return_tensors="np", max_length=64, truncation=True)
        tokens = jnp.array(tokens)

        logits = model(tokens, deterministic=False)
        loss = compute_loss(model, {"input_ids": tokens, "labels": tokens, "attention_mask": jnp.ones_like(tokens)})

        print(f"  Text: {text[:30]}... | Loss: {loss:.4f}")

    print(f"\nGeneral domain memory norm: {model.get_memory_norm():.4f}")

    # Now process technical domain
    print("\nProcessing technical domain:")

    for text in technical_texts:
        tokens = tokenizer.encode(text, return_tensors="np", max_length=64, truncation=True)
        tokens = jnp.array(tokens)

        logits = model(tokens, deterministic=False)
        loss = compute_loss(model, {"input_ids": tokens, "labels": tokens, "attention_mask": jnp.ones_like(tokens)})

        print(f"  Text: {text[:30]}... | Loss: {loss:.4f}")

    print(f"\nAdapted memory norm: {model.get_memory_norm():.4f}")

domain_adaptation_example()
```

## Next Steps

- Try different Titans variants (MAG, MAL, LMM) for comparison
- Experiment with longer sequences using the MAC variant
- Implement custom evaluation metrics
- Explore fine-tuning on specific domains
- Check out the [Memory Analysis](memory-analysis.md) example for deeper insights

## Resources

- [Titans Configuration Guide](../tutorials/architecture-variants.md)
- [Memory Mechanisms Tutorial](../tutorials/memory-mechanisms.md)
- [Performance Optimization Guide](../tutorials/performance-optimization.md)
- [API Reference](../api/core.md)
