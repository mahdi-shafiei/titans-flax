# Tutorials

Welcome to the Titans tutorials! These step-by-step guides will help you understand and use the Titans architecture effectively.

## 📚 Tutorial Sequence

We recommend following these tutorials in order:

### 1. [Understanding Titans](understanding-titans.md)
**Recommended first read** - Learn the core concepts and architecture behind Titans.

- What is neural long-term memory?
- How test-time adaptation works
- Overview of the four variants (MAC, MAG, MAL, LMM)
- When to use Titans vs traditional transformers

### 2. [Basic Usage](basic-usage.md)
Get hands-on experience with the Titans API.

- Creating your first Titans model
- Processing sequences
- Understanding model outputs
- Working with different variants

### 3. [Memory Mechanisms](memory-mechanisms.md)
Deep dive into how Titans memory works.

- Memory initialization and updates
- Gradient-based learning during inference
- Memory persistence across sequences
- Debugging memory behavior

### 4. [Architecture Variants](architecture-variants.md)
Complete guide to choosing and using different Titans variants.

- **MAC** (Memory as Context) - For very long sequences
- **MAG** (Memory as Gate) - Balanced performance
- **MAL** (Memory as Layer) - Simple and fast
- **LMM** (Long-term Memory Module) - Pure memory

### 5. [Training Models](training-models.md)
Learn how to train Titans models from scratch.

- Setting up training loops
- Loss functions and optimization
- Handling memory during training
- Best practices for convergence

### 6. [Test-Time Adaptation](test-time-adaptation.md)
Master the key feature that makes Titans unique.

- Enabling adaptation during inference
- Configuring adaptation parameters
- Multi-step adaptation strategies
- Measuring adaptation effectiveness

### 7. [Performance Optimization](performance-optimization.md)
Optimize Titans for your specific use case.

- Memory usage optimization
- Computational efficiency
- Gradient checkpointing
- Mixed precision training

## 🎯 By Use Case

### Language Modeling
If you're working on language modeling:
1. Start with [Basic Usage](basic-usage.md)
2. Follow [Training Models](training-models.md)
3. See [Language Modeling Example](../examples/language-modeling.md)

### Long Context Processing
For very long sequences:
1. Read [Understanding Titans](understanding-titans.md)
2. Focus on [Architecture Variants](architecture-variants.md) (MAC variant)
3. Check [Long Context Example](../examples/long-context.md)

### Research & Experimentation
For research purposes:
1. Complete [Understanding Titans](understanding-titans.md)
2. Study [Memory Mechanisms](memory-mechanisms.md)
3. Explore [Research Reproduction](../examples/research-reproduction.md)

## 💡 Tutorial Format

Each tutorial includes:

- **Learning objectives** - What you'll understand after completing it
- **Prerequisites** - What you should know before starting
- **Code examples** - Executable code you can run and modify
- **Exercises** - Practice problems to reinforce learning
- **Further reading** - Links to relevant papers and documentation

## 🚀 Quick Start Path

In a hurry? Follow this minimal path:

1. **5 minutes**: [Understanding Titans](understanding-titans.md) (concepts only)
2. **10 minutes**: [Basic Usage](basic-usage.md) (hands-on basics)
3. **15 minutes**: [Architecture Variants](architecture-variants.md) (choosing the right variant)

## 📖 Prerequisites

Before starting these tutorials, you should be familiar with:

- **Python programming** - Intermediate level
- **JAX basics** - Arrays, transformations, compilation
- **Neural networks** - Attention mechanisms, transformers
- **Deep learning** - Training loops, optimization

!!! tip "New to JAX?"
    If you're new to JAX, check out the [official JAX tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) first.

## 🛠️ Setup

Make sure you have Titans installed:

```bash
pip install titans-flax[all]
```

For the tutorials, we recommend using Jupyter notebooks:

```bash
pip install jupyter
jupyter notebook
```

All tutorial code is designed to run on both CPU and GPU. GPU is recommended for larger examples.

## 🤝 Contributing

Found an issue or want to improve a tutorial?

- [Report issues](https://github.com/mahdi-shafiei/titans-flax/issues)
- [Submit improvements](https://github.com/mahdi-shafiei/titans-flax/pulls)
- [Ask questions](https://github.com/mahdi-shafiei/titans-flax/discussions)

## 📚 Additional Resources

- [API Reference](../api/index.md) - Complete technical documentation
- [Examples](../examples/index.md) - Real-world applications
- [Research Paper](https://arxiv.org/abs/your-paper-id) - Original Titans paper
- [JAX Documentation](https://jax.readthedocs.io/) - JAX ecosystem guides
