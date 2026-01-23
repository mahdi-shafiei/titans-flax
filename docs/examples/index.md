# Examples

This section contains practical, real-world examples of using Titans for various tasks. Each example includes complete, runnable code and detailed explanations.

## 🚀 Quick Examples

### Basic Model Creation
```python
import jax
import jax.numpy as jnp
from titans.core import create_titans_model

# Create model optimized for long sequences
model = create_titans_model(variant="mac", d_model=512, memory_depth=8)

# Process very long sequence
x = jax.random.normal(jax.random.key(42), (1, 1024, 512))
output = model(x, deterministic=False)

print(f"Processed sequence: {x.shape} -> {output.shape}")
print(f"Memory adaptation: {model.get_memory_norm():.4f}")
```

### Test-Time Adaptation
```python
# Reset memory for new task
model.reset_memory()

# Process related sequences - memory adapts
sequences = [jax.random.normal(jax.random.key(i), (1, 512, 512)) for i in range(5)]

for i, seq in enumerate(sequences):
    output = model(seq, deterministic=False)
    memory_norm = model.get_memory_norm()
    print(f"Step {i+1}: Memory norm = {memory_norm:.4f}")
```

## 📚 Detailed Examples

### [Language Modeling](language-modeling.md)
Complete language modeling pipeline with Titans.

**What you'll learn:**
- Training a language model from scratch
- Handling tokenization and data loading
- Using test-time adaptation for domain adaptation
- Generating text with memory-enhanced models

**Key features:**
- Full training loop implementation
- Evaluation metrics and perplexity
- Text generation with different strategies
- Memory analysis during generation

### [Long Context Tasks](long-context.md)
Processing very long sequences (up to 2M tokens).

**What you'll learn:**
- Choosing the right variant for long contexts
- Memory efficiency techniques
- Handling extremely long documents
- Chunking strategies and memory persistence

**Key features:**
- Document summarization
- Long-range dependency tasks
- Memory usage optimization
- Performance benchmarking

### [Memory Analysis](memory-analysis.md)
Deep dive into understanding Titans memory behavior.

**What you'll learn:**
- Visualizing memory evolution
- Understanding memory adaptation patterns
- Debugging memory issues
- Analyzing memory effectiveness

**Key features:**
- Memory visualization tools
- Adaptation trajectory analysis
- Memory similarity metrics
- Diagnostic utilities

### [Benchmarking](benchmarking.md)
Full performance evaluation and comparison.

**What you'll learn:**
- Benchmarking different variants
- Performance profiling
- Memory usage analysis
- Scaling behavior studies

**Key features:**
- Multi-variant comparisons
- Scaling analysis tools
- Performance visualization
- Optimization recommendations

### [Research Reproduction](research-reproduction.md)
Reproducing results from the original Titans paper.

**What you'll learn:**
- Exact paper configurations
- Experimental setups
- Result reproduction
- Research methodology

**Key features:**
- Paper experiment reproduction
- Dataset preparation
- Evaluation metrics
- Result analysis and visualization

## 🎯 Examples by Use Case

=== "Natural Language Processing"
    - [Language Modeling](language-modeling.md)
    - [Long Context Tasks](long-context.md)
    - Text generation and completion
    - Domain adaptation

=== "Research & Development"
    - [Research Reproduction](research-reproduction.md)
    - [Memory Analysis](memory-analysis.md)
    - Architecture comparisons
    - Ablation studies

=== "Performance & Optimization"
    - [Benchmarking](benchmarking.md)
    - Memory efficiency analysis
    - Scaling studies
    - Production deployment

=== "Educational"
    - Step-by-step tutorials
    - Concept illustrations
    - Interactive demos
    - Debugging guides

## 🛠️ Running the Examples

### Setup
All examples require the full Titans installation:

```bash
pip install titans-flax[all]
```

Some examples may need additional dependencies:

```bash
pip install datasets transformers  # For NLP examples
pip install matplotlib seaborn     # For visualization
pip install jupyter                # For interactive examples
```

### GPU Recommendations

| Example | Minimum GPU | Recommended GPU |
|---------|-------------|-----------------|
| Language Modeling | 8GB | 16GB+ |
| Long Context Tasks | 16GB | 24GB+ |
| Memory Analysis | 4GB | 8GB+ |
| Benchmarking | 8GB | 16GB+ |
| Research Reproduction | 16GB | 32GB+ |

### CPU Mode
All examples can run on CPU, though performance will be slower:

```python
import jax
# Force CPU usage
jax.config.update('jax_platform_name', 'cpu')
```

## 📊 Example Structure

Each example follows this structure:

```
example-name/
├── README.md              # Overview and instructions
├── example_script.py      # Main executable script
├── config.yaml           # Configuration file
├── requirements.txt       # Additional dependencies
├── data/                 # Sample data (if needed)
├── outputs/              # Generated outputs
└── analysis/             # Analysis notebooks
```

## 🔬 Advanced Examples

### Custom Variants
Learn how to implement custom Titans variants:

```python
from titans import TitansBase
import flax.nnx as nnx

class CustomTitansVariant(TitansBase):
    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.custom_layer = nnx.Linear(d_model, d_model)

    def __call__(self, x, deterministic=True):
        # Custom implementation
        memory_output = self.memory_module(x, deterministic)
        return self.custom_layer(memory_output)
```

### Multi-Modal Applications
Using Titans with different modalities:

```python
# Image + Text processing
from titans.core import create_titans_model

text_model = create_titans_model(variant="mac", d_model=512)
image_encoder = CustomImageEncoder()

def multimodal_forward(text, images):
    text_features = text_model(text)
    image_features = image_encoder(images)
    return combine_features(text_features, image_features)
```

## 📈 Performance Tips

1. **Use the right variant** - MAC for very long sequences, MAG for balanced performance
2. **Enable gradient checkpointing** - For memory efficiency during training
3. **Use mixed precision** - bfloat16 for faster training
4. **Batch efficiently** - Use padding and attention masks
5. **Monitor memory usage** - Track memory adaptation effectiveness

## 🤝 Contributing Examples

We welcome contributions! To add a new example:

1. Fork the repository
2. Create a new example directory
3. Follow the standard structure
4. Include complete documentation
5. Add tests and verification
6. Submit a pull request

See our [Contributing Guide](../CONTRIBUTING.md) for detailed instructions.

## 🆘 Getting Help

If you encounter issues with examples:

1. Check the example's README for specific requirements
2. Verify your environment matches the recommendations
3. Look for similar issues in [GitHub Issues](https://github.com/mahdi-shafiei/titans-flax/issues)
4. Ask questions in [Discussions](https://github.com/mahdi-shafiei/titans-flax/discussions)

---

!!! tip "Start with Language Modeling"
    New to Titans? We recommend starting with the [Language Modeling](language-modeling.md) example as it covers most core concepts.
