# Attention Modules

Attention mechanisms used in Titans models including causal attention, sliding window attention, and multi-head attention.

::: titans.attention_modules
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2

## Key Classes

- **CausalAttention** - Standard causal attention for autoregressive modeling
- **SlidingWindowAttention** - Sliding window attention for efficiency
- **MultiHeadAttention** - Multi-head attention wrapper
- **ConvolutionalLayer** - Convolutional layer for local processing

## Features

- Memory-efficient attention implementations
- Support for very long sequences
- Causal masking for autoregressive tasks
- Configurable attention patterns
