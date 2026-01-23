"""Titans: Learning to Memorize at Test Time.

A complete implementation of the Titans architecture from the paper
"Titans: Learning to Memorize at Test Time" using Flax NNX.
"""

import jax

# Core model imports
# Attention mechanisms
from .attention_modules import (
    CausalAttention,
    ConvolutionalLayer,
    SlidingWindowAttention,
)
from .benchmark import TitansBenchmark

# Configuration imports
from .config import TitansConfig, create_config_for_task, get_config
from .core import TitansModel, TitansVariant, create_titans_model, titans_forward

# Memory components
from .neural_memory import MemoryState, NeuralLongTermMemory, PersistentMemory

# Optimization utilities
from .optimization_utilities import (
    GradientCheckpointing,
    InferenceOptimizer,
    MemoryOptimizer,
    PerformanceProfiler,
    QuantizationUtils,
)

# Parameter validation utilities (DRY principle)
from .param_validation import validate_3d_input, validate_positive, validate_range

# Serialization
from .serialization import MemoryStateManager, ModelExporter, TitansCheckpoint

# Training
from .training import (
    LearningRateScheduler,
    TitansTrainer,
    TrainingMetrics,
    create_synthetic_dataset,
)

# Variants
from .variants import TitansLMM, TitansMAC, TitansMAG, TitansMAL

# Public API
__all__ = [
    # Attention mechanisms
    "CausalAttention",
    "ConvolutionalLayer",
    # Optimization utilities
    "GradientCheckpointing",
    "InferenceOptimizer",
    # Training
    "LearningRateScheduler",
    "MemoryOptimizer",
    # Memory components
    "MemoryState",
    # Serialization
    "MemoryStateManager",
    "ModelExporter",
    "NeuralLongTermMemory",
    "PerformanceProfiler",
    "PersistentMemory",
    "QuantizationUtils",
    "SlidingWindowAttention",
    # Utilities
    "TitansBenchmark",
    "TitansCheckpoint",
    # Configuration
    "TitansConfig",
    # Variants
    "TitansLMM",
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    # Core functionality
    "TitansModel",
    "TitansTrainer",
    "TitansVariant",
    "TrainingMetrics",
    # Version
    "__version__",
    "create_config_for_task",
    "create_synthetic_dataset",
    "create_titans_model",
    "get_config",
    "titans_forward",
    "validate_3d_input",
    # Validation
    "validate_positive",
    "validate_range",
]

__version__ = "0.1.0"


def demo_basic_usage() -> None:
    """Demonstrate basic Titans model usage."""
    # Create model instance
    model = create_titans_model(variant="mac", d_model=128, chunk_size=64, seed=42)

    # Example forward pass
    example_input = jax.random.normal(jax.random.key(0), (2, 100, 128))
    model(example_input, deterministic=False)
