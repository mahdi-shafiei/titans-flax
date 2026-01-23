"""Configuration management for Titans models.

This module provides structured configuration classes for managing
hyperparameters and model settings across different Titans variants.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp

from .core import TitansVariant


@dataclass
class MemoryConfig:
    """Configuration for neural memory components."""

    memory_depth: int = 2
    """Depth of neural memory network (L_M in paper)."""

    momentum_decay: float = 0.9
    """Momentum decay rate (η_t in paper)."""

    learning_rate: float = 0.01
    """Learning rate for memory updates (θ_t in paper)."""

    forget_rate: float = 0.1
    """Forgetting gate rate (α_t in paper)."""

    num_persistent_tokens: int = 64
    """Number of persistent memory tokens (N_p in paper)."""


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""

    num_heads: int = 8
    """Number of attention heads."""

    dropout_rate: float = 0.0
    """Dropout rate for attention weights."""

    window_size: int | None = None
    """Sliding window size, None for full attention."""

    use_causal_mask: bool = True
    """Whether to use causal masking."""


@dataclass
class ArchitectureConfig:
    """Configuration for overall architecture."""

    d_model: int = 128
    """Model dimension."""

    chunk_size: int = 512
    """Chunk size for MAC variant."""

    dtype: str = "float32"
    """Data type for computations."""

    def get_dtype(self) -> Any:
        """Get JAX dtype from string."""
        return getattr(jnp, self.dtype)


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    learning_rate: float = 1e-4
    """Learning rate for model parameters."""

    batch_size: int = 8
    """Training batch size."""

    max_seq_length: int = 1024
    """Maximum sequence length during training."""

    num_epochs: int = 10
    """Number of training epochs."""

    gradient_clip_norm: float | None = 1.0
    """Gradient clipping norm, None to disable."""

    warmup_steps: int = 1000
    """Number of warmup steps for learning rate schedule."""

    save_interval: int = 1000
    """Interval for saving checkpoints."""

    eval_interval: int = 500
    """Interval for evaluation."""

    log_interval: int = 100
    """Interval for logging."""


@dataclass
class TitansConfig:
    """Complete configuration for Titans models."""

    variant: TitansVariant = "mac"
    """Which Titans variant to use."""

    seed: int = 42
    """Random seed for reproducibility."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    """Memory-related configuration."""

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    """Attention-related configuration."""

    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    """Architecture-related configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training-related configuration."""

    # Variant-specific overrides
    def __post_init__(self) -> None:
        """Apply variant-specific configuration overrides."""
        if self.variant == "mac":
            # MAC uses full causal attention with chunking
            self.attention.window_size = None

        elif self.variant == "mag":
            # MAG uses sliding window attention
            if self.attention.window_size is None:
                self.attention.window_size = 256

        elif self.variant == "mal":
            # MAL uses sliding window attention
            if self.attention.window_size is None:
                self.attention.window_size = 256

        elif self.variant == "lmm":
            # LMM doesn't use attention
            pass

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""

        def _dataclass_to_dict(obj: Any) -> Any:
            """Convert dataclass to dictionary recursively."""
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for field_name in obj.__dataclass_fields__:
                    field_value = getattr(obj, field_name)
                    result[field_name] = _dataclass_to_dict(field_value)
                return result
            if isinstance(obj, list):
                return [_dataclass_to_dict(item) for item in obj]
            if isinstance(obj, dict):
                return {key: _dataclass_to_dict(value) for key, value in obj.items()}
            return obj

        result = _dataclass_to_dict(self)
        # Ensure we return a dict[str, Any] as expected
        if isinstance(result, dict):
            return result
        # This should never happen for a dataclass, but for type safety
        return {"error": "unexpected_type"}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TitansConfig":
        """Create configuration from dictionary."""
        # Extract nested configs
        memory_config = MemoryConfig(**config_dict.get("memory", {}))
        attention_config = AttentionConfig(**config_dict.get("attention", {}))
        architecture_config = ArchitectureConfig(**config_dict.get("architecture", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        # Create main config
        main_config = {
            k: v
            for k, v in config_dict.items()
            if k not in ["memory", "attention", "architecture", "training"]
        }

        return cls(
            memory=memory_config,
            attention=attention_config,
            architecture=architecture_config,
            training=training_config,
            **main_config,
        )

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TitansConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


# Predefined configurations for different use cases
def get_small_config() -> TitansConfig:
    """Get configuration for small model (quick experimentation)."""
    return TitansConfig(
        variant="mac",
        architecture=ArchitectureConfig(
            d_model=64,
            chunk_size=128,
        ),
        memory=MemoryConfig(
            memory_depth=2,
            num_persistent_tokens=32,
        ),
        attention=AttentionConfig(
            num_heads=4,
        ),
        training=TrainingConfig(
            batch_size=4,
            max_seq_length=256,
        ),
    )


def get_medium_config() -> TitansConfig:
    """Get configuration for medium model (balanced performance)."""
    return TitansConfig(
        variant="mac",
        architecture=ArchitectureConfig(
            d_model=256,
            chunk_size=512,
        ),
        memory=MemoryConfig(
            memory_depth=3,
            num_persistent_tokens=64,
        ),
        attention=AttentionConfig(
            num_heads=8,
        ),
        training=TrainingConfig(
            batch_size=8,
            max_seq_length=1024,
        ),
    )


def get_large_config() -> TitansConfig:
    """Get configuration for large model (high capacity)."""
    return TitansConfig(
        variant="mac",
        architecture=ArchitectureConfig(
            d_model=512,
            chunk_size=1024,
        ),
        memory=MemoryConfig(
            memory_depth=4,
            num_persistent_tokens=128,
        ),
        attention=AttentionConfig(
            num_heads=16,
        ),
        training=TrainingConfig(
            batch_size=4,
            max_seq_length=2048,
        ),
    )


def get_long_context_config() -> TitansConfig:
    """Get configuration optimized for very long contexts."""
    return TitansConfig(
        variant="mac",  # MAC is best for very long sequences
        architecture=ArchitectureConfig(
            d_model=256,
            chunk_size=2048,  # Larger chunks
        ),
        memory=MemoryConfig(
            memory_depth=4,  # Deeper memory
            num_persistent_tokens=256,  # More persistent memory
            forget_rate=0.05,  # Less aggressive forgetting
        ),
        attention=AttentionConfig(
            num_heads=8,
        ),
        training=TrainingConfig(
            batch_size=2,  # Smaller batch for longer sequences
            max_seq_length=8192,
        ),
    )


def get_efficient_config() -> TitansConfig:
    """Get configuration optimized for efficiency."""
    return TitansConfig(
        variant="mag",  # MAG is efficient
        architecture=ArchitectureConfig(
            d_model=128,
        ),
        memory=MemoryConfig(
            memory_depth=2,  # Shallow memory
            num_persistent_tokens=32,
        ),
        attention=AttentionConfig(
            num_heads=4,
            window_size=128,  # Small window
        ),
        training=TrainingConfig(
            batch_size=16,
            max_seq_length=512,
        ),
    )


def get_memory_intensive_config() -> TitansConfig:
    """Get configuration for memory-intensive tasks."""
    return TitansConfig(
        variant="lmm",  # Pure memory
        architecture=ArchitectureConfig(
            d_model=256,
        ),
        memory=MemoryConfig(
            memory_depth=6,  # Very deep memory
            num_persistent_tokens=512,  # Lots of persistent memory
            momentum_decay=0.95,  # Higher momentum
            learning_rate=0.005,  # Lower learning rate
        ),
        training=TrainingConfig(
            batch_size=4,
            max_seq_length=1024,
        ),
    )


# Configuration registry
PREDEFINED_CONFIGS = {
    "small": get_small_config,
    "medium": get_medium_config,
    "large": get_large_config,
    "long_context": get_long_context_config,
    "efficient": get_efficient_config,
    "memory_intensive": get_memory_intensive_config,
}


def get_config(name: str) -> TitansConfig:
    """Get a predefined configuration by name.

    Args:
        name: Name of the configuration

    Returns:
        TitansConfig instance

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in PREDEFINED_CONFIGS:
        available = list(PREDEFINED_CONFIGS.keys())
        msg = f"Unknown configuration '{name}'. Available: {available}"
        raise ValueError(msg)

    return PREDEFINED_CONFIGS[name]()


def create_config_for_task(
    task_type: Literal[
        "language_modeling", "time_series", "long_sequence", "classification"
    ],
    model_size: Literal["small", "medium", "large"] = "medium",
) -> TitansConfig:
    """Create configuration optimized for specific task type.

    Args:
        task_type: Type of task
        model_size: Size of the model

    Returns:
        TitansConfig optimized for the task
    """
    # Start with base config
    if model_size == "small":
        config = get_small_config()
    elif model_size == "medium":
        config = get_medium_config()
    else:
        config = get_large_config()

    # Task-specific modifications
    if task_type == "language_modeling":
        # Good for autoregressive language modeling
        config.variant = "mac"
        config.architecture.chunk_size = 1024
        config.memory.memory_depth = 3

    elif task_type == "time_series":
        # Good for time series forecasting
        config.variant = "mal"
        config.attention.window_size = 256
        config.memory.momentum_decay = 0.95  # Better for temporal patterns

    elif task_type == "long_sequence":
        # Optimized for very long sequences
        config.variant = "mac"
        config.architecture.chunk_size = 2048
        config.memory.memory_depth = 4
        config.memory.num_persistent_tokens = 256

    elif task_type == "classification":
        # Good for sequence classification
        config.variant = "mag"
        config.attention.window_size = 512
        config.memory.memory_depth = 2

    return config
