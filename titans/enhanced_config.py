"""Enhanced configuration with parallel training support.

This extends the existing TitansConfig to include the parallel training
algorithm from Section 3.2 as configurable options.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp

from .config import get_small_config
from .core import TitansVariant


@dataclass
class ParallelTrainingConfig:
    """Configuration for parallel training algorithm (Section 3.2)."""

    enabled: bool = False
    """Whether to enable parallel training mode."""

    chunk_size: int = 64
    """Size of chunks for tensorized mini-batch gradient descent."""

    use_chunk_level_params: bool = False
    """Whether to use chunk-level vs token-level parameters."""

    min_sequence_length: int = 256
    """Minimum sequence length to trigger parallel training."""

    auto_enable_threshold: int = 4096
    """Automatically enable parallel training for sequences longer than this."""


@dataclass
class MemoryConfig:
    """Enhanced memory configuration with parallel training support."""

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

    parallel_training: ParallelTrainingConfig = field(
        default_factory=ParallelTrainingConfig
    )
    """Configuration for parallel training algorithm."""


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
class EnhancedTitansConfig:
    """Enhanced Titans configuration with parallel training support."""

    variant: TitansVariant = "mac"
    """Which Titans variant to use."""

    seed: int = 42
    """Random seed for reproducibility."""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    """Memory-related configuration with parallel training support."""

    attention: AttentionConfig = field(default_factory=AttentionConfig)
    """Attention-related configuration."""

    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    """Architecture-related configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training-related configuration."""

    def __post_init__(self) -> None:
        """Apply variant-specific configuration overrides and validate parallel training."""
        # Apply variant-specific overrides
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

        # Auto-enable parallel training for long sequences
        if (
            self.training.max_seq_length
            >= self.memory.parallel_training.auto_enable_threshold
            and not self.memory.parallel_training.enabled
        ):
            self.memory.parallel_training.enabled = True

        # Validate parallel training configuration
        self._validate_parallel_config()

    def _validate_parallel_config(self) -> None:
        """Validate parallel training configuration."""
        parallel_config = self.memory.parallel_training

        if parallel_config.enabled:
            # Validate chunk size
            if parallel_config.chunk_size <= 0:
                msg = f"chunk_size must be positive, got {parallel_config.chunk_size}"
                raise ValueError(msg)

            if parallel_config.chunk_size > self.training.max_seq_length:
                pass

            # Validate minimum sequence length
            if parallel_config.min_sequence_length <= 0:
                msg = f"min_sequence_length must be positive, got {parallel_config.min_sequence_length}"
                raise ValueError(msg)

            # Recommend optimal chunk size
            recommended_chunk_size = min(64, self.training.max_seq_length // 4)
            if parallel_config.chunk_size > recommended_chunk_size * 2:
                pass

    def enable_parallel_training(
        self,
        chunk_size: int = 64,
        chunk_level_params: bool = False,
        min_sequence_length: int | None = None,
    ) -> None:
        """Enable parallel training with specified parameters.

        Args:
            chunk_size: Size of chunks for parallel processing
            chunk_level_params: Whether to use chunk-level parameters
            min_sequence_length: Minimum sequence length to trigger parallel mode
        """
        self.memory.parallel_training.enabled = True
        self.memory.parallel_training.chunk_size = chunk_size
        self.memory.parallel_training.use_chunk_level_params = chunk_level_params

        if min_sequence_length is not None:
            self.memory.parallel_training.min_sequence_length = min_sequence_length

        # Re-validate configuration
        self._validate_parallel_config()

    def disable_parallel_training(self) -> None:
        """Disable parallel training."""
        self.memory.parallel_training.enabled = False

    def get_parallel_training_summary(self) -> str:
        """Get a summary of parallel training configuration."""
        config = self.memory.parallel_training

        if not config.enabled:
            return "Parallel training: Disabled"

        return (
            f"Parallel training: Enabled\n"
            f"  Chunk size: {config.chunk_size}\n"
            f"  Chunk-level params: {config.use_chunk_level_params}\n"
            f"  Min sequence length: {config.min_sequence_length}\n"
            f"  Auto-enable threshold: {config.auto_enable_threshold}"
        )

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
        if isinstance(result, dict):
            return result
        return {"error": "unexpected_type"}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "EnhancedTitansConfig":
        """Create configuration from dictionary."""
        # Extract parallel training config
        memory_dict = config_dict.get("memory", {})
        parallel_dict = memory_dict.get("parallel_training", {})

        parallel_config = ParallelTrainingConfig(**parallel_dict)

        # Create memory config
        memory_config_dict = {
            k: v for k, v in memory_dict.items() if k != "parallel_training"
        }
        memory_config = MemoryConfig(
            **memory_config_dict, parallel_training=parallel_config
        )

        # Extract other configs
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
    def load(cls, path: Path) -> "EnhancedTitansConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)


# Enhanced configuration presets with parallel training


def get_small_config_with_parallel() -> EnhancedTitansConfig:
    """Get small configuration with parallel training for experimentation."""
    return EnhancedTitansConfig(
        variant="mac",
        architecture=ArchitectureConfig(
            d_model=64,
            chunk_size=128,
        ),
        memory=MemoryConfig(
            memory_depth=2,
            num_persistent_tokens=32,
            parallel_training=ParallelTrainingConfig(
                enabled=True,
                chunk_size=32,
                use_chunk_level_params=False,
                min_sequence_length=64,
            ),
        ),
        attention=AttentionConfig(
            num_heads=4,
        ),
        training=TrainingConfig(
            batch_size=4,
            max_seq_length=256,
        ),
    )


def get_long_context_config_with_parallel() -> EnhancedTitansConfig:
    """Get configuration optimized for very long contexts with parallel training."""
    return EnhancedTitansConfig(
        variant="mac",  # MAC is best for very long sequences
        architecture=ArchitectureConfig(
            d_model=256,
            chunk_size=2048,  # Larger chunks
        ),
        memory=MemoryConfig(
            memory_depth=4,  # Deeper memory
            num_persistent_tokens=256,  # More persistent memory
            forget_rate=0.05,  # Less aggressive forgetting
            parallel_training=ParallelTrainingConfig(
                enabled=True,
                chunk_size=128,  # Optimal for long sequences
                use_chunk_level_params=True,  # More efficient for very long sequences
                min_sequence_length=512,
                auto_enable_threshold=1024,
            ),
        ),
        attention=AttentionConfig(
            num_heads=8,
        ),
        training=TrainingConfig(
            batch_size=2,  # Smaller batch for longer sequences
            max_seq_length=8192,
        ),
    )


def get_efficient_config_with_parallel() -> EnhancedTitansConfig:
    """Get configuration optimized for efficiency with selective parallel training."""
    return EnhancedTitansConfig(
        variant="mag",  # MAG is efficient
        architecture=ArchitectureConfig(
            d_model=128,
        ),
        memory=MemoryConfig(
            memory_depth=2,  # Shallow memory
            num_persistent_tokens=32,
            parallel_training=ParallelTrainingConfig(
                enabled=False,  # Disabled by default for efficiency
                chunk_size=64,
                use_chunk_level_params=True,
                min_sequence_length=512,  # Only for longer sequences
                auto_enable_threshold=1024,
            ),
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


# Enhanced configuration registry
ENHANCED_CONFIGS = {
    "small_parallel": get_small_config_with_parallel,
    "long_context_parallel": get_long_context_config_with_parallel,
    "efficient_parallel": get_efficient_config_with_parallel,
}


def get_enhanced_config(name: str) -> EnhancedTitansConfig:
    """Get an enhanced configuration by name.

    Args:
        name: Name of the configuration

    Returns:
        EnhancedTitansConfig instance

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in ENHANCED_CONFIGS:
        available = list(ENHANCED_CONFIGS.keys())
        msg = f"Unknown enhanced configuration '{name}'. Available: {available}"
        raise ValueError(msg)

    return ENHANCED_CONFIGS[name]()


def _convert_regular_to_enhanced(regular_config: Any) -> EnhancedTitansConfig:
    """Convert a regular TitansConfig to EnhancedTitansConfig."""
    return EnhancedTitansConfig(
        variant=regular_config.variant,
        seed=regular_config.seed,
        memory=MemoryConfig(
            memory_depth=regular_config.memory.memory_depth,
            momentum_decay=regular_config.memory.momentum_decay,
            learning_rate=regular_config.memory.learning_rate,
            forget_rate=regular_config.memory.forget_rate,
            num_persistent_tokens=regular_config.memory.num_persistent_tokens,
            parallel_training=ParallelTrainingConfig(enabled=False),
        ),
        attention=AttentionConfig(
            num_heads=regular_config.attention.num_heads,
            dropout_rate=regular_config.attention.dropout_rate,
            window_size=regular_config.attention.window_size,
            use_causal_mask=regular_config.attention.use_causal_mask,
        ),
        architecture=ArchitectureConfig(
            d_model=regular_config.architecture.d_model,
            chunk_size=regular_config.architecture.chunk_size,
            dtype=regular_config.architecture.dtype,
        ),
        training=TrainingConfig(
            learning_rate=regular_config.training.learning_rate,
            batch_size=regular_config.training.batch_size,
            max_seq_length=regular_config.training.max_seq_length,
            num_epochs=regular_config.training.num_epochs,
            gradient_clip_norm=regular_config.training.gradient_clip_norm,
            warmup_steps=regular_config.training.warmup_steps,
            save_interval=regular_config.training.save_interval,
            eval_interval=regular_config.training.eval_interval,
            log_interval=regular_config.training.log_interval,
        ),
    )


def _get_base_config(
    model_size: Literal["small", "medium", "large"], enable_parallel: bool
) -> EnhancedTitansConfig:
    """Get base configuration for the given model size."""
    if model_size == "small":
        if enable_parallel:
            return get_small_config_with_parallel()
        return _convert_regular_to_enhanced(get_small_config())

    if model_size == "large":
        config = get_long_context_config_with_parallel()
        if not enable_parallel:
            config.disable_parallel_training()
        return config

    # medium
    config = get_efficient_config_with_parallel()
    if enable_parallel:
        config.enable_parallel_training()
    return config


def _apply_task_config(
    config: EnhancedTitansConfig,
    task_type: Literal[
        "language_modeling", "time_series", "long_sequence", "classification"
    ],
    enable_parallel: bool,
) -> None:
    """Apply task-specific configuration in-place."""
    if task_type == "language_modeling":
        config.variant = "mac"
        config.architecture.chunk_size = 1024
        config.memory.memory_depth = 3
        if enable_parallel:
            config.memory.parallel_training.chunk_size = 128

    elif task_type == "time_series":
        config.variant = "mal"
        config.attention.window_size = 256
        config.memory.momentum_decay = 0.95
        if enable_parallel:
            config.memory.parallel_training.chunk_size = 64

    elif task_type == "long_sequence":
        config.variant = "mac"
        config.architecture.chunk_size = 2048
        config.memory.memory_depth = 4
        config.memory.num_persistent_tokens = 256
        if enable_parallel:
            config.memory.parallel_training.enabled = True
            config.memory.parallel_training.chunk_size = 128
            config.memory.parallel_training.use_chunk_level_params = True

    elif task_type == "classification":
        config.variant = "mag"
        config.attention.window_size = 512
        config.memory.memory_depth = 2
        if enable_parallel:
            config.memory.parallel_training.min_sequence_length = 256


def create_config_for_task_with_parallel(
    task_type: Literal[
        "language_modeling", "time_series", "long_sequence", "classification"
    ],
    model_size: Literal["small", "medium", "large"] = "medium",
    enable_parallel: bool = True,
) -> EnhancedTitansConfig:
    """Create enhanced configuration optimized for specific task type.

    Args:
        task_type: Type of task
        model_size: Size of the model
        enable_parallel: Whether to enable parallel training

    Returns:
        EnhancedTitansConfig optimized for the task
    """
    config = _get_base_config(model_size, enable_parallel)
    _apply_task_config(config, task_type, enable_parallel)
    return config


# Backward compatibility alias
TitansConfig = EnhancedTitansConfig
