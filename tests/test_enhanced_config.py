"""Tests for enhanced configuration with parallel training support.

This module tests the enhanced configuration system including parallel training
configuration and validation logic.
"""

import json
from pathlib import Path
import tempfile

import jax.numpy as jnp
import pytest

from titans.enhanced_config import (
    ArchitectureConfig,
    AttentionConfig,
    EnhancedTitansConfig,
    MemoryConfig,
    ParallelTrainingConfig,
    TitansConfig,
    TrainingConfig,
    create_config_for_task_with_parallel,
    get_efficient_config_with_parallel,
    get_enhanced_config,
    get_long_context_config_with_parallel,
    get_small_config_with_parallel,
)


class TestParallelTrainingConfig:
    """Tests for ParallelTrainingConfig dataclass."""

    def test_default_initialization(self) -> None:
        """Test default ParallelTrainingConfig initialization."""
        config = ParallelTrainingConfig()

        assert config.enabled is False
        assert config.chunk_size == 64
        assert config.use_chunk_level_params is False
        assert config.min_sequence_length == 256
        assert config.auto_enable_threshold == 4096

    def test_custom_initialization(self) -> None:
        """Test custom ParallelTrainingConfig initialization."""
        config = ParallelTrainingConfig(
            enabled=True,
            chunk_size=128,
            use_chunk_level_params=True,
            min_sequence_length=512,
            auto_enable_threshold=2048,
        )

        assert config.enabled is True
        assert config.chunk_size == 128
        assert config.use_chunk_level_params is True
        assert config.min_sequence_length == 512
        assert config.auto_enable_threshold == 2048


class TestEnhancedMemoryConfig:
    """Tests for enhanced MemoryConfig with parallel training."""

    def test_default_initialization(self) -> None:
        """Test default MemoryConfig initialization."""
        config = MemoryConfig()

        assert config.memory_depth == 2
        assert config.momentum_decay == 0.9
        assert config.learning_rate == 0.01
        assert config.forget_rate == 0.1
        assert config.num_persistent_tokens == 64
        assert isinstance(config.parallel_training, ParallelTrainingConfig)
        assert config.parallel_training.enabled is False

    def test_custom_parallel_config(self) -> None:
        """Test MemoryConfig with custom parallel training config."""
        parallel_config = ParallelTrainingConfig(
            enabled=True,
            chunk_size=128,
            use_chunk_level_params=True,
        )

        config = MemoryConfig(
            memory_depth=4,
            parallel_training=parallel_config,
        )

        assert config.memory_depth == 4
        assert config.parallel_training.enabled is True
        assert config.parallel_training.chunk_size == 128
        assert config.parallel_training.use_chunk_level_params is True


class TestEnhancedTitansConfig:
    """Tests for EnhancedTitansConfig."""

    def test_default_initialization(self) -> None:
        """Test default EnhancedTitansConfig initialization."""
        config = EnhancedTitansConfig()

        assert config.variant == "mac"
        assert config.seed == 42
        assert isinstance(config.memory, MemoryConfig)
        assert isinstance(config.attention, AttentionConfig)
        assert isinstance(config.architecture, ArchitectureConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_variant_specific_overrides(self) -> None:
        """Test that __post_init__ applies variant-specific overrides."""
        # Test MAC variant
        mac_config = EnhancedTitansConfig(variant="mac")
        assert mac_config.attention.window_size is None

        # Test MAG variant
        mag_config = EnhancedTitansConfig(variant="mag")
        assert mag_config.attention.window_size == 256

        # Test MAL variant
        mal_config = EnhancedTitansConfig(variant="mal")
        assert mal_config.attention.window_size == 256

        # Test LMM variant
        EnhancedTitansConfig(variant="lmm")
        # LMM doesn't change window_size from default

    def test_auto_enable_parallel_training(self) -> None:
        """Test auto-enabling parallel training for long sequences."""
        config = EnhancedTitansConfig()
        config.training.max_seq_length = (
            5000  # Above default auto_enable_threshold (4096)
        )
        config.memory.parallel_training.enabled = False

        # Trigger __post_init__ manually or create new config
        config.__post_init__()

        assert config.memory.parallel_training.enabled is True

    def test_parallel_config_validation_chunk_size(self) -> None:
        """Test parallel configuration validation for chunk size."""
        config = EnhancedTitansConfig()
        config.memory.parallel_training.enabled = True
        config.memory.parallel_training.chunk_size = 0
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config._validate_parallel_config()

    def test_parallel_config_validation_min_sequence(self) -> None:
        """Test parallel configuration validation for min_sequence_length."""
        config = EnhancedTitansConfig()
        config.memory.parallel_training.enabled = True
        config.memory.parallel_training.min_sequence_length = 0
        with pytest.raises(ValueError, match="min_sequence_length must be positive"):
            config._validate_parallel_config()

    def test_enable_parallel_training(self) -> None:
        """Test enable_parallel_training method."""
        config = EnhancedTitansConfig()
        assert config.memory.parallel_training.enabled is False

        config.enable_parallel_training(
            chunk_size=128, chunk_level_params=True, min_sequence_length=512
        )

        assert config.memory.parallel_training.enabled is True
        assert config.memory.parallel_training.chunk_size == 128
        assert config.memory.parallel_training.use_chunk_level_params is True
        assert config.memory.parallel_training.min_sequence_length == 512

    def test_disable_parallel_training(self) -> None:
        """Test disable_parallel_training method."""
        config = EnhancedTitansConfig()
        config.enable_parallel_training()
        assert config.memory.parallel_training.enabled is True

        config.disable_parallel_training()
        assert config.memory.parallel_training.enabled is False

    def test_get_parallel_training_summary(self) -> None:
        """Test get_parallel_training_summary method."""
        # Test disabled
        config = EnhancedTitansConfig()
        summary = config.get_parallel_training_summary()
        assert "Disabled" in summary

        # Test enabled
        config.enable_parallel_training(chunk_size=64)
        summary = config.get_parallel_training_summary()
        assert "Enabled" in summary
        assert "Chunk size: 64" in summary

    def test_to_dict_and_from_dict(self) -> None:
        """Test serialization to/from dictionary."""
        original_config = EnhancedTitansConfig(
            variant="mag",
            seed=123,
        )
        original_config.enable_parallel_training(chunk_size=128)

        # Convert to dict
        config_dict = original_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["variant"] == "mag"
        assert config_dict["seed"] == 123
        assert config_dict["memory"]["parallel_training"]["enabled"] is True
        assert config_dict["memory"]["parallel_training"]["chunk_size"] == 128

        # Convert back from dict
        restored_config = EnhancedTitansConfig.from_dict(config_dict)
        assert restored_config.variant == "mag"
        assert restored_config.seed == 123
        assert restored_config.memory.parallel_training.enabled is True
        assert restored_config.memory.parallel_training.chunk_size == 128

    def test_save_and_load(self) -> None:
        """Test saving and loading configuration to/from file."""
        original_config = EnhancedTitansConfig(variant="mal", seed=456)
        original_config.enable_parallel_training(chunk_size=256)

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            # Save config
            original_config.save(config_path)
            assert config_path.exists()

            # Verify JSON content
            with open(config_path) as f:
                json_content = json.load(f)
            assert json_content["variant"] == "mal"
            assert json_content["memory"]["parallel_training"]["chunk_size"] == 256

            # Load config
            loaded_config = EnhancedTitansConfig.load(config_path)
            assert loaded_config.variant == "mal"
            assert loaded_config.seed == 456
            assert loaded_config.memory.parallel_training.enabled is True
            assert loaded_config.memory.parallel_training.chunk_size == 256


class TestPredefinedConfigs:
    """Tests for predefined enhanced configurations."""

    def test_get_small_config_with_parallel(self) -> None:
        """Test small config with parallel training."""
        config = get_small_config_with_parallel()

        assert config.variant == "mac"
        assert config.architecture.d_model == 64
        assert config.memory.parallel_training.enabled is True
        assert config.memory.parallel_training.chunk_size == 32
        assert config.training.batch_size == 4

    def test_get_long_context_config_with_parallel(self) -> None:
        """Test long context config with parallel training."""
        config = get_long_context_config_with_parallel()

        assert config.variant == "mac"
        assert config.architecture.d_model == 256
        assert config.architecture.chunk_size == 2048
        assert config.memory.parallel_training.enabled is True
        assert config.memory.parallel_training.chunk_size == 128
        assert config.memory.parallel_training.use_chunk_level_params is True
        assert config.training.max_seq_length == 8192

    def test_get_efficient_config_with_parallel(self) -> None:
        """Test efficient config with selective parallel training."""
        config = get_efficient_config_with_parallel()

        assert config.variant == "mag"
        assert config.architecture.d_model == 128
        assert config.memory.parallel_training.enabled is False  # Disabled by default
        assert config.memory.parallel_training.auto_enable_threshold == 1024
        assert config.attention.window_size == 128

    def test_get_enhanced_config(self) -> None:
        """Test get_enhanced_config function."""
        # Test valid config names
        small_config = get_enhanced_config("small_parallel")
        assert isinstance(small_config, EnhancedTitansConfig)
        assert small_config.memory.parallel_training.enabled is True

        long_config = get_enhanced_config("long_context_parallel")
        assert isinstance(long_config, EnhancedTitansConfig)
        assert long_config.training.max_seq_length == 8192

        efficient_config = get_enhanced_config("efficient_parallel")
        assert isinstance(efficient_config, EnhancedTitansConfig)
        assert efficient_config.variant == "mag"

        # Test invalid config name
        with pytest.raises(ValueError, match="Unknown enhanced configuration"):
            get_enhanced_config("nonexistent_config")


class TestTaskSpecificConfigs:
    """Tests for task-specific configuration creation."""

    def test_language_modeling_config(self) -> None:
        """Test config creation for language modeling task."""
        config = create_config_for_task_with_parallel(
            task_type="language_modeling",
            model_size="medium",
            enable_parallel=True,
        )

        assert config.variant == "mac"
        assert config.architecture.chunk_size == 1024
        assert config.memory.memory_depth == 3
        assert config.memory.parallel_training.enabled is True
        assert config.memory.parallel_training.chunk_size == 128

    def test_time_series_config(self) -> None:
        """Test config creation for time series task."""
        config = create_config_for_task_with_parallel(
            task_type="time_series",
            model_size="small",
            enable_parallel=True,
        )

        assert config.variant == "mal"
        assert config.attention.window_size == 256
        assert config.memory.momentum_decay == 0.95
        assert config.memory.parallel_training.chunk_size == 64

    def test_long_sequence_config(self) -> None:
        """Test config creation for long sequence task."""
        config = create_config_for_task_with_parallel(
            task_type="long_sequence",
            model_size="large",
            enable_parallel=True,
        )

        assert config.variant == "mac"
        assert config.architecture.chunk_size == 2048
        assert config.memory.memory_depth == 4
        assert config.memory.num_persistent_tokens == 256
        assert config.memory.parallel_training.enabled is True
        assert config.memory.parallel_training.use_chunk_level_params is True

    def test_classification_config(self) -> None:
        """Test config creation for classification task."""
        config = create_config_for_task_with_parallel(
            task_type="classification",
            model_size="medium",
            enable_parallel=False,
        )

        assert config.variant == "mag"
        assert config.attention.window_size == 512
        assert config.memory.memory_depth == 2
        assert config.memory.parallel_training.enabled is False

    def test_config_without_parallel(self) -> None:
        """Test config creation with parallel training disabled."""
        config = create_config_for_task_with_parallel(
            task_type="language_modeling",
            model_size="small",
            enable_parallel=False,
        )

        # Should still have correct task-specific settings
        assert config.variant == "mac"
        assert config.architecture.chunk_size == 1024
        assert config.memory.memory_depth == 3
        # But parallel training should be disabled
        assert config.memory.parallel_training.enabled is False


class TestConfigValidation:
    """Tests for configuration validation and error handling."""

    def test_invalid_variant_handling(self) -> None:
        """Test handling of invalid variant types."""
        # This should not raise an error during initialization
        config = EnhancedTitansConfig(variant="invalid_variant")  # type: ignore
        # But the invalid variant should be preserved
        assert config.variant == "invalid_variant"

    def test_chunk_size_warnings(self) -> None:
        """Test warnings for suboptimal chunk sizes."""
        config = EnhancedTitansConfig()
        config.memory.parallel_training.enabled = True
        config.memory.parallel_training.chunk_size = 512  # Large chunk size
        config.training.max_seq_length = 1024

        # Should not raise error but may issue warnings
        config._validate_parallel_config()

    def test_memory_configuration_edge_cases(self) -> None:
        """Test edge cases in memory configuration."""
        # Test with minimal values
        config = EnhancedTitansConfig()
        config.memory.memory_depth = 1
        config.memory.num_persistent_tokens = 1
        config.memory.momentum_decay = 0.0
        config.memory.forget_rate = 0.0
        config.memory.learning_rate = 1e-6

        # Should still be valid
        assert config.memory.memory_depth == 1
        assert config.memory.num_persistent_tokens == 1

    def test_training_configuration_edge_cases(self) -> None:
        """Test edge cases in training configuration."""
        config = EnhancedTitansConfig()
        config.training.batch_size = 1
        config.training.max_seq_length = 1
        config.training.num_epochs = 1

        # Should still be valid
        assert config.training.batch_size == 1
        assert config.training.max_seq_length == 1

    def test_architecture_dtype_conversion(self) -> None:
        """Test architecture dtype conversion."""
        config = EnhancedTitansConfig()
        config.architecture.dtype = "float16"

        assert config.architecture.get_dtype() == jnp.float16

    def test_config_consistency_after_modifications(self) -> None:
        """Test that config remains consistent after modifications."""
        config = EnhancedTitansConfig(variant="mag")
        original_window_size = config.attention.window_size

        # Modify config
        config.enable_parallel_training(chunk_size=128)
        config.memory.memory_depth = 5

        # Variant-specific settings should be preserved
        assert config.variant == "mag"
        assert config.attention.window_size == original_window_size

        # New settings should be applied
        assert config.memory.parallel_training.enabled is True
        assert config.memory.parallel_training.chunk_size == 128
        assert config.memory.memory_depth == 5


class TestBackwardCompatibility:
    """Tests for backward compatibility with original TitansConfig."""

    def test_titans_config_alias(self) -> None:
        """Test that TitansConfig is aliased to EnhancedTitansConfig."""
        config = TitansConfig(variant="mal", seed=789)
        assert isinstance(config, EnhancedTitansConfig)
        assert config.variant == "mal"
        assert config.seed == 789

    def test_enhanced_config_works_as_regular_config(self) -> None:
        """Test that enhanced config can be used as regular config."""
        config = EnhancedTitansConfig(variant="lmm")

        # Should have all the basic properties
        assert hasattr(config, "variant")
        assert hasattr(config, "seed")
        assert hasattr(config, "memory")
        assert hasattr(config, "attention")
        assert hasattr(config, "architecture")
        assert hasattr(config, "training")

        # Should support serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)

        restored_config = EnhancedTitansConfig.from_dict(config_dict)
        assert restored_config.variant == "lmm"


if __name__ == "__main__":
    pytest.main([__file__])
