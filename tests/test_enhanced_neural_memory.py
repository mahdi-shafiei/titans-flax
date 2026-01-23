"""Tests for enhanced neural memory module with parallel training.

This module tests the enhanced neural memory implementation that includes
working parallel training algorithms from Section 3.2 of the Titans paper.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from titans.enhanced_neural_memory import (
    EnhancedNeuralLongTermMemory,
    MemoryState,
    NeuralLongTermMemory,  # Backward compatibility alias
    ParallelTrainingConfig,
    create_parallel_config,
    parallel_associative_scan,
)


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def sample_input() -> jax.Array:
    """Sample input tensor for testing."""
    key = jax.random.key(123)
    return jax.random.normal(key, (2, 32, 128)) * 0.1


@pytest.fixture
def d_model() -> int:
    """Standard model dimension for tests."""
    return 128


class TestMemoryState:
    """Tests for MemoryState NamedTuple."""

    def test_memory_state_creation(self) -> None:
        """Test MemoryState creation and access."""
        weights = [jnp.ones((64, 64)), jnp.ones((64, 64))]
        momentum = [jnp.zeros((64, 64)), jnp.zeros((64, 64))]

        state = MemoryState(weights=weights, momentum=momentum)

        assert len(state.weights) == 2
        assert len(state.momentum) == 2
        assert jnp.array_equal(state.weights[0], weights[0])
        assert jnp.array_equal(state.momentum[1], momentum[1])

    def test_memory_state_immutability(self) -> None:
        """Test that MemoryState is immutable."""
        weights = [jnp.ones((32, 32))]
        momentum = [jnp.zeros((32, 32))]

        state = MemoryState(weights=weights, momentum=momentum)

        # Should not be able to modify fields directly
        with pytest.raises(AttributeError):
            state.weights = []  # type: ignore


class TestParallelTrainingConfig:
    """Tests for ParallelTrainingConfig NamedTuple."""

    def test_default_config(self) -> None:
        """Test default ParallelTrainingConfig values."""
        config = ParallelTrainingConfig()

        assert config.enabled is False
        assert config.chunk_size == 32
        assert config.use_chunk_level_params is False
        assert config.min_sequence_length == 64

    def test_custom_config(self) -> None:
        """Test custom ParallelTrainingConfig values."""
        config = ParallelTrainingConfig(
            enabled=True,
            chunk_size=128,
            use_chunk_level_params=True,
            min_sequence_length=256,
        )

        assert config.enabled is True
        assert config.chunk_size == 128
        assert config.use_chunk_level_params is True
        assert config.min_sequence_length == 256

    def test_create_parallel_config_helper(self) -> None:
        """Test the helper function for creating config."""
        config = create_parallel_config(
            enabled=True,
            chunk_size=64,
            chunk_level_params=True,
            min_sequence_length=128,
        )

        assert config.enabled is True
        assert config.chunk_size == 64
        assert config.use_chunk_level_params is True
        assert config.min_sequence_length == 128


class TestParallelAssociativeScan:
    """Tests for enhanced parallel associative scan implementation."""

    def test_simple_scan_operation(self) -> None:
        """Test basic parallel associative scan operation."""
        seq_len = 8

        # Create simple test data
        transitions = jnp.array([0.9] * seq_len)
        elements = jnp.ones((seq_len, 4, 4)) * 0.1

        result = parallel_associative_scan(transitions, elements)

        # Verify shape
        assert result.shape == (seq_len, 4, 4)

        # Verify finite values
        assert jnp.all(jnp.isfinite(result))

    def test_scan_with_varying_transitions(self) -> None:
        """Test scan with different transition values."""
        seq_len = 16

        # Varying transition values
        transitions = jnp.linspace(0.1, 0.9, seq_len)
        elements = jax.random.normal(jax.random.key(42), (seq_len, 8, 8)) * 0.1

        result = parallel_associative_scan(transitions, elements)

        assert result.shape == (seq_len, 8, 8)
        assert jnp.all(jnp.isfinite(result))

    def test_small_sequence_fallback(self) -> None:
        """Test fallback to sequential scan for small sequences."""
        seq_len = 4  # Small enough to trigger fallback

        transitions = jnp.array([0.8] * seq_len)
        elements = jnp.ones((seq_len, 16, 16)) * 0.05

        # Should use sequential scan internally but still work
        result = parallel_associative_scan(transitions, elements)

        assert result.shape == (seq_len, 16, 16)
        assert jnp.all(jnp.isfinite(result))

    def test_scan_consistency(self) -> None:
        """Test consistency between different sequence lengths."""
        # Test that the scan gives consistent results for different sizes
        transitions_short = jnp.array([0.9] * 4)
        elements_short = jnp.ones((4, 8, 8)) * 0.1

        transitions_long = jnp.array([0.9] * 12)
        elements_long = jnp.ones((12, 8, 8)) * 0.1

        result_short = parallel_associative_scan(transitions_short, elements_short)
        result_long = parallel_associative_scan(transitions_long, elements_long)

        # First 4 elements should be similar
        assert jnp.allclose(result_short, result_long[:4], atol=1e-4)

    def test_edge_case_single_element(self) -> None:
        """Test scan with single element."""
        transitions = jnp.array([0.5])
        elements = jnp.ones((1, 4, 4)) * 2.0

        result = parallel_associative_scan(transitions, elements)

        assert result.shape == (1, 4, 4)
        # For single element: result should be just the element (no previous state)
        assert jnp.allclose(result[0], elements[0])


class TestEnhancedNeuralLongTermMemory:
    """Tests for EnhancedNeuralLongTermMemory class."""

    def test_initialization(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test basic initialization."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=3,
            rngs=rngs,
        )

        assert memory.d_model == d_model
        assert memory.memory_depth == 3
        assert len(memory.memory_layers) == 3
        assert memory.parallel_config.enabled is False

    def test_initialization_with_parallel_config(
        self, rngs: nnx.Rngs, d_model: int
    ) -> None:
        """Test initialization with parallel training config."""
        parallel_config = ParallelTrainingConfig(
            enabled=True,
            chunk_size=64,
            use_chunk_level_params=True,
            min_sequence_length=128,
        )

        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=2,
            parallel_config=parallel_config,
            rngs=rngs,
        )

        assert memory.parallel_config.enabled is True
        assert memory.parallel_config.chunk_size == 64
        assert memory.parallel_config.use_chunk_level_params is True

    def test_initialization_validation(self, rngs: nnx.Rngs) -> None:
        """Test input validation during initialization."""
        # Test invalid d_model
        with pytest.raises(ValueError, match="d_model must be positive"):
            EnhancedNeuralLongTermMemory(d_model=0, rngs=rngs)

        # Test invalid memory_depth
        with pytest.raises(ValueError, match="memory_depth must be positive"):
            EnhancedNeuralLongTermMemory(d_model=64, memory_depth=0, rngs=rngs)

        # Test invalid momentum_decay
        with pytest.raises(ValueError, match="momentum_decay must be in"):
            EnhancedNeuralLongTermMemory(d_model=64, momentum_decay=1.5, rngs=rngs)

        # Test invalid learning_rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            EnhancedNeuralLongTermMemory(d_model=64, learning_rate=-0.1, rngs=rngs)

        # Test invalid forget_rate
        with pytest.raises(ValueError, match="forget_rate must be in"):
            EnhancedNeuralLongTermMemory(d_model=64, forget_rate=1.2, rngs=rngs)

    def test_memory_state_creation(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test initial memory state creation."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model, memory_depth=2, rngs=rngs
        )

        state = memory._create_initial_memory_state()

        assert isinstance(state, MemoryState)
        assert len(state.weights) == 2
        assert len(state.momentum) == 2

        # Weights should be small random values, not zeros
        for weight in state.weights:
            assert weight.shape == (d_model, d_model)
            assert not jnp.allclose(weight, 0.0)
            assert jnp.std(weight) > 0.0

        # Momentum should be zeros initially
        for momentum in state.momentum:
            assert momentum.shape == (d_model, d_model)
            assert jnp.allclose(momentum, 0.0)

    def test_apply_memory_network(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test applying memory network with custom weights."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model, memory_depth=2, rngs=rngs
        )

        # Test with layer weights
        x = jnp.ones((1, d_model))
        output1 = memory._apply_memory_network(x)
        assert output1.shape == (1, d_model)

        # Test with custom weights
        custom_weights = [
            jnp.eye(d_model) * 0.5,
            jnp.eye(d_model) * 0.8,
        ]
        output2 = memory._apply_memory_network(x, custom_weights)
        assert output2.shape == (1, d_model)

        # Outputs should be different when using different weights
        assert not jnp.allclose(output1, output2)

    def test_sequential_training_step(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test sequential training step."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model, memory_depth=2, rngs=rngs
        )

        # Create test data
        keys = jax.random.normal(jax.random.key(42), (2, 16, d_model)) * 0.1
        values = jax.random.normal(jax.random.key(43), (2, 16, d_model)) * 0.1

        # Initial memory state
        initial_state = memory._create_initial_memory_state()

        # Perform sequential training step
        updated_state = memory._sequential_training_step(keys, values, initial_state)

        assert isinstance(updated_state, MemoryState)
        assert len(updated_state.weights) == 2
        assert len(updated_state.momentum) == 2

        # Memory should have been updated
        for init_w, new_w in zip(initial_state.weights, updated_state.weights):
            assert not jnp.allclose(init_w, new_w, atol=1e-6)

    def test_simplified_parallel_training_step(
        self, rngs: nnx.Rngs, d_model: int
    ) -> None:
        """Test simplified parallel training step."""
        parallel_config = ParallelTrainingConfig(
            enabled=True,
            chunk_size=16,
            min_sequence_length=32,
        )

        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=2,
            parallel_config=parallel_config,
            rngs=rngs,
        )

        # Create test data (long enough for parallel processing)
        keys = jax.random.normal(jax.random.key(42), (2, 64, d_model)) * 0.1
        values = jax.random.normal(jax.random.key(43), (2, 64, d_model)) * 0.1

        # Initial memory state
        initial_state = memory._create_initial_memory_state()

        # Perform parallel training step
        updated_state = memory._simplified_parallel_training_step(
            keys, values, initial_state
        )

        assert isinstance(updated_state, MemoryState)
        assert len(updated_state.weights) == 2
        assert len(updated_state.momentum) == 2

    def test_forward_pass_sequential_mode(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test forward pass in sequential mode."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=2,
            rngs=rngs,
        )

        # Forward pass
        output, memory_state = memory(sample_input, deterministic=False)

        # Verify output
        assert output.shape == sample_input.shape
        assert jnp.all(jnp.isfinite(output))

        # Verify memory state
        assert isinstance(memory_state, MemoryState)
        assert len(memory_state.weights) == 2
        assert len(memory_state.momentum) == 2

    def test_forward_pass_parallel_mode(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test forward pass in parallel mode."""
        parallel_config = ParallelTrainingConfig(
            enabled=True,
            chunk_size=32,
            min_sequence_length=64,
        )

        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=2,
            parallel_config=parallel_config,
            rngs=rngs,
        )

        # Long input to trigger parallel mode
        long_input = jax.random.normal(jax.random.key(42), (2, 128, d_model)) * 0.1

        # Forward pass
        output, memory_state = memory(long_input, deterministic=False)

        assert output.shape == long_input.shape
        assert jnp.all(jnp.isfinite(output))
        assert isinstance(memory_state, MemoryState)

    def test_deterministic_mode(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test deterministic mode behavior."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model, memory_depth=2, rngs=rngs
        )

        # First call (deterministic)
        _output1, state1 = memory(sample_input, deterministic=True)

        # Second call (deterministic) - should not update memory
        _output2, state2 = memory(sample_input, memory_state=state1, deterministic=True)

        # Memory state should not change in deterministic mode
        for w1, w2 in zip(state1.weights, state2.weights):
            assert jnp.allclose(w1, w2)

        for m1, m2 in zip(state1.momentum, state2.momentum):
            assert jnp.allclose(m1, m2)

    def test_force_parallel_override(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test forcing parallel mode via parameter."""
        # Create memory with parallel disabled
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            parallel_config=ParallelTrainingConfig(enabled=False),
            rngs=rngs,
        )

        # Force parallel mode
        output, _ = memory(sample_input, deterministic=False, force_parallel=True)

        assert output.shape == sample_input.shape
        assert jnp.all(jnp.isfinite(output))

    def test_input_validation(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test input validation in forward pass."""
        memory = EnhancedNeuralLongTermMemory(d_model=d_model, rngs=rngs)

        # Test wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 3D input"):
            memory(jnp.ones((10, 20)), deterministic=True)

        # Test wrong d_model dimension
        with pytest.raises(ValueError, match="Expected d_model"):
            memory(jnp.ones((2, 32, 64)), deterministic=True)

    def test_memory_state_validation_and_recovery(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test memory state validation and automatic recovery."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model, memory_depth=2, rngs=rngs
        )

        # Create invalid memory state (wrong number of layers)
        invalid_state = MemoryState(
            weights=[jnp.ones((d_model, d_model))],  # Only 1 layer instead of 2
            momentum=[jnp.zeros((d_model, d_model))],
        )

        # Should recover by creating new memory state
        output, recovered_state = memory(
            sample_input,
            memory_state=invalid_state,
            deterministic=False,
        )

        assert output.shape == sample_input.shape
        assert len(recovered_state.weights) == 2  # Correct number of layers
        assert len(recovered_state.momentum) == 2

    def test_error_handling_in_parallel_mode(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test error handling and fallback in parallel mode."""
        # Create memory with parallel config that might cause issues
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            parallel_config=ParallelTrainingConfig(
                enabled=True,
                chunk_size=1000,  # Very large chunk size
                min_sequence_length=1,
            ),
            rngs=rngs,
        )

        # Should handle errors gracefully and fall back to sequential
        output, state = memory(sample_input, deterministic=False)

        assert output.shape == sample_input.shape
        assert isinstance(state, MemoryState)

    def test_memory_parameter_updates(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test that memory parameters are properly updated."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=2,
            momentum_decay=0.8,
            learning_rate=0.02,
            forget_rate=0.2,
            rngs=rngs,
        )

        # Check initial parameter values
        assert abs(memory.momentum_decay.value - 0.8) < 1e-6
        assert abs(memory.learning_rate.value - 0.02) < 1e-6
        assert abs(memory.forget_rate.value - 0.2) < 1e-6

    def test_different_memory_depths(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test different memory depths."""
        for depth in [1, 2, 3, 5]:
            memory = EnhancedNeuralLongTermMemory(
                d_model=d_model,
                memory_depth=depth,
                rngs=rngs,
            )

            output, state = memory(sample_input, deterministic=False)

            assert output.shape == sample_input.shape
            assert len(state.weights) == depth
            assert len(state.momentum) == depth

    def test_parallel_vs_sequential_consistency(
        self, rngs: nnx.Rngs, d_model: int
    ) -> None:
        """Test that parallel and sequential modes give reasonable results."""
        # Sequential memory
        memory_seq = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=2,
            parallel_config=ParallelTrainingConfig(enabled=False),
            rngs=rngs,
        )

        # Parallel memory
        memory_par = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=2,
            parallel_config=ParallelTrainingConfig(
                enabled=True,
                chunk_size=16,
                min_sequence_length=32,
            ),
            rngs=rngs,
        )

        # Test input
        test_input = jax.random.normal(jax.random.key(42), (1, 64, d_model)) * 0.1

        # Sequential mode
        output_seq, _ = memory_seq(test_input, deterministic=False)

        # Parallel mode
        output_par, _ = memory_par(test_input, deterministic=False)

        # Both should produce valid outputs
        assert output_seq.shape == test_input.shape
        assert output_par.shape == test_input.shape
        assert jnp.all(jnp.isfinite(output_seq))
        assert jnp.all(jnp.isfinite(output_par))

    def test_memory_persistence_across_calls(
        self, rngs: nnx.Rngs, d_model: int
    ) -> None:
        """Test memory persistence across multiple calls."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model, memory_depth=2, rngs=rngs
        )

        # First call
        input1 = jax.random.normal(jax.random.key(1), (1, 16, d_model)) * 0.1
        _output1, state1 = memory(input1, deterministic=False)

        # Second call with memory from first call
        input2 = jax.random.normal(jax.random.key(2), (1, 16, d_model)) * 0.1
        _output2, state2 = memory(input2, memory_state=state1, deterministic=False)

        # Memory should evolve
        for w1, w2 in zip(state1.weights, state2.weights):
            assert not jnp.allclose(w1, w2, atol=1e-6)

    def test_large_sequence_processing(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test processing of large sequences."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            parallel_config=ParallelTrainingConfig(
                enabled=True,
                chunk_size=64,
                min_sequence_length=128,
            ),
            rngs=rngs,
        )

        # Large sequence
        large_input = jax.random.normal(jax.random.key(42), (1, 512, d_model)) * 0.1

        output, state = memory(large_input, deterministic=False)

        assert output.shape == large_input.shape
        assert jnp.all(jnp.isfinite(output))
        assert isinstance(state, MemoryState)


class TestBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_neural_long_term_memory_alias(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test that NeuralLongTermMemory is aliased to EnhancedNeuralLongTermMemory."""
        # Should be able to use the alias
        memory = NeuralLongTermMemory(d_model=d_model, rngs=rngs)

        assert isinstance(memory, EnhancedNeuralLongTermMemory)
        assert memory.d_model == d_model

    def test_enhanced_memory_as_drop_in_replacement(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test that enhanced memory can be used as drop-in replacement."""
        # Create enhanced memory without parallel config (should behave like original)
        memory = EnhancedNeuralLongTermMemory(d_model=d_model, rngs=rngs)

        # Should work like original memory
        output, state = memory(sample_input, deterministic=False)

        assert output.shape == sample_input.shape
        assert isinstance(state, MemoryState)

    def test_original_api_compatibility(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test compatibility with original API calls."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model, memory_depth=2, rngs=rngs
        )

        # Test original API style calls
        output1, state1 = memory(sample_input, deterministic=True)
        output2, state2 = memory(
            sample_input, memory_state=state1, deterministic=False, update_memory=True
        )

        assert output1.shape == sample_input.shape
        assert output2.shape == sample_input.shape
        assert isinstance(state1, MemoryState)
        assert isinstance(state2, MemoryState)


class TestEdgeCasesAndStressTests:
    """Tests for edge cases and stress testing."""

    def test_extreme_parameter_values(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test with extreme parameter values."""
        # Test with very small values
        memory_small = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            momentum_decay=0.01,
            learning_rate=1e-6,
            forget_rate=0.01,
            rngs=rngs,
        )

        output_small, _ = memory_small(sample_input, deterministic=False)
        assert jnp.all(jnp.isfinite(output_small))

        # Test with large values
        memory_large = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            momentum_decay=0.99,
            learning_rate=0.1,
            forget_rate=0.99,
            rngs=rngs,
        )

        output_large, _ = memory_large(sample_input, deterministic=False)
        assert jnp.all(jnp.isfinite(output_large))

    def test_zero_learning_rate(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test behavior with very small learning rate."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            learning_rate=1e-10,  # Very small but not zero
            rngs=rngs,
        )

        output, state = memory(sample_input, deterministic=False)
        assert jnp.all(jnp.isfinite(output))
        assert isinstance(state, MemoryState)

    def test_batch_size_extremes(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test with extreme batch sizes."""
        memory = EnhancedNeuralLongTermMemory(d_model=d_model, rngs=rngs)

        # Single sample
        tiny_input = jax.random.normal(jax.random.key(1), (1, 16, d_model))
        output_tiny, _ = memory(tiny_input, deterministic=True)
        assert output_tiny.shape == tiny_input.shape

        # Large batch
        large_input = jax.random.normal(jax.random.key(2), (32, 16, d_model))
        output_large, _ = memory(large_input, deterministic=True)
        assert output_large.shape == large_input.shape

    def test_sequence_length_extremes(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test with extreme sequence lengths."""
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            parallel_config=ParallelTrainingConfig(
                enabled=True,
                chunk_size=32,
                min_sequence_length=1,
            ),
            rngs=rngs,
        )

        # Very short sequence
        short_input = jax.random.normal(jax.random.key(1), (2, 1, d_model))
        output_short, _ = memory(short_input, deterministic=True)
        assert output_short.shape == short_input.shape

        # Very long sequence
        long_input = jax.random.normal(jax.random.key(2), (1, 1024, d_model))
        output_long, _ = memory(long_input, deterministic=True)
        assert output_long.shape == long_input.shape

    def test_numerical_stability(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test numerical stability with various input scales."""
        memory = EnhancedNeuralLongTermMemory(d_model=d_model, rngs=rngs)

        # Test different input scales
        for scale in [1e-6, 1e-3, 1.0, 1e3, 1e6]:
            test_input = jnp.ones((1, 16, d_model)) * scale
            output, _ = memory(test_input, deterministic=True)

            # Should remain finite even with extreme scales
            assert jnp.all(jnp.isfinite(output))


if __name__ == "__main__":
    pytest.main([__file__])
