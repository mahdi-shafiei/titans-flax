"""Test suite for Titans neural memory modules.

This module contains tests for the NeuralLongTermMemory and PersistentMemory
classes from titans.neural_memory.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from titans.neural_memory import (
    MemoryState,
    NeuralLongTermMemory,
    PersistentMemory,
)


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Standard fixture for rngs."""
    return nnx.Rngs(42)


@pytest.fixture
def sample_input() -> jax.Array:
    """Sample input tensor for testing."""
    # Use varied input instead of uniform ones to ensure memory updates
    key = jax.random.key(123)
    return jax.random.normal(key, (2, 16, 128)) * 0.1


@pytest.fixture
def d_model() -> int:
    """Standard model dimension for tests."""
    return 128


class TestNeuralLongTermMemory:
    """Tests for Neural Long-term Memory module."""

    def test_initialization(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test basic initialization."""
        memory = NeuralLongTermMemory(d_model=d_model, memory_depth=2, rngs=rngs)
        assert memory.d_model == d_model
        assert memory.memory_depth == 2
        assert len(memory.memory_layers) == 2

    def test_forward_pass_shape(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test forward pass produces correct shapes."""
        memory = NeuralLongTermMemory(d_model=d_model, memory_depth=2, rngs=rngs)

        output, memory_state = memory(sample_input, deterministic=True)

        # Check output shape
        assert output.shape == sample_input.shape

        # Check memory state structure
        assert isinstance(memory_state, MemoryState)
        assert len(memory_state.weights) == 2
        assert len(memory_state.momentum) == 2

        # Check weight shapes
        for weight in memory_state.weights:
            assert weight.shape == (d_model, d_model)

    def test_memory_state_creation(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test memory state creation."""
        memory = NeuralLongTermMemory(d_model=d_model, memory_depth=3, rngs=rngs)

        state = memory._create_initial_memory_state()

        assert len(state.weights) == 3
        assert len(state.momentum) == 3

        for weight, momentum in zip(state.weights, state.momentum):
            assert weight.shape == (d_model, d_model)
            assert momentum.shape == (d_model, d_model)
            # Weights should be small random values, not zeros
            assert not jnp.allclose(weight, 0.0)
            assert jnp.std(weight) > 0.0  # Should have some variance
            # Momentum should still be zeros initially
            assert jnp.allclose(momentum, 0.0)

    def test_memory_update_mechanism(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test that memory state is updated during training."""
        memory = NeuralLongTermMemory(d_model=d_model, memory_depth=2, rngs=rngs)

        # First forward pass
        _output1, state1 = memory(sample_input, deterministic=False)

        # Second forward pass with updated state
        _output2, state2 = memory(sample_input, state1, deterministic=False)

        # States should be different (memory was updated)
        for w1, w2 in zip(state1.weights, state2.weights):
            assert not jnp.allclose(w1, w2, atol=1e-6)

    def test_deterministic_mode(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test deterministic mode doesn't update memory."""
        memory = NeuralLongTermMemory(d_model=d_model, memory_depth=2, rngs=rngs)

        # First forward pass
        _output1, state1 = memory(sample_input, deterministic=True)

        # Second forward pass should not update state
        _output2, state2 = memory(sample_input, state1, deterministic=True)

        # States should be identical
        for w1, w2 in zip(state1.weights, state2.weights):
            assert jnp.allclose(w1, w2)


class TestPersistentMemory:
    """Tests for Persistent Memory module."""

    def test_initialization(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test basic initialization."""
        num_tokens = 32
        persistent = PersistentMemory(num_tokens=num_tokens, d_model=d_model, rngs=rngs)

        assert persistent.num_tokens == num_tokens
        assert persistent.d_model == d_model
        assert persistent.tokens.value.shape == (num_tokens, d_model)

    def test_forward_pass_shape(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test forward pass produces correct shapes."""
        num_tokens = 32
        batch_size = 4

        persistent = PersistentMemory(num_tokens=num_tokens, d_model=d_model, rngs=rngs)

        output = persistent(batch_size)
        assert output.shape == (batch_size, num_tokens, d_model)

    def test_batch_consistency(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test that same tokens are returned for all batch elements."""
        persistent = PersistentMemory(num_tokens=16, d_model=d_model, rngs=rngs)

        output = persistent(batch_size=3)

        # All batch elements should have identical tokens
        assert jnp.allclose(output[0], output[1])
        assert jnp.allclose(output[1], output[2])


if __name__ == "__main__":
    pytest.main([__file__])
