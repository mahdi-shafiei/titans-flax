"""Test suite for Titans attention modules.

This module contains tests for the attention modules from titans.attention_modules.
"""

from flax import nnx
import jax
import pytest

from titans.attention_modules import (
    CausalAttention,
    ConvolutionalLayer,
    SlidingWindowAttention,
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


class TestAttentionModules:
    """Tests for attention modules."""

    def test_causal_attention(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test causal attention."""
        attention = CausalAttention(in_features=d_model, num_heads=8, rngs=rngs)

        output = attention(sample_input, deterministic=True)
        assert output.shape == sample_input.shape

    def test_sliding_window_attention(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test sliding window attention."""
        attention = SlidingWindowAttention(
            in_features=d_model, num_heads=8, window_size=4, rngs=rngs
        )

        output = attention(sample_input, deterministic=True)
        assert output.shape == sample_input.shape

    def test_convolutional_layer(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test 1D convolutional layer."""
        conv = ConvolutionalLayer(features=d_model, kernel_size=3, rngs=rngs)

        output = conv(sample_input)
        assert output.shape == sample_input.shape


if __name__ == "__main__":
    pytest.main([__file__])
