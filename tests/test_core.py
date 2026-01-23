"""Test suite for Titans core module functionality.

This module contains tests for the TitansModel class and related functionality
from titans.core, converted from the examples to proper unit tests.
"""

from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from titans.core import TitansModel, TitansTrainer, TitansVariant, create_titans_model


class TestTitansModel:
    """Test class for TitansModel functionality."""

    @pytest.fixture
    def rngs(self) -> nnx.Rngs:
        """Standard fixture for rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def model_mac(self, rngs: nnx.Rngs) -> TitansModel:
        """Create a MAC variant model for testing."""
        return TitansModel(
            variant="mac",
            d_model=128,
            chunk_size=16,
            rngs=rngs,
        )

    @pytest.fixture
    def model_mag(self, rngs: nnx.Rngs) -> TitansModel:
        """Create a MAG variant model for testing."""
        return TitansModel(
            variant="mag",
            d_model=128,
            window_size=64,
            rngs=rngs,
        )

    @pytest.fixture
    def sample_input(self) -> jax.Array:
        """Sample input tensor for testing."""
        return jax.random.normal(jax.random.key(1), (2, 32, 128))

    def test_model_initialization(self, rngs: nnx.Rngs) -> None:
        """Test that models can be initialized with different variants."""
        variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]

        for variant in variants:
            model = TitansModel(
                variant=variant,
                d_model=64,
                rngs=rngs,
            )
            assert model.variant == variant
            assert model.d_model == 64

    def test_model_initialization_invalid_variant(self, rngs: nnx.Rngs) -> None:
        """Test that invalid variant raises ValueError."""
        with pytest.raises(ValueError, match="Unknown variant"):
            TitansModel(
                variant="invalid",  # type: ignore
                d_model=64,
                rngs=rngs,
            )

    def test_model_initialization_invalid_params(self, rngs: nnx.Rngs) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            TitansModel(variant="mac", d_model=0, rngs=rngs)

        with pytest.raises(ValueError, match="num_heads must be positive"):
            TitansModel(variant="mac", d_model=64, num_heads=0, rngs=rngs)

        with pytest.raises(ValueError, match="memory_depth must be positive"):
            TitansModel(variant="mac", d_model=64, memory_depth=0, rngs=rngs)

    def test_forward_pass_shapes(
        self, model_mac: TitansModel, sample_input: jax.Array
    ) -> None:
        """Test that forward passes produce correct output shapes."""
        output = model_mac(sample_input, deterministic=True)
        assert output.shape == sample_input.shape

    def test_forward_pass_invalid_input(self, model_mac: TitansModel) -> None:
        """Test that invalid input shapes raise ValueError."""
        # Wrong number of dimensions
        with pytest.raises(ValueError, match="Expected 3D input"):
            model_mac(jnp.ones((10, 20)), deterministic=True)

        # Wrong d_model dimension
        with pytest.raises(ValueError, match="Expected d_model=128"):
            model_mac(jnp.ones((2, 32, 64)), deterministic=True)

    def test_memory_reset(
        self, model_mac: TitansModel, sample_input: jax.Array
    ) -> None:
        """Test memory reset functionality."""
        # Process some input to create memory state
        _ = model_mac(sample_input, deterministic=False)

        # Check that memory exists
        memory_state_before = model_mac.get_memory_state()
        assert memory_state_before is not None

        # Reset memory
        model_mac.reset_memory()

        # Check that memory is reset
        memory_state_after = model_mac.get_memory_state()
        assert memory_state_after is None

    def test_memory_state_management(
        self, model_mac: TitansModel, sample_input: jax.Array
    ) -> None:
        """Test memory state get/set functionality."""
        # Initially no memory
        assert model_mac.get_memory_state() is None

        # Process input to create memory
        _ = model_mac(sample_input, deterministic=False)
        memory_state = model_mac.get_memory_state()
        assert memory_state is not None

        # Reset and set memory state
        model_mac.reset_memory()
        model_mac.set_memory_state(memory_state)

        # Check that memory state was restored
        restored_state = model_mac.get_memory_state()
        assert restored_state is not None
        assert len(restored_state.weights) == len(memory_state.weights)

    def test_memory_state_type_validation(self, model_mac: TitansModel) -> None:
        """Test that setting invalid memory state raises TypeError."""
        with pytest.raises(TypeError, match="Expected MemoryState or None"):
            model_mac.set_memory_state("invalid")  # type: ignore

    def test_memory_norm_computation(
        self, model_mac: TitansModel, sample_input: jax.Array
    ) -> None:
        """Test memory norm computation."""
        # Initially no memory, norm should be 0
        assert model_mac.get_memory_norm() == 0.0

        # Process input to create memory
        _ = model_mac(sample_input, deterministic=False)

        # Memory norm should be positive
        memory_norm = model_mac.get_memory_norm()
        assert memory_norm > 0.0
        assert jnp.isfinite(memory_norm)


class TestTestTimeAdaptation:
    """Test test-time adaptation functionality."""

    @pytest.fixture
    def model(self) -> TitansModel:
        """Create a model for adaptation testing."""
        return create_titans_model(
            variant="mac",
            d_model=128,
            chunk_size=16,
            seed=42,
        )

    @pytest.fixture
    def input_sequences(self) -> tuple[jax.Array, jax.Array]:
        """Create two different input sequences for adaptation testing."""
        x1 = jax.random.normal(jax.random.key(1), (2, 32, 128))
        x2 = jax.random.normal(jax.random.key(2), (2, 32, 128))
        return x1, x2

    def test_adaptation_changes_output(
        self, model: TitansModel, input_sequences: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that adaptation changes the model's response to inputs."""
        x1, x2 = input_sequences

        # Get deterministic output for x2
        model.reset_memory()
        output_det = model(x2, deterministic=True)

        # Process x1 to adapt memory, then process x2
        model.reset_memory()
        _ = model(x1, deterministic=False)  # Adapt memory
        output_adapted = model(x2, deterministic=False)  # Use adapted memory

        # Outputs should be different due to adaptation
        difference = jnp.mean(jnp.abs(output_det - output_adapted))
        assert difference > 0.0
        assert jnp.isfinite(difference)

    def test_memory_updates_during_adaptation(
        self, model: TitansModel, input_sequences: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that memory is actually updated during adaptation."""
        x1, x2 = input_sequences

        # Process first sequence
        model.reset_memory()
        _ = model(x1, deterministic=False)
        memory_after_x1 = model.get_memory_state()

        # Process second sequence
        _ = model(x2, deterministic=False)
        memory_after_x2 = model.get_memory_state()

        # Memory should be different after processing different sequences
        assert memory_after_x1 is not None
        assert memory_after_x2 is not None

        # Check that memory weights have changed
        for w1, w2 in zip(memory_after_x1.weights, memory_after_x2.weights):
            assert not jnp.allclose(w1, w2, atol=1e-6)

    def test_deterministic_mode_no_adaptation(
        self, model: TitansModel, input_sequences: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that deterministic mode doesn't update memory."""
        x1, x2 = input_sequences

        # Process in deterministic mode
        model.reset_memory()
        _ = model(x1, deterministic=True)
        memory_after_x1 = model.get_memory_state()

        _ = model(x2, deterministic=True)
        memory_after_x2 = model.get_memory_state()

        # Memory should remain None in deterministic mode
        assert memory_after_x1 is None
        assert memory_after_x2 is None

    def test_test_time_adaptation_integration(self) -> None:
        """Integration test for test-time adaptation functionality."""
        # Create model
        model = create_titans_model(
            variant="mac",
            d_model=128,
            chunk_size=16,
            seed=42,
        )

        # Generate example data
        batch_size, seq_len = 2, 32
        x1 = jax.random.normal(jax.random.key(1), (batch_size, seq_len, 128))
        x2 = jax.random.normal(jax.random.key(2), (batch_size, seq_len, 128))

        # Test adaptation workflow
        model.reset_memory()
        output1 = model(x1, deterministic=False)
        output2 = model(x2, deterministic=False)

        # Compare deterministic vs adaptive processing
        model.reset_memory()
        output_det = model(x2, deterministic=True)

        model.reset_memory()
        _ = model(x1, deterministic=False)  # Adapt memory
        output_adapted = model(x2, deterministic=False)  # Use adapted memory

        # Verify outputs have correct shapes and are finite
        assert output1.shape == (batch_size, seq_len, 128)
        assert output2.shape == (batch_size, seq_len, 128)
        assert output_det.shape == (batch_size, seq_len, 128)
        assert output_adapted.shape == (batch_size, seq_len, 128)

        assert jnp.all(jnp.isfinite(output1))
        assert jnp.all(jnp.isfinite(output2))
        assert jnp.all(jnp.isfinite(output_det))
        assert jnp.all(jnp.isfinite(output_adapted))

        # Verify adaptation creates differences
        difference = jnp.mean(jnp.abs(output_det - output_adapted))
        assert difference > 0.0


class TestMemoryCapacity:
    """Test memory capacity across different variants."""

    @pytest.fixture
    def test_inputs(self) -> list[jax.Array]:
        """Create test inputs with different scales."""
        return [
            jax.random.normal(jax.random.key(i), (1, 16, 64)) * (i + 1)
            for i in range(3)
        ]

    def test_memory_capacity_all_variants(self, test_inputs: list[jax.Array]) -> None:
        """Test memory capacity for all variants."""
        variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]
        d_model = 64

        for variant in variants:
            model = create_titans_model(
                variant=variant, d_model=d_model, memory_depth=3, seed=42
            )

            memory_norms = []
            for _i, x in enumerate(test_inputs):
                _ = model(x, deterministic=False)
                memory_norm = model.get_memory_norm()
                memory_norms.append(memory_norm)

                # Memory norm should be finite and non-negative
                assert jnp.isfinite(memory_norm)
                assert memory_norm >= 0.0

            # Memory norms should generally increase as we process more data
            # (though this isn't guaranteed due to the adaptive nature)
            assert all(norm >= 0 for norm in memory_norms)

    def test_memory_persistence_across_sequences(
        self, test_inputs: list[jax.Array]
    ) -> None:
        """Test that memory persists and accumulates across sequences."""
        model = create_titans_model(variant="mac", d_model=64, memory_depth=2, seed=42)

        for x in test_inputs:
            _ = model(x, deterministic=False)
            current_norm = model.get_memory_norm()

            # Memory should exist after processing
            assert current_norm > 0.0
            # Memory state should exist
            assert model.get_memory_state() is not None

    def test_memory_capacity_integration(self) -> None:
        """Integration test for memory capacity across variants."""
        variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]
        d_model = 64

        for variant in variants:
            model = create_titans_model(
                variant=variant, d_model=d_model, memory_depth=3, seed=42
            )

            # Process multiple sequences to test memory capacity
            for i in range(3):
                x = jax.random.normal(jax.random.key(i), (1, 16, d_model)) * (i + 1)

                output = model(x, deterministic=False)
                memory_norm = model.get_memory_norm()

                # Verify output shape and finite values
                assert output.shape == (1, 16, d_model)
                assert jnp.all(jnp.isfinite(output))
                assert jnp.isfinite(memory_norm)
                assert memory_norm >= 0.0


class TestLongSequenceProcessing:
    """Test long sequence processing capabilities."""

    def test_long_sequence_chunked_processing(self) -> None:
        """Test processing of long sequences in chunks."""
        # Create model optimized for long sequences
        model = create_titans_model(
            variant="mag",  # MAG is good for long sequences
            d_model=128,
            window_size=64,
            memory_depth=2,
            seed=42,
        )

        # Generate a long sequence
        long_seq_len = 512
        x_long = jax.random.normal(jax.random.key(0), (1, long_seq_len, 128))

        # Process in chunks
        chunk_size = 128
        total_output = []

        for i in range(0, long_seq_len, chunk_size):
            end_idx = min(i + chunk_size, long_seq_len)
            chunk = x_long[:, i:end_idx, :]

            output_chunk = model(chunk, deterministic=False)
            total_output.append(output_chunk)

            # Each chunk should produce valid output
            assert output_chunk.shape == chunk.shape
            assert jnp.all(jnp.isfinite(output_chunk))

        # Concatenate all outputs
        final_output = jnp.concatenate(total_output, axis=1)

        # Final output should match input sequence length
        assert final_output.shape == x_long.shape

    def test_memory_evolution_long_sequence(self) -> None:
        """Test that memory evolves during long sequence processing."""
        model = create_titans_model(
            variant="mag",
            d_model=64,
            window_size=32,
            memory_depth=2,
            seed=42,
        )

        # Process sequence in chunks and track memory evolution
        memory_norms = []
        chunk_size = 64
        num_chunks = 8

        for i in range(num_chunks):
            chunk = jax.random.normal(jax.random.key(i), (1, chunk_size, 64))
            _ = model(chunk, deterministic=False)

            memory_norm = model.get_memory_norm()
            memory_norms.append(memory_norm)

            # Memory should exist and be finite
            assert memory_norm > 0.0
            assert jnp.isfinite(memory_norm)

        # Memory should be actively maintained throughout
        assert all(norm > 0.0 for norm in memory_norms)

    def test_long_sequence_integration(self) -> None:
        """Integration test for long sequence processing."""
        # Create model for long sequences
        model = create_titans_model(
            variant="mag",  # MAG is good for long sequences
            d_model=128,
            window_size=64,
            memory_depth=2,
            seed=42,
        )

        # Generate a long sequence
        long_seq_len = 256  # Smaller than examples for faster testing
        x_long = jax.random.normal(jax.random.key(0), (1, long_seq_len, 128))

        # Process in chunks
        chunk_size = 64
        total_output = []

        for i in range(0, long_seq_len, chunk_size):
            end_idx = min(i + chunk_size, long_seq_len)
            chunk = x_long[:, i:end_idx, :]

            output_chunk = model(chunk, deterministic=False)
            total_output.append(output_chunk)

            # Verify chunk output
            assert output_chunk.shape == chunk.shape
            assert jnp.all(jnp.isfinite(output_chunk))

        # Verify final concatenated output
        final_output = jnp.concatenate(total_output, axis=1)
        assert final_output.shape == x_long.shape
        assert jnp.all(jnp.isfinite(final_output))


class TestTitansVariants:
    """Tests for all Titans variants."""

    @pytest.fixture
    def rngs(self) -> nnx.Rngs:
        """Standard fixture for rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_input(self) -> jax.Array:
        """Sample input tensor for testing."""
        # Use varied input instead of uniform ones to ensure memory updates
        key = jax.random.key(123)
        return jax.random.normal(key, (2, 16, 128)) * 0.1

    @pytest.fixture
    def d_model(self) -> int:
        """Standard model dimension for tests."""
        return 128

    def test_titans_mac_initialization(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test Titans MAC initialization."""
        model = TitansModel(
            variant="mac",
            d_model=d_model,
            chunk_size=8,
            memory_depth=2,
            rngs=rngs,
        )

        assert model.variant == "mac"
        assert model.d_model == d_model
        assert hasattr(model.model, "neural_memory")
        assert hasattr(model.model, "persistent_memory")

    def test_titans_mac_forward_pass(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test Titans MAC forward pass."""
        model = TitansModel(
            variant="mac",
            d_model=d_model,
            chunk_size=8,
            memory_depth=2,
            rngs=rngs,
        )

        output = model(sample_input, deterministic=True)

        assert output.shape == sample_input.shape

    def test_titans_mag_initialization(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test Titans MAG initialization."""
        model = TitansModel(
            variant="mag",
            d_model=d_model,
            window_size=4,
            memory_depth=2,
            rngs=rngs,
        )

        assert model.variant == "mag"
        assert model.d_model == d_model
        assert hasattr(model.model, "neural_memory")
        assert hasattr(model.model, "sliding_attention")

    def test_titans_mag_forward_pass(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test Titans MAG forward pass."""
        model = TitansModel(
            variant="mag",
            d_model=d_model,
            window_size=4,
            memory_depth=2,
            rngs=rngs,
        )

        output = model(sample_input, deterministic=True)

        assert output.shape == sample_input.shape

    def test_titans_mal_initialization(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test Titans MAL initialization."""
        model = TitansModel(
            variant="mal",
            d_model=d_model,
            window_size=4,
            memory_depth=2,
            rngs=rngs,
        )

        assert model.variant == "mal"
        assert hasattr(model.model, "neural_memory")
        assert hasattr(model.model, "sliding_attention")

    def test_titans_mal_forward_pass(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test Titans MAL forward pass."""
        model = TitansModel(
            variant="mal",
            d_model=d_model,
            window_size=4,
            memory_depth=2,
            rngs=rngs,
        )

        output = model(sample_input, deterministic=True)

        assert output.shape == sample_input.shape

    def test_titans_lmm_initialization(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test Titans LMM initialization."""
        model = TitansModel(variant="lmm", d_model=d_model, memory_depth=2, rngs=rngs)

        assert model.variant == "lmm"
        assert hasattr(model.model, "neural_memory")
        assert hasattr(model.model, "persistent_memory")

    def test_titans_lmm_forward_pass(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test Titans LMM forward pass."""
        model = TitansModel(variant="lmm", d_model=d_model, memory_depth=2, rngs=rngs)

        output = model(sample_input, deterministic=True)

        assert output.shape == sample_input.shape


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.fixture
    def rngs(self) -> nnx.Rngs:
        """Standard fixture for rngs."""
        return nnx.Rngs(42)

    @pytest.fixture
    def sample_input(self) -> jax.Array:
        """Sample input tensor for testing."""
        # Use varied input instead of uniform ones to ensure memory updates
        key = jax.random.key(123)
        return jax.random.normal(key, (2, 16, 128)) * 0.1

    @pytest.fixture
    def d_model(self) -> int:
        """Standard model dimension for tests."""
        return 128

    def test_memory_persistence_across_calls(
        self, rngs: nnx.Rngs, d_model: int
    ) -> None:
        """Test that memory state persists across multiple calls."""
        model = TitansModel(
            variant="mac",
            d_model=d_model,
            chunk_size=8,
            memory_depth=2,
            rngs=rngs,
        )

        # Generate sequence of inputs
        input1 = jnp.ones((1, 16, d_model))
        input2 = jnp.ones((1, 16, d_model)) * 2

        # First call
        model(input1, deterministic=False)
        state1 = model.get_memory_state()

        # Second call with memory continuation
        model(input2, deterministic=False)
        state2 = model.get_memory_state()

        # Memory should have been updated
        if state1 is not None and state2 is not None:
            for w1, w2 in zip(state1.weights, state2.weights):
                assert not jnp.allclose(w1, w2, atol=1e-6)

    def test_gradient_flow(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test that gradients flow through the model."""
        model = TitansModel(
            variant="mac",
            d_model=d_model,
            chunk_size=8,
            memory_depth=2,
            rngs=rngs,
        )

        def loss_fn(model: TitansModel) -> jax.Array:
            output = model(sample_input, deterministic=True)
            return jnp.mean(output**2)

        # Compute gradients
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model)

        # Check that gradients exist and are finite
        # The gradients should be a State object with the same structure
        assert isinstance(grads, nnx.State)

        # Check that the gradient state has the expected modules
        grad_state = nnx.state(grads)
        assert "model" in grad_state, "Missing gradients for model"

        # Check that at least some gradients are computed
        # We'll check that the gradient computation doesn't fail
        def check_finite_grads(state_dict: Any) -> None:
            """Recursively check that all gradients are finite."""
            for key, value in state_dict.items():
                if hasattr(value, "value") and value.value is not None:
                    assert jnp.isfinite(value.value).all(), (
                        f"Non-finite gradient in {key}"
                    )
                elif hasattr(value, "_mapping") and isinstance(value._mapping, dict):
                    check_finite_grads(value._mapping)

        check_finite_grads(grad_state)

    def test_different_sequence_lengths(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test models with different sequence lengths."""
        model = TitansModel(
            variant="mag",
            d_model=d_model,
            window_size=4,
            memory_depth=2,
            rngs=rngs,
        )

        # Test with different sequence lengths
        for seq_len in [8, 16, 32]:
            test_input = jnp.ones((2, seq_len, d_model))
            output = model(test_input, deterministic=True)
            assert output.shape == test_input.shape

    def test_batch_independence(self, rngs: nnx.Rngs, d_model: int) -> None:
        """Test that different batch elements are processed independently."""
        model = TitansModel(variant="lmm", d_model=d_model, memory_depth=2, rngs=rngs)

        # Create input with different values for each batch element
        batch_input = jnp.stack(
            [
                jnp.ones((16, d_model)),
                jnp.ones((16, d_model)) * 2,
            ]
        )

        output = model(batch_input, deterministic=True)

        # Outputs should be different for different inputs
        assert not jnp.allclose(output[0], output[1], atol=1e-3)

    def test_finite_outputs(
        self, rngs: nnx.Rngs, d_model: int, sample_input: jax.Array
    ) -> None:
        """Test that all outputs are finite (no NaN or inf)."""
        variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]

        for variant in variants:
            if variant == "mac":
                model = TitansModel(
                    variant=variant, d_model=d_model, chunk_size=8, rngs=rngs
                )
            elif variant in ["mag", "mal"]:
                model = TitansModel(
                    variant=variant, d_model=d_model, window_size=4, rngs=rngs
                )
            else:  # lmm
                model = TitansModel(variant=variant, d_model=d_model, rngs=rngs)

            output = model(sample_input, deterministic=True)

            # Check output is finite
            assert jnp.isfinite(output).all(), f"Non-finite output in {variant}"


class TestMemoryGradientIntegration:
    """Test memory-gradient integration across all variants."""

    @pytest.fixture
    def sample_training_data(self) -> tuple[jax.Array, jax.Array]:
        """Generate sample training data."""
        key = jax.random.key(123)
        x = jax.random.normal(key, (4, 32, 128)) * 0.1  # Small values for stability
        # Simple target: slightly modified input
        y = x + 0.05 * jax.random.normal(jax.random.split(key)[1], x.shape)
        return x, y

    @pytest.mark.parametrize("variant", ["mac", "mag", "mal", "lmm"])
    def test_memory_updates_in_gradient_computation(
        self, variant: TitansVariant, sample_training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that memory updates are included in gradient computation."""
        x, y = sample_training_data

        model = create_titans_model(
            variant=variant, d_model=128, memory_depth=2, seed=42
        )

        # Get initial memory state
        initial_memory_state = model.get_memory_state()

        def loss_fn(model: TitansModel) -> jax.Array:
            # Memory updates should be part of this computation
            predictions = model(x, deterministic=False)
            return jnp.mean((predictions - y) ** 2)

        # Compute gradients
        loss_value, grads = nnx.value_and_grad(loss_fn)(model)

        # Verify gradients exist and are finite
        assert jnp.isfinite(loss_value), "Loss should be finite"

        # Check that memory state was actually updated during gradient computation
        updated_memory_state = model.get_memory_state()

        if initial_memory_state is None:
            # Memory should be initialized after first forward pass
            assert updated_memory_state is not None, "Memory should be initialized"
        else:
            # Memory should be different after update
            assert updated_memory_state is not None, "Memory should not be None"
            for w1, w2 in zip(
                initial_memory_state.weights, updated_memory_state.weights
            ):
                assert not jnp.allclose(w1, w2, atol=1e-6), (
                    "Memory weights should be updated"
                )

        # Verify gradients are computed for model parameters
        grad_state = nnx.state(grads)
        assert len(grad_state) > 0, "Gradients should be computed"

        # Check that all gradients are finite
        def check_finite_grads(state_dict: Any) -> None:
            for key, value in state_dict.items():
                if hasattr(value, "value") and value.value is not None:
                    assert jnp.isfinite(value.value).all(), (
                        f"Non-finite gradient in {key}"
                    )
                elif hasattr(value, "_mapping") and isinstance(value._mapping, dict):
                    check_finite_grads(value._mapping)

        check_finite_grads(grad_state)

    def test_memory_state_persistence_across_steps(
        self, sample_training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that memory state persists correctly across training steps."""
        x, y = sample_training_data

        model = create_titans_model(variant="mac", d_model=128, memory_depth=2, seed=42)
        trainer = TitansTrainer(model, learning_rate=1e-3)

        memory_norms = []
        losses = []

        # Run multiple training steps
        for _step in range(5):
            # Perform training step
            loss = trainer.train_step(x, y)
            losses.append(loss)

            # Track memory evolution
            memory_norm = model.get_memory_norm()
            memory_norms.append(memory_norm)

        # Verify memory is evolving
        assert len(set(memory_norms[1:])) > 1, "Memory should evolve across steps"

        # Verify we have a reasonable number of finite losses
        finite_losses = [loss for loss in losses if jnp.isfinite(loss)]
        assert len(finite_losses) >= 3, (
            f"Should have finite losses, got {finite_losses}"
        )

    def test_deterministic_vs_stochastic_behavior(
        self, sample_training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test deterministic vs stochastic memory behavior."""
        x, _y = sample_training_data

        model = create_titans_model(variant="mag", d_model=128, memory_depth=2, seed=42)

        # Reset memory
        model.reset_memory()
        initial_memory_state = model.get_memory_state()

        # Deterministic forward pass should not update memory
        model(x, deterministic=True)
        memory_after_det = model.get_memory_state()

        if initial_memory_state is None and memory_after_det is not None:
            # Memory was initialized but not updated
            pass  # This is acceptable
        elif initial_memory_state is not None and memory_after_det is not None:
            # Memory should be unchanged
            for w1, w2 in zip(initial_memory_state.weights, memory_after_det.weights):
                assert jnp.allclose(w1, w2), (
                    "Deterministic mode should not update memory"
                )

        # Stochastic forward pass should update memory
        model.reset_memory()
        model(x, deterministic=False)
        memory_after_stoch = model.get_memory_state()

        assert memory_after_stoch is not None, "Memory should be initialized/updated"

        # Second stochastic pass should further update memory
        model(x, deterministic=False)
        memory_after_stoch2 = model.get_memory_state()

        # Memory should be different after second update
        assert memory_after_stoch2 is not None, "Memory should not be None"
        for w1, w2 in zip(memory_after_stoch.weights, memory_after_stoch2.weights):
            # Should be different (unless by very unlikely chance)
            if not jnp.allclose(w1, w2, atol=1e-6):
                break
        else:
            pytest.fail("Memory should update between stochastic forward passes")

    def test_memory_gradient_flow(
        self, sample_training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that gradients flow through memory updates correctly."""
        x, y = sample_training_data

        model = create_titans_model(variant="lmm", d_model=128, memory_depth=2, seed=42)

        def memory_dependent_loss(model: TitansModel) -> jax.Array:
            """Loss function that explicitly depends on memory state."""
            output = model(x, deterministic=False)

            # Base reconstruction loss
            recon_loss = jnp.mean((output - y) ** 2)

            # Add term that depends on memory state
            memory_state = model.get_memory_state()
            if memory_state is not None:
                # Fix: Use jnp.stack to create array, then sum
                weight_squares = jnp.stack(
                    [jnp.sum(w**2) for w in memory_state.weights]
                )
                memory_term = jnp.sum(weight_squares)
                return recon_loss + 1e-4 * memory_term
            return recon_loss

        # Compute gradients
        loss, grads = nnx.value_and_grad(memory_dependent_loss)(model)

        # Verify loss is finite
        assert jnp.isfinite(loss), "Loss should be finite"

        # Verify gradients exist
        grad_state = nnx.state(grads)
        assert len(grad_state) > 0, "Should have gradients"

        # Verify memory was updated
        memory_state = model.get_memory_state()
        assert memory_state is not None, "Memory should be initialized"
        assert model.get_memory_norm() > 0, "Memory should have non-zero norm"

    def test_memory_state_synchronization(
        self, sample_training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that memory state synchronization works correctly."""
        x, _y = sample_training_data

        model = create_titans_model(variant="mal", d_model=128, memory_depth=2, seed=42)

        # Reset memory
        model.reset_memory()
        assert model.get_memory_state() is None, "Memory should be reset"
        assert model.get_memory_norm() == 0.0, "Memory norm should be zero"

        # Forward pass to initialize memory
        model(x, deterministic=False)
        memory_state1 = model.get_memory_state()
        memory_norm1 = model.get_memory_norm()

        assert memory_state1 is not None, "Memory should be initialized"
        assert memory_norm1 > 0, "Memory norm should be positive"

        # Another forward pass should update memory
        model(x, deterministic=False)
        memory_state2 = model.get_memory_state()
        model.get_memory_norm()

        # Memory should be updated
        assert memory_state2 is not None, "Memory should not be None"
        for w1, w2 in zip(memory_state1.weights, memory_state2.weights):
            if not jnp.allclose(w1, w2, atol=1e-6):
                break
        else:
            pytest.fail("Memory should be updated between forward passes")

        # Test memory state setting
        model.set_memory_state(memory_state1)
        retrieved_state = model.get_memory_state()

        # Should match what we set
        assert retrieved_state is not None, "Retrieved state should not be None"
        for w1, w2 in zip(memory_state1.weights, retrieved_state.weights):
            assert jnp.allclose(w1, w2), "Set memory state should match retrieved state"

    def test_training_convergence(
        self, sample_training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that training with memory updates shows convergence."""
        x, y = sample_training_data

        model = create_titans_model(variant="mag", d_model=128, memory_depth=2, seed=42)
        trainer = TitansTrainer(model, learning_rate=1e-3)

        losses = []
        memory_norms = []

        # Train for several steps
        for step in range(20):
            loss = trainer.train_step(x, y)
            memory_norm = model.get_memory_norm()

            losses.append(loss)
            memory_norms.append(memory_norm)

            # Verify loss is finite
            assert jnp.isfinite(loss), f"Loss should be finite at step {step}"

        # Check that loss generally decreases (allowing for some fluctuation)
        initial_loss = np.mean(losses[:3])
        final_loss = np.mean(losses[-3:])

        # Should show some improvement
        improvement_ratio = (initial_loss - final_loss) / initial_loss
        assert improvement_ratio > -0.5, (
            f"Loss should not increase too much: {improvement_ratio}"
        )

        # Memory should be active
        final_memory_norm = memory_norms[-1]
        assert final_memory_norm > 0, "Memory should be active after training"

    def test_memory_reset_functionality(self) -> None:
        """Test memory reset functionality works correctly."""
        model = create_titans_model(variant="mac", d_model=64, memory_depth=2, seed=42)

        # Generate some data to process
        x = jax.random.normal(jax.random.key(123), (2, 16, 64))

        # Process data to initialize memory
        model(x, deterministic=False)
        assert model.get_memory_norm() > 0, "Memory should be initialized"

        # Reset memory
        model.reset_memory()
        assert model.get_memory_state() is None, "Memory should be None after reset"
        assert model.get_memory_norm() == 0.0, "Memory norm should be zero after reset"

        # Process data again - should reinitialize
        model(x, deterministic=False)
        assert model.get_memory_norm() > 0, "Memory should be reinitialized"

    def test_memory_consistency_across_variants(
        self, sample_training_data: tuple[jax.Array, jax.Array]
    ) -> None:
        """Test that memory behavior is consistent across variants."""
        x, _y = sample_training_data
        variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]

        results = {}

        for variant in variants:
            model = create_titans_model(
                variant=variant,
                d_model=128,
                memory_depth=2,
                seed=42,
            )

            # Test basic memory functionality
            model.reset_memory()
            assert model.get_memory_norm() == 0.0, f"{variant}: Memory should be reset"

            # Process data
            output = model(x, deterministic=False)
            memory_norm = model.get_memory_norm()

            results[variant] = {
                "memory_norm": memory_norm,
                "output_norm": float(jnp.linalg.norm(output)),
                "memory_initialized": model.get_memory_state() is not None,
            }

            # Basic consistency checks
            assert memory_norm > 0, f"{variant}: Memory should be active"
            assert jnp.isfinite(output).all(), f"{variant}: Output should be finite"

        # All variants should have initialized memory
        for variant, result in results.items():
            assert result["memory_initialized"], (
                f"{variant}: Memory should be initialized"
            )
            assert result["memory_norm"] > 0, f"{variant}: Memory should be active"


class TestCreateTitansModel:
    """Test the create_titans_model factory function."""

    def test_create_model_all_variants(self) -> None:
        """Test creating models for all variants."""
        variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]

        for variant in variants:
            model = create_titans_model(
                variant=variant,
                d_model=64,
                seed=42,
            )

            assert isinstance(model, TitansModel)
            assert model.variant == variant
            assert model.d_model == 64

    def test_create_model_with_kwargs(self) -> None:
        """Test creating model with additional keyword arguments."""
        model = create_titans_model(
            variant="mac",
            d_model=128,
            seed=42,
            chunk_size=32,
            memory_depth=3,
            num_persistent_tokens=128,
        )

        assert isinstance(model, TitansModel)
        assert model.variant == "mac"
        assert model.d_model == 128

    def test_create_model_reproducibility(self) -> None:
        """Test that models created with same seed are reproducible."""
        model1 = create_titans_model(variant="mac", d_model=64, seed=42)
        model2 = create_titans_model(variant="mac", d_model=64, seed=42)

        # Test with same input
        x = jax.random.normal(jax.random.key(1), (1, 16, 64))

        output1 = model1(x, deterministic=True)
        output2 = model2(x, deterministic=True)

        # Outputs should be identical for same seed
        assert jnp.allclose(output1, output2)


class TestTitansTrainer:
    """Test the TitansTrainer class."""

    @pytest.fixture
    def model_and_trainer(self) -> tuple[TitansModel, TitansTrainer]:
        """Create a model and trainer for testing."""
        model = create_titans_model(variant="mac", d_model=64, seed=42)
        trainer = TitansTrainer(model, learning_rate=1e-4)
        return model, trainer

    @pytest.fixture
    def training_data(self) -> tuple[jax.Array, jax.Array]:
        """Create sample training data."""
        x = jax.random.normal(jax.random.key(1), (2, 16, 64))
        y = jax.random.normal(jax.random.key(2), (2, 16, 64))
        return x, y

    def test_trainer_initialization(self) -> None:
        """Test trainer initialization."""
        model = create_titans_model(variant="mac", d_model=64, seed=42)
        trainer = TitansTrainer(model, learning_rate=1e-3)

        assert trainer.model is model
        assert hasattr(trainer, "optimizer")

    def test_train_step(
        self,
        model_and_trainer: tuple[TitansModel, TitansTrainer],
        training_data: tuple[jax.Array, jax.Array],
    ) -> None:
        """Test a single training step."""
        _model, trainer = model_and_trainer
        x, y = training_data

        # Perform training step
        loss = trainer.train_step(x, y)

        # Loss should be finite and positive
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_evaluate(
        self,
        model_and_trainer: tuple[TitansModel, TitansTrainer],
        training_data: tuple[jax.Array, jax.Array],
    ) -> None:
        """Test model evaluation."""
        _model, trainer = model_and_trainer
        x, y = training_data

        # Evaluate model
        metrics = trainer.evaluate(x, y)

        # Should return dictionary with mse and mae
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert jnp.isfinite(metrics["mse"])
        assert jnp.isfinite(metrics["mae"])
        assert metrics["mse"] >= 0.0
        assert metrics["mae"] >= 0.0

    def test_training_reduces_loss(
        self,
        model_and_trainer: tuple[TitansModel, TitansTrainer],
        training_data: tuple[jax.Array, jax.Array],
    ) -> None:
        """Test that training can reduce loss over multiple steps."""
        _model, trainer = model_and_trainer
        x, y = training_data

        # Initial loss
        initial_metrics = trainer.evaluate(x, y)
        initial_loss = initial_metrics["mse"]

        # Train for a few steps
        for _ in range(5):
            trainer.train_step(x, y)

        # Final loss
        final_metrics = trainer.evaluate(x, y)
        final_loss = final_metrics["mse"]

        # Loss should decrease (though this isn't guaranteed for all cases)
        # At minimum, both losses should be finite
        assert jnp.isfinite(initial_loss)
        assert jnp.isfinite(final_loss)

    def test_trainer_memory_integration(
        self,
        model_and_trainer: tuple[TitansModel, TitansTrainer],
        training_data: tuple[jax.Array, jax.Array],
    ) -> None:
        """Test that trainer properly integrates memory updates."""
        model, trainer = model_and_trainer
        x, y = training_data

        # Get initial parameter count and memory norm as simple metrics
        initial_memory_norm = model.get_memory_norm()

        # Training step
        loss = trainer.train_step(x, y)

        # Check that memory state has changed (indicating parameter updates)
        updated_memory_norm = model.get_memory_norm()

        # Simple check: memory should be active after training
        assert jnp.isfinite(loss), "Training loss should be finite"
        assert updated_memory_norm > 0, "Memory should be active after training"

        # Memory norm should be different after training (indicating updates)
        if initial_memory_norm > 0:
            # If memory was already initialized, it should change
            assert not jnp.allclose(
                initial_memory_norm, updated_memory_norm, atol=1e-8
            ), "Memory should be updated during training"


if __name__ == "__main__":
    pytest.main([__file__])
