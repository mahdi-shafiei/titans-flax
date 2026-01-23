"""Titans architecture variants: MAC, MAG, MAL, and LMM.

This implements the four variants of Titans with clean architecture:
- Memory as Context (MAC)
- Memory as Gate (MAG)
- Memory as Layer (MAL)
- Long-term Memory Module (LMM)
"""

from flax import nnx
import jax
import jax.numpy as jnp

from .attention_modules import CausalAttention, SlidingWindowAttention
from .neural_memory import MemoryState, NeuralLongTermMemory, PersistentMemory
from .param_validation import validate_3d_input, validate_positive


class TitansMAC(nnx.Module):
    """Titans with Memory as Context (MAC).

    This variant treats memory as context to the current information.
    The sequence is chunked and memory retrieval is concatenated with
    persistent memory and current segment before attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        chunk_size: int,
        memory_depth: int = 2,
        num_persistent_tokens: int = 64,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize Titans MAC."""
        super().__init__()

        # Validate inputs (DRY: using shared validation utilities)
        validate_positive(d_model, "d_model")
        validate_positive(chunk_size, "chunk_size")

        self.d_model = d_model
        self.chunk_size = chunk_size

        # Core components
        self.neural_memory = NeuralLongTermMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            rngs=rngs,
            dtype=dtype,
        )

        self.persistent_memory = PersistentMemory(
            num_tokens=num_persistent_tokens,
            d_model=d_model,
            rngs=rngs,
            dtype=dtype,
        )

        self.attention = CausalAttention(
            num_heads=num_heads,
            in_features=d_model,
            dropout_rate=dropout_rate,
            rngs=rngs,
            dtype=dtype,
        )

        # Layer normalization
        self.norm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs, dtype=dtype)
        self.norm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs, dtype=dtype)

        # Feed-forward network
        self.ffn = nnx.Sequential(
            nnx.Linear(d_model, 4 * d_model, rngs=rngs, dtype=dtype),
            nnx.gelu,
            nnx.Linear(4 * d_model, d_model, rngs=rngs, dtype=dtype),
        )

        # Output projection for memory interaction
        self.memory_gate = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            rngs=rngs,
            dtype=dtype,
        )

    def _chunk_sequence(self, x: jax.Array) -> list[jax.Array]:
        """Split sequence into chunks."""
        seq_len = x.shape[1]
        chunks = []

        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = x[:, i:end_idx, :]
            chunks.append(chunk)

        return chunks

    def _process_chunk(
        self,
        chunk: jax.Array,
        memory_state: MemoryState,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, MemoryState]:
        """Process a single chunk with memory retrieval."""
        batch_size = chunk.shape[0]

        # Retrieve from long-term memory
        retrieved_memory, new_memory_state = self.neural_memory(
            chunk,
            memory_state,
            deterministic=deterministic,
            update_memory=not deterministic,
        )

        # Get persistent memory
        persistent_tokens = self.persistent_memory(batch_size)

        # Concatenate: persistent_memory || retrieved_memory || current_chunk
        combined_input = jnp.concatenate(
            [persistent_tokens, retrieved_memory, chunk], axis=1
        )

        # Apply attention
        attn_out = self.attention(combined_input, deterministic=deterministic)

        # Extract the output corresponding to the original chunk
        chunk_start_idx = persistent_tokens.shape[1] + retrieved_memory.shape[1]
        chunk_output = attn_out[:, chunk_start_idx:, :]

        # Residual connection and layer norm
        chunk_output = self.norm1(chunk + chunk_output)

        # Feed-forward with residual
        ffn_out = self.ffn(chunk_output)
        chunk_output = self.norm2(chunk_output + ffn_out)

        # Compute final output with memory interaction
        memory_interaction = self.memory_gate(retrieved_memory)
        # Average pool memory interaction to match chunk length
        pooled_memory = jnp.mean(memory_interaction, axis=1, keepdims=True)
        pooled_memory = jnp.broadcast_to(pooled_memory, chunk_output.shape)

        final_output = chunk_output * nnx.sigmoid(pooled_memory)

        return final_output, new_memory_state

    def __call__(
        self,
        x: jax.Array,
        memory_state: MemoryState | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, MemoryState]:
        """Apply Titans MAC to input sequence."""
        # Validate input (DRY: using shared validation utilities)
        validate_3d_input(x, expected_d_model=self.d_model)

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self.neural_memory._create_initial_memory_state()

        # Split input into chunks
        chunks = self._chunk_sequence(x)

        # Process each chunk sequentially
        output_chunks = []
        current_memory_state = memory_state

        for chunk in chunks:
            chunk_output, current_memory_state = self._process_chunk(
                chunk,
                current_memory_state,
                deterministic=deterministic,
            )
            output_chunks.append(chunk_output)

        # Concatenate output chunks
        output = jnp.concatenate(output_chunks, axis=1)

        return output, current_memory_state


class TitansMAG(nnx.Module):
    """Titans with Memory as Gate (MAG).

    This variant uses sliding window attention combined with neural memory
    through a gating mechanism.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        memory_depth: int = 2,
        num_persistent_tokens: int = 64,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize Titans MAG."""
        super().__init__()

        # Validate inputs (DRY: using shared validation utilities)
        validate_positive(d_model, "d_model")
        validate_positive(window_size, "window_size")

        self.d_model = d_model

        # Core components
        self.neural_memory = NeuralLongTermMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            rngs=rngs,
            dtype=dtype,
        )

        self.persistent_memory = PersistentMemory(
            num_tokens=num_persistent_tokens,
            d_model=d_model,
            rngs=rngs,
            dtype=dtype,
        )

        self.sliding_attention = SlidingWindowAttention(
            num_heads=num_heads,
            in_features=d_model,
            window_size=window_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
            dtype=dtype,
        )

        # Gating mechanism
        self.gate_proj = nnx.Linear(
            in_features=2 * d_model,
            out_features=d_model,
            rngs=rngs,
            dtype=dtype,
        )

        # Normalization layers
        self.norm_attention = nnx.LayerNorm(
            num_features=d_model, rngs=rngs, dtype=dtype
        )
        self.norm_memory = nnx.LayerNorm(num_features=d_model, rngs=rngs, dtype=dtype)

    def __call__(
        self,
        x: jax.Array,
        memory_state: MemoryState | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, MemoryState]:
        """Apply Titans MAG to input sequence."""
        # Validate input (DRY: using shared validation utilities)
        validate_3d_input(x, expected_d_model=self.d_model)

        batch_size = x.shape[0]

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self.neural_memory._create_initial_memory_state()

        # Get persistent memory and prepend to input
        persistent_tokens = self.persistent_memory(batch_size)
        x_with_persistent = jnp.concatenate([persistent_tokens, x], axis=1)

        # Apply sliding window attention (short-term memory)
        attention_output = self.sliding_attention(
            x_with_persistent, deterministic=deterministic
        )

        # Extract attention output for original sequence
        attn_out = attention_output[:, persistent_tokens.shape[1] :, :]
        attn_out = self.norm_attention(attn_out)

        # Apply neural memory (long-term memory)
        memory_output, new_memory_state = self.neural_memory(
            x,
            memory_state,
            deterministic=deterministic,
            update_memory=not deterministic,
        )
        memory_out = self.norm_memory(memory_output)

        # Combine via gating mechanism
        combined = jnp.concatenate([attn_out, memory_out], axis=-1)
        gate = nnx.sigmoid(self.gate_proj(combined))

        # Gated combination
        output = gate * attn_out + (1 - gate) * memory_out

        return output, new_memory_state


class TitansMAL(nnx.Module):
    """Titans with Memory as Layer (MAL).

    This variant uses neural memory as a layer followed by sliding window attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        memory_depth: int = 2,
        num_persistent_tokens: int = 64,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize Titans MAL."""
        super().__init__()

        # Validate inputs (DRY: using shared validation utilities)
        validate_positive(d_model, "d_model")
        validate_positive(window_size, "window_size")

        # Core components
        self.neural_memory = NeuralLongTermMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            rngs=rngs,
            dtype=dtype,
        )

        self.persistent_memory = PersistentMemory(
            num_tokens=num_persistent_tokens,
            d_model=d_model,
            rngs=rngs,
            dtype=dtype,
        )

        self.sliding_attention = SlidingWindowAttention(
            num_heads=num_heads,
            in_features=d_model,
            window_size=window_size,
            dropout_rate=dropout_rate,
            rngs=rngs,
            dtype=dtype,
        )

        # Normalization
        self.norm = nnx.LayerNorm(num_features=d_model, rngs=rngs, dtype=dtype)

    def __call__(
        self,
        x: jax.Array,
        memory_state: MemoryState | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, MemoryState]:
        """Apply Titans MAL to input sequence."""
        # Validate input (DRY: using shared validation utilities)
        validate_3d_input(x)

        batch_size = x.shape[0]

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self.neural_memory._create_initial_memory_state()

        # Get persistent memory and prepend to input
        persistent_tokens = self.persistent_memory(batch_size)
        x_with_persistent = jnp.concatenate([persistent_tokens, x], axis=1)

        # Apply neural memory layer first
        memory_output, new_memory_state = self.neural_memory(
            x_with_persistent,
            memory_state,
            deterministic=deterministic,
            update_memory=not deterministic,
        )

        # Normalize memory output
        memory_output = self.norm(memory_output)

        # Apply sliding window attention
        output = self.sliding_attention(memory_output, deterministic=deterministic)

        # Extract output for original sequence (remove persistent tokens)
        final_output = output[:, persistent_tokens.shape[1] :, :]

        return final_output, new_memory_state


class TitansLMM(nnx.Module):
    """Titans with just the Neural Memory Module (LMM).

    This variant uses only the neural long-term memory without attention,
    as mentioned in the paper for testing memory capability independently.
    """

    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        num_persistent_tokens: int = 64,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize Titans LMM."""
        super().__init__()

        # Validate inputs (DRY: using shared validation utilities)
        validate_positive(d_model, "d_model")

        # Core components
        self.neural_memory = NeuralLongTermMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            rngs=rngs,
            dtype=dtype,
        )

        self.persistent_memory = PersistentMemory(
            num_tokens=num_persistent_tokens,
            d_model=d_model,
            rngs=rngs,
            dtype=dtype,
        )

        # Output projection
        self.output_proj = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(
        self,
        x: jax.Array,
        memory_state: MemoryState | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, MemoryState]:
        """Apply Titans LMM to input sequence."""
        # Validate input (DRY: using shared validation utilities)
        validate_3d_input(x)

        batch_size = x.shape[0]

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self.neural_memory._create_initial_memory_state()

        # Get persistent memory and prepend to input
        persistent_tokens = self.persistent_memory(batch_size)
        x_with_persistent = jnp.concatenate([persistent_tokens, x], axis=1)

        # Apply neural memory
        memory_output, new_memory_state = self.neural_memory(
            x_with_persistent,
            memory_state,
            deterministic=deterministic,
            update_memory=not deterministic,
        )

        # Extract output for original sequence
        output = memory_output[:, persistent_tokens.shape[1] :, :]

        # Apply output projection
        output = self.output_proj(output)

        return output, new_memory_state


def test_titans_variants() -> None:
    """Test all Titans variants."""
    # Test parameters
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 8
    rngs = nnx.Rngs(42)

    # Create test input
    inputs = jnp.ones((batch_size, seq_len, d_model))

    # Test TitansMAC
    mac_model = TitansMAC(
        d_model=d_model,
        num_heads=num_heads,
        chunk_size=16,
        memory_depth=2,
        rngs=rngs,
    )
    mac_out, _mac_memory = mac_model(inputs, deterministic=True)
    assert mac_out.shape == inputs.shape

    # Test TitansMAG
    mag_model = TitansMAG(
        d_model=d_model,
        num_heads=num_heads,
        window_size=16,
        memory_depth=2,
        rngs=rngs,
    )
    mag_out, _mag_memory = mag_model(inputs, deterministic=True)
    assert mag_out.shape == inputs.shape

    # Test TitansMAL
    mal_model = TitansMAL(
        d_model=d_model,
        num_heads=num_heads,
        window_size=16,
        memory_depth=2,
        rngs=rngs,
    )
    mal_out, _mal_memory = mal_model(inputs, deterministic=True)
    assert mal_out.shape == inputs.shape

    # Test TitansLMM
    lmm_model = TitansLMM(
        d_model=d_model,
        memory_depth=2,
        rngs=rngs,
    )
    lmm_out, _lmm_memory = lmm_model(inputs, deterministic=True)
    assert lmm_out.shape == inputs.shape


if __name__ == "__main__":
    test_titans_variants()
