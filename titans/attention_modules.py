"""Attention modules for Titans architecture using built-in Flax NNX components."""

from flax import nnx
from jax import Array
import jax.numpy as jnp


def create_causal_mask(seq_len: int) -> Array:
    """Create causal mask for attention."""
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))


def create_sliding_window_mask(seq_len: int, window_size: int) -> Array:
    """Create sliding window causal mask."""
    # Create indices
    i = jnp.arange(seq_len)[:, None]
    j = jnp.arange(seq_len)[None, :]

    # Causal mask: j <= i
    causal_mask = j <= i

    # Window mask: i - window_size + 1 <= j
    window_mask = i - window_size + 1 <= j

    return causal_mask & window_mask


class CausalAttention(nnx.Module):
    """Causal attention using built-in nnx.MultiHeadAttention."""

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()

        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            decode=False,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array, *, deterministic: bool = False) -> Array:
        seq_len = x.shape[1]
        mask = create_causal_mask(seq_len)
        return self.attention(x, mask=mask, deterministic=deterministic)


class SlidingWindowAttention(nnx.Module):
    """Sliding window attention using built-in nnx.MultiHeadAttention."""

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        window_size: int,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()

        if window_size <= 0:
            msg = f"window_size must be positive, got {window_size}"
            raise ValueError(msg)

        self.window_size = window_size
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=in_features,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            decode=False,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array, *, deterministic: bool = False) -> Array:
        seq_len = x.shape[1]
        mask = create_sliding_window_mask(seq_len, self.window_size)
        return self.attention(x, mask=mask, deterministic=deterministic)


class ConvolutionalLayer(nnx.Module):
    """1D convolutional layer for local processing."""

    def __init__(
        self,
        features: int,
        kernel_size: int = 3,
        use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()

        if features <= 0:
            msg = f"features must be positive, got {features}"
            raise ValueError(msg)
        if kernel_size <= 0:
            msg = f"kernel_size must be positive, got {kernel_size}"
            raise ValueError(msg)

        self.conv = nnx.Conv(
            in_features=features,
            out_features=features,
            kernel_size=(kernel_size,),
            padding="SAME",
            use_bias=use_bias,
            rngs=rngs,
            dtype=dtype,
        )

        self.norm = nnx.LayerNorm(
            num_features=features,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(self, x: Array) -> Array:
        # Reshape for 1D convolution: (batch, seq_len, 1, features)
        x_reshaped = x[..., None, :]
        conv_out = self.conv(x_reshaped)
        conv_out = conv_out[..., 0, :]  # Remove spatial dimension
        return self.norm(conv_out + x)  # Residual connection
