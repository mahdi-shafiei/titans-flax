"""Main module for Titans implementation with nnx.Module architecture."""

from typing import Any, Literal

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from .neural_memory import MemoryState
from .variants import TitansLMM, TitansMAC, TitansMAG, TitansMAL

# Named constants (per audit recommendation to extract magic numbers)
# Weight for memory regularization loss term
MEMORY_REGULARIZATION_WEIGHT: float = 1e-6

TitansVariant = Literal["mac", "mag", "mal", "lmm"]


class TitansModel(nnx.Module):
    """Unified interface for all Titans variants.

    This class provides a clean API for creating and using different
    Titans model variants with consistent interfaces. Now properly
    inherits from nnx.Module for JAX transformation compatibility.
    """

    def __init__(
        self,
        variant: TitansVariant,
        d_model: int,
        num_heads: int = 8,
        memory_depth: int = 2,
        num_persistent_tokens: int = 64,
        chunk_size: int = 512,
        window_size: int = 256,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
        dtype: Any = jnp.float32,
    ):
        """Initialize Titans model.

        Args:
            variant: Which Titans variant to use ("mac", "mag", "mal", "lmm")
            d_model: Model dimension
            num_heads: Number of attention heads (ignored for "lmm")
            memory_depth: Depth of neural memory network
            num_persistent_tokens: Number of persistent memory tokens
            chunk_size: Chunk size for MAC variant
            window_size: Window size for MAG/MAL variants
            dropout_rate: Dropout rate
            rngs: Random number generators
            dtype: Data type
        """
        super().__init__()

        # Validate inputs
        if variant not in ["mac", "mag", "mal", "lmm"]:
            msg = f"Unknown variant: {variant}"
            raise ValueError(msg)
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)
        if num_heads <= 0:
            msg = f"num_heads must be positive, got {num_heads}"
            raise ValueError(msg)
        if memory_depth <= 0:
            msg = f"memory_depth must be positive, got {memory_depth}"
            raise ValueError(msg)
        if num_persistent_tokens <= 0:
            msg = f"num_persistent_tokens must be positive, got {num_persistent_tokens}"
            raise ValueError(msg)

        self.variant = variant
        self.d_model = d_model

        # FIXED: Direct delegation - no memory state copying
        # Memory state is now managed entirely by the variant model
        if variant == "mac":
            self.model = TitansMAC(
                d_model=d_model,
                num_heads=num_heads,
                chunk_size=chunk_size,
                memory_depth=memory_depth,
                num_persistent_tokens=num_persistent_tokens,
                dropout_rate=dropout_rate,
                rngs=rngs,
                dtype=dtype,
            )
        elif variant == "mag":
            self.model = TitansMAG(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                memory_depth=memory_depth,
                num_persistent_tokens=num_persistent_tokens,
                dropout_rate=dropout_rate,
                rngs=rngs,
                dtype=dtype,
            )
        elif variant == "mal":
            self.model = TitansMAL(
                d_model=d_model,
                num_heads=num_heads,
                window_size=window_size,
                memory_depth=memory_depth,
                num_persistent_tokens=num_persistent_tokens,
                dropout_rate=dropout_rate,
                rngs=rngs,
                dtype=dtype,
            )
        elif variant == "lmm":
            self.model = TitansLMM(
                d_model=d_model,
                memory_depth=memory_depth,
                num_persistent_tokens=num_persistent_tokens,
                rngs=rngs,
                dtype=dtype,
            )

        # FIXED: Initialize internal memory state tracking variable
        # This is used only for the wrapper API consistency
        self._current_memory_state: nnx.Variable[MemoryState | None] = nnx.Variable(
            None
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = False,
        reset_memory: bool = False,
    ) -> jax.Array:
        """Apply model to input sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            deterministic: Whether to disable memory updates and dropout
            reset_memory: Whether to reset memory state

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Validate input
        if x.ndim != 3:
            msg = f"Expected 3D input (batch, seq_len, d_model), got {x.shape}"
            raise ValueError(msg)
        if x.shape[-1] != self.d_model:
            msg = f"Expected d_model={self.d_model}, got {x.shape[-1]}"
            raise ValueError(msg)

        # FIXED: Handle memory reset directly at variant level
        if reset_memory:
            self._current_memory_state.value = None

        # FIXED: Direct delegation with proper memory state management
        output, new_memory_state = self.model(
            x,
            memory_state=self._current_memory_state.value,
            deterministic=deterministic,
        )

        # FIXED: Update internal memory state tracking
        # This maintains consistency with the wrapper API
        if not deterministic:
            self._current_memory_state.value = new_memory_state

        return output

    def reset_memory(self) -> None:
        """Reset the memory state to initial conditions."""
        self._current_memory_state.value = None

    def get_memory_state(self) -> MemoryState | None:
        """Get current memory state."""
        return self._current_memory_state.value

    def set_memory_state(self, memory_state: MemoryState | None) -> None:
        """Set memory state.

        Args:
            memory_state: New memory state to set, or None to reset
        """
        # Runtime validation for defensive programming
        if memory_state is not None and not isinstance(memory_state, MemoryState):  # pyright: ignore[reportUnnecessaryIsInstance]
            msg = f"Expected MemoryState or None, got {type(memory_state)}"
            raise TypeError(msg)
        self._current_memory_state.value = memory_state

    def get_memory_norm(self) -> float | jax.Array:
        """Get the norm of the current memory state.

        Returns:
            The L2 norm of all memory weights, or 0.0 if no memory exists.
            When called within a JAX transformation context, returns a JAX array.
            Otherwise returns a concrete float.
        """
        if self._current_memory_state.value is None:
            return 0.0

        # Compute the norm using JAX operations, avoiding early concretization
        total_norm_squared = 0.0
        for weight in self._current_memory_state.value.weights:
            total_norm_squared = total_norm_squared + jnp.sum(weight**2)

        total_norm = jnp.sqrt(total_norm_squared)

        # Only convert to float if we're not in a JAX transformation context
        # This avoids the ConcretizationTypeError during gradient computation
        try:
            return float(total_norm)
        except jax.errors.ConcretizationTypeError:
            # If we're in a traced context, return the JAX array as float
            # This is a workaround for the type annotation constraint
            return total_norm  # type: ignore

    def count_params(self) -> int:
        """Count total number of parameters in the model.

        Uses nnx.iter_graph for efficient traversal with thorough parameter detection.
        This approach is faster than recursive state traversal while maintaining accuracy.

        Returns:
            Total number of parameters
        """
        total_params = 0

        # Use iter_graph to traverse all graph nodes
        # This returns individual objects, including parameter objects
        for _path, obj in nnx.iter_graph(self):
            # Check if this object is a parameter
            if (
                hasattr(obj, "value")
                and hasattr(obj.value, "size")
                and hasattr(obj, "__class__")
                and "Param" in str(obj.__class__)
            ):
                total_params += int(obj.value.size)

        return total_params


def create_titans_model(
    variant: TitansVariant,
    d_model: int,
    seed: int = 42,
    **kwargs: Any,
) -> TitansModel:
    """Convenience function to create a Titans model.

    Args:
        variant: Which Titans variant to use
        d_model: Model dimension
        seed: Random seed
        **kwargs: Additional arguments for TitansModel

    Returns:
        Initialized TitansModel
    """
    # Validate inputs
    if variant not in ["mac", "mag", "mal", "lmm"]:
        msg = f"Unknown variant: {variant}. Choose from: mac, mag, mal, lmm"
        raise ValueError(msg)
    if d_model <= 0:
        msg = f"d_model must be positive, got {d_model}"
        raise ValueError(msg)

    rngs = nnx.Rngs(seed)

    return TitansModel(variant=variant, d_model=d_model, rngs=rngs, **kwargs)


# JIT-compiled forward pass for optimized inference
# Note: static_argnames needed for boolean args that control tracing behavior
@nnx.jit(static_argnames=("deterministic", "reset_memory"))
def titans_forward(
    model: TitansModel,
    x: jax.Array,
    *,
    deterministic: bool = True,
    reset_memory: bool = False,
) -> jax.Array:
    """JIT-compiled forward pass for optimized inference.

    This function provides JIT-compiled inference for Titans models.
    Using @nnx.jit handles split/merge automatically, provides clean API,
    and propagates state updates back to original module.

    Args:
        model: TitansModel instance
        x: Input tensor of shape (batch, seq_len, d_model)
        deterministic: Whether to disable memory updates (default True for inference)
        reset_memory: Whether to reset memory state before forward pass

    Returns:
        Output tensor of shape (batch, seq_len, d_model)

    Example:
        >>> model = create_titans_model("mac", d_model=128)
        >>> x = jnp.ones((2, 64, 128))
        >>> output = titans_forward(model, x)  # JIT-compiled
    """
    return model(x, deterministic=deterministic, reset_memory=reset_memory)


class TitansTrainer:
    """Training utilities for Titans models with FIXED memory-gradient integration."""

    def __init__(
        self,
        model: TitansModel,
        learning_rate: float = 1e-4,
    ):
        """Initialize trainer.

        Args:
            model: Titans model to train
            learning_rate: Learning rate for optimizer
        """
        # Validate inputs before assignment
        if learning_rate <= 0:
            msg = f"learning_rate must be positive, got {learning_rate}"
            raise ValueError(msg)

        self.model = model
        # UPDATED for Flax 0.11.0+: wrt parameter is now required
        self.optimizer = nnx.Optimizer(
            model,
            optax.adam(learning_rate=learning_rate),
            wrt=nnx.Param,  # Specify which variables to optimize
        )

    def train_step(
        self,
        x: jax.Array,
        y: jax.Array,
    ) -> float:
        """Perform a single training step with FIXED memory-gradient integration.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Loss value
        """
        # Validate inputs
        if x.shape[:-1] != y.shape[:-1]:
            msg = f"Input and target shape mismatch: {x.shape} vs {y.shape}"
            raise ValueError(msg)

        def loss_fn(model: TitansModel) -> jax.Array:
            # FIXED: Memory updates are now properly included in gradient computation
            # The model handles memory state internally, and gradients flow through
            # both model parameters AND memory updates
            predictions = model(x, deterministic=False)  # Enable memory updates

            # Basic MSE loss
            mse_loss = jnp.mean((predictions - y) ** 2)

            # Optional: Add memory regularization to prevent explosion
            memory_state = model.get_memory_state()
            memory_reg = 0.0
            if memory_state is not None:
                # Small regularization on memory magnitude
                memory_norm = jnp.sqrt(sum(jnp.sum(w**2) for w in memory_state.weights))
                memory_reg = MEMORY_REGULARIZATION_WEIGHT * memory_norm

            return mse_loss + memory_reg

        # FIXED: Single gradient computation that includes memory updates
        # UPDATED for Flax 0.11.0+: use self.model instead of self.optimizer.model
        loss, grads = nnx.value_and_grad(loss_fn)(self.model)
        self.optimizer.update(self.model, grads)  # Pass both model and grads

        return float(loss)

    def evaluate(
        self,
        x: jax.Array,
        y: jax.Array,
    ) -> dict[str, float]:
        """Evaluate model on data.

        Args:
            x: Input tensor
            y: Target tensor

        Returns:
            Dictionary of metrics
        """
        # Use deterministic=True for evaluation (no memory updates)
        # UPDATED for Flax 0.11.0+: use self.model instead of self.optimizer.model
        predictions = self.model(x, deterministic=True)

        mse = jnp.mean((predictions - y) ** 2)
        mae = jnp.mean(jnp.abs(predictions - y))

        return {
            "mse": float(mse),
            "mae": float(mae),
        }
