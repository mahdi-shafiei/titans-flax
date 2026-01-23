"""Neural Long-term Memory module for Titans architecture."""

from typing import NamedTuple

from flax import nnx
import jax
from jax import Array
import jax.numpy as jnp

# Named constants (per audit recommendation to extract magic numbers)
# Standard deviation for memory weight initialization
MEMORY_INIT_STDDEV: float = 0.01


class MemoryState(NamedTuple):
    """State of the neural memory including weights and momentum."""

    weights: list[Array]
    momentum: list[Array]


class NeuralLongTermMemory(nnx.Module):
    """Neural Long-term Memory that learns to memorize at test time.

    This implements the core neural memory module from the Titans paper,
    which uses gradient descent with momentum and weight decay to learn
    associations between keys and values during inference.
    """

    def __init__(
        self,
        d_model: int,
        memory_depth: int = 2,
        momentum_decay: float = 0.9,
        learning_rate: float = 0.01,
        forget_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize Neural Long-term Memory.

        Args:
            d_model: Model dimension
            memory_depth: Depth of the memory network (L_M in paper)
            momentum_decay: Momentum decay rate (η_t in paper)
            learning_rate: Learning rate for gradient updates (θ_t in paper)
            forget_rate: Forgetting gate rate (α_t in paper)
            rngs: Random number generators
            dtype: Data type for computations
        """
        super().__init__()

        # Validate inputs
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)
        if memory_depth <= 0:
            msg = f"memory_depth must be positive, got {memory_depth}"
            raise ValueError(msg)
        if not 0 <= momentum_decay <= 1:
            msg = f"momentum_decay must be in [0,1], got {momentum_decay}"
            raise ValueError(msg)
        if learning_rate <= 0:
            msg = f"learning_rate must be positive, got {learning_rate}"
            raise ValueError(msg)
        if not 0 <= forget_rate <= 1:
            msg = f"forget_rate must be in [0,1], got {forget_rate}"
            raise ValueError(msg)

        self.d_model = d_model
        self.memory_depth = memory_depth
        self.dtype = dtype

        # FIX: Store the initialization key properly
        self._init_key = rngs.params()

        # Key, Value, Query projections for associative memory
        self.key_proj = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
        )
        self.value_proj = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
        )
        self.query_proj = nnx.Linear(
            in_features=d_model,
            out_features=d_model,
            use_bias=True,
            rngs=rngs,
            dtype=dtype,
        )

        # Deep memory network layers - using nnx.List for proper state tracking
        # (plain Python lists won't properly track parameters with nnx.state())
        self.memory_layers = nnx.List(
            [
                nnx.Linear(
                    in_features=d_model,
                    out_features=d_model,
                    use_bias=True,
                    rngs=rngs,
                    dtype=dtype,
                )
                for _ in range(memory_depth)
            ]
        )

        # Trainable parameters for update mechanism
        self.momentum_decay = nnx.Param(
            jnp.array(momentum_decay, dtype=dtype), trainable=True
        )
        self.learning_rate = nnx.Param(
            jnp.array(learning_rate, dtype=dtype), trainable=True
        )
        self.forget_rate = nnx.Param(
            jnp.array(forget_rate, dtype=dtype), trainable=True
        )

    def _create_initial_memory_state(self) -> MemoryState:
        """Create initial memory state with small random values."""
        # FIX: Use the properly stored initialization key
        weights = []
        momentum = []

        # Split key for each layer to ensure different initializations
        keys = jax.random.split(self._init_key, self.memory_depth)

        for i, layer in enumerate(self.memory_layers):
            # Initialize with small random values instead of zeros
            w_shape = layer.kernel.value.shape
            # Use split key for this layer
            init_weight = (
                jax.random.normal(keys[i], w_shape, dtype=self.dtype)
                * MEMORY_INIT_STDDEV
            )
            weights.append(init_weight)
            momentum.append(jnp.zeros(w_shape, dtype=self.dtype))

        return MemoryState(weights=weights, momentum=momentum)

    def _apply_memory_network(
        self, x: Array, memory_weights: list[Array] | None = None
    ) -> Array:
        """Apply memory network with optional custom weights.

        Args:
            x: Input tensor
            memory_weights: Optional custom weights, uses layer weights if None

        Returns:
            Output after applying memory network
        """
        for i, layer in enumerate(self.memory_layers):
            if memory_weights is not None:
                # Use custom weights from memory state
                x = jnp.dot(x, memory_weights[i])
                if layer.bias is not None:
                    x = x + layer.bias.value
            else:
                # Use regular layer weights
                x = layer(x)

            # Apply activation (SiLU) except for last layer
            if i < len(self.memory_layers) - 1:
                x = nnx.silu(x)

        return x

    def _compute_associative_loss(
        self,
        keys: Array,
        values: Array,
        memory_weights: list[Array],
    ) -> Array:
        """Compute associative memory loss ℓ(M; x_t) = ||M(k_t) - v_t||²."""
        predicted_values = self._apply_memory_network(keys, memory_weights)
        return jnp.mean((predicted_values - values) ** 2)

    def _update_memory_state(
        self,
        keys: Array,
        values: Array,
        memory_state: MemoryState,
    ) -> MemoryState:
        """Update memory using gradient descent with momentum and weight decay.

        This implements the update rules from the paper:
        M_t = (1 - α_t)M_{t-1} + S_t
        S_t = η_t S_{t-1} - θ_t ∇ℓ(M_{t-1}; x_t)
        """

        def loss_fn(weights: list[Array]) -> Array:
            return self._compute_associative_loss(keys, values, weights)

        # Compute gradients using JAX grad
        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(memory_state.weights)

        # Update momentum and weights
        new_momentum = []
        new_weights = []

        for weight, momentum, grad in zip(
            memory_state.weights, memory_state.momentum, gradients
        ):
            # Update momentum: S_t = η_t S_{t-1} - θ_t ∇ℓ
            new_mom = (
                self.momentum_decay.value * momentum - self.learning_rate.value * grad
            )
            new_momentum.append(new_mom)

            # Update weights with forget gate: M_t = (1 - α_t)M_{t-1} + S_t
            new_w = (1 - self.forget_rate.value) * weight + new_mom
            new_weights.append(new_w)

        return MemoryState(weights=new_weights, momentum=new_momentum)

    def __call__(
        self,
        x: Array,
        memory_state: MemoryState | None = None,
        *,
        deterministic: bool = False,
        update_memory: bool = True,
    ) -> tuple[Array, MemoryState]:
        """Apply neural long-term memory.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            memory_state: Current memory state, creates new if None
            deterministic: If True, disable memory updates
            update_memory: If True, update memory state (test-time learning)

        Returns:
            Tuple of (output_tensor, updated_memory_state)
        """
        # Validate input
        if x.ndim != 3:
            msg = f"Expected 3D input (batch, seq_len, d_model), got {x.shape}"
            raise ValueError(msg)
        if x.shape[-1] != self.d_model:
            msg = f"Expected d_model={self.d_model}, got {x.shape[-1]}"
            raise ValueError(msg)

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = self._create_initial_memory_state()

        # Project to keys, values, queries
        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.query_proj(x)

        # Retrieve from memory using current memory weights
        # M*(q_t) in the paper notation
        # Stop gradient on memory weights during retrieval to prevent tracer leaks
        memory_weights_detached = [
            jax.lax.stop_gradient(w) for w in memory_state.weights
        ]
        retrieved = self._apply_memory_network(queries, memory_weights_detached)

        # Update memory state if enabled and not deterministic
        new_memory_state = memory_state
        if update_memory and not deterministic:
            # Average over sequence length for memory update
            avg_keys = jnp.mean(keys, axis=1)  # (batch, d_model)
            avg_values = jnp.mean(values, axis=1)  # (batch, d_model)

            # Further average over batch for single memory update
            batch_keys = jnp.mean(avg_keys, axis=0)  # (d_model,)
            batch_values = jnp.mean(avg_values, axis=0)  # (d_model,)

            # Update memory state
            new_memory_state = self._update_memory_state(
                batch_keys[None, :],  # Add batch dim for consistency
                batch_values[None, :],
                memory_state,
            )

        return retrieved, new_memory_state


class PersistentMemory(nnx.Module):
    """Persistent memory module with learnable input-independent parameters."""

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize persistent memory.

        Args:
            num_tokens: Number of persistent memory tokens (N_p in paper)
            d_model: Model dimension
            rngs: Random number generators
            dtype: Data type
        """
        super().__init__()

        # Validate inputs
        if num_tokens <= 0:
            msg = f"num_tokens must be positive, got {num_tokens}"
            raise ValueError(msg)
        if d_model <= 0:
            msg = f"d_model must be positive, got {d_model}"
            raise ValueError(msg)

        self.num_tokens = num_tokens
        self.d_model = d_model

        # Learnable persistent memory parameters
        self.tokens = nnx.Param(
            jax.random.normal(rngs.params(), (num_tokens, d_model), dtype=dtype)
            * 0.02,  # Small initialization
            trainable=True,
        )

    def __call__(self, batch_size: int) -> Array:
        """Get persistent memory tokens for a batch.

        Args:
            batch_size: Batch size

        Returns:
            Persistent memory tokens of shape (batch, num_tokens, d_model)
        """
        if batch_size <= 0:
            msg = f"batch_size must be positive, got {batch_size}"
            raise ValueError(msg)

        # Expand for batch size
        return jnp.broadcast_to(
            self.tokens.value[None, :, :], (batch_size, self.num_tokens, self.d_model)
        )
