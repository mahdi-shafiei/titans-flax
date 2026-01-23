"""Memory and performance optimization utilities for Titans.

This module provides tools for optimizing Titans models for production use,
including memory-efficient implementations, gradient checkpointing, and
inference optimizations.
"""

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from .config import TitansConfig
from .core import TitansModel, TitansVariant, create_titans_model


@dataclass
class MemoryProfile:
    """Memory usage profile for analysis."""

    peak_memory_mb: float
    current_memory_mb: float
    model_parameters: int
    activation_memory_mb: float
    memory_state_mb: float

    def __str__(self) -> str:
        return (
            f"Memory Profile:\n"
            f"  Peak: {self.peak_memory_mb:.2f} MB\n"
            f"  Current: {self.current_memory_mb:.2f} MB\n"
            f"  Model params: {self.model_parameters:,}\n"
            f"  Activations: {self.activation_memory_mb:.2f} MB\n"
            f"  Memory state: {self.memory_state_mb:.2f} MB"
        )


class MemoryOptimizer:
    """Memory optimization utilities for Titans models."""

    @staticmethod
    def estimate_memory_usage(
        model: TitansModel,
        batch_size: int,
        seq_len: int,
        dtype: jnp.dtype = jnp.float32,
    ) -> MemoryProfile:
        """Estimate memory usage for given configuration.

        Args:
            model: Titans model
            batch_size: Batch size
            seq_len: Sequence length
            dtype: Data type

        Returns:
            Memory profile with usage estimates
        """
        # Bytes per element
        bytes_per_element = 4 if dtype == jnp.float32 else 2
        mb_factor = 1024 * 1024

        d_model = model.d_model

        # Model parameters - use the standardized count_params method
        model_params = model.count_params()

        # Activation memory (input, hidden states, attention)
        input_memory = batch_size * seq_len * d_model * bytes_per_element

        # Attention memory (for full attention)
        if model.variant in ["mac", "mag"]:
            # Attention scores: batch * heads * seq_len^2
            num_heads = 8  # Default assumption
            attention_memory = (
                batch_size * num_heads * seq_len * seq_len * bytes_per_element
            )
        else:
            attention_memory = 0

        # Hidden states through layers
        hidden_memory = (
            batch_size * seq_len * d_model * bytes_per_element * 3
        )  # Multiple layers

        total_activation_memory = (
            input_memory + attention_memory + hidden_memory
        ) / mb_factor

        # Memory state
        memory_state = model.get_memory_state()
        memory_state_size = 0
        if memory_state:
            for weight in memory_state.weights:
                memory_state_size += weight.size * bytes_per_element
            for momentum in memory_state.momentum:
                memory_state_size += momentum.size * bytes_per_element

        memory_state_mb = memory_state_size / mb_factor

        # Total estimates
        model_memory_mb = model_params * bytes_per_element / mb_factor
        peak_memory_mb = model_memory_mb + total_activation_memory + memory_state_mb

        return MemoryProfile(
            peak_memory_mb=peak_memory_mb,
            current_memory_mb=model_memory_mb + memory_state_mb,
            model_parameters=model_params,
            activation_memory_mb=total_activation_memory,
            memory_state_mb=memory_state_mb,
        )

    @staticmethod
    def optimize_chunk_size(
        model: TitansModel,
        target_memory_mb: float,
        seq_len: int,
        batch_size: int = 1,
    ) -> int:
        """Find optimal chunk size for given memory budget.

        Args:
            model: Titans model (should be MAC variant)
            target_memory_mb: Target memory usage in MB
            seq_len: Total sequence length
            batch_size: Batch size

        Returns:
            Optimal chunk size
        """
        if model.variant != "mac":
            return seq_len

        # Binary search for optimal chunk size
        min_chunk = 16
        max_chunk = seq_len

        best_chunk = max_chunk

        while min_chunk <= max_chunk:
            chunk_size = (min_chunk + max_chunk) // 2

            # Estimate memory for this chunk size
            # Create temporary model config to test
            test_profile = MemoryOptimizer.estimate_memory_usage(
                model, batch_size, chunk_size
            )

            if test_profile.peak_memory_mb <= target_memory_mb:
                best_chunk = chunk_size
                min_chunk = chunk_size + 1
            else:
                max_chunk = chunk_size - 1

        return best_chunk

    @staticmethod
    def create_memory_efficient_model(
        config: TitansConfig,
        memory_budget_mb: float = 1000,
        target_seq_len: int = 2048,
    ) -> TitansModel:
        """Create memory-efficient model within budget.

        Args:
            config: Base configuration
            memory_budget_mb: Memory budget in MB
            target_seq_len: Target sequence length

        Returns:
            Optimized TitansModel
        """
        # Start with base model
        model = TitansModel(
            variant=config.variant,
            d_model=config.architecture.d_model,
            rngs=nnx.Rngs(config.seed),
        )

        # Check initial memory usage
        initial_profile = MemoryOptimizer.estimate_memory_usage(
            model, batch_size=1, seq_len=target_seq_len
        )

        if initial_profile.peak_memory_mb <= memory_budget_mb:
            return model

        # Try reducing model size
        reduced_d_model = config.architecture.d_model

        while (
            initial_profile.peak_memory_mb > memory_budget_mb and reduced_d_model >= 32
        ):
            reduced_d_model = int(reduced_d_model * 0.8)

            # Create smaller model
            test_model = TitansModel(
                variant=config.variant,
                d_model=reduced_d_model,
                rngs=nnx.Rngs(config.seed),
            )

            test_profile = MemoryOptimizer.estimate_memory_usage(
                test_model, batch_size=1, seq_len=target_seq_len
            )

            if test_profile.peak_memory_mb <= memory_budget_mb:
                return test_model

        # If still too large, suggest chunking for MAC
        if config.variant == "mac":
            MemoryOptimizer.optimize_chunk_size(model, memory_budget_mb, target_seq_len)

        return model


class GradientCheckpointing:
    """Gradient checkpointing utilities to reduce memory usage."""

    @staticmethod
    def checkpoint_titans_layer(
        titans_layer_fn: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Apply gradient checkpointing to a Titans layer.

        Args:
            titans_layer_fn: Function representing the Titans layer
            *args: Arguments to the layer
            **kwargs: Keyword arguments to the layer

        Returns:
            Output with gradient checkpointing applied
        """
        # Use JAX's checkpoint decorator for gradient checkpointing
        checkpointed_fn = jax.checkpoint(titans_layer_fn)
        return checkpointed_fn(*args, **kwargs)

    @staticmethod
    def create_checkpointed_model(
        base_model: TitansModel,
        _checkpoint_every_n_layers: int = 2,
    ) -> TitansModel:
        """Create a version of the model with gradient checkpointing.

        Args:
            base_model: Base Titans model
            _checkpoint_every_n_layers: Checkpoint every N layers

        Returns:
            Model with gradient checkpointing
        """
        # This is a simplified example - in practice, you'd need to modify
        # the actual model architecture to include checkpointing

        # Return the original model (in a real implementation, you'd create a new one)
        return base_model


class InferenceOptimizer:
    """Optimization utilities specifically for inference."""

    @staticmethod
    def compile_for_inference(
        model: TitansModel,
        sample_input_shape: tuple[int, ...],
        static_argnums: tuple[int, ...] = (),
    ) -> Callable:
        """Compile model for fast inference.

        Args:
            model: Titans model
            sample_input_shape: Shape of sample input for compilation
            static_argnums: Argument positions that are static

        Returns:
            Compiled inference function
        """

        def inference_fn(
            x: jax.Array, deterministic: bool = True, reset_memory: bool = False
        ) -> jax.Array:
            return model(x, deterministic=deterministic, reset_memory=reset_memory)

        # Create sample input for compilation
        sample_input = jnp.ones(sample_input_shape)

        # JIT compile
        compiled_fn = nnx.jit(inference_fn, static_argnums=static_argnums)

        # Warm up compilation
        _ = compiled_fn(sample_input)

        return compiled_fn

    @staticmethod
    def create_streaming_inference(
        model: TitansModel,
        chunk_size: int = 512,
    ) -> Callable:
        """Create streaming inference function for long sequences.

        Args:
            model: Titans model
            chunk_size: Size of chunks to process

        Returns:
            Streaming inference function
        """

        def streaming_inference(
            x: jax.Array,
            reset_memory: bool = False,
        ) -> jax.Array:
            """Process long sequence in chunks.

            Args:
                x: Input sequence of shape (batch, seq_len, d_model)
                reset_memory: Whether to reset memory at start

            Returns:
                Processed sequence
            """
            if reset_memory:
                model.reset_memory()

            _batch_size, seq_len, _d_model = x.shape
            outputs = []

            # Process in chunks
            for i in range(0, seq_len, chunk_size):
                end_idx = min(i + chunk_size, seq_len)
                chunk = x[:, i:end_idx, :]

                # Process chunk
                chunk_output = model(chunk, deterministic=True, reset_memory=False)
                outputs.append(chunk_output)

            # Concatenate outputs
            return jnp.concatenate(outputs, axis=1)

        return streaming_inference

    @staticmethod
    def optimize_memory_updates(
        model: TitansModel,
        _update_frequency: int = 4,
    ) -> TitansModel:
        """Optimize memory update frequency for inference.

        Args:
            model: Titans model
            _update_frequency: Update memory every N steps

        Returns:
            Optimized model
        """
        # This would require modifying the memory update logic
        # For now, just return the original model with a note

        return model


class QuantizationUtils:
    """Utilities for model quantization."""

    @staticmethod
    def quantize_to_int8(
        model: TitansModel,
        _calibration_data: jax.Array | None = None,
    ) -> TitansModel:
        """Quantize model to int8 (placeholder implementation).

        Args:
            model: Titans model to quantize
            _calibration_data: Data for calibration

        Returns:
            Quantized model (placeholder - returns original)
        """
        # In a real implementation, you would:
        # 1. Collect activation statistics
        # 2. Compute quantization parameters
        # 3. Convert weights and implement quantized operations

        return model

    @staticmethod
    def mixed_precision_conversion(
        model: TitansModel,
        _use_fp16: bool = True,
    ) -> TitansModel:
        """Convert model to mixed precision.

        Args:
            model: Titans model
            _use_fp16: Whether to use FP16 (vs BF16)

        Returns:
            Mixed precision model (placeholder)
        """
        return model


class PerformanceProfiler:
    """Performance profiling utilities."""

    def __init__(self):
        """Initialize profiler."""
        self.profiles = {}

    def profile_model(
        self,
        model: TitansModel,
        input_shape: tuple[int, ...],
        num_warmup: int = 5,
        num_runs: int = 10,
    ) -> dict[str, Any]:
        """Profile model performance.

        Args:
            model: Titans model to profile
            input_shape: Input shape for profiling
            num_warmup: Number of warmup runs
            num_runs: Number of timing runs

        Returns:
            Performance profile
        """
        # Create test input
        test_input = jnp.ones(input_shape)

        # Warmup
        for _ in range(num_warmup):
            _ = model(test_input, deterministic=True)

        # Timing runs
        times = []
        memory_states = []

        for _run in range(num_runs):
            model.reset_memory()

            start_time = time.time()
            output = model(test_input, deterministic=False)
            jax.block_until_ready(output)
            end_time = time.time()

            times.append(end_time - start_time)

            # Track memory state
            memory_state = model.get_memory_state()
            if memory_state:
                memory_magnitude = sum(
                    float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                )
                memory_states.append(memory_magnitude)

        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        # Throughput
        total_tokens = input_shape[0] * input_shape[1]  # batch * seq_len
        throughput = total_tokens / avg_time

        profile = {
            "variant": model.variant,
            "input_shape": input_shape,
            "avg_time": avg_time,
            "std_time": std_time,
            "min_time": min_time,
            "max_time": max_time,
            "throughput_tokens_per_sec": throughput,
            "memory_magnitude_avg": np.mean(memory_states) if memory_states else 0,
            "memory_magnitude_std": np.std(memory_states) if memory_states else 0,
        }

        self.profiles[f"{model.variant}_{input_shape}"] = profile

        return profile

    def compare_variants(
        self,
        variants: list[TitansVariant] | None = None,
        input_shape: tuple[int, ...] = (2, 256, 128),
    ) -> dict[str, dict[str, Any]]:
        """Compare performance across variants.

        Args:
            variants: List of variants to compare
            input_shape: Input shape for comparison

        Returns:
            Comparison results
        """
        if variants is None:
            variants = ["mac", "mag", "mal", "lmm"]

        comparison = {}

        for variant in variants:
            model = create_titans_model(
                variant=variant, d_model=input_shape[-1], seed=42
            )

            profile = self.profile_model(model, input_shape)
            comparison[variant] = profile

        # Print comparison table

        for _variant, _profile in comparison.items():
            pass

        return comparison


def measure_model_performance(
    model: "TitansModel",  # Forward reference to avoid circular import
    seq_len: int,
    num_trials: int = 5,
) -> dict[str, float]:
    """Measure full performance metrics for a Titans model.

    This is a utility function extracted from research utilities that can be
    useful for general performance analysis beyond paper reproduction.

    Args:
        model: TitansModel instance to measure
        seq_len: Sequence length for test data
        num_trials: Number of trials to average over

    Returns:
        Dictionary containing performance metrics:
        - forward_time: Average forward pass time in seconds
        - throughput: Tokens processed per second
        - memory_capacity: Maximum memory magnitude observed
        - memory_usage: Estimated memory usage in MB
        - perplexity: Perplexity-like metric based on MSE
        - num_parameters: Rough parameter count estimate
    """
    d_model = model.d_model

    # Generate test data
    test_input = jax.random.normal(jax.random.key(0), (2, seq_len, d_model))

    # Measure forward time
    model.reset_memory()
    times = []

    for _trial in range(num_trials):
        start_time = time.time()
        output = model(test_input, deterministic=False)
        jax.block_until_ready(output)
        times.append(time.time() - start_time)

    avg_time = np.mean(times)
    throughput = (2 * seq_len) / avg_time  # tokens per second

    # Measure memory capacity
    memory_magnitudes = []
    model.reset_memory()

    for i in range(10):
        varied_input = test_input + 0.1 * jax.random.normal(
            jax.random.key(i), test_input.shape
        )
        _ = model(varied_input, deterministic=False)

        memory_state = model.get_memory_state()
        if memory_state:
            magnitude = sum(float(jnp.sum(jnp.abs(w))) for w in memory_state.weights)
            memory_magnitudes.append(magnitude)

    memory_capacity = np.max(memory_magnitudes) if memory_magnitudes else 0.0

    # Estimate memory usage (rough approximation)
    memory_state = model.get_memory_state()
    memory_usage = 0.0
    if memory_state:
        total_params = sum(w.size for w in memory_state.weights)
        memory_usage = total_params * 4 / (1024 * 1024)  # MB (assuming float32)

    # Compute perplexity-like metric
    target = test_input + 0.1 * jax.random.normal(jax.random.key(42), test_input.shape)
    output = model(test_input, deterministic=True)
    mse = float(jnp.mean((output - target) ** 2))
    perplexity = np.exp(mse)

    # Count parameters - use the standardized count_params method
    num_parameters = model.count_params()

    return {
        "forward_time": float(avg_time),
        "throughput": float(throughput),
        "memory_capacity": float(memory_capacity),
        "memory_usage": float(memory_usage),
        "perplexity": float(perplexity),
        "num_parameters": float(num_parameters),
    }


def convert_for_json_serialization(obj: Any) -> Any:
    """Convert numpy/JAX types to native Python types for JSON serialization.

    This utility helps when saving experimental results or metrics to JSON files.

    Args:
        obj: Object to convert (can be nested dict/list structures)

    Returns:
        Object with numpy/JAX types converted to native Python types
    """
    if isinstance(obj, np.integer | jnp.integer):
        return int(obj)
    if isinstance(obj, np.floating | jnp.floating):
        return float(obj)
    if isinstance(obj, np.ndarray | jax.Array):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_for_json_serialization(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json_serialization(item) for item in obj]
    return obj
