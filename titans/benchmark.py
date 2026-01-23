"""Benchmarking utilities for Titans models.

This module provides thorough benchmarking tools for comparing
different Titans variants, measuring performance metrics, and generating
comparative reports.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Import all variants and utilities
from .config import TitansConfig, get_config
from .core import TitansModel, TitansVariant, create_titans_model


class BenchmarkResult(NamedTuple):
    """Results from a single benchmark run."""

    forward_time: float
    memory_usage: float
    throughput: float
    accuracy: float | None = None
    loss: float | None = None


@dataclass
class PerformanceMetrics:
    """Full performance metrics."""

    # Timing metrics
    forward_times: list[float] = field(default_factory=list)
    backward_times: list[float] = field(default_factory=list)
    total_times: list[float] = field(default_factory=list)

    # Memory metrics
    peak_memory: float = 0.0
    memory_usage_history: list[float] = field(default_factory=list)

    # Throughput metrics
    tokens_per_second: list[float] = field(default_factory=list)
    samples_per_second: list[float] = field(default_factory=list)

    # Accuracy metrics
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)

    # Memory state analysis
    memory_state_magnitudes: list[float] = field(default_factory=list)
    memory_update_magnitudes: list[float] = field(default_factory=list)

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics as JSON-serializable floats."""
        return {
            "avg_forward_time": (
                float(jnp.mean(jnp.array(self.forward_times)))
                if self.forward_times
                else 0.0
            ),
            "avg_backward_time": (
                float(jnp.mean(jnp.array(self.backward_times)))
                if self.backward_times
                else 0.0
            ),
            "avg_total_time": (
                float(jnp.mean(jnp.array(self.total_times)))
                if self.total_times
                else 0.0
            ),
            "peak_memory_mb": float(self.peak_memory / (1024 * 1024)),
            "avg_tokens_per_sec": (
                float(jnp.mean(jnp.array(self.tokens_per_second)))
                if self.tokens_per_second
                else 0.0
            ),
            "avg_loss": (
                float(jnp.mean(jnp.array(self.losses))) if self.losses else 0.0
            ),
            "avg_accuracy": (
                float(jnp.mean(jnp.array(self.accuracies))) if self.accuracies else 0.0
            ),
            "avg_memory_magnitude": (
                float(jnp.mean(jnp.array(self.memory_state_magnitudes)))
                if self.memory_state_magnitudes
                else 0.0
            ),
        }


class TitansBenchmark:
    """Full benchmarking suite for Titans models."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize benchmark suite.

        Args:
            output_dir: Directory to save benchmark results
        """
        if output_dir is None:
            output_dir = Path("benchmark_results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Store results
        self.results: dict[str, PerformanceMetrics] = {}

    def benchmark_variant(
        self,
        variant: TitansVariant,
        config: TitansConfig,
        sequence_lengths: list[int] | None = None,
        batch_sizes: list[int] | None = None,
        num_trials: int = 5,
    ) -> dict[str, PerformanceMetrics]:
        """Benchmark a specific Titans variant across different configurations.

        Args:
            variant: Titans variant to benchmark
            config: Configuration to use
            sequence_lengths: List of sequence lengths to test
            batch_sizes: List of batch sizes to test
            num_trials: Number of trials per configuration

        Returns:
            Dictionary mapping configuration names to performance metrics
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
        if sequence_lengths is None:
            sequence_lengths = [128, 256, 512, 1024]

        variant_results = {}

        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                config_name = f"{variant}_{seq_len}_{batch_size}"

                # Update config for this test
                test_config = config
                test_config.training.batch_size = batch_size
                test_config.training.max_seq_length = seq_len

                # Run benchmark
                metrics = self._benchmark_single_config(
                    variant, test_config, seq_len, batch_size, num_trials
                )

                variant_results[config_name] = metrics

        return variant_results

    def _benchmark_single_config(
        self,
        variant: TitansVariant,
        config: TitansConfig,
        seq_len: int,
        batch_size: int,
        num_trials: int,
    ) -> PerformanceMetrics:
        """Benchmark a single configuration."""
        metrics = PerformanceMetrics()

        # Create model
        model = create_titans_model(
            variant=variant,
            d_model=config.architecture.d_model,
            num_heads=config.attention.num_heads,
            memory_depth=config.memory.memory_depth,
            chunk_size=config.architecture.chunk_size,
            window_size=config.attention.window_size,
            dropout_rate=config.attention.dropout_rate,
            seed=config.seed,
        )

        # Generate test data
        key = jax.random.key(config.seed)
        test_input = jax.random.normal(
            key, (batch_size, seq_len, config.architecture.d_model)
        )

        # Compile the model (first run for JIT compilation)
        _ = model(test_input, deterministic=True)

        # Run trials
        for trial in range(num_trials):
            # Reset memory for each trial
            model.reset_memory()

            # Forward pass timing
            start_time = time.time()
            output = model(test_input, deterministic=False)
            jax.block_until_ready(output)
            forward_time = time.time() - start_time

            metrics.forward_times.append(forward_time)

            # Calculate throughput
            total_tokens = batch_size * seq_len
            tokens_per_sec = total_tokens / forward_time
            metrics.tokens_per_second.append(tokens_per_sec)

            samples_per_sec = batch_size / forward_time
            metrics.samples_per_second.append(samples_per_sec)

            # Memory analysis
            memory_state = model.get_memory_state()
            if memory_state:
                memory_magnitude = sum(
                    float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                )
                metrics.memory_state_magnitudes.append(memory_magnitude)

            # Backward pass timing (for training)
            if trial == 0:  # Only do this once as it's expensive

                def loss_fn(_model: TitansModel) -> jax.Array:
                    pred = _model(test_input, deterministic=True)
                    return jnp.mean(pred**2)

                start_time = time.time()
                grad_fn = nnx.grad(loss_fn)
                grads = grad_fn(model)
                jax.block_until_ready(grads)
                backward_time = time.time() - start_time

                metrics.backward_times.append(backward_time)
                metrics.total_times.append(forward_time + backward_time)

        return metrics

    def compare_variants(
        self,
        variants: list[TitansVariant] | None = None,
        seq_len: int = 512,
        batch_size: int = 4,
        num_trials: int = 10,
    ) -> dict[TitansVariant, PerformanceMetrics]:
        """Compare performance across different Titans variants.

        Args:
            variants: List of variants to compare
            seq_len: Sequence length for comparison
            batch_size: Batch size for comparison
            num_trials: Number of trials per variant

        Returns:
            Dictionary mapping variants to their performance metrics
        """
        if variants is None:
            variants = ["mac", "mag", "mal", "lmm"]

        comparison_results: dict[TitansVariant, PerformanceMetrics] = {}

        for variant in variants:
            # Get appropriate config for variant
            config = get_config("medium")
            config.variant = variant
            # Call __post_init__ to apply variant-specific overrides
            config.__post_init__()

            metrics = self._benchmark_single_config(
                variant, config, seq_len, batch_size, num_trials
            )

            comparison_results[variant] = metrics

            # Print summary

        return comparison_results

    def scaling_analysis(
        self,
        variant: TitansVariant = "mac",
        sequence_lengths: list[int] | None = None,
        d_model: int = 128,
        num_trials: int = 3,
    ) -> dict[int, PerformanceMetrics]:
        """Analyze how performance scales with sequence length.

        Args:
            variant: Titans variant to analyze
            sequence_lengths: List of sequence lengths to test
            d_model: Model dimension
            num_trials: Number of trials per length

        Returns:
            Dictionary mapping sequence lengths to performance metrics
        """
        if sequence_lengths is None:
            sequence_lengths = [128, 256, 512, 1024, 2048]

        scaling_results: dict[int, PerformanceMetrics] = {}
        config = get_config("medium")
        config.variant = variant
        config.architecture.d_model = d_model
        # Call __post_init__ to apply variant-specific overrides
        config.__post_init__()

        for seq_len in sequence_lengths:
            metrics = self._benchmark_single_config(
                variant, config, seq_len, batch_size=2, num_trials=num_trials
            )

            scaling_results[seq_len] = metrics

        return scaling_results

    def memory_capacity_analysis(
        self,
        variant: TitansVariant = "mac",
        num_sequences: int = 10,
        seq_len: int = 256,
        memory_depths: list[int] | None = None,
    ) -> dict[int, list[float]]:
        """Analyze memory capacity with different memory depths.

        Args:
            variant: Titans variant to analyze
            num_sequences: Number of sequences to process
            seq_len: Length of each sequence
            memory_depths: List of memory depths to test

        Returns:
            Dictionary mapping memory depths to memory magnitude histories
        """
        if memory_depths is None:
            memory_depths = [1, 2, 3, 4, 5]

        capacity_results: dict[int, list[float]] = {}

        for depth in memory_depths:
            config = get_config("medium")
            config.variant = variant
            config.memory.memory_depth = depth
            # Call __post_init__ to apply variant-specific overrides
            config.__post_init__()

            model = create_titans_model(
                variant=variant,
                d_model=config.architecture.d_model,
                memory_depth=depth,
                seed=42,
            )

            memory_history = []

            for i in range(num_sequences):
                # Generate different input each time
                key = jax.random.key(i)
                x = jax.random.normal(key, (1, seq_len, config.architecture.d_model))

                # Process sequence
                _ = model(x, deterministic=False)

                # Track memory magnitude
                memory_state = model.get_memory_state()
                if memory_state:
                    magnitude = sum(
                        float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                    )
                    memory_history.append(magnitude)
                else:
                    memory_history.append(0.0)

            capacity_results[depth] = memory_history

        return capacity_results

    def save_results(self, results: dict[Any, Any], filename: str) -> None:
        """Save benchmark results to JSON file.

        Args:
            results: Results dictionary to save
            filename: Name of the output file
        """
        output_path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        def convert_arrays(obj: jax.Array | dict[str, Any] | list[Any] | Any) -> Any:
            if isinstance(obj, jax.Array):
                return obj.tolist()
            if isinstance(obj, dict):
                # Convert keys to strings for JSON serialization
                return {str(k): convert_arrays(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_arrays(item) for item in obj]
            if hasattr(obj, "get_summary"):
                # Convert summary to dict first, then recursively convert arrays
                summary = obj.get_summary()
                return convert_arrays(summary)
            return obj

        serializable_results = convert_arrays(results)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def plot_comparison(
        self,
        comparison_results: dict[TitansVariant, PerformanceMetrics],
        save_path: Path | None = None,
    ) -> None:
        """Plot comparison results across variants.

        Args:
            comparison_results: Results from compare_variants()
            save_path: Optional path to save the plot
        """
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Titans Variants Performance Comparison", fontsize=16)

        variants = list(comparison_results.keys())

        # Forward time comparison
        forward_times = [
            jnp.mean(jnp.array(comparison_results[v].forward_times)) for v in variants
        ]
        axes[0, 0].bar(variants, forward_times)
        axes[0, 0].set_title("Average Forward Time")
        axes[0, 0].set_ylabel("Time (seconds)")

        # Throughput comparison
        throughputs = [
            jnp.mean(jnp.array(comparison_results[v].tokens_per_second))
            for v in variants
        ]
        axes[0, 1].bar(variants, throughputs)
        axes[0, 1].set_title("Throughput")
        axes[0, 1].set_ylabel("Tokens/Second")

        # Memory magnitude comparison
        memory_mags = [
            (
                jnp.mean(jnp.array(comparison_results[v].memory_state_magnitudes))
                if comparison_results[v].memory_state_magnitudes
                else 0
            )
            for v in variants
        ]
        axes[1, 0].bar(variants, memory_mags)
        axes[1, 0].set_title("Average Memory State Magnitude")
        axes[1, 0].set_ylabel("Magnitude")

        # Forward time distribution
        for _i, variant in enumerate(variants):
            times = comparison_results[variant].forward_times
            axes[1, 1].hist(times, alpha=0.7, label=variant.upper(), bins=10)
        axes[1, 1].set_title("Forward Time Distribution")
        axes[1, 1].set_xlabel("Time (seconds)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def plot_scaling(
        self,
        scaling_results: dict[int, PerformanceMetrics],
        save_path: Path | None = None,
    ) -> None:
        """Plot scaling analysis results.

        Args:
            scaling_results: Results from scaling_analysis()
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Scaling Analysis", fontsize=16)

        seq_lengths = list(scaling_results.keys())

        # Time scaling
        avg_times = [
            jnp.mean(jnp.array(scaling_results[length].forward_times))
            for length in seq_lengths
        ]
        axes[0].plot(seq_lengths, avg_times, "o-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Sequence Length")
        axes[0].set_ylabel("Forward Time (seconds)")
        axes[0].set_title("Time vs Sequence Length")
        axes[0].grid(True, alpha=0.3)

        # Throughput scaling
        avg_throughputs = [
            jnp.mean(jnp.array(scaling_results[length].tokens_per_second))
            for length in seq_lengths
        ]
        axes[1].plot(seq_lengths, avg_throughputs, "o-", linewidth=2, markersize=8)
        axes[1].set_xlabel("Sequence Length")
        axes[1].set_ylabel("Throughput (tokens/sec)")
        axes[1].set_title("Throughput vs Sequence Length")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()


def run_full_benchmark() -> None:
    """Run a full benchmark of all Titans variants.

    This is the entry point for the benchmark CLI command.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running full Titans benchmark...")
    benchmark = TitansBenchmark()

    # Compare all variants
    comparison_results = benchmark.compare_variants()
    benchmark.save_results(comparison_results, "variant_comparison.json")

    # Run scaling analysis
    scaling_results = benchmark.scaling_analysis()
    benchmark.save_results(scaling_results, "scaling_analysis.json")

    # Run memory capacity analysis
    capacity_results = benchmark.memory_capacity_analysis()
    benchmark.save_results(capacity_results, "memory_capacity.json")

    logger.info("Benchmark complete. Results saved to benchmark_results/")
