"""Scaling Analysis Experiment for Titans Paper Reproduction.

This experiment analyzes how Titans performance scales with:
1. Memory depth (number of memory layers)
2. Sequence length (context window size)
3. Model size (d_model parameter)
4. Memory capacity (number of persistent tokens)

Following the methodology from the Titans paper.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from titans.core import TitansModel, TitansVariant

from .paper_reproduction_core import (
    BaseExperiment,
    ExperimentConfig,
    ExperimentResult,
    MetadataValue,
    MetricValue,
)

# Type alias for task metadata (flexible dict to handle various return structures)
TaskMetadata = dict[str, object]

# Type alias for detailed results (flexible dict to handle various metric types)
DetailedResult = dict[str, object]


@dataclass
class ScalingTask:
    """Configuration for a scaling analysis task."""

    scaling_dimension: (
        str  # "memory_depth", "sequence_length", "model_size", "memory_capacity"
    )
    parameter_value: int
    task_type: str  # "memorization", "recall", "computation"


class ScalingAnalysisExperiment(BaseExperiment):
    """Scaling Analysis experiment implementation.

    This experiment systematically varies different model parameters to understand:
    - How memory depth affects learning and recall capabilities
    - How sequence length impacts performance and efficiency
    - How model size scales with task complexity
    - How memory capacity affects information storage
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize the scaling analysis experiment."""
        super().__init__(output_dir)

        # Define scaling parameters to test
        self.scaling_configs = {
            "memory_depth": [1, 2, 3, 4, 5, 6],
            "sequence_length": [128, 256, 512, 1024, 2048, 4096, 8192],
            "model_size": [64, 128, 256, 512, 768, 1024],
            "memory_capacity": [16, 32, 64, 128, 256, 512],
        }

        # Task types for evaluation
        self.task_types = ["memorization", "recall", "computation"]

    def _create_memorization_task(
        self, seq_len: int, d_model: int, key: jax.Array, difficulty: str = "medium"
    ) -> tuple[jax.Array, jax.Array, TaskMetadata]:
        """Create a memorization task - learn to reproduce patterns."""
        keys = jax.random.split(key, 3)

        if difficulty == "easy":
            # Simple repeating patterns
            pattern_len = min(8, seq_len // 4)
            base_pattern = jax.random.normal(keys[0], (pattern_len, d_model))
            repeats = seq_len // pattern_len
            sequence = jnp.tile(base_pattern, (repeats + 1, 1))[:seq_len, :]
        elif difficulty == "medium":
            # Structured but varied patterns
            num_patterns = 4
            pattern_len = seq_len // num_patterns
            patterns = []
            for i in range(num_patterns):
                pattern = jax.random.normal(keys[1], (pattern_len, d_model)) * (i + 1)
                patterns.append(pattern)
            sequence = jnp.concatenate(patterns, axis=0)
            if sequence.shape[0] < seq_len:
                padding = jax.random.normal(
                    keys[2], (seq_len - sequence.shape[0], d_model)
                )
                sequence = jnp.concatenate([sequence, padding], axis=0)
            sequence = sequence[:seq_len, :]
        else:  # hard
            # Complex hierarchical patterns
            sequence = jax.random.normal(keys[0], (seq_len, d_model))
            # Add hierarchical structure
            for scale in [2, 4, 8, 16]:
                if scale < seq_len:
                    period = seq_len // scale
                    for i in range(scale):
                        start_idx = i * period
                        end_idx = min((i + 1) * period, seq_len)
                        sequence = sequence.at[start_idx:end_idx].add(
                            jax.random.normal(keys[1], (d_model,)) * 0.1
                        )

        # Add batch dimension
        sequence = sequence[None, :, :]
        target = sequence  # For autoregressive learning

        metadata: TaskMetadata = {
            "task_type": "memorization",
            "difficulty": difficulty,
            "sequence_length": seq_len,
        }

        return sequence, target, metadata

    def _create_recall_task(
        self, seq_len: int, d_model: int, key: jax.Array, memory_span: int = 10
    ) -> tuple[jax.Array, jax.Array, TaskMetadata]:
        """Create a recall task - remember information from earlier in sequence."""
        keys = jax.random.split(key, 4)

        # Create base sequence
        sequence = jax.random.normal(keys[0], (seq_len, d_model)) * 0.5

        # Insert memorable items at regular intervals
        num_items = seq_len // memory_span
        memorable_items = []
        positions = []

        for i in range(num_items):
            pos = i * memory_span + jax.random.randint(keys[1], (), 0, memory_span // 2)
            if pos < seq_len - 1:
                # Create distinctive memorable item
                item = jax.random.normal(keys[2], (d_model,)) * 2.0 + 3.0
                sequence = sequence.at[pos].set(item)
                memorable_items.append(item)
                positions.append(pos)

        # Target asks to reproduce memorable items at specific query positions
        target = sequence.copy()

        # Add batch dimension
        sequence = sequence[None, :, :]
        target = target[None, :, :]

        metadata: TaskMetadata = {
            "task_type": "recall",
            "memory_span": memory_span,
            "num_memorable_items": len(memorable_items),
            "memorable_positions": positions,
        }

        return sequence, target, metadata

    def _create_computation_task(
        self, seq_len: int, d_model: int, key: jax.Array, operation: str = "cumsum"
    ) -> tuple[jax.Array, jax.Array, TaskMetadata]:
        """Create a computation task - perform operations over sequence."""
        keys = jax.random.split(key, 2)

        # Create input sequence
        sequence = jax.random.normal(keys[0], (seq_len, d_model)) * 0.5

        if operation == "cumsum":
            # Cumulative sum along sequence
            target = jnp.cumsum(sequence, axis=0)
        elif operation == "running_mean":
            # Running average
            target = jnp.cumsum(sequence, axis=0) / jnp.arange(1, seq_len + 1)[:, None]
        elif operation == "diff":
            # Differences between consecutive elements
            target = jnp.diff(sequence, axis=0, prepend=sequence[:1])
        else:
            # Default: simple transformation
            target = sequence * 2.0 + 1.0

        # Add batch dimension
        sequence = sequence[None, :, :]
        target = target[None, :, :]

        metadata: TaskMetadata = {
            "task_type": "computation",
            "operation": operation,
        }

        return sequence, target, metadata

    def _evaluate_performance(
        self,
        model: TitansModel,
        sequence: jax.Array,
        target: jax.Array,
        metadata: TaskMetadata,
    ) -> dict[str, float | int | str | bool | jax.Array]:
        """Evaluate model performance on a task."""
        # Reset memory before evaluation
        model.reset_memory()

        # Time the forward pass
        start_time = time.time()
        output = model(sequence, deterministic=False)
        forward_time = time.time() - start_time

        # Compute basic metrics
        mse = float(jnp.mean((output - target) ** 2))
        mae = float(jnp.mean(jnp.abs(output - target)))

        # Compute relative error
        target_norm = float(jnp.linalg.norm(target))
        if target_norm > 0:
            relative_error = float(jnp.linalg.norm(output - target) / target_norm)
        else:
            relative_error = float(jnp.linalg.norm(output - target))

        # Task-specific metrics
        task_metrics = {}
        if metadata["task_type"] == "memorization":
            # Pattern fidelity
            pattern_similarity = float(
                jnp.corrcoef(output.flatten(), target.flatten())[0, 1]
            )
            task_metrics["pattern_similarity"] = pattern_similarity

        elif metadata["task_type"] == "recall":
            # Recall accuracy at specific positions
            if "memorable_positions" in metadata:
                positions = metadata["memorable_positions"]
                if positions:
                    recall_errors = []
                    for pos in positions:
                        if pos < output.shape[1]:
                            pos_error = float(
                                jnp.mean((output[0, pos] - target[0, pos]) ** 2)
                            )
                            recall_errors.append(pos_error)
                    if recall_errors:
                        task_metrics["avg_recall_error"] = np.mean(recall_errors)
                        task_metrics["recall_accuracy"] = float(
                            np.mean([e < 0.5 for e in recall_errors])
                        )

        elif metadata["task_type"] == "computation":
            # Computational accuracy
            computation_error = float(jnp.mean(jnp.abs(output - target)))
            task_metrics["computation_error"] = computation_error
            task_metrics["computation_accuracy"] = float(computation_error < 0.1)

        # Memory analysis
        memory_norm = model.get_memory_norm()
        memory_state = model.get_memory_state()

        # Throughput calculation
        seq_len = sequence.shape[1]
        throughput = seq_len / forward_time if forward_time > 0 else 0

        return {
            "mse": mse,
            "mae": mae,
            "relative_error": relative_error,
            "forward_time": forward_time,
            "throughput": throughput,
            "memory_norm": memory_norm,
            "memory_layers": len(memory_state.weights) if memory_state else 0,
            "parameter_count": model.count_params(),
            **task_metrics,
        }

    def _run_scaling_dimension(
        self,
        dimension: str,
        variant: TitansVariant,
        config: ExperimentConfig,
        trial_id: int,
    ) -> list[DetailedResult]:
        """Run scaling analysis for a specific dimension."""
        results = []
        parameter_values = self.scaling_configs[dimension]

        key = jax.random.key(config.seed + trial_id)
        keys = jax.random.split(key, len(parameter_values) * len(self.task_types))
        key_idx = 0

        for param_value in parameter_values:
            for task_type in self.task_types:
                try:
                    # Create model with scaled parameter
                    model_kwargs: dict[str, int] = {}
                    # Initialize defaults for d_model and seq_len
                    d_model: int = config.d_model
                    seq_len: int = 512
                    if dimension == "memory_depth":
                        model_kwargs["memory_depth"] = param_value
                        d_model = config.d_model
                        seq_len = 512  # Fixed for memory depth scaling
                    elif dimension == "sequence_length":
                        seq_len = param_value
                        d_model = config.d_model
                    elif dimension == "model_size":
                        d_model = param_value
                        seq_len = 512  # Fixed for model size scaling
                    elif dimension == "memory_capacity":
                        model_kwargs["num_persistent_tokens"] = param_value
                        d_model = config.d_model
                        seq_len = 512  # Fixed for memory capacity scaling

                    # Create model
                    model = self.create_model(
                        variant, config, d_model=d_model, **model_kwargs
                    )

                    # Create task
                    if task_type == "memorization":
                        sequence, target, metadata = self._create_memorization_task(
                            seq_len, d_model, keys[key_idx]
                        )
                    elif task_type == "recall":
                        sequence, target, metadata = self._create_recall_task(
                            seq_len, d_model, keys[key_idx]
                        )
                    else:  # computation
                        sequence, target, metadata = self._create_computation_task(
                            seq_len, d_model, keys[key_idx]
                        )

                    key_idx += 1

                    # Evaluate performance
                    metrics = self._evaluate_performance(
                        model, sequence, target, metadata
                    )

                    # Add scaling information
                    metrics.update(
                        {
                            "scaling_dimension": dimension,
                            "parameter_value": param_value,
                            "task_type": task_type,
                            "variant": variant,
                            "trial_id": trial_id,
                            "d_model": d_model,
                            "sequence_length": seq_len,
                        }
                    )

                    results.append(metrics)

                except Exception as e:
                    self.logger.warning(
                        f"Failed scaling {dimension}={param_value}, task={task_type}, variant={variant}, trial={trial_id}: {e!s}"
                    )
                    continue

        return results

    def run_single_trial(
        self, model: TitansModel, config: ExperimentConfig, trial_id: int
    ) -> ExperimentResult:
        """Run a single trial of scaling analysis."""
        # Note: model parameter is not used directly since we create models per scaling dimension

        all_results: list[DetailedResult] = []

        # Get variant with proper type
        variant: TitansVariant = model.variant  # type: ignore[assignment]

        # Run scaling analysis for each dimension
        for dimension in self.scaling_configs:
            dimension_results = self._run_scaling_dimension(
                dimension, variant, config, trial_id
            )
            all_results.extend(dimension_results)

        # Aggregate results
        aggregated_metrics: dict[str, MetricValue] = {}
        metadata: dict[str, MetadataValue] = {
            "scaling_dimensions": list(self.scaling_configs.keys()),
            "task_types": self.task_types,
        }

        if all_results:
            # Group by scaling dimension
            by_dimension: dict[str, list[DetailedResult]] = {}
            for result in all_results:
                dim = str(result["scaling_dimension"])
                if dim not in by_dimension:
                    by_dimension[dim] = []
                by_dimension[dim].append(result)

            # Compute scaling statistics
            for dim, dim_results in by_dimension.items():
                param_values = [r["parameter_value"] for r in dim_results]
                mse_values = [r["mse"] for r in dim_results]
                throughput_values = [r["throughput"] for r in dim_results]

                # Compute scaling trends
                if len(param_values) > 1:
                    # Simple linear fit to log-log plot
                    log_params = np.log(np.array(param_values) + 1e-8)
                    log_mse = np.log(np.array(mse_values) + 1e-8)
                    log_throughput = np.log(np.array(throughput_values) + 1e-8)

                    mse_slope = float(np.polyfit(log_params, log_mse, 1)[0])
                    throughput_slope = float(
                        np.polyfit(log_params, log_throughput, 1)[0]
                    )

                    aggregated_metrics[f"{dim}_mse_scaling_slope"] = mse_slope
                    aggregated_metrics[f"{dim}_throughput_scaling_slope"] = (
                        throughput_slope
                    )
                    aggregated_metrics[f"{dim}_num_points"] = len(param_values)

            # Overall statistics
            aggregated_metrics["total_evaluations"] = len(all_results)
            aggregated_metrics["avg_mse"] = float(
                np.mean([float(r["mse"]) for r in all_results])  # type: ignore[arg-type]
            )
            aggregated_metrics["avg_throughput"] = float(
                np.mean([float(r["throughput"]) for r in all_results])  # type: ignore[arg-type]
            )
            aggregated_metrics["avg_memory_norm"] = float(
                np.mean([float(r["memory_norm"]) for r in all_results])  # type: ignore[arg-type]
            )

            # Store detailed results as JSON
            metadata["detailed_results_json"] = json.dumps(all_results)

        else:
            aggregated_metrics["total_evaluations"] = 0
            aggregated_metrics["error"] = "all evaluations failed"

        return ExperimentResult(
            config=config,
            variant=None,  # Multiple variants tested
            trial_id=trial_id,
            metrics=aggregated_metrics,
            metadata=metadata,
            execution_time=0.0,  # Will be set by base class
        )

    def run(self) -> list[ExperimentResult]:
        """Run the complete scaling analysis experiment."""
        config = ExperimentConfig(
            name="scaling_analysis",
            description="Analyze performance scaling with different parameters",
            variants=["mac", "mag", "mal", "lmm"],
            d_model=256,  # Base model size
            num_trials=3,  # Fewer trials due to extensive coverage
            seed=42,
            timeout_seconds=600.0,  # Longer timeout for scaling analysis
        )

        # Run experiment
        results = self.run_experiment(config, self.run_single_trial)

        # Save results
        self.save_results(results, "scaling_analysis_results.json")

        # Analyze results
        analysis = self.analyze_results(results)

        # Save analysis
        with open(self.output_dir / "scaling_analysis_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Create plots
        self.plot_results(results, self.output_dir / "scaling_analysis_plots.png")

        # Create specialized scaling plots
        self._create_scaling_plots(results)

        self.logger.info("Scaling Analysis experiment completed")
        return results

    def _create_scaling_plots(self, results: list[ExperimentResult]) -> None:
        """Create specialized scaling analysis plots."""
        successful_results = [r for r in results if r.error is None]
        if not successful_results:
            return

        # Extract detailed results
        all_detailed_results: list[DetailedResult] = []
        for result in successful_results:
            if "detailed_results" in result.metadata:
                detailed = result.metadata["detailed_results"]
                if isinstance(detailed, list):
                    all_detailed_results.extend(detailed)  # type: ignore[arg-type]

        if not all_detailed_results:
            self.logger.warning("No detailed results found for scaling plots")
            return

        # Create thorough scaling plots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle("Scaling Analysis - Complete View", fontsize=16)

        # Plot scaling for each dimension
        dimensions = [
            "memory_depth",
            "sequence_length",
            "model_size",
            "memory_capacity",
        ]

        for i, dimension in enumerate(dimensions):
            if i < len(axes[0]):
                # Performance scaling
                self._plot_dimension_scaling(
                    all_detailed_results, dimension, "mse", axes[0, i]
                )
                axes[0, i].set_title(f"MSE vs {dimension.replace('_', ' ').title()}")

                # Throughput scaling
                self._plot_dimension_scaling(
                    all_detailed_results, dimension, "throughput", axes[1, i]
                )
                axes[1, i].set_title(
                    f"Throughput vs {dimension.replace('_', ' ').title()}"
                )

                # Memory scaling
                self._plot_dimension_scaling(
                    all_detailed_results, dimension, "memory_norm", axes[2, i]
                )
                axes[2, i].set_title(
                    f"Memory Norm vs {dimension.replace('_', ' ').title()}"
                )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "scaling_complete_plots.png",
            dpi=300,
            bbox_inches="tight",
        )
        if plt.isinteractive():
            plt.show()
        plt.close()

        # Create variant comparison plots
        self._create_variant_scaling_comparison(all_detailed_results)

    def _plot_dimension_scaling(
        self,
        detailed_results: list[DetailedResult],
        dimension: str,
        metric: str,
        ax: Any,
    ) -> None:
        """Plot scaling for a specific dimension and metric."""
        # Filter results for this dimension
        dim_results = [
            r for r in detailed_results if r["scaling_dimension"] == dimension
        ]

        if not dim_results:
            ax.text(0.5, 0.5, f"No data for {dimension}", ha="center", va="center")
            return

        # Group by variant and task type
        variant_data = {}
        for result in dim_results:
            variant = result["variant"]
            task_type = result["task_type"]
            key = f"{variant}_{task_type}"

            if key not in variant_data:
                variant_data[key] = {"param_values": [], "metric_values": []}

            variant_data[key]["param_values"].append(result["parameter_value"])
            variant_data[key]["metric_values"].append(result[metric])

        # Plot each variant/task combination
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i / len(variant_data)) for i in range(len(variant_data))]

        for (key, data), color in zip(variant_data.items(), colors):
            if len(data["param_values"]) > 1:
                # Sort by parameter value
                sorted_indices = np.argsort(data["param_values"])
                x_vals = np.array(data["param_values"])[sorted_indices]
                y_vals = np.array(data["metric_values"])[sorted_indices]

                ax.plot(x_vals, y_vals, marker="o", label=key, color=color, alpha=0.7)

        ax.set_xlabel(dimension.replace("_", " ").title())
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Only show legend if not too many items
        if len(variant_data) <= 8:
            ax.legend(fontsize="small")

    def _create_variant_scaling_comparison(
        self, detailed_results: list[DetailedResult]
    ) -> None:
        """Create plots comparing variants across scaling dimensions."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Variant Comparison Across Scaling Dimensions", fontsize=16)

        dimensions = [
            "memory_depth",
            "sequence_length",
            "model_size",
            "memory_capacity",
        ]

        for i, dimension in enumerate(dimensions[:4]):
            row = i // 2
            col = i % 2

            # Get results for this dimension
            dim_results = [
                r for r in detailed_results if r["scaling_dimension"] == dimension
            ]

            if not dim_results:
                continue

            # Group by variant
            variant_performance = {}
            for result in dim_results:
                variant = result["variant"]
                if variant not in variant_performance:
                    variant_performance[variant] = []
                variant_performance[variant].append(result["mse"])

            # Create box plot
            if variant_performance:
                data = [variant_performance[variant] for variant in variant_performance]
                labels = list(variant_performance.keys())

                bp = axes[row, col].boxplot(data, labels=labels, patch_artist=True)
                axes[row, col].set_title(
                    f"Performance on {dimension.replace('_', ' ').title()}"
                )
                axes[row, col].set_ylabel("MSE")

                # Color the boxes
                colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]
                for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                    patch.set_facecolor(color)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "variant_scaling_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        if plt.isinteractive():
            plt.show()
        plt.close()
