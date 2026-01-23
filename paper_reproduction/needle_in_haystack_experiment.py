"""Needle-in-Haystack Experiment for Titans Paper Reproduction.

This experiment tests the model's ability to recall specific information
placed at different positions within long sequences, which is a key
capability for memory-augmented architectures.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from titans.core import TitansModel

from .paper_reproduction_core import (
    BaseExperiment,
    ExperimentConfig,
    ExperimentResult,
    MetadataValue,
    MetricValue,
)

# Type alias for task metadata (flexible dict)
TaskMetadata = dict[str, object]

# Type alias for detailed results (flexible dict)
DetailedResult = dict[str, object]


@dataclass
class NeedleTask:
    """Configuration for a single needle-in-haystack task."""

    sequence_length: int
    needle_position: (
        str  # "beginning", "middle", "end", or specific fraction like "0.25"
    )
    needle_type: str  # "value", "pattern", "sequence"
    difficulty: str  # "easy", "medium", "hard"


class NeedleInHaystackExperiment(BaseExperiment):
    """Needle-in-Haystack experiment implementation.

    This experiment follows the methodology from the Titans paper where:
    1. A "needle" (specific pattern/value) is inserted at various positions
    2. The model must retrieve/recall this information later
    3. Performance is measured across different sequence lengths and positions
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize the experiment."""
        super().__init__(output_dir)
        self.needle_tasks = self._generate_needle_tasks()

    def _generate_needle_tasks(self) -> list[NeedleTask]:
        """Generate complete set of needle tasks."""
        tasks = []

        # Sequence lengths to test
        sequence_lengths = [512, 1024, 2048, 4096, 8192]

        # Positions to test
        positions = ["beginning", "middle", "end", "0.1", "0.25", "0.75", "0.9"]

        # Different needle types
        needle_types = ["value", "pattern", "sequence"]

        # Difficulty levels
        difficulties = ["easy", "medium", "hard"]

        for seq_len in sequence_lengths:
            for position in positions:
                for needle_type in needle_types:
                    for difficulty in difficulties:
                        tasks.append(
                            NeedleTask(
                                sequence_length=seq_len,
                                needle_position=position,
                                needle_type=needle_type,
                                difficulty=difficulty,
                            )
                        )

        return tasks

    def _create_needle_sequence(
        self, task: NeedleTask, d_model: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, TaskMetadata]:
        """Create a sequence with embedded needle.

        Args:
            task: Needle task configuration
            d_model: Model dimension
            key: JAX random key

        Returns:
            Tuple of (sequence, target, metadata)
        """
        keys = jax.random.split(key, 5)

        # Create base haystack sequence
        haystack = self._create_haystack(task.sequence_length, d_model, keys[0])

        # Create needle based on type and difficulty
        needle, _needle_target = self._create_needle(
            task.needle_type, task.difficulty, d_model, keys[1]
        )

        # Determine needle position
        needle_pos = self._get_needle_position(
            task.needle_position, task.sequence_length
        )

        # Insert needle into haystack
        sequence = haystack.at[:, needle_pos : needle_pos + needle.shape[1], :].set(
            needle
        )

        # Create target sequence (same as input for autoregressive task)
        target = sequence

        task_metadata: TaskMetadata = {
            "needle_position": needle_pos,
            "needle_length": needle.shape[1],
            "sequence_length": task.sequence_length,
            "needle_type": task.needle_type,
            "difficulty": task.difficulty,
        }

        return sequence, target, task_metadata

    def _create_haystack(self, seq_len: int, d_model: int, key: jax.Array) -> jax.Array:
        """Create background haystack sequence."""
        # Create diverse background patterns to make needle detection challenging
        keys = jax.random.split(key, 3)

        # Mix of different patterns
        base_noise = jax.random.normal(keys[0], (1, seq_len, d_model)) * 0.5

        # Add some structured patterns to increase difficulty
        sin_pattern = jnp.sin(jnp.arange(seq_len)[:, None] * 0.1) * 0.3
        sin_pattern = jnp.broadcast_to(sin_pattern, (1, seq_len, d_model))

        # Random structured blocks - ensure we have enough tokens after repeat
        block_size = max(1, seq_len // 10)
        num_repeats = (seq_len + block_size - 1) // block_size  # Ceiling division
        block_pattern = jax.random.uniform(keys[1], (1, block_size, d_model))
        block_pattern = jnp.repeat(block_pattern, num_repeats, axis=1)[:, :seq_len, :]

        # Combine patterns
        return base_noise + 0.2 * sin_pattern + 0.1 * block_pattern

    def _create_needle(
        self, needle_type: str, difficulty: str, d_model: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Create needle pattern and its target."""
        keys = jax.random.split(key, 3)

        if needle_type == "value":
            return self._create_value_needle(difficulty, d_model, keys[0])
        if needle_type == "pattern":
            return self._create_pattern_needle(difficulty, d_model, keys[1])
        if needle_type == "sequence":
            return self._create_sequence_needle(difficulty, d_model, keys[2])
        msg = f"Unknown needle type: {needle_type}"
        raise ValueError(msg)

    def _create_value_needle(
        self, difficulty: str, d_model: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Create a distinctive value needle."""
        if difficulty == "easy":
            # Very distinctive value
            needle = jnp.ones((1, 1, d_model)) * 5.0
        elif difficulty == "medium":
            # Moderately distinctive
            needle = jnp.ones((1, 1, d_model)) * 2.0
        else:  # hard
            # Subtle but learnable pattern
            needle = jax.random.normal(key, (1, 1, d_model)) * 0.1 + 1.0

        target = needle  # For autoregressive modeling
        return needle, target

    def _create_pattern_needle(
        self, difficulty: str, d_model: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Create a pattern-based needle."""
        keys = jax.random.split(key, 2)

        if difficulty == "easy":
            # Simple alternating pattern - shape: (1, pattern_length, d_model)
            pattern_length = 4
            # Create pattern values and broadcast to d_model
            pattern_values = jnp.array([1.0, -1.0, 1.0, -1.0])[:, None]  # (4, 1)
            needle = jnp.broadcast_to(pattern_values, (pattern_length, d_model))
            needle = needle[None, :, :]  # (1, 4, d_model)
        elif difficulty == "medium":
            # More complex but regular pattern - shape: (1, pattern_length, d_model)
            pattern_length = 8
            t = jnp.arange(pattern_length)
            pattern_values = jnp.sin(t * 2 * jnp.pi / pattern_length)[:, None]  # (8, 1)
            needle = jnp.broadcast_to(pattern_values, (pattern_length, d_model))
            needle = needle[None, :, :]  # (1, 8, d_model)
        else:  # hard
            # Learned pattern with noise - shape: (1, pattern_length, d_model)
            pattern_length = 6
            base_pattern = jax.random.normal(keys[0], (pattern_length, d_model))
            noise = jax.random.normal(keys[1], (pattern_length, d_model)) * 0.1
            needle = (base_pattern + noise)[None, :, :]

        target = needle  # For autoregressive modeling
        return needle, target

    def _create_sequence_needle(
        self, difficulty: str, d_model: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Create a sequence-based needle."""
        keys = jax.random.split(key, 2)

        if difficulty == "easy":
            # Simple increasing sequence
            seq_len = 3
            sequence = jnp.arange(seq_len)[:, None] * jnp.ones(d_model)[None, :]
            needle = sequence[None, :, :]
        elif difficulty == "medium":
            # Fibonacci-like sequence
            seq_len = 5
            fib = jnp.array([1, 1, 2, 3, 5])
            sequence = fib[:, None] * jnp.ones(d_model)[None, :]
            needle = sequence[None, :, :]
        else:  # hard
            # Complex learned sequence
            seq_len = 8
            needle = jax.random.normal(keys[0], (1, seq_len, d_model))
            # Add some structure
            needle = needle + jnp.cumsum(needle, axis=1) * 0.1

        target = needle  # For autoregressive modeling
        return needle, target

    def _get_needle_position(self, position_spec: str, seq_len: int) -> int:
        """Convert position specification to actual index."""
        if position_spec == "beginning":
            return seq_len // 10  # 10% into sequence
        if position_spec == "middle":
            return seq_len // 2
        if position_spec == "end":
            return seq_len - seq_len // 10  # 90% into sequence
        # Assume it's a fraction like "0.25"
        fraction = float(position_spec)
        return int(seq_len * fraction)

    def _evaluate_needle_recall(
        self,
        model: TitansModel,
        sequence: jax.Array,
        target: jax.Array,
        metadata: TaskMetadata,
    ) -> dict[str, float | int | str | bool | jax.Array]:
        """Evaluate how well the model recalls the needle."""
        # Reset memory before each evaluation
        model.reset_memory()

        # Process sequence
        output = model(sequence, deterministic=False)

        # Extract needle position output
        needle_pos = int(metadata["needle_position"])  # type: ignore[arg-type]
        needle_len = int(metadata["needle_length"])  # type: ignore[arg-type]

        needle_output = output[:, needle_pos : needle_pos + needle_len, :]
        needle_target = target[:, needle_pos : needle_pos + needle_len, :]

        # Compute recall metrics
        mse = float(jnp.mean((needle_output - needle_target) ** 2))
        mae = float(jnp.mean(jnp.abs(needle_output - needle_target)))

        # Compute relative error (normalized by target magnitude)
        target_norm = float(jnp.linalg.norm(needle_target))
        if target_norm > 0:
            relative_error = float(
                jnp.linalg.norm(needle_output - needle_target) / target_norm
            )
        else:
            relative_error = float(jnp.linalg.norm(needle_output - needle_target))

        # Compute recall accuracy (within threshold)
        threshold = 0.5  # Configurable threshold
        recall_accuracy = float(
            jnp.mean(jnp.abs(needle_output - needle_target) < threshold)
        )

        # Compute sequence-level metrics
        seq_mse = float(jnp.mean((output - target) ** 2))
        seq_mae = float(jnp.mean(jnp.abs(output - target)))

        return {
            "needle_mse": mse,
            "needle_mae": mae,
            "needle_relative_error": relative_error,
            "needle_recall_accuracy": recall_accuracy,
            "sequence_mse": seq_mse,
            "sequence_mae": seq_mae,
            "memory_norm": model.get_memory_norm(),
        }

    def run_single_trial(
        self, model: TitansModel, config: ExperimentConfig, trial_id: int
    ) -> ExperimentResult:
        """Run a single trial of the needle-in-haystack experiment."""
        key = jax.random.key(config.seed + trial_id)

        # Sample tasks for this trial
        num_tasks_per_trial = 10  # Configurable
        keys = jax.random.split(key, num_tasks_per_trial + 1)

        trial_metrics = []
        memory_evolution = []

        for i, task in enumerate(
            random.sample(
                self.needle_tasks, min(num_tasks_per_trial, len(self.needle_tasks))
            )
        ):
            try:
                # Create needle sequence
                sequence, target, task_meta = self._create_needle_sequence(
                    task, config.d_model, keys[i]
                )

                # Evaluate model
                metrics = self._evaluate_needle_recall(
                    model, sequence, target, task_meta
                )

                # Add task-specific metadata
                metrics.update(
                    {
                        f"task_{i}_seq_len": task.sequence_length,
                        f"task_{i}_needle_type": task.needle_type,
                        f"task_{i}_difficulty": task.difficulty,
                        f"task_{i}_position": task.needle_position,
                    }
                )

                trial_metrics.append(metrics)
                memory_evolution.append(metrics["memory_norm"])

            except Exception as e:
                self.logger.warning(f"Task {i} failed in trial {trial_id}: {e!s}")
                continue

        # Aggregate metrics across tasks
        aggregated_metrics: dict[str, MetricValue] = {}

        # Store detailed per-task results for plotting
        detailed_task_results = []
        for i, m in enumerate(trial_metrics):
            task_result = {
                "needle_mse": m.get("needle_mse", 0.0),
                "needle_mae": m.get("needle_mae", 0.0),
                "needle_recall_accuracy": m.get("needle_recall_accuracy", 0.0),
                "needle_relative_error": m.get("needle_relative_error", 0.0),
                "seq_len": m.get(f"task_{i}_seq_len", 512),
                "needle_type": m.get(f"task_{i}_needle_type", "unknown"),
                "difficulty": m.get(f"task_{i}_difficulty", "unknown"),
            }
            detailed_task_results.append(task_result)

        metadata: dict[str, MetadataValue] = {
            "num_needle_tasks": len(self.needle_tasks),
            "tasks_per_trial": num_tasks_per_trial,
            "detailed_task_results": json.dumps(detailed_task_results),
        }

        if trial_metrics:
            for key in trial_metrics[0]:
                if key.startswith("task_"):
                    continue  # Skip task-specific metadata
                values = [m[key] for m in trial_metrics if key in m]
                if values:
                    aggregated_metrics[f"avg_{key}"] = float(np.mean(values))
                    aggregated_metrics[f"std_{key}"] = float(np.std(values))
                    aggregated_metrics[f"min_{key}"] = float(np.min(values))
                    aggregated_metrics[f"max_{key}"] = float(np.max(values))

            # Add summary metrics
            aggregated_metrics["num_tasks_completed"] = len(trial_metrics)
            aggregated_metrics["avg_recall_success_rate"] = float(
                np.mean([m["needle_recall_accuracy"] for m in trial_metrics])
            )

        else:
            aggregated_metrics["num_tasks_completed"] = 0
            aggregated_metrics["error"] = "all tasks failed"

        return ExperimentResult(
            config=config,
            variant=None,  # Multiple variants tested
            trial_id=trial_id,
            metrics=aggregated_metrics,
            metadata=metadata,
            execution_time=0.0,  # Will be set by base class
            memory_evolution=memory_evolution,
        )

    def run(self) -> list[ExperimentResult]:
        """Run the complete needle-in-haystack experiment."""
        config = ExperimentConfig(
            name="needle_in_haystack",
            description="Test model's ability to recall information at different positions",
            variants=["mac", "mag", "mal", "lmm"],
            d_model=256,
            memory_depths=[2, 3, 4],
            sequence_lengths=[512, 1024, 2048, 4096],
            batch_size=1,  # Single sequence per trial
            num_trials=5,
            seed=42,
            timeout_seconds=300.0,
        )

        # Run experiment using base class infrastructure
        results = self.run_experiment(config, self.run_single_trial)

        # Save results
        self.save_results(results, "needle_in_haystack_results.json")

        # Analyze results
        analysis = self.analyze_results(results)

        # Save analysis
        with open(self.output_dir / "needle_in_haystack_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        # Create plots
        self.plot_results(results, self.output_dir / "needle_in_haystack_plots.png")

        # Create specialized plots for needle experiment
        self._create_specialized_plots(results)

        self.logger.info("Needle-in-Haystack experiment completed")
        return results

    def _create_specialized_plots(self, results: list[ExperimentResult]) -> None:
        """Create specialized plots for needle-in-haystack experiment."""
        successful_results = [r for r in results if r.error is None]
        if not successful_results:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Needle-in-Haystack Detailed Analysis", fontsize=16)

        # Plot 1: Recall accuracy by variant
        self._plot_recall_by_variant(successful_results, axes[0, 0])

        # Plot 2: Performance by sequence length
        self._plot_performance_by_sequence_length(successful_results, axes[0, 1])

        # Plot 3: Memory evolution comparison
        self._plot_memory_evolution_comparison(successful_results, axes[0, 2])

        # Plot 4: Error analysis
        self._plot_error_analysis(successful_results, axes[1, 0])

        # Plot 5: Task completion rates
        self._plot_task_completion_rates(successful_results, axes[1, 1])

        # Plot 6: Relative error distribution
        self._plot_relative_error_distribution(successful_results, axes[1, 2])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "needle_specialized_plots.png",
            dpi=300,
            bbox_inches="tight",
        )
        if plt.isinteractive():
            plt.show()
        plt.close()

    def _plot_recall_by_variant(self, results: list[ExperimentResult], ax: Any) -> None:
        """Plot recall accuracy by variant."""
        variant_accuracy = {}
        for result in results:
            if "avg_needle_recall_accuracy" in result.metrics:
                variant = result.variant
                if variant not in variant_accuracy:
                    variant_accuracy[variant] = []
                variant_accuracy[variant].append(
                    result.metrics["avg_needle_recall_accuracy"]
                )

        if variant_accuracy:
            variants = list(variant_accuracy.keys())
            means = [np.mean(variant_accuracy[v]) for v in variants]
            stds = [np.std(variant_accuracy[v]) for v in variants]

            bars = ax.bar(variants, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title("Needle Recall Accuracy by Variant")
            ax.set_ylabel("Recall Accuracy")
            ax.set_xlabel("Variant")
            ax.set_ylim(0, 1)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                )

    def _collect_seq_len_performance(
        self, results: list[ExperimentResult]
    ) -> dict[int, dict[str, list[float]]]:
        """Collect per-sequence-length performance from detailed results."""
        seq_len_performance: dict[int, dict[str, list[float]]] = {}

        for result in results:
            if result.error:
                continue
            self._process_result_for_seq_len(result, seq_len_performance)

        return seq_len_performance

    def _process_result_for_seq_len(
        self,
        result: ExperimentResult,
        seq_len_performance: dict[int, dict[str, list[float]]],
    ) -> None:
        """Process a single result for sequence length performance."""
        variant = result.variant or "unknown"
        detailed_json = result.metadata.get("detailed_task_results", "[]")

        if not isinstance(detailed_json, str):
            return

        try:
            detailed_results = json.loads(detailed_json)
        except json.JSONDecodeError:
            return

        for task_result in detailed_results:
            seq_len = task_result.get("seq_len", 512)
            accuracy = task_result.get("needle_recall_accuracy", 0.0)
            seq_len_performance.setdefault(seq_len, {}).setdefault(variant, []).append(
                accuracy
            )

    def _plot_performance_by_sequence_length(
        self, results: list[ExperimentResult], ax: Any
    ) -> None:
        """Plot performance degradation with sequence length."""
        seq_len_performance = self._collect_seq_len_performance(results)

        if not seq_len_performance:
            ax.text(
                0.5,
                0.5,
                "No sequence length data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        self._render_seq_len_plot(seq_len_performance, ax)

    def _render_seq_len_plot(
        self, seq_len_performance: dict[int, dict[str, list[float]]], ax: Any
    ) -> None:
        """Render the sequence length performance plot."""
        seq_lengths = sorted(seq_len_performance.keys())
        variants: set[str] = set()
        for sl_data in seq_len_performance.values():
            variants.update(sl_data.keys())

        colors = {"mac": "blue", "mag": "green", "mal": "red", "lmm": "purple"}

        for variant in sorted(variants):
            means, stds, valid_seq_lens = self._compute_variant_stats(
                variant, seq_lengths, seq_len_performance
            )
            if valid_seq_lens:
                color = colors.get(variant, "gray")
                ax.errorbar(
                    valid_seq_lens,
                    means,
                    yerr=stds,
                    label=variant.upper(),
                    marker="o",
                    capsize=3,
                    color=color,
                    alpha=0.7,
                )

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Recall Accuracy")
        ax.set_title("Performance by Sequence Length")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)

    def _compute_variant_stats(
        self,
        variant: str,
        seq_lengths: list[int],
        seq_len_performance: dict[int, dict[str, list[float]]],
    ) -> tuple[list[float], list[float], list[int]]:
        """Compute mean and std for a variant across sequence lengths."""
        means: list[float] = []
        stds: list[float] = []
        valid_seq_lens: list[int] = []

        for sl in seq_lengths:
            if variant in seq_len_performance[sl]:
                values = seq_len_performance[sl][variant]
                means.append(float(np.mean(values)))
                stds.append(float(np.std(values)))
                valid_seq_lens.append(sl)

        return means, stds, valid_seq_lens

    def _plot_memory_evolution_comparison(
        self, results: list[ExperimentResult], ax: Any
    ) -> None:
        """Plot memory evolution for different variants."""
        variant_evolutions: dict[str, list[list[float]]] = {}
        for result in results:
            if result.memory_evolution:
                variant = result.variant or "unknown"
                if variant not in variant_evolutions:
                    variant_evolutions[variant] = []
                variant_evolutions[variant].append(result.memory_evolution)

        # Define styles for each variant to handle overlapping lines
        variant_colors = {
            "mac": "#E74C3C",
            "mag": "#F39C12",
            "mal": "#27AE60",
            "lmm": "#3498DB",
        }
        variant_markers = {"mac": "o", "mag": "s", "mal": "^", "lmm": "D"}
        variant_linestyles = {"mac": "-", "mag": "--", "mal": "-.", "lmm": ":"}
        variant_offsets = {"mac": 0, "mag": 0.02, "mal": 0.04, "lmm": 0.06}

        for variant in sorted(variant_evolutions.keys()):
            evolutions = variant_evolutions[variant]
            if evolutions:
                # Average evolution
                max_len = max(len(evo) for evo in evolutions)
                avg_evolution = []
                for i in range(max_len):
                    values = [evo[i] for evo in evolutions if i < len(evo)]
                    if values:
                        avg_evolution.append(np.mean(values))

                if avg_evolution:
                    color = variant_colors.get(variant, "gray")
                    marker = variant_markers.get(variant, "o")
                    linestyle = variant_linestyles.get(variant, "-")
                    offset = variant_offsets.get(variant, 0)
                    offset_evolution = [v + offset for v in avg_evolution]
                    ax.plot(
                        offset_evolution,
                        label=variant.upper(),
                        marker=marker,
                        color=color,
                        linestyle=linestyle,
                        alpha=0.7,
                    )

        ax.set_title("Memory Evolution During Trials")
        ax.set_ylabel("Memory Magnitude")
        ax.set_xlabel("Task Number")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_error_analysis(self, results: list[ExperimentResult], ax: Any) -> None:
        """Plot error analysis across variants."""
        variant_errors = {}
        for result in results:
            if "avg_needle_relative_error" in result.metrics:
                variant = result.variant
                if variant not in variant_errors:
                    variant_errors[variant] = []
                variant_errors[variant].append(
                    result.metrics["avg_needle_relative_error"]
                )

        if variant_errors:
            data = [variant_errors[variant] for variant in variant_errors]
            labels = list(variant_errors.keys())

            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            ax.set_title("Relative Error Distribution")
            ax.set_ylabel("Relative Error")
            ax.set_xlabel("Variant")

            # Color the boxes
            colors = ["lightblue", "lightgreen", "lightcoral", "lightyellow"]
            for patch, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
                patch.set_facecolor(color)

    def _plot_task_completion_rates(
        self, results: list[ExperimentResult], ax: Any
    ) -> None:
        """Plot task completion rates by variant."""
        variant_completion = {}
        for result in results:
            if "num_tasks_completed" in result.metrics:
                variant = result.variant
                if variant not in variant_completion:
                    variant_completion[variant] = []
                variant_completion[variant].append(
                    result.metrics["num_tasks_completed"]
                )

        if variant_completion:
            variants = list(variant_completion.keys())
            avg_completion = [np.mean(variant_completion[v]) for v in variants]

            bars = ax.bar(variants, avg_completion, alpha=0.7, color="skyblue")
            ax.set_title("Average Tasks Completed per Trial")
            ax.set_ylabel("Number of Tasks")
            ax.set_xlabel("Variant")

            # Add value labels
            for bar, avg in zip(bars, avg_completion):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{avg:.1f}",
                    ha="center",
                    va="bottom",
                )

    def _plot_relative_error_distribution(
        self, results: list[ExperimentResult], ax: Any
    ) -> None:
        """Plot distribution of relative errors."""
        all_errors = []
        for result in results:
            if "avg_needle_relative_error" in result.metrics:
                all_errors.append(result.metrics["avg_needle_relative_error"])

        if all_errors:
            ax.hist(all_errors, bins=20, alpha=0.7, color="salmon", edgecolor="black")
            ax.set_title("Distribution of Relative Errors")
            ax.set_xlabel("Relative Error")
            ax.set_ylabel("Frequency")
            ax.axvline(
                np.mean(all_errors),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(all_errors):.3f}",
            )
            ax.legend()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Needle-in-Haystack Experiment for Titans Paper Reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run with defaults
  %(prog)s --output-dir ./results             # Custom output directory
  %(prog)s --quick                            # Quick run with reduced settings
  %(prog)s --d-model 128 --num-trials 3       # Custom model size and trials
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="needle_in_haystack_results",
        help="Output directory for results (default: needle_in_haystack_results)",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension (default: 256)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of trials per configuration (default: 5)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with reduced settings for faster execution",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["mac", "mag", "mal", "lmm"],
        default=["mac", "mag", "mal", "lmm"],
        help="Variants to test (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.d_model <= 0:
        parser.error("--d-model must be a positive integer")
    if args.num_trials <= 0:
        parser.error("--num-trials must be a positive integer")
    if args.seed < 0:
        parser.error("--seed must be a non-negative integer")

    # Apply quick settings if requested
    if args.quick:
        args.d_model = min(args.d_model, 128)
        args.num_trials = min(args.num_trials, 2)

    print("🔍 Needle-in-Haystack Experiment")
    print("=" * 50)
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model dimension:  {args.d_model}")
    print(f"  Number of trials: {args.num_trials}")
    print(f"  Variants:         {', '.join(args.variants)}")
    print(f"  Seed:             {args.seed}")
    print(f"  Quick mode:       {args.quick}")
    print("=" * 50)

    # Create and run the experiment
    experiment = NeedleInHaystackExperiment(output_dir=Path(args.output_dir))

    # Override default config with command line args
    config = ExperimentConfig(
        name="needle_in_haystack",
        description="Test model's ability to recall information at different positions",
        variants=args.variants,
        d_model=args.d_model,
        memory_depths=[2, 3] if args.quick else [2, 3, 4],
        sequence_lengths=[512, 1024] if args.quick else [512, 1024, 2048, 4096],
        batch_size=1,
        num_trials=args.num_trials,
        seed=args.seed,
        timeout_seconds=300.0,
    )

    # Run experiment
    print("\n🚀 Starting experiment...")
    results = experiment.run_experiment(config, experiment.run_single_trial)

    # Save and analyze results
    experiment.save_results(results, "needle_in_haystack_results.json")
    analysis = experiment.analyze_results(results)

    with open(Path(args.output_dir) / "needle_in_haystack_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Create plots
    experiment.plot_results(
        results, Path(args.output_dir) / "needle_in_haystack_plots.png"
    )
    experiment._create_specialized_plots(results)

    # Print summary
    successful = sum(1 for r in results if r.error is None)
    print("\n" + "=" * 50)
    print("📊 Experiment Summary")
    print("=" * 50)
    print(f"  Total trials:      {len(results)}")
    print(f"  Successful trials: {successful}")
    print(f"  Failed trials:     {len(results) - successful}")
    print(f"  Results saved to:  {args.output_dir}/")
    print("\n✅ Needle-in-Haystack experiment completed!")
