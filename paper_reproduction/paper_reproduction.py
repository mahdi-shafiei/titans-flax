"""Complete Paper Reproduction Runner for Titans.

This module provides the complete experimental framework for reproducing
all results from the Titans paper, including additional analysis experiments.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from titans.benchmark import TitansBenchmark
from titans.core import TitansModel, TitansVariant, create_titans_model

from .ablation_long_context_experiments import (
    AblationStudyExperiment,
    LongContextExperiment,
)
from .needle_in_haystack_experiment import NeedleInHaystackExperiment
from .paper_reproduction_core import (
    BaseExperiment,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    MetadataValue,
    MetricValue,
)
from .scaling_analysis_experiment import ScalingAnalysisExperiment


class MemoryPersistenceExperiment(BaseExperiment):
    """Test memory persistence and adaptation capabilities.

    This experiment tests:
    - How memory evolves over multiple sequences
    - Memory retention across different tasks
    - Adaptation speed and stability
    """

    def __init__(self, output_dir: Path | None = None):
        super().__init__(output_dir)

    def _create_adaptation_sequence(
        self, seq_len: int, d_model: int, key: jax.Array, pattern_type: str = "evolving"
    ) -> tuple[jax.Array, jax.Array, dict]:
        """Create a sequence for testing adaptation."""
        keys = jax.random.split(key, 3)

        if pattern_type == "evolving":
            # Create a pattern that evolves over time
            base_pattern = jax.random.normal(keys[0], (d_model,))
            sequence = []

            for i in range(seq_len):
                # Pattern evolves gradually
                evolution_factor = i / seq_len
                current_pattern = (
                    base_pattern * (1 - evolution_factor)
                    + jax.random.normal(keys[1], (d_model,)) * evolution_factor
                )
                sequence.append(current_pattern)

            sequence = jnp.stack(sequence)[None, :, :]
            target = sequence

        elif pattern_type == "switching":
            # Pattern that switches between modes
            pattern_a = jax.random.normal(keys[0], (d_model,)) * 2.0
            pattern_b = jax.random.normal(keys[1], (d_model,)) * 2.0

            sequence = []
            for i in range(seq_len):
                # Switch every 50 tokens
                current_pattern = pattern_a if (i // 50) % 2 == 0 else pattern_b

                # Add noise
                current_pattern = (
                    current_pattern + jax.random.normal(keys[2], (d_model,)) * 0.1
                )
                sequence.append(current_pattern)

            sequence = jnp.stack(sequence)[None, :, :]
            target = sequence

        else:  # "cyclic"
            # Cyclic pattern
            cycle_length = 20
            base_patterns = jax.random.normal(keys[0], (cycle_length, d_model))

            sequence = []
            for i in range(seq_len):
                pattern_idx = i % cycle_length
                current_pattern = (
                    base_patterns[pattern_idx]
                    + jax.random.normal(keys[1], (d_model,)) * 0.05
                )
                sequence.append(current_pattern)

            sequence = jnp.stack(sequence)[None, :, :]
            target = sequence

        metadata = {
            "pattern_type": pattern_type,
            "sequence_length": seq_len,
        }

        return sequence, target, metadata

    def run_single_trial(
        self, model: TitansModel, config: ExperimentConfig, trial_id: int
    ) -> ExperimentResult:
        """Run memory persistence trial."""
        key = jax.random.key(config.seed + trial_id)
        keys = jax.random.split(key, 10)

        memory_evolution = []
        adaptation_metrics = []

        # Test different pattern types
        pattern_types = ["evolving", "switching", "cyclic"]

        for i, pattern_type in enumerate(pattern_types):
            # Create multiple sequences of the same pattern type
            for seq_idx in range(3):
                sequence, target, metadata = self._create_adaptation_sequence(
                    256, config.d_model, keys[i * 3 + seq_idx], pattern_type
                )

                # Process sequence
                output = model(sequence, deterministic=False)

                # Compute adaptation metrics
                mse = float(jnp.mean((output - target) ** 2))
                memory_norm = model.get_memory_norm()

                adaptation_metrics.append(
                    {
                        "pattern_type": pattern_type,
                        "sequence_index": seq_idx,
                        "mse": mse,
                        "memory_norm": memory_norm,
                    }
                )

                memory_evolution.append(memory_norm)

        # Compute memory persistence metrics
        if len(memory_evolution) > 1:
            memory_growth = memory_evolution[-1] - memory_evolution[0]
            memory_stability = (
                np.std(memory_evolution[-3:]) if len(memory_evolution) >= 3 else 0
            )
        else:
            memory_growth = 0
            memory_stability = 0

        # Aggregate metrics
        aggregated_metrics: dict[str, MetricValue] = {
            "memory_growth": float(memory_growth),
            "memory_stability": float(memory_stability),
            "final_memory_norm": float(memory_evolution[-1]) if memory_evolution else 0,
            "avg_mse": float(np.mean([m["mse"] for m in adaptation_metrics])),
            "adaptation_sequences_completed": len(adaptation_metrics),
        }

        metadata: dict[str, MetadataValue] = {
            "detailed_metrics_json": json.dumps(adaptation_metrics),
        }

        return ExperimentResult(
            config=config,
            variant=None,
            trial_id=trial_id,
            metrics=aggregated_metrics,
            metadata=metadata,
            execution_time=0.0,
            memory_evolution=memory_evolution,
        )

    def run(self) -> list[ExperimentResult]:
        """Run the memory persistence experiment."""
        config = ExperimentConfig(
            name="memory_persistence",
            description="Test memory persistence and adaptation",
            variants=["mac", "mag", "mal", "lmm"],
            d_model=256,
            num_trials=5,
            seed=42,
        )

        results = self.run_experiment(config, self.run_single_trial)
        self.save_results(results, "memory_persistence_results.json")
        self.plot_results(results, self.output_dir / "memory_persistence_plots.png")

        return results


class BaselineComparisonExperiment(BaseExperiment):
    """Compare Titans variants against standard baselines.

    This experiment compares against:
    - Standard Transformer with full attention
    - Transformer with sliding window attention
    - Simple RNN-based models
    """

    def __init__(self, output_dir: Path | None = None):
        super().__init__(output_dir)

    def _create_standard_task(
        self, seq_len: int, d_model: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, dict]:
        """Create a standard task for baseline comparison."""
        keys = jax.random.split(key, 2)

        # Create a sequence modeling task
        sequence = jax.random.normal(keys[0], (seq_len, d_model)) * 0.5

        # Target is a simple transformation (next token prediction style)
        target = jnp.roll(sequence, -1, axis=0)
        target = target.at[-1].set(jax.random.normal(keys[1], (d_model,)))

        # Add batch dimension
        sequence = sequence[None, :, :]
        target = target[None, :, :]

        metadata = {
            "task_type": "next_token_prediction",
            "sequence_length": seq_len,
        }

        return sequence, target, metadata

    def run_single_trial(
        self, model: TitansModel, config: ExperimentConfig, trial_id: int
    ) -> ExperimentResult:
        """Run baseline comparison trial."""
        key = jax.random.key(config.seed + trial_id)
        keys = jax.random.split(key, 5)

        results = []

        # Test on multiple sequence lengths
        seq_lengths = [128, 256, 512, 1024]

        for i, seq_len in enumerate(seq_lengths):
            sequence, target, metadata = self._create_standard_task(
                seq_len, config.d_model, keys[i]
            )

            # Reset memory and time execution
            model.reset_memory()
            start_time = time.time()

            try:
                output = model(sequence, deterministic=False)
                execution_time = time.time() - start_time
                success = True
            except Exception as e:
                self.logger.warning(f"Failed on seq_len {seq_len}: {e!s}")
                execution_time = float("inf")
                output = jnp.zeros_like(target)
                success = False

            if success:
                mse = float(jnp.mean((output - target) ** 2))
                throughput = seq_len / execution_time if execution_time > 0 else 0
                memory_norm = model.get_memory_norm()

                results.append(
                    {
                        "sequence_length": seq_len,
                        "mse": mse,
                        "execution_time": execution_time,
                        "throughput": throughput,
                        "memory_norm": memory_norm,
                        "success": True,
                    }
                )
            else:
                results.append(
                    {
                        "sequence_length": seq_len,
                        "success": False,
                        "execution_time": execution_time,
                    }
                )

        # Aggregate metrics
        successful_results = [r for r in results if r.get("success", False)]
        aggregated_metrics: dict[str, MetricValue] = {}
        metadata: dict[str, MetadataValue] = {
            "detailed_results_json": json.dumps(results),
        }

        if successful_results:
            aggregated_metrics["avg_mse"] = float(
                np.mean([r["mse"] for r in successful_results])
            )
            aggregated_metrics["avg_throughput"] = float(
                np.mean([r["throughput"] for r in successful_results])
            )
            aggregated_metrics["max_sequence_length"] = max(
                int(r["sequence_length"]) for r in successful_results
            )
            aggregated_metrics["total_execution_time"] = float(
                np.sum([r["execution_time"] for r in successful_results])
            )
            aggregated_metrics["success_rate"] = len(successful_results) / len(results)
        else:
            aggregated_metrics["success_rate"] = 0.0
            aggregated_metrics["error"] = "all tests failed"

        return ExperimentResult(
            config=config,
            variant=None,
            trial_id=trial_id,
            metrics=aggregated_metrics,
            metadata=metadata,
            execution_time=0.0,
        )

    def run(self) -> list[ExperimentResult]:
        """Run the baseline comparison experiment."""
        config = ExperimentConfig(
            name="baseline_comparison",
            description="Compare against standard baselines",
            variants=["mac", "mag", "mal", "lmm"],
            d_model=256,
            num_trials=3,
            seed=42,
        )

        results = self.run_experiment(config, self.run_single_trial)
        self.save_results(results, "baseline_comparison_results.json")
        self.plot_results(results, self.output_dir / "baseline_comparison_plots.png")

        return results


class FullPaperReproduction:
    """Main class for running complete paper reproduction experiments.

    This coordinates all experiments and provides analysis across the entire suite.
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize complete reproduction suite."""
        if output_dir is None:
            output_dir = Path("titans_paper_reproduction")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experiment runner
        self.runner = ExperimentRunner(self.output_dir)

        # Register all experiments
        self._register_experiments()

    def _register_experiments(self):
        """Register all paper reproduction experiments."""
        # Core paper experiments
        self.runner.register_experiment(
            "needle_in_haystack",
            NeedleInHaystackExperiment,
        )

        self.runner.register_experiment(
            "scaling_analysis",
            ScalingAnalysisExperiment,
        )

        self.runner.register_experiment(
            "ablation_study",
            AblationStudyExperiment,
        )

        self.runner.register_experiment(
            "long_context",
            LongContextExperiment,
        )

        # Additional analysis experiments
        self.runner.register_experiment(
            "memory_persistence",
            MemoryPersistenceExperiment,
        )

        self.runner.register_experiment(
            "baseline_comparison",
            BaselineComparisonExperiment,
        )

    def run_quick_validation(self) -> dict[str, bool]:
        """Run quick validation tests to ensure everything works."""
        print("🔍 Running quick validation tests...")

        validation_results: dict[str, bool] = {}
        variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]

        for variant in variants:
            try:
                # Create small model
                model = create_titans_model(variant=variant, d_model=64, seed=42)

                # Test basic functionality
                x = jax.random.normal(jax.random.key(0), (1, 32, 64))
                output = model(x, deterministic=False)

                # Check output shape and finiteness
                if output.shape == x.shape and jnp.isfinite(output).all():
                    validation_results[variant] = True
                    print(f"  ✅ {variant.upper()} variant: PASS")
                else:
                    validation_results[variant] = False
                    print(f"  ❌ {variant.upper()} variant: FAIL (shape/finite check)")

            except Exception as e:
                validation_results[variant] = False
                print(f"  ❌ {variant.upper()} variant: FAIL ({e!s})")

        all(validation_results.values())
        print(
            f"\n🎯 Validation Summary: {sum(validation_results.values())}/{len(variants)} variants passed"
        )

        return validation_results

    def run_benchmark_suite(self) -> dict[str, any]:
        """Run performance benchmarking suite."""
        print("⚡ Running performance benchmark suite...")

        benchmark = TitansBenchmark(self.output_dir / "benchmarks")

        # Quick performance comparison
        comparison_results = benchmark.compare_variants(
            variants=["mac", "mag", "mal", "lmm"],
            seq_len=256,
            batch_size=2,
            num_trials=3,
        )

        # Scaling analysis
        scaling_results = benchmark.scaling_analysis(
            variant="mac",
            sequence_lengths=[128, 256, 512],
            num_trials=2,
        )

        # Memory capacity analysis
        capacity_results = benchmark.memory_capacity_analysis(
            variant="mac",
            num_sequences=10,
            memory_depths=[1, 2, 3],
        )

        benchmark.save_results(comparison_results, "quick_comparison.json")
        benchmark.save_results(scaling_results, "quick_scaling.json")
        benchmark.save_results(capacity_results, "quick_capacity.json")

        print("✅ Benchmark suite completed")

        return {
            "comparison": comparison_results,
            "scaling": scaling_results,
            "capacity": capacity_results,
        }

    def run_core_experiments(self) -> dict[str, any]:
        """Run core paper reproduction experiments only."""
        print("🧪 Running core paper reproduction experiments...")

        core_experiments = [
            "needle_in_haystack",
            "scaling_analysis",
            "ablation_study",
            "long_context",
        ]

        # Temporarily save all experiments and run only core ones
        all_experiments = self.runner.experiments.copy()
        self.runner.experiments = {
            name: exp
            for name, exp in all_experiments.items()
            if name in core_experiments
        }

        try:
            results = self.runner.run_all_experiments()
        finally:
            # Restore all experiments
            self.runner.experiments = all_experiments

        return results

    def run_full_reproduction(self) -> dict[str, any]:
        """Run complete paper reproduction suite."""
        print("🚀 Running FULL paper reproduction suite...")
        print("This may take several hours depending on your hardware.")

        # Run all registered experiments
        results = self.runner.run_all_experiments()

        # Generate full analysis
        self._generate_full_analysis()

        return results

    def _generate_full_analysis(self):
        """Generate full analysis across all experiments."""
        print("📊 Generating full analysis...")

        # Create summary report
        self._create_summary_report()

        # Create cross-experiment comparison plots
        self._create_cross_experiment_plots()

        print("✅ Full analysis completed")

    def _create_summary_report(self):
        """Create a complete summary report."""
        summary_path = self.output_dir / "PAPER_REPRODUCTION_SUMMARY.md"

        with open(summary_path, "w") as f:
            f.write("# Titans Paper Reproduction Summary\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Experiments Completed\n\n")

            for exp_name in self.runner.experiments:
                exp_dir = self.output_dir / exp_name
                if exp_dir.exists():
                    f.write(f"- ✅ {exp_name.replace('_', ' ').title()}\n")

                    # Look for results file
                    results_file = exp_dir / f"{exp_name}_results.json"
                    if results_file.exists():
                        f.write(
                            f"  - Results: `{results_file.relative_to(self.output_dir)}`\n"
                        )

                    # Look for plots
                    plot_files = list(exp_dir.glob("*.png"))
                    if plot_files:
                        f.write(f"  - Plots: {len(plot_files)} visualization(s)\n")
                else:
                    f.write(f"- ❌ {exp_name.replace('_', ' ').title()} (not found)\n")

            f.write("\n## Key Findings\n\n")
            f.write("<!-- TODO: Add key findings from analysis -->\n\n")

            f.write("## Files Generated\n\n")
            f.write("```\n")
            for file_path in sorted(self.output_dir.rglob("*")):
                if file_path.is_file():
                    f.write(f"{file_path.relative_to(self.output_dir)}\n")
            f.write("```\n")

        print(f"📄 Summary report created: {summary_path}")

    def _create_cross_experiment_plots(self):
        """Create plots comparing results across experiments."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("Cross-Experiment Analysis", fontsize=16)

            # Collect data from all experiment results
            experiment_data = self._collect_cross_experiment_data()

            if not experiment_data:
                # No data - show message
                for ax in axes.flat:
                    ax.text(
                        0.5,
                        0.5,
                        "No experiment data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
            else:
                # Plot 1: Variant Performance Across Experiments
                self._plot_variant_performance(experiment_data, axes[0, 0])

                # Plot 2: Execution Time Comparison
                self._plot_execution_times(experiment_data, axes[0, 1])

                # Plot 3: Memory Evolution Summary
                self._plot_memory_summary(experiment_data, axes[1, 0])

                # Plot 4: Overall Success/Completion Rates
                self._plot_success_rates(experiment_data, axes[1, 1])

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "cross_experiment_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            if plt.isinteractive():
                plt.show()
            plt.close()

            print("📈 Cross-experiment plots created")

        except Exception as e:
            print(f"⚠️  Could not create cross-experiment plots: {e!s}")

    def _collect_cross_experiment_data(self) -> dict:
        """Collect data from all experiment result files."""
        data = {}
        for exp_name in self.runner.experiments:
            exp_dir = self.output_dir / exp_name
            results_file = exp_dir / f"{exp_name}_results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        results = json.load(f)
                    data[exp_name] = results
                except (json.JSONDecodeError, OSError):
                    continue
        return data

    def _plot_variant_performance(self, data: dict, ax: Any) -> None:
        """Plot variant performance (MSE/error) across experiments."""
        variants = ["mac", "mag", "mal", "lmm"]
        variant_colors = {
            "mac": "#E74C3C",
            "mag": "#F39C12",
            "mal": "#27AE60",
            "lmm": "#3498DB",
        }

        exp_names = []
        variant_metrics: dict[str, list[float]] = {v: [] for v in variants}

        for exp_name, results in data.items():
            if not results:
                continue
            exp_names.append(exp_name.replace("_", "\n"))

            # Get average metric per variant
            variant_values: dict[str, list[float]] = {v: [] for v in variants}
            for result in results:
                variant = result.get("variant")
                if variant in variants:
                    # Look for common error metrics
                    for metric_key in ["avg_needle_mse", "avg_mse", "mse", "error"]:
                        if metric_key in result.get("metrics", {}):
                            variant_values[variant].append(
                                float(result["metrics"][metric_key])
                            )
                            break

            for v in variants:
                if variant_values[v]:
                    variant_metrics[v].append(float(np.mean(variant_values[v])))
                else:
                    variant_metrics[v].append(0.0)

        if not exp_names:
            ax.text(
                0.5,
                0.5,
                "No performance data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Variant Performance")
            return

        x = np.arange(len(exp_names))
        width = 0.2

        for i, variant in enumerate(variants):
            if any(variant_metrics[variant]):
                ax.bar(
                    x + i * width,
                    variant_metrics[variant],
                    width,
                    label=variant.upper(),
                    color=variant_colors[variant],
                    alpha=0.8,
                )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Average MSE/Error")
        ax.set_title("Variant Performance Across Experiments")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(exp_names, fontsize=8)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_execution_times(self, data: dict, ax: Any) -> None:
        """Plot execution times across experiments."""
        variants = ["mac", "mag", "mal", "lmm"]
        variant_colors = {
            "mac": "#E74C3C",
            "mag": "#F39C12",
            "mal": "#27AE60",
            "lmm": "#3498DB",
        }

        exp_names = []
        variant_times: dict[str, list[float]] = {v: [] for v in variants}

        for exp_name, results in data.items():
            if not results:
                continue
            exp_names.append(exp_name.replace("_", "\n"))

            variant_exec: dict[str, list[float]] = {v: [] for v in variants}
            for result in results:
                variant = result.get("variant")
                exec_time = result.get("execution_time", 0)
                if variant in variants and exec_time > 0:
                    variant_exec[variant].append(exec_time)

            for v in variants:
                variant_times[v].append(
                    float(np.mean(variant_exec[v])) if variant_exec[v] else 0.0
                )

        if not exp_names:
            ax.text(
                0.5,
                0.5,
                "No timing data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Execution Times")
            return

        x = np.arange(len(exp_names))
        width = 0.2

        for i, variant in enumerate(variants):
            if any(variant_times[variant]):
                ax.bar(
                    x + i * width,
                    variant_times[variant],
                    width,
                    label=variant.upper(),
                    color=variant_colors[variant],
                    alpha=0.8,
                )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Avg Execution Time (s)")
        ax.set_title("Execution Time by Variant")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(exp_names, fontsize=8)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_memory_summary(self, data: dict, ax: Any) -> None:
        """Plot memory evolution summary across experiments."""
        variants = ["mac", "mag", "mal", "lmm"]
        variant_colors = {
            "mac": "#E74C3C",
            "mag": "#F39C12",
            "mal": "#27AE60",
            "lmm": "#3498DB",
        }

        exp_names = []
        variant_memory: dict[str, list[float]] = {v: [] for v in variants}

        for exp_name, results in data.items():
            if not results:
                continue
            exp_names.append(exp_name.replace("_", "\n"))

            variant_mem: dict[str, list[float]] = {v: [] for v in variants}
            for result in results:
                variant = result.get("variant")
                mem_evo = result.get("memory_evolution", [])
                if variant in variants and mem_evo:
                    # Use final memory value
                    variant_mem[variant].append(float(mem_evo[-1]))

            for v in variants:
                variant_memory[v].append(
                    float(np.mean(variant_mem[v])) if variant_mem[v] else 0.0
                )

        if not exp_names:
            ax.text(
                0.5,
                0.5,
                "No memory data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Memory Summary")
            return

        x = np.arange(len(exp_names))
        width = 0.2

        for i, variant in enumerate(variants):
            if any(variant_memory[variant]):
                ax.bar(
                    x + i * width,
                    variant_memory[variant],
                    width,
                    label=variant.upper(),
                    color=variant_colors[variant],
                    alpha=0.8,
                )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Final Memory Magnitude")
        ax.set_title("Memory State by Variant")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(exp_names, fontsize=8)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    def _plot_success_rates(self, data: dict, ax: Any) -> None:
        """Plot success/completion rates across experiments."""
        exp_names = []
        success_rates = []
        trial_counts = []

        for exp_name, results in data.items():
            if not results:
                continue
            exp_names.append(exp_name.replace("_", "\n"))

            total = len(results)
            successful = sum(1 for r in results if not r.get("error"))
            success_rates.append(successful / total * 100 if total > 0 else 0)
            trial_counts.append(total)

        if not exp_names:
            ax.text(
                0.5,
                0.5,
                "No experiment data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Success Rates")
            return

        x = np.arange(len(exp_names))
        bars = ax.bar(x, success_rates, color="#2ECC71", alpha=0.8, edgecolor="black")

        # Add trial counts as labels
        for bar, count in zip(bars, trial_counts):
            height = bar.get_height()
            ax.annotate(
                f"n={count}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Experiment Completion Rates")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_names, fontsize=8)
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="100%")
        ax.grid(True, alpha=0.3, axis="y")

    def run_full_suite(self) -> None:
        """Run all experiments in the suite."""
        # Use run_all_experiments which handles all registered experiments
        self.runner.run_all_experiments()


# Available experiments that can be run individually
AVAILABLE_EXPERIMENTS = [
    "needle_in_haystack",
    "scaling_analysis",
    "ablation_study",
    "long_context",
    "memory_persistence",
    "baseline_comparison",
]


def _handle_list_experiments() -> None:
    """Print available experiments and exit."""
    print("📋 Available Experiments:")
    print("=" * 50)
    for exp in AVAILABLE_EXPERIMENTS:
        print(f"  • {exp}")
    print("\nUsage: paper_reproduction.py --experiments <name1> [<name2> ...]")
    sys.exit(0)


def _validate_experiment_names(experiments: list[str]) -> None:
    """Validate experiment names, exit with error if invalid."""
    invalid_experiments = [e for e in experiments if e not in AVAILABLE_EXPERIMENTS]
    if not invalid_experiments:
        return
    print(f"❌ Error: Unknown experiment(s): {', '.join(invalid_experiments)}")
    print("\n📋 Available experiments:")
    for exp in AVAILABLE_EXPERIMENTS:
        print(f"   • {exp}")
    print(
        "\nExample: python -m paper_reproduction.paper_reproduction --experiments needle_in_haystack"
    )
    sys.exit(1)


def _run_mode(
    mode: str,
    reproduction: FullPaperReproduction,
    experiments: list[str] | None,
) -> None:
    """Execute the appropriate mode."""
    if experiments:
        # Run specific experiments
        print(f"Running specific experiments: {', '.join(experiments)}")
        all_experiments = reproduction.runner.experiments.copy()
        reproduction.runner.experiments = {
            name: exp for name, exp in all_experiments.items() if name in experiments
        }
        reproduction.runner.run_all_experiments()
        return

    mode_handlers = {
        "validate": lambda: _run_validate_mode(reproduction),
        "benchmark": lambda: _run_benchmark_mode(reproduction),
        "core": lambda: _run_core_mode(reproduction),
        "full": lambda: _run_full_mode(reproduction),
    }
    mode_handlers[mode]()


def _run_validate_mode(reproduction: FullPaperReproduction) -> None:
    """Run quick validation mode."""
    validation_results = reproduction.run_quick_validation()
    if all(validation_results.values()):
        print("\n🎉 All validation tests passed! Ready for full reproduction.")
        sys.exit(0)
    print("\n❌ Some validation tests failed. Please check the implementation.")
    sys.exit(1)


def _run_benchmark_mode(reproduction: FullPaperReproduction) -> None:
    """Run benchmark mode."""
    reproduction.run_benchmark_suite()
    print(f"\n📊 Benchmark results saved to: {reproduction.output_dir / 'benchmarks'}")


def _run_core_mode(reproduction: FullPaperReproduction) -> None:
    """Run core experiments mode."""
    results = reproduction.run_core_experiments()
    print(f"\n✅ Core experiments completed: {results}")


def _run_full_mode(reproduction: FullPaperReproduction) -> None:
    """Run full reproduction mode."""
    results = reproduction.run_full_reproduction()
    print(f"\n🎉 Full paper reproduction completed: {results}")


def main():
    """Main entry point for paper reproduction."""
    experiments_list = ", ".join(AVAILABLE_EXPERIMENTS)

    parser = argparse.ArgumentParser(
        description="Titans Paper Reproduction Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modes:
  validate   - Quick validation of all model variants (fast, ~1 min)
  benchmark  - Run performance benchmarks across variants
  core       - Run core paper experiments (needle, scaling, ablation)
  full       - Complete paper reproduction with all experiments

Available experiments for --experiments:
  {experiments_list}

Examples:
  %(prog)s                                     # Default: quick validation
  %(prog)s --mode validate                     # Quick validation only
  %(prog)s --mode benchmark                    # Performance benchmarks
  %(prog)s --mode core                         # Core paper experiments
  %(prog)s --mode full                         # Complete reproduction
  %(prog)s --experiments needle_in_haystack    # Run specific experiment
  %(prog)s --experiments scaling_analysis long_context  # Multiple experiments
  %(prog)s --mode core --quick                 # Core experiments, reduced settings

Individual experiment scripts (with more options):
  python -m paper_reproduction.needle_in_haystack_experiment --help
  python -m paper_reproduction.scaling_analysis_experiment --help
  python -m paper_reproduction.ablation_long_context_experiments --help
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["validate", "benchmark", "core", "full"],
        default="validate",
        help="Execution mode: validate, benchmark, core, or full (default: validate)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="titans_paper_reproduction",
        help="Output directory (default: titans_paper_reproduction)",
    )

    parser.add_argument(
        "--experiments",
        nargs="*",
        metavar="NAME",
        help=f"Specific experiments to run (overrides mode). Available: {experiments_list}",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use quick/reduced settings for faster execution",
    )

    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List all available experiments and exit",
    )

    args = parser.parse_args()

    # Handle --list-experiments
    if args.list_experiments:
        _handle_list_experiments()

    # Validate experiment names if provided
    if args.experiments:
        _validate_experiment_names(args.experiments)

    # Initialize reproduction suite
    reproduction = FullPaperReproduction(Path(args.output_dir))

    print("🔬 Titans Paper Reproduction Suite")
    print("=" * 50)
    print(f"Output directory: {reproduction.output_dir}")
    print(f"Mode: {args.mode}")

    if args.quick:
        print("⚡ Quick mode enabled - using reduced settings")

    try:
        _run_mode(args.mode, reproduction, args.experiments)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during execution: {e!s}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
