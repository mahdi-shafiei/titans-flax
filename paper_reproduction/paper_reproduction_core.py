"""Paper reproduction framework for Titans: Learning to Memorize at Test Time.

This module provides a complete experimental framework for reproducing
the results from the Titans paper, following the critical technical guidelines
and using proper Flax NNX patterns.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import signal
import time
from types import FrameType
from typing import Protocol, TypedDict

import jax
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from titans.config import get_config
from titans.core import TitansModel, TitansVariant, create_titans_model

# Type alias for model configuration parameter values
ConfigValue = int | float | str | bool

# Type alias for serializable values (JSON-compatible)
SerializableValue = (
    None
    | bool
    | int
    | float
    | str
    | list["SerializableValue"]
    | dict[str, "SerializableValue"]
)


# Type definitions for analysis results
class MetricStats(TypedDict):
    """Statistics for a single metric."""

    mean: float
    std: float
    min: float
    max: float


# Type alias for variant analysis results
VariantAnalysis = dict[str, int | float | dict[str, MetricStats] | list[float]]

# Type alias for full analysis results (flexible dict to accommodate various result types)
AnalysisResult = dict[
    str,
    int
    | float
    | list[str | None]
    | dict[str | None, VariantAnalysis]
    | dict[str, MetricStats],
]


class TrialTimeoutError(Exception):
    """Raised when a trial exceeds its timeout."""


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    description: str
    variants: list[TitansVariant] = field(
        default_factory=lambda: ["mac", "mag", "mal", "lmm"]
    )
    d_model: int = 256
    memory_depths: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    sequence_lengths: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    batch_size: int = 4
    num_trials: int = 5
    seed: int = 42
    timeout_seconds: float = 300.0  # 5 minutes per experiment


# Type alias for metric values (numeric, string, or boolean)
MetricValue = float | int | str | bool | jax.Array

# Type alias for metadata values
MetadataValue = float | int | str | bool | list[float] | list[int] | list[str] | None


@dataclass
class ExperimentResult:
    """Results from a single experimental trial."""

    config: ExperimentConfig
    variant: TitansVariant | None
    trial_id: int
    metrics: dict[str, MetricValue]
    metadata: dict[str, MetadataValue]
    execution_time: float
    memory_evolution: list[float] = field(default_factory=list)
    error: str | None = None


class ExperimentProtocol(Protocol):
    """Protocol for experiment implementations."""

    def run_trial(
        self,
        model: TitansModel,
        config: ExperimentConfig,
        trial_id: int,
        **kwargs: ConfigValue,
    ) -> ExperimentResult:
        """Run a single experimental trial."""
        ...


class BaseExperiment:
    """Base class for all paper reproduction experiments."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize base experiment."""
        if output_dir is None:
            output_dir = Path("paper_reproduction_results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def create_model(
        self,
        variant: TitansVariant,
        config: ExperimentConfig,
        **override_params: ConfigValue,
    ) -> TitansModel:
        """Create a model for the experiment."""
        model_config = get_config("medium")
        model_config.variant = variant

        # Use d_model from override_params if provided, otherwise from config
        d_model_value = override_params.pop("d_model", None)
        d_model: int = (
            int(d_model_value) if d_model_value is not None else config.d_model
        )
        model_config.architecture.d_model = d_model

        # Apply any remaining overrides
        for param, value in override_params.items():
            if hasattr(model_config.architecture, param):
                setattr(model_config.architecture, param, value)
            elif hasattr(model_config.memory, param):
                setattr(model_config.memory, param, value)
            elif hasattr(model_config.attention, param):
                setattr(model_config.attention, param, value)

        return create_titans_model(
            variant=variant, d_model=d_model, seed=config.seed, **override_params
        )

    def run_experiment(
        self,
        config: ExperimentConfig,
        experiment_fn: Callable[[TitansModel, ExperimentConfig, int], ExperimentResult],
    ) -> list[ExperimentResult]:
        """Run a complete experiment across all variants and trials."""
        results = []

        logger.info(f"Starting experiment: {config.name}")
        logger.info(f"Description: {config.description}")

        total_trials = len(config.variants) * config.num_trials

        with tqdm(total=total_trials, desc=f"Running {config.name}") as pbar:
            for variant in config.variants:
                logger.info(f"Testing variant: {variant.upper()}")

                for trial in range(config.num_trials):
                    try:
                        # Create fresh model for each trial
                        model = self.create_model(variant, config)

                        # Run trial with timeout
                        result = self._run_trial_with_timeout(
                            experiment_fn, model, config, trial, config.timeout_seconds
                        )
                        result.variant = variant
                        results.append(result)

                        logger.debug(
                            f"Trial {trial + 1}/{config.num_trials} completed for {variant} in {result.execution_time:.2f}s"
                        )

                    except Exception as exc:
                        logger.exception(f"Trial {trial} failed for {variant}")
                        # Create error result
                        error_result = ExperimentResult(
                            config=config,
                            variant=variant,
                            trial_id=trial,
                            metrics={},
                            metadata={"error": str(exc)},
                            execution_time=0.0,
                            error=str(exc),
                        )
                        results.append(error_result)

                    pbar.update(1)

        return results

    def _run_trial_with_timeout(
        self,
        experiment_fn: Callable[[TitansModel, ExperimentConfig, int], ExperimentResult],
        model: TitansModel,
        config: ExperimentConfig,
        trial_id: int,
        timeout_seconds: float,
    ) -> ExperimentResult:
        """Run a trial with timeout protection."""

        def timeout_handler(_signum: int, _frame: FrameType | None) -> None:
            msg = f"Trial exceeded {timeout_seconds} seconds"
            raise TrialTimeoutError(msg)

        # Set up timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))

        try:
            start_time = time.time()
            result = experiment_fn(model, config, trial_id)
            result.execution_time = time.time() - start_time
            return result
        finally:
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)

    def save_results(self, results: list[ExperimentResult], filename: str) -> None:
        """Save experimental results to JSON."""
        output_path = self.output_dir / filename

        # Convert results to serializable format
        serializable_results = []
        for result in results:
            result_dict = {
                "config": {
                    "name": result.config.name,
                    "description": result.config.description,
                    "variants": result.config.variants,
                    "d_model": result.config.d_model,
                    "memory_depths": result.config.memory_depths,
                    "sequence_lengths": result.config.sequence_lengths,
                    "batch_size": result.config.batch_size,
                    "num_trials": result.config.num_trials,
                    "seed": result.config.seed,
                },
                "variant": result.variant,
                "trial_id": result.trial_id,
                "metrics": self._convert_metrics_to_serializable(result.metrics),
                "metadata": self._convert_metrics_to_serializable(result.metadata),
                "execution_time": result.execution_time,
                "memory_evolution": [float(x) for x in result.memory_evolution],
                "error": result.error,
            }
            serializable_results.append(result_dict)

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def _convert_metrics_to_serializable(
        self,
        obj: (
            MetricValue
            | MetadataValue
            | dict[str, MetricValue]
            | dict[str, MetadataValue]
            | list[MetricValue]
            | list[MetadataValue]
        ),
    ) -> SerializableValue:
        """Convert JAX arrays and other types to JSON-serializable format."""
        # Handle array types
        if isinstance(obj, jax.Array):
            return float(obj) if obj.size == 1 else obj.tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle numeric types
        if isinstance(obj, np.integer | np.floating):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        # Handle containers recursively
        if isinstance(obj, dict):
            return {
                str(k): self._convert_metrics_to_serializable(v) for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self._convert_metrics_to_serializable(item) for item in obj]
        # Return primitives as-is, convert unknown types to string
        return (
            obj
            if isinstance(obj, bool | int | float | str) or obj is None
            else str(obj)
        )

    def analyze_results(self, results: list[ExperimentResult]) -> AnalysisResult:
        """Analyze experimental results and compute summary statistics."""
        if not results:
            return {}

        variants_tested: list[str | None] = list({r.variant for r in results})
        analysis: AnalysisResult = {
            "total_trials": len(results),
            "successful_trials": len([r for r in results if r.error is None]),
            "failed_trials": len([r for r in results if r.error is not None]),
            "variants_tested": variants_tested,
        }

        # Analyze by variant
        by_variant: dict[str | None, VariantAnalysis] = {}
        for variant in variants_tested:
            variant_results = [
                r for r in results if r.variant == variant and r.error is None
            ]
            if variant_results:
                by_variant[variant] = self._analyze_variant_results(variant_results)

        analysis["by_variant"] = by_variant

        # Overall statistics
        successful_results = [r for r in results if r.error is None]
        if successful_results:
            execution_times = [r.execution_time for r in successful_results]
            analysis["avg_execution_time"] = float(np.mean(execution_times))
            analysis["total_execution_time"] = float(np.sum(execution_times))

            # Aggregate metrics across all variants (only numeric values)
            all_metrics: dict[str, list[float]] = {}
            for result in successful_results:
                for metric, value in result.metrics.items():
                    # Only aggregate numeric values
                    if isinstance(value, int | float) or (
                        isinstance(value, jax.Array) and value.size == 1
                    ):
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(float(value))

            analysis["aggregate_metrics"] = {
                metric: MetricStats(
                    mean=float(np.mean(values)),
                    std=float(np.std(values)),
                    min=float(np.min(values)),
                    max=float(np.max(values)),
                )
                for metric, values in all_metrics.items()
                if values  # Only include metrics that have values
            }

        return analysis

    def _analyze_variant_results(
        self, results: list[ExperimentResult]
    ) -> VariantAnalysis:
        """Analyze results for a specific variant."""
        if not results:
            return {}

        # Execution time stats
        execution_times = [r.execution_time for r in results]

        # Metric aggregation (only numeric values)
        metrics_aggregated: dict[str, list[float]] = {}
        for result in results:
            for metric, value in result.metrics.items():
                if isinstance(value, int | float) or (
                    isinstance(value, jax.Array) and value.size == 1
                ):
                    if metric not in metrics_aggregated:
                        metrics_aggregated[metric] = []
                    metrics_aggregated[metric].append(float(value))

        # Memory evolution analysis
        memory_evolutions = [r.memory_evolution for r in results if r.memory_evolution]
        avg_memory_evolution: list[float] = []
        if memory_evolutions:
            max_length = max(len(evolution) for evolution in memory_evolutions)
            for i in range(max_length):
                values = [
                    evolution[i]
                    for evolution in memory_evolutions
                    if i < len(evolution)
                ]
                if values:
                    avg_memory_evolution.append(float(np.mean(values)))

        return {
            "num_trials": len(results),
            "avg_execution_time": float(np.mean(execution_times)),
            "std_execution_time": float(np.std(execution_times)),
            "metrics": {
                metric: MetricStats(
                    mean=float(np.mean(values)),
                    std=float(np.std(values)),
                    min=float(np.min(values)),
                    max=float(np.max(values)),
                )
                for metric, values in metrics_aggregated.items()
                if values  # Only include metrics with values
            },
            "memory_evolution": avg_memory_evolution,
        }

    def plot_results(
        self, results: list[ExperimentResult], save_path: Path | None = None
    ) -> None:
        """Create plots for experimental results."""
        if not results:
            logger.warning("No results to plot")
            return

        successful_results = [r for r in results if r.error is None]
        if not successful_results:
            logger.warning("No successful results to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Experiment: {successful_results[0].config.name}", fontsize=16)

        # Plot 1: Performance by variant
        self._plot_performance_by_variant(successful_results, axes[0, 0])

        # Plot 2: Execution time by variant
        self._plot_execution_time_by_variant(successful_results, axes[0, 1])

        # Plot 3: Memory evolution
        self._plot_memory_evolution(successful_results, axes[1, 0])

        # Plot 4: Distribution of key metric
        self._plot_metric_distribution(successful_results, axes[1, 1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Plot saved to: {save_path}")
        else:
            plt.savefig(
                self.output_dir / f"{successful_results[0].config.name}_results.png",
                dpi=300,
                bbox_inches="tight",
            )

        if plt.isinteractive():
            plt.show()
        plt.close()

    def _plot_performance_by_variant(
        self, results: list[ExperimentResult], ax: Axes
    ) -> None:
        """Plot performance metrics by variant."""
        # Find the most common metric to plot
        metric_counts: dict[str, int] = {}
        for result in results:
            for metric in result.metrics:
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

        if not metric_counts:
            ax.text(0.5, 0.5, "No metrics to plot", ha="center", va="center")
            return

        primary_metric = max(metric_counts, key=lambda k: metric_counts[k])

        # Group by variant
        variant_data = {}
        for result in results:
            if primary_metric in result.metrics:
                variant = result.variant
                if variant not in variant_data:
                    variant_data[variant] = []
                variant_data[variant].append(result.metrics[primary_metric])

        if variant_data:
            variants = list(variant_data.keys())
            means = [np.mean(variant_data[v]) for v in variants]
            stds = [np.std(variant_data[v]) for v in variants]

            ax.bar(variants, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_title(f"{primary_metric} by Variant")
            ax.set_ylabel(primary_metric)
            ax.set_xlabel("Variant")

    def _plot_execution_time_by_variant(
        self, results: list[ExperimentResult], ax: Axes
    ) -> None:
        """Plot execution time by variant."""
        variant_times = {}
        for result in results:
            variant = result.variant
            if variant not in variant_times:
                variant_times[variant] = []
            variant_times[variant].append(result.execution_time)

        if variant_times:
            variants = list(variant_times.keys())
            means = [np.mean(variant_times[v]) for v in variants]
            stds = [np.std(variant_times[v]) for v in variants]

            ax.bar(variants, means, yerr=stds, capsize=5, alpha=0.7, color="orange")
            ax.set_title("Execution Time by Variant")
            ax.set_ylabel("Time (seconds)")
            ax.set_xlabel("Variant")

    def _plot_memory_evolution(self, results: list[ExperimentResult], ax: Axes) -> None:
        """Plot memory evolution over time."""
        variant_evolutions: dict[str, list[list[float]]] = {}
        for result in results:
            if result.memory_evolution:
                variant = result.variant or "unknown"
                if variant not in variant_evolutions:
                    variant_evolutions[variant] = []
                variant_evolutions[variant].append(result.memory_evolution)

        # Define consistent colors for each variant (with good visibility)
        variant_colors = {
            "mac": "#E74C3C",  # Red
            "mag": "#F39C12",  # Orange
            "mal": "#27AE60",  # Green
            "lmm": "#3498DB",  # Blue
        }
        # Define marker styles to differentiate overlapping lines
        variant_markers = {"mac": "o", "mag": "s", "mal": "^", "lmm": "D"}
        # Define line styles for additional differentiation
        variant_linestyles = {"mac": "-", "mag": "--", "mal": "-.", "lmm": ":"}
        # Small vertical offsets for overlapping constant lines
        variant_offsets = {"mac": 0, "mag": 0.02, "mal": 0.04, "lmm": 0.06}

        if variant_evolutions:
            # Sort variants for consistent plotting order
            for variant in sorted(variant_evolutions.keys()):
                evolutions = variant_evolutions[variant]
                # Average evolution
                if evolutions:
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
                        # Apply small offset to separate overlapping lines
                        offset_evolution = [v + offset for v in avg_evolution]
                        ax.plot(
                            offset_evolution,
                            label=variant.upper(),
                            marker=marker,
                            color=color,
                            linestyle=linestyle,
                            alpha=0.8,
                            linewidth=2,
                            markersize=6,
                        )

            ax.set_title("Memory Evolution")
            ax.set_ylabel("Memory Magnitude")
            ax.set_xlabel("Step")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No memory evolution data", ha="center", va="center")

    def _plot_metric_distribution(
        self, results: list[ExperimentResult], ax: Axes
    ) -> None:
        """Plot distribution of primary metric."""
        # Find the most common metric
        metric_counts: dict[str, int] = {}
        for result in results:
            for metric in result.metrics:
                metric_counts[metric] = metric_counts.get(metric, 0) + 1

        if not metric_counts:
            ax.text(0.5, 0.5, "No metrics to plot", ha="center", va="center")
            return

        primary_metric = max(metric_counts, key=lambda k: metric_counts[k])

        # Collect values by variant
        variant_values = {}
        for result in results:
            if primary_metric in result.metrics:
                variant = result.variant
                if variant not in variant_values:
                    variant_values[variant] = []
                variant_values[variant].append(result.metrics[primary_metric])

        if variant_values:
            # Create box plot
            data = [variant_values[variant] for variant in variant_values]
            tick_labels = list(variant_values.keys())

            ax.boxplot(data, tick_labels=tick_labels)
            ax.set_title(f"Distribution of {primary_metric}")
            ax.set_ylabel(primary_metric)
            ax.set_xlabel("Variant")


class ExperimentRunner:
    """Main experiment runner for paper reproduction."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize experiment runner."""
        if output_dir is None:
            output_dir = Path("paper_reproduction_results")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up full logging
        log_file = self.output_dir / "experiment_log.txt"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

        self.experiments = {}
        self.results = {}

    def register_experiment(
        self, name: str, experiment_class: type, **kwargs: ConfigValue
    ) -> None:
        """Register an experiment for execution."""
        self.experiments[name] = (experiment_class, kwargs)
        self.logger.info(f"Registered experiment: {name}")

    def run_all_experiments(self) -> dict[str, int | float | str | dict[str, str]]:
        """Run all registered experiments."""
        self.logger.info("Starting paper reproduction experiments")
        start_time = time.time()

        summary_results = {}

        for name, (experiment_class, kwargs) in self.experiments.items():
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Running experiment: {name}")
            self.logger.info(f"{'=' * 60}")

            try:
                # Create experiment instance
                experiment = experiment_class(
                    output_dir=self.output_dir / name, **kwargs
                )

                # Run experiment (implementation specific)
                if hasattr(experiment, "run"):
                    results = experiment.run()
                    self.results[name] = results
                    summary_results[name] = "completed"

                    self.logger.info(f"✅ Experiment {name} completed successfully")
                else:
                    self.logger.error(f"❌ Experiment {name} missing run() method")
                    summary_results[name] = "failed - missing run method"

            except Exception as exc:
                self.logger.exception(f"❌ Experiment {name} failed")
                summary_results[name] = f"failed - {exc!s}"

        total_time = time.time() - start_time

        # Generate overall summary
        summary: dict[str, int | float | str | dict[str, str]] = {
            "total_experiments": len(self.experiments),
            "completed": len([r for r in summary_results.values() if r == "completed"]),
            "failed": len([r for r in summary_results.values() if r != "completed"]),
            "total_execution_time": total_time,
            "results_per_experiment": summary_results,
        }

        # Save summary
        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Total experiments: {summary['total_experiments']}")
        self.logger.info(f"Completed: {summary['completed']}")
        self.logger.info(f"Failed: {summary['failed']}")
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info(f"Summary saved to: {summary_path}")

        return summary
