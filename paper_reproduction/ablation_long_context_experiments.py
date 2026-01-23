"""Ablation Study and Long Context Experiments for Titans Paper Reproduction.

This module implements:
1. Ablation Study - Tests the impact of different architectural components
2. Long Context Experiment - Tests performance on very long sequences
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

# Type alias for simple task metadata
SimpleMetadata = dict[str, object]

# Type alias for complex task metadata (includes lists)
TaskMetadata = dict[str, object]

# Type alias for detailed results
DetailedResult = dict[str, object]


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""

    component_name: str
    description: str
    modification_type: str  # "remove", "replace", "modify"
    baseline_variant: TitansVariant
    modified_params: dict[str, int | float | str | bool]


class AblationStudyExperiment(BaseExperiment):
    """Ablation Study experiment implementation.

    Tests the impact of different architectural components by:
    - Removing memory components
    - Varying memory update rules
    - Testing different attention mechanisms
    - Modifying persistent memory configurations
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize the ablation study experiment."""
        super().__init__(output_dir)

        # Define ablation configurations
        self.ablation_configs = [
            AblationConfig(
                component_name="shallow_memory",
                description="Reduce memory depth to 1",
                modification_type="modify",
                baseline_variant="mac",
                modified_params={"memory_depth": 1},
            ),
            AblationConfig(
                component_name="no_persistent_memory",
                description="Remove persistent memory tokens",
                modification_type="modify",
                baseline_variant="mac",
                modified_params={"num_persistent_tokens": 1},  # Minimum value
            ),
            AblationConfig(
                component_name="large_persistent_memory",
                description="Large persistent memory",
                modification_type="modify",
                baseline_variant="mac",
                modified_params={"num_persistent_tokens": 512},
            ),
            AblationConfig(
                component_name="small_chunks",
                description="Very small chunk size",
                modification_type="modify",
                baseline_variant="mac",
                modified_params={"chunk_size": 32},
            ),
            AblationConfig(
                component_name="large_chunks",
                description="Very large chunk size",
                modification_type="modify",
                baseline_variant="mac",
                modified_params={"chunk_size": 2048},
            ),
            AblationConfig(
                component_name="small_window",
                description="Small attention window",
                modification_type="modify",
                baseline_variant="mag",
                modified_params={"window_size": 32},
            ),
            AblationConfig(
                component_name="large_window",
                description="Large attention window",
                modification_type="modify",
                baseline_variant="mag",
                modified_params={"window_size": 1024},
            ),
        ]

    def _create_ablation_task(
        self,
        seq_len: int,
        d_model: int,
        key: jax.Array,
        task_difficulty: str = "medium",
    ) -> tuple[jax.Array, jax.Array, SimpleMetadata]:
        """Create a task for ablation study evaluation."""
        keys = jax.random.split(key, 4)

        if task_difficulty == "easy":
            # Simple pattern completion
            pattern_len = 8
            num_patterns = seq_len // pattern_len
            base_pattern = jax.random.normal(keys[0], (pattern_len, d_model))

            sequence = jnp.tile(base_pattern, (num_patterns + 1, 1))[:seq_len, :]
            target = sequence  # Autoregressive target

        elif task_difficulty == "medium":
            # Mixed memorization and computation
            # First half: memorization
            mem_len = seq_len // 2
            mem_sequence = jax.random.normal(keys[0], (mem_len, d_model))

            # Second half: transformation of first half
            if mem_len > 0:
                transformed = (
                    mem_sequence * 2.0
                    + jax.random.normal(keys[1], (mem_len, d_model)) * 0.1
                )
                if transformed.shape[0] + mem_len > seq_len:
                    transformed = transformed[: seq_len - mem_len]
                sequence = jnp.concatenate([mem_sequence, transformed], axis=0)
            else:
                sequence = jax.random.normal(keys[0], (seq_len, d_model))

            target = sequence

        else:  # hard
            # Complex associative memory task
            # Create key-value pairs scattered throughout sequence
            sequence = jax.random.normal(keys[0], (seq_len, d_model)) * 0.5

            # Insert key-value associations
            num_pairs = min(seq_len // 20, 10)
            for i in range(num_pairs):
                key_pos = i * (seq_len // num_pairs) + jax.random.randint(
                    keys[1], (), 0, 5
                )
                value_pos = key_pos + 10 + jax.random.randint(keys[2], (), 0, 5)

                if value_pos < seq_len:
                    # Create associated key-value pair
                    key_vector = jax.random.normal(keys[3], (d_model,)) * 2.0
                    value_vector = (
                        key_vector * 0.5 + jax.random.normal(keys[3], (d_model,)) * 0.1
                    )

                    sequence = sequence.at[key_pos].set(key_vector)
                    sequence = sequence.at[value_pos].set(value_vector)

            target = sequence

        # Add batch dimension
        sequence = sequence[None, :, :]
        target = target[None, :, :]

        task_metadata: SimpleMetadata = {
            "task_difficulty": task_difficulty,
            "sequence_length": seq_len,
            "task_type": "ablation_evaluation",
        }

        return sequence, target, task_metadata

    def _evaluate_ablation_performance(
        self,
        model: TitansModel,
        sequence: jax.Array,
        target: jax.Array,
        _metadata: SimpleMetadata,
    ) -> dict[str, float | int | str | bool | jax.Array]:
        """Evaluate model performance for ablation study."""
        # Reset memory
        model.reset_memory()

        # Time forward pass
        start_time = time.time()
        output = model(sequence, deterministic=False)
        forward_time = time.time() - start_time

        # Basic metrics
        mse = float(jnp.mean((output - target) ** 2))
        mae = float(jnp.mean(jnp.abs(output - target)))

        # Relative error
        target_norm = float(jnp.linalg.norm(target))
        relative_error = float(jnp.linalg.norm(output - target) / (target_norm + 1e-8))

        # Memory metrics
        memory_norm = model.get_memory_norm()
        memory_state = model.get_memory_state()

        # Efficiency metrics
        seq_len = sequence.shape[1]
        throughput = seq_len / forward_time if forward_time > 0 else 0

        # Parameter count
        param_count = model.count_params()

        return {
            "mse": mse,
            "mae": mae,
            "relative_error": relative_error,
            "forward_time": forward_time,
            "throughput": throughput,
            "memory_norm": memory_norm,
            "memory_layers": len(memory_state.weights) if memory_state else 0,
            "parameter_count": param_count,
            "efficiency": throughput / (param_count / 1000) if param_count > 0 else 0,
        }

    def run_single_trial(
        self, _model: TitansModel, config: ExperimentConfig, trial_id: int
    ) -> ExperimentResult:
        """Run a single trial of ablation study."""
        key = jax.random.key(config.seed + trial_id)
        keys = jax.random.split(key, len(self.ablation_configs) + 1)

        results = []

        # Test baseline models
        baseline_variants: list[TitansVariant] = ["mac", "mag", "mal", "lmm"]

        for _i, variant in enumerate(baseline_variants):
            try:
                # Create baseline model
                baseline_model = self.create_model(variant, config)

                # Test on multiple difficulties
                for difficulty in ["easy", "medium", "hard"]:
                    task_key = jax.random.split(keys[0], 3)[
                        {"easy": 0, "medium": 1, "hard": 2}[difficulty]
                    ]
                    sequence, target, task_meta = self._create_ablation_task(
                        512, config.d_model, task_key, difficulty
                    )

                    metrics = self._evaluate_ablation_performance(
                        baseline_model, sequence, target, task_meta
                    )

                    metrics.update(
                        {
                            "configuration": f"baseline_{variant}",
                            "variant": variant,
                            "difficulty": difficulty,
                            "is_baseline": True,
                        }
                    )

                    results.append(metrics)

            except Exception as e:
                self.logger.warning(f"Baseline {variant} failed: {e!s}")
                continue

        # Test ablation configurations
        for i, ablation_config in enumerate(self.ablation_configs):
            try:
                # Create modified model
                modified_model = self.create_model(
                    ablation_config.baseline_variant,
                    config,
                    **ablation_config.modified_params,
                )

                # Test on multiple difficulties
                for difficulty in ["easy", "medium", "hard"]:
                    task_key = jax.random.split(keys[i + 1], 3)[
                        {"easy": 0, "medium": 1, "hard": 2}[difficulty]
                    ]
                    sequence, target, task_meta = self._create_ablation_task(
                        512, config.d_model, task_key, difficulty
                    )

                    metrics = self._evaluate_ablation_performance(
                        modified_model, sequence, target, task_meta
                    )

                    metrics.update(
                        {
                            "configuration": ablation_config.component_name,
                            "description": ablation_config.description,
                            "baseline_variant": ablation_config.baseline_variant,
                            "modification_type": ablation_config.modification_type,
                            "difficulty": difficulty,
                            "is_baseline": False,
                        }
                    )

                    results.append(metrics)

            except Exception as e:
                self.logger.warning(
                    f"Ablation {ablation_config.component_name} failed: {e!s}"
                )
                continue

        # Aggregate results
        aggregated_metrics: dict[str, MetricValue] = {}
        metadata: dict[str, MetadataValue] = {
            "ablation_configs": len(self.ablation_configs),
        }

        if results:
            # Compute relative performance compared to baselines
            baseline_results = [r for r in results if r.get("is_baseline", False)]
            ablation_results = [r for r in results if not r.get("is_baseline", False)]

            if baseline_results:
                baseline_mse = float(np.mean([r["mse"] for r in baseline_results]))
                baseline_throughput = float(
                    np.mean([r["throughput"] for r in baseline_results])
                )

                aggregated_metrics["baseline_avg_mse"] = baseline_mse
                aggregated_metrics["baseline_avg_throughput"] = baseline_throughput

                # Compute relative performance for ablations
                if ablation_results:
                    relative_performances: list[float] = []
                    for result in ablation_results:
                        rel_perf = baseline_mse / (float(result["mse"]) + 1e-8)
                        relative_performances.append(rel_perf)

                    aggregated_metrics["avg_relative_performance"] = float(
                        np.mean(relative_performances)
                    )
                    aggregated_metrics["std_relative_performance"] = float(
                        np.std(relative_performances)
                    )

            aggregated_metrics["num_configurations_tested"] = len(
                self.ablation_configs
            ) + len(baseline_variants)
            aggregated_metrics["num_successful_tests"] = len(results)
            aggregated_metrics["avg_mse"] = float(np.mean([r["mse"] for r in results]))
            aggregated_metrics["avg_throughput"] = float(
                np.mean([r["throughput"] for r in results])
            )

            # Store detailed results as serialized JSON string for metadata
            metadata["detailed_results_json"] = json.dumps(results)

        else:
            aggregated_metrics["num_successful_tests"] = 0
            aggregated_metrics["error"] = "all tests failed"

        return ExperimentResult(
            config=config,
            variant=None,  # Multiple variants tested
            trial_id=trial_id,
            metrics=aggregated_metrics,
            metadata=metadata,
            execution_time=0.0,
        )

    def run(self) -> list[ExperimentResult]:
        """Run the complete ablation study."""
        config = ExperimentConfig(
            name="ablation_study",
            description="Test impact of different architectural components",
            variants=["mac"],  # Will test multiple configurations
            d_model=256,
            num_trials=3,
            seed=42,
            timeout_seconds=450.0,
        )

        results = self.run_experiment(config, self.run_single_trial)

        # Save and analyze results
        self.save_results(results, "ablation_study_results.json")
        analysis = self.analyze_results(results)

        with open(self.output_dir / "ablation_study_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        self.plot_results(results, self.output_dir / "ablation_study_plots.png")
        self._create_ablation_plots(results)

        self.logger.info("Ablation Study completed")
        return results

    def _create_ablation_plots(self, results: list[ExperimentResult]) -> None:
        """Create specialized ablation study plots."""
        successful_results = [r for r in results if r.error is None]
        if not successful_results:
            return

        # Extract detailed results (now stored as JSON string)
        all_detailed_results: list[dict[str, float | str | bool]] = []
        for result in successful_results:
            if "detailed_results_json" in result.metadata:
                json_str = result.metadata["detailed_results_json"]
                if isinstance(json_str, str):
                    all_detailed_results.extend(json.loads(json_str))

        if not all_detailed_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Ablation Study Analysis", fontsize=16)

        # Plot 1: Performance comparison
        self._plot_ablation_performance_comparison(all_detailed_results, axes[0, 0])

        # Plot 2: Efficiency analysis
        self._plot_ablation_efficiency(all_detailed_results, axes[0, 1])

        # Plot 3: Memory usage comparison
        self._plot_ablation_memory_usage(all_detailed_results, axes[1, 0])

        # Plot 4: Relative performance by difficulty
        self._plot_ablation_by_difficulty(all_detailed_results, axes[1, 1])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "ablation_specialized_plots.png",
            dpi=300,
            bbox_inches="tight",
        )
        if plt.isinteractive():
            plt.show()
        plt.close()

    def _plot_ablation_performance_comparison(
        self, detailed_results: list[dict], ax: Any
    ) -> None:
        """Plot performance comparison across ablation configurations."""
        # Group by configuration
        config_performance = {}
        for result in detailed_results:
            config = result.get("configuration", "unknown")
            if config not in config_performance:
                config_performance[config] = []
            config_performance[config].append(result["mse"])

        if config_performance:
            configs = list(config_performance.keys())
            means = [np.mean(config_performance[c]) for c in configs]
            stds = [np.std(config_performance[c]) for c in configs]

            x_pos = np.arange(len(configs))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)

            # Color baselines differently
            for _i, (bar, config) in enumerate(zip(bars, configs)):
                if "baseline" in config:
                    bar.set_color("lightblue")
                else:
                    bar.set_color("lightcoral")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.set_ylabel("MSE")
            ax.set_title("Performance by Configuration")
            ax.grid(True, alpha=0.3)

    def _plot_ablation_efficiency(self, detailed_results: list[dict], ax: Any) -> None:
        """Plot efficiency (throughput/params) comparison."""
        config_efficiency = {}
        for result in detailed_results:
            config = result.get("configuration", "unknown")
            if config not in config_efficiency:
                config_efficiency[config] = []
            config_efficiency[config].append(result.get("efficiency", 0))

        if config_efficiency:
            configs = list(config_efficiency.keys())
            means = [np.mean(config_efficiency[c]) for c in configs]

            x_pos = np.arange(len(configs))
            ax.bar(x_pos, means, alpha=0.7, color="lightgreen")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.set_ylabel("Efficiency (throughput/1K params)")
            ax.set_title("Efficiency by Configuration")
            ax.grid(True, alpha=0.3)

    def _plot_ablation_memory_usage(
        self, detailed_results: list[dict], ax: Any
    ) -> None:
        """Plot memory usage comparison."""
        config_memory = {}
        for result in detailed_results:
            config = result.get("configuration", "unknown")
            if config not in config_memory:
                config_memory[config] = []
            config_memory[config].append(result["memory_norm"])

        if config_memory:
            configs = list(config_memory.keys())
            means = [np.mean(config_memory[c]) for c in configs]

            x_pos = np.arange(len(configs))
            ax.bar(x_pos, means, alpha=0.7, color="orange")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(configs, rotation=45, ha="right")
            ax.set_ylabel("Memory Norm")
            ax.set_title("Memory Usage by Configuration")
            ax.grid(True, alpha=0.3)

    def _plot_ablation_by_difficulty(
        self, detailed_results: list[dict], ax: Any
    ) -> None:
        """Plot performance by task difficulty."""
        difficulties = ["easy", "medium", "hard"]

        baseline_by_diff = {d: [] for d in difficulties}
        ablation_by_diff = {d: [] for d in difficulties}

        for result in detailed_results:
            difficulty = result.get("difficulty", "unknown")
            is_baseline = result.get("is_baseline", False)

            if difficulty in difficulties:
                if is_baseline:
                    baseline_by_diff[difficulty].append(result["mse"])
                else:
                    ablation_by_diff[difficulty].append(result["mse"])

        x = np.arange(len(difficulties))
        width = 0.35

        baseline_means = [
            np.mean(baseline_by_diff[d]) if baseline_by_diff[d] else 0
            for d in difficulties
        ]
        ablation_means = [
            np.mean(ablation_by_diff[d]) if ablation_by_diff[d] else 0
            for d in difficulties
        ]

        ax.bar(
            x - width / 2,
            baseline_means,
            width,
            label="Baseline",
            alpha=0.7,
            color="lightblue",
        )
        ax.bar(
            x + width / 2,
            ablation_means,
            width,
            label="Ablations",
            alpha=0.7,
            color="lightcoral",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(difficulties)
        ax.set_ylabel("MSE")
        ax.set_title("Performance by Task Difficulty")
        ax.legend()
        ax.grid(True, alpha=0.3)


class LongContextExperiment(BaseExperiment):
    """Long Context experiment implementation.

    Tests the ability to process very long sequences effectively:
    - Memory efficiency with increasing context length
    - Information retention across long contexts
    - Computational scalability
    """

    def __init__(self, output_dir: Path | None = None):
        """Initialize the long context experiment."""
        super().__init__(output_dir)

        # Context lengths to test (in tokens)
        self.context_lengths = [1024, 2048, 4096, 8192, 16384, 32768]

        # Task types for long context evaluation
        self.long_context_tasks = ["document_qa", "summarization", "retrieval"]

    def _create_long_document(
        self, length: int, d_model: int, key: jax.Array, task_type: str = "document_qa"
    ) -> tuple[jax.Array, jax.Array, TaskMetadata]:
        """Create a long document with embedded questions/tasks."""
        keys = jax.random.split(key, 5)

        if task_type == "document_qa":
            # Create document with question-answer pairs
            document = jax.random.normal(keys[0], (length, d_model)) * 0.3

            # Embed questions and answers throughout
            num_qa_pairs = min(length // 1000, 20)  # One per ~1000 tokens
            qa_positions = []

            for i in range(num_qa_pairs):
                q_pos = i * (length // num_qa_pairs) + jax.random.randint(
                    keys[1], (), 0, 100
                )
                a_pos = q_pos + 50 + jax.random.randint(keys[2], (), 0, 50)

                if a_pos < length:
                    # Question pattern
                    question = jax.random.normal(keys[3], (d_model,)) * 2.0 + 5.0
                    # Answer pattern (related to question)
                    answer = (
                        question * 0.7 + jax.random.normal(keys[4], (d_model,)) * 0.5
                    )

                    document = document.at[q_pos].set(question)
                    document = document.at[a_pos].set(answer)
                    qa_positions.append((q_pos, a_pos))

            target = document  # Autoregressive target
            task_meta: TaskMetadata = {
                "qa_positions": qa_positions,
                "num_qa_pairs": num_qa_pairs,
                "task_type": task_type,
                "context_length": length,
            }

        elif task_type == "summarization":
            # Create document with summary at the end
            # Ensure summary_length evenly divides content_length for reshape
            summary_length = max(1, length // 10)  # ~10% for summary
            content_length = summary_length * (length // summary_length - 1)
            # Adjust to fill the remaining length
            actual_total = content_length + summary_length
            if actual_total < length:
                content_length += length - actual_total

            # Main content
            content = jax.random.normal(keys[0], (content_length, d_model)) * 0.5

            # Summary (compressed representation of content)
            # Reshape content into chunks and take mean of each chunk
            chunk_size = max(1, content_length // summary_length)
            usable_content = content[: chunk_size * summary_length]
            summary = jnp.mean(
                usable_content.reshape(summary_length, chunk_size, d_model), axis=1
            )
            summary = summary + jax.random.normal(keys[1], summary.shape) * 0.1

            document = jnp.concatenate([content, summary], axis=0)
            target = document
            task_meta = {
                "content_length": content_length,
                "summary_length": summary_length,
                "task_type": task_type,
                "context_length": document.shape[0],  # Actual length
            }

        else:  # retrieval
            # Create document with retrieval targets
            document = jax.random.normal(keys[0], (length, d_model)) * 0.3

            # Insert retrievable items
            num_items = min(length // 500, 50)
            retrieval_positions: list[int] = []

            for i in range(num_items):
                pos = jax.random.randint(
                    keys[1],
                    (),
                    i * (length // num_items),
                    (i + 1) * (length // num_items),
                )
                if pos < length:
                    item = jax.random.normal(keys[2], (d_model,)) * 3.0
                    document = document.at[pos].set(item)
                    retrieval_positions.append(int(pos))

            target = document
            task_meta = {
                "retrieval_positions": retrieval_positions,
                "num_items": num_items,
                "task_type": task_type,
                "context_length": length,
            }

        # Add batch dimension
        document = document[None, :, :]
        target = target[None, :, :]

        return document, target, task_meta

    def _process_sequence(
        self,
        model: TitansModel,
        document: jax.Array,
        context_length: int,
        chunk_size: int,
    ) -> tuple[jax.Array, float, bool]:
        """Process document sequence, chunked if necessary."""
        if context_length <= chunk_size:
            start_time = time.time()
            try:
                output = model(document, deterministic=False)
                return output, time.time() - start_time, True
            except Exception as e:
                self.logger.warning(
                    f"Failed to process sequence of length {context_length}: {e!s}"
                )
                return jnp.zeros_like(document), float("inf"), False
        else:
            start_time = time.time()
            try:
                outputs = []
                for i in range(0, context_length, chunk_size):
                    end_idx = min(i + chunk_size, context_length)
                    chunk = document[:, i:end_idx, :]
                    chunk_output = model(chunk, deterministic=False, reset_memory=False)
                    outputs.append(chunk_output)
                output = jnp.concatenate(outputs, axis=1)
                return output, time.time() - start_time, True
            except Exception as e:
                self.logger.warning(
                    f"Failed chunked processing for length {context_length}: {e!s}"
                )
                return jnp.zeros_like(document), float("inf"), False

    def _evaluate_long_context_performance(
        self,
        model: TitansModel,
        document: jax.Array,
        target: jax.Array,
        metadata: TaskMetadata,
    ) -> DetailedResult:
        """Evaluate model performance on long context task."""
        # Extract and type values from metadata
        context_length = int(metadata["context_length"])  # type: ignore[arg-type]
        task_type = str(metadata["task_type"])

        # Reset memory
        model.reset_memory()

        # Process sequence
        output, processing_time, success = self._process_sequence(
            model, document, context_length, chunk_size=1024
        )

        if not success:
            return {
                "success": False,
                "processing_time": processing_time,
                "context_length": context_length,
                "mse": float("inf"),
                "throughput": 0.0,
            }

        # Basic metrics
        mse = float(jnp.mean((output - target) ** 2))
        mae = float(jnp.mean(jnp.abs(output - target)))

        # Context-specific metrics
        task_metrics = {}
        if task_type == "document_qa" and "qa_positions" in metadata:
            # Evaluate QA performance
            qa_errors = []
            for _q_pos, a_pos in metadata["qa_positions"]:
                if a_pos < output.shape[1]:
                    qa_error = float(
                        jnp.mean((output[0, a_pos] - target[0, a_pos]) ** 2)
                    )
                    qa_errors.append(qa_error)

            if qa_errors:
                task_metrics["avg_qa_error"] = np.mean(qa_errors)
                task_metrics["qa_accuracy"] = float(
                    np.mean([e < 1.0 for e in qa_errors])
                )

        elif task_type == "summarization" and "summary_length" in metadata:
            # Evaluate summary quality
            summary_start = int(metadata["content_length"])  # type: ignore[arg-type]
            if summary_start < output.shape[1]:
                summary_output = output[:, summary_start:, :]
                summary_target = target[:, summary_start:, :]
                summary_error = float(jnp.mean((summary_output - summary_target) ** 2))
                task_metrics["summary_error"] = summary_error

        elif task_type == "retrieval" and "retrieval_positions" in metadata:
            # Evaluate retrieval accuracy
            retrieval_errors = []
            ret_positions = list(metadata["retrieval_positions"])  # type: ignore[arg-type]
            for pos in ret_positions:
                pos_int = int(pos)
                if pos_int < output.shape[1]:
                    ret_error = float(
                        jnp.mean((output[0, pos_int] - target[0, pos_int]) ** 2)
                    )
                    retrieval_errors.append(ret_error)

            if retrieval_errors:
                task_metrics["avg_retrieval_error"] = np.mean(retrieval_errors)
                task_metrics["retrieval_accuracy"] = float(
                    np.mean([e < 1.0 for e in retrieval_errors])
                )

        # Efficiency metrics
        throughput = context_length / processing_time if processing_time > 0 else 0
        memory_norm = model.get_memory_norm()

        return {
            "success": success,
            "mse": mse,
            "mae": mae,
            "processing_time": processing_time,
            "throughput": throughput,
            "context_length": context_length,
            "memory_norm": memory_norm,
            "memory_efficiency": memory_norm / context_length
            if context_length > 0
            else 0,
            **task_metrics,
        }

    def run_single_trial(
        self, model: TitansModel, config: ExperimentConfig, trial_id: int
    ) -> ExperimentResult:
        """Run a single trial of long context experiment."""
        key = jax.random.key(config.seed + trial_id)

        results = []
        memory_evolution = []

        # Test each context length
        for context_length in self.context_lengths:
            for task_type in self.long_context_tasks:
                try:
                    task_key = jax.random.split(key, 2)[0]
                    key = jax.random.split(key, 2)[1]

                    # Create long context task
                    document, target, task_meta = self._create_long_document(
                        context_length, config.d_model, task_key, task_type
                    )

                    # Evaluate performance
                    metrics = self._evaluate_long_context_performance(
                        model, document, target, task_meta
                    )

                    metrics.update(
                        {
                            "task_type": task_type,
                            "variant": model.variant,
                            "trial_id": trial_id,
                        }
                    )

                    results.append(metrics)
                    memory_evolution.append(metrics["memory_norm"])

                except Exception as e:
                    self.logger.warning(
                        f"Long context task failed: length={context_length}, task={task_type}, trial={trial_id}: {e!s}"
                    )
                    continue

        # Aggregate results
        aggregated_metrics: dict[str, MetricValue] = {}
        metadata: dict[str, MetadataValue] = {
            "context_lengths_tested": self.context_lengths,
            "task_types_tested": self.long_context_tasks,
        }

        if results:
            successful_results = [r for r in results if r["success"]]

            aggregated_metrics["total_tests"] = len(results)
            aggregated_metrics["successful_tests"] = len(successful_results)
            aggregated_metrics["success_rate"] = len(successful_results) / len(results)

            if successful_results:
                aggregated_metrics["avg_mse"] = float(
                    np.mean([r["mse"] for r in successful_results])
                )
                aggregated_metrics["avg_throughput"] = float(
                    np.mean([r["throughput"] for r in successful_results])
                )
                aggregated_metrics["avg_memory_norm"] = float(
                    np.mean([r["memory_norm"] for r in successful_results])
                )
                aggregated_metrics["max_context_length"] = max(
                    int(r["context_length"]) for r in successful_results
                )
                aggregated_metrics["avg_processing_time"] = float(
                    np.mean([r["processing_time"] for r in successful_results])
                )

            # Store detailed results as JSON string
            metadata["detailed_results_json"] = json.dumps(results)
        else:
            aggregated_metrics["total_tests"] = 0
            aggregated_metrics["error"] = "all tests failed"

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
        """Run the complete long context experiment."""
        config = ExperimentConfig(
            name="long_context",
            description="Test performance on very long sequences",
            variants=["mac", "mag"],  # Focus on variants good for long context
            d_model=256,
            num_trials=2,  # Fewer trials due to computational cost
            seed=42,
            timeout_seconds=900.0,  # Longer timeout for long sequences
        )

        results = self.run_experiment(config, self.run_single_trial)

        # Save and analyze results
        self.save_results(results, "long_context_results.json")
        analysis = self.analyze_results(results)

        with open(self.output_dir / "long_context_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

        self.plot_results(results, self.output_dir / "long_context_plots.png")
        self._create_long_context_plots(results)

        self.logger.info("Long Context experiment completed")
        return results

    def _create_long_context_plots(self, results: list[ExperimentResult]) -> None:
        """Create specialized long context plots."""
        successful_results = [r for r in results if r.error is None]
        if not successful_results:
            return

        # Extract detailed results (stored as JSON string)
        all_detailed_results: list[dict[str, float | int | str | bool]] = []
        for result in successful_results:
            if "detailed_results_json" in result.metadata:
                json_str = result.metadata["detailed_results_json"]
                if isinstance(json_str, str):
                    all_detailed_results.extend(json.loads(json_str))

        if not all_detailed_results:
            return

        # Filter successful results
        successful_detailed = [
            r for r in all_detailed_results if r.get("success", False)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Long Context Analysis", fontsize=16)

        # Plot 1: Throughput vs context length
        self._plot_throughput_scaling(successful_detailed, axes[0, 0])

        # Plot 2: Memory efficiency vs context length
        self._plot_memory_efficiency(successful_detailed, axes[0, 1])

        # Plot 3: Success rate by context length
        self._plot_success_rate(all_detailed_results, axes[1, 0])

        # Plot 4: Processing time scaling
        self._plot_processing_time_scaling(successful_detailed, axes[1, 1])

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "long_context_specialized_plots.png",
            dpi=300,
            bbox_inches="tight",
        )
        if plt.isinteractive():
            plt.show()
        plt.close()

    def _plot_throughput_scaling(self, results: list[dict], ax: Any) -> None:
        """Plot throughput vs context length."""
        variant_data = {}
        for result in results:
            variant = result.get("variant", "unknown")
            if variant not in variant_data:
                variant_data[variant] = {"lengths": [], "throughputs": []}

            variant_data[variant]["lengths"].append(result["context_length"])
            variant_data[variant]["throughputs"].append(result["throughput"])

        for variant, data in variant_data.items():
            if data["lengths"]:
                ax.scatter(
                    data["lengths"],
                    data["throughputs"],
                    label=variant.upper(),
                    alpha=0.7,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput vs Context Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_memory_efficiency(self, results: list[dict], ax: Any) -> None:
        """Plot memory efficiency vs context length."""
        lengths = [r["context_length"] for r in results]
        efficiencies = [r["memory_efficiency"] for r in results]

        ax.scatter(lengths, efficiencies, alpha=0.7, color="green")
        ax.set_xscale("log")
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Memory Efficiency (norm/length)")
        ax.set_title("Memory Efficiency vs Context Length")
        ax.grid(True, alpha=0.3)

    def _plot_success_rate(self, results: list[dict], ax: Any) -> None:
        """Plot success rate by context length."""
        length_success = {}
        for result in results:
            length = result["context_length"]
            if length not in length_success:
                length_success[length] = {"total": 0, "success": 0}

            length_success[length]["total"] += 1
            if result.get("success", False):
                length_success[length]["success"] += 1

        lengths = sorted(length_success.keys())
        success_rates = [
            length_success[seq_len]["success"] / length_success[seq_len]["total"]
            for seq_len in lengths
        ]

        ax.plot(lengths, success_rates, marker="o", linewidth=2, markersize=8)
        ax.set_xscale("log")
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Success Rate")
        ax.set_title("Success Rate vs Context Length")
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

    def _plot_processing_time_scaling(self, results: list[dict], ax: Any) -> None:
        """Plot processing time scaling."""
        lengths = [r["context_length"] for r in results]
        times = [r["processing_time"] for r in results]

        ax.scatter(lengths, times, alpha=0.7, color="red")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Context Length")
        ax.set_ylabel("Processing Time (seconds)")
        ax.set_title("Processing Time vs Context Length")
        ax.grid(True, alpha=0.3)
