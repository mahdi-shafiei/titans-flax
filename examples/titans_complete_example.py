"""Complete example demonstrating all Titans functionality.

This example shows:
1. Creating different Titans variants
2. Training with test-time adaptation
3. Saving and loading models
4. Benchmarking performance
5. Using memory state management
"""

import contextlib
from pathlib import Path
import time
from typing import Literal, cast

from flax import nnx
import jax
import jax.numpy as jnp

from titans.benchmark import TitansBenchmark

# Titans imports
# Import all Titans modules
from titans.config import (
    TitansConfig,
    TitansVariant,
    create_config_for_task,
    get_config,
)
from titans.core import create_titans_model
from titans.serialization import (
    ModelExporter,
    TitansCheckpoint,
)


class TitansDemo:
    """Complete demonstration of Titans capabilities."""

    def __init__(self, output_dir: Path | None = None):
        """Initialize demo with output directory."""
        if output_dir is None:
            output_dir = Path("titans_demo_output")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def demo_configuration_management(self) -> None:
        """Demonstrate configuration management capabilities."""
        # Show predefined configurations
        configs = [
            "small",
            "medium",
            "large",
            "long_context",
            "efficient",
            "memory_intensive",
        ]

        for config_name in configs:
            config = get_config(config_name)

        # Task-specific configurations
        tasks: list[
            Literal[
                "language_modeling", "time_series", "long_sequence", "classification"
            ]
        ] = [
            "language_modeling",
            "time_series",
            "long_sequence",
            "classification",
        ]

        for task in tasks:
            config = create_config_for_task(task, "medium")

        # Save and load configuration
        config = get_config("medium")
        config_path = self.output_dir / "example_config.json"
        config.save(config_path)

        TitansConfig.load(config_path)

    def demo_model_variants(self) -> None:
        """Demonstrate different Titans model variants."""
        variants = ["mac", "mag", "mal", "lmm"]
        d_model = 64
        seq_len = 32
        batch_size = 2

        for variant in variants:
            # Create model
            model = create_titans_model(
                variant=cast(TitansVariant, variant), d_model=d_model, seed=42
            )

            # Test input
            x = jnp.ones((batch_size, seq_len, d_model))

            # Forward pass
            start_time = time.time()
            model(x, deterministic=False)
            _elapsed = time.time() - start_time

            # Memory state analysis
            memory_state = model.get_memory_state()
            if memory_state:
                _total_params = sum(w.size for w in memory_state.weights)
            else:
                pass

    def demo_test_time_adaptation(self) -> None:
        """Demonstrate test-time adaptation capabilities."""
        # Create model
        model = create_titans_model(
            variant="mac",
            d_model=128,
            chunk_size=16,
            seed=42,
        )

        # Generate sequence of related inputs
        base_input = jax.random.normal(jax.random.key(0), (1, 32, 128))

        outputs_without_adaptation: list[jax.Array] = []
        outputs_with_adaptation: list[jax.Array] = []
        memory_magnitudes: list[float] = []

        for i in range(5):
            # Create input with increasing similarity to base
            similarity_factor = 0.8 + 0.1 * i  # 0.8 to 1.2
            current_input = base_input * similarity_factor + 0.1 * jax.random.normal(
                jax.random.key(i + 1), base_input.shape
            )

            # Without adaptation (reset memory each time)
            model.reset_memory()
            output_no_adapt = model(current_input, deterministic=True)
            outputs_without_adaptation.append(output_no_adapt)

            # With adaptation (preserve memory)
            if i == 0:
                model.reset_memory()
            output_adapt = model(current_input, deterministic=False)
            outputs_with_adaptation.append(output_adapt)

            # Track memory magnitude
            memory_state = model.get_memory_state()
            if memory_state:
                magnitude = sum(
                    float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                )
                memory_magnitudes.append(magnitude)
            else:
                memory_magnitudes.append(0.0)

        # Compare outputs
        differences = []
        for i in range(len(outputs_without_adaptation)):
            diff = jnp.mean(
                jnp.abs(outputs_with_adaptation[i] - outputs_without_adaptation[i])
            )
            differences.append(float(diff))

    def demo_long_sequence_processing(self) -> None:
        """Demonstrate long sequence processing capabilities."""
        # Test different sequence lengths
        sequence_lengths = [128, 256, 512, 1024]
        d_model = 64

        # Create model optimized for long sequences
        model = create_titans_model(
            variant="mac",  # Best for long sequences
            d_model=d_model,
            chunk_size=64,
            memory_depth=3,
            seed=42,
        )

        processing_times: list[float] = []
        memory_usage: list[float] = []

        for seq_len in sequence_lengths:
            # Generate long sequence
            x = jax.random.normal(jax.random.key(0), (1, seq_len, d_model))

            # Process sequence
            model.reset_memory()
            start_time = time.time()

            model(x, deterministic=False)

            end_time = time.time()
            processing_time = end_time - start_time
            processing_times.append(processing_time)

            # Calculate throughput
            _throughput = seq_len / processing_time

            # Memory analysis
            memory_state = model.get_memory_state()
            if memory_state:
                total_memory = sum(w.size for w in memory_state.weights)
                memory_usage.append(total_memory)

        # Show scaling characteristics
        for i, seq_len in enumerate(sequence_lengths):
            if i > 0:
                time_ratio = processing_times[i] / processing_times[i - 1]
                length_ratio = seq_len / sequence_lengths[i - 1]
                _efficiency = length_ratio / time_ratio

    def demo_memory_analysis(self) -> None:
        """Demonstrate memory capacity and behavior analysis."""
        # Test different memory depths
        memory_depths = [1, 2, 3, 4]
        d_model = 64
        num_sequences = 10

        for depth in memory_depths:
            model = create_titans_model(
                variant="lmm",
                d_model=d_model,
                memory_depth=depth,
                seed=42,  # Pure memory
            )

            memory_history: list[float] = []

            for seq_idx in range(num_sequences):
                # Generate sequence
                x = jax.random.normal(jax.random.key(seq_idx), (1, 16, d_model))

                # Process sequence
                _ = model(x, deterministic=False)

                # Track memory
                memory_state = model.get_memory_state()
                if memory_state:
                    magnitude = sum(
                        float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                    )
                    memory_history.append(magnitude)

                if seq_idx % 3 == 0:
                    pass

            # Memory capacity analysis
            if len(memory_history) >= 3:
                _growth_rate = (memory_history[-1] - memory_history[0]) / len(
                    memory_history
                )
                _recent_change = memory_history[-1] - memory_history[-2]

    def demo_benchmarking(self) -> None:
        """Demonstrate benchmarking capabilities."""
        # Create benchmark suite
        benchmark = TitansBenchmark(self.output_dir / "benchmark_results")

        # Quick benchmark with small models
        comparison_results = benchmark.compare_variants(
            variants=["mac", "mag", "mal", "lmm"],
            seq_len=128,
            batch_size=2,
            num_trials=3,
        )

        # Display results

        for _variant, metrics in comparison_results.items():
            metrics.get_summary()

        # Save results
        benchmark.save_results(comparison_results, "demo_benchmark.json")

        # Create comparison plot
        with contextlib.suppress(Exception):
            benchmark.plot_comparison(
                comparison_results,
                save_path=self.output_dir / "benchmark_comparison.png",
            )

    def demo_checkpointing(self) -> None:
        """Demonstrate checkpointing and serialization."""
        # Create model
        config = get_config("small")
        model = create_titans_model(
            variant=config.variant, d_model=config.architecture.d_model, seed=42
        )

        # Initialize checkpoint manager
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_manager = TitansCheckpoint(checkpoint_dir)

        # Simulate training steps
        for step in [100, 200, 300]:
            # Simulate training
            x = jnp.ones((2, 32, config.architecture.d_model))
            _ = model(x, deterministic=False)

            # Save checkpoint
            metrics = {"loss": 1.0 - step * 0.001, "accuracy": 0.5 + step * 0.001}
            checkpoint_manager.save_model(model, step, config, metrics=metrics)

        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        for _checkpoint in checkpoints:
            pass

        # Load checkpoint
        loaded_model, _loaded_config, _metadata = checkpoint_manager.load_model()

        # Export for inference
        export_path = self.output_dir / "exported_model"
        ModelExporter.export_for_inference(
            loaded_model,
            export_path,
            sample_input_shape=(1, 64, config.architecture.d_model),
        )

        # Load exported model
        ModelExporter.load_for_inference(export_path, nnx.Rngs(42))

    def demo_ablation_study(self) -> dict[int, dict[str, float]]:
        """Demonstrate ablation study capabilities."""
        d_model = 64
        seq_len = 64
        test_input = jnp.ones((1, seq_len, d_model))

        depths = [1, 2, 3, 4]
        results: dict[int, dict[str, float]] = {}

        for depth in depths:
            model = create_titans_model(
                variant="lmm", d_model=d_model, memory_depth=depth, seed=42
            )

            # Test performance
            start_time = time.time()
            model(test_input, deterministic=False)
            forward_time = time.time() - start_time

            # Test adaptation
            adaptation_outputs: list[jax.Array] = []
            for i in range(5):
                varied_input = test_input + 0.1 * jax.random.normal(
                    jax.random.key(i), test_input.shape
                )
                adapt_output = model(varied_input, deterministic=False)
                adaptation_outputs.append(adapt_output)

            # Calculate adaptation variance
            adaptation_variance = float(jnp.var(jnp.stack(adaptation_outputs)))

            results[depth] = {
                "forward_time": forward_time,
                "adaptation_variance": adaptation_variance,
                "parameter_count": model.count_params(),
            }

        for _depth, metrics in results.items():
            # Log efficiency ratio (unused result is intentional for demo)
            _ = metrics["adaptation_variance"] / metrics["forward_time"]

        return results

    def demo_memory_persistence(self) -> None:
        """Show how memory persists across multiple sequences."""
        print("🧠 Testing Memory Persistence...")

        # Create model
        model = create_titans_model(variant="lmm", d_model=64, seed=42)

        # Process multiple sequences and track memory growth
        memory_magnitudes: list[float] = []

        for i in range(5):
            # Create sequence
            x = jax.random.normal(jax.random.key(i), (1, 20, 64))

            # Process sequence (memory should grow/adapt)
            _ = model(x, deterministic=False)

            # Track memory state
            memory_state = model.get_memory_state()
            if memory_state:
                total_magnitude = sum(
                    float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                )
                memory_magnitudes.append(total_magnitude)
                print(f"   Step {i + 1}: Memory magnitude = {total_magnitude:.4f}")
            else:
                print(f"   Step {i + 1}: No memory state available")

        print(f"Memory growth pattern: {memory_magnitudes}")

    def run_complete_demo(self) -> None:
        """Run the complete demonstration."""
        demos = [
            ("Configuration Management", self.demo_configuration_management),
            ("Model Variants", self.demo_model_variants),
            ("Test-Time Adaptation", self.demo_test_time_adaptation),
            ("Long Sequence Processing", self.demo_long_sequence_processing),
            ("Memory Analysis", self.demo_memory_analysis),
            ("Benchmarking", self.demo_benchmarking),
            ("Checkpointing", self.demo_checkpointing),
            ("Ablation Study", self.demo_ablation_study),
            ("Memory Persistence", self.demo_memory_persistence),
        ]

        start_time = time.time()

        for _name, demo_func in demos:
            try:
                demo_func()
            except Exception:  # nosec B112  # Broad catch for demo robustness
                continue

        _total_time = time.time() - start_time

        # List generated files
        for file_path in self.output_dir.rglob("*"):
            if file_path.is_file():
                _relative_path = file_path.relative_to(self.output_dir)


def quick_start_example() -> None:
    """Quick start example for new users."""
    print("🚀 Titans Quick Start Example")
    print("=" * 40)

    # 1. Create a model
    print("1. Creating Titans model (MAC variant, d_model=128)...")
    model = create_titans_model(variant="mac", d_model=128, seed=42)
    print("   ✓ Model created successfully")

    # 2. Process some data
    print("\n2. Processing initial data...")
    x = jnp.ones((2, 64, 128))  # (batch, seq_len, d_model)
    output = model(x, deterministic=False)  # Enable test-time learning
    print(f"   ✓ Input shape: {x.shape}")
    print(f"   ✓ Output shape: {output.shape}")

    # 3. Test adaptation
    print("\n3. Testing test-time adaptation...")
    model.reset_memory()

    memory_magnitudes: list[float] = []
    # Process similar sequences
    for i in range(3):
        x_variant = x + 0.1 * jax.random.normal(jax.random.key(i), x.shape)
        model(x_variant, deterministic=False)

        memory_state = model.get_memory_state()
        if memory_state:
            total_magnitude = sum(
                float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
            )
            memory_magnitudes.append(total_magnitude)
            print(f"   Step {i + 1}: Memory magnitude = {total_magnitude:.4f}")
        else:
            print(f"   Step {i + 1}: No memory state available")

    print("\n4. Adaptation Summary:")
    print(f"   Memory growth: {memory_magnitudes}")
    if len(memory_magnitudes) > 1:
        growth_rate = (
            memory_magnitudes[-1] / memory_magnitudes[0]
            if memory_magnitudes[0] > 0
            else 0
        )
        print(f"   Growth factor: {growth_rate:.2f}x")

    print("\n✅ Quick start example completed successfully!")
    print("   Try running with '--mode complete' for a full demonstration.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Titans Implementation Demo")
    parser.add_argument(
        "--mode",
        choices=["quick", "complete"],
        default="quick",
        help="Demo mode: 'quick' for basic example, 'complete' for full demo",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="titans_demo_output",
        help="Output directory for demo results",
    )

    args = parser.parse_args()

    if args.mode == "quick":
        quick_start_example()
    else:
        demo = TitansDemo(Path(args.output_dir))
        demo.run_complete_demo()
