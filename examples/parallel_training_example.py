"""Example usage of the parallel training algorithm for Titans neural memory.

This script demonstrates how to use the Section 3.2 parallel training algorithm
as an optional component in the Titans architecture.
"""

import time
import traceback

from flax import nnx
import jax
import jax.numpy as jnp

from titans.config import get_config
from titans.enhanced_neural_memory import (
    EnhancedNeuralLongTermMemory,
    ParallelTrainingConfig,
    create_parallel_config,
)


def _run_trials(
    memory_factory: callable,
    test_input: jax.Array,
    num_trials: int,
    memory_depth: int,
    mode_name: str,
) -> float:
    """Run benchmark trials and return average time."""
    times = []
    for trial in range(num_trials):
        try:
            memory = memory_factory(trial)
            start_time = time.time()
            output, memory_state = memory(test_input, deterministic=False)
            jax.block_until_ready(output)
            times.append(time.time() - start_time)
            assert output.shape == test_input.shape
            assert memory_state is not None
            assert len(memory_state.weights) == memory_depth
        except Exception as e:
            print(f"    Warning: {mode_name} trial {trial} failed: {e}")
            times.append(float("inf"))

    valid_times = [t for t in times if t != float("inf")]
    return sum(valid_times) / len(valid_times) if valid_times else float("inf")


def _report_benchmark_results(
    avg_sequential_time: float, avg_parallel_time: float
) -> None:
    """Report benchmark results with speedup calculation."""
    if avg_sequential_time == float("inf") or avg_parallel_time == float("inf"):
        print("  ⚠️  Could not complete benchmark (some trials failed)")
        return

    speedup = avg_sequential_time / avg_parallel_time
    print(f"  Sequential: {avg_sequential_time:.4f}s")
    print(f"  Parallel:   {avg_parallel_time:.4f}s")
    print(f"  Speedup:    {speedup:.2f}x")
    if speedup > 1.0:
        print("  ✅ Parallel training is faster")
    else:
        print("  ⚠️  Sequential training is faster (overhead may dominate)")


def benchmark_parallel_vs_sequential() -> dict[str, dict[str, list[float]]]:
    """Benchmark parallel vs sequential training modes with proper error handling."""
    print("🔬 Benchmarking Parallel vs Sequential Training")
    print("=" * 60)

    # Test parameters
    d_model = 256
    memory_depth = 3
    batch_size = 4
    seq_lengths = [128, 256, 512, 1024]
    num_trials = 3

    results: dict[str, dict[str, list[float]]] = {
        "sequential": {"times": [], "seq_lens": []},
        "parallel": {"times": [], "seq_lens": []},
    }

    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        key = jax.random.key(42)
        test_input = jax.random.normal(key, (batch_size, seq_len, d_model))

        # Sequential training factory
        def make_sequential(trial: int) -> EnhancedNeuralLongTermMemory:
            return EnhancedNeuralLongTermMemory(
                d_model=d_model,
                memory_depth=memory_depth,
                parallel_config=ParallelTrainingConfig(enabled=False),
                rngs=nnx.Rngs(42 + trial),
            )

        avg_sequential_time = _run_trials(
            make_sequential, test_input, num_trials, memory_depth, "Sequential"
        )
        results["sequential"]["times"].append(avg_sequential_time)
        results["sequential"]["seq_lens"].append(seq_len)

        # Parallel training factory - capture seq_len explicitly
        def make_parallel(
            trial: int, *, _seq_len: int = seq_len
        ) -> EnhancedNeuralLongTermMemory:
            parallel_config = create_parallel_config(
                enabled=True,
                chunk_size=min(64, _seq_len // 2),
                chunk_level_params=False,
                min_sequence_length=64,
            )
            return EnhancedNeuralLongTermMemory(
                d_model=d_model,
                memory_depth=memory_depth,
                parallel_config=parallel_config,
                rngs=nnx.Rngs(42 + trial),
            )

        avg_parallel_time = _run_trials(
            make_parallel, test_input, num_trials, memory_depth, "Parallel"
        )
        results["parallel"]["times"].append(avg_parallel_time)
        results["parallel"]["seq_lens"].append(seq_len)

        _report_benchmark_results(avg_sequential_time, avg_parallel_time)

    # Summary
    valid_seq_times = [t for t in results["sequential"]["times"] if t != float("inf")]
    valid_par_times = [t for t in results["parallel"]["times"] if t != float("inf")]

    if valid_seq_times and valid_par_times:
        print("\n📊 Summary")
        print(
            f"Sequential avg time: {sum(valid_seq_times) / len(valid_seq_times):.4f}s"
        )
        print(
            f"Parallel avg time:   {sum(valid_par_times) / len(valid_par_times):.4f}s"
        )

    return results


def demo_parallel_configuration() -> None:
    """Demonstrate different parallel training configurations with error handling."""
    print("\n🛠️  Demonstrating Parallel Training Configurations")
    print("=" * 60)

    d_model = 128
    seq_len = 512
    batch_size = 2

    # Create test data
    key = jax.random.key(42)
    test_input = jax.random.normal(key, (batch_size, seq_len, d_model))

    configs: list[tuple[str, ParallelTrainingConfig]] = [
        ("Disabled (Sequential)", create_parallel_config(enabled=False)),
        (
            "Small Chunks (32)",
            create_parallel_config(
                enabled=True, chunk_size=32, chunk_level_params=False
            ),
        ),
        (
            "Medium Chunks (64)",
            create_parallel_config(
                enabled=True, chunk_size=64, chunk_level_params=False
            ),
        ),
        (
            "Large Chunks (128)",
            create_parallel_config(
                enabled=True, chunk_size=128, chunk_level_params=False
            ),
        ),
    ]

    for config_name, parallel_config in configs:
        print(f"\n Testing: {config_name}")

        try:
            # Create memory with this configuration
            memory = EnhancedNeuralLongTermMemory(
                d_model=d_model,
                memory_depth=2,
                parallel_config=parallel_config,
                rngs=nnx.Rngs(42),
            )

            # Run forward pass
            start_time = time.time()
            output, memory_state = memory(test_input, deterministic=False)
            jax.block_until_ready(output)
            elapsed_time = time.time() - start_time

            # Check memory evolution
            if memory_state and memory_state.weights:
                total_memory_norm = sum(
                    float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                )
                memory_layers = len(memory_state.weights)
            else:
                total_memory_norm = 0.0
                memory_layers = 0

            print("  ✅ Success!")
            print(f"  Time: {elapsed_time:.4f}s")
            print(f"  Output shape: {output.shape}")
            print(f"  Memory norm: {total_memory_norm:.4f}")
            print(f"  Memory layers: {memory_layers}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")


def demo_titans_with_parallel_training() -> None:
    """Demonstrate using parallel training in full Titans models."""
    print("\n🏗️  Titans Models with Parallel Training")
    print("=" * 60)

    # Create enhanced Titans configuration
    config = get_config("medium")
    config.architecture.d_model = 256

    # Enable parallel training for longer sequences
    parallel_config = create_parallel_config(
        enabled=True,
        chunk_size=64,
        min_sequence_length=256,
    )

    print(f"Configuration: {config.variant} variant")
    print(f"Model dimension: {config.architecture.d_model}")
    print(f"Parallel training: {'Enabled' if parallel_config.enabled else 'Disabled'}")
    print(f"Chunk size: {parallel_config.chunk_size}")

    # Create test scenarios: (name, seq_len, should_be_parallel)
    test_cases: list[tuple[str, int, bool]] = [
        ("Short sequence (128)", 128, False),
        ("Medium sequence (512)", 512, True),
        ("Long sequence (1024)", 1024, True),
    ]

    for test_name, seq_len, should_be_parallel in test_cases:
        print(f"\n  {test_name}:")

        try:
            # Create enhanced neural memory (this would be integrated into Titans)
            memory = EnhancedNeuralLongTermMemory(
                d_model=config.architecture.d_model,
                memory_depth=config.memory.memory_depth,
                parallel_config=parallel_config,
                rngs=nnx.Rngs(42),
            )

            # Create test input
            test_input = jax.random.normal(
                jax.random.key(42),
                (2, seq_len, config.architecture.d_model),
            )

            # Test with deterministic=False to trigger memory updates
            start_time = time.time()
            output, memory_state = memory(test_input, deterministic=False)
            jax.block_until_ready(output)
            elapsed_time = time.time() - start_time

            expected_parallel = (
                parallel_config.enabled
                and seq_len >= parallel_config.min_sequence_length
            )

            print("    ✅ Success!")
            print(
                f"    Expected parallel: {expected_parallel} (should be: {should_be_parallel})"
            )
            print(f"    Time: {elapsed_time:.4f}s")
            print(f"    Memory updated: {memory_state}")
            if memory_state:
                print(f"    Memory layers: {len(memory_state.weights)}")

        except Exception as e:
            print(f"    ❌ Failed: {e}")


def demonstrate_memory_evolution() -> None:
    """Demonstrate how memory evolves with parallel training."""
    print("\n🧠 Memory Evolution with Parallel Training")
    print("=" * 60)

    d_model = 128
    memory_depth = 2
    seq_len = 384

    # Create memory with parallel training
    parallel_config = create_parallel_config(
        enabled=True,
        chunk_size=32,
        min_sequence_length=64,
    )

    try:
        memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            parallel_config=parallel_config,
            rngs=nnx.Rngs(42),
        )

        # Track memory evolution over multiple sequences
        memory_norms = []
        memory_state = None

        print("Processing sequences and tracking memory evolution...")

        for seq_idx in range(5):
            try:
                # Create different input for each sequence
                key = jax.random.key(42 + seq_idx)
                test_input = jax.random.normal(key, (1, seq_len, d_model))

                # Process sequence
                _output, memory_state = memory(
                    test_input, memory_state=memory_state, deterministic=False
                )

                # Calculate memory magnitude
                if memory_state and memory_state.weights:
                    total_norm = sum(
                        float(jnp.sum(jnp.abs(w))) for w in memory_state.weights
                    )
                else:
                    total_norm = 0.0

                memory_norms.append(total_norm)

                print(f"  Sequence {seq_idx + 1}: Memory norm = {total_norm:.4f}")

            except Exception as e:
                print(f"  ❌ Sequence {seq_idx + 1} failed: {e}")
                memory_norms.append(0.0)

        if memory_norms and any(norm > 0 for norm in memory_norms):
            print(f"\nMemory evolution: {[f'{norm:.4f}' for norm in memory_norms]}")
            if memory_norms[0] > 0:
                growth = memory_norms[-1] / memory_norms[0]
                print(f"Growth trend: {growth:.2f}x change")
            else:
                print("Growth trend: Unable to calculate (starting from zero)")
        else:
            print("⚠️  No memory evolution detected - check implementation")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def test_basic_functionality() -> None:
    """Test basic functionality of the enhanced neural memory."""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 60)

    # Test basic initialization and forward pass
    d_model = 64
    memory_depth = 2
    seq_len = 128
    batch_size = 2

    try:
        # Test sequential mode
        print("Testing sequential mode...")
        sequential_memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            parallel_config=create_parallel_config(enabled=False),
            rngs=nnx.Rngs(42),
        )

        test_input = jax.random.normal(
            jax.random.key(123), (batch_size, seq_len, d_model)
        )

        # Test deterministic mode
        output1, _state1 = sequential_memory(test_input, deterministic=True)
        output2, _state2 = sequential_memory(test_input, deterministic=True)

        assert jnp.allclose(output1, output2), (
            "Deterministic outputs should be identical"
        )
        print("  ✅ Deterministic mode works")

        # Test non-deterministic mode
        output3, state3 = sequential_memory(test_input, deterministic=False)
        output4, _state4 = sequential_memory(test_input, state3, deterministic=False)

        assert not jnp.allclose(output3, output4), (
            "Non-deterministic outputs should differ"
        )
        print("  ✅ Non-deterministic mode works")

        # Test parallel mode
        print("Testing parallel mode...")
        parallel_memory = EnhancedNeuralLongTermMemory(
            d_model=d_model,
            memory_depth=memory_depth,
            parallel_config=create_parallel_config(enabled=True, chunk_size=32),
            rngs=nnx.Rngs(42),
        )

        output5, state5 = parallel_memory(test_input, deterministic=False)
        assert output5.shape == test_input.shape, "Output shape should match input"
        assert state5 is not None, "Memory state should be updated"
        assert len(state5.weights) == memory_depth, (
            "Should have correct number of weight layers"
        )
        print("  ✅ Parallel mode works")

        print("🎉 All basic functionality tests passed!")

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()


def main() -> None:
    """Run all parallel training demonstrations with extensive error handling."""
    print("🚀 Titans Parallel Training Algorithm Demo (Fixed Version)")
    print("Based on Section 3.2: Tensorized Mini-batch Gradient Descent")
    print("=" * 80)

    try:
        # Test basic functionality first
        test_basic_functionality()

        # Benchmark performance
        print("\n" + "=" * 60)
        benchmark_parallel_vs_sequential()

        # Demonstrate configurations
        print("\n" + "=" * 60)
        demo_parallel_configuration()

        # Show integration with Titans
        print("\n" + "=" * 60)
        demo_titans_with_parallel_training()

        # Demonstrate memory evolution
        print("\n" + "=" * 60)
        demonstrate_memory_evolution()

        print("\n✅ All demonstrations completed!")

        # Summary recommendations
        print("\n💡 Recommendations:")
        print("  - Use parallel training for sequences longer than 256 tokens")
        print("  - Start with chunk_size=64 for balanced performance")
        print("  - Monitor memory evolution to ensure stable learning")
        print("  - Use sequential training as fallback for short sequences")
        print("  - The implementation includes automatic error recovery")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        traceback.print_exc()

        print("\n🔧 Troubleshooting:")
        print("  - Ensure you have the latest version of JAX installed")
        print("  - Check that the enhanced_neural_memory module is properly imported")
        print(
            "  - Try reducing sequence lengths or model dimensions if you encounter memory issues"
        )


if __name__ == "__main__":
    main()
