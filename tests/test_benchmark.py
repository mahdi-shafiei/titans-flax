"""Full tests for Titans benchmarking utilities.

This module tests the benchmarking suite including performance metrics,
variant comparisons, scaling analysis, and report generation functionality.
"""

from collections.abc import Iterator
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from titans.benchmark import (
    BenchmarkResult,
    PerformanceMetrics,
    TitansBenchmark,
)
from titans.config import get_config
from titans.core import TitansModel, TitansVariant, create_titans_model


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_model() -> TitansModel:
    """Create a sample TitansModel for testing."""
    return create_titans_model(
        variant="mac",
        d_model=64,
        chunk_size=32,
        memory_depth=2,
        seed=42,
    )


@pytest.fixture
def sample_input() -> jax.Array:
    """Sample input tensor for benchmarking."""
    return jax.random.normal(jax.random.key(123), (2, 128, 64))


class TestBenchmarkResult:
    """Tests for BenchmarkResult NamedTuple."""

    def test_benchmark_result_creation(self) -> None:
        """Test BenchmarkResult creation and access."""
        result = BenchmarkResult(
            forward_time=0.123,
            memory_usage=45.6,
            throughput=1000.0,
            accuracy=0.85,
            loss=0.234,
        )

        assert result.forward_time == 0.123
        assert result.memory_usage == 45.6
        assert result.throughput == 1000.0
        assert result.accuracy == 0.85
        assert result.loss == 0.234

    def test_benchmark_result_optional_fields(self) -> None:
        """Test BenchmarkResult with optional fields."""
        result = BenchmarkResult(
            forward_time=0.1,
            memory_usage=30.0,
            throughput=500.0,
        )

        assert result.forward_time == 0.1
        assert result.memory_usage == 30.0
        assert result.throughput == 500.0
        assert result.accuracy is None
        assert result.loss is None

    def test_benchmark_result_immutability(self) -> None:
        """Test that BenchmarkResult is immutable."""
        result = BenchmarkResult(
            forward_time=0.1,
            memory_usage=30.0,
            throughput=500.0,
        )

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            result.forward_time = 0.2  # type: ignore


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_default_initialization(self) -> None:
        """Test default PerformanceMetrics initialization."""
        metrics = PerformanceMetrics()

        assert len(metrics.forward_times) == 0
        assert len(metrics.backward_times) == 0
        assert len(metrics.total_times) == 0
        assert metrics.peak_memory == 0.0
        assert len(metrics.memory_usage_history) == 0
        assert len(metrics.tokens_per_second) == 0
        assert len(metrics.samples_per_second) == 0
        assert len(metrics.losses) == 0
        assert len(metrics.accuracies) == 0
        assert len(metrics.memory_state_magnitudes) == 0
        assert len(metrics.memory_update_magnitudes) == 0

    def test_custom_initialization(self) -> None:
        """Test PerformanceMetrics with custom values."""
        metrics = PerformanceMetrics(
            forward_times=[0.1, 0.12, 0.11],
            backward_times=[0.2, 0.22, 0.21],
            peak_memory=100.5,
            tokens_per_second=[1000, 1100, 1050],
        )

        assert len(metrics.forward_times) == 3
        assert len(metrics.backward_times) == 3
        assert metrics.peak_memory == 100.5
        assert len(metrics.tokens_per_second) == 3

    def test_get_summary(self) -> None:
        """Test get_summary method."""
        metrics = PerformanceMetrics(
            forward_times=[0.1, 0.12, 0.11],
            backward_times=[0.2, 0.22, 0.21],
            total_times=[0.3, 0.34, 0.32],
            peak_memory=100.5 * 1024 * 1024,  # In bytes
            tokens_per_second=[1000, 1100, 1050],
            losses=[0.5, 0.4, 0.3],
            accuracies=[0.8, 0.85, 0.9],
            memory_state_magnitudes=[10.0, 12.0, 11.0],
        )

        summary = metrics.get_summary()

        assert "avg_forward_time" in summary
        assert "avg_backward_time" in summary
        assert "avg_total_time" in summary
        assert "peak_memory_mb" in summary
        assert "avg_tokens_per_sec" in summary
        assert "avg_loss" in summary
        assert "avg_accuracy" in summary
        assert "avg_memory_magnitude" in summary

        # Check specific values
        assert abs(summary["avg_forward_time"] - 0.11) < 1e-6
        assert abs(summary["peak_memory_mb"] - 100.5) < 1e-6
        assert abs(summary["avg_tokens_per_sec"] - 1050.0) < 1e-6

    def test_get_summary_empty_lists(self) -> None:
        """Test get_summary with empty lists."""
        metrics = PerformanceMetrics()
        summary = metrics.get_summary()

        assert summary["avg_forward_time"] == 0.0
        assert summary["avg_backward_time"] == 0.0
        assert summary["avg_tokens_per_sec"] == 0.0
        assert summary["avg_loss"] == 0.0


class TestTitansBenchmark:
    """Tests for TitansBenchmark class."""

    def test_benchmark_initialization(self, temp_dir: Path) -> None:
        """Test TitansBenchmark initialization."""
        benchmark = TitansBenchmark(output_dir=temp_dir / "benchmark_results")

        assert benchmark.output_dir.exists()
        assert isinstance(benchmark.results, dict)
        assert len(benchmark.results) == 0

    def test_benchmark_initialization_default_dir(self) -> None:
        """Test TitansBenchmark initialization with default directory."""
        benchmark = TitansBenchmark()

        assert benchmark.output_dir.name == "benchmark_results"
        assert benchmark.output_dir.exists()

    def test_benchmark_single_config(self, temp_dir: Path) -> None:
        """Test benchmarking a single configuration."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Create test config
        config = get_config("small")

        # Benchmark single configuration
        metrics = benchmark._benchmark_single_config(
            variant="mac",
            config=config,
            seq_len=128,
            batch_size=2,
            num_trials=2,  # Small number for testing
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert len(metrics.forward_times) == 2
        assert len(metrics.tokens_per_second) == 2
        assert all(time > 0 for time in metrics.forward_times)
        assert all(tps > 0 for tps in metrics.tokens_per_second)

    def test_benchmark_variant(self, temp_dir: Path) -> None:
        """Test benchmarking a specific variant."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        config = get_config("small")

        # Benchmark MAC variant with small parameters for testing
        variant_results = benchmark.benchmark_variant(
            variant="mac",
            config=config,
            sequence_lengths=[64, 128],
            batch_sizes=[1, 2],
            num_trials=1,  # Single trial for speed
        )

        assert isinstance(variant_results, dict)
        assert len(variant_results) == 4  # 2 seq_lens × 2 batch_sizes

        # Check result structure
        for config_name, metrics in variant_results.items():
            assert isinstance(metrics, PerformanceMetrics)
            assert "mac" in config_name
            assert len(metrics.forward_times) == 1

    def test_compare_variants(self, temp_dir: Path) -> None:
        """Test comparing multiple variants."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Compare subset of variants for testing speed
        comparison_results = benchmark.compare_variants(
            variants=["mac", "mag"],
            seq_len=64,
            batch_size=2,
            num_trials=1,
        )

        assert isinstance(comparison_results, dict)
        assert len(comparison_results) == 2
        assert "mac" in comparison_results
        assert "mag" in comparison_results

        for _variant, metrics in comparison_results.items():
            assert isinstance(metrics, PerformanceMetrics)
            assert len(metrics.forward_times) == 1

    def test_scaling_analysis(self, temp_dir: Path) -> None:
        """Test scaling analysis across sequence lengths."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        scaling_results = benchmark.scaling_analysis(
            variant="mac",
            sequence_lengths=[32, 64],
            d_model=64,
            num_trials=1,
        )

        assert isinstance(scaling_results, dict)
        assert len(scaling_results) == 2
        assert 32 in scaling_results
        assert 64 in scaling_results

        for _seq_len, metrics in scaling_results.items():
            assert isinstance(metrics, PerformanceMetrics)
            assert len(metrics.forward_times) == 1

    def test_memory_capacity_analysis(self, temp_dir: Path) -> None:
        """Test memory capacity analysis."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        capacity_results = benchmark.memory_capacity_analysis(
            variant="mac",
            num_sequences=3,
            seq_len=32,
            memory_depths=[1, 2],
        )

        assert isinstance(capacity_results, dict)
        assert len(capacity_results) == 2
        assert 1 in capacity_results
        assert 2 in capacity_results

        for _depth, memory_history in capacity_results.items():
            assert isinstance(memory_history, list)
            assert len(memory_history) == 3  # num_sequences
            assert all(isinstance(val, int | float) for val in memory_history)

    def test_save_results(self, temp_dir: Path) -> None:
        """Test saving benchmark results to JSON."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Create test results with JAX arrays
        test_results = {
            "variant_performance": {
                "mac": PerformanceMetrics(
                    forward_times=[0.1, 0.12],
                    tokens_per_second=[1000, 1100],
                ),
            },
            "scaling_data": {
                128: [0.1, 0.12],
                256: [0.2, 0.22],
            },
            "jax_array": jnp.array([1, 2, 3]),
        }

        # Save results
        filename = "test_results.json"
        benchmark.save_results(test_results, filename)

        # Check that file was created
        output_file = benchmark.output_dir / filename
        assert output_file.exists()

        # Check that file contains valid JSON
        with open(output_file) as f:
            loaded_data = json.load(f)

        assert isinstance(loaded_data, dict)
        assert "variant_performance" in loaded_data
        assert "scaling_data" in loaded_data
        assert "jax_array" in loaded_data

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_comparison(
        self,
        mock_savefig: Mock,
        mock_show: Mock,
        temp_dir: Path,
    ) -> None:
        """Test plotting comparison results."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Create mock comparison results
        comparison_results: dict[TitansVariant, PerformanceMetrics] = {
            "mac": PerformanceMetrics(
                forward_times=[0.1, 0.12],
                tokens_per_second=[1000, 1100],
                memory_state_magnitudes=[10.0, 12.0],
            ),
            "mag": PerformanceMetrics(
                forward_times=[0.15, 0.17],
                tokens_per_second=[800, 900],
                memory_state_magnitudes=[8.0, 9.0],
            ),
        }

        # Test plotting (should not raise exceptions)
        benchmark.plot_comparison(
            comparison_results, save_path=temp_dir / "comparison.png"
        )

        # Verify that matplotlib functions were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.savefig")
    def test_plot_scaling(
        self,
        mock_savefig: Mock,
        mock_show: Mock,
        temp_dir: Path,
    ) -> None:
        """Test plotting scaling analysis results."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Create mock scaling results
        scaling_results = {
            128: PerformanceMetrics(
                forward_times=[0.1, 0.12],
                tokens_per_second=[1000, 1100],
            ),
            256: PerformanceMetrics(
                forward_times=[0.2, 0.22],
                tokens_per_second=[1200, 1300],
            ),
        }

        # Test plotting
        benchmark.plot_scaling(scaling_results, save_path=temp_dir / "scaling.png")

        # Verify that matplotlib functions were called
        mock_show.assert_called_once()
        mock_savefig.assert_called_once()


class TestBenchmarkingWorkflows:
    """Tests for complete benchmarking workflows."""

    def test_end_to_end_benchmarking_workflow(self, temp_dir: Path) -> None:
        """Test complete end-to-end benchmarking workflow."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Run a minimal benchmarking workflow
        config = get_config("small")

        # 1. Benchmark a single variant
        variant_results = benchmark.benchmark_variant(
            variant="mac",
            config=config,
            sequence_lengths=[32, 64],
            batch_sizes=[1, 2],
            num_trials=1,
        )

        assert len(variant_results) == 4

        # 2. Compare variants
        comparison_results = benchmark.compare_variants(
            variants=["mac", "mag"],
            seq_len=32,
            batch_size=1,
            num_trials=1,
        )

        assert len(comparison_results) == 2

        # 3. Scaling analysis
        scaling_results = benchmark.scaling_analysis(
            variant="mac",
            sequence_lengths=[32, 64],
            num_trials=1,
        )

        assert len(scaling_results) == 2

        # 4. Memory capacity analysis
        capacity_results = benchmark.memory_capacity_analysis(
            variant="mac",
            num_sequences=2,
            memory_depths=[1, 2],
        )

        assert len(capacity_results) == 2

        # 5. Save all results
        all_results = {
            "variant_benchmark": variant_results,
            "variant_comparison": comparison_results,
            "scaling_analysis": scaling_results,
            "memory_capacity": capacity_results,
        }

        benchmark.save_results(all_results, "complete_benchmark.json")

        # Verify file was saved
        assert (benchmark.output_dir / "complete_benchmark.json").exists()

    def test_performance_regression_detection(self, temp_dir: Path) -> None:
        """Test using benchmarks to detect performance regressions."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Simulate baseline performance
        get_config("small")
        baseline_results = benchmark.compare_variants(
            variants=["mac"],
            seq_len=64,
            batch_size=2,
            num_trials=2,
        )

        baseline_time = jnp.mean(jnp.array(baseline_results["mac"].forward_times))
        baseline_throughput = jnp.mean(
            jnp.array(baseline_results["mac"].tokens_per_second)
        )

        # Simulate current performance (should be similar)
        current_results = benchmark.compare_variants(
            variants=["mac"],
            seq_len=64,
            batch_size=2,
            num_trials=2,
        )

        current_time = jnp.mean(jnp.array(current_results["mac"].forward_times))
        current_throughput = jnp.mean(
            jnp.array(current_results["mac"].tokens_per_second)
        )

        # Performance should be in similar range (allowing for variance)
        time_ratio = current_time / baseline_time
        throughput_ratio = current_throughput / baseline_throughput

        # Allow up to 50% variance (benchmarks can be noisy)
        assert 0.5 < time_ratio < 2.0
        assert 0.5 < throughput_ratio < 2.0

    def test_memory_efficiency_comparison(self, temp_dir: Path) -> None:
        """Test comparing memory efficiency across variants."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Compare memory usage for different variants
        variants: list[TitansVariant] = ["mac", "mag", "mal"]
        memory_results: dict[TitansVariant, float] = {}

        for variant in variants:
            capacity_results = benchmark.memory_capacity_analysis(
                variant=variant,
                num_sequences=3,
                seq_len=32,
                memory_depths=[2],
            )

            # Get final memory magnitude
            final_memory = capacity_results[2][-1]  # depth=2, last sequence
            memory_results[variant] = final_memory

        # All variants should have some memory usage
        for _variant, memory_usage in memory_results.items():
            assert memory_usage >= 0.0
            assert jnp.isfinite(memory_usage)

    def test_throughput_scaling_validation(self, temp_dir: Path) -> None:
        """Test validating throughput scaling behavior."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Test scaling with sequence length
        scaling_results = benchmark.scaling_analysis(
            variant="mag",
            sequence_lengths=[32, 64, 128],
            num_trials=1,
        )

        throughputs = []
        for seq_len in [32, 64, 128]:
            throughput = jnp.mean(jnp.array(scaling_results[seq_len].tokens_per_second))
            throughputs.append(throughput)

        # All throughputs should be positive
        assert all(tp > 0 for tp in throughputs)

        # Throughput might decrease with longer sequences (not always linear scaling)
        # Just verify they're all reasonable values
        assert all(100 < tp < 100000 for tp in throughputs)  # Reasonable range


class TestBenchmarkErrorHandling:
    """Tests for error handling and edge cases in benchmarking."""

    def test_invalid_variant_handling(self, temp_dir: Path) -> None:
        """Test handling of invalid variant specifications."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # This should raise an error when trying to create the model
        with pytest.raises(ValueError, match="invalid"):
            benchmark.compare_variants(
                variants=["invalid_variant"],  # type: ignore
                num_trials=1,
            )

    def test_zero_trials_handling(self, temp_dir: Path) -> None:
        """Test handling of zero trials."""
        benchmark = TitansBenchmark(output_dir=temp_dir)
        config = get_config("small")

        # Should handle zero trials gracefully
        metrics = benchmark._benchmark_single_config(
            variant="mac",
            config=config,
            seq_len=32,
            batch_size=1,
            num_trials=0,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert len(metrics.forward_times) == 0

    def test_extreme_sequence_lengths(self, temp_dir: Path) -> None:
        """Test benchmarking with extreme sequence lengths."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Very short sequences
        results_short = benchmark.scaling_analysis(
            variant="lmm",
            sequence_lengths=[1, 2],
            d_model=32,
            num_trials=1,
        )

        assert len(results_short) == 2
        assert all(
            len(metrics.forward_times) == 1 for metrics in results_short.values()
        )

    def test_large_batch_sizes(self, temp_dir: Path) -> None:
        """Test benchmarking with large batch sizes."""
        benchmark = TitansBenchmark(output_dir=temp_dir)
        config = get_config("small")

        # Large batch size (but still reasonable for testing)
        variant_results = benchmark.benchmark_variant(
            variant="lmm",
            config=config,
            sequence_lengths=[16],
            batch_sizes=[16],  # Large batch
            num_trials=1,
        )

        assert len(variant_results) == 1
        config_name = next(iter(variant_results.keys()))
        assert "16" in config_name  # Should contain batch size

    def test_memory_overflow_handling(self, temp_dir: Path) -> None:
        """Test handling of potential memory overflow scenarios."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Test with many sequences (might cause memory buildup)
        capacity_results = benchmark.memory_capacity_analysis(
            variant="mac",
            num_sequences=20,  # Many sequences
            seq_len=16,
            memory_depths=[1],
        )

        # Should complete without crashing
        assert len(capacity_results[1]) == 20
        assert all(jnp.isfinite(val) for val in capacity_results[1])

    def test_concurrent_benchmarking_safety(self, temp_dir: Path) -> None:
        """Test that benchmarking is safe when run concurrently."""
        # This is a simplified test - in practice, you'd need threading
        benchmark1 = TitansBenchmark(output_dir=temp_dir / "bench1")
        benchmark2 = TitansBenchmark(output_dir=temp_dir / "bench2")

        # Both should be able to run independently
        results1 = benchmark1.compare_variants(variants=["mac"], num_trials=1)
        results2 = benchmark2.compare_variants(variants=["mag"], num_trials=1)

        assert "mac" in results1
        assert "mag" in results2


class TestBenchmarkResultAnalysis:
    """Tests for analyzing and interpreting benchmark results."""

    def test_performance_summary_generation(self, temp_dir: Path) -> None:
        """Test generating performance summaries from benchmark results."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Generate comparison results
        comparison_results = benchmark.compare_variants(
            variants=["mac", "mag"],
            seq_len=64,
            batch_size=2,
            num_trials=2,
        )

        # Generate summaries for each variant
        summaries = {}
        for variant, metrics in comparison_results.items():
            summaries[variant] = metrics.get_summary()

        # Verify summaries
        assert len(summaries) == 2
        for _variant, summary in summaries.items():
            assert "avg_forward_time" in summary
            assert "avg_tokens_per_sec" in summary
            assert summary["avg_forward_time"] > 0
            assert summary["avg_tokens_per_sec"] > 0

    def test_scaling_trend_analysis(self, temp_dir: Path) -> None:
        """Test analyzing scaling trends from benchmark results."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Get scaling results
        scaling_results = benchmark.scaling_analysis(
            variant="mac",
            sequence_lengths=[32, 64, 128],
            num_trials=1,
        )

        # Analyze trends
        seq_lengths = sorted(scaling_results.keys())
        times = []
        throughputs = []

        for seq_len in seq_lengths:
            metrics = scaling_results[seq_len]
            avg_time = jnp.mean(jnp.array(metrics.forward_times))
            avg_throughput = jnp.mean(jnp.array(metrics.tokens_per_second))
            times.append(avg_time)
            throughputs.append(avg_throughput)

        # Verify trend analysis
        assert len(times) == 3
        assert len(throughputs) == 3
        assert all(t > 0 for t in times)
        assert all(tp > 0 for tp in throughputs)

        # Time should generally increase with sequence length
        # (though not strictly monotonic due to optimization effects)
        assert times[2] >= times[0]  # 128 should take >= time than 32

    def test_memory_growth_analysis(self, temp_dir: Path) -> None:
        """Test analyzing memory growth patterns."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Test memory capacity with increasing depth
        capacity_results = benchmark.memory_capacity_analysis(
            variant="mac",
            num_sequences=5,
            seq_len=32,
            memory_depths=[1, 2, 3],
        )

        # Analyze memory growth for each depth
        for _depth, memory_history in capacity_results.items():
            assert len(memory_history) == 5

            # All memory values should be finite and positive
            assert all(jnp.isfinite(val) and val > 0 for val in memory_history)

            # Memory should show some evolution (not all identical)
            # Allow for some tolerance due to numerical precision
            memory_values = jnp.array(memory_history)
            memory_variance = jnp.var(memory_values)

            # If variance is very small, all values are essentially the same
            # Otherwise, there should be some meaningful change
            if memory_variance > 1e-6:
                # Memory has changed over sequences - this is expected behavior
                initial_memory = memory_history[0]
                final_memory = memory_history[-1]

                # The absolute change should be reasonable (not extreme)
                relative_change = abs(final_memory - initial_memory) / initial_memory
                assert relative_change < 10.0  # Less than 1000% change

    def test_performance_comparison_metrics(self, temp_dir: Path) -> None:
        """Test computing performance comparison metrics."""
        benchmark = TitansBenchmark(output_dir=temp_dir)

        # Compare two variants
        comparison_results = benchmark.compare_variants(
            variants=["mac", "mag"],
            seq_len=64,
            batch_size=2,
            num_trials=2,
        )

        # Compute relative performance
        mac_metrics = comparison_results["mac"]
        mag_metrics = comparison_results["mag"]

        mac_time = jnp.mean(jnp.array(mac_metrics.forward_times))
        mag_time = jnp.mean(jnp.array(mag_metrics.forward_times))

        mac_throughput = jnp.mean(jnp.array(mac_metrics.tokens_per_second))
        mag_throughput = jnp.mean(jnp.array(mag_metrics.tokens_per_second))

        # Compute speedup ratios
        time_ratio = mac_time / mag_time
        throughput_ratio = mac_throughput / mag_throughput

        # Both ratios should be positive and finite
        assert jnp.isfinite(time_ratio)
        assert time_ratio > 0
        assert jnp.isfinite(throughput_ratio)
        assert throughput_ratio > 0


if __name__ == "__main__":
    pytest.main([__file__])
