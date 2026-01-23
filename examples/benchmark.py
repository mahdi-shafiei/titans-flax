"""Test for Titans benchmarking functionality.

This test demonstrates how to use the benchmarking capabilities
of the Titans library by running a full benchmark.
"""

from typing import Any

from titans.benchmark import TitansBenchmark


def run_full_benchmark() -> dict[str, Any]:
    """Run a full benchmark of all Titans variants."""
    benchmark = TitansBenchmark()

    # 1. Compare variants
    comparison_results = benchmark.compare_variants(
        variants=["mac", "mag", "mal", "lmm"],
        seq_len=512,
        batch_size=4,
        num_trials=5,
    )

    benchmark.save_results(comparison_results, "variant_comparison.json")
    benchmark.plot_comparison(
        comparison_results, save_path=benchmark.output_dir / "variant_comparison.png"
    )

    # 2. Scaling analysis
    scaling_results = benchmark.scaling_analysis(
        variant="mac",
        sequence_lengths=[128, 256, 512, 1024],
        num_trials=3,
    )

    benchmark.save_results(scaling_results, "scaling_analysis.json")
    benchmark.plot_scaling(
        scaling_results, save_path=benchmark.output_dir / "scaling_analysis.png"
    )

    # 3. Memory capacity analysis
    capacity_results = benchmark.memory_capacity_analysis(
        variant="mac",
        num_sequences=20,
        memory_depths=[1, 2, 3, 4],
    )

    benchmark.save_results(capacity_results, "memory_capacity.json")

    return {
        "comparison": comparison_results,
        "scaling": scaling_results,
        "capacity": capacity_results,
    }


def test_full_benchmark() -> None:
    """Test the full benchmark functionality."""
    # Run full benchmark
    results = run_full_benchmark()

    # Basic validation that results were generated
    assert "comparison" in results
    assert "scaling" in results
    assert "capacity" in results

    # Verify comparison results have expected variants
    comparison_results = results["comparison"]
    expected_variants = ["mac", "mag", "mal", "lmm"]
    for variant in expected_variants:
        assert variant in comparison_results

    print("✓ Full benchmark test completed successfully!")
    print(f"✓ Generated comparison results for {len(comparison_results)} variants")
    print(f"✓ Generated scaling results for {len(results['scaling'])} sequence lengths")
    print(f"✓ Generated capacity results for {len(results['capacity'])} memory depths")


if __name__ == "__main__":
    test_full_benchmark()
