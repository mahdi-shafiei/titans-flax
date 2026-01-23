#!/usr/bin/env python3
"""Titans-Flax Quick Start Example.

This script demonstrates basic usage of the Titans model for test-time
learning.
"""

import jax.numpy as jnp

# Core imports
from titans.core import create_titans_model


def main() -> None:
    """Run the quick start example."""
    print("🚀 Titans-Flax Quick Start Example")
    print("=" * 50)

    print("\n📦 Creating Titans model...")
    print("   - Variant: MAC (Memory-Augmented Convolution)")
    print("   - Model dimension: 128")
    print("   - Number of heads: 8")
    print("   - Memory depth: 2")

    # Create a Titans model
    model = create_titans_model(
        variant="mac",  # or "mag", "mal", "lmm"
        d_model=128,
        num_heads=8,
        memory_depth=2,
        seed=42,
    )
    print("✅ Model created successfully!")

    print("\n🔢 Preparing input data...")
    # Process input sequence
    x = jnp.ones((2, 64, 128))  # (batch, seq_len, d_model)
    print(f"   - Input shape: {x.shape}")
    print("   - Batch size: 2")
    print("   - Sequence length: 64")
    print("   - Model dimension: 128")

    print("\n🧠 Running forward pass with test-time learning...")
    # Enable test-time learning
    output = model(x, deterministic=False)

    print("✅ Forward pass completed!")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Output dtype: {output.dtype}")
    print(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")

    print("\n🎯 Testing deterministic mode...")
    output_det = model(x, deterministic=True)
    print("✅ Deterministic pass completed!")
    print(f"   - Deterministic output shape: {output_det.shape}")

    # Check if outputs are different (they should be due to test-time learning)
    difference = jnp.mean(jnp.abs(output - output_det))
    print(f"   - Mean absolute difference: {difference:.6f}")

    if difference > 1e-6:
        print("✅ Test-time learning is working! (outputs differ between modes)")
    else:
        print("ℹ️  Outputs are very similar (test-time learning may be minimal)")

    print("\n🎉 Quick start example completed successfully!")
    print("\n💡 Next steps:")
    print("   - Try different model variants: 'mag', 'mal', 'lmm'")
    print("   - Experiment with different model dimensions")
    print("   - Check out the language modeling example:")
    print("     python3 -m examples.language_modeling_app")


if __name__ == "__main__":
    main()
