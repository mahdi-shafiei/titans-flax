#!/bin/bash

# Quick activation script for Titans-Flax environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run './setup_env.sh' first to create the environment."
    exit 1
fi

echo "⚡ Activating Titans-Flax environment..."
source venv/bin/activate

echo "✅ Environment activated!"
echo "🔍 Current Python: $(which python3)"
echo "📦 Available packages:"
pip list | grep -E "(jax|flax|titans)"

echo ""
echo "🚀 Ready to use Titans-Flax!"
echo "Try: python3 examples/quick_start.py"
