#!/bin/bash

# Titans-Flax Environment Setup Script
echo "🚀 Setting up Titans-Flax development environment..."

# Find Python 3.11+
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3; do
    if command -v $cmd &> /dev/null; then
        version=$($cmd --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Error: Python 3.11+ is required but not found."
    echo "   Please install Python 3.11 first:"
    echo "   Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "   Or use pyenv: pyenv install 3.11"
    exit 1
fi

python_version=$($PYTHON_CMD --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
echo "✅ Using $PYTHON_CMD (Python $python_version)"

# Create virtual environment
echo "📦 Creating virtual environment with Python $python_version..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install JAX with CUDA support (for GPU)
echo "🎯 Installing JAX with CUDA support..."
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install the package in development mode
echo "📚 Installing titans-flax in development mode..."
pip install -e .

# Install additional development dependencies
echo "🛠️ Installing development tools..."
pip install -e ".[dev,research]"

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import jax
import jax.numpy as jnp
import flax
print(f'✅ JAX version: {jax.__version__}')
print(f'✅ JAX devices: {jax.devices()}')
print(f'✅ Flax version: {flax.__version__}')
try:
    from titans.core import create_titans_model
    print('✅ Titans package imported successfully')
except ImportError as e:
    print(f'❌ Titans import failed: {e}')
"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo "Or use the activation script:"
echo "  source ./activate_env.sh"
echo ""
echo "To deactivate the environment, run:"
echo "  deactivate"
echo ""
echo "To test the installation, run:"
echo "  python3 examples/quick_start.py"
