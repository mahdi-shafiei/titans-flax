# Titans-Flax Setup Guide

This guide will help you set up a proper development environment for Titans-Flax.

## Quick Setup

### 1. Initial Setup (run once)

```bash
# Clone the repository and navigate to it
git clone <your-repo-url>
cd titans-flax

# Run the setup script to create virtual environment
./setup_env.sh
```

This script will:
- ✅ Check Python version (requires 3.9+)
- 📦 Create a virtual environment in `venv/`
- 🎯 Install JAX with CUDA support for GPU
- 📚 Install the project in development mode
- 🛠️ Install development and research dependencies
- 🔍 Verify the installation

### 2. Daily Usage

Each time you want to work with the project:

```bash
# Option 1: Manual activation
source venv/bin/activate

# Option 2: Use the activation script
source ./activate_env.sh
```

### 3. Test the Installation

```bash
# After activating the environment
python3 examples/quick_start.py
```

## What's Included

The setup installs:

- **Core JAX ecosystem**: JAX, JAXlib, Flax, Optax
- **GPU support**: JAX with CUDA 12 for GPU acceleration
- **Development tools**: pytest, black, isort, mypy, pre-commit
- **Research tools**: Jupyter, wandb, matplotlib, seaborn
- **Titans package**: Installed in development mode

## Troubleshooting

### GPU Issues
If you don't have a GPU or want CPU-only installation, modify `setup_env.sh` and replace:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
with:
```bash
pip install --upgrade jax jaxlib
```

### Python Version Issues
Ensure you have Python 3.9+ installed:
```bash
python3 --version
```

### Virtual Environment Issues
If the environment gets corrupted, remove it and recreate:
```bash
rm -rf venv/
./setup_env.sh
```

## Complete Environment Reset

If you want to completely remove all traces of the current environment and start fresh (useful for debugging dependency conflicts or major updates), you have two options:

### Option 1: Automated Script (Recommended)

Use the provided script for an interactive, guided reset process:

```bash
# Run the automated cleanup script
./clean_environment.sh
```

This script will:
- 🛡️ **Ask for confirmation** before making any changes
- 🧹 **Remove all caches and environments** systematically
- 🎨 **Provide colored, informative output**
- ✅ **Verify each step** was completed successfully
- 🚀 **Optionally recreate** the environment automatically
- ⚠️ **Handle errors gracefully** with helpful messages

### Option 2: Manual Commands (Full Control)

If you prefer manual control or the script doesn't work, follow these steps:

### 1. Remove Virtual Environment and Caches

```bash
# Deactivate the current environment if active
deactivate

# Remove the virtual environment directory
rm -rf venv/

# Clear pip cache
pip cache purge

# Clear Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Clear pytest cache
rm -rf .pytest_cache/

# Clear mypy cache
rm -rf .mypy_cache/

# Clear ruff cache
rm -rf .ruff_cache/

# Clear coverage files
rm -f .coverage
rm -rf htmlcov/

# Clear any build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
```

### 2. Clear System-wide Package Manager Caches (Optional)

If you want to ensure completely fresh downloads:

```bash
# Clear apt cache (Ubuntu/Debian)
sudo apt clean

# Clear homebrew cache (macOS)
brew cleanup

# Clear conda cache (if using conda)
conda clean --all

# Clear global pip cache (if different from user cache)
sudo pip cache purge
```

### 3. Create Fresh Environment

```bash
# Create a completely fresh virtual environment
./setup_env.sh

# Verify the installation
source venv/bin/activate
python3 -c "import jax, flax, titans; print('Fresh environment ready!')"
```

### 4. When to Use Complete Reset

Consider a complete environment reset when:
- 🔄 **Dependency conflicts**: Incompatible package versions
- 🚀 **Major updates**: Upgrading Python or JAX versions
- 🐛 **Strange errors**: Unexplained import or runtime issues
- 🧹 **Clean slate**: Starting development on a new machine
- 📦 **Package corruption**: Suspected corrupted installations
- 🔬 **Debugging**: Isolating environment-specific issues

> **Note**: This process will remove all installed packages and require re-downloading everything, which may take several minutes depending on your internet connection.

## Environment Structure

```
titans-flax/
├── venv/                    # Virtual environment (created by setup)
├── setup_env.sh            # Initial setup script
├── activate_env.sh         # Quick activation script
├── clean_environment.sh    # Complete environment reset script
├── examples/
│   └── quick_start.py      # Test the installation
├── titans/                 # Main package
├── tests/                  # Test suite
└── requirements.txt        # Dependencies
```

## Deactivating

To deactivate the virtual environment:
```bash
deactivate
```

## Development Workflow

1. Activate environment: `source venv/bin/activate`
2. Make changes to the code
3. Run tests: `pytest`
4. Format code: `black .`
5. Check types: `mypy titans/`
6. Run examples: `python3 examples/quick_start.py`

## Running Applications

### Language Modeling Application

The repository includes a complete language modeling application that demonstrates training a Titans model for autoregressive text generation:

```bash
# Option 1: Run as module
python3 -m examples.language_modeling_app

# Option 2: Import and run
python3 -c "from examples.language_modeling_app import language_modeling_example; language_modeling_example()"
```
