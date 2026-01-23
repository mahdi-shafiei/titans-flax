# Development commands
# Titans Development Makefile
#
# This Makefile provides convenient commands for common development tasks.
# Run 'make help' to see all available commands.

.PHONY: help setup install install-dev install-gpu install-tpu clean lint format type-check test test-fast test-integration test-coverage docs build upload check-all demo benchmark profile docs-install docs-serve docs-build docs-deploy docs-check docs-clean

# Configuration
PYTHON := python
PIP := pip
PYTEST := pytest
PACKAGE_NAME := titans
SRC_DIR := $(PACKAGE_NAME)
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
BOLD := \033[1m
NC := \033[0m # No Color

# Default target
help: ## Show this help message
	@echo "$(BOLD)Titans Development Makefile$(NC)"
	@echo "================================"
	@echo ""
	@echo "$(BOLD)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)Examples:$(NC)"
	@echo "  make setup          # Set up development environment"
	@echo "  make test           # Run all tests"
	@echo "  make format         # Format code"
	@echo "  make check-all      # Run all quality checks"
	@echo "  make demo           # Run demo example"

# Environment setup
setup: ## Set up complete development environment
	@echo "$(YELLOW)Setting up development environment...$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -e ".[dev,research]"
	@if command -v pre-commit >/dev/null 2>&1; then \
		pre-commit install; \
		echo "$(GREEN)Pre-commit hooks installed$(NC)"; \
	else \
		echo "$(YELLOW)Warning: pre-commit not found, skipping hook installation$(NC)"; \
	fi
	@echo "$(GREEN)Development environment ready!$(NC)"

install: ## Install package in normal mode
	@echo "$(YELLOW)Installing package...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)Package installed!$(NC)"

install-dev: ## Install package with development dependencies
	@echo "$(YELLOW)Installing with development dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Development installation complete!$(NC)"

install-gpu: ## Install package with GPU support
	@echo "$(YELLOW)Installing with GPU support...$(NC)"
	$(PIP) install -e ".[gpu]"
	@echo "$(GREEN)GPU installation complete!$(NC)"

install-tpu: ## Install package with TPU support
	@echo "$(YELLOW)Installing with TPU support...$(NC)"
	$(PIP) install -e ".[tpu]"
	@echo "$(GREEN)TPU installation complete!$(NC)"

# Code quality
lint: ## Run linting checks
	@echo "$(YELLOW)Running linting checks...$(NC)"
	@echo "$(BLUE)Checking with ruff...$(NC)"
	ruff $(SRC_DIR)/
	@echo "$(BLUE)Checking import sorting...$(NC)"
	isort --check-only $(SRC_DIR)/
	@echo "$(BLUE)Checking code formatting...$(NC)"
	black --check $(SRC_DIR)/
	@echo "$(GREEN)Linting passed!$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	@echo "$(BLUE)Sorting imports...$(NC)"
	isort $(SRC_DIR)/
	@echo "$(BLUE)Formatting with black...$(NC)"
	black $(SRC_DIR)/
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(YELLOW)Running type checks...$(NC)"
	mypy $(SRC_DIR)/
	@echo "$(GREEN)Type checking passed!$(NC)"

# Testing
test: ## Run all tests
	@echo "$(YELLOW)Running all tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v

test-fast: ## Run fast tests only (exclude slow and integration tests)
	@echo "$(YELLOW)Running fast tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v -m "not slow and not integration"

test-integration: ## Run integration tests only
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v -m "integration"

test-coverage: ## Run tests with coverage report
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR)/ \
		--cov=$(SRC_DIR) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-report=xml \
		-v
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-watch: ## Run tests in watch mode (requires pytest-watch)
	@echo "$(YELLOW)Running tests in watch mode...$(NC)"
	@if command -v ptw >/dev/null 2>&1; then \
		ptw $(TEST_DIR)/ -- -v; \
	else \
		echo "$(RED)Error: pytest-watch not installed. Install with: pip install pytest-watch$(NC)"; \
		exit 1; \
	fi

# Documentation
docs: ## Build documentation (placeholder)
	@echo "$(YELLOW)Building documentation...$(NC)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		echo "$(BLUE)Documentation directory found$(NC)"; \
		echo "$(YELLOW)Note: Add actual documentation build commands here$(NC)"; \
	else \
		echo "$(YELLOW)Creating docs directory...$(NC)"; \
		mkdir -p $(DOCS_DIR); \
	fi
	@echo "$(GREEN)Documentation build complete!$(NC)"

docs-serve: ## Serve documentation locally (placeholder)
	@echo "$(YELLOW)Serving documentation locally...$(NC)"
	@echo "$(YELLOW)Note: Add documentation server commands here$(NC)"

# Package building and distribution
build: ## Build package for distribution
	@echo "$(YELLOW)Building package...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)Package built in dist/$(NC)"

clean: ## Clean build artifacts and cache files
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "$(GREEN)Cleanup complete!$(NC)"

upload-test: ## Upload package to Test PyPI
	@echo "$(YELLOW)Uploading to Test PyPI...$(NC)"
	twine upload --repository testpypi dist/*

upload: ## Upload package to PyPI
	@echo "$(YELLOW)Uploading to PyPI...$(NC)"
	twine upload dist/*

# Quality checks
check-all: lint type-check test ## Run all quality checks
	@echo "$(GREEN)All quality checks passed!$(NC)"

security: ## Run security checks with bandit
	@echo "$(YELLOW)Running security checks...$(NC)"
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r $(SRC_DIR)/ -f json -o bandit-report.json; \
		echo "$(GREEN)Security check complete! Report saved to bandit-report.json$(NC)"; \
	else \
		echo "$(YELLOW)Warning: bandit not installed. Install with: pip install bandit$(NC)"; \
	fi

# Development utilities
demo: ## Run quick demo example
	@echo "$(YELLOW)Running demo example...$(NC)"
	$(PYTHON) -c "import titans; titans.quick_start_example()"

demo-complete: ## Run complete demo with all features
	@echo "$(YELLOW)Running complete demo...$(NC)"
	$(PYTHON) examples/titans_complete_example.py --mode complete --output-dir demo_output

benchmark: ## Run performance benchmarks
	@echo "$(YELLOW)Running performance benchmarks...$(NC)"
	$(PYTHON) -c "from titans.benchmark import run_full_benchmark; run_full_benchmark()"

profile: ## Profile model performance
	@echo "$(YELLOW)Profiling model performance...$(NC)"
	$(PYTHON) -c "\
from titans.optimization_utilities import PerformanceProfiler; \
from titans import create_titans_model; \
profiler = PerformanceProfiler(); \
model = create_titans_model('mac', d_model=128); \
profiler.compare_variants(['mac', 'mag', 'mal', 'lmm'])"

memory-analysis: ## Analyze memory usage
	@echo "$(YELLOW)Analyzing memory usage...$(NC)"
	$(PYTHON) -c "\
from titans.optimization_utilities import MemoryOptimizer; \
from titans import create_titans_model; \
model = create_titans_model('mac', d_model=256); \
profile = MemoryOptimizer.estimate_memory_usage(model, 4, 512); \
print(profile)"

paper-reproduction: ## Run paper reproduction experiments
	@echo "$(YELLOW)Running paper reproduction experiments...$(NC)"
	$(PYTHON) -c "from titans.research_utilities import run_paper_reproduction; run_paper_reproduction()"

# Training examples
train-example: ## Run training example
	@echo "$(YELLOW)Running training example...$(NC)"
	$(PYTHON) training.py

language-model: ## Run language modeling example
	@echo "$(YELLOW)Running language modeling example...$(NC)"
	$(PYTHON) examples/language_modeling_app.py

# Environment info
info: ## Show environment and package information
	@echo "$(BOLD)Environment Information$(NC)"
	@echo "======================="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Working directory: $$(pwd)"
	@echo ""
	@echo "$(BOLD)Installed packages:$(NC)"
	@$(PIP) list | grep -E "(jax|flax|optax|titans)" || echo "Core packages not found"
	@echo ""
	@echo "$(BOLD)JAX devices:$(NC)"
	@$(PYTHON) -c "import jax; print('Available devices:', jax.devices())" 2>/dev/null || echo "JAX not available"

verify: ## Verify installation and basic functionality
	@echo "$(YELLOW)Verifying installation...$(NC)"
	@echo "$(BLUE)Testing basic imports...$(NC)"
	@$(PYTHON) -c "import titans; print('✓ Titans imported successfully')"
	@echo "$(BLUE)Testing model creation...$(NC)"
	@$(PYTHON) -c "import titans; model = titans.create_titans_model('mac', d_model=64); print('✓ Model creation successful')"
	@echo "$(BLUE)Testing basic inference...$(NC)"
	@$(PYTHON) -c "\
import titans, jax.numpy as jnp; \
model = titans.create_titans_model('mac', d_model=64); \
x = jnp.ones((1, 16, 64)); \
y = model(x); \
print(f'✓ Basic inference successful: {x.shape} -> {y.shape}')"
	@echo "$(GREEN)All verifications passed!$(NC)"

# Development workflow
dev-setup: setup verify ## Complete development setup with verification
	@echo "$(GREEN)Development environment is ready!$(NC)"
	@echo ""
	@echo "$(BOLD)Next steps:$(NC)"
	@echo "  1. Run 'make test' to ensure everything works"
	@echo "  2. Run 'make demo' to see Titans in action"
	@echo "  3. Check 'make help' for more commands"

pre-commit: format lint type-check test-fast ## Run pre-commit checks
	@echo "$(GREEN)Pre-commit checks passed!$(NC)"

ci: check-all ## Run all CI checks locally
	@echo "$(GREEN)All CI checks passed locally!$(NC)"

# Release workflow
prepare-release: clean build ## Prepare package for release
	@echo "$(YELLOW)Preparing release...$(NC)"
	twine check dist/*
	@echo "$(GREEN)Release preparation complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Review dist/ contents"
	@echo "  2. Test with 'make upload-test'"
	@echo "  3. Upload with 'make upload'"

# Debugging helpers
debug-imports: ## Debug import issues
	@echo "$(YELLOW)Debugging imports...$(NC)"
	$(PYTHON) -c "\
import sys; \
print('Python path:'); \
for p in sys.path: print('  ', p); \
print(); \
try: \
    import titans; \
    print('✓ Titans imported from:', titans.__file__); \
    print('✓ Titans version:', titans.__version__); \
except Exception as e: \
    print('✗ Import failed:', e)"

debug-jax: ## Debug JAX installation
	@echo "$(YELLOW)Debugging JAX installation...$(NC)"
	$(PYTHON) -c "\
try: \
    import jax; \
    import jax.numpy as jnp; \
    print('✓ JAX version:', jax.__version__); \
    print('✓ JAX devices:', jax.devices()); \
    x = jnp.array([1, 2, 3]); \
    print('✓ JAX operations work:', x + 1); \
except Exception as e: \
    print('✗ JAX issue:', e)"

# Performance testing
perf-test: ## Run performance tests
	@echo "$(YELLOW)Running performance tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v -m "performance" --tb=short

stress-test: ## Run stress tests with large models
	@echo "$(YELLOW)Running stress tests...$(NC)"
	$(PYTHON) -c "\
from titans import create_titans_model; \
import jax.numpy as jnp; \
import time; \
print('Creating large model...'); \
model = create_titans_model('mac', d_model=512); \
x = jnp.ones((4, 1024, 512)); \
print(f'Processing large input: {x.shape}'); \
start = time.time(); \
y = model(x); \
elapsed = time.time() - start; \
print(f'✓ Stress test passed: {elapsed:.2f}s for {x.size:,} elements')"

# Help for specific topics
help-testing: ## Show testing help
	@echo "$(BOLD)Testing Commands$(NC)"
	@echo "==============="
	@echo "  make test           - Run all tests"
	@echo "  make test-fast      - Run fast tests only"
	@echo "  make test-integration - Run integration tests"
	@echo "  make test-coverage  - Run tests with coverage"
	@echo "  make test-watch     - Run tests in watch mode"
	@echo ""
	@echo "$(BOLD)Test Markers:$(NC)"
	@echo "  @pytest.mark.slow         - Slow tests (>30s)"
	@echo "  @pytest.mark.integration  - Integration tests"
	@echo "  @pytest.mark.gpu          - GPU-required tests"

help-formatting: ## Show formatting help
	@echo "$(BOLD)Code Formatting$(NC)"
	@echo "==============="
	@echo "  make format     - Format code with black and isort"
	@echo "  make lint       - Check code formatting and style"
	@echo "  make type-check - Run type checking with mypy"
	@echo ""
	@echo "$(BOLD)Tools Used:$(NC)"
	@echo "  black     - Code formatting"
	@echo "  isort     - Import sorting"
	@echo "  ruff      - Fast linting"
	@echo "  mypy      - Type checking"

# Version management
version: ## Show current version
	@$(PYTHON) -c "import titans; print('Titans version:', titans.__version__)"

bump-version: ## Bump version (requires bump2version)
	@if command -v bump2version >/dev/null 2>&1; then \
		bump2version patch; \
	else \
		echo "$(RED)Error: bump2version not installed$(NC)"; \
		echo "Install with: pip install bump2version"; \
	fi

# Cleanup specific artifacts
clean-cache: ## Clean Python cache files only
	@echo "$(YELLOW)Cleaning Python cache...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/ .mypy_cache/

clean-build: ## Clean build artifacts only
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf build/ dist/ *.egg-info/

clean-coverage: ## Clean coverage reports
	@echo "$(YELLOW)Cleaning coverage reports...$(NC)"
	rm -rf .coverage htmlcov/ coverage.xml

# Development status
status: ## Show development status
	@echo "$(BOLD)Development Status$(NC)"
	@echo "=================="
	@git status --porcelain | wc -l | xargs printf "Modified files: %d\n"
	@git log --oneline -5 | head -5
	@echo ""
	@echo "$(BOLD)Recent commits:$(NC)"
	@git log --oneline -3

# Make all targets .PHONY to avoid conflicts with files
.PHONY: $(MAKECMDGOALS)

clean-artifacts: ## Clean all generated artifacts (checkpoints, outputs, etc.)
	@echo "$(YELLOW)Cleaning all generated artifacts...$(NC)"
	rm -rf language_model_checkpoints/ titans_demo_output/ memory_states/
	rm -rf *_checkpoints/ example_output/ demo_results/ exported_model/
	rm -rf paper_reproduction/ benchmark_results/ .benchmarks/
	rm -rf *.json *.png *.pkl profile_output/ performance_results/
	@echo "$(GREEN)Artifacts cleaned!$(NC)"

clean-all: clean clean-artifacts ## Clean everything (build + artifacts)
	@echo "$(GREEN)Complete cleanup finished!$(NC)"

# Documentation commands
docs-install:
	@echo "📦 Installing documentation dependencies..."
	pip install ".[docs]"

docs-serve:
	@echo "🚀 Starting documentation server..."
	@echo "📖 Documentation will be available at: http://localhost:8000"
	mkdocs serve

docs-build:
	@echo "🏗️  Building documentation..."
	mkdocs build

docs-deploy:
	@echo "🚀 Deploying documentation to GitHub Pages..."
	mike deploy --push --update-aliases main latest
	mike set-default --push latest

docs-check:
	@echo "🔍 Checking documentation..."
	mkdocs build --strict

docs-clean:
	@echo "🧹 Cleaning documentation artifacts..."
	rm -rf site/ .cache/
