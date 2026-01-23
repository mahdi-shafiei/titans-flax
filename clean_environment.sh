#!/bin/bash

# Titans-Flax Environment Reset Script
# This script completely removes the virtual environment and all caches,
# then optionally recreates a fresh environment.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to ask for user confirmation
confirm() {
    while true; do
        read -p "$(echo -e "${YELLOW}$1 (y/n): ${NC}")" yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Main script
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  Titans-Flax Environment Reset              ║"
echo "║                                                              ║"
echo "║  This script will completely remove your virtual            ║"
echo "║  environment and all caches for a fresh start.              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

print_warning "This will permanently delete:"
echo "  • Virtual environment (venv/)"
echo "  • All Python cache files"
echo "  • Pip cache"
echo "  • Tool-specific caches (pytest, mypy, ruff)"
echo "  • Coverage files"
echo "  • Build artifacts"

if ! confirm "Are you sure you want to proceed?"; then
    print_info "Operation cancelled."
    exit 0
fi

# Deactivate virtual environment if active
print_section "Deactivating Environment"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_info "Deactivating current virtual environment..."
    deactivate 2>/dev/null || true
    print_success "Virtual environment deactivated"
else
    print_info "No active virtual environment detected"
fi

# Remove virtual environment
print_section "Removing Virtual Environment"
if [ -d "venv/" ]; then
    print_info "Removing venv/ directory..."
    rm -rf venv/
    print_success "Virtual environment removed"
else
    print_info "No venv/ directory found"
fi

# Clear pip cache
print_section "Clearing Pip Cache"
print_info "Clearing pip cache..."
pip cache purge 2>/dev/null || print_warning "Could not clear pip cache (may not be available)"
print_success "Pip cache cleared"

# Clear Python cache files
print_section "Clearing Python Cache Files"
print_info "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

print_info "Removing .pyc files..."
find . -name "*.pyc" -delete 2>/dev/null || true

print_info "Removing .pyo files..."
find . -name "*.pyo" -delete 2>/dev/null || true

print_success "Python cache files cleared"

# Clear tool-specific caches
print_section "Clearing Tool Caches"

# Pytest cache
if [ -d ".pytest_cache/" ]; then
    print_info "Removing pytest cache..."
    rm -rf .pytest_cache/
    print_success "Pytest cache removed"
fi

# MyPy cache
if [ -d ".mypy_cache/" ]; then
    print_info "Removing mypy cache..."
    rm -rf .mypy_cache/
    print_success "MyPy cache removed"
fi

# Ruff cache
if [ -d ".ruff_cache/" ]; then
    print_info "Removing ruff cache..."
    rm -rf .ruff_cache/
    print_success "Ruff cache removed"
fi

# Clear coverage files
print_section "Clearing Coverage Files"
if [ -f ".coverage" ]; then
    print_info "Removing .coverage file..."
    rm -f .coverage
    print_success "Coverage file removed"
fi

if [ -d "htmlcov/" ]; then
    print_info "Removing htmlcov/ directory..."
    rm -rf htmlcov/
    print_success "HTML coverage directory removed"
fi

# Clear build artifacts
print_section "Clearing Build Artifacts"
for dir in build/ dist/ *.egg-info/; do
    if [ -d "$dir" ] 2>/dev/null; then
        print_info "Removing $dir..."
        rm -rf "$dir"
        print_success "Build artifact $dir removed"
    fi
done

# Optional: Clear system-wide caches
echo ""
if confirm "Do you want to clear system-wide package manager caches? (optional but thorough)"; then
    print_section "Clearing System Caches"

    # Check for different package managers
    if command -v apt >/dev/null 2>&1; then
        print_info "Clearing apt cache..."
        sudo apt clean 2>/dev/null || print_warning "Could not clear apt cache"
    fi

    if command -v brew >/dev/null 2>&1; then
        print_info "Clearing homebrew cache..."
        brew cleanup 2>/dev/null || print_warning "Could not clear homebrew cache"
    fi

    if command -v conda >/dev/null 2>&1; then
        print_info "Clearing conda cache..."
        conda clean --all -y 2>/dev/null || print_warning "Could not clear conda cache"
    fi

    # Global pip cache
    print_info "Clearing global pip cache..."
    sudo pip cache purge 2>/dev/null || print_warning "Could not clear global pip cache"

    print_success "System caches cleared"
fi

print_section "Environment Reset Complete"
print_success "All traces of the previous environment have been removed!"

# Option to recreate environment
echo ""
if confirm "Do you want to create a fresh virtual environment now?"; then
    print_section "Creating Fresh Environment"

    if [ -f "setup_env.sh" ]; then
        print_info "Running setup_env.sh..."
        chmod +x setup_env.sh
        ./setup_env.sh

        print_section "Verifying Installation"
        print_info "Testing the fresh environment..."

        # Test installation
        if source venv/bin/activate && python3 -c "import jax, flax, titans; print('Fresh environment ready!')" 2>/dev/null; then
            print_success "Fresh environment created and verified successfully!"
            echo ""
            print_info "To activate the environment, run:"
            echo "  source venv/bin/activate"
        else
            print_error "There was an issue with the fresh environment setup"
            print_info "Please check the output above and try running ./setup_env.sh manually"
        fi
    else
        print_error "setup_env.sh not found. Please create the environment manually."
    fi
else
    print_info "Environment reset complete. Run ./setup_env.sh when ready to create a fresh environment."
fi

echo ""
print_success "Script completed successfully!"
