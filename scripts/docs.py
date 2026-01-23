#!/usr/bin/env python3
"""Documentation management script for Titans project.

This script provides convenient commands to build, serve, and deploy
the MkDocs documentation.
"""

import argparse
import os
from pathlib import Path
import subprocess  # nosec B404  # Legitimate use for build automation
import sys


def run_command(
    cmd: list[str], check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    """Run a shell command and return the result."""
    print(f"🔧 Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)  # nosec B603  # Trusted command execution


def serve_docs():
    """Start the MkDocs development server."""
    print("🚀 Starting MkDocs development server...")
    print("📖 Documentation will be available at: http://localhost:8000")
    print("🔄 Auto-reload is enabled for live editing")

    try:
        run_command(["mkdocs", "serve"])
    except KeyboardInterrupt:
        print("\n👋 Documentation server stopped")


def build_docs():
    """Build the static documentation site."""
    print("🏗️  Building static documentation...")

    # Clean previous build
    site_dir = Path("site")
    if site_dir.exists():
        print("🧹 Cleaning previous build...")
        import shutil

        shutil.rmtree(site_dir)

    # Build documentation
    run_command(["mkdocs", "build"])

    print("✅ Documentation built successfully!")
    print(f"📁 Static files are in: {site_dir.absolute()}")


def deploy_docs():
    """Deploy documentation to GitHub Pages."""
    print("🚀 Deploying documentation to GitHub Pages...")

    # Deploy using mike for versioning
    run_command(["mike", "deploy", "--push", "--update-aliases", "main", "latest"])

    # Set as default version
    run_command(["mike", "set-default", "--push", "latest"])

    print("✅ Documentation deployed successfully!")


def check_docs():
    """Check documentation for issues."""
    print("🔍 Checking documentation...")

    # Check if MkDocs can build without errors
    try:
        run_command(["mkdocs", "build", "--strict"], check=False)
        print("✅ Documentation builds without errors")
    except subprocess.CalledProcessError:
        print("❌ Documentation has build errors")
        return False

    # Check for broken links (if available)
    try:
        run_command(["mkdocs", "build", "--strict"])
        print("✅ No broken internal links found")
    except subprocess.CalledProcessError:
        print("⚠️  Some links may be broken")

    return True


def install_deps():
    """Install documentation dependencies."""
    print("📦 Installing documentation dependencies...")

    run_command([sys.executable, "-m", "pip", "install", "titans-flax[docs]"])

    print("✅ Documentation dependencies installed!")


def clean_docs():
    """Clean documentation build artifacts."""
    print("🧹 Cleaning documentation build artifacts...")

    artifacts = ["site", ".cache"]

    for artifact in artifacts:
        artifact_path = Path(artifact)
        if artifact_path.exists():
            if artifact_path.is_dir():
                import shutil

                shutil.rmtree(artifact_path)
                print(f"🗑️  Removed directory: {artifact}")
            else:
                artifact_path.unlink()
                print(f"🗑️  Removed file: {artifact}")

    print("✅ Documentation cleaned!")


def validate_structure():
    """Validate documentation structure."""
    print("🔍 Validating documentation structure...")

    required_files = [
        "mkdocs.yml",
        "docs/index.md",
        "docs/getting-started/installation.md",
        "docs/getting-started/quick-start.md",
        "docs/tutorials/index.md",
        "docs/examples/index.md",
        "docs/api/index.md",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False

    print("✅ Documentation structure is valid!")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Titans Documentation Management")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Serve command
    subparsers.add_parser("serve", help="Start development server")

    # Build command
    subparsers.add_parser("build", help="Build static documentation")

    # Deploy command
    subparsers.add_parser("deploy", help="Deploy to GitHub Pages")

    # Check command
    subparsers.add_parser("check", help="Check documentation for issues")

    # Install command
    subparsers.add_parser("install", help="Install documentation dependencies")

    # Clean command
    subparsers.add_parser("clean", help="Clean build artifacts")

    # Validate command
    subparsers.add_parser("validate", help="Validate documentation structure")

    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    if args.command == "serve":
        serve_docs()
    elif args.command == "build":
        build_docs()
    elif args.command == "deploy":
        deploy_docs()
    elif args.command == "check":
        check_docs()
    elif args.command == "install":
        install_deps()
    elif args.command == "clean":
        clean_docs()
    elif args.command == "validate":
        validate_structure()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
