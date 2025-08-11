#!/usr/bin/env python3
"""Environment setup script"""

import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install required dependencies"""
    print("Installing Python dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."])
    
    print("Installing development dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])


def setup_directories():
    """Create necessary directories"""
    dirs = ["output", "output/memory", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True, parents=True)
    print("Created necessary directories")


def setup_git_hooks():
    """Setup git pre-commit hooks"""
    try:
        subprocess.run(["pre-commit", "install"], check=True)
        print("Installed pre-commit hooks")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not install pre-commit hooks (optional)")


def main():
    """Main setup function"""
    print("Setting up LLM_ATC7 environment...")
    
    install_dependencies()
    setup_directories()
    setup_git_hooks()
    
    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Test the CLI: python -m src.cdr.cli health-check")
    print("2. Run tests: pytest")
    print("3. Try a demo: python -m src.cdr.cli run-e2e --scat-path data/sample_scat.json")
    print("4. View help: python -m src.cdr.cli --help")


if __name__ == "__main__":
    main()
