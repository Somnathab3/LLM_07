# LLM_ATC7 Makefile

.PHONY: help install test lint format clean demo health-check

help:
	@echo "LLM_ATC7 Development Commands"
	@echo "============================"
	@echo "install      - Install project and dependencies"
	@echo "test         - Run test suite"
	@echo "lint         - Run linting checks"
	@echo "format       - Format code with black"
	@echo "clean        - Clean build artifacts"
	@echo "demo         - Run system demo"
	@echo "health-check - Check system health"
	@echo "docs         - Generate documentation"

install:
	pip install -e .
	pip install -e .[dev]

test:
	pytest --cov=src/cdr --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	black --check src/ tests/
	mypy src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

demo:
	python demo.py

health-check:
	python -m src.cdr.cli health-check

docs:
	@echo "Documentation available in docs/ directory"
	@echo "Architecture: docs/architecture.md"
	@echo "README: README.md"

# Development shortcuts
run-e2e:
	python -m src.cdr.cli run-e2e --scat-path data/sample_scat.json

metrics:
	python -m src.cdr.cli metrics --results-dir output/

# Setup commands
setup:
	python scripts/setup_environment.py

git-hooks:
	pre-commit install
