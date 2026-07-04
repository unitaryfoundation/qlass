# Makefile for QLASS project

.PHONY: test lint format typecheck check-examples

test:
	uv run --group test pytest

lint:
	uv run --group lint ruff check src/ tests/
	uv run --group lint ruff format --check src/ tests/

format:
	uv run --group lint ruff format src/ tests/

typecheck:
	uv run --group type-check mypy src/qlass

check-examples:
	@echo "Running all Python example scripts in examples/..."
	@set -e; for f in $$(git ls-files 'examples/*.py'); do \
	  echo "Running $$f..."; \
	  uv run python "$$f"; \
	done
	@echo "All example scripts ran successfully."
