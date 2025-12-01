# Makefile for QLASS project

.PHONY: check-examples

check-examples:
	@echo "Running all Python example scripts in examples/..."
	@set -e; for f in examples/*.py; do \
	  echo "Running $$f..."; \
	  python3 "$$f"; \
	done
	@echo "All example scripts ran successfully."
