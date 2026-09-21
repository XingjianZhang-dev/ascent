# ASCENT — CPU-only recomputation of every reported number from the retained records.
# Requires only reproduction/requirements-reproduce.txt (numpy, scipy). No GPU, no model weights.
PYTHON ?= python3

.PHONY: reproduce test

reproduce:
	$(PYTHON) experiments/reproduce_all_tables.py

test:
	$(PYTHON) -m pytest -q tests/
