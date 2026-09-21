"""Skip, rather than fail, tests that need benchmark row bodies absent from the public tree.

The public repository redistributes no BABILong/RULER row bodies (see
THIRD_PARTY_NOTICES.md). Tests that open a withheld ``data/**/*.jsonl`` file
raise ``FileNotFoundError``; this hook reports them as skipped with the reason,
so the skip is visible in the test summary instead of silently passing. In the
development tree, where the panels exist, the tests run in full.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


OPTIONAL_DEPENDENCIES = {"torch", "transformers", "sentencepiece", "tokenizers", "safetensors", "bibtexparser", "accelerate"}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    try:
        outcome.get_result()
    except FileNotFoundError as error:
        missing = Path(str(error.filename or "")).resolve() if error.filename else None
        if missing is not None and missing.suffix == ".jsonl" and (ROOT / "data") in missing.parents:
            pytest.skip(f"benchmark row bodies are not redistributed ({missing.relative_to(ROOT)}); rebuild with the construction scripts")
        if missing is not None and (ROOT / "paper") in missing.parents and not (ROOT / "paper/main.tex").exists():
            pytest.skip("manuscript LaTeX sources are not part of the public tree (only paper/data/evidence and paper/generated are released)")
        raise
    except ModuleNotFoundError as error:
        if error.name and error.name.split(".")[0] in OPTIONAL_DEPENDENCIES:
            pytest.skip(f"optional GPU-stack dependency {error.name!r} is not installed (reproduction/requirements-core.txt)")
        raise
    except Exception as error:  # importlib.metadata.PackageNotFoundError for a pinned runtime dependency
        if type(error).__name__ == "PackageNotFoundError" and str(error).strip("'\"") in OPTIONAL_DEPENDENCIES | {"No package metadata was found for sentencepiece"}:
            pytest.skip(f"optional GPU-stack dependency is not installed: {error}")
        if type(error).__name__ == "PackageNotFoundError":
            pytest.skip(f"optional GPU-stack dependency is not installed: {error}")
        raise
