"""Column-wise CSV comparison used by ``make reproduce`` for the 13,824-row transition audit.

NumPy's SIMD dispatch (AVX-512 versus AVX2/NEON kernels) moves the last bit of some exp/log
results, so a byte-for-byte comparison of the audit CSV fails on some CPUs while every reported
number is unchanged. The comparison must therefore accept last-ulp float differences and reject
everything else.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.reproduce_all_tables import CSV_FLOAT_TOLERANCE, compare_csv

ROOT = Path(__file__).resolve().parents[1]
REAL_CSV = ROOT / "reports/row_audits/FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv"

HEADER = ["endpoint", "row_id", "k_low", "nested_prefix", "row_check_pass", "exact_posterior_delta", "model_ascent_delta"]
ROWS = [
    ["qwen2p5-1p5b-instruct", "p02_r0000", "2", "True", "True", "-0.32308685051712926", "-0.3953262832865978"],
    ["qwen2p5-3b-instruct", "p02_r0001", "3", "True", "True", "1.1881652148863393", "0.86507836436921"],
    ["qwen2p5-7b-instruct", "p03_r0002", "5", "True", "False", "0.0", "2.5e-05"],
]


def write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def nudged(value: str) -> str:
    """The same float moved by one ulp, written with repr (as the audit CSV writes floats)."""
    x = float(value)
    return repr(float(np.nextafter(x, np.inf)))


def test_one_ulp_float_differences_pass(tmp_path: Path) -> None:
    retained = write(tmp_path / "a.csv", HEADER, ROWS)
    rows = [row[:5] + [nudged(row[5]), nudged(row[6])] for row in ROWS]
    recomputed = write(tmp_path / "b.csv", HEADER, rows)
    worst, problems = compare_csv(retained, recomputed)
    assert problems == []
    assert 0.0 < worst <= CSV_FLOAT_TOLERANCE


def test_float_changed_by_1e6_fails(tmp_path: Path) -> None:
    retained = write(tmp_path / "a.csv", HEADER, ROWS)
    rows = [list(row) for row in ROWS]
    rows[1][5] = repr(float(rows[1][5]) + 1e-6)
    recomputed = write(tmp_path / "b.csv", HEADER, rows)
    worst, problems = compare_csv(retained, recomputed)
    assert problems and "exact_posterior_delta" in problems[0]
    assert worst > CSV_FLOAT_TOLERANCE


@pytest.mark.parametrize("column, new_value", [(4, "False"), (1, "p02_r9999"), (2, "4")])
def test_non_float_cell_change_fails(tmp_path: Path, column: int, new_value: str) -> None:
    retained = write(tmp_path / "a.csv", HEADER, ROWS)
    rows = [list(row) for row in ROWS]
    rows[0][column] = new_value
    recomputed = write(tmp_path / "b.csv", HEADER, rows)
    worst, problems = compare_csv(retained, recomputed)
    assert problems and HEADER[column] in problems[0]
    assert worst == 0.0


def test_row_removed_fails(tmp_path: Path) -> None:
    retained = write(tmp_path / "a.csv", HEADER, ROWS)
    recomputed = write(tmp_path / "b.csv", HEADER, ROWS[:-1])
    _, problems = compare_csv(retained, recomputed)
    assert problems == ["row count differs: 3 vs 2"]


def test_column_renamed_fails(tmp_path: Path) -> None:
    retained = write(tmp_path / "a.csv", HEADER, ROWS)
    header = list(HEADER)
    header[5] = "exact_posterior_change"
    recomputed = write(tmp_path / "b.csv", header, ROWS)
    _, problems = compare_csv(retained, recomputed)
    assert len(problems) == 1 and problems[0].startswith("header differs")


def test_integer_like_column_is_compared_exactly(tmp_path: Path) -> None:
    # k_low holds integers; a numerically equal but textually different value must not pass
    retained = write(tmp_path / "a.csv", HEADER, ROWS)
    rows = [list(row) for row in ROWS]
    rows[0][2] = "2.0"
    recomputed = write(tmp_path / "b.csv", HEADER, rows)
    _, problems = compare_csv(retained, recomputed)
    assert problems and "k_low" in problems[0]


def test_real_row_audit_against_itself() -> None:
    if not REAL_CSV.is_file():
        pytest.skip("released row audit not present")
    worst, problems = compare_csv(REAL_CSV, REAL_CSV)
    assert problems == []
    assert worst == 0.0
