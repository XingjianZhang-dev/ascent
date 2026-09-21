#!/usr/bin/env python3
"""One-command, CPU-only recomputation of every reported number (item C-60).

``make reproduce`` runs this script. From the retained per-row result records
it (1) re-verifies every SHA-256 manifest in the result trees it reads,
(2) recomputes the four authoritative analysis JSONs and compares them with the
retained ones (exact for every non-float field; floats to an absolute tolerance
that only absorbs the ~1e-12 difference between SciPy versions in the Student-t
critical value, which the ledger documents), (3) re-renders every generated
manuscript table and compares it byte for byte with ``paper/generated/``,
(4) regenerates the power analysis and the 13,824-row transition audit and
compares them (the audit CSV column-wise: header, row count and every non-float
column exact, float columns to an absolute tolerance of 1e-12, because NumPy's
SIMD dispatch differs between CPUs and moves the last bit of some exp/log
results), and (5) prints every reported number at manuscript precision.
It exits non-zero on any mismatch. Nothing under ``artifacts/`` is written.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import subprocess
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable



def fmt(value: float, places: int) -> str:
    """Manuscript rounding rule: round half up on the decimal value.

    Identical to ``decimal()`` in ``render_paper_additional_tables.py`` and ``d()`` in
    ``render_appendix_tables.py``. Panel means are exact decimals (multiples of 1/800 for
    ten 80-row panels), so ties such as 85.625 occur; Python's ``f"{x:.2f}"`` rounds the
    binary double (85.625 is exactly representable and rounds half to even → 85.62), whereas
    the manuscript reports 85.63. Every printed number below therefore uses this helper.
    """
    return str(Decimal(f"{value:.12f}").quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


OFFICIAL = "artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9"
SEMANTIC = "artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6"
FACTORIAL = "artifacts/sum_numeric_candidate"
AROUND7B = "artifacts/around7b_formal"
SYSTEMS = "artifacts/remote_results/babilong_4k_systems/analysis.json"
FLOPS = "artifacts/remote_results/flops_e49eae9/analysis.json"

MANIFESTS = [
    (f"{OFFICIAL}/SHA256SUMS.txt", OFFICIAL),
    (f"{SEMANTIC}/SHA256SUMS", SEMANTIC),
    (f"{AROUND7B}/node1-completed-canonical.sha256", AROUND7B),
    (f"{AROUND7B}/node1-raw4.sha256", AROUND7B),
    (f"{AROUND7B}/node2-completed-falcon-fixed.sha256", AROUND7B),
    (f"{AROUND7B}/node2-granite.sha256", AROUND7B),
]

FLOAT_TOLERANCE = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)


def verify_manifest(manifest: Path, base: Path) -> tuple[int, list[str]]:
    checked = 0
    failures = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        name = name.strip().lstrip("*")
        candidates = [base / name, manifest.parent / name, ROOT / name]
        target = next((c for c in candidates if c.is_file()), None)
        if target is None:
            failures.append(f"missing: {name}")
            continue
        checked += 1
        if sha256_file(target) != digest.strip():
            failures.append(f"hash mismatch: {name}")
    return checked, failures


CSV_FLOAT_TOLERANCE = 1e-12


def _is_float_column(values: list[str]) -> bool:
    """A column is compared numerically when every non-empty retained value parses as a float
    and at least one is written with a decimal point, an exponent, or as nan/inf. Integer-like
    and text columns (ids, counts, flags, retained-observation strings) are compared exactly."""
    seen_float_form = False
    for value in values:
        if value == "":
            continue
        try:
            float(value)
        except ValueError:
            return False
        if any(ch in value for ch in ".eE") or value.strip().lower().lstrip("+-") in {"nan", "inf"}:
            seen_float_form = True
    return seen_float_form


def compare_csv(retained_path: Path, recomputed_path: Path, tolerance: float = CSV_FLOAT_TOLERANCE) -> tuple[float, list[str]]:
    """Column-wise comparison of two CSV files.

    Header (names and order) and row count must be identical; every non-float column must be
    identical as text; float columns must agree to ``tolerance`` (absolute; nan equals nan).
    Returns (max absolute float difference, problems). Any problem means the files differ.
    """
    import csv

    with retained_path.open(newline="") as fh:
        retained = list(csv.reader(fh))
    with recomputed_path.open(newline="") as fh:
        recomputed = list(csv.reader(fh))
    problems: list[str] = []
    worst = 0.0
    if not retained or not recomputed:
        return worst, ["empty CSV"]
    if retained[0] != recomputed[0]:
        return worst, [f"header differs: {retained[0][:8]} vs {recomputed[0][:8]}"]
    if len(retained) != len(recomputed):
        return worst, [f"row count differs: {len(retained) - 1} vs {len(recomputed) - 1}"]
    header = retained[0]
    float_columns = {j for j in range(len(header)) if _is_float_column([row[j] for row in retained[1:] if j < len(row)])}
    for i, (a, b) in enumerate(zip(retained[1:], recomputed[1:]), start=1):
        if len(a) != len(b):
            problems.append(f"row {i}: field count differs")
            continue
        for j, (x, y) in enumerate(zip(a, b)):
            if j in float_columns:
                try:
                    fx, fy = float(x), float(y)
                except ValueError:
                    problems.append(f"row {i} column {header[j]}: non-numeric value {y!r}")
                    continue
                if fx != fx and fy != fy:  # both nan
                    continue
                diff = abs(fx - fy)
                if not diff <= tolerance:
                    problems.append(f"row {i} column {header[j]}: {x} vs {y}")
                worst = max(worst, diff if diff == diff else float("inf"))
            elif x != y:
                problems.append(f"row {i} column {header[j]}: {x!r} vs {y!r}")
        if len(problems) > 50:
            problems.append("... further differences omitted")
            break
    return worst, problems


def compare_json(retained: Any, recomputed: Any, path: str = "") -> tuple[float, list[str]]:
    worst = 0.0
    problems: list[str] = []
    if isinstance(retained, dict) and isinstance(recomputed, dict):
        if set(retained) != set(recomputed):
            problems.append(f"{path}: key sets differ {sorted(set(retained) ^ set(recomputed))[:5]}")
        for key in set(retained) & set(recomputed):
            w, p = compare_json(retained[key], recomputed[key], f"{path}/{key}")
            worst = max(worst, w)
            problems += p
    elif isinstance(retained, list) and isinstance(recomputed, list):
        if len(retained) != len(recomputed):
            problems.append(f"{path}: list lengths differ")
        for index, (a, b) in enumerate(zip(retained, recomputed)):
            w, p = compare_json(a, b, f"{path}[{index}]")
            worst = max(worst, w)
            problems += p
    elif isinstance(retained, bool) or isinstance(recomputed, bool):
        if retained != recomputed:
            problems.append(f"{path}: {retained!r} != {recomputed!r}")
    elif isinstance(retained, (int, float)) and isinstance(recomputed, (int, float)):
        diff = abs(float(retained) - float(recomputed))
        worst = max(worst, diff)
        if diff > FLOAT_TOLERANCE:
            problems.append(f"{path}: {retained!r} != {recomputed!r} (diff {diff:.3e})")
    elif retained != recomputed:
        problems.append(f"{path}: {retained!r} != {recomputed!r}")
    return worst, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=Path, help="directory to keep the recomputed outputs in (default: temporary)")
    args = parser.parse_args()
    tmp = args.keep or Path(tempfile.mkdtemp(prefix="ascent_reproduce_"))
    tmp.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print("== 1. SHA-256 manifests of the result trees ==")
    for manifest, base in MANIFESTS:
        checked, problems = verify_manifest(ROOT / manifest, ROOT / base)
        status = "ok" if not problems else "FAIL"
        print(f"  [{status}] {manifest}: {checked} files verified" + (f"; {problems[:3]}" if problems else ""))
        failures += [f"manifest {manifest}: {p}" for p in problems]

    print("== 2. Recompute the authoritative analysis JSONs from per-row records ==")
    analyses = {
        "official_16k": (
            [PY, "-m", "experiments.analyze_babilong_canonical_coscale_16k_confirmation",
             "--config", "configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json",
             "--small-root", f"{OFFICIAL}/canonical/0p5b", "--mid-root", f"{OFFICIAL}/canonical/1p5b",
             "--large-root", f"{OFFICIAL}/canonical/3b", "--raw-large-root", f"{OFFICIAL}/raw/3b",
             "--output", str(tmp / "official_16k_analysis.json")],
            f"{OFFICIAL}/analysis.json",
        ),
        "semantic_holdout": (
            [PY, "-m", "experiments.analyze_babilong_canonical_semantic_holdout",
             "--config", "configs/babilong_qwen2p5_canonical_semantic_holdout_8k_confirmatory.json",
             "--small-root", f"{SEMANTIC}/0p5b", "--mid-root", f"{SEMANTIC}/1p5b", "--large-root", f"{SEMANTIC}/3b",
             "--output", str(tmp / "semantic_holdout_analysis.json")],
            f"{SEMANTIC}/analysis.json",
        ),
        "factorial": (
            [PY, "-m", "experiments.analyze_noisy_composition_candidate",
             "--config", "configs/posttraining_noisy_composition_sum_numeric_candidate.json",
             "--root", FACTORIAL, "--phase", "confirmation",
             "--run-commit", "f133a9689adb870680d462049c0dd4e0bfc41e9a",
             "--output", str(tmp / "factorial_analysis.json"),
             "--emit-row-audit", str(tmp / "FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv")],
            f"{FACTORIAL}/confirmation_analysis.json",
        ),
        "around7b": (
            [PY, "-m", "experiments.analyze_babilong_around7b_extension",
             "--config", "configs/babilong_around7b_16k_extension_confirmatory.json",
             "--canonical-root", f"qwen2p5-7b-instruct={AROUND7B}/qwen2p5-7b/canonical_slots_4",
             "--canonical-root", f"mistral-7b-instruct-v0p3={AROUND7B}/mistral-7b/canonical_slots_4",
             "--canonical-root", f"falcon3-7b-instruct={AROUND7B}/falcon3-7b/canonical_slots_4",
             "--canonical-root", f"granite-3p3-8b-instruct={AROUND7B}/granite-8b/canonical_slots_4",
             "--canonical-root", f"qwen3-8b-nonthinking={AROUND7B}/qwen3-8b/canonical_slots_4",
             "--qwen-fixed3-root", f"{AROUND7B}/qwen2p5-7b/canonical_fixed_slots_3",
             "--qwen-raw4-root", f"{AROUND7B}/qwen2p5-7b/raw_slots_4",
             "--source-config", "configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json",
             "--frozen-qwen3b-root", "artifacts/frozen_qwen2p5_3b_16k",
             "--output", str(tmp / "around7b_analysis.json")],
            f"{AROUND7B}/analysis.json",
        ),
    }
    recomputed: dict[str, Any] = {}
    for name, (cmd, retained_path) in analyses.items():
        try:
            run(cmd)
        except subprocess.CalledProcessError as error:
            failures.append(f"{name}: analysis command failed\n{error.stderr[-2000:]}")
            print(f"  [FAIL] {name}: command failed")
            continue
        retained = json.loads((ROOT / retained_path).read_text())
        out_name = {"official_16k": "official_16k_analysis.json", "semantic_holdout": "semantic_holdout_analysis.json",
                    "factorial": "factorial_analysis.json", "around7b": "around7b_analysis.json"}[name]
        recomputed[name] = json.loads((tmp / out_name).read_text())
        worst, problems = compare_json(retained, recomputed[name])
        status = "ok" if not problems else "FAIL"
        print(f"  [{status}] {name}: max |float diff| = {worst:.3e} vs {retained_path}")
        failures += [f"{name}: {p}" for p in problems[:10]]

    print("== 3. Re-render the manuscript tables and compare byte for byte ==")
    renders = [
        [PY, "-m", "experiments.render_paper_qwen_confirmation_table", "--official", f"{OFFICIAL}/analysis.json",
         "--semantic", f"{SEMANTIC}/analysis.json", "--output", str(tmp / "qwen_confirmations.tex")],
        [PY, "-m", "experiments.render_paper_additional_tables", "--mechanism", f"{FACTORIAL}/confirmation_analysis.json",
         "--around7b", f"{AROUND7B}/analysis.json", "--systems", SYSTEMS, "--flops", FLOPS,
         "--mechanism-output", str(tmp / "scale_interaction_confirmation.tex"),
         "--around7b-output", str(tmp / "around7b_results.tex"), "--systems-output", str(tmp / "systems_results.tex")],
        [PY, "-m", "experiments.render_theory_evidence_map", "--official", f"{OFFICIAL}/analysis.json",
         "--mechanism", f"{FACTORIAL}/confirmation_analysis.json", "--around7b", f"{AROUND7B}/analysis.json",
         "--systems", SYSTEMS, "--output", str(tmp / "theory_evidence_map.tex")],
        [PY, "-m", "experiments.render_cross_task_table",
         "--qwen-multiquery-archive", "artifacts/remote_results/qwen_niah_fullctx_16k_multiquery_width_linear_confirmation_372xxx.tar.gz",
         "--smol-multiquery-archive", "artifacts/remote_results/smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz",
         "--smol-cwe", "artifacts/remote_results/cwe_confirm_c749f24/analysis.json",
         "--qwen-cwe", "artifacts/remote_results/qwen_cwe_confirm_cd8352f/analysis.json",
         "--output", str(tmp / "cross_task_confirmations.tex")],
        [PY, "experiments/analyze_panel_power.py", "--output", str(tmp / "PANEL_POWER_ANALYSIS.json"), "--tex-dir", str(tmp)],
        [PY, "experiments/render_appendix_tables.py", "--out-dir", str(tmp)],
    ]
    ablation_run_config = ROOT / "artifacts_revision/schema_blind_2026-09/babilong_qwen2p5_generic_writer_16k_ablation.run.json"
    if ablation_run_config.is_file():
        renders.append([PY, "-m", "experiments.analyze_generic_writer_ablation", "--config", str(ablation_run_config),
                        "--results-root", "artifacts_revision/schema_blind_2026-09", "--canonical-root", "artifacts_revision/target_blindness_2026-09/official16k",
                        "--output", str(tmp / "GENERIC_WRITER_ABLATION.json"), "--tex-output", str(tmp / "generic_writer_ablation.tex"),
                        "--tex-text-output", str(tmp / "generic_writer_ablation_text.tex")])
    for cmd in renders:
        try:
            run(cmd)
        except subprocess.CalledProcessError as error:
            failures.append(f"render failed: {' '.join(cmd[:3])}\n{error.stderr[-1500:]}")
            print(f"  [FAIL] {cmd[2] if cmd[1] == '-m' else cmd[1]}")
    appendix_tables = ("factorial_cells", "babilong_4k_8k_results", "smollm2_factorials", "smollm2_factorial_contrasts", "model_revisions")
    ablation_tables = ("generic_writer_ablation", "generic_writer_ablation_text") if ablation_run_config.is_file() else ()
    for name in ("qwen_confirmations", "scale_interaction_confirmation", "around7b_results", "systems_results",
                 "theory_evidence_map", "cross_task_confirmations", "panel_power", "statistical_consistency",
                 *appendix_tables, *ablation_tables):
        same = (tmp / f"{name}.tex").is_file() and filecmp.cmp(tmp / f"{name}.tex", ROOT / "paper/generated" / f"{name}.tex", shallow=False)
        print(f"  [{'ok' if same else 'FAIL'}] paper/generated/{name}.tex byte-identical")
        if not same:
            failures.append(f"table {name}.tex differs from paper/generated")
    for name, retained_path in (("PANEL_POWER_ANALYSIS.json", "reports/PANEL_POWER_ANALYSIS.json"),):
        worst, problems = compare_json(json.loads((ROOT / retained_path).read_text()), json.loads((tmp / name).read_text()))
        print(f"  [{'ok' if not problems else 'FAIL'}] {retained_path}: max |float diff| = {worst:.3e}")
        failures += [f"{name}: {p}" for p in problems[:5]]
    worst, problems = compare_csv(ROOT / "reports/row_audits/FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv",
                                  tmp / "FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv")
    print(f"  [{'ok' if not problems else 'FAIL'}] reports/row_audits/FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv: "
          f"header, row count and non-float columns identical; max |float diff| = {worst:.3e} (tolerance {CSV_FLOAT_TOLERANCE:.0e})")
    failures += [f"13,824-row audit CSV: {p}" for p in problems[:5]]

    print("== 4. Every reported number, from the recomputed analyses ==")
    if all(k in recomputed for k in ("official_16k", "semantic_holdout", "factorial", "around7b")):
        for study, key in (("Official 16K", "official_16k"), ("Semantic holdout", "semantic_holdout")):
            a = recomputed[key]
            for ep, res in a["endpoint_results"].items():
                g = res["primary"]["gain"]
                print(f"  {study} {ep}: gain {fmt(100*g['mean'], 2)} pts [{fmt(g['ci95_low'], 4)}, {fmt(g['ci95_high'], 4)}]")
            for c in a["adjacent_scale_contrasts"]:
                print(f"  {study} adjacent increment {c.get('lower_endpoint', '')}->{c.get('upper_endpoint', '')}: "
                      f"{fmt(100*c['gain_increment']['mean'], 2)} pts, p_Holm = {c['one_sided_p_holm']:.4g}")
        f = recomputed["factorial"]
        for name, e in f["estimands"].items():
            b = f["supplementary_bonferroni_familywise_estimands"].get(name, {})
            print(f"  Factorial {name}: {fmt(e['mean'], 3)} nats [{fmt(e['ci95_low'], 3)}, {fmt(e['ci95_high'], 3)}]"
                  + (f"; Bonferroni [{fmt(b['ci95_low'], 3)}, {fmt(b['ci95_high'], 3)}]" if b else ""))
        prov = f["provenance"]
        row_checks = sum(v for c in prov["nonredundant_rows_by_endpoint_and_transition"].values() for v in c.values())
        # The retained record's key ``exact_posterior_strictly_improves_every_transition_every_panel`` asserts that
        # the panel-MEAN exact-posterior NLL (config field ``exact_posterior_mean_nll_by_panel_and_rounds``) decreases
        # on every K transition in every panel. It is not a per-row statement; the per-row tally comes from the CSV.
        import csv as _csv
        with (tmp / "FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv").open() as handle:
            audit_rows = list(_csv.DictReader(handle))
        per_row_improved = sum(1 for r in audit_rows if float(r["exact_posterior_delta"]) > 0)
        print(f"  Factorial row checks (nesting + non-redundancy, by construction): {row_checks:,} of {len(audit_rows):,} "
              f"(every_transition_changes_every_row={prov['every_transition_changes_every_row']}, "
              f"panel_mean_exact_posterior_nll_decreases_every_transition_every_panel={prov['exact_posterior_strictly_improves_every_transition_every_panel']})")
        print(f"  per-row exact-posterior improvement: {per_row_improved:,} of {len(audit_rows):,} "
              "(not guaranteed; Proposition 2 is an expectation-level statement)")
        a7 = recomputed["around7b"]
        for ep, res in a7.get("endpoint_results", {}).items():
            g = res["primary"]["gain"] if "primary" in res else res.get("gain", {})
            if g:
                print(f"  Around-7B {ep}: gain {fmt(100*g['mean'], 2)} pts [{fmt(100*g['ci95_low'], 1)}, {fmt(100*g['ci95_high'], 1)}]")

    print("== Result ==")
    if failures:
        print(f"FAILED with {len(failures)} problem(s):")
        for item in failures:
            print("  - " + item.splitlines()[0])
        print(f"(recomputed outputs kept in {tmp})")
        return 1
    print(f"ALL REPORTED NUMBERS REPRODUCED (recomputed outputs in {tmp})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
