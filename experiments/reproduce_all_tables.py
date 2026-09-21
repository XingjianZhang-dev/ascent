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
compares them, and (5) prints every reported number at manuscript precision.
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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

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
    audit_same = sha256_file(tmp / "FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv") == sha256_file(
        ROOT / "reports/row_audits/FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv")
    print(f"  [{'ok' if audit_same else 'FAIL'}] reports/row_audits/FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv byte-identical")
    if not audit_same:
        failures.append("13,824-row audit CSV differs")

    print("== 4. Every reported number, from the recomputed analyses ==")
    if all(k in recomputed for k in ("official_16k", "semantic_holdout", "factorial", "around7b")):
        for study, key in (("Official 16K", "official_16k"), ("Semantic holdout", "semantic_holdout")):
            a = recomputed[key]
            for ep, res in a["endpoint_results"].items():
                g = res["primary"]["gain"]
                print(f"  {study} {ep}: gain {100*g['mean']:.2f} pts [{g['ci95_low']:.4f}, {g['ci95_high']:.4f}]")
            for c in a["adjacent_scale_contrasts"]:
                print(f"  {study} adjacent increment {c.get('lower_endpoint', '')}->{c.get('upper_endpoint', '')}: "
                      f"{100*c['gain_increment']['mean']:.2f} pts, p_Holm = {c['one_sided_p_holm']:.4g}")
        f = recomputed["factorial"]
        for name, e in f["estimands"].items():
            b = f["supplementary_bonferroni_familywise_estimands"].get(name, {})
            print(f"  Factorial {name}: {e['mean']:.3f} nats [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}]"
                  + (f"; Bonferroni [{b['ci95_low']:.3f}, {b['ci95_high']:.3f}]" if b else ""))
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
                print(f"  Around-7B {ep}: gain {100*g['mean']:.2f} pts [{100*g['ci95_low']:.1f}, {100*g['ci95_high']:.1f}]")

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
