#!/usr/bin/env python3
"""Compare a revision-stage rerun against its retained reference as a ladder.

Phase 2A (Array revision 1). The reference (retained 2026-08 record, produced
on an RTX PRO 6000 Blackwell) is fixed before the rerun executes; the rerun is
produced on a different GPU architecture (A100, Ampere). The comparison is
reported as a ladder of separately stated agreement levels, never as one
pass/fail verdict:

1. ``discrete`` — per-row identity of every discrete scientific field
   (predicted answer / generated string / score / argmax candidate), with the
   number of differing rows per field;
2. ``continuous`` — per-row maximum absolute and relative differences of every
   stored floating-point quantity (NLLs, probability vectors), reported as
   numbers;
3. ``panel_precision`` — whether each panel-level statistic prints identically
   at the precision used in the manuscript (accuracies to 4 decimals, NLL
   nats to 3 decimals), alongside the raw values from both sides;
4. ``binary64`` — whether every stored float is identical in its IEEE-754
   binary64 representation (expected to hold on matched hardware; on a
   different architecture a failure here is an expected consequence of
   differing kernel paths, not a reproduction failure).

Both record families are supported: the numeric factorial
(``run_noisy_composition_candidate.py``) and BABILong prompt records
(``run_babilong_prompt.py``). Nothing under ``artifacts/`` is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


FACTORIAL_TOP_SCIENTIFIC = (
    "experiment", "config_sha256", "data_sha256", "panel", "rounds", "endpoint",
    "model_files", "model_identity", "preflight", "foundation_accuracy", "ascent_accuracy",
)
FACTORIAL_TOP_CONTINUOUS = ("foundation_nll", "ascent_nll", "gain_nll")  # dicts with mean/se/ci
FACTORIAL_ROW_DISCRETE = ("row_id", "target", "foundation_answer", "ascent_answer",
                          "retained_first_observations", "retained_second_observations")
FACTORIAL_ROW_SCALARS = ("foundation_nll", "ascent_nll", "gain_nll")
FACTORIAL_ROW_VECTORS = ("foundation_probabilities", "ascent_probabilities")

BABILONG_TOP_SCIENTIFIC = (
    "experiment", "endpoint", "config", "data", "model_files", "model_identity",
    "parser_accuracy", "wins", "regressions", "state",
)
BABILONG_TOP_CONTINUOUS = ("foundation", "ascent", "gain")  # dicts with mean/se/ci
BABILONG_ROW_DISCRETE = (
    "row_id", "task", "target", "foundation_output", "foundation_answer", "foundation_location",
    "foundation_score", "ascent_output", "ascent_answer", "ascent_location", "ascent_score",
    "supporting_fact_count", "retained_fact_count", "retained_facts", "structured_inventory",
    "structured_count", "retained_read_state_bytes", "persistent_payload_bytes",
    "write_budget_bytes", "write_payload_utilization", "foundation_prompt_tokens",
    "ascent_prompt_tokens",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def binary64_equal(a: float, b: float) -> bool:
    return struct.pack(">d", float(a)) == struct.pack(">d", float(b))


def rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom > 0 else 0.0


def fmt(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _flatten_floats(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten_floats(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.update(_flatten_floats(v, f"{prefix}[{i}]"))
    return out


def continuous_summary(ref: dict[str, float], new: dict[str, float]) -> dict[str, Any]:
    keys = sorted(set(ref) | set(new))
    missing = [k for k in keys if k not in ref or k not in new]
    common = [k for k in keys if k in ref and k in new]
    max_abs = 0.0
    max_rel = 0.0
    max_abs_key = None
    max_rel_key = None
    b64_all = True
    b64_mismatch = 0
    for k in common:
        a, b = ref[k], new[k]
        if not binary64_equal(a, b):
            b64_all = False
            b64_mismatch += 1
        d = abs(a - b)
        r = rel_diff(a, b)
        if d > max_abs:
            max_abs, max_abs_key = d, k
        if r > max_rel:
            max_rel, max_rel_key = r, k
    return {
        "values_compared": len(common),
        "keys_missing_on_one_side": missing,
        "max_abs_diff": max_abs,
        "max_abs_diff_at": max_abs_key,
        "max_rel_diff": max_rel,
        "max_rel_diff_at": max_rel_key,
        "binary64_identical": b64_all,
        "binary64_mismatching_values": b64_mismatch,
    }


def compare_rows(ref_rows: list[dict], new_rows: list[dict], discrete: tuple[str, ...],
                 scalars: tuple[str, ...], vectors: tuple[str, ...]) -> dict[str, Any]:
    if len(ref_rows) != len(new_rows):
        raise RuntimeError(f"row counts differ: reference={len(ref_rows)} rerun={len(new_rows)}")
    row_ids_same_order = all(a.get("row_id") == b.get("row_id") for a, b in zip(ref_rows, new_rows, strict=True))
    mismatch_counts = {f: 0 for f in discrete}
    differing_row_ids: dict[str, list[str]] = {f: [] for f in discrete}
    for a, b in zip(ref_rows, new_rows, strict=True):
        for f in discrete:
            if canonical(a.get(f)) != canonical(b.get(f)):
                mismatch_counts[f] += 1
                differing_row_ids[f].append(str(a.get("row_id")))
    argmax_flips = None
    if vectors:
        argmax_flips = {}
        for f in vectors:
            flips = []
            for a, b in zip(ref_rows, new_rows, strict=True):
                va, vb = a.get(f), b.get(f)
                if va is None or vb is None:
                    continue
                ia = max(range(len(va)), key=lambda i: va[i])
                ib = max(range(len(vb)), key=lambda i: vb[i])
                if ia != ib:
                    flips.append(str(a.get("row_id")))
            argmax_flips[f] = {"rows_flipped": len(flips), "row_ids": flips}
    ref_floats: dict[str, float] = {}
    new_floats: dict[str, float] = {}
    for idx, (a, b) in enumerate(zip(ref_rows, new_rows, strict=True)):
        for f in scalars + vectors:
            if f in a:
                ref_floats.update(_flatten_floats(a[f], f"row[{idx}].{f}"))
            if f in b:
                new_floats.update(_flatten_floats(b[f], f"row[{idx}].{f}"))
    return {
        "rows": len(ref_rows),
        "row_ids_same_order": row_ids_same_order,
        "discrete": {
            "fields": list(discrete),
            "all_discrete_fields_identical": all(v == 0 for v in mismatch_counts.values()),
            "rows_differing_per_field": {k: v for k, v in mismatch_counts.items() if v},
            "differing_row_ids": {k: v for k, v in differing_row_ids.items() if v},
            "argmax_flips": argmax_flips,
        },
        "continuous": continuous_summary(ref_floats, new_floats) if (scalars or vectors) else None,
    }


def compare_records(ref_path: Path, new_path: Path) -> dict[str, Any]:
    ref = json.loads(ref_path.read_text())
    new = json.loads(new_path.read_text())
    family = "factorial" if "foundation_probabilities" in (ref["predictions"][0] if ref["predictions"] else {}) else "babilong"

    if family == "factorial":
        top_sci = FACTORIAL_TOP_SCIENTIFIC
        top_cont = FACTORIAL_TOP_CONTINUOUS
        rows = compare_rows(ref["predictions"], new["predictions"], FACTORIAL_ROW_DISCRETE,
                            FACTORIAL_ROW_SCALARS, FACTORIAL_ROW_VECTORS)
        panel_stats = {
            "foundation_accuracy": (ref["foundation_accuracy"], new["foundation_accuracy"], 4),
            "ascent_accuracy": (ref["ascent_accuracy"], new["ascent_accuracy"], 4),
            "foundation_nll.mean": (ref["foundation_nll"]["mean"], new["foundation_nll"]["mean"], 3),
            "ascent_nll.mean": (ref["ascent_nll"]["mean"], new["ascent_nll"]["mean"], 3),
            "gain_nll.mean": (ref["gain_nll"]["mean"], new["gain_nll"]["mean"], 3),
        }
    else:
        top_sci = BABILONG_TOP_SCIENTIFIC
        top_cont = BABILONG_TOP_CONTINUOUS
        rows = compare_rows(ref["predictions"], new["predictions"], BABILONG_ROW_DISCRETE, (), ())
        panel_stats = {
            "foundation.mean": (ref["foundation"]["mean"], new["foundation"]["mean"], 4),
            "ascent.mean": (ref["ascent"]["mean"], new["ascent"]["mean"], 4),
            "gain.mean": (ref["gain"]["mean"], new["gain"]["mean"], 4),
            "remaining_error_elimination": (ref["remaining_error_elimination"], new["remaining_error_elimination"], 4),
        }

    # A field that the reference record's schema did not have yet (e.g. ``model_identity``,
    # added to the BABILong runner in commit 0d60263 after the official-16K reference
    # runs) cannot be compared; it is reported as ``None`` rather than as a mismatch.
    top_level = {
        f: (None if (f not in ref) != (f not in new) else canonical(ref.get(f)) == canonical(new.get(f)))
        for f in top_sci
    }
    fields_absent_from_reference_schema = [f for f in top_sci if f not in ref and f in new]
    if family == "babilong":
        ref_cond = {k: v for k, v in ref["condition"].items() if k != "diagnostic_no_promotion"}
        new_cond = {k: v for k, v in new["condition"].items() if k != "diagnostic_no_promotion"}
        top_level["condition_excluding_diagnostic_flag"] = canonical(ref_cond) == canonical(new_cond)
        top_level["by_task"] = canonical(ref.get("by_task")) == canonical(new.get("by_task"))
    top_cont_summary = continuous_summary(
        {k: v for f in top_cont for k, v in _flatten_floats(ref[f], f).items()},
        {k: v for f in top_cont for k, v in _flatten_floats(new[f], f).items()},
    )
    panel_precision = {
        name: {
            "reference": a, "rerun": b, "decimals": d,
            "reference_printed": fmt(a, d), "rerun_printed": fmt(b, d),
            "prints_identically": fmt(a, d) == fmt(b, d),
            "abs_diff": abs(a - b),
        }
        for name, (a, b, d) in panel_stats.items()
    }
    all_b64 = top_cont_summary["binary64_identical"] and (
        rows["continuous"] is None or rows["continuous"]["binary64_identical"]
    )
    env_ref = ref.get("environment", {})
    env_new = new.get("environment", {})
    return {
        "family": family,
        "reference": {"path": str(ref_path), "sha256": sha256_file(ref_path), "status": ref.get("status"),
                       "gpu": env_ref.get("cuda_device") or env_ref.get("gpu"), "git_commit": env_ref.get("git_commit"),
                       "git_dirty": env_ref.get("git_dirty"), "started_utc": ref.get("started_utc")},
        "rerun": {"path": str(new_path), "sha256": sha256_file(new_path), "status": new.get("status"),
                   "gpu": env_new.get("cuda_device") or env_new.get("gpu"), "git_commit": env_new.get("git_commit"),
                   "git_dirty": env_new.get("git_dirty"), "started_utc": new.get("started_utc")},
        "ladder": {
            "1_discrete": {
                "top_level_scientific_fields_equal": top_level,
                "fields_absent_from_reference_schema_not_compared": fields_absent_from_reference_schema,
                "fields_skipped_explanation": (
                    "these top-level fields exist in the rerun record but not in the reference record because the "
                    "runner added them to its output schema after the reference was produced (e.g. model_identity, "
                    "added in commit 0d60263 on 2026-08-15 after the official-16K reference runs of 2026-08-14); "
                    "they were skipped, not found equal, and do not enter any verdict"
                    if fields_absent_from_reference_schema else None
                ),
                "all_top_level_scientific_fields_equal": all(v for v in top_level.values() if v is not None),
                "rows": rows["rows"],
                "row_ids_same_order": rows["row_ids_same_order"],
                **rows["discrete"],
            },
            "2_continuous": {
                "top_level_statistics": top_cont_summary,
                "per_row": rows["continuous"],
            },
            "3_panel_precision": {
                "all_print_identically": all(v["prints_identically"] for v in panel_precision.values()),
                "statistics": panel_precision,
            },
            "4_binary64": {
                "all_stored_floats_binary64_identical": all_b64,
                "note": "expected on matched hardware; on a different GPU architecture a failure here reflects differing kernel reduction paths and is not a reproduction failure",
            },
        },
        "verdicts": {
            "scientific_fields_and_predictions_equal": all(v for v in top_level.values() if v is not None) and rows["discrete"]["all_discrete_fields_identical"],
            "panel_statistics_print_identically": all(v["prints_identically"] for v in panel_precision.values()),
            "binary64_identical": all_b64,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="JSON file: list of {reference, rerun, label} objects")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text())
    results = []
    for pair in pairs:
        result = compare_records(Path(pair["reference"]), Path(pair["rerun"]))
        result["label"] = pair.get("label")
        results.append(result)
        v = result["verdicts"]
        c = result["ladder"]["2_continuous"]
        per_row = c["per_row"]
        print(f"[{pair.get('label')}] family={result['family']} rows={result['ladder']['1_discrete']['rows']} "
              f"discrete_equal={v['scientific_fields_and_predictions_equal']} "
              f"panel_prints_identically={v['panel_statistics_print_identically']} "
              f"binary64={v['binary64_identical']} "
              f"max_abs={per_row['max_abs_diff'] if per_row else c['top_level_statistics']['max_abs_diff']:.3e} "
              f"max_rel={per_row['max_rel_diff'] if per_row else c['top_level_statistics']['max_rel_diff']:.3e}")
        flips = result["ladder"]["1_discrete"].get("argmax_flips")
        if flips:
            for f, info in flips.items():
                print(f"    argmax flips in {f}: {info['rows_flipped']}")
        diffs = result["ladder"]["1_discrete"]["rows_differing_per_field"]
        if diffs:
            print(f"    rows differing per field: {diffs}")
    summary = {
        "comparison": "cross-architecture reproduction ladder (retained Blackwell reference vs A100 rerun)",
        "pairs": len(results),
        "all_scientific_fields_and_predictions_equal": all(r["verdicts"]["scientific_fields_and_predictions_equal"] for r in results),
        "all_panel_statistics_print_identically": all(r["verdicts"]["panel_statistics_print_identically"] for r in results),
        "all_binary64_identical": all(r["verdicts"]["binary64_identical"] for r in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"written: {args.output}  sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
