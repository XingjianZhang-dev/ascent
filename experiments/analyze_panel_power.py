#!/usr/bin/env python3
"""Retrospective power, minimum detectable effect, and statistical-consistency audit.

Array revision 1, item C-39 (Reviewer 2 point 4; Reviewer 1 request). Every
quantity is recomputed from the panel-level evidence CSVs that the manuscript
tables are rendered from:

* ``paper/data/evidence/primary_panel.csv`` — ten-panel ASCENT-minus-Foundation
  accuracy gains for the official 16K study and the semantic holdout at
  0.5B/1.5B/3B (unit: fraction correct; reported in percentage points);
* ``paper/data/evidence/factorial_panel.csv`` — nine-panel candidate-normalized
  NLL gains for the 3x3 Qwen2.5 x state-size factorial;
* ``paper/data/evidence/interaction_intervals.csv`` — the reported
  Bonferroni-simultaneous intervals, used only as a cross-check target.

For each registered contrast it reports the paired panel SD, SE, t statistic,
raw one-sided p, Holm-adjusted p (primary studies) or Bonferroni critical
value (factorial), the retrospective minimum detectable effect at the
registered error rate and 80% power (exact non-central t), and the achieved
power for the observed effect. Nothing is hard-coded except the family
structure that the frozen configs registered.

Outputs: ``reports/PANEL_POWER_ANALYSIS.json`` and two LaTeX tables under
``paper/generated/``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import nct, t as student_t

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "paper/data/evidence"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    n = int(array.size)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    se = sd / math.sqrt(n)
    t_stat = mean / se if se > 0 else math.inf
    return {"n": n, "df": n - 1, "mean": mean, "sd": sd, "se": se, "t": t_stat}


def power_one_sided(delta: float, se: float, df: int, alpha: float) -> float:
    """Power of a one-sided t test at level alpha for true effect delta."""
    crit = student_t.ppf(1.0 - alpha, df)
    return float(nct.sf(crit, df, delta / se))


def mde_one_sided(se: float, df: int, alpha: float, power: float) -> float:
    """Smallest effect detected with the given power (exact non-central t)."""
    return float(brentq(lambda d: power_one_sided(d, se, df, alpha) - power, 1e-12, 100.0 * se))


def mde_approx(se: float, df: int, alpha: float, power: float) -> float:
    return float(se * (student_t.ppf(1.0 - alpha, df) + student_t.ppf(power, df)))


def read_primary() -> dict[str, dict[str, list[float]]]:
    studies: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with (EVIDENCE / "primary_panel.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["contrast"] != "ASCENT - Foundation":
                continue
            studies[row["study"]][row["scale_label"]].append((int(row["panel"]), float(row["value"])))
    out: dict[str, dict[str, list[float]]] = {}
    for study, scales in studies.items():
        out[study] = {scale: [v for _, v in sorted(rows)] for scale, rows in scales.items()}
    return out


def read_factorial() -> dict[tuple[str, int], list[float]]:
    cells: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    with (EVIDENCE / "factorial_panel.csv").open() as handle:
        for row in csv.DictReader(handle):
            cells[(row["model"], int(row["slots"]))].append((int(row["panel"]), float(row["gain_nll"])))
    return {key: [v for _, v in sorted(rows)] for key, rows in cells.items()}


def read_reported_intervals() -> dict[str, dict[str, float]]:
    out = {}
    with (EVIDENCE / "interaction_intervals.csv").open() as handle:
        for row in csv.DictReader(handle):
            out[row["estimand"]] = {k: float(row[k]) for k in ("mean", "ci_low", "ci_high")}
    return out


def analyze_primary(study: str, scales: dict[str, list[float]], power_target: float) -> dict[str, Any]:
    order = ["0.5B", "1.5B", "3B"]
    n = len(scales[order[0]])
    df = n - 1
    endpoints = {}
    for scale in order:
        stats = paired_stats(scales[scale])
        crit = student_t.ppf(0.975, df)
        endpoints[scale] = {
            **stats,
            "ci95_low": stats["mean"] - crit * stats["se"],
            "ci95_high": stats["mean"] + crit * stats["se"],
            "mean_points": 100 * stats["mean"],
            "se_points": 100 * stats["se"],
        }
    # Registered directional family: the two adjacent gain increments, one-sided,
    # Holm step-down at family-wise 0.05. The smallest Holm threshold is 0.05/2.
    family_size = 2
    alpha_family = 0.05
    alpha_smallest = alpha_family / family_size
    contrasts = []
    raw_p = []
    for low, high in zip(order[:-1], order[1:], strict=True):
        diff = [b - a for a, b in zip(scales[low], scales[high], strict=True)]
        stats = paired_stats(diff)
        p_raw = float(student_t.sf(stats["t"], df))
        raw_p.append(p_raw)
        crit_two_sided = student_t.ppf(0.975, df)
        contrasts.append(
            {
                "contrast": f"{low} -> {high}",
                **stats,
                "mean_points": 100 * stats["mean"],
                "sd_points": 100 * stats["sd"],
                "se_points": 100 * stats["se"],
                "ci95_two_sided_points": [100 * (stats["mean"] - crit_two_sided * stats["se"]), 100 * (stats["mean"] + crit_two_sided * stats["se"])],
                "one_sided_p_raw": p_raw,
                "mde_points_at_alpha_smallest_power_target": 100 * mde_one_sided(stats["se"], df, alpha_smallest, power_target),
                "mde_points_approx_central_t": 100 * mde_approx(stats["se"], df, alpha_smallest, power_target),
                "achieved_power_at_alpha_smallest": power_one_sided(stats["mean"], stats["se"], df, alpha_smallest),
                "achieved_power_at_alpha_family": power_one_sided(stats["mean"], stats["se"], df, alpha_family),
            }
        )
    for row, p_holm in zip(contrasts, holm_adjust(raw_p), strict=True):
        row["one_sided_p_holm"] = p_holm
        row["passes_registered_family"] = row["mean"] > 0 and p_holm < alpha_family
    return {
        "study": study,
        "panels": n,
        "df": df,
        "family": {"size": family_size, "procedure": "Holm step-down, one-sided", "family_alpha": alpha_family, "smallest_step_alpha": alpha_smallest},
        "power_target": power_target,
        "t_critical_two_sided_0975": float(student_t.ppf(0.975, df)),
        "t_critical_one_sided_alpha_smallest": float(student_t.ppf(1 - alpha_smallest, df)),
        "endpoints": endpoints,
        "adjacent_increments": contrasts,
    }


def analyze_factorial(cells: dict[tuple[str, int], list[float]], reported: dict[str, dict[str, float]], power_target: float) -> dict[str, Any]:
    order = [("1.5B", 2), ("3B", 3), ("7B", 5)]
    n = len(cells[order[0]])
    df = n - 1
    g = lambda m, k: np.asarray(cells[(m, k)], dtype=np.float64)  # noqa: E731
    diag = [g(m, k) for m, k in order]
    estimands: dict[str, np.ndarray] = {
        "Diagonal gain: 1.5B/K2": diag[0],
        "Diagonal gain: 3B/K3": diag[1],
        "Diagonal gain: 7B/K5": diag[2],
        "Diagonal: 1.5B to 3B": diag[1] - diag[0],
        "Diagonal: 3B to 7B": diag[2] - diag[1],
        "7B refinement: K3 to K5": g("7B", 5) - g("7B", 3),
        "Interaction: 1.5B to 3B": (g("3B", 3) - g("3B", 2)) - (g("1.5B", 3) - g("1.5B", 2)),
        "Interaction: 3B to 7B": (g("7B", 5) - g("7B", 3)) - (g("3B", 5) - g("3B", 3)),
    }
    family_size = len(estimands)
    alpha_two_sided = 0.05 / family_size
    alpha_one_sided_equivalent = alpha_two_sided / 2  # a positive Bonferroni lower bound is a one-sided test at this level
    t_bonf = float(student_t.ppf(1 - alpha_one_sided_equivalent, df))
    rows = []
    for name, values in estimands.items():
        stats = paired_stats(values.tolist())
        ci_low = stats["mean"] - t_bonf * stats["se"]
        ci_high = stats["mean"] + t_bonf * stats["se"]
        rep = reported.get(name)
        rows.append(
            {
                "estimand": name,
                **stats,
                "bonferroni_ci_low": ci_low,
                "bonferroni_ci_high": ci_high,
                "reported": rep,
                "matches_reported_to_1e-9": (
                    rep is not None and abs(rep["mean"] - stats["mean"]) < 1e-9 and abs(rep["ci_low"] - ci_low) < 1e-9 and abs(rep["ci_high"] - ci_high) < 1e-9
                ),
                "one_sided_p_raw": float(student_t.sf(stats["t"], df)),
                "one_sided_p_bonferroni": min(1.0, family_size * float(student_t.sf(stats["t"], df))),
                "mde_nats_at_bonferroni_power_target": mde_one_sided(stats["se"], df, alpha_one_sided_equivalent, power_target),
                "achieved_power_at_bonferroni": power_one_sided(stats["mean"], stats["se"], df, alpha_one_sided_equivalent),
            }
        )
    return {
        "study": "Numeric factorial (candidate-normalized NLL gain, nats)",
        "panels": n,
        "df": df,
        "family": {"size": family_size, "procedure": "Bonferroni-simultaneous two-sided 95% intervals; positive lower bound required", "two_sided_alpha_per_estimand": alpha_two_sided},
        "power_target": power_target,
        "t_critical_bonferroni": t_bonf,
        "estimands": rows,
    }


def latex_escape(text: str) -> str:
    return text.replace("%", "\\%").replace("&", "\\&").replace("_", "\\_").replace(" -> ", " $\\to$ ")


def sci(p: float) -> str:
    if p >= 1e-3:
        return f"{p:.4f}"
    mantissa, exponent = f"{p:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def render_power_table(primary: list[dict[str, Any]], factorial: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated by experiments/analyze_panel_power.py; do not edit.",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Study & Contrast & Effect & SE & $t$ & $p_{\\mathrm{raw}}$ & MDE$_{80}$ & Power \\\\",
        "\\midrule",
    ]
    for study in primary:
        for c in study["adjacent_increments"]:
            lines.append(
                f"{latex_escape(study['study'])} & {latex_escape(c['contrast'])} & {c['mean_points']:.2f} pts & {c['se_points']:.2f} & "
                f"{c['t']:.2f} & {sci(c['one_sided_p_raw'])} & {c['mde_points_at_alpha_smallest_power_target']:.2f} pts & {c['achieved_power_at_alpha_smallest']:.2f} \\\\"
            )
    lines.append("\\midrule")
    for e in factorial["estimands"]:
        lines.append(
            f"Factorial & {latex_escape(e['estimand'])} & {e['mean']:.3f} nats & {e['se']:.3f} & {e['t']:.2f} & {sci(e['one_sided_p_raw'])} & "
            f"{e['mde_nats_at_bonferroni_power_target']:.3f} nats & {e['achieved_power_at_bonferroni']:.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_consistency_table(primary: list[dict[str, Any]], factorial: dict[str, Any]) -> str:
    official = next(s for s in primary if s["study"] == "Official 16K")
    lines = [
        "% Auto-generated by experiments/analyze_panel_power.py; do not edit.",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "Reported quantity & Reported value & Implied SE & Implied $t$ & df \\\\",
        "\\midrule",
    ]
    for c in official["adjacent_increments"]:
        lines.append(
            f"Official 16K {latex_escape(c['contrast'])} increment & {c['mean_points']:.2f} pts, $p_{{\\mathrm{{Holm}}}}$={sci(c['one_sided_p_holm'])} & "
            f"{c['se_points']:.2f} pts & {c['t']:.2f} & {c['df']} \\\\"
        )
    for scale, e in official["endpoints"].items():
        lines.append(
            f"Official 16K {scale} endpoint gain & {e['mean']:.4f} [{e['ci95_low']:.4f}, {e['ci95_high']:.4f}] & {e['se']:.4f} & {e['t']:.2f} & {e['df']} \\\\"
        )
    for name in ("Diagonal: 1.5B to 3B", "Interaction: 1.5B to 3B", "Diagonal: 3B to 7B", "Interaction: 3B to 7B"):
        e = next(x for x in factorial["estimands"] if x["estimand"] == name)
        lines.append(
            f"Factorial {latex_escape(name)} & {e['mean']:.3f} [{e['bonferroni_ci_low']:.3f}, {e['bonferroni_ci_high']:.3f}] (Bonf.) & {e['se']:.3f} & {e['t']:.2f} & {e['df']} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/PANEL_POWER_ANALYSIS.json")
    parser.add_argument("--tex-dir", type=Path, default=ROOT / "paper/generated")
    args = parser.parse_args()

    primary_data = read_primary()
    primary = [analyze_primary(study, scales, args.power) for study, scales in primary_data.items() if study in ("Official 16K", "Semantic holdout")]
    primary.sort(key=lambda s: 0 if s["study"] == "Official 16K" else 1)
    factorial = analyze_factorial(read_factorial(), read_reported_intervals(), args.power)

    official = next(s for s in primary if s["study"] == "Official 16K")
    first, second = official["adjacent_increments"]
    sanity = {
        "t_0975_df9": official["t_critical_two_sided_0975"],
        "official_first_increment_raw_p": first["one_sided_p_raw"],
        "official_first_increment_t": first["t"],
        "official_first_increment_se_points": first["se_points"],
        "official_second_increment_raw_p": second["one_sided_p_raw"],
        "official_second_increment_holm_p": second["one_sided_p_holm"],
        "official_second_increment_t": second["t"],
        "official_second_increment_se_points": second["se_points"],
        "official_mde_points_first_contrast_se": first["mde_points_at_alpha_smallest_power_target"],
        "official_mde_points_second_contrast_se": second["mde_points_at_alpha_smallest_power_target"],
        "official_achieved_power_second_increment": second["achieved_power_at_alpha_smallest"],
        "bonferroni_t_crit_1_minus_005_over_16_df8": factorial["t_critical_bonferroni"],
        "factorial_all_reported_intervals_reproduced": all(e["matches_reported_to_1e-9"] for e in factorial["estimands"] if e["reported"] is not None),
    }
    result = {
        "schema_version": 1,
        "inputs": {
            str(p.relative_to(ROOT)): sha256_file(p) for p in (EVIDENCE / "primary_panel.csv", EVIDENCE / "factorial_panel.csv", EVIDENCE / "interaction_intervals.csv")
        },
        "method": {
            "unit_of_inference": "panel (paired across panels); Student t with df = panels - 1",
            "power_model": "exact non-central t: power = P(T_{df, ncp} > t_{1-alpha, df}) with ncp = delta / SE, SE taken from the observed between-panel SD",
            "mde": "delta solving power(delta) = power_target at the smallest registered per-test alpha (Holm: 0.05/2 one-sided; Bonferroni: 0.05/8 two-sided i.e. 0.05/16 one-sided)",
            "caveat": "retrospective power uses the observed SE as if it were the true SD; it characterizes the design's sensitivity, not the probability that the observed result is true",
        },
        "primary_studies": primary,
        "factorial": factorial,
        "sanity": sanity,
        "headline": (
            "The official-16K design is well powered for the first adjacent increment and underpowered for the second: "
            f"at the registered one-sided level {official['family']['smallest_step_alpha']:.3f} with 80% power the minimum detectable increment is "
            f"{second['mde_points_at_alpha_smallest_power_target']:.2f} points (using the second contrast's SE), so the observed "
            f"{second['mean_points']:.2f}-point 1.5B->3B increment was detected at {second['achieved_power_at_alpha_smallest']:.2f} power "
            f"and is significant but marginal (p_Holm = {second['one_sided_p_holm']:.4f})."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.tex_dir.mkdir(parents=True, exist_ok=True)
    (args.tex_dir / "panel_power.tex").write_text(render_power_table(primary, factorial))
    (args.tex_dir / "statistical_consistency.tex").write_text(render_consistency_table(primary, factorial))
    print(json.dumps(sanity, indent=2))
    print(result["headline"])
    print(f"written: {args.output}  sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
