#!/usr/bin/env python3
"""Generate AI_ASSISTANCE.md: a per-file record of AI assistance and verification.

Every file in the release tree gets one row. Provenance is derived from the
development repository: files first added or modified in the revision commits
(after ``98462d9``, 2026-09-20 onward) were drafted by Claude Code under the
author's direction; files present before that were author-written with
editing/refactoring assistance from ChatGPT and OpenAI Codex (code) or
language refinement (prose), as declared by the author; machine-generated
records (result JSON, configs written by freeze scripts, manifests, generated
tables, evidence CSVs) had no AI involvement of their own and are marked "no".
The verification column names a mechanism that exists in this repository.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
REVISION_BASE = "98462d9df7214571bd4cff28a01f56177fc9e467"

DECLARATION = (
    "During the preparation of this work the author used ChatGPT and OpenAI Codex for "
    "editing and refactoring author-written code and for language refinement of "
    "author-written text, and, during the revision, Claude Code (Anthropic) to draft the "
    "revision-stage audit, comparison, instrumentation, and documentation files from the "
    "author's specifications. All AI-assisted code was reviewed by the author and "
    "independently verified by unit tests, hash-verified inputs, fail-closed validation "
    "audits, and independent recomputation of every reported value from immutable analysis "
    "artifacts (`make reproduce`). A per-file record of assistance and verification is "
    "released as AI_ASSISTANCE.md in the public repository. The author reviewed and edited "
    "all resulting material and takes full responsibility for the content of the article."
)

ORIGINAL_CODE = "yes — author-written; ChatGPT (chat) and OpenAI Codex used for editing and refactoring"
ORIGINAL_PROSE = "yes — author-written; language refinement (ChatGPT/Codex)"
AUDIT_2026_09 = "yes — submission-audit tooling/report prepared with AI assistance (OpenAI Codex/ChatGPT); author reviewed"
REVISION = "yes — drafted by Claude Code (Anthropic) from the author's specification, 2026-09; author reviewed"
NONE = "no — machine-generated record"
NONE_UPSTREAM = "no — third-party licence text / upstream data card facts"


def git_changed_since_base() -> set[str]:
    out = subprocess.run(["git", "diff", "--name-only", f"{REVISION_BASE}..HEAD"], cwd=DEV, capture_output=True, text=True, check=True).stdout
    return set(out.split())


def tracked_at_base() -> set[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", REVISION_BASE], cwd=DEV, capture_output=True, text=True, check=True).stdout
    return set(out.split())


from datetime import datetime, timedelta, timezone

# Files not in git and last modified before the revision session began (2026-09-20 16:00 EDT)
# were produced by the earlier submission audits, not by this revision.
SESSION_START = datetime(2026, 9, 20, 16, 0, tzinfo=timezone(timedelta(hours=-4))).timestamp()


def pre_session_untracked(rel: str, changed: set[str], base: set[str]) -> bool:
    if rel in base or rel in changed:
        return False
    dev_path = DEV / rel
    return dev_path.exists() and dev_path.stat().st_mtime < SESSION_START


def classify(rel: str, changed: set[str], base: set[str]) -> tuple[str, str, str]:
    """Return (assistance, nature, verification) for a release-tree path."""
    p = Path(rel)
    top = p.parts[0] if p.parts else ""
    if pre_session_untracked(rel, changed, base) and p.suffix in {".py", ".md"}:
        return AUDIT_2026_09, "submission-audit script or report (2026-08/09, before the revision)", "every cited path and hash resolves in this repository; scripts re-run during the revision (reports/PHASE0_OPEN_QUESTIONS_RESOLVED.md)"
    # --- machine-generated records -------------------------------------------------
    if top in {"artifacts", "artifacts_revision"} and p.suffix in {".json", ".csv", ".txt", ".log", ".gz", ".tgz", ".npz", ".sha256", ".patch"} and not p.name.endswith((".md",)) and p.name != "PREREGISTRATION.json":
        return NONE, "runner/analysis output or environment capture", "hash manifests re-verified (reports/RELEASE_TREE_INTEGRITY.txt); analyses recomputed by make reproduce"
    if top == "configs":
        return NONE, "frozen configuration written by the freeze/prepare scripts", "SHA-256 recorded in every result record and re-hashed by the runner before each run; PREREGISTRATION_TIMELINE.md"
    if top == "data":
        return NONE, "panel manifest, row-identifier list, or synthetic panel written by prepare_* scripts", "panel SHA-256 recorded in MANIFEST.json and re-hashed by the runner"
    if rel.startswith("paper/generated/") or rel.startswith("paper/data/evidence/"):
        return NONE, "table/CSV rendered from analysis JSON", "byte-identical regeneration by make reproduce"
    if rel.startswith("reproduction/environment_captures/"):
        return NONE, "environment capture extracted from the frozen archives", "SHA256SUMS.txt beside the files; source archives hashed in reports/NODE_IDENTITY_EVIDENCE_2026-09-20.json"
    if rel.startswith("reports/phase0_archive_evidence/") or rel == "docs/DEV_HISTORY_LOG.txt" or rel.endswith(".ids.json") or rel.endswith(".sha256.json"):
        return NONE, "extracted/generated from archives or git history by a script", "regenerable by the named script; hashes recorded"
    if rel in {"LICENSE"}:
        return NONE_UPSTREAM, "Apache License 2.0 text", "verbatim from apache.org"
    if rel.startswith("reports/") and p.suffix in {".json", ".csv", ".tsv"}:
        return NONE, "audit output written by an experiments/ script", "regenerable by the named script; make reproduce compares PANEL_POWER_ANALYSIS.json and the row audit byte for byte"
    # --- revision-stage (this revision) ------------------------------------------------
    if rel in changed or rel not in base and (top in {"ascent", "experiments", "tests", "reproduction", "docs", "reports", "artifacts_revision", "paper"} or top == "" ):
        if top == "ascent":
            return REVISION, "instrumentation module (target-blind row view, provenance record)", "tests/test_target_blindness.py; every instrumented run row records blocked_target_access_attempts=0; same-instance instrumented-vs-uninstrumented comparison identical (artifacts_revision/target_blindness_2026-09/)"
        if top == "experiments" and p.suffix == ".py":
            if p.name.startswith(("compare_", "audit_", "extract_", "build_", "record_", "reproduce_", "analyze_panel_power", "analyze_generic")):
                return REVISION, "audit/comparison/analysis script", "self-tests on known-identical and deliberately different pairs; outputs cross-checked against retained manifests; make reproduce"
            if p.name in {"run_babilong_prompt.py", "analyze_noisy_composition_candidate.py"}:
                return ORIGINAL_CODE + "; revision additions (audit label, per-row target-blindness record, sentence-window source / row-audit flag) drafted by Claude Code", "runner/analysis script with revision-stage additions", "analysis JSON output unchanged (make reproduce); same-instance instrumented run identical to the uninstrumented run; tests/"
            return REVISION, "script", "executed and outputs retained under artifacts_revision/"
        if top == "experiments" and p.suffix == ".sh":
            return REVISION, "job script", "environment snapshots before/after; SHA256SUMS of outputs"
        if top == "tests":
            return REVISION if rel not in base else ORIGINAL_CODE + "; skip guard added in revision", "unit test", "executed (see reports/FRESH_CLONE_REPRODUCTION_TRANSCRIPT.txt)"
        if p.name in {"PREREGISTRATION.json", "DEVIATIONS.md"} or p.suffix == ".md":
            return REVISION, "document", "every quantitative statement cites a file path and SHA-256 that this repository contains; numbers cross-checked by make reproduce"
        if rel == "Makefile" or rel.startswith("reproduction/requirements"):
            return REVISION, "build/requirements file", "used by the fresh-clone reproduction transcript"
        return REVISION, "revision-stage file", "see VERIFICATION.md"
    # --- original tree ---------------------------------------------------------------
    if top in {"ascent"}:
        return ORIGINAL_CODE, "core implementation", "unit tests in tests/; preflight parser audit over all rows (experiments/audit_babilong_qwen_preflight.py); static target-blind test (tests/test_babilong_target_blind_static.py); cross-node reproduction of outputs (VERIFICATION.md §4)"
    if top == "experiments":
        if p.name.startswith("run_"):
            return ORIGINAL_CODE, "experiment runner", "hash-verified inputs and weights; fail-closed validation audit (reports/C15_FAIL_CLOSED_VALIDATION_AUDIT_2026-09-20.md); outputs reproduced across instances"
        if p.name.startswith(("analyze_", "render_", "export_")):
            return ORIGINAL_CODE, "analysis/rendering script", "recomputed by make reproduce from immutable records; byte-identical tables; independent SciPy check recorded in docs/EXPERIMENT_LEDGER.md"
        if p.name.startswith(("prepare_", "fetch_", "freeze_")):
            return ORIGINAL_CODE, "data construction / freeze script", "panel and config hashes recorded in MANIFEST.json and configs/, re-verified by runners"
        if p.name.startswith(("audit_", "verify_", "check_", "build_", "plot_", "profile_", "diagnose_", "generate_", "create_")):
            return AUDIT_2026_09 if rel not in base else ORIGINAL_CODE, "audit/build/plot script", "outputs cross-checked by experiments/audit_manuscript_consistency.py and experiments/audit_paper_claims.py"
        return ORIGINAL_CODE, "script", "tests/"
    if top == "tests":
        return ORIGINAL_CODE, "unit test", "executed (see reports/FRESH_CLONE_REPRODUCTION_TRANSCRIPT.txt)"
    if top == "docs" or rel.startswith("reproduction/") or rel == "pyproject.toml":
        return ORIGINAL_PROSE if p.suffix == ".md" else ORIGINAL_CODE, "documentation / metadata", "claims traced to configs and result records (PREREGISTRATION_TIMELINE.md)"
    if top == "reports":
        return AUDIT_2026_09, "audit report", "every cited path and hash resolves in this repository"
    return ORIGINAL_PROSE, "file", "see VERIFICATION.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEV / "release/ascent")
    parser.add_argument("--output", type=Path, default=DEV / "release_docs/AI_ASSISTANCE.md")
    args = parser.parse_args()
    changed = git_changed_since_base()
    base = tracked_at_base()
    rows = []
    for path in sorted(p for p in args.tree.rglob("*") if p.is_file() and ".git" not in p.parts):
        rel = str(path.relative_to(args.tree))
        if rel == "AI_ASSISTANCE.md":
            rows.append((rel, REVISION, "this document (generated by experiments/build_ai_assistance_table.py)", "the declaration text below is the one used in the manuscript, verbatim"))
            continue
        rows.append((rel, *classify(rel, changed, base)))
    counts = {}
    for _, a, _, _ in rows:
        key = a.split(" — ")[0]
        counts[key] = counts.get(key, 0) + 1
    lines = [
        "# AI assistance record",
        "",
        "This file records, for every file in this repository, whether AI tools were involved in producing it, in what role, and which mechanism in this repository independently verifies it. It is generated by `experiments/build_ai_assistance_table.py` from the development repository's history and the author's declaration; edit the rules there, not this table.",
        "",
        "## Tools and roles",
        "",
        "- **ChatGPT (OpenAI, chat interface)** and **OpenAI Codex (agentic editor)** — used on the original codebase (2026-08) to edit and refactor author-written code and to prepare submission-audit tooling and reports; used on the manuscript for language refinement of author-written text. The core logic of the writers, runners and analyses was written by the author.",
        "- **Claude Code (Anthropic, Claude Opus)** — used during the Array revision (from 2026-09-20) to draft, from the author's specifications, the revision-stage files: Phase 0 evidence scripts and reports, the environment recorder, the cross-architecture comparison, the target-blindness instrumentation, the power analysis, the row-level audit, the pre-registration timeline, the schema-blind ablation scripts, the release assembler, and the public documentation. The author reviewed each file and its outputs.",
        "- No AI tool produced any experimental measurement: every number in the manuscript is the output of the frozen runners on the pinned models and data, recomputed by `make reproduce`.",
        "",
        "## Manuscript declaration (verbatim)",
        "",
        "> " + DECLARATION,
        "",
        "## Summary",
        "",
        "| Assistance | Files |", "|---|---:|",
        *(f"| {k} | {v} |" for k, v in sorted(counts.items())),
        "",
        "## Per-file table",
        "",
        "| Path | AI assistance | Nature | Independent verification |",
        "|---|---|---|---|",
    ]
    for rel, a, n, v in rows:
        lines.append(f"| `{rel}` | {a} | {n} | {v} |")
    args.output.write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} files; {counts}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
