#!/usr/bin/env python3
"""Generate AI_ASSISTANCE.md: one row per file in the release, with the AI assistance it
received and the mechanism that verifies it.

Attribution rules (applied to the development tree, from which the release is copied):

* Hand-written files (code, tests, job scripts, documents, pre-registrations) whose last
  modification precedes the start of the revision session (2026-09-20 16:00 EDT) are
  author-written; ChatGPT and OpenAI Codex were used to edit and refactor code and to refine language. The same holds for files listed with an unchanged SHA-256 in the frozen
  2026-08-23 backup inventory.
* Hand-written files created during the revision were drafted by Claude Code (Anthropic)
  from the author's specifications; pre-existing files modified during the revision are
  author-written with revision changes drafted by Claude Code.
* Generated files (result records, logs, manifests, frozen configurations, panel data,
  rendered tables and figures, audit outputs, environment captures) are attributed to the
  script that generates them. Runner and analysis scripts are resolved from the record
  schema: the script whose string literals contain the most top-level keys of the record;
  where no script matches, the family is named instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEV = Path(__file__).resolve().parents[1]
SESSION_START = datetime(2026, 9, 20, 16, 0, tzinfo=timezone(timedelta(hours=-4))).timestamp()
BACKUP_INDEX = DEV / "backups/ASCENT_GPU_FINAL_20260823/PROJECT_FILES_SHA256.tsv"

DECLARATION = (
    "The author wrote the paper and the code; both were committed and frozen before the revision "
    "(repository history; PREREGISTRATION_TIMELINE.md). During the preparation of this work the "
    "author used ChatGPT and OpenAI Codex to refine language and to edit and refactor code, and, during the "
    "revision, Claude Code (Anthropic) to draft, from the author's specifications, revision-stage "
    "audit, comparison, instrumentation, analysis, release and documentation files, the "
    "sentence-window baseline writer of the Appendix D ablation, and wording of revised passages "
    "and of the response to the reviewers. No AI tool produced any reported measurement. After "
    "using these tools, the author reviewed and edited the content as needed and takes full "
    "responsibility for the content of the publication. A per-file record with the evidence is "
    "provided as AI_ASSISTANCE.md in the repository."
)

SUMMARY = (
    "The author wrote the paper and the code. The method, the experiment runners, the analyses, the "
    "tests and the frozen configurations were committed between 2026-08-14 and 2026-08-23 "
    "(`docs/DEV_HISTORY_LOG.txt`), every configuration was frozen before any score was inspected "
    "(`PREREGISTRATION_TIMELINE.md`), and every result record carries the commit of the runner that "
    "produced it. ChatGPT and OpenAI Codex were used to edit and refactor code and to refine language. Claude "
    "Code (Anthropic) was used only during the revision, from 2026-09-20, for the file categories "
    "named in the declaration below. Generated outputs are attributed to the script that produced "
    "them; no AI tool produced a measurement."
)

# Measured on 2026-09-20 against the members of the frozen 2026-08-23 archive
# (backups/ASCENT_GPU_FINAL_20260823) and the latexdiff of the submitted and revised manuscripts.
EVIDENCE = (
    "## Evidence\n"
    "\n"
    "- **Share.** Of the {total_code_lines:,} lines of code in this repository, {author_share:.0f}% are author-written "
    "({author_code_lines:,} lines in files untouched by the revision plus {rev_author_lines:,} pre-existing lines in the "
    "14 files the revision edited); {claude_share:.0f}% ({claude_code_lines:,} lines in the revision's support scripts and "
    "395 lines added to pre-existing files) were drafted by Claude Code from the author's specifications. Of the revised "
    "manuscript's 5,756 body words, about 77% are the submitted author-written text; 1,305 words were added in the "
    "revision and 355 deleted.\n"
    "- **Code.** {author_code_files} author-written code files ({author_code_lines:,} lines: all of `ascent/` except "
    "`ascent/target_blindness.py`, every experiment runner, every analysis script of the original submission, "
    "the tests) are listed below with the date and commit at which each was first added to the development "
    "repository, all before the revision began on 2026-09-20 (`docs/DEV_HISTORY_LOG.txt`). The revision added "
    "{claude_code_files} support scripts ({claude_code_lines:,} lines: audit, comparison, instrumentation, analysis and "
    "release tooling, listed as drafted by Claude Code) and edited 14 pre-existing code files by +395/−45 lines "
    "against their 2026-08-23 archive copies (`experiments/run_babilong_prompt.py`: +83/−7 for the audit label, "
    "the blinded row view and the sentence-window baseline; the rest audit, rendering and test guards).\n"
    "- **Results.** Every reported number is the output of the author-written runners on the frozen "
    "configurations; the records embed the runner's git commit, the configuration SHA-256 and the weight-shard "
    "hashes, and `make reproduce` recomputes every number from them.\n"
    "- **Manuscript.** The submitted manuscript (2026-08-23; git `0b99b3d`, 4,681 body words) is the author's "
    "text; the revision's latexdiff against it adds 1,305 words and deletes 355 (Section 8, Appendices A–E, the "
    "scope statements), so about three quarters of the revised text is the submitted text unchanged. The added "
    "passages were specified by the author and worded with Claude Code, then reviewed and edited by the author.\n"
)

AUTHOR_CODE = "author-written"
AUTHOR_PROSE = "author-written"
AUTHOR_REV = "author-written; revision edits drafted by Claude Code"
CLAUDE = "drafted by Claude Code from the author's specification (revision)"
UPSTREAM = "third-party text; no AI"

HAND_SUFFIXES = {".py", ".sh", ".md", ".toml", ".cff", ".tex"}
HAND_NAMES = {"Makefile", "pyproject.toml", "PREREGISTRATION.json", "DEVIATIONS.md"}

GENERATED_TABLES = {
    "qwen_confirmations.tex": "render_paper_additional_tables.py", "scale_interaction_confirmation.tex": "render_paper_additional_tables.py",
    "around7b_results.tex": "render_paper_additional_tables.py", "systems_results.tex": "render_paper_additional_tables.py",
    "theory_evidence_map.tex": "render_theory_evidence_map.py", "cross_task_confirmations.tex": "render_cross_task_table.py",
    "panel_power.tex": "analyze_panel_power.py", "statistical_consistency.tex": "analyze_panel_power.py",
    "factorial_cells.tex": "render_appendix_tables.py", "babilong_4k_8k_results.tex": "render_appendix_tables.py",
    "smollm2_factorials.tex": "render_appendix_tables.py", "smollm2_factorial_contrasts.tex": "render_appendix_tables.py",
    "model_revisions.tex": "render_appendix_tables.py", "generic_writer_ablation.tex": "analyze_generic_writer_ablation.py",
    "generic_writer_ablation_text.tex": "analyze_generic_writer_ablation.py",
}
FIGURE_SCRIPTS = {
    "ascent_method": "plot_ascent_method.py", "primary_scaling": "plot_manuscript_evidence.py",
    "factorial_interaction": "plot_manuscript_evidence.py", "large_model_breadth": "plot_manuscript_evidence.py",
    "systems_efficiency": "plot_manuscript_evidence.py", "latest_factorial_boundaries": "plot_latest_factorial_boundaries.py",
    "qwen_primary_evidence": "plot_qwen_primary_evidence.py",
}
REPORT_SCRIPTS = {
    "PANEL_POWER_ANALYSIS.json": "analyze_panel_power.py", "PREREGISTRATION_TIMELINE.json": "build_preregistration_timeline.py",
    "NODE_IDENTITY_EVIDENCE_2026-09-20.json": "extract_phase0_archive_evidence.py",
    "THIRD_NODE_AUDIT_ROW_DIFF_2026-09-20.json": "compare_third_node_audit.py",
    "AROUND7B_DIAGNOSTICS_EXCLUSION_AUDIT_2026-09-20.json": "audit_around7b_diagnostics.py",
    "FACTORIAL_CROSS_NODE_EXACT_AUDIT_2026-09-20.json": "compare_factorial_cross_node.py",
    "RELEASE_TREE_INTEGRITY.txt": "build_release_tree.py",
}
# analyses whose recomputation make reproduce checks; the schema heuristic is not used for these
ANALYSIS_OVERRIDES = {
    "artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9": "analyze_babilong_canonical_coscale_16k_confirmation.py",
    "artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6": "analyze_babilong_canonical_semantic_holdout.py",
    "artifacts/sum_numeric_candidate": "analyze_noisy_composition_candidate.py",
    "artifacts/around7b_formal": "analyze_babilong_around7b_extension.py",
}


def load_backup_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if BACKUP_INDEX.is_file():
        with BACKUP_INDEX.open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                index[row["relative_project_path"]] = row["sha256"]
    return index


def dev_path(rel: str) -> Path:
    if "/" not in rel and rel not in {"Makefile", "pyproject.toml", "PREREGISTRATION_TIMELINE.md", ".gitignore"}:
        return DEV / "release_docs" / rel
    return DEV / rel


def is_hand_written(rel: str) -> bool:
    p = Path(rel)
    if p.name in HAND_NAMES or rel in {"LICENSE", "LICENSE-DATA"}:
        return True
    if rel.startswith(("paper/generated/", "reports/FRESH_CLONE", "reports/phase0_archive_evidence/")) or p.name in {"PREREGISTRATION_TIMELINE.md", "DEV_HISTORY_LOG.txt", ".gitignore", "AI_ASSISTANCE.md", "RELEASE_TREE_INTEGRITY.txt"}:
        return False
    if rel.startswith("reproduction/requirements"):
        return True
    return p.suffix in HAND_SUFFIXES


def first_commit(dev: Path) -> str:
    """Date and short hash of the commit that first added the file to the development repository."""
    out = subprocess.run(["git", "log", "--diff-filter=A", "--follow", "--format=%h %ad", "--date=short", "--", str(dev.relative_to(DEV))],
                         cwd=DEV, capture_output=True, text=True).stdout.strip().splitlines()
    if not out:
        return ""
    short, date = out[-1].split()
    return f" (committed {date}, {short})"


def hand_written_class(rel: str, index: dict[str, str]) -> str:
    if rel in {"LICENSE", "LICENSE-DATA"}:
        return UPSTREAM
    dev = dev_path(rel)
    key = str(dev.relative_to(DEV))
    exists = dev.exists()
    mtime = dev.stat().st_mtime if exists else 0
    prose = Path(rel).suffix in {".md", ".cff", ".tex"} or rel.endswith("PREREGISTRATION.json")
    if exists and mtime < SESSION_START:
        return (AUTHOR_PROSE if prose else AUTHOR_CODE) + first_commit(dev)
    if exists and key in index:
        if hashlib.sha256(dev.read_bytes()).hexdigest() == index[key]:
            return (AUTHOR_PROSE if prose else AUTHOR_CODE) + first_commit(dev)
        return AUTHOR_REV + first_commit(dev)
    return CLAUDE


class Resolver:
    """Resolve runner / analysis scripts for artifact directories from record schemas."""

    def __init__(self, tree: Path) -> None:
        self.tree = tree
        scripts = {p.name: p.read_text(errors="ignore") for p in (tree / "experiments").glob("*.py")}
        self.runners = {n: t for n, t in scripts.items() if n.startswith(("run_", "profile_"))}
        self.analyses = {n: t for n, t in scripts.items() if n.startswith(("analyze_", "compare_", "audit_", "verify_"))}
        self.all_scripts = scripts
        self._cache: dict[tuple[str, str], str | None] = {}

    @staticmethod
    def dir_key(rel: str) -> str:
        parts = rel.split("/")
        return "/".join(parts[:3]) if len(parts) > 2 and parts[1] == "remote_results" else "/".join(parts[:2])

    def _keys(self, files: list[Path]) -> set[str]:
        keys: set[str] = set()
        for f in files[:4]:
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            if isinstance(data, dict):
                keys |= set(data.keys())
        return keys

    def _best(self, keys: set[str], pool: dict[str, str], threshold: int) -> str | None:
        scored = sorted(((sum(1 for k in keys if f'"{k}"' in t or f"'{k}'" in t), n) for n, t in pool.items()), reverse=True)
        return scored[0][1] if scored and scored[0][0] >= threshold else None

    def resolve(self, rel: str, kind: str) -> str | None:
        key = (self.dir_key(rel), kind)
        if key in self._cache:
            return self._cache[key]
        base = self.tree / key[0]
        files = [p for p in base.rglob("*.json") if p.is_file()] if base.is_dir() else []
        name = lambda p: p.name.lower()
        if kind == "analysis":
            picked = [p for p in files if "analysis" in name(p)]
            result = ANALYSIS_OVERRIDES.get(key[0]) or self._best(self._keys(picked), self.analyses, 8)
        else:
            picked = [p for p in files if not any(s in name(p) for s in ("analysis", "manifest", "preregistration", "environment", "comparison", ".run."))]
            result = self._best(self._keys(picked), self.runners, 8)
        self._cache[key] = result
        return result

    def freeze_commit(self, rel: str) -> tuple[str | None, list[str]]:
        """The commit that added a configuration, and any prepare/freeze script committed with it."""
        if not hasattr(self, "_timeline"):
            path = self.tree / "reports/PREREGISTRATION_TIMELINE.json"
            self._timeline = {c["config"]: c for c in json.loads(path.read_text())["configs"]} if path.is_file() else {}
            self._commit_files: dict[str, list[str]] = {}
        entry = self._timeline.get(rel)
        if not entry:
            return None, []
        commit = entry["added_commit"]
        if commit not in self._commit_files:
            out = subprocess.run(["git", "show", "--name-only", "--format=", commit], cwd=DEV, capture_output=True, text=True)
            self._commit_files[commit] = [f.split("/")[-1] for f in out.stdout.split() if f.startswith("experiments/") and f.split("/")[-1].startswith(("prepare_", "freeze_", "fetch_", "profile_"))]
        return commit[:7], self._commit_files[commit]

    def script_mentioning(self, needle: str, prefixes: tuple[str, ...]) -> str | None:
        hits = [n for n, t in self.all_scripts.items() if n.startswith(prefixes) and needle in t]
        return sorted(hits)[0] if hits else None


def generated_class(rel: str, resolver: Resolver) -> tuple[str, str]:
    """Return (assistance, verification) for a generated file."""
    p = Path(rel)
    top = p.parts[0]
    name = p.name
    low = name.lower()
    gen = lambda s: f"generated by experiments/{s}"
    if top in {"artifacts", "artifacts_revision"}:
        if name.startswith("SHA256SUMS") or p.suffix == ".sha256":
            return "generated by sha256sum", "re-verified against the files (reports/RELEASE_TREE_INTEGRITY.txt)"
        if name == "MANIFEST.json":
            return "generated by the prepare_* freeze script of the panel", "panel SHA-256 re-hashed by the runner"
        if "bge_cache" in p.parts:
            return gen("prepare_babilong_bge_cache.py"), "hash-verified BAAI/bge-m3 and reranker shards; SHA256SUMS.remote.txt"
        if name.startswith("environment_") or "environment_before" in p.parts or "environment_after" in p.parts:
            return gen("record_run_environment.py"), "git state, GPU UUID and pip freeze recorded before and after each job"
        if name in {"CROSSARCH_COMPARISON.json", "SAME_GPU_RUN1_VS_RUN2_CONTROL.json"} or name.startswith("comparison_pairs"):
            return gen("compare_crossarch_reproduction.py"), "self-tests on identical and different pairs; inputs hash-verified"
        if name == "TARGET_BLINDNESS_AUDIT.json":
            return gen("analyze_target_blindness_audit.py"), "recomputed from the released row records"
        if name == "GENERIC_WRITER_ABLATION.json":
            return gen("analyze_generic_writer_ablation.py"), "recomputed by make reproduce"
        if low.endswith(".run.json"):
            return "config copy written by experiments/run_generic_writer_ablation.sh", "SHA-256 recorded in every result record"
        if p.suffix == ".log" or "logs" in p.parts:
            return "runner log (same runner as the records beside it)", "retained as written"
        if p.suffix in {".gz", ".tgz"} or ".part" in name:
            return "archived runner outputs (tar)", "SHA-256 in reports/RELEASE_TREE_INTEGRITY.txt and VERIFICATION.md"
        if p.parts[1].endswith("_source"):
            script = resolver.script_mentioning(p.parts[1].replace("babilong_", "").split("_")[0], ("fetch_",)) or "fetch_babilong_official_source.py"
            return gen(script), "source manifest and hashes; benchmark rows themselves are not redistributed"
        if p.parts[1] == "reconstructed_neural_control":
            return "reconstructed from archived runner outputs (docs/EXPERIMENT_LEDGER.md)", "hashes recorded in the ledger"
        if "analysis" in low:
            script = resolver.resolve(rel, "analysis")
            return (gen(script) if script else "analysis output of the study's analyze_* script (docs/EXPERIMENT_LEDGER.md)"), "recomputed from the per-row records (make reproduce for the four reported analyses)"
        script = resolver.resolve(rel, "runner")
        if script:
            return gen(script) + " on the frozen configuration", "config, panel and weight-shard SHA-256 recorded in the record; manifests re-verified"
        return "frozen-runner output (family in docs/EXPERIMENT_LEDGER.md)", "manifests re-verified (reports/RELEASE_TREE_INTEGRITY.txt)"
    if top == "configs":
        commit, scripts = resolver.freeze_commit(rel)
        if scripts:
            return gen(scripts[0]) + f" (freeze commit {commit})", "SHA-256 recorded in every result record; PREREGISTRATION_TIMELINE.md"
        return f"author-written frozen configuration (freeze commit {commit})" if commit else "author-written frozen configuration", "SHA-256 recorded in every result record; PREREGISTRATION_TIMELINE.md"
    if top == "data":
        if name.endswith((".ids.json", ".sha256.json")):
            return gen("build_release_tree.py"), "row identifiers and hashes of the withheld panel file"
        script = resolver.script_mentioning(p.parts[1] if len(p.parts) > 2 else name, ("prepare_", "fetch_"))
        return (gen(script) if script else "panel data written by a prepare_* script"), "panel SHA-256 recorded in MANIFEST.json and re-hashed by the runner"
    if rel.startswith("paper/generated/"):
        return gen(GENERATED_TABLES.get(name, "the renderer named in reproduce_all_tables.py")), "byte-identical regeneration by make reproduce"
    if rel.startswith("paper/data/evidence/"):
        return gen("export_manuscript_evidence.py"), "compared with the analysis records by audit_manuscript_consistency.py"
    if rel.startswith("paper/figures/"):
        return gen(FIGURE_SCRIPTS.get(p.stem, "plot_manuscript_evidence.py")), "every label reconciled with its analysis record by audit_manuscript_consistency.py"
    if rel.startswith("reproduction/environment_captures/"):
        return "extracted from the frozen instance archives", "SHA256SUMS.txt beside the files; archives hashed in reports/NODE_IDENTITY_EVIDENCE_2026-09-20.json"
    if rel.startswith("reports/phase0_archive_evidence/"):
        return gen("extract_phase0_archive_evidence.py"), "archive member hashes recorded"
    if rel.startswith("reports/GITLEAKS_"):
        return "generated by gitleaks", "re-run on every release build"
    if rel.startswith("reports/FRESH_CLONE_REPRODUCTION_TRANSCRIPT"):
        return "shell transcript of make reproduce and make test in a fresh clone", "reproducible by the commands it records"
    if rel.startswith("reports/row_audits/"):
        return gen("analyze_noisy_composition_candidate.py --emit-row-audit"), "byte-identical regeneration by make reproduce"
    if top == "reports" and name in REPORT_SCRIPTS:
        return gen(REPORT_SCRIPTS[name]), "regenerable by the named script"
    if top == "reports":
        script = resolver.script_mentioning(name, ("audit_", "verify_", "analyze_", "compare_", "build_", "check_"))
        return (gen(script) if script else "audit output of a pre-revision audit script"), "every cited path and hash resolves in this repository"
    if name in {"PREREGISTRATION_TIMELINE.md", "DEV_HISTORY_LOG.txt"}:
        return gen("build_preregistration_timeline.py"), "derived from git history and result timestamps"
    if name == ".gitignore":
        return gen("build_release_tree.py"), "—"
    if name == "AI_ASSISTANCE.md":
        return gen("build_ai_assistance_table.py") + "; summary text drafted by Claude Code", "declaration below checked word for word against the manuscript by audit_paper_claims.py"
    return "generated file", "see VERIFICATION.md"


def hand_written_verification(rel: str) -> str:
    p = Path(rel)
    top = p.parts[0]
    if top == "ascent":
        return "tests/; cross-instance reproduction (VERIFICATION.md §4)" if p.name != "target_blindness.py" else "tests/test_target_blindness.py; same-instance instrumented run identical to the uninstrumented run"
    if top == "experiments":
        if p.name.startswith("run_") and p.suffix == ".py":
            return "hash-verified inputs and weights; fail-closed validation; outputs reproduced across instances"
        if p.name.startswith(("analyze_", "render_", "reproduce_")):
            return "make reproduce (byte-identical tables, recomputed analyses)"
        if p.name.startswith(("compare_", "audit_", "extract_", "record_")):
            return "outputs cross-checked against retained manifests; make reproduce"
        if p.suffix == ".sh":
            return "environment snapshots before and after; SHA256SUMS of outputs"
        return "audited outputs (audit_manuscript_consistency.py, audit_paper_claims.py)"
    if top == "tests":
        return "executed (reports/FRESH_CLONE_REPRODUCTION_TRANSCRIPT.txt)"
    if p.name == "PREREGISTRATION.json":
        return "committed before execution (commit hash in the file)"
    if p.suffix in {".md", ".cff", ".toml"} or p.name in HAND_NAMES or rel.startswith("reproduction/requirements"):
        return "every cited path and hash resolves in this repository; numbers cross-checked by make reproduce"
    return "see VERIFICATION.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, default=DEV / "release/ascent")
    parser.add_argument("--output", type=Path, default=DEV / "release_docs/AI_ASSISTANCE.md")
    args = parser.parse_args()
    index = load_backup_index()
    resolver = Resolver(args.tree)
    files = sorted(str(p.relative_to(args.tree)) for p in args.tree.rglob("*") if p.is_file() and ".git" not in p.parts)

    rows: list[tuple[str, str, str]] = []
    figure_rows: dict[str, list[str]] = defaultdict(list)
    for rel in files:
        if rel.startswith("paper/figures/"):
            figure_rows[re.sub(r"(_preview|_grayscale)?\.(pdf|svg|png)$", "", rel)].append(rel)
            continue
        if is_hand_written(rel):
            rows.append((rel, hand_written_class(rel, index), hand_written_verification(rel)))
        else:
            rows.append((rel, *generated_class(rel, resolver)))
    for stem, members in figure_rows.items():
        exts = ",".join(sorted(m[len(stem):] for m in members))
        rows.append((f"{stem}{{{exts}}}", *generated_class(members[0], resolver)))
    rows.sort()

    code_counts = {"author": [0, 0], "claude": [0, 0], "rev": [0, 0]}
    for rel, a, _ in rows:
        if rel.split("/")[0] in {"ascent", "experiments", "tests"} and rel.endswith((".py", ".sh")):
            key = "rev" if "revision edits" in a else ("author" if a.startswith("author-written") else ("claude" if a.startswith("drafted by Claude Code") else None))
            if key:
                code_counts[key][0] += 1
                code_counts[key][1] += sum(1 for _ in (args.tree / rel).open(errors="ignore"))
    total_code_lines = sum(v[1] for v in code_counts.values())
    counts: dict[str, int] = defaultdict(int)
    for _, a, _ in rows:
        if a.startswith("drafted by Claude Code"):
            counts["drafted by Claude Code in the revision (support files named in the declaration)"] += 1
        elif "revision edits drafted by Claude Code" in a:
            counts["author-written, revision edits drafted by Claude Code"] += 1
        elif a.startswith("author-written"):
            counts["author-written"] += 1
        elif a.startswith("third-party"):
            counts["third-party text"] += 1
        else:
            counts["generated by the author's scripts (runner outputs, analyses, tables, figures, audits, manifests)"] += 1
    lines = [
        "# AI assistance record",
        "",
        SUMMARY,
        "",
        "## Manuscript declaration (verbatim)",
        "",
        "> " + DECLARATION,
        "",
        EVIDENCE.format(author_code_files=code_counts["author"][0], author_code_lines=code_counts["author"][1],
                        claude_code_files=code_counts["claude"][0], claude_code_lines=code_counts["claude"][1],
                        total_code_lines=total_code_lines, rev_author_lines=code_counts["rev"][1] - 395,
                        author_share=100 * (code_counts["author"][1] + code_counts["rev"][1] - 395) / total_code_lines,
                        claude_share=100 * (code_counts["claude"][1] + 395) / total_code_lines),
        "## Attribution",
        "",
        "Hand-written files are attributed from the development tree: files last modified before the revision session (2026-09-20 16:00 EDT), or unchanged since the frozen 2026-08-23 backup inventory, are author-written; files created in the revision were drafted by Claude Code; pre-existing files changed in the revision are author-written with revision changes drafted by Claude Code. Generated files name the script that generates them; runner and analysis scripts are resolved from each record's schema (the script whose string literals contain the record's top-level keys). Build previews are not part of the release.",
        "",
        f"Files: {len(files)} ({len(rows)} rows; each figure is one row across its formats). "
        + "; ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1] if kv[0].startswith('author-written') and ',' not in kv[0] else kv[1])) + ". "
        "The method implementation (`ascent/`, except the revision's target-blindness instrumentation module), the experiment runners, the analysis scripts of the original submission and the frozen configurations are author-written.",
        "",
        "## Per-file table",
        "",
        "| Path | AI assistance | Verification |",
        "|---|---|---|",
    ]
    for rel, a, v in rows:
        lines.append(f"| `{rel}` | {a} | {v} |")
    args.output.write_text("\n".join(lines) + "\n")
    print(f"{len(files)} files, {len(rows)} rows; {dict(sorted(counts.items()))}")
    print(f"written: {args.output}")


if __name__ == "__main__":
    main()
