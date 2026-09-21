#!/usr/bin/env python3
"""Build PREREGISTRATION_TIMELINE.md and docs/DEV_HISTORY_LOG.txt from records.

Array revision 1, items C-38 / W-09. For every frozen configuration under
``configs/`` this script derives, from the development repository and the
retained result records only:

* the commit that introduced the file (``git log --diff-filter=A``), its author
  date, and the file's current SHA-256 together with the commit at which the
  file last changed;
* the panel hashes the config registers (``panel_sha256_by_name`` or the
  equivalent field);
* the earliest and latest ``started_utc`` among all retained result records
  whose ``config.sha256`` / ``config_sha256`` equals the config's hash, and how
  many such records exist.

"Scores were not inspected before the freeze" is then a checkable statement:
the freeze commit's author date precedes the earliest retained score for that
configuration. Configs with no retained scoring record are listed with that
fact rather than omitted.

``docs/DEV_HISTORY_LOG.txt`` is ``git log`` as hash, ISO author date and
subject only (no diffs, no bodies), so a referee can verify the timeline
against the private repository on request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def config_hash_index() -> dict[str, list[tuple[str, str]]]:
    """Map config SHA-256 -> [(started_utc, artifact path)] over all retained records."""
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path in (ROOT / "artifacts").rglob("*.json"):
        if "checkpoints" in path.parts or path.stat().st_size > 20_000_000:
            continue
        try:
            head = path.read_bytes()[:4000].decode("utf-8", "ignore")
        except OSError:
            continue
        if "sha256" not in head or "started_utc" not in head:
            continue
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        digest = None
        config = record.get("config")
        if isinstance(config, dict):
            digest = config.get("sha256")
        digest = digest or record.get("config_sha256")
        started = record.get("started_utc")
        if digest and started:
            index[digest].append((started, str(path.relative_to(ROOT))))
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, default=ROOT / "PREREGISTRATION_TIMELINE.md")
    parser.add_argument("--history-log", type=Path, default=ROOT / "docs/DEV_HISTORY_LOG.txt")
    parser.add_argument("--json", type=Path, default=ROOT / "reports/PREREGISTRATION_TIMELINE.json")
    args = parser.parse_args()

    index = config_hash_index()
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "configs").glob("*.json")):
        rel = str(path.relative_to(ROOT))
        added = git("log", "--diff-filter=A", "--follow", "--format=%H|%aI|%s", "--", rel).splitlines()
        added_commit, added_date, added_subject = (added[-1].split("|", 2) if added else ("", "", ""))
        last = git("log", "-1", "--format=%H|%aI", "--", rel)
        last_commit, last_date = (last.split("|", 1) if last else ("", ""))
        tracked = bool(added)
        digest = sha256_file(path)
        try:
            config = json.loads(path.read_text())
        except json.JSONDecodeError:
            config = {}
        panels = config.get("panel_sha256_by_name") or config.get("panel_hashes") or {}
        manifest = config.get("panel_manifest") or config.get("data_manifest_sha256")
        scored = sorted(index.get(digest, []))
        rows.append(
            {
                "config": rel,
                "sha256": digest,
                "tracked": tracked,
                "status": config.get("status"),
                "frozen_utc": config.get("frozen_utc") or config.get("frozen_at"),
                "added_commit": added_commit,
                "added_author_date": added_date,
                "added_subject": added_subject,
                "last_changed_commit": last_commit,
                "last_changed_author_date": last_date,
                "panel_hash_count": len(panels) if isinstance(panels, dict) else 0,
                "panel_hashes": panels if isinstance(panels, dict) else {},
                "panel_manifest": manifest,
                "retained_records_with_this_config_hash": len(scored),
                "earliest_retained_score_utc": scored[0][0] if scored else None,
                "latest_retained_score_utc": scored[-1][0] if scored else None,
                "earliest_record_path": scored[0][1] if scored else None,
                "freeze_precedes_earliest_score": (
                    None
                    if not scored or not last_date
                    else datetime.fromisoformat(last_date) <= datetime.fromisoformat(scored[0][0])
                ),
            }
        )

    log_lines = git("log", "--format=%H %aI %s").splitlines()
    args.history_log.parent.mkdir(parents=True, exist_ok=True)
    args.history_log.write_text(
        "# Development repository history: commit hash, ISO author date, subject only.\n"
        f"# Exported {datetime.now().astimezone().isoformat(timespec='seconds')} from HEAD {git('rev-parse', 'HEAD')}.\n"
        "# No commit bodies or diffs are included; the private repository is available to the editor on request.\n"
        + "\n".join(log_lines)
        + "\n"
    )
    args.json.write_text(json.dumps({"generated_from_head": git("rev-parse", "HEAD"), "configs": rows}, indent=2) + "\n")

    scored_rows = [r for r in rows if r["retained_records_with_this_config_hash"]]
    unscored_rows = [r for r in rows if not r["retained_records_with_this_config_hash"]]
    lines = [
        "# Pre-registration timeline",
        "",
        "Every row is derived from two sources only: the development repository's git",
        "history (`docs/DEV_HISTORY_LOG.txt`, hash + ISO date + subject) and the",
        "`started_utc` fields of the retained result records under `artifacts/` whose",
        "recorded `config.sha256` equals the configuration file's SHA-256. The column",
        "**freeze ≤ first score** is true when the author date of the commit that last",
        "changed the configuration precedes the earliest retained score produced under",
        "it; that is the checkable form of \"scores were not inspected before the",
        "freeze\". Panel hashes are the ones the configuration registers; the runner",
        "re-hashes each panel file and aborts on mismatch.",
        "",
        f"Generated by `experiments/build_preregistration_timeline.py` from HEAD `{git('rev-parse', 'HEAD')}`;",
        f"machine-readable copy with all panel hashes: `reports/PREREGISTRATION_TIMELINE.json`.",
        "",
        "## Configurations with retained scoring records",
        "",
        "| Configuration | SHA-256 (first 16) | Frozen by commit | Author date | Panels | Retained records | First score (UTC) | Freeze ≤ first score |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for r in scored_rows:
        lines.append(
            f"| `{r['config'].removeprefix('configs/')}` | `{r['sha256'][:16]}` | `{r['last_changed_commit'][:7]}` | "
            f"{r['last_changed_author_date']} | {r['panel_hash_count']} | {r['retained_records_with_this_config_hash']} | "
            f"{(r['earliest_retained_score_utc'] or '')[:19]} | {'yes' if r['freeze_precedes_earliest_score'] else '**no**'} |"
        )
    lines += [
        "",
        "## Configurations without a retained scoring record",
        "",
        "These are designs, development freezes superseded before scoring, manifests, or",
        "configurations whose records are stored in archives rather than as loose JSON.",
        "",
        "| Configuration | SHA-256 (first 16) | Added by commit | Author date | Status field |",
        "|---|---|---|---|---|",
    ]
    for r in unscored_rows:
        lines.append(
            f"| `{r['config'].removeprefix('configs/')}` | `{r['sha256'][:16]}` | `{r['added_commit'][:7] if r['added_commit'] else 'untracked'}` | "
            f"{r['added_author_date']} | {r['status'] or ''} |"
        )
    violations = [r for r in scored_rows if r["freeze_precedes_earliest_score"] is False]
    lines += [
        "",
        "## Summary",
        "",
        f"- {len(rows)} configuration files; {len(scored_rows)} have retained scoring records with a matching config hash.",
        f"- Freeze-precedes-first-score holds for {len(scored_rows) - len(violations)} of {len(scored_rows)} scored configurations."
        + (" The exceptions are listed above with **no** and discussed in VERIFICATION.md." if violations else ""),
        "- The git author dates are those recorded in the private development repository; they are not independently timestamped. The frozen backup archive (SHA-256 `431049c38a3c91e7b0fb1f8a3d5a52484aeae609a81a12af0c9c5272a5e00821`, 2026-08-23) contains the full repository and is available to the editor on request.",
        "",
    ]
    args.timeline.write_text("\n".join(lines))
    print(f"{len(rows)} configs, {len(scored_rows)} scored, {len(violations)} freeze-order violations")
    for r in violations:
        print("  VIOLATION", r["config"], r["last_changed_author_date"], "->", r["earliest_retained_score_utc"], r["earliest_record_path"])
    print(f"written: {args.timeline}, {args.history_log} ({len(log_lines)} commits), {args.json}")


if __name__ == "__main__":
    main()
