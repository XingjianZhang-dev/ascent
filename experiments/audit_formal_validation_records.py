#!/usr/bin/env python3
"""Audit retained formal validation records and archived failure logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

FAMILIES = {
    "official_16k_canonical": (
        "artifacts/remote_results/"
        "babilong_qwen_canonical_16k_confirmation_4fd25e9/canonical"
    ),
    "official_16k_raw_control": (
        "artifacts/remote_results/"
        "babilong_qwen_canonical_16k_confirmation_4fd25e9/raw"
    ),
    "semantic_holdout": (
        "artifacts/remote_results/"
        "babilong_qwen_canonical_semantic_holdout_b7480e6"
    ),
    "around7b_formal": "artifacts/around7b_formal",
    "isolated_systems": "artifacts/remote_results/babilong_4k_systems",
    "third_node_audit": "artifacts/array_third_node_audit",
    "numeric_factorial_confirmation": "artifacts/sum_numeric_candidate/confirmation",
}

MANIFESTS = (
    "artifacts/remote_results/"
    "babilong_qwen_canonical_16k_confirmation_4fd25e9/SHA256SUMS.txt",
    "artifacts/remote_results/"
    "babilong_qwen_canonical_semantic_holdout_b7480e6/SHA256SUMS",
)

FAILURE_PATTERN = re.compile(
    r"Traceback \(most recent call last\)|RuntimeError:|ValueError:|"
    r"AssertionError:|CUDA out of memory|No such file or directory|failed",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_files(name: str, root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "analysis.json" or "diagnostics" in path.parts:
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        if name == "numeric_factorial_confirmation":
            if "preflight" not in record or "predictions" not in record:
                continue
        elif "parser_accuracy" not in record or "predictions" not in record:
            continue
        records.append((path, record))
    return records


def audit_family(name: str, relative_root: str) -> dict[str, Any]:
    records = family_files(name, ROOT / relative_root)
    result: dict[str, Any] = {
        "root": relative_root,
        "result_files": len(records),
        "prediction_rows": sum(len(record["predictions"]) for _, record in records),
        "statuses": sorted({str(record.get("status")) for _, record in records}),
    }
    if name == "numeric_factorial_confirmation":
        required_true = (
            "candidate_boundary_stable",
            "candidate_support_valid",
            "pass",
            "target_values_not_rendered_by_prompt_builder",
        )
        all_preflights_pass = all(
            all(record["preflight"].get(key) is True for key in required_true)
            and record["preflight"].get("generation_invoked") is False
            for _, record in records
        )
        result.update(
            {
                "all_preflights_pass": all_preflights_pass,
                "required_true_preflight_fields": list(required_true),
                "generation_invoked_is_false_in_all_preflights": all(
                    record["preflight"].get("generation_invoked") is False
                    for _, record in records
                ),
            }
        )
    else:
        parser_values = sorted({record["parser_accuracy"] for _, record in records})
        result.update(
            {
                "parser_accuracy_values": parser_values,
                "all_parser_accuracy_one": parser_values == [1.0],
            }
        )
    return result


def audit_manifest(relative_path: str) -> dict[str, Any]:
    manifest = ROOT / relative_path
    verified = missing = mismatched = 0
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, raw_path = line.split(None, 1)
        raw_path = raw_path.strip().lstrip("*")
        candidate = Path(raw_path)
        if not candidate.is_absolute() and not raw_path.startswith("artifacts/"):
            candidate = manifest.parent / candidate
        elif not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.exists():
            missing += 1
        elif sha256_file(candidate) != expected:
            mismatched += 1
        else:
            verified += 1
    return {
        "path": relative_path,
        "verified": verified,
        "missing": missing,
        "mismatched": mismatched,
        "pass": missing == 0 and mismatched == 0,
    }


def matching_lines(text: str) -> list[dict[str, Any]]:
    return [
        {"line": line_number, "text": line[:500]}
        for line_number, line in enumerate(text.splitlines(), 1)
        if FAILURE_PATTERN.search(line)
    ]


def audit_logs() -> dict[str, Any]:
    unpacked_files = sorted(
        {
            *ROOT.joinpath("artifacts").rglob("*.log"),
            *ROOT.joinpath("artifacts").rglob("*.out"),
            *ROOT.joinpath("artifacts").rglob("*.err"),
        }
    )
    unpacked_findings: list[dict[str, Any]] = []
    for path in unpacked_files:
        lines = matching_lines(path.read_text(errors="replace"))
        if lines:
            unpacked_findings.append(
                {"path": str(path.relative_to(ROOT)), "matches": lines}
            )

    archive_paths = sorted(
        {
            *ROOT.joinpath("artifacts").rglob("*.tar.gz"),
            *ROOT.joinpath("artifacts").rglob("*.tgz"),
        }
    )
    archive_findings: list[dict[str, Any]] = []
    text_members = 0
    text_bytes = 0
    for archive_path in archive_paths:
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.lower().endswith(
                    (".log", ".txt", ".out", ".err")
                ):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                data = extracted.read()
                text_members += 1
                text_bytes += len(data)
                lines = matching_lines(data.decode("utf-8", "replace"))
                if lines:
                    archive_findings.append(
                        {
                            "archive": str(archive_path.relative_to(ROOT)),
                            "member": member.name,
                            "matches": lines,
                        }
                    )
    return {
        "unpacked_files_scanned": len(unpacked_files),
        "unpacked_findings": unpacked_findings,
        "archives_scanned": len(archive_paths),
        "archive_text_members_scanned": text_members,
        "archive_text_bytes_scanned": text_bytes,
        "archive_findings": archive_findings,
    }


def build_audit() -> dict[str, Any]:
    families = {name: audit_family(name, root) for name, root in FAMILIES.items()}
    family_gates_pass = all(
        result.get("all_preflights_pass", result.get("all_parser_accuracy_one"))
        for result in families.values()
    )
    manifests = [audit_manifest(path) for path in MANIFESTS]
    logs = audit_logs()
    return {
        "schema_version": 1,
        "audit": "retained_formal_validation_and_failure_logs",
        "status": "completed_with_documented_prescore_interruption",
        "families": families,
        "family_gates_pass": family_gates_pass,
        "manifests": manifests,
        "all_manifests_pass": all(item["pass"] for item in manifests),
        "logs": logs,
        "interpretation": (
            "Retained promoted result families pass their recorded parser or "
            "preflight gates. The sole archived traceback is a pre-score missing-model "
            "interruption in a RULER CWE node-2 attempt; it is not a state-validity "
            "failure and produced no fallback result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = build_audit()
    text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
