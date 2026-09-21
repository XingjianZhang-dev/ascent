#!/usr/bin/env python3
"""Read-only integrity verifier for the ASCENT core reproduction package."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(document: dict[str, Any], keys: list[str]) -> Any:
    value: Any = document
    for key in keys:
        value = value[key]
    return value


def verify(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    rows: list[dict[str, Any]] = []
    for specification in manifest["files"]:
        path = ROOT / specification["path"]
        exists = path.is_file()
        actual_hash = sha256_file(path) if exists else None
        row: dict[str, Any] = {
            "path": specification["path"],
            "role": specification["role"],
            "exists": exists,
            "expected_sha256": specification["sha256"],
            "actual_sha256": actual_hash,
            "hash_matches": actual_hash == specification["sha256"],
        }
        if exists and specification.get("json"):
            json.loads(path.read_text())
            row["json_parses"] = True
        if exists and "tar_analysis_member" in specification:
            with tarfile.open(path, "r:gz") as archive:
                extracted = archive.extractfile(specification["tar_analysis_member"])
                if extracted is None:
                    raise FileNotFoundError(specification["tar_analysis_member"])
                analysis = json.load(extracted)
            row["embedded_gate_value"] = nested_value(
                analysis, specification["required_true_path"]
            )
            row["embedded_gate_passes"] = row["embedded_gate_value"] is True
        rows.append(row)
    gates = {
        "all_files_exist": all(row["exists"] for row in rows),
        "all_hashes_match": all(row["hash_matches"] for row in rows),
        "all_json_files_parse": all(
            row.get("json_parses", True) for row in rows
        ),
        "all_embedded_gates_pass": all(
            row.get("embedded_gate_passes", True) for row in rows
        ),
    }
    gates["package_verification_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "files": rows,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest)
    print(json.dumps(result, indent=2))
    if not result["gates"]["package_verification_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
