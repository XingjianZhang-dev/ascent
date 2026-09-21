#!/usr/bin/env python3
"""Fetch pinned official BABILong parquet files and freeze JSONL sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_row(row: dict[str, Any]) -> dict[str, str]:
    return {
        "input": str(row["input"]),
        "question": str(row["question"]),
        "target": str(row["target"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--revision",
        default="fc4d1a584dfc498c37578753bee4cdd91b987ae2",
    )
    parser.add_argument("--dataset-config", default="8k")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["qa1", "qa2", "qa3"],
    )
    args = parser.parse_args()

    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task in args.tasks:
        repository_path = (
            f"{args.dataset_config}/{task}-00000-of-00001.parquet"
        )
        downloaded = Path(
            hf_hub_download(
                repo_id="RMT-team/babilong-1k-samples",
                filename=repository_path,
                repo_type="dataset",
                revision=args.revision,
            )
        )
        table = parquet.read_table(downloaded, columns=["input", "question", "target"])
        rows = [normalized_row(row) for row in table.to_pylist()]
        output = args.output_root / f"{task}.jsonl"
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        records.append(
            {
                "task": task,
                "repository_path": repository_path,
                "downloaded_sha256": sha256_file(downloaded),
                "jsonl_path": output.name,
                "jsonl_sha256": sha256_file(output),
                "rows": len(rows),
            }
        )
    manifest = {
        "repository": "RMT-team/babilong-1k-samples",
        "repository_type": "dataset",
        "revision": args.revision,
        "dataset_config": args.dataset_config,
        "files": records,
    }
    manifest_path = args.output_root / "SOURCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


if __name__ == "__main__":
    main()
