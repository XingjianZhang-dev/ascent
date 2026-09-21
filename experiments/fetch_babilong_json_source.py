#!/usr/bin/env python3
"""Fetch and normalize a pinned BABILong JSON source split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.fetch_babilong_official_source import normalized_row, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task in args.tasks:
        repository_path = f"data/{task}/{args.dataset_config}.json"
        downloaded = Path(
            hf_hub_download(
                repo_id=args.repository,
                filename=repository_path,
                repo_type="dataset",
                revision=args.revision,
            )
        )
        document = json.loads(downloaded.read_text())
        if not isinstance(document, list):
            raise RuntimeError(f"expected row list: {repository_path}")
        rows = [normalized_row(row) for row in document]
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
        "repository": args.repository,
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
