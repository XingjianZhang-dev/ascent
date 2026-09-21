#!/usr/bin/env python3
"""Record a complete environment snapshot for a revision-stage run directory.

Every Phase 2 run in the Array revision writes its results into a fresh
directory under ``artifacts_revision/``. This script is invoked before the
first GPU job of that directory and again after the last one. It records what
the retained 2026-08 records did not: GPU UUID, serial number, PCI bus id and
driver; full ``nvidia-smi -q``; ``lscpu``; ``/etc/os-release``; ``pip freeze``;
Python/PyTorch/Transformers/CUDA/cuDNN versions; the repository commit and
dirty flag; and the SHA-256 of the pre-registration file the run is bound to.

Usage::

    python experiments/record_run_environment.py \
        --run-dir artifacts_revision/crossnode_2026-09 \
        --preregistration artifacts_revision/crossnode_2026-09/PREREGISTRATION.json \
        --phase before

The snapshot is ``<run-dir>/environment_<phase>.json``; the raw command
outputs are stored beside it under ``<run-dir>/environment_<phase>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_COMMANDS = {
    "nvidia_smi_q.txt": ["nvidia-smi", "-q"],
    "nvidia_smi_table.txt": ["nvidia-smi"],
    "lscpu.txt": ["lscpu"],
    "free.txt": ["free", "-b"],
    "os_release.txt": ["cat", "/etc/os-release"],
    "uname.txt": ["uname", "-a"],
    "pip_freeze.txt": [sys.executable, "-m", "pip", "freeze"],
    "nvcc_version.txt": ["nvcc", "--version"],
    "git_status_porcelain.txt": ["git", "-C", str(ROOT), "status", "--porcelain"],
    "git_head.txt": ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
    "git_log_1.txt": ["git", "-C", str(ROOT), "log", "-1", "--format=%H%n%ai%n%s"],
}


def run_capture(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        return 127, f"{error}"
    return completed.returncode, completed.stdout + completed.stderr


def gpu_query() -> list[dict[str, str]]:
    fields = "index,name,uuid,serial,driver_version,pci.bus_id,memory.total,compute_cap"
    code, out = run_capture(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    rows = []
    if code == 0:
        for line in out.strip().splitlines():
            values = [v.strip() for v in line.split(",")]
            rows.append(dict(zip(fields.split(","), values, strict=False)))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda
        versions["cudnn"] = str(torch.backends.cudnn.version())
        versions["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        versions["cuda_capability"] = (
            ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None
        )
        versions["tf32_matmul_allowed"] = str(torch.backends.cuda.matmul.allow_tf32)
        versions["tf32_cudnn_allowed"] = str(torch.backends.cudnn.allow_tf32)
        versions["cudnn_deterministic"] = str(torch.backends.cudnn.deterministic)
        versions["cudnn_benchmark"] = str(torch.backends.cudnn.benchmark)
        versions["deterministic_algorithms"] = str(torch.are_deterministic_algorithms_enabled())
    except Exception as error:  # pragma: no cover - environment dependent
        versions["torch_error"] = repr(error)
    for name in ("transformers", "tokenizers", "safetensors", "sentencepiece", "numpy", "accelerate", "huggingface_hub"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except Exception:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()

    raw_dir = args.run_dir / f"environment_{args.phase}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_records = {}
    for filename, command in RAW_COMMANDS.items():
        code, out = run_capture(command)
        (raw_dir / filename).write_text(out)
        raw_records[filename] = {"command": command, "exit_code": code, "sha256": sha256_file(raw_dir / filename)}

    head = (raw_dir / "git_head.txt").read_text().strip()
    porcelain = (raw_dir / "git_status_porcelain.txt").read_text()
    snapshot = {
        "schema_version": 1,
        "phase": args.phase,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpus": gpu_query(),
        "libraries": library_versions(),
        "git": {
            "commit": head,
            "dirty": bool(porcelain.strip()),
            "porcelain_lines": porcelain.strip().splitlines(),
        },
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": sha256_file(args.preregistration),
        },
        "raw_captures": raw_records,
    }
    out_path = args.run_dir / f"environment_{args.phase}.json"
    out_path.write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({k: snapshot[k] for k in ("hostname", "gpus", "git", "preregistration")}, indent=2))
    print(f"written: {out_path}")


if __name__ == "__main__":
    main()
