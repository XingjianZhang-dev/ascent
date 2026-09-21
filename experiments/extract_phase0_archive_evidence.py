#!/usr/bin/env python3
"""Extract node-identity and third-instance worktree evidence from frozen archives.

Phase 0 (Array revision 1), questions Q2 and Q3. The three compute instances
used for cross-node audits each produced a metadata capture that lives only
inside a frozen ``.tar.zst`` archive:

* node 1 and node 2: ``GPU_NODES_DISASTER_RECOVERY_20260815/*.tar.zst``
  (captured 2026-08-15T09:28Z, ``GPU_BACKUP_METADATA/``)
* node 3: ``backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst``
  (captured 2026-08-23T14:59Z, ``BACKUP_METADATA/``)

This script verifies each archive's recorded SHA-256, streams the archive
read-only, copies the small metadata files it needs into
``reports/phase0_archive_evidence/<node>/`` with their SHA-256 values, and
writes a machine-readable summary. The archives are never modified.

Requires the ``zstd`` binary (tar streaming) — no Python zstandard module.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any


ARCHIVES = {
    "node1": {
        "archive": "GPU_NODES_DISASTER_RECOVERY_20260815/ASCENT_GPU_node1_port32784_20260815_dr.tar.zst",
        "sha256_file": "GPU_NODES_DISASTER_RECOVERY_20260815/ASCENT_GPU_node1_port32784_20260815_dr.tar.zst.sha256",
        "member_dir_suffix": "GPU_BACKUP_METADATA",
        "role": "factorial and official-panel primary node (node1 in artifact paths)",
    },
    "node2": {
        "archive": "GPU_NODES_DISASTER_RECOVERY_20260815/ASCENT_GPU_node2_port10826_20260815_dr.tar.zst",
        "sha256_file": "GPU_NODES_DISASTER_RECOVERY_20260815/ASCENT_GPU_node2_port10826_20260815_dr.tar.zst.sha256",
        "member_dir_suffix": "GPU_BACKUP_METADATA",
        "role": "second node used for audit reruns (node2 / audit_node2 in artifact paths)",
    },
    "node3": {
        "archive": "backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst",
        "sha256_file": "backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst.sha256",
        "member_dir_suffix": "BACKUP_METADATA",
        "role": "third instance used for the Qwen3-8B post-hoc audit rerun (artifacts/array_third_node_audit)",
    },
}
WANTED = (
    "HOSTNAME.txt",
    "DATE_UTC.txt",
    "UNAME.txt",
    "NVIDIA_SMI.txt",
    "NVIDIA_SMI_TABLE.txt",
    "CPU.txt",
    "MEMORY.txt",
    "OS_RELEASE.txt",
    "GIT_HEAD.txt",
    "GIT_STATUS.txt",
    "GIT_DIFF.patch",
    "GIT_REPOSITORIES.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(text: str) -> str:
    """The capture files store '$ cmd', '# exit_code=N', then output."""
    lines = text.splitlines()
    body = [line for line in lines if not (line.startswith("$ ") or line.startswith("# exit_code="))]
    return "\n".join(body).strip()


def nvidia_field(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text, flags=re.M)
    return match.group(1) if match else None


def stream_members(archive: Path, suffix: str) -> dict[str, bytes]:
    proc = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    found: dict[str, bytes] = {}
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) < 2 or parts[-2] != suffix or parts[-1] not in WANTED:
                continue
            handle = tar.extractfile(member)
            if handle is not None:
                found[parts[-1]] = handle.read()
    proc.stdout.close()
    proc.wait()
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--evidence-dir", type=Path, default=Path("reports/phase0_archive_evidence"))
    parser.add_argument("--output", type=Path, default=Path("reports/NODE_IDENTITY_EVIDENCE_2026-09-20.json"))
    args = parser.parse_args()

    nodes: dict[str, Any] = {}
    for node, spec in ARCHIVES.items():
        archive = args.root / spec["archive"]
        recorded = (args.root / spec["sha256_file"]).read_text().split()[0]
        actual = sha256_file(archive)
        if actual != recorded:
            raise RuntimeError(f"{archive}: sha256 {actual} != recorded {recorded}")
        members = stream_members(archive, spec["member_dir_suffix"])
        out_dir = args.evidence_dir / node
        out_dir.mkdir(parents=True, exist_ok=True)
        copied = {}
        for name, data in sorted(members.items()):
            (out_dir / name).write_bytes(data)
            copied[name] = {"bytes": len(data), "sha256": sha256_bytes(data)}
        smi = members["NVIDIA_SMI.txt"].decode()
        nodes[node] = {
            "role": spec["role"],
            "archive": spec["archive"],
            "archive_sha256_verified": actual,
            "metadata_member_dir": spec["member_dir_suffix"],
            "capture_utc": command_output(members["DATE_UTC.txt"].decode()),
            "hostname": command_output(members["HOSTNAME.txt"].decode()),
            "uname": command_output(members["UNAME.txt"].decode()),
            "gpu": {
                "product_name": nvidia_field(smi, "Product Name"),
                "uuid": nvidia_field(smi, "GPU UUID"),
                "serial_number": nvidia_field(smi, "Serial Number"),
                "pci_bus_id": nvidia_field(smi, "Bus Id"),
                "driver_version": nvidia_field(smi, "Driver Version"),
                "cuda_version": nvidia_field(smi, "CUDA Version"),
            },
            "cpu_model": (re.search(r"Model name:\s*(.+)", members["CPU.txt"].decode()) or [None, None])[1],
            "memory_total": (re.search(r"Mem:\s+(\S+)", members["MEMORY.txt"].decode()) or [None, None])[1],
            "git_head": command_output(members["GIT_HEAD.txt"].decode()) if "GIT_HEAD.txt" in members else None,
            "git_status_modified_tracked": (
                [line[3:] for line in command_output(members["GIT_STATUS.txt"].decode()).splitlines() if line.startswith(" M ")]
                if "GIT_STATUS.txt" in members
                else None
            ),
            "git_diff_patch_sha256": copied.get("GIT_DIFF.patch", {}).get("sha256"),
            "copied_files": copied,
        }

    uuids = [n["gpu"]["uuid"] for n in nodes.values()]
    serials = [n["gpu"]["serial_number"] for n in nodes.values()]
    drivers = [n["gpu"]["driver_version"] for n in nodes.values()]
    summary = {
        "question": "Q3: are the three cross-node compute instances physically distinct?",
        "gpu_uuids_all_distinct": len(set(uuids)) == len(uuids),
        "gpu_serials_all_distinct": len(set(serials)) == len(serials),
        "driver_versions_all_distinct": len(set(drivers)) == len(drivers),
        "node1_node2_captured_same_instant_with_different_drivers": (
            nodes["node1"]["capture_utc"] == nodes["node2"]["capture_utc"]
            and nodes["node1"]["gpu"]["driver_version"] != nodes["node2"]["gpu"]["driver_version"]
        ),
        "caveat": (
            "GPU identity was captured from the same container hostnames after the runs "
            "(node1/node2: 2026-08-15T09:28Z, about nine hours after their last retained record; "
            "node3: 2026-08-23T14:59Z, about thirteen hours after its run). The result records "
            "themselves store hostname and GPU product name, not UUID."
        ),
        "nodes": nodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for node, info in nodes.items():
        print(f"{node}: {info['hostname']} | {info['gpu']['uuid']} | serial {info['gpu']['serial_number']} | driver {info['gpu']['driver_version']} | captured {info['capture_utc']}")
    print(f"uuids distinct={summary['gpu_uuids_all_distinct']} serials distinct={summary['gpu_serials_all_distinct']} drivers distinct={summary['driver_versions_all_distinct']}")
    print(f"written: {args.output}  sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
