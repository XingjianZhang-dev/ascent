#!/usr/bin/env python3
"""Build the upload-ready Array supplementary-material archive."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import re
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ARCHIVE_ROOT = "ASCENT_Array_Supplementary_Material"
ALLOWED_TREE_SUFFIXES = {".json", ".jsonl", ".csv", ".tex", ".py"}
ARCHIVE_TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt"}
EXCLUDED_PARTS = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    "development",
    "diagnostics",
    "logs",
}
EXCLUDED_NAMES = {"DONE", "git_status.txt", ".DS_Store"}
FORBIDDEN_TEXT = {
    "AI-use disclosure marker": re.compile(
        r"OpenAI|ChatGPT|GPT-5|AI-assisted|generative AI", re.IGNORECASE
    ),
    "publisher tool name": re.compile(r"Codex|Claude|Gemini|Anthropic", re.IGNORECASE),
    "local application state": re.compile(r"\.codex", re.IGNORECASE),
    "compute credential or endpoint": re.compile(
        r"connect\.westd\.seetacloud\.com|root@|sshpass|"
        r"QRuS\+j/jtKut|eTlfB1\+lSGmn|dDT1usMJVPmX|"
        r"(?:api|access|secret)[_-]?(?:key|token)\s*[:=]",
        re.IGNORECASE,
    ),
    "local machine path": re.compile(r"/root/|/Users/|Ascent -- TNNLS", re.IGNORECASE),
    "prior submission venue": re.compile(
        r"\bTNNLS\b|NEUNET(?:-D)?|IEEE Transactions on Neural|"
        r"Neural_Networks_Submission|\bUPLOAD\b|submitted to Neural Networks",
        re.IGNORECASE,
    ),
}

PATH_REPLACEMENTS = (
    ("/root/autodl-tmp/ascent-results/", "artifacts/"),
    ("/root/autodl-tmp/ascent-result-archives/", "artifacts/"),
    ("/root/autodl-tmp/ascent/.venv-gpu/bin", ".venv-gpu/bin"),
    ("/root/autodl-tmp/ascent/", "./"),
    ("/root/autodl-tmp/", "./"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def eligible(path: Path, source_root: Path) -> bool:
    relative = path.relative_to(source_root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.name.startswith("DONE"):
        return False
    if path.suffix == ".log" or path.suffix == ".pyc":
        return False
    return path.suffix in ALLOWED_TREE_SUFFIXES


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_file() and eligible(path, source):
            copy_file(path, destination / path.relative_to(source))


def normalize_text(text: str) -> str:
    """Replace machine-specific paths without changing scientific fields."""
    for source, replacement in PATH_REPLACEMENTS:
        text = text.replace(source, replacement)
    return text


def normalize_release_paths(root: Path) -> None:
    text_suffixes = {".md", ".txt", ".py", ".tex", ".json", ".jsonl", ".csv"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in text_suffixes:
            continue
        original = path.read_text(encoding="utf-8", errors="strict")
        normalized = normalize_text(original)
        if normalized != original:
            path.write_text(normalized, encoding="utf-8")


def clean_tar_member(name: str) -> bool:
    """Keep only portable, structured scientific records from raw run bundles."""
    normalized = name.removeprefix("./")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        return False
    if any(
        part.startswith("._") or part in EXCLUDED_PARTS or part == "cache"
        for part in path.parts
    ):
        return False
    if (
        path.name in EXCLUDED_NAMES
        or path.name in {"MANIFEST.sha256", "SHA256SUMS.txt"}
        or "development" in path.name.lower()
    ):
        return False
    if path.parts and path.parts[0] in {"root", "Users", "private", "tmp"}:
        return False
    return Path(path.name).suffix in ARCHIVE_TEXT_SUFFIXES


def copy_clean_tar(source: Path, destination: Path) -> None:
    """Create a deterministic result-only tarball from a raw experiment archive."""
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[str, bytes]] = []
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not clean_tar_member(member.name):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            payload = extracted.read()
            if Path(member.name).suffix in ARCHIVE_TEXT_SUFFIXES:
                payload = normalize_text(payload.decode("utf-8", errors="strict")).encode("utf-8")
            selected.append((member.name, payload))
    if not selected:
        raise RuntimeError(f"no structured scientific records found in {source}")
    with destination.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as clean_archive:
                for name, payload in sorted(selected):
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    clean_archive.addfile(info, io.BytesIO(payload))


def scan_nested_tar(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not clean_tar_member(member.name):
                raise RuntimeError(f"disallowed member in {path.name}: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"unreadable member in {path.name}: {member.name}")
            if Path(member.name).suffix not in ARCHIVE_TEXT_SUFFIXES:
                continue
            text = extracted.read().decode("utf-8", errors="strict")
            for label, pattern in FORBIDDEN_TEXT.items():
                if label == "publisher tool name":
                    continue
                if pattern.search(text):
                    raise RuntimeError(
                        f"disallowed text in {path.name}:{member.name}: {label}"
                    )


def experiment_closure(project: Path, seeds: list[str]) -> list[Path]:
    pending = [project / "experiments" / name for name in seeds]
    selected: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith("experiments."):
                continue
            dependency = project / (node.module.replace(".", "/") + ".py")
            if dependency.is_file() and dependency not in selected:
                pending.append(dependency)
    return sorted(selected)


def write_manifest(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            lines.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def scan_release(root: Path) -> None:
    bad_paths = []
    bad_text = []
    text_suffixes = {".md", ".txt", ".py", ".tex", ".json", ".jsonl", ".csv"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts):
            bad_paths.append(relative)
        if path.suffix in {".log", ".pyc", ".ipynb"} or path.name.startswith("DONE"):
            bad_paths.append(relative)
        if path.suffix in text_suffixes:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in FORBIDDEN_TEXT.items():
                # Benchmark stories contain the ordinary proper noun "Codex".
                # Scientific records must remain byte-identical, so tool-name
                # screening applies to authored release material, while
                # credential/endpoint screening continues to apply everywhere.
                if (
                    label == "publisher tool name"
                    and path.relative_to(root).parts[0] in {"data", "artifacts"}
                ):
                    continue
                if pattern.search(text):
                    bad_text.append(f"{relative}: {label}")
        if path.name.endswith(".tar.gz"):
            scan_nested_tar(path)
    if bad_paths or bad_text:
        message = "release scan failed"
        if bad_paths:
            message += f"; disallowed paths={sorted(set(bad_paths))}"
        if bad_text:
            message += f"; disallowed text={sorted(set(bad_text))}"
        raise RuntimeError(message)


def make_zip(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            arcname = f"{ARCHIVE_ROOT}/{path.relative_to(source).as_posix()}"
            info = zipfile.ZipInfo(arcname, date_time=(2026, 8, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.name == "verify.py" else 0o644) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build(project: Path, supplement_pdf: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="ascent_supplement_") as temporary:
        staging = Path(temporary) / ARCHIVE_ROOT
        staging.mkdir(parents=True)

        static_files = {
            project / "supplementary/README.md": staging / "README.md",
            project / "supplementary/REPRODUCE.md": staging / "REPRODUCE.md",
            project / "supplementary/requirements.txt": staging / "requirements.txt",
            project / "supplementary/requirements-gpu.txt": staging / "requirements-gpu.txt",
            project / "supplementary/verify.py": staging / "verify.py",
            project / "paper/supplement.tex": staging / "Supplementary_Information.tex",
            supplement_pdf: staging / "Supplementary_Information.pdf",
            project / "pyproject.toml": staging / "pyproject.toml",
        }
        for source, destination in static_files.items():
            copy_file(source, destination)

        copy_tree(project / "ascent", staging / "ascent")

        experiment_seeds = [
            "analyze_array_third_node_audit.py",
            "analyze_babilong_canonical_coscale_16k_confirmation.py",
            "analyze_babilong_canonical_semantic_holdout.py",
            "analyze_babilong_around7b_extension.py",
            "analyze_babilong_qrag_direct_peer.py",
            "analyze_noisy_composition_candidate.py",
            "analyze_babilong_systems.py",
            "analyze_babilong_flops_profile.py",
            "analyze_babilong_generative_factorial.py",
            "analyze_babilong_certified_factorial.py",
            "analyze_ruler_aggregation_confirmation.py",
            "analyze_cross_node_replication.py",
            "render_paper_qwen_confirmation_table.py",
            "render_paper_additional_tables.py",
            "render_cross_task_table.py",
            "run_babilong_prompt.py",
            "run_babilong_qrag_retrieval.py",
            "run_babilong_qrag_reader.py",
            "run_noisy_composition_candidate.py",
            "run_ruler_aggregation_certified.py",
            "profile_babilong_flops.py",
        ]
        for source in experiment_closure(project, experiment_seeds):
            copy_file(source, staging / "experiments" / source.name)

        configs = [
            "babilong_qwen2p5_canonical_coscale_16k_confirmatory.json",
            "babilong_qwen2p5_canonical_semantic_holdout_8k_confirmatory.json",
            "posttraining_noisy_composition_sum_numeric_candidate.json",
            "babilong_around7b_16k_extension_confirmatory.json",
            "babilong_qrag_direct_peer_frozen.json",
            "babilong_qwen2p5_8k_generative_factorial_replication_frozen.json",
            "babilong_smollm2_4k_systems.json",
            "babilong_smollm2_8k_flops_profile.json",
            "babilong_smollm2_8k_generative_factorial_frozen.json",
            "babilong_smollm2_8k_neural_controls_confirmatory.json",
            "babilong_qa78_certified_factorial_smollm2_8k_frozen.json",
            "ruler_cwe_smollm2_8k_certified_confirmatory.json",
            "ruler_cwe_qwen2p5_16k_certified_confirmatory.json",
            "ruler_niah_qwen2p5_full_context_16k_multiquery_width_linear_confirmatory.json",
            "ruler_niah_qwen_full_context_16k_multiquery_width_linear_confirmation_manifest.json",
            "ruler_niah_smollm2_full_context_4k_multiquery_width075_confirmatory.json",
            "ruler_niah_smollm2_full_context_4k_multiquery_width075_confirmation_manifest.json",
        ]
        for name in configs:
            copy_file(project / "configs" / name, staging / "configs" / name)

        data_maps = {
            "babilong_16k_canonical_confirmation": "babilong_16k_canonical_confirmation",
            "babilong_train_8k_semantic_holdout_confirmation": "babilong_train_8k_semantic_holdout_confirmation",
            "posttraining_noisy_composition_sum_numeric_candidate": "posttraining_noisy_composition_sum_numeric_candidate",
            "babilong_1k_8k_neural_control": "babilong_1k_8k_neural_control",
            "babilong_structured_qa78_8k_certified_confirmation": "babilong_structured_qa78_8k_certified_confirmation",
            "ruler_cwe_confirm_seed382101_8k_safe": "ruler_cwe_smollm2_seed382101",
            "ruler_cwe_confirm_seed382202_8k_safe": "ruler_cwe_smollm2_seed382202",
            "ruler_cwe_confirm_seed382303_8k_safe": "ruler_cwe_smollm2_seed382303",
            "ruler_cwe_qwen2p5_confirm_seed384101_16k_safe": "ruler_cwe_qwen2p5_seed384101",
            "ruler_cwe_qwen2p5_confirm_seed384202_16k_safe": "ruler_cwe_qwen2p5_seed384202",
            "ruler_cwe_qwen2p5_confirm_seed384303_16k_safe": "ruler_cwe_qwen2p5_seed384303",
        }
        for source_name, target_name in data_maps.items():
            copy_tree(project / "data" / source_name, staging / "data" / target_name)

        artifact_maps = {
            "artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9": "artifacts/official_16k",
            "artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6": "artifacts/semantic_holdout",
            "artifacts/around7b_formal": "artifacts/around7b",
            "artifacts/remote_results/babilong_qrag_direct_peer_89a8280": "artifacts/qrag",
            "artifacts/remote_results/babilong_4k_systems": "artifacts/systems",
            "artifacts/remote_results/babilong_8k_generative_factorial_audit_693c414": "artifacts/smollm2_generation_factorial",
            "artifacts/remote_results/babilong_qa78_certified_factorial_ab4222a": "artifacts/smollm2_certified_factorial",
            "artifacts/remote_results/cwe_confirm_c749f24": "artifacts/ruler/smollm2_cwe",
            "artifacts/remote_results/qwen_cwe_confirm_cd8352f": "artifacts/ruler/qwen2p5_cwe",
            "artifacts/remote_results/cross_node_replication_c87fa8c": "artifacts/cross_node_replication",
            "artifacts/array_third_node_audit": "artifacts/array_third_node_audit",
        }
        for source_name, target_name in artifact_maps.items():
            copy_tree(project / source_name, staging / target_name)

        copy_tree(
            project / "artifacts/sum_numeric_candidate/confirmation",
            staging / "artifacts/qwen_factorial/confirmation",
        )
        copy_file(
            project / "artifacts/sum_numeric_candidate/confirmation_analysis.json",
            staging / "artifacts/qwen_factorial/analysis.json",
        )
        cross_node = project / "artifacts/sum_numeric_candidate/audit_node2/rounds_5_sum_numeric_candidate_panel_1_qwen2p5-7b-instruct.json"
        copy_file(cross_node, staging / "artifacts/qwen_factorial/cross_node" / cross_node.name)

        copy_file(
            project / "artifacts/remote_results/flops_e49eae9/analysis.json",
            staging / "artifacts/flops/analysis.json",
        )
        copy_tree(
            project / "artifacts/remote_results/flops_e49eae9/flops_e49eae9",
            staging / "artifacts/flops/raw",
        )
        copy_clean_tar(
            project / "artifacts/remote_results/babilong_8k_neural_controls_072adbd.tar.gz",
            staging / "artifacts/flops/babilong_8k_neural_controls_072adbd.tar.gz",
        )

        for source_name, target_name in (
            ("qwen_niah_fullctx_16k_multiquery_width_linear_confirmation_372xxx.tar.gz", "qwen_niah_16k_multiquery.tar.gz"),
            ("smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz", "smollm2_4k_multiquery.tar.gz"),
        ):
            copy_clean_tar(
                project / "artifacts/remote_results" / source_name,
                staging / "artifacts/ruler" / target_name,
            )

        copy_tree(project / "paper/generated", staging / "derived/tables")
        copy_tree(project / "paper/generated", staging / "generated")
        copy_tree(project / "paper/data/evidence", staging / "derived/figure_data")
        copy_file(
            project / "paper/figures/systems_efficiency.pdf",
            staging / "figures/systems_efficiency.pdf",
        )

        normalize_release_paths(staging)
        write_manifest(staging)
        scan_release(staging)
        make_zip(staging, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--supplement-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.root.resolve()
    output = args.output.resolve()
    build(project, args.supplement_pdf.resolve(), output)
    archive_digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{archive_digest}  {output.name}\n", encoding="utf-8"
    )
    print(output)
    print(f"sha256 {archive_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
