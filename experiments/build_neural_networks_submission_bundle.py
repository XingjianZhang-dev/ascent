#!/usr/bin/env python3
"""Build the curated Array submission-source archive."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


ARCHIVE_ROOT = "ASCENT_Array_Submission"
DEFAULT_NAME = "ASCENT_Array_submission_bundle_2026-08-23.zip"
FIXED_TIME = (2026, 8, 23, 0, 0, 0)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_map(root: Path) -> dict[str, Path]:
    paper = root / "paper"
    submission = paper / "submission"
    mapping = {
        "README.txt": submission / "BUNDLE_README.txt",
        "Manuscript/ASCENT_anonymous_manuscript.pdf": paper / "build/main.pdf",
        "Manuscript/ASCENT_named_internal.pdf": paper / "build/main_named.pdf",
        "Manuscript/main.tex": paper / "main.tex",
        "Manuscript/main_named.tex": paper / "main_named.tex",
        "Manuscript/references.bib": paper / "references.bib",
        "Manuscript/references_verified.bib": paper / "references_verified.bib",
        "Manuscript/generated/qwen_confirmations.tex": paper / "generated/qwen_confirmations.tex",
        "Manuscript/generated/theory_evidence_map.tex": paper / "generated/theory_evidence_map.tex",
        "Submission_Items/Highlights.txt": submission / "Highlights.txt",
        "Submission_Items/Title_Page.docx": submission / "Title_Page.docx",
        "Submission_Items/Cover_Letter.pdf": submission / "Cover_Letter.pdf",
        "Submission_Items/Cover_Letter.tex": submission / "Cover_Letter.tex",
        "Submission_Items/Cover_Letter.md": submission / "Cover_Letter.md",
        "Submission_Items/Declaration_of_Interests.docx": submission / "Declaration_of_Interests.docx",
        "Submission_Items/Portal_Declarations.docx": submission / "Portal_Declarations.docx",
        "Submission_Items/Graphical_Abstract.pdf": submission / "Graphical_Abstract.pdf",
        "Submission_Items/Graphical_Abstract.png": submission / "Graphical_Abstract.png",
        "Submission_Items/Graphical_Abstract.svg": submission / "Graphical_Abstract.svg",
        "Submission_Items/SUBMISSION_METADATA.md": submission / "SUBMISSION_METADATA.md",
        "Submission_Items/AUTHOR_DECLARATIONS.md": submission / "AUTHOR_DECLARATIONS.md",
    }
    for stem in ("ascent_method", "primary_scaling", "factorial_interaction", "large_model_breadth"):
        for suffix in ("pdf", "png", "svg"):
            mapping[f"Manuscript/figures/{stem}.{suffix}"] = paper / "figures" / f"{stem}.{suffix}"
    return mapping


def archive_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build(root: Path, output: Path) -> tuple[int, str]:
    mapping = source_map(root)
    missing = [str(path) for path in mapping.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required bundle files: " + ", ".join(missing))

    payloads = {name: path.read_bytes() for name, path in mapping.items()}
    banned = (".log", ".aux", ".pyc", ".ipynb", ".DS_Store")
    if any(name.endswith(banned) for name in payloads):
        raise RuntimeError("development artifact entered the submission whitelist")

    manifest = "".join(
        f"{sha256(data)}  {name}\n" for name, data in sorted(payloads.items())
    ).encode("utf-8")
    payloads["MANIFEST.sha256"] = manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payloads.items()):
            archive.writestr(archive_info(name), data)

    archive_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    sidecar.write_text(f"{archive_digest}  {output.name}\n", encoding="utf-8")
    return len(payloads), archive_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "paper/submission" / DEFAULT_NAME
    count, digest = build(root, output)
    print(f"built {output} ({count} files)")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
