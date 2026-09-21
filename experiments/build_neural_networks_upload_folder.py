#!/usr/bin/env python3
"""Assemble the exact files needed for an Array initial submission."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path


FOLDER_NAME = "Array_UPLOAD_READY_2026-08-23_v9_PORTAL_MATCHED_FINAL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1])
    raise RuntimeError(f"unable to read page count: {path}")


def build(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing folder: {output}")

    paper = root / "paper"
    submission = paper / "submission"
    files = {
        "01_ASCENT_Anonymous_Manuscript.pdf": paper / "build/main.pdf",
        "02_ASCENT_Cover_Letter.pdf": submission / "Cover_Letter.pdf",
        "03_ASCENT_Highlights.txt": submission / "Highlights.txt",
        "04_REQUIRED_System_Competing_Interests.docx": (
            submission / "Declaration_of_Interests.docx"
        ),
        "05_REQUIRED_System_Portal_Declarations.docx": submission / "Portal_Declarations.docx",
        "06_OPTIONAL_ASCENT_Graphical_Abstract.pdf": submission / "Graphical_Abstract.pdf",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing upload files: " + ", ".join(missing))

    if pdf_pages(files["01_ASCENT_Anonymous_Manuscript.pdf"]) != 28:
        raise RuntimeError("unexpected manuscript page count")
    if pdf_pages(files["02_ASCENT_Cover_Letter.pdf"]) != 1:
        raise RuntimeError("cover letter is not one page")
    highlights = [
        line.strip()[1:].strip()
        for line in files["03_ASCENT_Highlights.txt"].read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("•")
    ]
    if not (3 <= len(highlights) <= 5 and all(len(line) <= 85 for line in highlights)):
        raise RuntimeError("highlights violate the journal limits")
    output.mkdir(parents=True)
    for name, source in files.items():
        destination = output / name
        shutil.copy2(source, destination)

    readme = """ARRAY — INITIAL SUBMISSION UPLOAD ORDER

This folder contains the checked files for a regular Article submission. The
compiled manuscript PDF is the main manuscript; editable source is retained in
the separate archival submission bundle.

UPLOAD THESE FIVE REQUIRED FILES

1. 01_ASCENT_Anonymous_Manuscript.pdf
   File type: Manuscript
   This is the double-anonymized reviewer PDF. It contains no author name,
   affiliation, contact details, acknowledgements, funding, interests, or
   CRediT statement, and its PDF author metadata is blank.

2. 02_ASCENT_Cover_Letter.pdf
   File type: Cover letter

3. 03_ASCENT_Highlights.txt
   File type: Highlights

4. 04_REQUIRED_System_Competing_Interests.docx
   File type: Declaration of competing interests
   This is the unmodified Word file downloaded from Elsevier's declaration
   interface; it is not a locally authored substitute.

5. 05_REQUIRED_System_Portal_Declarations.docx
   File type: Declarations
   This is the separate, unmodified Word file generated through Elsevier's
   portal declaration step. Both declaration uploads are system-generated.

OPTIONAL FILE

6. 06_OPTIONAL_ASCENT_Graphical_Abstract.pdf
   Upload only if the portal presents a dedicated Graphical Abstract file type.
   Do not misclassify this file as an ordinary Figure.

DO NOT UPLOAD AT INITIAL SUBMISSION

- The LaTeX source bundle: the portal explicitly says it is not needed until
  revision.
- Separate Figure or Table files: all article figures and tables are embedded
  in the manuscript PDF.
- The named internal manuscript PDF or `main_named.tex`. The portal requires a
  double-anonymized reviewer manuscript. Author details are entered in
  Editorial Manager's author-information fields.
- `Title_Page.docx`: the current portal provides no Title page item type. Keep
  this author-identifying file only in the local archival bundle; do not
  misclassify it under Manuscript, Declarations, Cover letter, or Supplementary
  Material.
- Any Supplementary Material ZIP. The manuscript states that the data and code
  supporting the findings are available from the corresponding author upon
  reasonable request.
- This README or SHA256SUMS.txt.

The required generative-AI disclosure is already placed directly before the
references in the manuscript PDF.
"""
    (output / "00_READ_ME_FIRST.txt").write_text(readme, encoding="utf-8")

    checksums = "".join(
        f"{sha256(output / name)}  {name}\n" for name in sorted(files)
    )
    (output / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")

    for name, source in files.items():
        if sha256(output / name) != sha256(source):
            raise RuntimeError(f"copy verification failed: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "paper/submission" / FOLDER_NAME
    build(root, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
