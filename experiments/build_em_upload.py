#!/usr/bin/env python3
"""Assemble the Editorial Manager upload folder for the Array revision.

Produces one self-contained LaTeX file (every ``\\input`` flattened, including the
generated tables, and the compiled bibliography inlined so no BibTeX run is
needed), the figures with flat file names, EPS copies of the figures, the .bbl and
.bib files, and the letters. Usage::

    python experiments/build_em_upload.py --out ~/Desktop/EM_Upload

Requires ``paper/build/main.bbl`` from the current manuscript build and
``pdftops`` (poppler) for the EPS copies.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX_NAME = "ASCENT_manuscript_Revision"
LETTERS = ["RESPONSE_TO_REVIEWERS_Revision.pdf", "COVER_LETTER_Revision.pdf", "Highlights_Revision.txt"]


def flatten(text: str) -> str:
    def repl(match: re.Match) -> str:
        name = match.group(1)
        path = PAPER / (name if name.endswith(".tex") else name + ".tex")
        return flatten(path.read_text().rstrip("\n"))
    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path.home() / "Desktop/EM_Upload")
    args = parser.parse_args()
    out = args.out
    (out / "EPS_backup").mkdir(parents=True, exist_ok=True)

    flat = flatten((PAPER / "main.tex").read_text())
    flat, n_fig = re.subn(r"\\includegraphics(\[[^\]]*\])?\{figures/([^}]+)\}", r"\\includegraphics\1{\2}", flat)
    bbl = (PAPER / "build/main.bbl").read_text().rstrip("\n")
    assert flat.count("\\bibliography{references,references_verified}") == 1
    flat = flat.replace("\\bibliographystyle{elsarticle-num}\n", "")
    flat = flat.replace(
        "\\bibliography{references,references_verified}",
        "% bibliography compiled with elsarticle-num from references.bib and references_verified.bib (main.bbl inlined)\n" + bbl,
    )
    assert "\\input{" not in flat and "figures/" not in flat
    (out / f"{TEX_NAME}.tex").write_text(flat)

    figures = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", flat)))
    for name in figures:
        shutil.copy2(PAPER / "figures" / name, out / name)
        subprocess.run(["pdftops", "-eps", str(PAPER / "figures" / name), str(out / "EPS_backup" / (Path(name).stem + ".eps"))], check=True)
    shutil.copy2(PAPER / "build/main.bbl", out / f"{TEX_NAME}.bbl")
    for bib in ("references.bib", "references_verified.bib"):
        shutil.copy2(PAPER / bib, out / bib)
    for letter in LETTERS:
        shutil.copy2(PAPER / "submission" / letter, out / letter)
    print(f"{out}: {TEX_NAME}.tex ({flat.count(chr(10))} lines, {flat.count('tabular')//2} tables, "
          f"{flat.count(chr(92)+'bibitem')} references), {n_fig} figures + EPS copies, .bbl, 2 .bib, {len(LETTERS)} letters")


if __name__ == "__main__":
    main()
