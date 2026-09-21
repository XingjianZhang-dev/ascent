#!/usr/bin/env python3
"""Record the Zenodo *version* DOI everywhere the placeholder appears.

Usage (from the development tree, after Zenodo has minted the DOI for the
GitHub release):

    python experiments/record_zenodo_doi.py 10.5281/zenodo.1234567

Replaces ``10.5281/zenodo.<version-id>`` in the manuscript, the cover and
response letters, README.md, CITATION.cff (also un-commenting the identifier
block) and VERIFICATION.md, then prints what to do next (recompile, re-run the
audits, commit, rebuild the release tree, push to main; no new tag). Nothing under
artifacts/ is touched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "10.5281/zenodo.<version-id>"
FILES = [
    "paper/main.tex",
    "paper/submission/COVER_LETTER_R1.tex",
    "paper/submission/RESPONSE_TO_REVIEWERS_R1.md",
    "paper/submission/SUBMISSION_METADATA_R1.md",
    "release_docs/README.md",
    "release_docs/VERIFICATION.md",
    "release_docs/CITATION.cff",
]


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"10\.5281/zenodo\.\d+", sys.argv[1]):
        raise SystemExit("usage: record_zenodo_doi.py 10.5281/zenodo.<digits>")
    doi = sys.argv[1]
    for rel in FILES:
        path = ROOT / rel
        text = path.read_text()
        count = text.count(PLACEHOLDER)
        text = text.replace(PLACEHOLDER, doi)
        if rel.endswith("CITATION.cff"):
            text = text.replace(
                "# identifiers:\n#   - type: doi\n#     value: 10.5281/zenodo.<version-id>   # Zenodo version DOI, added once minted\n",
                f"identifiers:\n  - type: doi\n    value: {doi}\n",
            ).replace(f"#     value: {doi}", f"    value: {doi}")
            count += 1
        path.write_text(text)
        print(f"{rel}: {count} replacement(s)")
    print(
        "\nnext:\n"
        "  (cd paper && tectonic --keep-logs --keep-intermediates --outdir build main.tex)\n"
        "  (cd paper/submission && tectonic --outdir . COVER_LETTER_R1.tex && pandoc RESPONSE_TO_REVIEWERS_R1.md -o RESPONSE_TO_REVIEWERS_R1.pdf --pdf-engine=tectonic)\n"
        "  .venv/bin/python -m experiments.audit_manuscript_consistency --root . && .venv/bin/python -m experiments.audit_paper_claims --root .\n"
        "  git commit -am 'Record the Zenodo version DOI'\n"
        "  python experiments/build_release_tree.py && python experiments/build_ai_assistance_table.py && python experiments/build_release_tree.py\n"
        "  (cd release/ascent && git add -A && git commit -m 'Record the Zenodo version DOI' && git push)   # no new tag: the manuscript cites the v1.0 version DOI\n"
    )


if __name__ == "__main__":
    main()
