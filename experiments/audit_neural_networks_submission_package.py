#!/usr/bin/env python3
"""Fail-closed audit for the Array Editorial Manager package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

from experiments.audit_manuscript_consistency import audit as audit_consistency
from experiments.audit_paper_claims import audit as audit_claims, word_count
from experiments.build_neural_networks_submission_bundle import (
    ARCHIVE_ROOT as SUBMISSION_ARCHIVE_ROOT,
    source_map as submission_source_map,
)


BUNDLE = "ASCENT_Array_submission_bundle_2026-08-23.zip"
UPLOAD_FOLDER = "Array_UPLOAD_READY_2026-08-23_v9_PORTAL_MATCHED_FINAL"
GUIDE_URL = "https://www.sciencedirect.com/journal/array/publish/guide-for-authors"
SYSTEM_INTEREST_SHA256 = "b13ee67aeb715a2c52ea373da1d077ddfda834a0f433f187774682ff7ff90206"
SYSTEM_DECLARATIONS_SHA256 = "198b30f36ed33ca9ea11a0369cb71747d4ef2e635e12be66f937fbfad41b2926"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(tex: str, name: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(.*?)\\end\{{{re.escape(name)}\}}",
        tex,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"missing {name} environment")
    return match.group(1)


def pdf_pages(path: Path) -> int:
    output = subprocess.check_output(["pdfinfo", str(path)], text=True)
    match = re.search(r"^Pages:\s+(\d+)", output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot read page count: {path}")
    return int(match.group(1))


def docx_text_and_revision_state(path: Path) -> tuple[str, bool, bool]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        document = archive.read("word/document.xml").decode("utf-8")
        text = " ".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", document))
        tracked = bool(re.search(r"<w:(?:ins|del)(?:\s|>)", document))
        comments = any(
            name.startswith("word/comments")
            and bool(re.search(r"<w:comment(?:\s|>)", archive.read(name).decode("utf-8")))
            for name in names
        )
    return text, tracked, comments


def check_submission_bundle(path: Path, root: Path) -> dict[str, bool | int]:
    prefix = f"{SUBMISSION_ARCHIVE_ROOT}/"
    manifest_name = prefix + "MANIFEST.sha256"
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        names = archive.namelist()
        manifest = archive.read(manifest_name).decode("utf-8").splitlines()
        expected = {}
        for line in manifest:
            digest, relative = line.split("  ", 1)
            expected[relative] = digest
        actual = {
            name[len(prefix) :]
            for name in names
            if name.startswith(prefix) and not name.endswith("/") and name != manifest_name
        }
        hashes_match = set(expected) == actual and all(
            hashlib.sha256(archive.read(prefix + relative)).hexdigest() == digest
            for relative, digest in expected.items()
        )
        clean = not any(
            name.endswith((".log", ".aux", ".pyc", ".ipynb", ".DS_Store"))
            or re.search(r"(?:^|/)(?:tmp|diagnostics|development)(?:/|$)", name)
            for name in names
        )
        required = all(
            prefix + name in names
            for name in (
                "Manuscript/ASCENT_anonymous_manuscript.pdf",
                "Manuscript/ASCENT_named_internal.pdf",
                "Manuscript/main.tex",
                "Manuscript/main_named.tex",
                "Submission_Items/Highlights.txt",
                "Submission_Items/Title_Page.docx",
                "Submission_Items/Cover_Letter.pdf",
                "Submission_Items/Declaration_of_Interests.docx",
                "Submission_Items/Portal_Declarations.docx",
                "Submission_Items/Graphical_Abstract.pdf",
            )
        )
        no_supplement = not any("Supplementary_Material" in name for name in names)
        matches_workspace = all(
            archive.read(prefix + name) == source.read_bytes()
            for name, source in submission_source_map(root).items()
        )
    return {
        "submission_bundle_crc_pass": bad is None,
        "submission_bundle_manifest_pass": hashes_match,
        "submission_bundle_clean_paths": clean,
        "submission_bundle_required_files": required,
        "no_supplement_in_submission_bundle": no_supplement,
        "submission_bundle_matches_workspace": matches_workspace,
        "submission_bundle_manifested_files": len(expected),
    }


def audit(root: Path) -> dict:
    paper = root / "paper"
    submission = paper / "submission"
    tex = (paper / "main.tex").read_text(encoding="utf-8")
    named_wrapper = (paper / "main_named.tex").read_text(encoding="utf-8")
    anonymous_pdf = paper / "build/main.pdf"
    named_pdf = paper / "build/main_named.pdf"
    anonymous_text = subprocess.check_output(
        ["pdftotext", str(anonymous_pdf), "-"], text=True
    )
    named_text = subprocess.check_output(["pdftotext", str(named_pdf), "-"], text=True)
    named_text_flat = re.sub(r"\s+", " ", named_text)
    anonymous_info = subprocess.check_output(["pdfinfo", str(anonymous_pdf)], text=True)
    cover = (submission / "Cover_Letter.md").read_text(encoding="utf-8")
    cover_flat = re.sub(r"\s+", " ", cover)
    metadata = (submission / "SUBMISSION_METADATA.md").read_text(encoding="utf-8")
    highlights = [
        line.strip()[1:].strip()
        for line in (submission / "Highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("•")
    ]

    checks: dict[str, bool] = {}
    checks["official_elsarticle_source"] = (
        r"\documentclass[preprint,12pt]{elsarticle}" in tex
        and r"\journal{Array}" in tex
    )
    identity_tokens = (
        "Xingjian Zhang",
        "Carnegie Mellon University",
        "xingjiaz@andrew.cmu.edu",
        "5000 Forbes Avenue",
        "0009-0009-9141-8819",
        "+1 917 940 6399",
        "+86 139 5192 4078",
    )
    checks["double_anonymized_source_workflow"] = all(
        phrase in tex
        for phrase in (
            r"\newif\ifblindreview",
            r"\ifdefined\ASCENTNamedVersion",
            r"\author[]{}",
            r"\hypersetup{pdfauthor={}}",
        )
    ) and all(
        phrase in named_wrapper for phrase in (r"\def\ASCENTNamedVersion{}", r"\input{main.tex}")
    )
    checks["anonymous_pdf_has_no_identity"] = not any(
        token.casefold() in anonymous_text.casefold() for token in identity_tokens
    )
    author_metadata = re.search(r"^Author:\s*(.*)$", anonymous_info, re.MULTILINE)
    checks["anonymous_pdf_author_metadata_blank"] = (
        author_metadata is None or not author_metadata.group(1).strip(" ;")
    )
    checks["named_internal_record_complete"] = all(
        token in named_text_flat for token in identity_tokens[:5]
    ) and r"\corref{cor1}" in tex
    checks["array_scope_selected"] = all(
        phrase in metadata
        for phrase in (
            "Regular paper (original research article; not a Technical Note)",
            "Artificial Intelligence and Machine Learning",
            "Data, Knowledge and Intelligent Systems",
        )
    )
    abstract_words = word_count(environment(tex, "abstract"))
    checks["abstract_at_most_250_words"] = abstract_words <= 250
    keywords = [item.strip() for item in environment(tex, "keyword").split(r"\sep")]
    checks["at_most_six_keywords"] = 1 <= len(keywords) <= 6
    checks["numbered_article_sections"] = all(
        marker in tex
        for marker in (
            r"\section{Introduction}",
            r"\section{Problem Formulation}",
            r"\section{ASCENT}",
            r"\section{Experimental Protocol}",
            r"\section{Results}",
            r"\section{Conclusion}",
        )
    )
    checks["main_has_no_duplicate_appendix"] = r"\input{appendix}" not in tex
    checks["author_declarations_conditionally_separated"] = all(
        tex.index(marker) < tex.index(r"\bibliography{references,references_verified}")
        for marker in (
            r"\section*{CRediT authorship contribution statement}",
            r"\section*{Funding}",
            r"\section*{Declaration of competing interest}",
            r"\section*{Data and code availability}",
            r"\section*{Declaration of generative AI",
        )
    ) and not any(
        heading in anonymous_text
        for heading in (
            "CRediT authorship contribution statement",
            "Funding",
            "Declaration of competing interest",
        )
    ) and all(
        heading in named_text
        for heading in (
            "CRediT authorship contribution statement",
            "Funding",
            "Declaration of competing interest",
        )
    )
    checks["highlights_count_and_length"] = (
        3 <= len(highlights) <= 5 and all(len(line) <= 85 for line in highlights)
    )
    checks["cover_establishes_array_fit"] = all(
        phrase in cover_flat
        for phrase in (
            "Contribution",
            "Significance and evidence",
            "Artificial Intelligence and Machine Learning",
            "Data, Knowledge and Intelligent Systems",
            "third physical-node audit",
        )
    )
    checks["cover_confirms_original_submission"] = all(
        phrase in cover_flat
        for phrase in (
            "no prior conference version",
            "not under consideration elsewhere",
            "approved for submission",
        )
    )
    checks["cover_letter_one_page"] = pdf_pages(submission / "Cover_Letter.pdf") == 1
    checks["cover_confirms_double_anonymized_portal_workflow"] = (
        "double-anonymized review workflow" in cover_flat
        and "Editorial Manager's author metadata" in cover_flat
        and "reviewer manuscript is fully anonymized" in cover_flat
    )
    request_statement = (
        "the data and code supporting the findings of this study are available "
        "from the corresponding author upon reasonable request"
    )
    checks["request_based_data_code_statement_consistent"] = all(
        request_statement in re.sub(r"\s+", " ", text).lower()
        for text in (tex, cover, metadata, (submission / "AUTHOR_DECLARATIONS.md").read_text(encoding="utf-8"))
    )
    checks["no_supplement_claim_in_current_submission_text"] = not any(
        re.search(r"Supplementary Material archive|Supplementary Material ZIP|separate Supplementary", text, re.IGNORECASE)
        for text in (tex, cover, metadata, (submission / "AUTHOR_DECLARATIONS.md").read_text(encoding="utf-8"))
    )

    graphical = submission / "Graphical_Abstract.png"
    with Image.open(graphical) as image:
        width, height = image.size
        dpi = image.info.get("dpi", (0, 0))
    checks["graphical_abstract_dimensions"] = width >= 1328 and height >= 531
    checks["graphical_abstract_resolution"] = min(dpi) >= 300

    declaration = submission / "Declaration_of_Interests.docx"
    declaration_text, tracked, comments = docx_text_and_revision_state(declaration)
    checks["system_generated_interest_docx"] = (
        "no known competing financial interests" in declaration_text
        and not tracked
        and not comments
        and sha256(declaration) == SYSTEM_INTEREST_SHA256
    )
    portal_declarations = submission / "Portal_Declarations.docx"
    portal_text, portal_tracked, portal_comments = docx_text_and_revision_state(
        portal_declarations
    )
    checks["portal_generated_declarations_docx"] = (
        "Declaration of interests" in portal_text
        and "no known competing financial interests" in portal_text
        and not portal_tracked
        and not portal_comments
        and sha256(portal_declarations) == SYSTEM_DECLARATIONS_SHA256
    )
    title_page = submission / "Title_Page.docx"
    title_text, title_tracked, title_comments = docx_text_and_revision_state(title_page)
    checks["separate_title_page_complete"] = all(
        phrase in title_text
        for phrase in (
            "ASCENT: Scale-Complementary External State",
            "Xingjian Zhang",
            "Carnegie Mellon University",
            "xingjiaz@andrew.cmu.edu",
            "5000 Forbes Avenue",
            "0009-0009-9141-8819",
            "Acknowledgements",
            "Funding",
            "Declaration of interests",
            "CRediT authorship contribution statement",
        )
    ) and not title_tracked and not title_comments
    bundle_path = submission / BUNDLE
    bundle_checks = check_submission_bundle(bundle_path, root)
    checks.update({key: bool(value) for key, value in bundle_checks.items() if isinstance(value, bool)})
    bundle_sidecar = submission / f"{BUNDLE}.sha256"
    bundle_expected_hash = bundle_sidecar.read_text(encoding="utf-8").split()[0]
    checks["submission_bundle_sidecar_matches"] = sha256(bundle_path) == bundle_expected_hash

    upload_dir = submission / UPLOAD_FOLDER
    upload_names = {
        path.name for path in upload_dir.iterdir() if path.is_file()
    } if upload_dir.is_dir() else set()
    expected_upload_names = {
        "00_READ_ME_FIRST.txt",
        "01_ASCENT_Anonymous_Manuscript.pdf",
        "02_ASCENT_Cover_Letter.pdf",
        "03_ASCENT_Highlights.txt",
        "04_REQUIRED_System_Competing_Interests.docx",
        "05_REQUIRED_System_Portal_Declarations.docx",
        "06_OPTIONAL_ASCENT_Graphical_Abstract.pdf",
        "SHA256SUMS.txt",
    }
    checks["portal_matched_upload_folder"] = upload_names == expected_upload_names
    checks["no_title_page_in_upload_folder"] = not any(
        "title_page" in name.casefold() for name in upload_names
    )

    claims = audit_claims(root)
    consistency = audit_consistency(root)
    checks["scientific_claim_audit"] = bool(claims["pass"])
    checks["numeric_reference_audit"] = bool(consistency["pass"])
    checks["main_pdf_technical_range"] = 15 <= pdf_pages(anonymous_pdf) <= 35

    failures = [name for name, passed in checks.items() if not passed]
    return {
        "target": "Array (Elsevier), regular paper",
        "guide_checked": "2026-08-23",
        "guide_url": GUIDE_URL,
        "pass": not failures,
        "failed_checks": failures,
        "checks": checks,
        "metrics": {
            "anonymous_main_pdf_pages": pdf_pages(anonymous_pdf),
            "named_internal_pdf_pages": pdf_pages(named_pdf),
            "abstract_words": abstract_words,
            "keywords": len(keywords),
            "highlights": len(highlights),
            "highlight_max_characters": max(map(len, highlights)),
            "graphical_abstract_pixels": [width, height],
            "graphical_abstract_dpi": list(dpi),
            "submission_bundle_manifested_files": bundle_checks[
                "submission_bundle_manifested_files"
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
