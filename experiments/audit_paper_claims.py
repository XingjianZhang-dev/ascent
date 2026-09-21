#!/usr/bin/env python3
"""Fail closed when the Array manuscript drifts from source evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from PIL import Image

from experiments.render_paper_additional_tables import (
    render_around7b,
    render_mechanism,
    render_systems,
)
from experiments.render_paper_qwen_confirmation_table import render as render_qwen
from experiments.render_cross_task_table import (
    load_single_analysis,
    render as render_cross_task,
)
from experiments.render_theory_evidence_map import render as render_theory_evidence
from experiments.audit_manuscript_consistency import audit as audit_consistency


def citation_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        keys.update(key.strip() for key in group.split(","))
    return keys


def bibliography_keys(*bibs: str) -> set[str]:
    return set(re.findall(r"@[A-Za-z]+\{([^,]+),", "\n".join(bibs)))


def environment_text(tex: str, name: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(name)}\}}(.*?)\\end\{{{re.escape(name)}\}}",
        tex,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"{name} environment is missing")
    return match.group(1)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*", text))


def section_text(tex: str, title: str) -> str:
    match = re.search(
        rf"\\section\{{{re.escape(title)}\}}(.*?)(?=\\section\{{|\\section\*\{{|$)",
        tex,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError(f"section is missing: {title}")
    return match.group(1)


def command_output(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def pdf_pages(pdf: Path) -> int:
    text = command_output("pdfinfo", str(pdf))
    match = re.search(r"^Pages:\s+(\d+)$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not read page count: {pdf}")
    return int(match.group(1))


def font_audit(pdf: Path) -> tuple[bool, bool]:
    lines = command_output("pdffonts", str(pdf)).splitlines()[2:]
    data = [line for line in lines if line.strip()]
    no_type3 = all("Type 3" not in line for line in data)
    all_embedded = all("yes" in line.split() for line in data)
    return no_type3, all_embedded


def audit(root: Path) -> dict:
    paper = root / "paper"
    tex = (paper / "main.tex").read_text()
    appendix = (paper / "appendix.tex").read_text()
    legacy_bib = (paper / "references.bib").read_text()
    verified_bib = (paper / "references_verified.bib").read_text()

    official = json.loads(
        (root / "artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json").read_text()
    )
    semantic = json.loads(
        (root / "artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6/analysis.json").read_text()
    )
    mechanism = json.loads(
        (root / "artifacts/sum_numeric_candidate/confirmation_analysis.json").read_text()
    )
    around7b = json.loads((root / "artifacts/around7b_formal/analysis.json").read_text())
    systems = json.loads(
        (root / "artifacts/remote_results/babilong_4k_systems/analysis.json").read_text()
    )
    flops = json.loads(
        (root / "artifacts/remote_results/flops_e49eae9/analysis.json").read_text()
    )
    qwen_multiquery = load_single_analysis(
        root
        / "artifacts/remote_results/"
        "qwen_niah_fullctx_16k_multiquery_width_linear_confirmation_372xxx.tar.gz"
    )
    smol_multiquery = load_single_analysis(
        root
        / "artifacts/remote_results/"
        "smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz"
    )
    smol_cwe = json.loads(
        (
            root / "artifacts/remote_results/cwe_confirm_c749f24/analysis.json"
        ).read_text()
    )
    qwen_cwe = json.loads(
        (
            root / "artifacts/remote_results/qwen_cwe_confirm_cd8352f/analysis.json"
        ).read_text()
    )
    generative_factorial = json.loads(
        (
            root
            / "artifacts/remote_results/babilong_8k_generative_factorial_audit_693c414/factorial_analysis.json"
        ).read_text()
    )
    generative_factorial_audit = json.loads(
        (
            root
            / "artifacts/remote_results/babilong_8k_generative_factorial_audit_693c414/audit.json"
        ).read_text()
    )
    certified_factorial = json.loads(
        (root / "artifacts/remote_results/babilong_qa78_certified_factorial_ab4222a/factorial_analysis.json").read_text()
    )
    reference_manifest = json.loads((paper / "data/reference_verification.json").read_text())
    submission = paper / "submission"
    highlights = (submission / "Highlights.txt").read_text()
    cover_letter = (submission / "Cover_Letter.md").read_text()
    submission_metadata = (submission / "SUBMISSION_METADATA.md").read_text()

    abstract_words = word_count(environment_text(tex, "abstract"))
    keywords = [item.strip() for item in environment_text(tex, "keyword").split(r"\sep")]
    cited = citation_keys(tex + "\n" + appendix)
    defined = bibliography_keys(legacy_bib, verified_bib)
    bbl = (paper / "build/main.bbl").read_text()
    reference_list = set(
        re.findall(r"\\bibitem(?:\[[^]]*\])?\{([^}]+)\}", bbl)
    )
    limitations = section_text(tex, "Limitations")
    nonlimitations = tex.replace(limitations, "")

    checks: dict[str, bool] = {}
    consistency = audit_consistency(root)
    checks["numeric_and_reference_consistency"] = consistency["pass"]
    checks["elsarticle_array_target"] = (
        r"\documentclass[preprint,12pt]{elsarticle}" in tex
        and r"\journal{Array}" in tex
    )
    # Array uses single-anonymized review: the compiled manuscript carries the
    # complete title page (author, affiliation, ORCID, corresponding email) and
    # the named build is the default.
    checks["single_anonymized_title_page_ready"] = all(
        marker not in tex
        for marker in (
            "Author details required before submission",
            "Affiliation required before submission",
            "double-anonymized",
        )
    ) and all(
        marker in tex
        for marker in (
            r"\newif\ifblindreview",
            r"\ifdefined\ASCENTBlindVersion",
            r"\blindreviewfalse",
            r"\author[aff1]{Xingjian Zhang\corref{cor1}}",
            "xingjiaz@andrew.cmu.edu",
            "https://orcid.org/0009-0009-9141-8819",
            "Carnegie Mellon University",
            "https://orcid.org/0009-0009-9141-8819",
            "Carnegie Mellon University",
            "5000 Forbes Avenue",
        )
    ) and "aff2" not in tex and (paper / "submission/Title_Page.docx").is_file()
    checks["abstract_150_to_250_words"] = 150 <= abstract_words <= 250
    checks["five_keywords"] = len(keywords) == 5 and all(keywords)
    checks["at_least_80_unique_citations"] = len(cited) >= 80
    checks["all_citations_defined"] = cited <= defined
    checks["reference_list_exactly_matches_citations"] = reference_list == cited
    checks["verified_reference_manifest"] = (
        reference_manifest.get("count") == 80
        and len(reference_manifest.get("records", [])) == 80
        and all(
            record.get("key")
            and record.get("title")
            and record.get("authors")
            and record.get("year")
            and str(record.get("url", "")).startswith("https://arxiv.org/abs/")
            for record in reference_manifest.get("records", [])
        )
    )

    checks["primary_table_exactly_regenerated"] = (
        (paper / "generated/qwen_confirmations.tex").read_text()
        == render_qwen(official, semantic)
    )
    checks["mechanism_table_exactly_regenerated"] = (
        (paper / "generated/scale_interaction_confirmation.tex").read_text()
        == render_mechanism(mechanism)
    )
    checks["around7b_table_exactly_regenerated"] = (
        (paper / "generated/around7b_results.tex").read_text()
        == render_around7b(around7b)
    )
    checks["systems_table_exactly_regenerated"] = (
        (paper / "generated/systems_results.tex").read_text()
        == render_systems(systems, flops)
    )
    checks["theory_evidence_table_exactly_regenerated"] = (
        (paper / "generated/theory_evidence_map.tex").read_text()
        == render_theory_evidence(official, mechanism, around7b, systems)
    )
    checks["cross_task_table_exactly_regenerated"] = (
        (paper / "generated/cross_task_confirmations.tex").read_text()
        == render_cross_task(qwen_multiquery, smol_multiquery, smol_cwe, qwen_cwe)
    )
    checks["cross_task_inference_scoped"] = (
        "with a\npositive log-scale slope" in tex
        and "upper adjacent interval is .0956" in limitations
        and "Positive scaling slope" in (
            paper / "generated/cross_task_confirmations.tex"
        ).read_text()
    )
    checks["theory_experiment_closure_explicit"] = all(
        marker in tex
        for marker in (
            r"\label{prop:no-universal}",
            r"\label{prop:information-value}",
            r"\label{prop:finite-reader}",
            r"\label{prop:footprint}",
            r"\subsection{Theory--Experiment Correspondence}",
            r"\input{generated/theory_evidence_map.tex}",
            "13,824 endpoint--transition row checks",
        )
    )
    checks["primary_confirmation_gates_pass"] = (
        official["primary_gate_pass"] is True
        and semantic["primary_gate_pass"] is True
    )
    checks["mechanism_simultaneous_family_passes"] = (
        mechanism["registered_performance_pass"] is True
        and mechanism["supplementary_bonferroni_familywise_performance_pass"] is True
        and all(
            item["ci95_low"] > 0
            for item in mechanism["supplementary_bonferroni_familywise_estimands"].values()
        )
    )
    checks["latest_factorials_faithfully_scoped"] = (
        generative_factorial_audit["audit_pass"] is True
        and generative_factorial_audit["all_27_decode_and_score_runs_exact_to_previous_factorial"] is True
        and generative_factorial_audit["all_nine_diagonal_state_payloads_exact_to_pre_inventory_reference"] is True
        and generative_factorial_audit["factorial_interval_fields_exact"] is True
        and generative_factorial["gates"]["diagonal_gain_adjacent_lcbs_positive"] is True
        and generative_factorial["gates"]["registered_interaction_lcbs_positive"] is False
        and certified_factorial["gates"]["diagonal_gain_adjacent_lcbs_positive"] is True
        and certified_factorial["gates"]["foundation_by_state_interaction_lcbs_positive"] is False
        and re.search(r"same-refinement\s+interactions are \.0417", appendix) is not None
        and "same-refinement interaction intervals overlap zero" in limitations
    )
    checks["registered_fourth_scale_failure_in_limitations"] = (
        "increment of $-.1588$\n[$-.2055,-.1120$]" in limitations
        and "-.2055,-.1120" not in nonlimitations
    )
    checks["systems_wall_failure_in_limitations_or_appendix"] = (
        "wall-time ratios of .9008, .8651, and .7262 are reported as measured" in limitations
        and "Wall ratios of .9008, .8651, and .7262\nare reported separately" in appendix
    )

    figure_names = (
        "ascent_method",
        "primary_scaling",
        "factorial_interaction",
        "large_model_breadth",
        "systems_efficiency",
        "latest_factorial_boundaries",
    )
    checks["complete_publication_figure_assets"] = all(
        (paper / "figures" / f"{name}{suffix}").is_file()
        for name in figure_names
        for suffix in (".pdf", ".svg", ".png", "_grayscale.png", "_preview.png")
    )
    checks["main_and_appendix_figures_included"] = all(
        marker in tex + "\n" + appendix
        for marker in (
            "figures/ascent_method.pdf",
            "figures/primary_scaling.pdf",
            "figures/factorial_interaction.pdf",
            "figures/large_model_breadth.pdf",
            "figures/systems_efficiency.pdf",
            "figures/latest_factorial_boundaries.pdf",
        )
    )
    highlight_bullets = [
        line.removeprefix("•").strip()
        for line in highlights.splitlines()
        if line.strip().startswith("•")
    ]
    checks["elsevier_highlights_ready"] = (
        3 <= len(highlight_bullets) <= 5
        and all(0 < len(item) <= 85 for item in highlight_bullets)
    )
    graphical_abstract = submission / "Graphical_Abstract.png"
    graphical_width, graphical_height = (
        Image.open(graphical_abstract).size
        if graphical_abstract.is_file()
        else (0, 0)
    )
    checks["graphical_abstract_package_ready"] = (
        graphical_width >= 1328
        and graphical_height >= 531
        and 2.45 <= graphical_width / graphical_height <= 2.55
        and all(
            (submission / f"Graphical_Abstract{suffix}").is_file()
            for suffix in (".pdf", ".svg", ".png", "_grayscale.png", "_preview.png")
        )
    )
    graphical_no_type3, graphical_embedded = (
        font_audit(submission / "Graphical_Abstract.pdf")
        if (submission / "Graphical_Abstract.pdf").is_file()
        else (False, False)
    )
    checks["graphical_abstract_fonts_ready"] = (
        graphical_no_type3 and graphical_embedded
    )
    checks["cover_letter_scientific_content_ready"] = all(
        phrase in cover_letter
        for phrase in (
            "Contribution",
            "Significance and evidence",
            "Artificial Intelligence and Machine Learning",
            "Data, Knowledge and Intelligent Systems",
            "BABILong",
            "third physical-node audit",
        )
    )
    checks["array_scope_selected"] = all(
        phrase in submission_metadata
        for phrase in (
            "Regular paper (original research article; not a Technical Note)",
            "Artificial Intelligence and Machine Learning",
            "Data, Knowledge and Intelligent Systems",
            "double anonymized",
        )
    )
    checks["public_repository_availability_ready"] = (
        r"\input{appendix}" not in tex
        and "https://github.com/XingjianZhang-dev/ascent" in tex
        and "upon reasonable request" not in tex
        and "Supplementary Material archive submitted with this" not in tex
    )
    checks["elsevier_ai_disclosure_present"] = all(
        phrase in tex
        for phrase in (
            "Declaration of generative AI and AI-assisted technologies",
            "ChatGPT and OpenAI Codex",
            "Claude Code (Anthropic)",
            "takes full responsibility for the content of",
        )
    )
    # The manuscript declaration must equal, word for word, the declaration
    # published in the repository's AI_ASSISTANCE.md (release_docs/ in the
    # development tree, repository root in the public tree).
    def _normal(text: str) -> str:
        text = text.replace("\\texttt{make reproduce}", "`make reproduce`").replace("\\texttt{AI\\_ASSISTANCE.md}", "AI_ASSISTANCE.md")
        return re.sub(r"\s+", " ", text).strip()
    declaration_match = re.search(r"manuscript preparation process\}\s*(.*?)\s*\\bibliographystyle", tex, re.S)
    repo_file = next((c for c in (root / "release_docs/AI_ASSISTANCE.md", root / "AI_ASSISTANCE.md") if c.is_file()), None)
    repo_declaration = None
    if repo_file is not None:
        quoted = re.search(r"^> (.+)$", repo_file.read_text(), re.M)
        repo_declaration = quoted.group(1) if quoted else None
    checks["ai_declaration_matches_repository_file"] = bool(
        declaration_match and repo_declaration and _normal(declaration_match.group(1)) == _normal(repo_declaration)
    )
    checks["no_defensive_stock_phrasing_outside_limitations"] = not any(
        phrase in nonlimitations
        for phrase in (
            "does not invalidate",
            "not universally superior",
            "the appropriate conclusion is scoped",
            "failed development",
            "strict wall gate",
            "not labeled an official test score",
            "retained rather than removed",
        )
    )
    checks["scope_boundaries_confined_to_limitations"] = all(
        marker in limitations and marker not in nonlimitations
        for marker in (
            "-.2055,-.1120",
            "-.0043",
            "not at 0.5B",
            "interaction intervals overlap zero",
        )
    )
    checks["report_style_sectioning_removed"] = not any(
        marker in tex + "\n" + appendix
        for marker in (
            r"\section{Failure-Preserving Interface Audit}",
            r"\section{Registered Statistical Estimands}",
            r"\section{Failure Analysis and Limitations}",
            r"\subsection{Prospective Confirmation}",
            "confirmation panels",
            "strict wall gate",
        )
    )
    checks["ai_disclosure_localized"] = (
        tex.count("ChatGPT") == 1
        and tex.count("Claude Code") == 1
        and "ChatGPT" not in appendix
        and "GPT-5.6" not in tex + "\n" + appendix
    )
    normalized_appendix = re.sub(r"\s+", " ", appendix)
    checks["reproduction_contract_present"] = all(
        phrase in normalized_appendix
        for phrase in (
            "configuration hash",
            "SHA-256 hashes",
            "all 256 fifteen-way distributions",
            "configuration metadata agree exactly",
            "identity of code, checkpoints, alignment, and numerical results",
        )
    )
    normalized_tex = re.sub(r"\s+", " ", tex)
    checks["author_declarations_ready"] = all(
        phrase in normalized_tex
        for phrase in (
            "CRediT authorship contribution statement",
            "Xingjian Zhang: Conceptualization, Methodology, Software",
            "This research did not receive any specific grant",
            "The author declares that he has no known competing financial interests",
            "Data and code availability",
            "https://github.com/XingjianZhang-dev/ascent",
            "10.5281/zenodo.",
            "benchmark corpora are not redistributed",
            "Third-party model weights are not redistributed",
        )
    )

    pdf = paper / "build/main.pdf"
    log = (paper / "build/main.log").read_text() if (paper / "build/main.log").is_file() else ""
    checks["compiled_pdf_exists"] = pdf.is_file()
    pages = pdf_pages(pdf) if pdf.is_file() else 0
    checks["substantive_long_form_article"] = 20 <= pages <= 40
    checks["no_undefined_references_or_citations"] = not re.search(
        r"undefined references|undefined citation|Citation .* undefined|Reference .* undefined",
        log,
        re.IGNORECASE,
    )
    overfull = re.findall(
        r"Overfull \\[hv]box \(([0-9.]+)pt too (?:wide|high)\)([^\n]*)",
        log,
    )
    checks["no_material_overfull_boxes"] = all(
        float(amount) <= 3.0 and "while \\output is active" in context
        for amount, context in overfull
    )
    if pdf.is_file():
        no_type3, all_embedded = font_audit(pdf)
    else:
        no_type3, all_embedded = False, False
    checks["no_type3_fonts"] = no_type3
    checks["all_fonts_embedded"] = all_embedded

    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "pass": not failed,
        "target": "Array (Elsevier), regular paper",
        "abstract_words": abstract_words,
        "keywords": len(keywords),
        "unique_citations": len(cited),
        "compiled_pages": pages,
        "consistency_audit": {
            "statistical_summaries": consistency[
                "independently_recalculated_statistical_summaries"
            ],
            "plot_data_comparisons": consistency["plot_data_value_comparisons"],
            "promoted_claims": consistency["promoted_narrative_claims_checked"],
            "cited_references": consistency["references"]["cited_keys"],
        },
        "checks": checks,
        "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
