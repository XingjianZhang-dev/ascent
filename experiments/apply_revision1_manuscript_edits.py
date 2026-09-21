#!/usr/bin/env python3
"""Apply the Array revision-1 edits to paper/main.tex as exact-match replacements.

Each edit is (id, reviewer point, old, new). The script refuses to run if any
``old`` string is not found exactly once, so the manuscript state is always
known. It writes paper/CHANGELOG_REVISION1.md from the same list.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper/main.tex"

EDITS: list[tuple[str, str, str, str]] = [
    (
        "front-matter-single-blind",
        "R2-8 (metadata); Array is single-anonymized",
        r"""% Editorial Manager requires a double-anonymized review manuscript. Compile
% main.tex directly for the anonymized reviewer version. Define
% \ASCENTNamedVersion before inputting this file to build the internal named
% record used for production after acceptance.
\newif\ifblindreview
\ifdefined\ASCENTNamedVersion
  \blindreviewfalse
\else
  \blindreviewtrue
\fi""",
        r"""% Array uses single-anonymized review: the title page carries the author,
% affiliation, ORCID, funding, competing-interest and CRediT statements.
% Define \ASCENTBlindVersion before inputting this file only if an anonymized
% build is ever required.
\newif\ifblindreview
\ifdefined\ASCENTBlindVersion
  \blindreviewtrue
\else
  \blindreviewfalse
\fi""",
    ),
    (
        "abstract-7b-regression",
        "R2-1",
        r"""0.5B, 1.5B, and 3B parameters, with both adjacent directional contrasts
significant after family-wise correction. A three-model by three-state""",
        r"""0.5B, 1.5B, and 3B parameters, with both adjacent directional contrasts
significant after family-wise correction. Accuracy gains are not monotone
across every scale: at 7B the Foundation reader improves and the four-slot
state saturates, producing a 3B-to-7B accuracy increment of $-15.88$ points
[$-20.55$, $-11.20$]. The accuracy-side claim is therefore established from
0.5B to 3B; the 7B conclusion rests on the non-saturated
negative-log-likelihood factorial. A three-model by three-state""",
    ),
    (
        "abstract-scope-and-reproduction",
        "R2-5, R2-7; cross-node scope",
        r"""checkpoints at the two larger shared scales. Cross-node reproduction,
representation controls, retrieval baselines, and systems accounting complete
the empirical closure. The results establish scale-complementary external
state across the evaluated frozen long-context language-model systems.""",
        r"""checkpoints at the two larger shared scales. Matched-hardware reproduction, a
cross-architecture sensitivity analysis, controls, baselines, and systems
accounting complete the closure. The results establish scale-complementary
external state under the evaluated benchmarks, metrics, and state schema.""",
    ),
    (
        "intro-exact-reproduction",
        "cross-node scope",
        r"""ablations, systems accounting, and exact cross-node reproduction. This design""",
        r"""ablations, systems accounting, cross-node reproduction on matched hardware,
and a cross-architecture sensitivity analysis. This design""",
    ),
    (
        "contribution-exact-reproducibility",
        "cross-node scope",
        r"""  cross-task controls, systems measurements, and exact cross-node reproduction
  establish architectural breadth and exact reproducibility.""",
        r"""  cross-task controls, systems measurements, and cross-node reproduction
  establish architectural breadth and reproducibility; every reported number
  is recomputable from the released per-row records.""",
    ),
    (
        "prop3-indices-remark",
        "theory clarification (7.11)",
        r"""The preceding statement concerns an oracle decoder. To connect it to frozen
finite readers, define the nonnegative excess risk""",
        r"""The indices $m_i$ in Proposition~\ref{prop:oracle-complementarity} carry no
force: with oracle decoders the gain depends only on the nested state. The
reader-dependent content of scale complementarity is isolated in
Proposition~\ref{prop:finite-reader}, whose interaction term is what the
factorial estimates.

The preceding statement concerns an oracle decoder. To connect it to frozen
finite readers, define the nonnegative excess risk""",
    ),
    (
        "sec4.1-fail-closed",
        "C-15 (implementation-description correction)",
        r"""Each endpoint executes the same five operations: (i) write state from $X$
before $Q$ and $Y$ are revealed; (ii) validate types, ordering, and the budget;
(iii) select at most $s$ query-relevant entries; (iv) serialize those entries
with the fixed canonical template; and (v) decode with the unchanged model and
scorer. A failed validity check sets $E_{s,Q}=\varnothing$ and therefore
returns the foundation path. Equation~\eqref{eq:ascent-contract} isolates the""",
        r"""Each endpoint executes the same five operations: (i) write state from $X$
before $Q$ and $Y$ are revealed; (ii) validate registered types, ordering,
hashes, alignment, and budget constraints; (iii) select at most $s$
query-relevant entries; (iv) serialize those entries with the fixed canonical
template; and (v) decode with the unchanged model and scorer. Validation is
fail-closed: a missing, malformed, or mismatched state aborts the invocation
rather than silently substituting a Foundation output.
Equation~\eqref{eq:ascent-contract} isolates the""",
    ),
    (
        "sec4.2-per-row-blindness",
        "R2-2, R2-6 (target-blindness instrumented per row)",
        r"""contains no instance answer. Exhaustive parser checks precede model decoding,
and state entries preserve document order under deterministic tie-breaking.""",
        r"""contains no instance answer. Exhaustive parser checks precede model decoding,
and state entries preserve document order under deterministic tie-breaking.
Every BABILong result row released with this revision carries a
machine-checkable record that the writer, readout, and prompt builder read
only the input and the question, that the reference answer was inaccessible
until after the prompt had been rendered, and that every retained fact is a
verbatim span of the input at its recorded position (Section~\ref{sec:repro}).
Target-blindness is a property of the pipeline; it does not by itself
establish that the state schema generalizes beyond the evaluated task
families, which Section~\ref{sec:limitations} addresses separately.""",
    ),
    (
        "sec4.3-fail-closed",
        "C-15",
        r"""full factorial estimates their interaction. Deterministic validity checks
route missing or malformed state to the foundation path.""",
        r"""full factorial estimates their interaction. Deterministic validity checks
are fail-closed; missing or malformed state is not scored as an \ascent{}
observation.""",
    ),
    (
        "sec5.1-smollm2-scales",
        "clarity",
        r"""and 7B parameters and SmolLM2 \cite{allal2025smollm2} readers at three ordered
scales.""",
        r"""and 7B parameters and SmolLM2 \cite{allal2025smollm2} readers at 135M, 360M,
and 1.7B parameters.""",
    ),
    (
        "sec6.2-cross-node-scope",
        "R2-5; integrity bullet 3",
        r"""probability identity across $K$, and all 13,824 endpoint--transition row checks
hold across the factorial. The complete 7B/$K=5$ probability vectors and joint NLLs reproduce
exactly on a second physical node.""",
        r"""probability identity across $K$, and all 13,824 endpoint--transition row checks
hold across the factorial; each row check confirms that the state retained at
the larger $K$ extends the smaller state as a prefix and adds an observation,
which holds by construction of the nested writer and is an implementation
check rather than empirical evidence (Table~\ref{tab:theory-evidence}, note;
the per-row exact-posterior tallies, which are not guaranteed, are in
\ref{app:power}). On the preregistered development
panel, an audit rerun on a second compute instance with a physically distinct
GPU reproduced all 256 prediction records---the 512 arm-level 15-way
probability vectors (7,680 stored probabilities), all row-level joint NLLs,
and every prespecified scientific field---identically in their saved IEEE-754
binary64 representation. A revision-stage rerun of factorial panels 2 and 3
on a third instance with a different GPU architecture (NVIDIA A100) changed
the argmax candidate of one to two rows per 256-row panel and moved each
panel's NLL gain by about 0.01 nats (Section~\ref{sec:repro}).""",
    ),
    (
        "sec6.3-third-node-scope",
        "R2-5; Phase 0 Q1--Q3",
        r"""practically important 7B--8B regime. A posthoc audit on a third physical node
reproduces all 800 Qwen3-8B paired predictions and every scientific field
exactly; it is excluded from the prespecified inferential family.""",
        r"""practically important 7B--8B regime. A post-hoc audit rerun on a third,
separately provisioned compute instance with a physically distinct GPU
reproduced all 800 Qwen3-8B paired predictions and every prespecified
scientific field identically; it is a code audit on matched hardware rather
than an independent replication (Section~\ref{sec:repro}) and is excluded
from the prespecified inferential family.""",
    ),
    (
        "sec7-scope-statement-and-holm",
        "R2-7; R1 (marginal p_Holm); R2-4",
        r"""\section{Limitations}

The empirical claim is deliberately scale- and estimand-specific. Official
BABILong accuracy establishes increasing gains from 0.5B to 3B. At 7B,""",
        r"""\section{Limitations}
\label{sec:limitations}

\paragraph{Scope of the claim} Scale complementarity is demonstrated here for
Qwen2.5 readers on official BABILong and RULER tasks, under accuracy and under
candidate-normalized negative log-likelihood, with a structured-fact state
schema. It is not shown to be a general property of frozen long-context
readers with external memory. Within this scope three results are
unfavourable and are reported as such: the accuracy gain decreases from 3B to
7B; the upper adjacent interval for Qwen2.5 common-word aggregation includes
zero; and the SmolLM2 same-refinement interaction intervals include zero. The
explanations that follow do not remove these results from the record.

\paragraph{Schema scope} The writer schema used here records typed events,
assignment edges, occurrence-indexed key--value evidence, and count summaries.
These are the structures present in the evaluated structured-fact-retrieval
benchmarks. The present evidence therefore establishes scale complementarity
for external state constructed against such structures, and does not
establish it as a general property of frozen readers with arbitrary external
memory. \ref{app:schema-blind} reports the identical pipeline with
three schema-free writers (a sentence-recency window, a lexical retrieval
chain, and BGE-M3 hybrid retrieval with reranking) in place of the
structured-fact writer. None satisfies the registered co-scaling rule: at
the same one-to-three-slot budget the generic state is worse than no state
at every scale (gains of $-5$ to $-18$ points), and the structured-fact
writer's advantage over each generic writer is 15 points at 0.5B and 89--99
points at 1.5B and 3B. The pattern reported here is therefore a property of
external state constructed against this schema. Extending the writer to
non-templated natural text is the principal open direction.

\paragraph{Unfavourable and marginal results} The empirical claim is
deliberately scale- and estimand-specific. Official BABILong accuracy
establishes increasing gains from 0.5B to 3B; the second increment
(1.5B$\to$3B, 3.75 points) is significant but marginal after Holm correction
($p_{\mathrm{Holm}}=0.0255$), and \ref{app:power} shows that the
ten-panel design detects an increment of this size with power of about .52,
against a minimum detectable increment of about 5.3--5.8 points at 80\%
power. At 7B,""",
    ),
    (
        "sec7-delete-tenfold",
        "R2-7",
        r"""supported-operator FLOP reductions; its profiler-enclosed wall-time ratios of
.9008, .8651, and .7262 do not constitute a tenfold end-to-end speedup.""",
        r"""supported-operator FLOP reductions; its profiler-enclosed end-to-end
wall-time ratios of .9008, .8651, and .7262 are reported as measured.""",
    ),
    (
        "sec8-rewrite",
        "R2-3, R2-5, R2-6; integrity bullets; Phase 0; Phase 2A/2B",
        r"""\section{Reproducibility and Ethics}

The retained research record pins dataset and model revisions, records file
sizes and SHA-256 hashes, captures hardware and software environments, and
regenerates the analyses from immutable manifests. Cross-node reruns reproduce every
selected prediction, including all 256 probability vectors and scientific
fields in the 7B mechanism study. A separate third-node audit reproduces all
800 Qwen3-8B paired predictions and scientific fields exactly across the ten
official 16K panels. The writer is target-blind, and no test answer is inserted
into state. The implementation preserves provenance and supports abstention
for domain-specific deployment.""",
        r"""\section{Reproducibility and Ethics}
\label{sec:repro}

\paragraph{Public record} The code, frozen configurations, per-row result
records, analysis files, and audit reports are released at
\url{https://github.com/XingjianZhang-dev/ascent} and archived at Zenodo
(version DOI \texttt{10.5281/zenodo.<version-id>}). A single CPU command
(\texttt{make reproduce}) re-verifies the SHA-256 manifests, recomputes every
analysis file from the per-row records, re-renders every table in this paper
byte for byte, and prints every reported number; \texttt{VERIFICATION.md}
maps each value to its records. Configurations were committed before scores
existed: for all 25 frozen configurations with retained scores, the freeze
commit precedes the first score (\texttt{PREREGISTRATION\_TIMELINE.md}).

\paragraph{What was reproduced on other hardware, and how exactly} Every
formal run executed on NVIDIA RTX PRO 6000 Blackwell instances. Cross-instance
audit reruns on matched hardware reproduced the selected audit cells
identically at the level of prediction and, where floats are stored,
IEEE-754 binary64 identity: the factorial development panel (256 records, node
1 versus node 2), official 16K panel 1 for Qwen2.5-7B (80 rows, node 1
versus node 2), and all ten official 16K panels for Qwen3-8B (800 rows, third
instance). The three instances have distinct GPU serial numbers and, for the
first two, distinct host driver versions captured at the same instant. The
Qwen3-8B rerun is retained as a post-hoc code audit rather than an independent
replication: it executed from a one-commit snapshot of the repository whose
working tree also held uncommitted edits to manuscript-packaging files made
concurrently on that instance; the full diff is retained, none of the
modified files is imported by the runner, and every execution-path file is
byte-identical to the formal-run commit. Its status label names a
state-accounting fix committed nine days earlier, which predates every formal
7--8B run. These audits do not assert that every formal experimental cell was
rerun on another instance.

A revision-stage rerun on a separately provisioned instance with a different
GPU architecture (NVIDIA A100, Ampere) and identical software pins reproduced
the retained results at the level of scientific conclusions but not
bit-identically. Across all sixty official-16K and semantic-holdout cells
(4,800 rows, both arms), 124 of 9,600 scored outputs changed (103 Foundation,
21 \ascent{}); twenty cells were unchanged, and single 80-row panels moved by
up to 5.00 accuracy points (four rows). The ten-panel means moved by at most
1.00 point (official 16K: 10.50/78.88/82.63 $\to$ 9.75/78.88/83.25 points;
semantic holdout: 3.50/77.25/80.75 $\to$ 4.50/77.00/80.62), inside every
reported interval half-width (1.66--3.70 points), and every registered
directional contrast still passes (second official increment 4.38 points,
$p_{\mathrm{Holm}}=0.0028$; semantic holdout 3.62 points,
$p_{\mathrm{Holm}}=0.022$). On factorial panels 2 and 3, one to two of 256
argmax candidates per arm flipped and the panel NLL gains moved by
approximately 0.01 nats against a reported half-width of 0.16 nats. Stored
floating-point values differ in their low-order bits, as expected across
differing kernel implementations. Same-instance controls were bit-identical,
so the differences are attributable to the architecture change.
Bit-identical reproduction is claimed only for the matched-architecture
instance pair where it was observed.

\paragraph{Target-blindness, instrumented} The BABILong writer never receives
the target: it is called with the input and the question only. In this
revision the write, readout, and prompt-construction path runs behind a
blinded row view that raises on any access to the answer field until the
prompt has been rendered, and every result row records the fields read, the
number of blocked accesses (zero), and a three-link provenance check that each
retained fact is the verbatim input span at its recorded position. All 4,800
rows of the official 16K and semantic-holdout studies were rerun with this
instrumentation; on the same instance the instrumented and uninstrumented
runners produced identical outputs.

\paragraph{Validation, interruptions, and determinism} Validation is
fail-closed in every runner. Two pre-score interruptions occurred in the RULER
common-word confirmation---a missing legacy configuration field and a node
without the SmolLM2-1.7B weights---neither of which produced a score; both are
logged in the release. Decoding is greedy in bfloat16 with pinned weights,
revisions, panels, and seeds; \texttt{torch.use\_deterministic\_algorithms}
and a TF32 policy were not set, and this is stated rather than retroactively
introduced (\texttt{reproduction/DETERMINISM\_POLICY.md}). The released
per-row records omit benchmark row bodies and model weights, which remain
under their original licences; panel identifiers, hashes, pinned revisions,
and construction scripts are released instead. The implementation preserves
provenance and supports abstention for domain-specific deployment.""",
    ),
    (
        "sec9-conclusion-scope",
        "cross-node scope",
        r"""systems measurements, representation controls, and exact cross-node
reproduction complete the theory-to-evidence chain. Scale complementarity is
therefore identifiable, achievable, and reproducible in the evaluated frozen
long-context language-model systems.""",
        r"""systems measurements, representation controls, and cross-node reproduction
complete the theory-to-evidence chain. Scale complementarity is therefore
identifiable, achievable, and reproducible in the evaluated frozen
long-context language-model systems, within the scope stated in
Section~\ref{sec:limitations}.""",
    ),
    (
        "data-availability",
        "R2-3",
        r"""\section*{Data and code availability}
The data and code supporting the findings of this study are available from the
corresponding author upon reasonable request. Third-party model weights are not
redistributed and remain subject to the licenses and access conditions of their
original providers.""",
        r"""\section*{Data and code availability}
The code and analysis artifacts supporting this study are publicly available at
\url{https://github.com/XingjianZhang-dev/ascent} and archived at Zenodo,
\url{https://doi.org/10.5281/zenodo.<version-id>}. Panel identifiers, hashes,
and construction scripts are released for BABILong and RULER; the underlying
benchmark corpora are not redistributed and remain under their original
licences (Apache-2.0 and BSD components; see \path{THIRD_PARTY_NOTICES.md}).
Third-party model weights are not redistributed and remain subject to their
original licences.""",
    ),
    (
        "ai-declaration",
        "R2-6",
        r"""During preparation of this manuscript, the author used ChatGPT for language
refinement and limited assistance with code editing. The author reviewed and
edited the resulting material and takes full responsibility for the article.""",
        r"""During the preparation of this work the author used ChatGPT and OpenAI Codex
for editing and refactoring author-written code and for language refinement of
author-written text, and, during the revision, Claude Code (Anthropic) to draft
the revision-stage audit, comparison, instrumentation, and documentation files
from the author's specifications. All AI-assisted code was reviewed by the
author and independently verified by unit tests, hash-verified inputs,
fail-closed validation audits, and independent recomputation of every reported
value from immutable analysis artifacts (\texttt{make reproduce}). A per-file
record of assistance and verification is released as
\texttt{AI\_ASSISTANCE.md} in the public repository. The author reviewed and
edited all resulting material and takes full responsibility for the content of
the article.""",
    ),
    (
        "table1-footnote",
        "R2 integrity bullet 3 (13,824 as a design check)",
        r"""\input{generated/theory_evidence_map.tex}
\end{table}""",
        r"""\input{generated/theory_evidence_map.tex}
\end{table}
\footnotetext[1]{Under the nested construction each additional slot extends
the retained observation prefix of the same latent registers, so the row
checks (prefix nesting and non-redundancy across all 13,824
endpoint--transition rows) hold by construction. They verify implementation
correctness; they are not independent empirical evidence. Per row, the exact
posterior improves on 10,791 of the 13,824 transitions under the noisy
channel (\ref{app:power}).}""",
    ),
    (
        "results-label",
        "cross-reference",
        r"""\section{Results}

\subsection{Qwen2.5 Scaling on Official BABILong}""",
        r"""\section{Results}
\label{sec:results}

\subsection{Qwen2.5 Scaling on Official BABILong}""",
    ),
    (
        "include-appendix",
        "new appendices A--E",
        r"""\bibliographystyle{elsarticle-num}
\bibliography{references,references_verified}""",
        r"""\bibliographystyle{elsarticle-num}
\bibliography{references,references_verified}

\input{appendix_revision1}""",
    ),
]


def main() -> None:
    text = MAIN.read_text()
    log = ["# Changelog — Array revision 1 (ARRAY-D-26-05300)", "",
           "Each entry is one exact-match edit applied to `paper/main.tex` by `experiments/apply_revision1_manuscript_edits.py`. The response letter is generated from this list.", "",
           "| Edit | Reviewer point / reason | Location |", "|---|---|---|"]
    for edit_id, point, old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"edit {edit_id!r}: expected exactly one match, found {text.count(old)}")
        text = text.replace(old, new)
        anchor = old.strip().splitlines()[0][:70].replace("|", "\\|")
        log.append(f"| `{edit_id}` | {point} | `{anchor}…` |")
    MAIN.write_text(text)
    (ROOT / "paper/CHANGELOG_REVISION1.md").write_text("\n".join(log) + "\n")
    print(f"applied {len(EDITS)} edits; changelog written")


if __name__ == "__main__":
    main()
