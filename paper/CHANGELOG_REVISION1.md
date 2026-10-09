# Changelog — Array revision 1 (ARRAY-D-26-05300)

Each entry is one exact-match edit applied to `paper/main.tex` by `experiments/apply_revision1_manuscript_edits.py`. The response letter is generated from this list.

| Edit | Reviewer point / reason | Location |
|---|---|---|
| `front-matter-single-blind` | R2-8 (metadata); Array is single-anonymized | `% Editorial Manager requires a double-anonymized review manuscript. Co…` |
| `abstract-7b-regression` | R2-1 | `0.5B, 1.5B, and 3B parameters, with both adjacent directional contrast…` |
| `abstract-scope-and-reproduction` | R2-5, R2-7; cross-node scope | `checkpoints at the two larger shared scales. Cross-node reproduction,…` |
| `intro-exact-reproduction` | cross-node scope | `ablations, systems accounting, and exact cross-node reproduction. This…` |
| `contribution-exact-reproducibility` | cross-node scope | `cross-task controls, systems measurements, and exact cross-node reprod…` |
| `prop3-indices-remark` | theory clarification (7.11) | `The preceding statement concerns an oracle decoder. To connect it to f…` |
| `sec4.1-fail-closed` | C-15 (implementation-description correction) | `Each endpoint executes the same five operations: (i) write state from …` |
| `sec4.2-per-row-blindness` | R2-2, R2-6 (target-blindness instrumented per row) | `contains no instance answer. Exhaustive parser checks precede model de…` |
| `sec4.3-fail-closed` | C-15 | `full factorial estimates their interaction. Deterministic validity che…` |
| `sec5.1-smollm2-scales` | clarity | `and 7B parameters and SmolLM2 \cite{allal2025smollm2} readers at three…` |
| `sec6.2-cross-node-scope` | R2-5; integrity bullet 3 | `probability identity across $K$, and all 13,824 endpoint--transition r…` |
| `sec6.3-third-node-scope` | R2-5; Phase 0 Q1--Q3 | `practically important 7B--8B regime. A posthoc audit on a third physic…` |
| `sec7-scope-statement-and-holm` | R2-7; R1 (marginal p_Holm); R2-4 | `\section{Limitations}…` |
| `sec7-delete-tenfold` | R2-7 | `supported-operator FLOP reductions; its profiler-enclosed wall-time ra…` |
| `sec8-rewrite` | R2-3, R2-5, R2-6; integrity bullets; Phase 0; Phase 2A/2B | `\section{Reproducibility and Ethics}…` |
| `sec9-conclusion-scope` | cross-node scope | `systems measurements, representation controls, and exact cross-node…` |
| `data-availability` | R2-3 | `\section*{Data and code availability}…` |
| `ai-declaration` | R2-6 | `During preparation of this manuscript, the author used ChatGPT for lan…` |
| `table1-footnote` | R2 integrity bullet 3 (13,824 as a design check) | `\input{generated/theory_evidence_map.tex}…` |
| `results-label` | cross-reference | `\section{Results}…` |
| `include-appendix` | new appendices A--E | `\bibliographystyle{elsarticle-num}…` |

## Post-script edits (2026-09-20, after the scripted edits above)

- **Rounding convention.** Every rendering path (`reproduce_all_tables.py`, `render_appendix_tables.py`, `plot_manuscript_evidence.py`, `analyze_panel_power.py`) now rounds half up on the exact decimal value, as the manuscript tables always did. Printed digits that changed: Fig. 2 labels 77.2 → 77.3 (77.25) and 26.2 → 26.3 (26.25); Appendix C SmolLM2 certified cells .4437/.6937/.9437/.4062/.6562/.9062 → .4438/.6938/.9438/.4063/.6563/.9063; §8 A100 semantic-holdout figures 80.62 → 80.63 (80.625) and 3.62 → 3.63 (3.625); Appendix D (`analyze_generic_writer_ablation.py`) 0.12/1.12/5.62/16.12/18.62/−11.37/−11.62/88.62/96.37 → 0.13/1.13/5.63/16.13/18.63/−11.38/−11.63/88.63/96.38, wording unchanged. No underlying value changed (`VERIFICATION.md` §2, ledger entry of the same date).
- **Prose pass.** Process narration and duplicated statements removed from §4.1–4.3, §4.5, §5.2, §5.4, §6.2, §6.3, §7, §8, §9 and Appendix B; §8 paragraphs retitled. No number, result or scope statement changed; both audits pass; the abstract is 249 words. `experiments/apply_revision1_manuscript_edits.py` reproduces the state before this pass.
- **Zenodo archive updated in place (2026-09-21).** The file on record 10.5281/zenodo.22865847 was replaced within Zenodo's 30-day correction window with the archive of the current `main` (Figure 4 axis range; attribution corrections in `AI_ASSISTANCE.md`; no reported number changed). No new Zenodo version, DOI, GitHub release or tag; `v1.0-array-revision-1` is untouched.

## Second revision (ARRAY-D-26-05300R1 → R2, October 2026)

Reviewer 1 (2 October 2026): "My concerns are resolved." Reviewer 2 (28 September 2026) asked for three clarifications and no new experiments. Each manuscript change below is an exact-match edit to `paper/main.tex` or `paper/references.bib`. No reported value, table entry or figure changed, no analysis was rerun, and `make reproduce` output is unchanged.

| Edit | Reviewer point | Location | Change |
|---|---|---|---|
| `abstract-7b-mechanism` | R2: clarity on the 7B divergence | abstract | The 7B sentence states the mechanism: Foundation accuracy rises to 33.25% while four-slot ASCENT answers every question (1.000 on all ten panels, 800 of 800 rows), which caps the gain at 66.75 points. The −15.88 [−20.55, −11.20] increment and the R1 scope sentence are unchanged. The abstract stays at 249 words (audit count); the trade is "Accuracy gains are not monotone across every scale" and "the Foundation reader improves and the four-slot state saturates, producing" out, the mechanism in. |
| `sec4.2-writer-structure` | R2: structural breakdown of the writer for non-bAbI documents | §4.2; new Table 1 | The writer is described as a recogniser (typed records carrying position and verbatim span), a state update over three primitives (last-write map, append-only log, counter) and a question map; Table 1 gives these, with registered bounds and budgets, for all four evaluated document families, two of which are not bAbI-style (RULER multi-query retrieval, RULER common-word aggregation). Adapting to a new family means supplying its schema. The former sentence pointing to §7 for the schema is replaced. |
| `sec7-schema-table-ref` | cross-reference | §7 "Schema scope" | Points to Table 1; wording otherwise unchanged. |
| `sec8-determinism` | R2: strict GPU determinism and throughput | §8, new "Determinism" paragraph | What `torch.use_deterministic_algorithms` guarantees (same software and hardware); every same-hardware comparison in the record already has that property with default kernels; why the flag does not extend across GPU architectures (cuBLAS and PyTorch documentation), so it would not have removed the 124 cross-architecture changes. The decoding-settings sentence moved here from the former "Interruptions and determinism" paragraph. |
| `sec8-interruptions-and-licences` | follows from the previous edit | §8 | Paragraph retitled "Interruptions and licences"; first sentence reworded for line breaking. |
| `bib-determinism-references` | R2: strict GPU determinism | `paper/references.bib` | PyTorch 2.8 documentation (`torch.use_deterministic_algorithms`; "Reproducibility") and the cuBLAS 12.8 User Guide, §2.1.4 "Results Reproducibility". They are references 90–92; references 1–89 keep their numbers. |

Consequences: Tables 1–2 of the first revision are now Tables 2–3; the compiled manuscript has 39 pages (Tectonic) and 92 references. Repository documents updated to match: `WRITER_SCHEMA_AND_SCOPE.md` (restructured around the three-part writer and the Table 1 schemas; §3 states the fail-closed behaviour of a BABILong read on input without bAbI-style sentences, as in manuscript §4.1 and the C-15 audit), `reproduction/DETERMINISM_POLICY.md` (section on the flag), `VERIFICATION.md` (rows for the new statements; table numbers), `AI_ASSISTANCE.md` and its generator (second-revision word counts), and the docstring of `experiments/analyze_generic_writer_ablation.py` (table number). `audit_paper_claims` and `audit_manuscript_consistency` pass, and the Editorial Manager source (`ASCENT_manuscript_Revision2.tex`, bibliography inlined) compiles in a clean TeX Live 2026 container holding only the upload files. The marked-up PDF is `latexdiff --type=FONTSTRIKE` of the R1 submission source against the R2 source (additions blue, deletions red). No new tag, GitHub release or Zenodo version.
