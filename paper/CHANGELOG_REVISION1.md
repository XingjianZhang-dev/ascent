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
