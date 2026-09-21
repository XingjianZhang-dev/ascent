# VERIFICATION

How every number in the manuscript can be traced to a retained record, what
"exact" means where the word is used, what was rerun on other hardware and
what happened, and what this repository does not cover.

Manuscript: *ASCENT: Scale-Complementary External State for Frozen
Long-Context Language Models* (Array, ARRAY-D-26-05300, revision 1).
Public repository: <https://github.com/XingjianZhang-dev/ascent>.
Archive: Zenodo version DOI `10.5281/zenodo.22865847`
(<https://doi.org/10.5281/zenodo.22865847>, release `v1.0-array-revision-1`). The private development repository (276 commits; hash/date/subject
log in `docs/DEV_HISTORY_LOG.txt`) and the frozen 2026-08-23 backup archive
(SHA-256 `431049c38a3c91e7b0fb1f8a3d5a52484aeae609a81a12af0c9c5272a5e00821`)
are available to the editor or a data-integrity reviewer on request.

## 1. One-command recomputation (CPU, minutes, no weights)

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r reproduction/requirements-reproduce.txt      # numpy, scipy
make reproduce
```

`experiments/reproduce_all_tables.py` (1) re-verifies every SHA-256 manifest
in the result trees it reads, (2) recomputes the four authoritative analysis
JSONs from the per-row records and compares them with the retained ones —
exact for every non-float field, floats to 1e-9 (the only observed
differences are ≈1.7e-12 in confidence-interval endpoints, from the SciPy
version's Student-t critical value; the ledger records the same effect),
(3) re-renders all eight generated manuscript tables and compares them byte
for byte with `paper/generated/`, (4) regenerates the power analysis and the
13,824-line transition audit and compares them, and (5) prints every reported
number. It exits non-zero on any mismatch. Expected final line:

```
ALL REPORTED NUMBERS REPRODUCED
```

The transcript of this command from a fresh clone is
`reports/FRESH_CLONE_REPRODUCTION_TRANSCRIPT.txt`.

## 2. Number-to-artifact traceability

| Reported value (manuscript) | Authoritative analysis record | Per-row records | Recompute command |
|---|---|---|---|
| Official 16K accuracy curve 10.50 → 78.88 → 82.63 pts; increments 68.38 (p_Holm = 3.43e-11) and 3.75 (p_Holm = 0.0255); Table 2 intervals | `artifacts/remote_results/babilong_qwen_canonical_16k_confirmation_4fd25e9/analysis.json` | `…/canonical/{0p5b,1p5b,3b}/canonical_16k_confirmation_panel_{1..10}.json` (+ `raw/3b/` control); manifest `…/SHA256SUMS.txt` (41 files) | `analyze_babilong_canonical_coscale_16k_confirmation` |
| Semantic holdout 3.50 → 77.25 → 80.75 pts; increments 73.75 (1.65e-13) and 3.50 (0.02897) | `artifacts/remote_results/babilong_qwen_canonical_semantic_holdout_b7480e6/analysis.json` | `…/{0p5b,1p5b,3b}/canonical_train_8k_semantic_panel_{1..10}.json`; manifest `…/SHA256SUMS` (31 files) | `analyze_babilong_canonical_semantic_holdout` |
| Factorial gains .388 → 2.099 → 7.406 nats; increments 1.711 [1.540, 1.882], 5.307 [5.053, 5.561]; interactions .781 [.539, 1.023], .968 [.671, 1.264]; 13,824 row checks | `artifacts/sum_numeric_candidate/confirmation_analysis.json` | `artifacts/sum_numeric_candidate/confirmation/{node1,node2}/rounds_{2,3,5}_sum_numeric_candidate_panel_{2..10}_<endpoint>.json` (81 files) | `analyze_noisy_composition_candidate --phase confirmation` |
| Around-7B gains: Qwen2.5-7B 66.75, Mistral-7B 80.75, Falcon3-7B 80.88, Granite-3.3-8B 85.63, Qwen3-8B 72.00 [67.7, 76.3]; 3B→7B increment −15.88 [−20.55, −11.20] | `artifacts/around7b_formal/analysis.json` | `artifacts/around7b_formal/<model>/canonical_slots_4/…` (70 files) plus the frozen 3B root `artifacts/frozen_qwen2p5_3b_16k/`; manifests `artifacts/around7b_formal/*.sha256` | `analyze_babilong_around7b_extension` |
| Systems / FLOPs table | `artifacts/remote_results/babilong_4k_systems/analysis.json`, `artifacts/remote_results/flops_e49eae9/analysis.json` | `…/babilong_4k_systems_ef2322d.tar.gz`, `…/flops_e49eae9/` | `render_paper_additional_tables` |
| Cross-task (RULER) table | `artifacts/remote_results/{cwe_confirm_c749f24,qwen_cwe_confirm_cd8352f}/analysis.json`; NIAH archives | archives named in `paper/README.md` | `render_cross_task_table` |
| Power / MDE appendix; statistical-consistency table | `reports/PANEL_POWER_ANALYSIS.json` | `paper/data/evidence/{primary_panel,factorial_panel,interaction_intervals}.csv` | `analyze_panel_power.py` |
| Third-instance Qwen3-8B audit (800 rows identical) | `reports/THIRD_NODE_AUDIT_ROW_DIFF_2026-09-20.json` | `artifacts/array_third_node_audit/` vs `artifacts/around7b_formal/qwen3-8b/canonical_slots_4/` | `compare_third_node_audit.py` |
| Matched-architecture cross-node exactness (development panel) | `reports/FACTORIAL_CROSS_NODE_EXACT_AUDIT_2026-09-20.json`; `artifacts/around7b_crossnode/qwen7b-panel1-exact-audit.json` | `artifacts/sum_numeric_candidate/{development/node1,audit_node2}/…panel_1…`; `artifacts/around7b_crossnode/` | `compare_factorial_cross_node.py` |
| Cross-architecture sensitivity (this revision) | `artifacts_revision/crossnode_2026-09/CROSSARCH_COMPARISON.json` | `artifacts_revision/crossnode_2026-09/{factorial,official16k}/` | `compare_crossarch_reproduction.py` |
| Per-row target-blindness records (this revision) | `artifacts_revision/target_blindness_2026-09/TARGET_BLINDNESS_AUDIT.json` | `artifacts_revision/target_blindness_2026-09/{official16k,semantic_holdout}/` | see §6 |
| Schema-blind writer ablation (this revision, Appendix D) | `artifacts_revision/schema_blind_2026-09/GENERIC_WRITER_ABLATION.json` | `artifacts_revision/schema_blind_2026-09/<condition>/` | `analyze_generic_writer_ablation.py` |

**Rounding convention.** Every reported number is the exact decimal value of
the panel statistic rounded half up at the printed precision
(`Decimal(...).quantize(..., ROUND_HALF_UP)` in `render_paper_additional_tables.py`,
`render_appendix_tables.py`, `analyze_panel_power.py`, `plot_manuscript_evidence.py`
and `reproduce_all_tables.py`). Ten-panel means of 80-row accuracies are multiples
of 1/800, so exact ties occur: the Granite-3.3-8B gain is 137/160 = 0.85625 and
is reported as 85.63; the A100 semantic-holdout 3B mean 645/8 = 80.625 as 80.63;
its second increment 3.625 as 3.63. Python's `f"{x:.2f}"` rounds the binary
double half to even and prints 85.62 for the same value; it is not used for any
reported number, with one exception: the schema-blind ablation table and
paragraph of Appendix D (`analyze_generic_writer_ablation.py`) print with
`f"{x:.2f}"`, and nine of their values are ties (0.125, 1.125, 5.625, 16.125,
18.625, −11.375, −11.625, 88.625, 96.375 points), which appear there as 0.12,
1.12, 5.62, 16.12, 18.62, −11.37, −11.62, 88.62, 96.37. Under the half-up rule
each would end in 3 or 8; the exact values are in
`GENERIC_WRITER_ABLATION.json`.

## 3. The 13,824 row checks are a design property, not an empirical finding

The numeric factorial crosses three readers with nested state sizes
K ∈ {2, 3, 5} on nine panels of 256 rows. "All 13,824 endpoint–transition row
checks hold" means: for each reader (3), each K-transition (2 → 3, 3 → 5), each
panel (9) and each row (256) — 3 × 2 × 9 × 256 = 13,824 — the observations
retained at the larger K extend those retained at the smaller K as a prefix
**and** differ from them. That is what the writer does by construction: it
retains the first K noisy observations of each latent register, so K_high ⊃
K_low and K_high ≠ K_low on every row. The check verifies that the
implementation did what the design says (no dropped or reordered
observations, no cross-K leakage, Foundation probabilities identical across
K); it cannot fail unless the code is wrong, and it is not evidence that
external state helps.

What is *not* guaranteed per row, and is reported as found in
`reports/row_audits/FACTORIAL_TRANSITION_ROW_AUDIT_13824.csv`: under the
symmetric noisy channel (crossover 0.28 over 8 values, target = sum of two
latents among 15 equal-length candidates) a further observation can mislead.
The exact Bayesian posterior NLL of the target improves on 10,791 of the
13,824 transitions (78 %), and the frozen reader's own candidate NLL improves
on 7,435 (54 %). The registered exact-posterior check is at the panel-mean
level (strictly decreasing mean exact-posterior NLL with K on every panel),
which holds; in the retained analysis record this panel-mean check is stored
under the legacy key `provenance.exact_posterior_strictly_improves_every_transition_every_panel`
(kept unchanged for record compatibility), and `make reproduce` prints it as
`panel_mean_exact_posterior_nll_decreases_every_transition_every_panel` next
to the per-row tally. The reported gains are panel averages, and their
intervals are panel-clustered.

## 4. What "exact" means, and where each level was observed

| Level | Definition | Where it holds (all matched architecture: RTX PRO 6000 Blackwell) |
|---|---|---|
| **Scientific-field identity** | every top-level field entering an analysis is equal after JSON parsing | factorial development panel 1, Qwen2.5-7B/K=5, node 1 vs node 2 (256 records); official-16K panel 1, Qwen2.5-7B, node 1 vs node 2 (80 rows); Qwen3-8B, all ten official-16K panels, formal (node 1) vs third instance (800 rows) |
| **Prediction identity** | every per-row discrete output (generated string, parsed answer, score, argmax candidate) is equal | the same three cells |
| **IEEE-754 binary64 identity** | every stored float has the same 8-byte representation | factorial development panel 1 (7,680 stored probabilities and 768 NLLs) |

The three compute instances are physically distinct: three GPU serial numbers
and UUIDs, and — for instances 1 and 2, captured at the same second — two
different NVIDIA driver versions, which are host-kernel properties
(`reports/NODE_IDENTITY_EVIDENCE_2026-09-20.json`; raw `nvidia-smi -q`
captures under `reports/phase0_archive_evidence/`). The third-instance rerun
is a post-hoc code audit, not an independent replication: it executed from a
one-commit snapshot whose working tree also held uncommitted edits to
manuscript-packaging files made concurrently on that instance; the full diff
is retained in the backup, none of the modified files is imported by the
runner, and every execution-path file is byte-identical to the formal-run
commit (`reports/PHASE0_OPEN_QUESTIONS_RESOLVED.md`). Its status string
`posthoc_code_audit_correction_rerun_no_independence_claim` is the runner's
generic `--audit-rerun` label; the "correction" it names is the 2026-08-14
state-byte accounting fix (commit `693c414`), which predates every formal
7–8B run and was audited as decode-exact.

**"Exact" is never applied to the cross-architecture rerun (§5).**

## 5. Cross-architecture reproduction — a sensitivity analysis

A revision-stage rerun on a separately provisioned instance with a different
GPU architecture (NVIDIA A100-SXM4-80GB, Ampere; UUID
`GPU-5b892f6f-ac8e-8c99-71ed-7efad31c6765`) with identical software pins
(Python 3.12.3, torch 2.8.0+cu128, transformers 4.57.6, tokenizers 0.22.2,
safetensors 0.8.0, sentencepiece 0.2.2, numpy 2.3.2, cuDNN 9.10.2) reproduced
the retained results at the level of scientific conclusions but not
bit-identically. The complete figures, over all sixty official-16K and
semantic-holdout cells (4,800 rows, both arms; Phase 2B records, §6) and the
two pre-registered factorial cells (Phase 2A):

- **124 of 9,600 scored outputs changed** (103 Foundation, 21 ASCENT); 20
  cells were unchanged, 23 moved by 1.25 points, 11 by 2.50, 3 by 3.75 and 3
  by 5.00.
- **Single 80-row panels moved by up to 5.00 points** (four rows of eighty).
- **Ten-panel means moved by at most 1.00 point** (official 16K
  10.50/78.88/82.63 → 9.75/78.88/83.25; semantic holdout 3.50/77.25/80.75 →
  4.50/77.00/80.63), inside every reported interval half-width (1.66–3.70
  points).
- **Every registered directional contrast still passes on the A100**
  (second official increment 4.38 points, p_Holm = 0.0028; semantic holdout
  3.63 points, p_Holm = 0.022).
- On the two factorial cells, 1–2 of 256 argmax candidates per arm flipped
  and the panel NLL gains moved by 0.008 and 0.011 nats against a reported
  half-width of 0.160 nats.
- **Same-GPU controls are bit-identical** (two A100 executions of the 2A
  cells; six instrumented-vs-uninstrumented pairs of the 2B cells), so every
  difference from the reference records is attributable to the architecture
  change.

Records: `artifacts_revision/target_blindness_2026-09/TARGET_BLINDNESS_AUDIT.json`
(`cross_architecture`), `artifacts_revision/crossnode_2026-09/CROSSARCH_COMPARISON.json`;
pre-registrations committed before execution; full 2A report
`artifacts_revision/crossnode_2026-09/CROSSARCH_REPRODUCTION_REPORT.md`.
The five pre-registered 2A cells in detail:

| Cell | Rows whose scored output changed | Panel statistic: reference → A100 | Shift | Reported CI half-width |
|---|---|---|---|---|
| factorial panel 2, 7B/K5 | Foundation argmax 2/256; ASCENT argmax 2/256 | gain 7.557 → 7.566 nats | +0.008 | 0.160 nats (Bonferroni 0.255) on the 9-panel mean 7.406 |
| factorial panel 3, 7B/K5 | 1/256; 1/256 | gain 7.299 → 7.310 nats | +0.011 | same |
| official 16K panel 1, 0.5B | Foundation 1/80 (wrong → right); ASCENT 1/80 (wrong → wrong) | gain 6.25 → 5.00 pts | −1.25 | 3.70 pts on the 10-panel mean 10.50 |
| official 16K panel 1, 1.5B | Foundation 1/80 (wrong → right); ASCENT 0 | gain 86.25 → 85.00 pts | −1.25 | 2.71 pts on 78.88 |
| official 16K panel 1, 3B | Foundation 0 scored changes; ASCENT 1/80 (wrong → right) | gain 80.00 → 81.25 pts | +1.25 | 1.66 pts on 82.63 |

On these five cells each panel moved by exactly one row; the sixty-cell
extension above shows that larger single-panel movements (up to four rows)
occur elsewhere while the ten-panel statistics stay within their intervals.
Per-row stored floats
differ in their low-order bits (8,440 of 8,448 values per factorial cell) and
individual row NLLs move by up to 1.4 nats, as expected across differing
kernel implementations; the panel means absorb this. A same-instance control
(two A100 executions under Python 3.12.14 and 3.12.3) was bit-identical at
every level, so the A100 executes the frozen runners deterministically and
every difference from the reference is attributable to the architecture
change. No hardware was substituted to obtain agreement. Bit-identical
reproduction is claimed only for the matched-architecture instance pair where
it was observed.

Two deviations from the pre-registration are recorded in
`artifacts_revision/crossnode_2026-09/DEVIATIONS.md`: the first execution
used Python 3.12.14 (retained in full as
`artifacts_revision/crossnode_2026-09_run1_python3.12.14/`), and the
comparison output states explicitly that `model_identity`, a runner output
field added after the official-16K references were produced, was skipped as
absent from the reference schema.

## 6. Per-row target-blindness rerun (Phase 2B) and schema-blind ablation (Phase 2C)

**Target-blindness, instrumented.** `ascent/target_blindness.py` wraps each
panel row in a view on which reading the answer field (or iterating the row)
raises until `reveal()` is called; the BABILong runner calls `reveal()` only
after the ASCENT prompt has been rendered, and records on every row the keys
read before the reveal, the number of blocked attempts, the reveal phase, and
a three-link provenance check (each retained fact is the verbatim input span
at its recorded position; every parsed field occurs in that sentence; the
rendered state string is the raw sentence or the canonical template of the
fact's own fields). The writer module `ascent/babilong_memory.py` is
unchanged and still passes the static test that it never names the target
field. Pre-registration:
`artifacts_revision/target_blindness_2026-09/PREREGISTRATION.json`.

Result (`TARGET_BLINDNESS_AUDIT.json`): all 60 cells (official 16K and
semantic holdout, Qwen2.5-0.5B/1.5B/3B, ten panels each, both arms) — 4,800
rows — record `blocked_target_access_attempts = 0`,
`target_read_before_reveal = false`, keys read before reveal ⊆ {input,
question, task}, and `state_derived_from_input_only = true`. Six same-instance
control pairs (instrumented vs uninstrumented runner, same A100, same
environment: official-16K panel 1 and semantic-holdout panel 1 for each
reader) are identical at every ladder level including binary64 identity of
every stored float, so the instrumentation changed no output. The
cross-architecture extension of these 60 cells is summarized in §5.

**Schema-blind ablation.** Pre-registration and interpretation rule:
`artifacts_revision/schema_blind_2026-09/PREREGISTRATION.json`; result:
`artifacts_revision/schema_blind_2026-09/GENERIC_WRITER_ABLATION.json` and
Appendix D of the manuscript. Result: all three schema-free writers collapse under the pre-registered
rule. Ten-panel gains (points, 0.5B/1.5B/3B): sentence window
−5.25/−17.50/−16.00; lexical BM25 chain −5.00/−16.38/−11.62; BGE-M3 hybrid +
reranker −5.50/−11.00/−5.38 — negative at every scale, i.e. at a
one-to-three-slot budget the generic state is worse than no state, and no
adjacent-increment family passes Holm with increasing gains. The
structured-fact writer's advantage in ASCENT accuracy over each generic
writer, paired over panels on the same A100 instance, is 15 points at 0.5B
and 89–99 points at 1.5B and 3B. Foundation outputs are identical across
all three conditions and the same-instance canonical run (0 of 2,400 rows
differ per condition). Consequently the manuscript takes Option B: §7 scopes
the claim to external state constructed against the evaluated
structured-fact schema (Appendix D; `paper/generated/generic_writer_ablation*.tex`
are generated from the record and compared by `make reproduce`). The BGE-M3
retrieval caches (three passages per row) are released under
`artifacts_revision/schema_blind_2026-09/bge_cache/` with their hashes
pinned in the run configuration.

## 7. Freeze evidence

`PREREGISTRATION_TIMELINE.md` lists, for each of the 147 configuration
files, the commit that last changed it, its author date, the config SHA-256,
the panel hashes it registers, and the earliest retained score produced under
it. For all 25 configurations with retained scoring records the freeze commit
precedes the first score (tightest margin: `65e1491`, 90 s before the first
around-7B score). Panel files are re-hashed by the runner before every run and
the run aborts on mismatch; the manifests in `data/*/MANIFEST.json` record
panel and upstream-source hashes.

## 8. Disclosed interruptions

Two pre-score interruptions occurred in the RULER CWE confirmation, neither of
which produced a score or a fallback output: the first launch stopped during
config parsing because the legacy singular `evaluation_samples` compatibility
field was absent (added in commit `c749f24` before any confirmation score);
and one node-2 invocation for SmolLM2-1.7B stopped before scoring because the
node lacked the local weights (the 1.7B confirmation completed on node 1).
Their logs are retained
(`artifacts/remote_results/cwe_confirm_c749f24/cwe_confirm_c749f24_node2.tar.gz`);
see `reports/C15_FAIL_CLOSED_VALIDATION_AUDIT_2026-09-20.md`. Validation in
every runner is fail-closed: a missing, malformed, or mismatched state aborts
the invocation; nothing is silently substituted with a Foundation output.

## 9. Coverage limits

- Cross-node audits cover the selected cells named in §4 and §5, not every
  formal experimental cell; every other cell was run once.
- The third instance's environment capture records Python, Torch,
  Transformers, tokenizer, SentencePiece, safetensors, GPU model, peak CUDA
  bytes, commit and dirty flag, but no full CPU/RAM/pip snapshot at run time;
  the backup taken thirteen hours later supplies those
  (`reproduction/environment_captures/final_node3_2026-08-23/`). The
  project-venv pip freeze in that capture is empty because the venv had been
  copied from another machine; the base-environment freeze is the applicable
  one. The `reproduction/environment.json` hashes for node 1/2 captures cannot
  be resolved to a retained file; the genuine per-node captures are in
  `reproduction/environment_captures/node{1,2}_2026-08-15/`.
- `reproduction/DETERMINISM_POLICY.md` states what was and was not pinned
  (no `torch.use_deterministic_algorithms`, no TF32 policy, no attention
  backend pin).
- Three result files under `artifacts/around7b_formal/diagnostics/` are
  post-hoc decode-order checks that reproduce their formal counterparts
  exactly and enter no reported number
  (`reports/AROUND7B_DIAGNOSTICS_EXCLUSION_AUDIT_2026-09-20.json`).
- Held back from this repository because they embed benchmark row bodies
  (available to the editor on request; none is referenced by the manuscript or
  by any script here): `artifacts/remote_results/ascent_core_reproduction_ffe6c78.tar.gz`
  (SHA-256 `b67d521e7c5e6153395d35225b168ba6d5b9babffa227f279c76727da271069c`),
  `…/ruler_qa2_headroom_052cd2a/…node1.tar.gz` (`5dfa237a…`),
  `…/ruler_qa2_hotpot_dualpath_factorial_8566dae/…node1.tar.gz` (`2e5582a0…`),
  `…/ruler_qa2_hotpot_fullstate_factorial_59cf4e6/…node1.tar.gz` (`2e3b23d8…`).
- One loose result file,
  `artifacts/remote_results/babilong_8k_confirm/node2/panel1_smollm2-360m-node2.json`,
  is an unreadable file-provider placeholder in the development tree; the
  released bytes are the same-named member of the transfer archive it was
  extracted from (`reports/RELEASE_TREE_INTEGRITY.txt`).
- Model weights and benchmark corpora are not redistributed
  (`THIRD_PARTY_NOTICES.md`); every checkpoint's shard hashes are pinned in
  `configs/` and verified before loading.
