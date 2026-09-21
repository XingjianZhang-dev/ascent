# Cross-architecture reproduction audit — Phase 2A (Array revision 1)

Date: 2026-09-20. Pre-registration: `PREREGISTRATION.json` (committed as
`9c15400132f2c809625d6848a70ab9d3bad2f36f` before execution). Comparison
script: `experiments/compare_crossarch_reproduction.py`. Machine-readable
result: `CROSSARCH_COMPARISON.json`. Same-GPU control:
`SAME_GPU_RUN1_VS_RUN2_CONTROL.json`. Deviations from the pre-registration:
`DEVIATIONS.md`.

## What was run

Five pre-registered cells, each a full rerun of a retained 2026-08 record, on
one separately provisioned instance with a different GPU architecture from
every retained run:

| | Retained reference | Revision rerun |
|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (cc 12.0) | NVIDIA A100-SXM4-80GB (cc 8.0), UUID `GPU-5b892f6f-ac8e-8c99-71ed-7efad31c6765`, serial `1652523038517` |
| Driver | 580.95.05 / 595.71.05 | 580.159.03 |
| Python / torch / transformers | 3.12.3 / 2.8.0+cu128 / 4.57.6 | 3.12.3 / 2.8.0+cu128 / 4.57.6 |
| tokenizers / safetensors / sentencepiece / numpy / cuDNN | 0.22.2 / 0.8.0 / 0.2.2 / 2.3.2 / 9.10.2 | identical |
| Code | commits `f133a96` (factorial), `4fd25e9` (official 16K) | `9c15400`, clean worktree; runner computation paths unchanged (`run_babilong_prompt.py` gained only `--audit-label`) |
| Weights | pinned revisions, shard SHA-256 verified | same revisions, all eight shard hashes verified before loading |

Cells: Qwen2.5-7B/K=5 on factorial confirmation panels 2 and 3 (256 rows
each, both arms); Qwen2.5-0.5B/1.5B/3B on official-16K confirmation panel 1
(80 rows each, Foundation and ASCENT arms). All ten runner jobs completed;
`environment_before.json` and `environment_after.json` both record
`git_dirty=false` at `9c15400`.

## Result — a sensitivity analysis, not a bit-level reproduction

The rerun reproduced every pre-registered panel at the level of scientific
conclusions but **not** row-for-row and **not** bit-for-bit.

### Level 1 — discrete identity: fails for 1–3 rows per cell

| Cell | Rows | Rows whose *scored* output changed | Rows whose generated string changed but parsed to the same answer/score |
|---|---|---|---|
| factorial panel 2, 7B/K5 | 256 | Foundation argmax: 2; ASCENT argmax: 2 | n/a (candidate scoring) |
| factorial panel 3, 7B/K5 | 256 | Foundation argmax: 1; ASCENT argmax: 1 | n/a |
| official 16K panel 1, 0.5B | 80 | Foundation: 1 (wrong→right); ASCENT: 1 (wrong→wrong, different string) | Foundation: 9 further rows |
| official 16K panel 1, 1.5B | 80 | Foundation: 1 (wrong→right); ASCENT: 0 | Foundation: 2 rows parsed to a different wrong answer, 2 rows different string |
| official 16K panel 1, 3B | 80 | Foundation: 0 scored changes (2 rows parsed to a different wrong answer); ASCENT: 1 (wrong→right) | Foundation: 6 further rows |

The changed BABILong rows are listed with both generated strings in
`CROSSARCH_COMPARISON.json` (`differing_row_ids`) and in the appendix below.
They are low-margin rows: the Foundation arm's 16K-token prompts produce
near-tied continuations (e.g. `'The apple is located in Beulah'` vs
`"The apple is found in Sandra's bathroom"`), and the flipped factorial rows
are candidates whose top two log-likelihoods are within bf16 rounding.

### Level 2 — continuous agreement: last-bit differences with occasional large per-row swings

| Cell | Floats compared | binary64-identical | Max abs diff | Where |
|---|---|---|---|---|
| factorial panel 2 | 8,448 | 8 | 1.123 nats | one row's `gain_nll` |
| factorial panel 3 | 8,448 | 5 | 1.356 nats | one row's `gain_nll` |
| official 16K (per cell) | 15 top-level statistics | 7 | 0.016 / 0.015 / 0.037 | `remaining_error_elimination` |

Per-row NLLs are sensitive: a bf16 forward pass over a 300–350-token prompt
with different tensor-core reduction orders shifts individual row NLLs by up
to ~1.4 nats; these shifts average out at the panel level (next section).

### Level 3 — panel statistics at manuscript precision: one-row shifts on these five cells

| Cell | Statistic | Reference | A100 | Shift | Reported CI half-width for this quantity |
|---|---|---|---|---|---|
| factorial panel 2 | gain NLL (nats) | 7.557 | 7.566 | +0.008 | 0.160 (t₈), 0.255 (Bonferroni) on the 9-panel mean 7.406 |
| factorial panel 3 | gain NLL (nats) | 7.299 | 7.310 | +0.011 | same |
| factorial panel 2 | ASCENT accuracy | .9023 | .8984 | −1/256 | — |
| factorial panel 3 | ASCENT accuracy | .8320 | .8359 | +1/256 | — |
| official 16K, 0.5B | gain (pts) | 6.25 | 5.00 | −1.25 | 3.70 on the 10-panel mean 10.50 |
| official 16K, 1.5B | gain (pts) | 86.25 | 85.00 | −1.25 | 2.71 on the 10-panel mean 78.88 |
| official 16K, 3B | gain (pts) | 80.00 | 81.25 | +1.25 | 1.66 on the 10-panel mean 82.63 |

On these five pre-registered cells every shift is one row (1.25 points on an
80-row panel; 1/256 on a factorial panel) or ≈0.01 nats — smaller than the
reported confidence-interval half-width and one to two orders of magnitude
smaller than the reported effects (78.88 and 82.63 points; 7.406 nats). No
reported conclusion changes: the ordering 0.5B < 1.5B < 3B on panel 1 is
unchanged, the factorial gains remain within 0.15 % of their reference
values, and every panel-level sign is unchanged. These five cells are not
the complete picture; see the sixty-cell extension below before quoting a
maximum shift.

### Extension to all sixty official-16K and semantic-holdout cells (Phase 2B records)

The Phase 2B instrumented rerun (`../target_blindness_2026-09/`, same
instance and environment; instrumented and uninstrumented runners
bit-identical on six control pairs) covers every panel of both studies and
is compared with the retained Blackwell records in
`../target_blindness_2026-09/TARGET_BLINDNESS_AUDIT.json`
(`cross_architecture`):

| Quantity | Value |
|---|---|
| Scored outputs changed | 124 of 9,600 (103 Foundation, 21 ASCENT) |
| Cells by single-panel shift | 20 unchanged; 23 by 1.25 pts; 11 by 2.50; 3 by 3.75; 3 by 5.00 (four rows of eighty) |
| Largest single-panel shifts | semantic holdout panel 4, 0.5B (+5.00); official 16K panel 2, 3B (+5.00); official 16K panel 9, 3B (−5.00) |
| Ten-panel means, official 16K | 10.50/78.88/82.63 → 9.75/78.88/83.25 points |
| Ten-panel means, semantic holdout | 3.50/77.25/80.75 → 4.50/77.00/80.62 points |
| Largest ten-panel-mean shift | 1.00 point, inside every reported interval half-width (1.66–3.70) |
| Registered contrasts on the A100 | all pass; second official increment 4.38 pts (p_Holm = 0.0028); semantic 3.62 pts (p_Holm = 0.022) |

Single panels can move by several rows on a different architecture; the
reported quantities are ten-panel means and their intervals, and those are
what the manuscript's sensitivity statement (Section 8) refers to.

### Level 4 — binary64 identity: fails, as expected across architectures

8,440 of 8,448 stored floats differ in panel 2 and 8,443 of 8,448 in panel
3; the 8 and 5 identical values are exact zeros or ones. This is the expected
consequence of different kernel implementations and reduction orders. It is
not described anywhere as a reproduction failure, and the word "exact" is
not applied to this rerun.

## Same-GPU control: the A100 is run-to-run deterministic

The first execution used Python 3.12.14 by mistake (`DEVIATIONS.md`); the
pre-registered execution used 3.12.3. Comparing the two A100 runs
(`SAME_GPU_RUN1_VS_RUN2_CONTROL.json`): all five cells are identical at every
level — every generated string, every argmax, and all 8,448 stored floats
per factorial cell bit-for-bit. Therefore (i) the A100 executes the frozen
runners deterministically, (ii) the Python patch version has no effect on any
output, and (iii) every difference from the reference records is attributable
to the change of GPU architecture.

## What this changes in the manuscript

- Bit-identical ("exact") reproduction is claimed only for the retained
  matched-architecture instance pair on the cells where it was observed
  (factorial development panel 1, node 1 vs node 2; official-16K panel 1 for
  Qwen2.5-7B, node 1 vs node 2; Qwen3-8B on all ten panels, third instance).
- Cross-architecture reproduction is reported as a sensitivity result with
  the numbers above (§8 of the manuscript; `VERIFICATION.md`).
- Coverage sentence: cross-instance reruns now cover factorial confirmation
  panels 2 and 3 (7B/K5) and official-16K panel 1 for the three co-scaled
  readers, in addition to the previously audited cells; every other formal
  cell was run once.

## Appendix — the changed BABILong rows

```
0.5B  ASCENT      row 09bac03b qa2 target=garden   : ref 'kitchen' (wrong)                        -> A100 'floor' (unparsed, wrong)
0.5B  Foundation  row 0b3d68b5 qa2 target=bathroom : ref 'The apple is located in Beulah' (wrong) -> A100 "The apple is found in Sandra's bathroom" (right)
1.5B  Foundation  row 05238a96 qa2 target=garden   : ref 'kitchen' (wrong)                        -> A100 'The milk was obtained from Mary.' (unparsed, wrong)
1.5B  Foundation  row 07487021 qa2 target=bedroom  : ref 'kitchen' (wrong)                        -> A100 'bathroom' (wrong)
1.5B  Foundation  row 00b0bfaa qa3 target=hallway  : ref 'kitchen' (wrong)                        -> A100 'hallway' (right)
3B    Foundation  row 065d2fab qa2 target=kitchen  : ref 'the apple is found in multiple locations:' (unparsed) -> A100 'the apple is found in the hallway.' (wrong)
3B    ASCENT      row 03795d41 qa3 target=bathroom : ref 'sandwiches' (unparsed, wrong)           -> A100 'bathroom' (right)
3B    Foundation  row 06e023fe qa3 target=garden   : ref 'the kitchen' (wrong)                    -> A100 'bathroom' (wrong)
```

Factorial rows whose argmax flipped: panel 2 Foundation `p02_r0141`,
`p02_r0210`; panel 2 ASCENT `p02_r0179`, `p02_r0193`; panel 3 Foundation
`p03_r0200`; panel 3 ASCENT `p03_r0156`.
