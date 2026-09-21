# Experiment ledger

## 2026-08-13 — natural-text direct ASCENT development

- Added a bounded 4,096-slot LRU ASCENT test over causal, within-document repeated 4-token contexts in WikiText-103 and PG19. Queries contain 128 prior tokens and never contain the target; stored source fragments come only from earlier occurrences at least 128 tokens away.
- Development used eligible hits 0–5,119 (1,024 calibration; 4,096 test), Pythia-410M and Pythia-2.8B, frozen weights, and a calibration-only conservative simplex over foundation, exact retrieval, and latent replay.
- WikiText-103 paired NLL gains: 410M ASCENT 0.631377 nats, 95% CI [0.569409, 0.693344]; 2.8B ASCENT-Scale 0.400200 [0.354628, 0.445771]. At 2.8B, 16-token Scale exceeded 4-token Fixed by 0.087465 [0.075274, 0.099657].
- PG19 paired NLL gains: 410M ASCENT 0.358975 nats, 95% CI [0.316136, 0.401814]; 2.8B ASCENT-Scale 0.266855 [0.234024, 0.299687]. At 2.8B, Scale exceeded Fixed by 0.041491 [0.033375, 0.049606].
- Exact retrieved-token accuracy was 44.46% on the WikiText test slice and 30.00% on PG19, so the benchmark contains both useful and misleading memory hits. WikiText 410M exact-only gain was 0.454790 nats and latent-only gain was 0.542275, while dual-path ASCENT reached 0.631377.
- Boundary: absolute ASCENT gain was smaller at 2.8B than 410M on both datasets. These development runs support direct natural-text utility and a within-2.8B storage-breadth effect, but not monotone absolute gain with foundation scale.
- Before any confirmatory execution, froze `configs/natural_repeat_confirmatory.json`: skip the first 6,000 eligible hits, use the next 2,048 only for gate calibration and the following 8,192 for testing. This is disjoint from development hits 0–5,119.

## 2026-08-13 — natural-text direct ASCENT confirmation

- Executed all four frozen cells at clean commit `a3cb0c4`; config SHA-256 was `3b72b3d76ccb79f96595202f510c00fa82169b0fb2f3521d8325d692cdd73571` in every artifact, and endpoint event identities matched exactly within each dataset.
- WikiText-103 ASCENT-Scale paired NLL gains were 0.652231 nats at 410M, 95% CI [0.610615, 0.693846], and 0.417317 at 2.8B [0.385535, 0.449100].
- PG19 ASCENT-Scale paired NLL gains were 0.275467 nats at 410M, 95% CI [0.249933, 0.301000], and 0.195365 at 2.8B [0.173361, 0.217369].
- At 2.8B, the registered 16-token Scale state exceeded the 4-token Fixed state by 0.084624 nats on WikiText [0.075393, 0.093855] and 0.027941 on PG19 [0.022805, 0.033077]. Equal-event pooled Scale-minus-Fixed was 0.056283 [0.050983, 0.061582].
- The dual path exceeded the better registered single path by 0.084761 nats (WikiText 410M), 0.042745 (WikiText 2.8B), and 0.016935 (PG19 2.8B); PG19 410M safely selected latent-only and tied it exactly.
- Post-confirmation dependence sensitivity used 10,000 deterministic block-bootstrap replicates at 64, 256, and 1,024 consecutive events. Every ASCENT-gain interval remained strictly positive; at 1,024-event blocks, lower bounds ranged from 0.130847 to 0.533015 nats.
- All preregistered primary gates and all block-bootstrap sensitivity gates passed. Scope remains the causally retrieved repeated-context subset; it is not unconditional whole-corpus perplexity. Absolute gain remained smaller at 2.8B than 410M on both datasets.

## 2026-08-13 — post-confirmation 1.4B natural-text interpolation

- Froze `configs/natural_repeat_1p4b_interpolation.json` only after the two-endpoint confirmation was known. This is explicitly diagnostic, not a preregistered primary result.
- On WikiText-103, Pythia-1.4B ASCENT-Scale gained 0.498376 nats over foundation [0.463769, 0.532982], and its 8-token state exceeded the 4-token Fixed state by 0.070671 [0.061623, 0.079719]. This breadth effect lies between 410M (zero by identical arms) and 2.8B (0.084624).
- On PG19, Pythia-1.4B ASCENT-Scale gained 0.213025 nats [0.191837, 0.234213], and Scale exceeded Fixed by 0.044231 [0.037021, 0.051441]. This is positive but larger than the 2.8B breadth effect (0.027941), so the three-point PG19 breadth curve is not monotone.
- The 1.4B Scale-minus-Fixed conclusion remained positive under 10,000 block-bootstrap replicates through 1,024-event blocks: WikiText [0.060581, 0.081333], PG19 [0.024793, 0.069209].
- Boundary: ASCENT absolute natural-text gain decreases from 410M to 1.4B to 2.8B on both datasets. Natural-text evidence supports positive bounded-memory utility and positive richer-state value at every nontrivial scale, but not monotone absolute gain with foundation size.

## 2026-08-13 — direct ASCENT on RULER NIAH

- Added a causal bounded parser and latent-replay evaluator for all seven official 4k RULER NIAH tasks. The parser writes fact sentences before seeing the final query, retrieves only query-named keys, and is explicitly treated as a matched-raw upper boundary rather than neural evidence. FPVM remains excluded.
- On the development seed 314159, the matched-raw parser recovered 700/700 answers. Pythia-410M latent-only replay was positive on all seven tasks and improved micro answer-token NLL by 2.282875 nats [2.212024, 2.353726]. Pythia-2.8B 64-token latent replay improved it by 1.360829 [1.312871, 1.408788].
- At 2.8B, 64-token Scale exceeded 16-token Fixed on every task in development. The micro paired effect was 0.674852 nats [0.646761, 0.702943], including 1.522549 nats on multiquery.
- Generated a new 700-example official RULER set with independent seed 271828. All task hashes were frozen in `configs/ruler_niah_4k_confirmatory.json` at clean commit `04dd29d` before neural execution; the causal parser again recovered 700/700 answers.
- Independent-seed confirmation passed all 14 task-by-endpoint latent-only gates. A sample-weighted audit first averaged each answer's tokens, then used 20,000 bootstrap resamples over 80 test samples per task. Equal-task latent-only gains were 4.054156 nats at 410M, bootstrap 95% CI [3.962969, 4.139962], and 2.635269 at 2.8B [2.557031, 2.711687].
- All seven 2.8B Scale-minus-Fixed gates passed under both normal and sample-bootstrap intervals. The equal-task effect was 0.478447 nats [0.461618, 0.495254]; task-level mean effects ranged from 0.102170 to 1.606092 nats.
- Boundary: these are teacher-forced answer-token NLL and top-1 diagnostics. They are not official autoregressive RULER exact-match scores. The dual fusion selects the exact certified path and therefore ties the matched-raw parser; neural claims use the separately reported latent-only arm. Absolute latent gain remains smaller at 2.8B than 410M.

## 2026-08-13 — strict RULER foundation-by-memory factorial falsification

- Generated a third independent official RULER NIAH seed (161803), froze all seven hashes, and ran a strict 2×2 design in which both Pythia-410M and Pythia-2.8B used exactly the same 16-token Fixed and 64-token Rich states at `rho=0.25`.
- All 14 within-endpoint Rich-minus-Fixed memory effects were positive, but the decisive cross-model interaction passed only 3/7 task gates. Positive interactions appeared on `niah_multikey_1`, `niah_single_1`, and `niah_single_2`.
- UUID and multiquery tasks showed large negative interactions because the direct-text replay interface helped the smaller model disproportionately. Equal-task interaction was -0.542162 nats with hierarchical bootstrap 95% CI [-0.564631, -0.519788].
- Decision: the current direct-text replay interface fails the core general scale-complementarity objective on RULER and cannot support a cross-model scaling claim. Do not subset away failed tasks. Development moves to a deeper/compositional read interface; any successful redesign requires a new untouched confirmation seed.

## 2026-08-13 — RULER variable-tracking redesign and four-seed interaction

- Replaced direct answer-copy replay with a causal bounded RULER Variable
  Tracking reader. It stores only the final unsolved assignment instance,
  recovers the queried chain, and replays assignment evidence while the frozen
  foundation must compose the output. The parser matched all 100/100 official
  development rows and is retained as a matched-raw boundary, not neural
  evidence.
- At `rho=0.75`, both endpoints benefited from Rich64 over Fixed16, but the
  model-by-memory interaction was negative: -0.037036
  [-0.056543, -0.017528]. At the mechanism-motivated `rho=0.875`, development
  passed: 410M Rich-minus-Fixed +0.028287, 2.8B +0.048280, interaction
  +0.019993 with bootstrap 95% CI [0.004282, 0.035565].
- The first untouched confirmation seed 223607 retained a positive interaction
  mean (+0.011269) but its interval crossed zero
  [-0.002642, 0.024994], so that single confirmation was correctly recorded as
  failed rather than tuned.
- A frozen high-power panel added seeds 244949, 264575, and 282843 with 200
  rows each and included seed 223607 in the equal-seed analysis. All 4/4 seed
  interaction means were positive. The equal-seed Rich64-minus-Fixed16 effects
  were +0.024388 at 410M [0.019224, 0.029438] and +0.044720 at 2.8B
  [0.034982, 0.053733]. The decisive interaction was +0.020332 with
  hierarchical bootstrap 95% CI [0.010957, 0.029458].
- Two provenance commits appear because the earlier confirmation config and
  later panel configs were committed separately. The executed runner content
  is byte-identical across the corresponding local commits (SHA-256
  `af5d5ef5ee1deea2ffa2498b7572a317ebf57c5a83c5d979291bf75d4ec4adaf`);
  both runtime commit hashes remain in the artifact.
- Decision: the replicated 2x2 interaction passes and shows that the larger
  foundation extracts about 1.83x as much benefit from the same ASCENT state
  expansion. This is a strong causal ablation, not yet the primary co-scaled
  ASCENT curve.

## 2026-08-13 — primary co-scaled RULER-VT development

- The user clarified that Fixed memory is an ablation only. The primary claim
  must co-scale the frozen foundation and ASCENT state, then require ASCENT's
  gain over the same-scale foundation to increase with scale.
- A suffix-state curve froze `B_s=8(d_s/1024)` at 8/16/20 replay tokens for
  Pythia 410M/1.4B/2.8B. Gains were +0.030170, 0, and +0.027781 nats; the slope
  was -0.003956 [-0.010766, 0.002854]. It failed.
- A progressive-prefix redesign made larger states append later causal-chain
  assignments and froze `B_s=12(d_s/1024)` at 12/24/30 tokens. Gains were
  +0.021087, 0, and +0.038312 nats. The 410M-to-2.8B endpoint gain increased
  by 1.82x and the three-point slope turned positive (+0.005861), but its 95%
  interval still crossed zero [-0.000210, 0.011931] because the 1.4B safe gate
  disabled the latent path.
- Decision: retain the four-seed 2x2 pass and the positive co-scaled endpoint
  trend, but do not claim a monotone three-scale curve. Test a same-family
  alternative middle checkpoint under the unchanged width-coupled budget; any
  favorable design still requires untouched-seed confirmation.

## 2026-08-13 — provenance-clean Pythia-1B middle-point boundary

- After the source audit, the official Pythia-1B `step143000` artifact with
  SHA-256 `d2523eec...` replaced the unexecuted old-revision download. The
  evaluator rejects a model before loading unless its frozen filename, byte
  count, and SHA-256 match the config.
- Under the width-linear 12/24/30-token rule, the provenance-clean gains were
  0.021087, 0.038239, and 0.038312 nats for 410M/1B/2.8B. The global slope was
  positive at 0.008708 [0.002531, 0.014884], but the 1B-to-2.8B adjacent
  difference was only 0.000074 [-0.012822, 0.012969]. A strict adjacent gate
  therefore failed, and the already frozen 12/24/30 confirmation configs were
  not executed.
- Decision: the earlier 1.4B zero is checkpoint/interface-specific rather than
  a general middle-scale collapse. Nevertheless, a nearly flat upper
  transition is not strong enough for the intended claim.

## 2026-08-13 — parameter-sublinear co-scaled RULER-VT confirmation

- Replaced width-linear replay bandwidth with one endpoint-independent rule,
  `B_s = round_half_up(12 (N_s/405334016)^0.75)`, yielding 12/24/51 tokens for
  410M/1B/2.8B. The registered replay-token/parameter ratios decrease from
  2.96e-8 to 2.40e-8 to 1.82e-8. The three confirmation configs and their
  untouched data hashes were committed before the redesigned development run.
- Development seed 314159 passed every gate. Gains were 0.021087, 0.038239,
  and 0.050980 nats. Adjacent differences were +0.017152
  [0.005501, 0.028804] and +0.012741 [0.000607, 0.024875]. The slope was
  +0.015396 [0.008433, 0.022358]; all corresponding 20,000-replicate bootstrap
  lower bounds were positive.
- Three untouched 200-row seeds (316228/331663/346410) then ran at one clean
  remote commit `988900f...`. Equal-seed endpoint gains were 0.027202 at 410M
  [0.020478, 0.033926], 0.038763 at 1B [0.035942, 0.041584], and 0.054761 at
  2.8B [0.051561, 0.057962].
- The seed-clustered 410M-to-1B increment was +0.011561 with t(2) 95% CI
  [0.003398, 0.019725]; the 1B-to-2.8B increment was +0.015998
  [0.012016, 0.019981]. The primary slope was +0.014290
  [0.009250, 0.019329]. Every seed's slope and both adjacent point estimates
  were positive, and every preregistered panel gate passed.
- Secondary boundary: seed 316228 alone had a 410M-to-1B normal lower bound of
  -0.000091, so that seed's stricter stand-alone adjacent-significance gate
  failed narrowly. The equal-seed panel gate was the registered replication
  estimand and passed; the individual failure remains reported.
- Accounting boundary: the curve currently registers neural replay bandwidth,
  not a complete byte-normalized persistent-state implementation. Actual 2.8B
  replay lengths were capped by source length (47--51 tokens in confirmation),
  while the causal parser's raw payload was recorded separately. Do not use
  this result to claim final persistent-byte efficiency until token/latent
  state storage and transient compute are separately charged.
- Decision: the core teacher-forced, single-task co-scaled gain curve is now
  independently confirmed. It is not yet TNNLS promotion evidence by itself;
  official autoregressive scoring, additional task families, cross-architecture
  replication, complete state/compute accounting, robustness, and competitive
  baselines remain required.

## 2026-08-13 — cross-task co-scaled composition confirmation

- Tested whether the increasing-gain law transfers from RULER-VT to the
  independently implemented two-fact addition composition task. The first
  endpoint-independent 1/2/4-round rule failed development because the gains
  were 0.170484, 0.137710, and 0.519272 nats: the 410M-to-1B difference was
  -0.032774 with 95% CI [-0.044977, -0.020571]. Its already frozen confirmation
  seeds were not executed.
- A second uniform, decreasing-relative-cost rule was frozen as
  `R_s = round_half_up(2 (N_s/405334016)^0.4)`, yielding 2/3/4 signal rounds.
  Its rounds/parameter ratio strictly decreases with scale. Three confirmation
  seeds (20261111/20261121/20261131) were committed before the development
  result was observed.
- Development passed every endpoint, adjacent, and slope gate. The gains were
  0.021885, 0.387988, and 0.519272 nats; adjacent differences were +0.366103
  [0.356388, 0.375819] and +0.131284 [0.119967, 0.142601].
- All three untouched confirmation seeds passed individually. Equal-seed gains
  were 0.021990 at 410M [0.014884, 0.029097], 0.383647 at 1B
  [0.375435, 0.391859], and 0.521680 at 2.8B [0.507808, 0.535552]. The
  410M-to-1B cluster increment was +0.361656 with t(2) 95% CI
  [0.360028, 0.363285], and the 1B-to-2.8B increment was +0.138034
  [0.128398, 0.147670]. The scale slope was +0.255658
  [0.249880, 0.261437]. Every seed's two adjacent point estimates and slope
  were positive.
- Boundary: this is a controlled synthetic composition task and teacher-forced
  answer-token NLL. Clean-KV text and a zero-NLL clean oracle remain separately
  reported, so this result makes no raw-storage-superiority claim. Official
  generative benchmarks and stronger composition tasks remain required.
- Result archive SHA-256:
  `ae2e7e766105972278d23fb4fa55fd7055ed11975ba28aed9b776f32e5155880`.

## 2026-08-13 — official autoregressive RULER-VT metric bridge

- Implemented NVIDIA RULER's official case-insensitive `string_match_all`
  metric and its registered 30-token Variable Tracking generation budget.
  Test decoding is greedy and never reads target tokens. A calibration-only
  gate selects the foundation/latent probability mixture from the first 20
  rows; the next 40 rows are scored.
- The first attempted execution emitted an ambiguous attention-mask warning and
  was interrupted before producing an artifact. The runner was fixed to pass an
  explicit all-one mask, the official 30-token limit was committed, and all
  three endpoints were then executed from the clean remote commit.
- The development bridge failed. Official fractional match changed from 0.025
  to 0.010 at 410M (paired difference -0.015, 95% CI [-0.0444, 0.0144]),
  tied 0.010 at 1B, and tied 0.065 at 2.8B. All latent gates selected weight
  one with positive calibration NLL lower bounds, but the improved target-token
  likelihoods did not change most greedy argmax sequences.
- Decision: do not translate the confirmed teacher-forced NLL scaling curve
  into an official generative-accuracy claim. Redesign sequence-level readout
  or use a generation-capable, provenance-locked model family; then freeze new
  untouched data seeds. Merely repeating the present decoder is not justified.
- Result archive SHA-256:
  `5e9e26cf1d79f29aa0703dbaf0825180b7be1043272178235ecc243181af1df7`.

## 2026-08-13 — Qwen2.5 generation-family screen and fairness correction

- Added a second Transformer implementation for the official Qwen2.5-Instruct
  0.5B/1.5B/3B family. Every repository revision and weight shard was frozen
  and SHA-256 checked before loading. The uniform state rule allocated
  12/28/48 replay tokens; replay tokens per parameter strictly decreased from
  2.43e-8 to 1.81e-8 to 1.56e-8.
- The first Qwen screen and scale run are invalid for arm comparison. Audit
  found that the library `generate()` path applied Qwen's registered repetition
  penalty to foundation, while ASCENT used a custom repetition-free greedy
  loop. Their apparent 0.5B +1.5-point and 1.5B -1-point changes are discarded.
- A corrected config and runner were committed before any corrected execution.
  Both arms use the same custom 30-token, repetition-free greedy loop;
  foundation is exactly the zero-latent-weight arm. Under this fair decoder,
  official fractional `string_match_all` improved from 0.020 to 0.060 at 0.5B:
  paired +0.040 with normal 95% CI [0.00234, 0.07766]. At 1.5B it tied
  0.050-to-0.050, and at 3B it tied 0.070-to-0.070.
- The corrected three-scale generative curve therefore failed its gate and no
  confirmation seeds will run. In contrast, calibration answer-token NLL gains
  increased sharply with scale: +0.140611, +0.490270, and +0.787655 nats, with
  respective 95% lower bounds +0.092815, +0.366607, and +0.650444.
- Mechanism decision: information gain grows strongly, but a convex
  foundation/latent probability mixture does not flip the larger models'
  greedy sequences. The next development redesign will calibration-select a
  bounded logit-delta scale on a new data seed, with identical decoding and an
  untouched safety split, rather than repeat the failed mixture.
- Boundary: the experiment uses official RULER data generation, 30-token budget,
  and `string_match_all` metric, but evaluates the registered 192-token query
  plus external-memory protocol rather than the standard full-context RULER
  input. It is a metric/protocol bridge, not a standard leaderboard score.
- Result archive SHA-256:
  `0a4a0f87ba7292fba491feb505faf13a4a8f7cdc6f5171131a857048dfef4ee6`.

## 2026-08-13 — sequence-calibrated logit-delta falsification

- Generated a new 200-row, Qwen-tokenized 4K RULER-VT seed (361803) from
  NVIDIA RULER commit `c3f5e3b...`. Row lengths were 4068--4086 tokens and the
  causal memory parser exactly recovered all 200 answers. The data SHA-256,
  model shards, 12/28/48 state rule, and delta grid `[0,.5,1,2,4,8]` were
  committed before execution.
- Rows 0--39 selected the delta; disjoint rows 40--79 were a safety gate that
  required a paired official-score LCB95 above zero; rows 80--199 were held out.
  Both arms again used the identical repetition-free greedy decoder.
- At 0.5B, selection scores for delta 0/.5/1/2/4/8 were
  .05/.05/.05/.04/0/0, so the registered smallest-tie rule chose zero. At 1.5B
  they were .05/.04/.03/.02/0/.04, again choosing zero. Both test arms tied
  exactly at .051667 and .053333, respectively.
- At 3B, selection preferred delta 8 (.07 versus .05 at zero), but the disjoint
  safety effect was -0.030 with 95% CI [-0.08354, 0.02354]. The gate correctly
  disabled the delta, and test tied exactly at .055.
- Answer-token calibration NLL gains still rose +0.140277 -> +0.426517 ->
  +0.726016 nats, with all lower bounds positive. The experiment therefore
  falsifies simple logit-direction amplification: increasing token information
  does not solve multi-variable sequence formatting and exposure-bias errors.
- Decision: do not tune more delta values. Move the official generative bridge
  to single-answer RULER NIAH before considering a structure-aware readout.
- Result archive SHA-256:
  `aa4ea60fcce06e794d832d6654626c67faa301cffe49dc3bce854abb8d346c88`.

## 2026-08-13 — single-answer RULER NIAH generation boundary

- Generated a new 200-row official `niah_single_1` set (seed 362803) with the
  NVIDIA RULER script at commit `c3f5e3b...`, Qwen tokenizer, 4K limit, and
  official 128-token generation budget. The exact causal reader recovered all
  200 answers; row lengths were 3491--4094 tokens. Data and model hashes were
  frozen before any endpoint execution.
- To isolate retrieval from VT's multi-variable formatting, development used
  one seven-digit answer per row, 20 calibration rows, and 60 test rows. Both
  arms used the identical repetition-free greedy decoder and the same
  12/28/48 parameter-sublinear state rule.
- Official fractional match tied exactly at .066667 for foundation and ASCENT
  at 0.5B, 1.5B, and 3B. The generative curve therefore failed and no
  confirmation seed will run.
- Calibration answer-token NLL gain remained positive and increased with scale:
  +0.015809 (LCB95 +0.002082), +0.076926 (+0.051605), and +0.102606
  (+0.043574). Removing multi-variable formatting did not make those
  likelihood improvements flip new greedy answers.
- Decision: stop tuning the same probability/logit readout. The next
  generative redesign must be structure-aware and must retain the exact reader
  as a separately reported raw-memory boundary. Standard full-context RULER
  and BABILong remain mandatory after a readout passes development.
- Result archive SHA-256:
  `51ddbd81a2397e567f3c22da452fe09a8a7912f5d952e31b43c0dd314e371647`.

## 2026-08-13 — active baseline scope update

- The user explicitly directed that FPVM not be used or retained as a baseline.
- All subsequent experiments test ASCENT directly against the frozen
  foundation, matched raw/clean-KV storage, same-signal exact decoding,
  fixed/scale variants, and ASCENT path/mechanism ablations.
- Previously frozen configs and artifacts that list FPVM blockers remain
  byte-identical historical records; they do not define the active matrix.

## 2026-08-14 — matched-search boundaries and learned ASCENT evidence breakthrough

- A newly generated official `niah_single_1` development set (seed 363803,
  100 test rows) first tested identical width-4 beam search in Foundation and
  ASCENT. Answer-token NLL gains increased +0.023691 -> +0.095583 ->
  +0.134924 nats across Qwen2.5 0.5B/1.5B/3B, with all calibration lower
  bounds positive, but exact match tied 1% -> 1% at all scales. Ordinary beam
  collapsed mostly to short strings such as `3.` or `42.`.
- A second untouched development set (seed 364803) imposed the same public
  seven-ASCII-digit output grammar and width-64 beam on both arms. The grammar
  never received a memory value. It restored valid-length predictions and
  retained a strictly increasing NLL curve (+0.024389 -> +0.085846 ->
  +0.137138 nats), but exact match again tied 5% -> 5% at every scale. This
  falsifies the hypothesis that formatting plus a wider probability search is
  sufficient.
- The first dual-path attempt treated the frozen suffix plus LM head as a
  latent autoencoder. Seeds 365803 at 0.5B and 1.5B produced no key-matched
  evidence prefixes and tied 4% -> 4%. A one-sample diagnostic showed why:
  hidden states predict the next token rather than reconstructing their own
  token, so this readout was structurally mismatched and the 3B full run was
  not promoted.
- The substantive redesign implements the proposal's explicit evidence path:
  an 11-class linear head maps each charged latent token to digit 0--9 or
  non-digit. The head is fit only on 2,048 independent seed-366000 episodes;
  its hyperparameters and the 20/23/25-token parameter-sublinear state rule
  were frozen before seed-366803 evaluation. Its held-out digit-token
  accuracies were 99.925%, 100%, and 100% across the three endpoint-specific
  heads.
- One invalid 0.5B run exposed a dispatch bug that computed the correct
  evidence prefix and then overwrote its prediction with an unconstrained
  numeric beam. The invalid score was discarded; commit `466ab77` moves that
  call back to the ordinary digit-beam branch. Data, head training, and every
  registered hyperparameter remained unchanged before the clean rerun.
- Clean development exact-match results were 4% -> 71% at 0.5B, 4% -> 99% at
  1.5B, and 2% -> 100% at 3B. Paired gains were +0.67
  [0.5774, 0.7626], +0.95 [0.9071, 0.9929], and +0.98
  [0.9524, 1.0076], with adjacent point differences +0.28 and +0.03. All
  endpoint cells had zero Foundation-only wins against ASCENT.
- Before accessing confirmation scores, three new official 420-row seeds
  367101/367202/367303 were hash-frozen, with 20 calibration and 400 test rows
  each. The registered panel requires every seed's endpoint gains and adjacent
  differences to be positive, plus positive t(2) 95% lower bounds for both
  adjacent differences and the log-parameter slope.
- The first complete untouched seed, 367101, passed its individual gates:
  Foundation/ASCENT scores were .0225/.4675, .0225/.9900, and .0150/1.0000.
  Gains were +.4450 [.3962, .4938], +.9675 [.9501, .9849], and +.9850
  [.9731, .9969]; adjacent gain differences were +.5225 and +.0175, and the
  gain slope was +.310999 per natural-log parameter. Wins/losses were
  178/0, 387/0, and 394/0.
- This is the first complete untouched official autoregressive scale curve to
  satisfy ASCENT's core point-estimate objective. It is not yet a confirmed
  panel: seeds 367202 and 367303 are still required. The bridge uses a
  192-token current query plus charged external state and is not a standard
  full-context leaderboard result.
- Development result archive SHA-256:
  `e467d4984f7b8f4dcf158695274c69699a743a1453c196d0fdc653217d301502`.
  First-confirmation-seed archive SHA-256:
  `132dbbee9e1936b00d63c0d283d1a0ac2804f2893708c39eb43ac896ae41b499`.

## 2026-08-14 — learned-evidence 20/23/25-token confirmation boundary

- The complete preregistered panel ran three untouched seeds
  367101/367202/367303 with 20 calibration and 400 test rows per seed/endpoint.
  All nine model cells used clean commit `59f74e0c...`; every task hash matched
  the frozen seed map, every endpoint refit its linear evidence head only on
  immutable seed 366000, and all nine cells had zero Foundation-over-ASCENT
  regressions.
- Seed-specific official exact-match gains at 0.5B/1.5B/3B were
  `.4450/.9675/.9850`, `.4625/.9675/.9775`, and `.4375/.9525/.9600`.
  Every endpoint gain, every adjacent gain difference, and every per-seed
  gain slope was positive. The seed-clustered endpoint gain means and t(2) 95%
  intervals were `.44833 [.41646,.48020]`, `.96250 [.94099,.98401]`, and
  `.97417 [.94230,1.00604]`.
- The 0.5B-to-1.5B gain increment passed strongly at `.51417`
  `[.49236,.53598]`. The 1.5B-to-3B increments were `.0175/.0100/.0075`,
  all individually positive, but their cluster mean `.01167` had interval
  `[-.00126,.02459]`. The preregistered requirement that both adjacent cluster
  lower bounds exceed zero therefore **failed**. The overall gain slope did
  pass at `.30331` per natural-log parameter with interval
  `[.28582,.32080]`.
- This is a ceiling-limited statistical failure, not a reversal: 1.5B ASCENT
  already scored 99.0--99.5% and 3B scored 100% on every seed. The gate will
  not be weakened or retrospectively relabeled. The next development rule
  uses fewer state tokens at every fixed model scale so that all endpoints
  remain away from saturation while preserving the identical task
  distribution and a decreasing state/model ratio. It requires a fresh
  development seed and, if successful, new untouched confirmation seeds.
- Complete panel result archive SHA-256:
  `33873911efc205074ab9e7dfc8ee42534accb9ec34c2e2b1213869c127d01a00`.

## 2026-08-14 — ceiling-aware 18/21/23-token ASCENT confirmation

- After the preceding strict failure, a single global parameter-sublinear state
  law was selected on fresh development seed 368803 and frozen before any new
  confirmation score was accessed:
  `B_s = round_half_up(18 * (N_s / 494032768)^0.135)`. This gives 18/21/23
  latent tokens at Qwen2.5 0.5B/1.5B/3B. The probe training set, linear head,
  optimizer, seven-digit grammar, width-64 matched beam, task distribution,
  calibration size, and query budget were unchanged.
- The preregistered confirmation panel used three new official RULER
  `niah_single_1` seeds 369101/369202/369303, each with 20 calibration and 400
  test examples. Seed-specific exact-match gains at 0.5B/1.5B/3B were
  `.0075/.7475/.9725`, `.0125/.8500/.9675`, and
  `.0075/.7325/.9750`. All nine gains and all six within-seed adjacent gain
  differences were strictly positive.
- Equal-seed endpoint gain means and t(2) 95% intervals were `.00917`
  `[.00200,.01634]`, `.77667 [.61781,.93553]`, and `.97167`
  `[.96218,.98115]`. The adjacent gain increments were `.76750`
  `[.61576,.91924]` and `.19500 [.02686,.36314]`. The gain slope on natural-log
  parameter count was `.54008 [.53491,.54524]`. The complete frozen scale gate
  therefore **passed**, including both adjacent cluster lower bounds.
- The post-run provenance audit also passed. All nine cells used the same clean
  commit `22b39c5b...`; every config, task, model revision, and weight-file hash
  matched its frozen manifest; the independently trained evidence-head summary
  was identical across evaluation seeds at each endpoint; and the 3,600 paired
  test rows contained 2,109 ASCENT-only wins and zero Foundation-only
  regressions. Probe validation digit accuracy was 99.894% at 0.5B and 100% at
  both larger endpoints.
- Persistent bf16 latent payloads were exactly 31.5/63/92 KiB, while linear
  evidence heads contained 9,867/16,907/22,539 parameters. State elements per
  model parameter decreased monotonically
  `3.2646e-5 -> 2.0895e-5 -> 1.5264e-5`, and each head was below one basis point
  of its foundation model. These figures exclude transient activations, beam
  KV work, latency, optimizer state, and FLOPs, which remain mandatory systems
  measurements.
- Boundary: this is strong confirmation of the core increasing-gain claim on
  one official NIAH task under a 192-token current-query plus external-state
  protocol. It is not a standard full-context RULER leaderboard result and is
  not sufficient for TNNLS readiness without multi-task/full-context RULER,
  BABILong, a second architecture, competitive controls, robustness, and
  end-to-end systems evidence.
- Audited result archive SHA-256:
  `e418543d3793c0ff4f95e7767d6e47f907d00d539fb1e761d016b2bb670feaa3`.

## 2026-08-14 — full-context 16K multiquery development breakthrough

- Reconstructed the official RULER Paul Graham essay corpus with RULER's own
  URL list and download script: all 218 sources succeeded; the 3,108,618-byte
  corpus SHA-256 is `58e35253...756c65`. Official 4K and 16K single-answer
  screens saturated at 40/40 for Qwen2.5-0.5B, while 16K four-distractor-key
  greedy remained 39/40. These null screens were stopped at 0.5B rather than
  expanded into uninformative scale curves.
- Official 16K `niah_multiquery` created usable headroom: on development seed
  370805, the 0.5B Foundation recovered an average `.6625` of four answers.
  A frozen multi-evidence ASCENT decoder retains the complete greedy Foundation
  output and appends only complete seven-digit runs decoded in query order from
  its latent state by the independently trained linear head. It reads no test
  targets and cannot remove a correct Foundation answer.
- Under the task-scaled parameter-sublinear 82/87/90-token state rule, the
  0.5B/1.5B/3B Foundation-to-ASCENT scores were
  `.6625 -> .8875`, `.6875 -> .9375`, and `.6750 -> 1.0000`.
  Paired gains rose `.2250 [.1701,.2799] -> .2500 [.1974,.3026] ->
  .3250 [.2890,.3610]`; adjacent point increments were `+.0250` and `+.0750`,
  and the gain slope on natural-log parameter count was `+.05135`. The three
  endpoints had 29/32/40 improvements and zero regressions.
- Complete four-value latent evidence coverage increased from 1/40 to 28/40
  to 40/40. Meanwhile state elements per model parameter decreased from about
  `1.49e-4` to `8.66e-5` to `5.97e-5`. Clean runtime commit was
  `a0819de1...`; endpoint wall times were 207/508/961 seconds, exposing the
  current no-KV-cache implementation as a systems limitation to optimize.
- This result is development-only. Three fresh official seeds
  371101/371202/371303 were generated and hash-frozen before execution. The
  confirmation gate requires positive endpoint and adjacent gains for every
  seed plus positive t(2) cluster lower bounds for both adjacent increments and
  the scale slope; it will not be weakened after scores are accessed.
- Development result archive SHA-256:
  `115c41f2e0589897afb9abd62634d041abb20aca6e6539feb494ec6a54ec4fea`.

## 2026-08-14 — first 16K multiquery confirmation falsification

- The frozen seed-371101 confirmation began on clean remote commit
  `e6ae80d...`. At 0.5B, Foundation/ASCENT scored `.6625/.91875`, for gain
  `+.25625 [.19943,.31307]`, 30 improvements, and zero regressions. At 1.5B,
  they scored `.7375/.95625`, for gain `+.21875 [.16557,.27193]`, 29
  improvements, and zero regressions.
- The first adjacent gain difference was therefore `-.03750`. This violates
  the preregistered requirement that every seed have positive adjacent gain
  differences, so the panel **failed immediately**. The 3B cell and remaining
  seeds were not run because no later result could rescue that gate.
- Diagnosis: the 82-token low-scale state recovered three complete values on
  every test row, so its evidence coverage was too close to the 87-token
  middle-scale state, while the 1.5B Foundation itself improved from `.6625`
  to `.7375`. A future development schedule must create a materially steeper
  low-to-middle evidence-capacity gradient. Seed 371101 is now development
  information and can never be reused as untouched confirmation evidence; the
  gate will not be weakened.

## 2026-08-14 — full-context state-accounting repair

- Two further schedules were screened only on the now-development seed 371101.
  The 60/78-token low/middle schedule still produced a negative adjacent gain
  difference (`.13750 -> .10625`) and was rejected. The 38/78/92 schedule
  produced a promising performance curve (`.03750 -> .10625 -> .30000`) with
  zero regressions, but an exact persistent-state audit found that it violated
  the proposal's decreasing-relative-cost invariant.
- Persistent latent state is `B_s * d_s` elements rather than merely `B_s`
  tokens. Consequently, the 38/78/92 schedule's state-elements/model-parameter
  ratios were `6.89185e-5 -> 7.76102e-5 -> 6.10563e-5`: the middle footprint
  increased by 12.61%. The configuration is relabeled as an accounting
  falsification and is ineligible for confirmation regardless of its metric.
- The repaired global law is
  `B_s = round_half_up(38 + 54*x_s^1.17)`, where
  `x_s = log(N_s/N_low)/log(N_high/N_low)`. It yields 38/69/92 tokens and exact
  relative footprints `6.89185e-5 -> 6.86552e-5 -> 6.10563e-5`, strictly
  decreasing at both transitions. This law was hash-frozen before execution
  and will first be screened on seed 371101. Even if successful, it requires
  three entirely new official-data seeds; 371101/371202/371303 will not be
  promoted to the repaired confirmation panel.
- The repaired 38/69/92 schedule then completed cleanly on seed 371101. Its
  Foundation-to-ASCENT gains were `.03750 [.00948,.06552]`,
  `.10625 [.06368,.14882]`, and `.30000 [.26861,.33139]`; adjacent increments
  were `+.06875` and `+.19375`, with natural-log-parameter slope `+.13506`.
  Evidence coverage was exactly one/three/four complete values on every test
  row, producing 6/16/40 wins and zero regressions. All cells used clean remote
  commit `5e597377...`. This is positive development evidence, not confirmation.
- Re-reading the proposal exposed a second registration issue: 38/69/92 was a
  log-parameter interpolation, whereas Section 4.2 explicitly specifies the
  width-coupled law `B_s=B_0(d_s/d_0)^alpha`, `0<alpha<=1`. The final candidate
  therefore uses `B_0=40, alpha=1`, yielding 40/69/91 tokens. Its exact bf16
  payloads are 70/207/364 KiB and its state-elements/model-parameter ratios
  decrease `7.25458e-5 -> 6.86552e-5 -> 6.03926e-5`. The analysis gate now
  verifies both the proposal law and the relative-state inequality.
- A standard autoregressive KV-cache optimization was separately frozen and
  tested before use. On two 16K rows, one cached prediction differed from the
  full-recompute reference and changed its RULER score from `.50` to `.75`;
  the exact-equivalence audit therefore failed. Cached decoding is prohibited
  from the development/confirmation results unless a future implementation
  passes byte-identical prediction checks. The unchanged full-recompute path
  remains the authoritative decoder.
- The proposal-exact 40/69/91 rule then completed on development seed 371101
  without changing the performance curve: gains were again
  `.03750 [.00948,.06552] -> .10625 [.06368,.14882] ->
  .30000 [.26861,.33139]`, with adjacent increments `+.06875/+.19375` and
  slope `+.13506`. Evidence coverage remained one/three/four values on every
  test row, for 6/16/40 wins and zero regressions. All three cells used clean
  remote commit `9a96f60c...`, explicit `foundation_decode_cache=false`, and
  the same frozen config; the archived result SHA-256 is
  `be1f0c7eb4773c68a2fdc44e53e1e88b65ee79e4f9e86a74b0ffb274719f9bdc`.
- Before inspecting repaired-rule scores on seeds 371202/371303, a development
  pressure panel was frozen at config SHA-256
  `74f5fe90746edb203f81d9e72bb78f7c153c54d7f6a0dbf4c6529b7ee0552a64`.
  These seeds were members of the superseded confirmation panel and will never
  be used as confirmation for 40/69/91. They are used only to detect instability
  before generating an entirely new three-seed confirmation panel.
- An automated full-context data audit passed all 180 rows before pressure-test
  scoring: the causal parser recovered every ordered four-answer target; each
  file had 60 unique inputs and 60 unique complete rows; chat-template lengths
  were 15,685--16,285 tokens and selected source spans were 82--92 tokens.
  Official seed 371202 reused random metadata index `197` three times, but the
  three inputs, outputs, and complete-row hashes were distinct. The audit now
  treats indices as metadata and gates on content identity; its artifact SHA is
  `146b10f0deb4695a95c444dc956c15e64b3eb4cc9fe386ab8636dba2e8460234`.
- The frozen three-seed development pressure panel passed every registered
  curve gate. Seed-specific gains at 0.5B/1.5B/3B were
  `.03750/.10625/.30000`, `.05000/.11250/.28750`, and
  `.01875/.10000/.28125`. Every one of the six adjacent differences was
  positive; all nine endpoint gains were positive; there were 185 paired wins
  and zero regressions.
- Equal-seed mean gains were `.03542 -> .10625 -> .28958`. The low-to-middle
  increment was `.07083` with t(2) 95% CI `[.04712,.09455]`; the middle-to-high
  increment was `.18333 [.15962,.20705]`; and the natural-log-parameter slope
  was `.13115 [.11191,.15039]`. Although these statistics pass the planned
  confirmation-style gates, the panel remains development evidence because
  371101 informed the rule and all seeds were associated with the superseded
  first confirmation. No further state-law tuning is permitted; the next step
  is an entirely new hash-frozen confirmation panel. Analysis SHA is
  `be50dcdcc48698d51617d2bce7906ebf60592055176361204a4f459f47ebad24`;
  complete archive SHA is
  `0b646f60715b551d1914a096150ed32bdbf6e9d6d6f10203354b7c89a84279b4`.
- With all model scores still inaccessible, three entirely new official RULER
  seeds 372101/372202/372303 were generated using the official repository at
  `c3f5e3b4...`, Qwen's frozen tokenizer, 16,384-token target length, and 60
  rows per seed. Their validation SHA-256 values are respectively
  `9e240e29...`, `23a9a985...`, and `26e3e87a...`. A pre-score audit passed
  180/180 ordered parser checks and found all inputs/full rows unique; audit SHA
  is `1b9a83d835fd6907fce4b7b753225d533a053b4546d14af96d03d99c95f30c05`.
- Before executing any confirmation cell, config SHA-256
  `4aa9024167d644ea4d996fe7562882aa23bb7b5df32648863f374bde5840c5be`
  froze the proposal-exact `B_s=round_half_up(40*(d_s/896)^1.0)` law, full-
  recompute greedy decoding (`foundation_decode_cache=false`), the independent
  old probe training set, all model/data hashes, and the unchanged seed-
  clustered gates. Manifest SHA-256 is
  `8517fdcabd645b68ea17212b8f50a8f87373ce817c30229dbb2ba128a1942c08`.
  No further decoder, state-law, or gate change is permitted after the first
  confirmation score is accessed.

## 2026-08-14 — proposal-exact 16K full-context confirmation passes

- All nine frozen cells completed on the same clean remote commit
  `283b79c9450547d7a3783990863db2b35e6163d2`. The integrated audit verified
  every config/task/model hash, 40-row test count, endpoint identity, model
  artifact, `foundation_decode_cache=false` flag, zero Foundation-only
  regression, identical endpoint-specific probe fit across data seeds, the
  exact proposal width law, and strictly decreasing relative state ratios.
- Seed-specific Foundation-to-ASCENT gains at 0.5B/1.5B/3B were
  `.06875/.15000/.31875`, `.02500/.10000/.31250`, and
  `.03750/.11250/.32500`. Every endpoint gain and every within-seed adjacent
  difference was positive. Across 360 paired test rows per endpoint there were
  194 ASCENT-only wins and zero Foundation-only regressions; 3B ASCENT scored
  `1.0000` on all three seeds and recovered all four values on all 120 tests.
- Equal-seed mean gains rose `.04375 -> .12083 -> .31875`. The low-to-middle
  increment was `.07708` with t(2) 95% CI `[.06812,.08605]`; the middle-to-high
  increment was `.19792 [.13517,.26066]`; and the gain slope on natural-log
  parameter count was `.14194 [.11627,.16760]`. All preregistered statistical
  and provenance gates therefore **passed**.
- Persistent bf16 latent state is exactly 70/207/364 KiB under the registered
  40/69/91-token width-linear law. State elements per model parameter decrease
  `7.25458e-5 -> 6.86552e-5 -> 6.03926e-5`, so absolute state grows while its
  relative footprint shrinks at both adjacent transitions.
- The metric boundary is official NVIDIA RULER `string_match_all` with the
  complete registered 16K current context visible to both arms, but ASCENT's
  independently trained latent-evidence decoder is a reported method component
  rather than the stock leaderboard decoder. This confirmation covers one
  `niah_multiquery` task and does not satisfy broader RULER/BABILong,
  cross-architecture, control, or systems gates by itself.
- Analysis SHA-256 is
  `4de0751f6fd5918c286e9775981a77492e82822561ea270ef53445b0e59448af`;
  complete archive SHA-256 is
  `d3780e16ff45dab3a8b05cb9066072ab62466a4f8cccf1f867e5e885fec4a1c7`.

## 2026-08-14 — second RULER task and 16K saturation boundary

- Added a separate bounded append-only NIAH reader for official
  `niah_multivalue`, where four values for the same key are all task targets.
  It preserves repeated events in character order and does not alter the
  existing latest-only overwrite reader. Official 16K seed 373803 passed 60/60
  causal exact-value checks; validation SHA is `b428b197...`.
- Under the unchanged proposal-exact 40/69/91 law, 0.5B Foundation/ASCENT
  scored `.78125/.85000`, gain `+.06875 [.03372,.10378]`, with 11 wins and no
  regressions. At 1.5B Foundation/ASCENT scored `.91875/.96250`, gain
  `+.04375 [.00916,.07834]`, with 6 wins and no regressions.
- The low-to-middle gain difference was therefore `-.02500`; the frozen
  development curve **failed** and the 3B cell was not run. The cause is a
  Foundation ceiling jump rather than ASCENT degradation. State budgets and
  decoder are held fixed; a separately generated official 32K development
  seed 374803 (SHA `4f46b946...`) tested whether increased context difficulty
  restores headroom. The 16K failure remains a reported external-validity
  boundary regardless of the 32K outcome.
- The initial 32K pre-score audit correctly extracted all four target facts but
  failed because the auditor imposed ordered-list equality. Official RULER
  shuffles multivalue needle positions independently of the reference list and
  its `string_match_all` metric is order-independent. The auditor was repaired
  to require exact multiset equality, including duplicate counts; a regression
  test was added. The repaired audit passed 60/60 rows before any model score,
  with chat lengths 32,047--32,667 tokens and audit SHA-256
  `df1fee55ce98e8b27ccad37bab2c241840a907e4d7e2895b068a23cde14639c1`.
- At 32K, 0.5B Foundation/ASCENT scored `.59375/.74375`, gain
  `+.15000 [.10775,.19225]`, with 23 wins and zero regressions. The 1.5B
  scores were `.87500/.98125`, gain `+.10625 [.06368,.14882]`, with 16 wins
  and zero regressions. The low-to-middle gain difference was `-.04375`, so
  the frozen 32K curve also **failed** and the 3B cell was not run. Doubling
  context restored low-end headroom but did not prevent Foundation saturation
  from rising faster than the ASCENT gain at the middle scale.
- Both 32K cells used clean remote commit `a12f43e...`, explicit full
  recomputation (`foundation_decode_cache=false`), and the unchanged 40/69/91
  law. Wall times were 589.66 s and 1357.97 s; peak allocated GPU memory was
  42.13 GB and 45.85 GB. Complete archive SHA-256 is
  `9fed9d33863bdeb9db4663d86d3b4aafba709bf8f0071be6b7b5422d15383f74`.

## 2026-08-14 — SmolLM2 cross-architecture development freeze

- Selected the official public `HuggingFaceTB` SmolLM2-Instruct Llama family,
  not a community mirror. Immutable repository revisions are `12fd25f...`,
  `a10cc151...`, and `31b70e2...` for 135M/360M/1.7B. The 135M and 360M
  safetensors were downloaded and independently hashed before any score;
  exact parameter counts are 134,515,008 and 361,821,120. The 1.7B file's
  registered linked SHA-256 is `f55217be...`, size 3,422,777,952 bytes, and
  exact parameter count derived from the frozen architecture is 1,711,376,384.
- Generated 60 untouched official 4K `niah_multiquery` rows from NVIDIA RULER
  commit `c3f5e3b...` with seed 375803. Data SHA-256 is `48f0c96a...`; the
  causal parser recovered all four query-ordered values on 60/60 rows, chat
  lengths are 3,340--3,998 tokens, and selected evidence spans are 90--97
  SmolLM2 tokens.
- Before accessing any SmolLM2 prediction, froze the width-sublinear state law
  `B_s=round_half_up(52*(d_s/896)^0.75)`, giving 37/55/97 tokens. This exposes
  exactly 1/2/4 complete seven-digit values on every registered row. Persistent
  bf16 state is 41.625/103.125/388 KiB while state elements per model parameter
  decrease `1.58436e-4 -> 1.45928e-4 -> 1.16080e-4`.
- Development gate: every endpoint gain and both adjacent gain differences
  must be positive. Stop before 1.7B if low-to-middle is nonpositive; any pass
  requires entirely new confirmation seeds. No model score had been accessed
  at this freeze point.

## 2026-08-14 — SmolLM2 cross-architecture confirmation

- The frozen development seed passed before confirmation. Foundation/ASCENT
  scores at 135M, 360M, and 1.7B were `.49375/.60625`, `.53750/.75000`, and
  `.48750/1.00000`, so ASCENT gains rose `.11250 -> .21250 -> .51250`.
  The adjacent gain differences were `+.10000` and `+.30000`; there were
  18/28/40 improved test rows and zero regressions at all three endpoints.
- Generated three entirely new official RULER 4K `niah_multiquery` datasets
  with seeds 376101/376202/376303 before confirmation scoring. Their immutable
  SHA-256 values are `a3895ffd...`, `872c76e4...`, and `a01757c3...`.
  The pre-score audit passed exact ordered parsing on 60/60 rows per seed,
  unique inputs and rows, and complete-context token ranges of 3,337--3,998.
  Audit SHA-256 is `a17ea226...`.
- The frozen 37/55/97-token rule passed on every confirmation seed. Endpoint
  gains were `.08750/.19375/.50625`, `.11875/.26250/.53750`, and
  `.10625/.19375/.52500`. Equal-seed means therefore rose
  `.104167 -> .216667 -> .522917`.
- Treating the three independent data seeds as the statistical units, the
  135M-to-360M gain increment was `+.112500`, t(2) 95% CI
  `[.041352,.183648]`; the 360M-to-1.7B increment was `+.306250`,
  `[.235102,.377398]`. The gain slope against log parameter count was
  `+.167528`, `[.163456,.171600]`. All registered scale gates passed.
- Across the nine confirmation cells there were 255 improved test rows and
  zero Foundation-only regressions. Every cell used the same frozen probe,
  exact registered model/data/config identities, full recomputation without a
  Foundation KV cache, and clean runtime commit `6900387...`. Relative state
  elements per model parameter strictly decreased at both scale transitions.
- The integrated audit passed every registration, state-law, clean-commit,
  probe-identity, and no-regression gate. The complete confirmation archive is
  `artifacts/remote_results/smollm2_4k_multiquery_width075_confirm_376101_376202_376303.tar.gz`,
  SHA-256 `6373ba3b660fb0419a548515fb8f09df015ec180ad849fcb07845963b8c6f0e1`.
  This independently reproduces the primary scale-complementarity phenomenon
  beyond Qwen2.5, but it does not erase the negative `niah_multivalue`
  boundary or complete the remaining task, control, and systems gates.

## 2026-08-13 — Pythia source and weight-lineage audit

- Audited Hugging Face repository ownership, branch refs, commit history, LFS
  metadata, and the SHA-256 of every weight file actually loaded on the GPU
  node. The principal 410M/1.4B/2.8B experiments all use the official
  EleutherAI standard Pythia suite and resolve to the respective official
  `step143000` branch targets. They are not community mirrors or v0 models.
- The apparently different 2.8B `main` and `step143000` safetensors sizes are a
  serialization difference. `main` contains 96 per-layer attention mask
  buffers in addition to 388 parameter tensors; the branch artifact contains
  the same 388 parameter tensors without those buffers. Common tensor names,
  dtypes, shapes, and offsets match, and sampled bytes across embeddings,
  layers 0/15/31, the output head, and final norm were identical.
- The 410M branch safetensors stores parameters in float32 whereas `main`
  stores them in float16 plus attention buffers. The evaluator explicitly casts
  every endpoint to bfloat16. Samples from embeddings, layers 0/12/23, the
  output head, and final norm were exactly equal after that cast (maximum
  absolute difference zero), so this serialization difference also does not
  alter the executed model.
- The temporary Pythia-1B middle-point diagnostic did contain a provenance
  error: it referenced commit `020ae513...` rather than the official
  `step143000` target `c0fea4b9...`. The incomplete old download and its watcher
  were stopped before execution, so it contributed no result. The config now
  points at the correct official branch target and requires weight SHA-256
  `d2523eec...`.
- Decision: existing 410M/1.4B/2.8B findings are not invalidated by model
  source. The 1.4B zero-gain boundary remains a real observed model/interface
  result, not presently attributable to a wrong public checkpoint. Preserve
  the source manifest in `configs/model_provenance_pythia_step143000.json` and
  reject any future endpoint whose runtime hash does not match it.

## 2026-08-13 — environment and protocol audit

- Remote host: AutoDL Ubuntu 22.04, 22 vCPU, 110 GB advertised container RAM
  (host reports ~1 TiB available), one NVIDIA RTX PRO 6000 Blackwell Server
  Edition with 97,887 MiB VRAM, driver 580.95.05.
- Storage: 30 GB system disk and 550 GB empty data disk at `/root/autodl-tmp`.
- GPU was idle at audit time.
- Existing local FPVM package contains Mamba2 endpoint configs, prior FPVM
  experiment/audit code, RULER, BABILong, and natural-text assets. It is treated
  as read-only source material; ASCENT uses this separate Git repository.
- Decision: implement and pass Stage A before copying large assets or installing
  the foundation-model environment.

## 2026-08-13 — Stage A certified noisy-refinement channel

- Command: `python experiments/run_stage_a.py --config configs/stage_a.json`.
- Local and remote environments both passed 6/6 original tests; the expanded
  suite subsequently passed 8/8 tests.
- Config SHA-256:
  `197a3ac2a24dc1294355345f5a013e74b12ca76e5f0a4e9ae3ee3a16a73cc9de`.
- Exact cumulative gains for 1/2/3/4 rounds: 0.887014775, 1.428409813,
  1.808085574, and 2.070118673 nats.
- Exact adjacent conditional-information increments: 0.887014775,
  0.541395038, 0.379675762, and 0.262033098 nats; every value is strictly
  positive. The decreasing increment is expected posterior saturation and does
  not contradict strictly increasing cumulative gain.
- On 200,000 Monte Carlo episodes, every adjacent increment had a strictly
  positive paired 95% interval. The exact and empirical NLLs agreed within the
  preregistered 0.01-nat tolerance.
- Remote result: `status=passed`, Python 3.12.3, NumPy 2.5.2, Linux 5.15,
  hostname `autodl-container-d91lagjqwt-00a4d7fe`.
- Decision: Stage A passes. Proceed to a diagnostic endpoint screen, but do not
  promote Stage B until matched-raw, FPVM-v1, latent-replay, and factorial
  interaction arms are implemented.

## 2026-08-13 — endpoint asset decision

- The GPU node cannot reach Hugging Face directly; the local workstation can.
- Public checkpoints are downloaded locally at immutable revisions, hashed,
  then transferred to `/root/autodl-tmp`; no endpoint result is observed before
  the endpoint configuration is frozen.
- Pythia is selected as the first Transformer family because the proposal names
  its 410M/1.4B/2.8B ladder as the cleanest controlled scaling family. Mamba2 is
  retained for the recurrent-family replication.

## 2026-08-13 — exact-KV falsification diagnostic

- An optimized clean exact-KV cache with the same registered key fingerprint
  was compared analytically against the BSC certified channel under identical
  byte budgets.
- At the small endpoint, the clean cache gain was 1.299651 nats versus
  ASCENT's 0.415788 nats; at the large endpoint, 2.637208 versus 1.532373 nats.
- Decision: the proposal's simple-recall matched-raw superiority criterion is
  internally inconsistent with a clean `(K,V)` write. The certified theorem is
  a monotone-information result, not superiority over a clean exact-KV oracle.
  Strong raw storage remains mandatory; future practical claims must come from
  overwrite, composition, semantic/latent state, retention, or natural text.

## 2026-08-13 — corrected Pythia certified endpoint diagnostics

- An audit invalidated the first endpoint screen: sampling a tensor whose last
  dimension depended on endpoint refinement count changed the nominally fixed
  one-round signal. Those results are discarded.
- Commit `e3541b2ded176e3e9f7821e44e0616c427ad75ab` samples four registered
  rounds once, then exposes strict prefixes. Ten tests pass locally and on the
  GPU node. All corrected endpoints have identical label SHA-256
  `b1fe07c6249869fca1fbe9cbd80f5acb6880ebb37d1cae7785a75fced16cb742`
  and fixed-prefix SHA-256
  `0f37cf5e35696d1203e28b9f9f2848b6c3e7039405355294547125e8f0e8a0d9`.
- Corrected certified gain (nats; 16,384 held-out episodes): 410M one round
  0.904736 [0.887000, 0.922472]; 1.4B one/two rounds 0.904736 and
  1.429113 [1.411160, 1.447065]; 2.8B one/four rounds 0.904736 and
  2.070310 [2.054968, 2.085652]. Same-signal exact controls tie exactly.
- Raw foundation candidate NLLs are 2.898026, 2.935050, and 2.942061 nats for
  410M/1.4B/2.8B, respectively, versus the registered uniform 2.772589. They
  are tokenizer/model-bias diagnostics and are excluded from the theorem.
- Peak GPU memory and wall time: 410M 1.495 GB/6.52 s, 1.4B 3.842 GB/13.27 s,
  2.8B 7.018 GB/18.74 s on the RTX PRO 6000.
- Decision: the curve validates nested certified information only. The fixed
  arm is identical across model scales, proving that this implementation has no
  foundation-state interaction. Stage B remains ineligible until matched raw,
  FPVM-v1, latent replay, and the preregistered two-factor interaction pass.

## 2026-08-13 — latent-replay interface falsification and repair

- The initially registered one-token posterior-mean replay at relative depth
  `rho=0.5` failed scale complementarity. At 410M it gained 0.081920/0.295959
  nats for one/four signal rounds, while the 2.8B safe gate disabled both arms;
  the paired interaction was -0.214040 [-0.220254, -0.207826].
- A symmetric post-hoc depth audit (`rho` 0.25/0.50/0.75) could not rescue the
  one-token interface. A same-signal MAP text-replay control did work at 2.8B
  (0.684498/1.375373 nats), localizing the failure to the latent interface.
- Replaying the full static codebook hidden sequence from the MAP state decoded
  the signal at both endpoints. At development seed 20260815, `rho=0.25` was
  the only audited depth with a positive interaction: 0.067538
  [0.058276, 0.076799]. The 0.50 and 0.75 interactions were negative. This
  depth selection is explicitly post-hoc and is not confirmation evidence.

## 2026-08-13 — preregistered latent-replay factorial confirmation

- Before accessing seed 20260821, config SHA-256
  `16821587ae105ce3887037e29fec78afd5d17781e1bdbb3cc73b5778c3e185fe`
  froze full-sequence MAP replay, `rho=0.25`, 4,096 calibration episodes,
  16,384 tests, and one primary paired difference-in-differences estimand.
- Both endpoints ran from clean commit
  `21750806e69dc4cde08208530d09721daf9cd4ff`; config, label, full-signal,
  query-input, source-commit, and test-label equality checks all passed.
- Memory gains (nats) were 0.638920 [0.618658, 0.659183] and 1.370406
  [1.345973, 1.394840] at 410M for fixed/rich signals, versus 0.667897
  [0.646921, 0.688874] and 1.466745 [1.440822, 1.492668] at 2.8B.
- The preregistered foundation-by-signal interaction was 0.067361 nats with
  95% CI [0.060663, 0.074060]. The fixed-signal foundation effect was 0.028977
  [0.022876, 0.035078] and the rich-signal effect 0.096338
  [0.087749, 0.104928]. The primary and all-gain secondary gates passed.
- Peak GPU memory/wall time were 1.798 GB/22.30 s at 410M and
  7.665 GB/73.07 s at 2.8B.
- Decision: retain the latent-replay direction and proceed to strong-baseline,
  overwrite/composition, and replicated scale tests. Stage B is still not
  promotion eligible; this confirmation does not erase the exact-KV simple-
  recall falsification or satisfy the remaining proposal gates.

## 2026-08-13 — preregistered three-scale, three-seed latent curve

- Config SHA-256
  `c8fbf7f664bf8a42d9f5238899e9f3715cd22b235e135b4556d45f4c409b454e`
  froze Pythia 410M/1.4B/2.8B, seeds 20260831/32/33, full-sequence replay at
  `rho=0.25`, 2,048 calibration and 8,192 test episodes per cell, and the
  seed-clustered slope as the primary estimand before any of the nine cells ran.
- All runs used clean commit `b11cd36927ba74a271051b2cf27e53524ca1d5ce`.
  Within every seed, config, labels, complete noisy signals, query inputs, and
  test labels matched exactly across all endpoints. No depth or replay-mode
  override was present.
- Rich-signal gain increased at every scale for every seed. The seed-specific
  slopes of gain on natural-log parameter count were 0.048766, 0.048315, and
  0.049145 nats. The three-seed cluster mean was 0.048742 with the preregistered
  t(2) 95% interval [0.047709, 0.049775].
- Adjacent foundation-by-signal interactions also passed: 410M-to-1.4B mean
  0.017275 [0.005460, 0.029090], and 1.4B-to-2.8B mean 0.049024
  [0.043978, 0.054070]. All three seed estimates were positive in both
  transitions.
- Representative rich gains for seeds 20260831/32/33 were respectively
  1.332229/1.354952/1.355204 at 410M,
  1.386091/1.405290/1.409659 at 1.4B, and
  1.427847/1.450315/1.451530 at 2.8B.
- Decision: the preregistered synthetic latent-replay scale gate passes. It is
  still not a paper-level promotion: clean exact-KV capacity-pressure tasks,
  FPVM-v1 external validity, overwrite/composition, natural text, long-context,
  cross-family replication, and system/safety evidence remain outstanding.

## 2026-08-13 — two-fact addition composition and strong-raw boundary

- A two-fact task independently samples two fresh values in `{0,...,7}` and
  asks the frozen foundation to predict their sum in `{0,...,14}` after a
  context reset. ASCENT replays two role-specific latent sequences at
  `rho=0.25`. Controls include the no-memory foundation, exact convolution of
  the two noisy posteriors, same-signal MAP text, clean-KV text reconstruction,
  and the unbeatable zero-NLL clean-KV oracle.
- Development seed 20260901 showed a positive endpoint interaction of 0.124891
  [0.115869, 0.133913], while also showing that clean-KV text was much stronger.
  The interface and task were then frozen before accessing seed 20260911.
- Confirmatory config SHA-256
  `f778fc600dace7747f4cb1e16cdcde46e23c9392b3d868f379335578776d1524`
  ran from clean commit `11c7f9f6fe575e2073a10e6d53549ecb909a1e3f` with 4,096 calibration
  and 16,384 test episodes per endpoint. Config, answer, full-signal, query,
  source-commit, and clean-tree invariants all passed.
- Confirmatory ASCENT gains at 410M were 0.166959 [0.159919, 0.173999]
  and 0.157257 [0.150413, 0.164101] for fixed/rich signals. At 2.8B they
  were 0.403428 [0.395898, 0.410957] and 0.518902
  [0.513121, 0.524684]. The preregistered interaction was 0.125176 nats
  [0.119022, 0.131330].
- Strong-raw superiority failed by a wide margin. ASCENT-rich minus clean-KV
  text gain was -0.177316 [-0.183599, -0.171032] at 410M and
  -0.387297 [-0.393726, -0.380868] at 2.8B. The clean-KV oracle NLL is exactly
  zero and cannot be beaten by a noisy representation.
- Decision: the composition scale-interaction gate passes, but the result is a
  capability/scale claim, not clean-storage superiority. Stage B remains
  ineligible pending overwrite, FPVM/external validity, and natural/long-context
  evidence; the raw boundary must remain explicit in any paper.

## 2026-08-13 — versioned overwrite mechanism and falsification

- The initial overwrite diagnostic exposed both obsolete and current replay
  sequences to the main path. This contradicted the locked versioned-memory
  semantics, whose read returns only the latest version. The implementation was
  corrected to latest-only masking; unmasked history and reversed replay order
  were retained as explicit failure ablations rather than discarded.
- Development seed 20260921 showed that latest-only masking substantially
  improved every cell, but the scale interaction remained negative at
  -0.081462 [-0.092132, -0.070791]. A new falsification replication was frozen
  before accessing seed 20260931.
- Falsification config SHA-256
  `4c6f78de39188e5aeb1cf29dc754c16157677e140eea47ef3a0e00f58dfbfb80`
  ran from clean commit `30a7b4766e02bd6a948c23126911c1cf622e63e6` with 4,096 calibration
  and 16,384 test episodes per endpoint. All config, current-label, signal,
  query, source-commit, and latest-version invariants passed.
- Within-endpoint gains were all significant: 410M fixed/rich gains were
  0.699815 [0.678232, 0.721398] and 1.477227 [1.449825, 1.504630]; 2.8B
  gains were 0.716293 [0.694286, 0.738301] and 1.417821
  [1.392404, 1.443237].
- Latest-only masking beat unmasked history in all four cells by 0.101686,
  0.252294, 0.125635, and 0.261271 nats, with all 95% lower bounds positive.
  This supports the version-control mechanism.
- The preregistered foundation-by-signal interaction was nevertheless
  -0.075884 [-0.083119, -0.068650]. Its upper bound is strictly negative, so
  overwrite scale complementarity is falsified for the locked Pythia interface.
- Decision: retain version masking as a validated mechanism but record
  overwrite as an external-validity failure. Under the proposal's decision
  rule, the full project cannot be promoted or written as a successful TNNLS
  paper unless a substantively revised, independently confirmed architecture
  resolves this boundary without post-hoc relabeling.

## 2026-08-14 — official BABILong scale-expanding confirmation

- The official BABILong repository was locked at commit
  `7a6efee29f5cac03c3c410e6799c80fd2ffe3610`; the official
  `RMT-team/babilong-1k-samples` 4K QA1/QA2/QA3 files were converted with
  recorded hashes. A target-blind bounded causal state reader reconstructed
  all 2,998 available answers exactly and selected exactly 1/2/3 supporting
  facts for every QA1/QA2/QA3 row, respectively.
- Before any BABILong model score, clean commit
  `d53b21f977a04ea800acfa378ceb685f7f5f8700` froze a task-balanced 60-row
  development panel, three disjoint untouched 120-row panels, exact SmolLM2
  model artifacts, greedy one-location scoring, and the scale-expanding
  1/2/3-fact state schedule. Development gain increased
  `0.1500 -> 0.5000 -> 0.6500` for 135M/360M/1.7B.
- Clean confirmation commit
  `c075cec97384b98e57ffe08d61b7c5bea48ae778` and config SHA-256
  `d30b6fcb98cf7d4a9fcc18ce1b98600ae8eae9fe2e03102619f93ae1166becce`
  preserved the development schedule before any confirmation prediction.
  The 360M endpoint was independently executed on a second RTX PRO 6000 node;
  its development result reproduced exactly.
- The three untouched panel gain curves were
  `0.3000/0.4833/0.6917`, `0.2667/0.5250/0.7000`, and
  `0.2167/0.5083/0.6333`. Every panel was strictly monotone. Panel-clustered
  endpoint mean gains were `0.2611`, `0.5056`, and `0.6750`, with t(2) 95%
  lower bounds `0.1569`, `0.4535`, and `0.5848`.
- Adjacent gain increments were `+0.2444 [0.1066, 0.3823]` and
  `+0.1694 [0.0653, 0.2736]`. The gain-on-log-parameter slope was
  `0.1580 [0.1414, 0.1745]`. Single-commit, single-config, disjoint-data,
  clean-worktree, parser, and exact model-artifact gates all passed.
- The raw artifact archive is
  `artifacts/remote_results/babilong_4k_scale_confirm_c075cec.tar.gz`, SHA-256
  `1e60a89f9f4e7ea93a387a28918bd3b69ba53297918e75a6da4cc90e011962a3`.
- Decision: this is the first confirmed structurally different official
  compositional-memory result whose scale-gain increments have positive
  panel-clustered lower bounds. It materially strengthens the primary claim,
  but does not by itself satisfy the remaining fixed-state, irrelevant-memory,
  matched-compute, systems, broader-context, and competitive-baseline gates.

## 2026-08-14 — untouched BABILong causal robustness controls

- Before any robustness-panel score, clean commit
  `d93ff34831f58e835ed484c1fc4a2d90068a7b76` and config SHA-256
  `e6498755d5d2668718078e8070bc4db9f538d287c219945e69207dc34934092c`
  froze three further disjoint official panels (stable-hash positions 140:260)
  and four conditions: scale-relevant 1/2/3 slots, fixed-relevant 1/1/1,
  scale-matched irrelevant memory, and scale-matched cyclically corrupted
  locations. No robustness model prediction was accessed before the freeze.
- Scale-relevant gains on the three new panels were
  `0.2000/0.4833/0.6250`, `0.2500/0.5667/0.7000`, and
  `0.2167/0.3667/0.6833`; every panel remained strictly monotone. Fixed-state
  mean gains were only `0.2222/0.2139/0.1444` and did not increase with scale.
- The scale-expansion minus fixed-state adjacent interactions were
  `+0.2583 [0.0196, 0.4971]` and `+0.2667 [0.0226, 0.5107]`, using t(2)
  panel intervals. Both preregistered lower bounds were positive.
- Relevant ASCENT accuracy exceeded same-slot irrelevant memory by
  `+0.2778 [0.2347, 0.3209]`, `+0.5750 [0.4257, 0.7243]`, and
  `+0.8028 [0.7007, 0.9049]` across 135M/360M/1.7B. It exceeded corrupted
  memory by `+0.3417 [0.2670, 0.4163]`, `+0.6722 [0.4766, 0.8679]`, and
  `+0.9028 [0.8597, 0.9459]`.
- Foundation outputs were identical sample-by-sample across all four
  conditions. Single-commit, single-config, row-alignment, clean-worktree,
  parser, data, and model-artifact gates passed. The 360M cells ran on the
  independent second RTX PRO 6000 node.
- The raw archive is
  `artifacts/remote_results/babilong_4k_robustness_d93ff34.tar.gz`, SHA-256
  `610934f57b11dcc6b71dd6d3d894e9449fcf8eeda1ba1a8e4c62582890d8d474`.
- Decision: all registered causal robustness gates pass. The increasing gain
  cannot be explained by foundation improvement alone, a fixed amount of
  memory, shorter prompts, or arbitrary/corrupted memory content. Systems,
  less task-specific retrieval, longer-context replication, and competitive
  matched-compute baselines remain open.

## 2026-08-14 — isolated BABILong systems accounting

- Clean commit `ef2322d093679b3b3bbb1f1ed3f090c7ae046424` froze an
  arm-isolated CUDA-synchronized systems protocol before measurement. Each of
  the three SmolLM2 endpoints ran in six fresh processes: three repetitions for
  each decode order, with eight warm-up samples per arm and all 120 rows from
  robustness panel 1. A single run made dirty only by newly copied, untracked
  8K data was rejected and rerun clean before analysis.
- Predictions and accuracy were identical across every repetition. Median
  ASCENT-to-Foundation decode-time ratios for 135M/360M/1.7B were
  `0.4130`, `0.4247`, and `0.2791`; the complete fresh-process ranges were
  `[0.3975,0.4184]`, `[0.4224,0.4331]`, and `[0.2722,0.2822]`.
  Both decode orders agreed at every endpoint.
- Median Foundation-versus-ASCENT incremental CUDA peaks were
  `1541.3/26.6 MB`, `1120.0/26.2 MB`, and `2067.7/57.4 MB` for
  135M/360M/1.7B. Prompt-token ratios were `0.02279`, `0.02435`, and
  `0.02501`. These are isolated decode measurements, not claims that the
  external parser itself runs on the GPU.
- The causal write-state payload was fixed at mean `536.65` bytes across
  endpoints. Query-conditioned read state grew `52.36 -> 87.94 -> 103.95`
  bytes under the frozen 1/2/3-slot schedule. Total external state therefore
  grew `589.01 -> 624.59 -> 640.60` bytes, while total-state/model-artifact
  ratio strictly fell `2.189e-6 -> 8.631e-7 -> 1.872e-7`.
- Retrieval medians were `948.1`, `952.5`, and `953.0` microseconds/sample.
  Seventeen of 18 repetitions were below the frozen 1 ms threshold; one 360M
  repetition was `1075.1` microseconds/sample. Under the strict every-run
  interpretation, the retrieval sub-gate fails even though its median passes.
- The raw archive is
  `artifacts/remote_results/babilong_4k_systems_ef2322d.tar.gz`, SHA-256
  `e110b963d095b47e94388e37a30b675a45391fe6050480bf235a22d7da4f2995`.
- Decision: latency, peak-memory, accuracy-invariance, decode-order, and
  decreasing-relative-state gates pass. The overall frozen systems gate is not
  marked fully passed because of the single strict retrieval-threshold miss;
  FLOP accounting and matched systems baselines remain open.

## 2026-08-14 — official BABILong 8K confirmation

- The official `RMT-team/babilong-1k-samples` dataset was pinned at revision
  `fc4d1a584dfc498c37578753bee4cdd91b987ae2`. QA1/QA2/QA3 8K parquet and
  converted JSONL hashes were recorded. The target-blind causal reader was
  exact on all 2,998 rows and returned exactly 1/2/3 supporting facts for every
  QA1/QA2/QA3 row.
- Clean commit `e159aa3ffed8ba991b264a58fb0bf0517a7eaf6e` froze a 60-row
  development panel and three untouched 120-row confirmation panels before any
  8K model score. Development passed with gains
  `0.3000 -> 0.5000 -> 0.7000`; both adjacent increments were `+0.2000`.
- Clean confirmation commit
  `440ef7f42f19f907017370c0677228b663914ede` and config SHA-256
  `b5a15b831499c9f60a1e56edd2c4a0d2946af93c0f559748f173c21d336897bd`
  preserved the 1/2/3-slot schedule, prompts, decoding, model revisions, and
  panel hashes before any confirmation score. The 360M cells ran on the second
  RTX PRO 6000 node.
- The three untouched gain curves were
  `0.2833/0.5250/0.7500`, `0.3083/0.5250/0.6417`, and
  `0.2500/0.4833/0.6750`; every panel was strictly monotone. Endpoint mean
  gains were `0.2806`, `0.5111`, and `0.6889`, with panel-t 95% lower bounds
  `0.2079`, `0.4514`, and `0.5511`.
- Adjacent mean gain increments were `+0.2306 [0.1989,0.2622]` and
  `+0.1778 [0.0399,0.3156]`. The gain-on-log-parameter slope was
  `0.1564 [0.0879,0.2250]`. All frozen confirmation and provenance gates
  passed.
- Foundation prompt lengths ranged from 4,891 to 8,192 tokens across the
  confirmation cells. One unique panel-3 row reached the frozen 8,192-token
  cap at all three endpoints; all other unique rows were below the cap. The
  result is therefore an official 8K-config evaluation with this isolated
  truncation explicitly disclosed, not a claim that every row was untruncated.
- Task-stratified results expose a boundary: at 135M, QA3 gain was
  `-0.050/-0.050/0.000` across the three panels, while QA1 was strongly
  positive; 360M and 1.7B were positive on every task-panel cell. The primary
  task-balanced scale curve passes, but small-model QA3 must remain visible.
- The raw archive is
  `artifacts/remote_results/babilong_8k_confirm_440ef7f.tar.gz`, SHA-256
  `c6c162a6e90b2da6cdbc006c49507d7baed0071172e73f5cb326cbac23117d2c`.
- Decision: longer-context replication passes and materially strengthens the
  official compositional scale-complementarity claim. Less task-specific
  retrieval, matched-compute/current retrieval baselines, and broader tasks
  remain required before TNNLS promotion.

## 2026-08-14 — generic lexical retrieval control at 8K

- Clean commit `1d435662e1c6a55b62107a71357f740687461f3a` and config
  SHA-256 `aed634c807740ae4ed6de1966d68ed25934e24fee5d1499ec9fac85607d64d38`
  froze three additional official 8K panels (stable-hash positions 140:260),
  all model artifacts, and two same-read-slot 1/2/3 conditions before any
  control-panel score: structured causal ASCENT and a generic BM25-like lexical
  chain retriever.
- The generic code contains no BABILong people, objects, locations, events, or
  target vocabulary. It splits at generic punctuation, uses fixed
  alphanumeric tokens and question stop words, applies deterministic lexical
  scoring, expands from selected passage tokens, breaks exact ties by recency,
  and returns selected passages chronologically. It never reads the target.
- ASCENT gains on the three new panels were
  `0.3333/0.5750/0.6667`, `0.3083/0.5667/0.7000`, and
  `0.3000/0.5250/0.7083`; every curve was strictly monotone. Adjacent mean
  increments were `+0.2417 [0.2003,0.2831]` and
  `+0.1361 [0.0221,0.2501]`, and the log-parameter slope was
  `0.1431 [0.1019,0.1843]`.
- ASCENT exact accuracy exceeded the same-read-slot generic retriever by mean
  `+0.0694 [0.0575,0.0814]`, `+0.4972 [0.4340,0.5605]`, and
  `+0.6472 [0.5516,0.7428]` at 135M/360M/1.7B. Foundation predictions were
  identical sample-by-sample between conditions, and all three endpoint lower
  bounds were positive.
- The generic retriever itself had mean gains `0.2444`, `0.0583`, and
  `0.0444`; its 1.7B interval included zero. It was advantaged in write
  capacity: the full-corpus UTF-8 lower bound averaged `30,190.46` bytes,
  versus only `544.70` bytes for ASCENT's causal write state. Generic read
  payloads averaged `78.74/205.15/372.55` bytes versus ASCENT
  `52.58/88.24/104.49` bytes.
- Task strata localize the advantage. At 135M, both methods tied on QA3 and the
  aggregate advantage came mainly from QA2; generic retrieval was already
  strong on one-hop QA1. At 360M, every task-stratified panel interval favored
  ASCENT. At 1.7B, QA2 and QA3 advantages were large and had positive lower
  bounds, while the QA1 lower bound was slightly negative. The primary
  task-balanced control passes, but this is not a claim of uniform task-level
  superiority.
- All commit/config/data/model/parser/clean-tree gates passed; no input in
  these three panels reached the 8,192-token cap. The raw archive is
  `artifacts/remote_results/babilong_8k_generic_1d43566.tar.gz`, SHA-256
  `6cb7832fe615ddf6407b524a58d69e99437a2dd29a29dc4402d1920850353941`.
- Decision: the frozen generic lexical-retrieval control passes and rules out
  the weak explanation that any short, full-corpus lexical retrieval prompt
  produces the scale curve. This is not a same-write-byte or same-FLOP result,
  nor a comparison with current neural retrieval/memory systems; those gates
  remain open.

## 2026-08-14 — official BABILong 8K neural retrieval and byte-matched controls

- The retrieval protocol was frozen at clean commit `78ab8d1` before any
  cache generation or model score. It uses the official BGE-M3 revision
  `5617a9f61b028005a4858fdac845db406aefb181`, generic BM25, reciprocal-rank
  fusion, three iterative target-blind hops, and the official
  `bge-reranker-v2-m3` revision
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`. Both 2.27 GB weight artifacts
  were checked by SHA-256 before use. A second control stores exact raw event
  strings in a FIFO under each sample's exact ASCENT semantic write-byte
  capacity, then exposes the same 1/2/3 read slots.
- The 60-row development panel passed from clean commit `335c030`: ASCENT
  accuracy was `.400/.600/.933`, compared with `.217/.183/.283` for the
  byte-matched raw FIFO and `.100/.150/.233` for BGE. ASCENT-minus-Foundation
  gains were exactly `.300/.500/.700`. The pass decision was committed as
  `0a797ac` before confirmation cache generation.
- Commit `f0c8d44` froze three untouched official panels (stable-hash positions
  260:380), conditions, endpoints, metrics, and gates before any confirmation
  cache was generated. Cache hashes were then attached without inspecting
  confirmation recall or decoder scores. The executable confirmation commit
  was `072adbd014ce4123c8eba284ea7ab635e912afde`, and its config SHA-256 was
  `d6987b5780b12ee4e39666536805c133408c6055a2defee57bbda080196c405c`.
- ASCENT accuracy on the three untouched panels was
  `.350/.717/.917`, `.425/.683/.933`, and `.350/.667/.908` across
  135M/360M/1.7B. Corresponding gain curves were
  `.208/.517/.708`, `.275/.500/.700`, and `.242/.483/.600`; every panel was
  strictly monotone.
- Panel-clustered adjacent gain increments were
  `+.2583 [+.1488,+.3679]` and `+.1694 [+.0554,+.2835]`. The gain-on-log-model-
  parameter slope was `+.1629 [+.0946,+.2313]`. All three preregistered lower
  bounds were positive.
- ASCENT accuracy advantages over the byte-capacity-matched raw FIFO were
  `+.1444 [+.0267,+.2622]`, `+.5167 [+.3732,+.6601]`, and
  `+.6083 [+.5536,+.6631]`. Advantages over BGE-M3 hybrid+rereanking were
  `+.2833 [+.1384,+.4282]`, `+.5667 [+.5460,+.5874]`, and
  `+.6556 [+.5087,+.8024]`. Thus all six endpoint-by-control lower bounds were
  positive.
- Mean ASCENT write payload was 555.23 bytes. The raw FIFO had the exact
  per-sample ASCENT capacity but used 481.15 bytes on average because an
  indivisible next raw event could not always fit. BGE's reported raw-text plus
  float16 dense-index lower bound averaged 518,401.47 bytes, excluding sparse
  structures, model weights, reranker compute, and transient activations.
- Only after all 27 decoder results were frozen, the BGE support audit was
  opened. Across panels, complete support-chain recall at 1/2/3 passages ranged
  `.092-.150`, `.175-.258`, and `.258-.392`; fact recall ranged `.147-.221`,
  `.324-.364`, and `.419-.474`. These results explain the neural retriever's
  difficulty on causal multi-hop chains and were not used for tuning.
- Single-commit, single-config, exact cache-hash, clean-worktree, parser,
  condition-source, row-alignment, and sample-by-sample Foundation-invariance
  gates all passed. One unique Foundation row reached the 8,192-token cap.
  Task-stratified lower bounds were positive at 360M and 1.7B; at 135M, QA2
  intervals crossed zero and QA3 tied, so the claim remains task-balanced and
  scale-dependent rather than uniform at the smallest endpoint.
- The raw archive is
  `artifacts/remote_results/babilong_8k_neural_controls_072adbd.tar.gz`,
  SHA-256
  `17f9fda974f75daf3f6ed555c2bf6ade8d420f4c14a2b92c77f45c2ee9ca2192`.
- Decision: the current-neural-retrieval and same-byte-capacity controls pass
  all frozen primary gates and materially strengthen the central claim that
  ASCENT extracts increasing benefit with model scale. They do not close the
  same-FLOP/static-adapter, broader-task, overwrite-boundary, or independent
  rerun-package gates.

## 2026-08-14 — same-inference-FLOP static replay adapter

- A development control used the already confirmed Pythia latent-replay
  interface at relative depth 0.25 with 11 full-sequence replay tokens. The
  comparison adapter is a single calibration-trained replay block shared by
  every episode. It receives no episode observation, value, key-specific
  parameter, test target, or query-dependent parameter. Dynamic ASCENT and the
  static adapter execute the same `replay_from_hidden` call, replay-token
  count, hidden width, injection layer, frozen suffix, and candidate
  projection. Offline adapter training, dynamic memory decoding, parameter
  bytes, and transient bytes are reported separately rather than called equal.
- V1 development at commit `2ae9b3e` used 64 optimizer steps and passed, but
  retained the final iterate. To strengthen the control, V2 was disclosed and
  frozen at commit `a659a27` before its score: 128 steps, learning rate .01,
  full-calibration evaluation after every epoch, and minimum-calibration-NLL
  checkpoint selection. V2 static gains were +.1274 nats at 410M and +.1686 at
  2.8B. Dynamic rich ASCENT gains were +1.3281 and +1.4273; dynamic-over-static
  paired lower bounds were +1.1518 and +1.2055. The dynamic rich gain also
  increased by +.0992 [.0826,.1159] from 410M to 2.8B.
- Clean commit `bbe577ce389a23e844ddc7a8c2527b04ded4cd0f` and config SHA-256
  `338e2f9534ccaecf4fbfa8d21d5d297174bb4720278cdcb646fa1d2b7440f61f`
  froze three untouched seeds (20260941/20260951/20260961), 2,048 calibration
  and 8,192 test episodes per endpoint, the V2 optimizer/checkpoint protocol,
  exact model artifacts, and all gates before any registered seed score.
- The static adapter was a nontrivial baseline: its mean gain was +.1097 nats
  at 410M and +.1757 at 2.8B. Dynamic ASCENT gains averaged +1.3451 and
  +1.4577. Per-seed dynamic-over-static advantages were
  `1.2228/1.2436/1.2399` nats at 410M and
  `1.2713/1.2821/1.2927` at 2.8B; every within-seed paired 95% lower bound was
  positive.
- Seed-clustered t(2) intervals for dynamic-over-static advantage were
  `+1.2354 [+1.2079,+1.2629]` at 410M and
  `+1.2820 [+1.2554,+1.3086]` at 2.8B. The dynamic rich gain increase from
  410M to 2.8B was `+.1125 [+.0969,+.1282]` across seeds.
- The 410M endpoint ran on the second RTX PRO 6000 node after its official
  weight artifact was transferred and independently verified. The 2.8B
  endpoint ran on the first node. Single-config, single-commit, clean-tree,
  registered-seed, exact-model-hash, identical-label, identical-signal,
  identical-query, exact-test-label, and exact same-suffix-FLOP gates all
  passed.
- The raw archive is
  `artifacts/remote_results/static_replay_same_flop_bbe577c.tar.gz`, SHA-256
  `abb801b7066a9965b6fbecb7082d167f0a960044aad61b0e2d25cf0bd155a61a`.
- Decision: the registered same-inference-FLOP static-adapter gate passes. The
  dynamic advantage cannot be explained by simply adding an equally shaped
  static replay block or by executing the frozen suffix on extra tokens. This
  closes the proposal's principal same-FLOP mechanism control, while broader
  task-level matched-compute systems comparisons and the overwrite boundary
  remain open.

## 2026-08-14 — cross-node prediction replication and latency-tail audit

- Clean commit `c87fa8c8c3308cdf347ffa0d5d50280f7fc81e08` froze two audits
  before any new score: a three-panel cross-node rerun of the 360M BABILong 8K
  scale-relevant arm, and six additional fresh-process measurements of the
  single 360M retrieval-latency tail. Neither audit is counted as a new
  independent data seed, and the original 1.075 ms failure remains retained.
- The 360M arm had originally run on the second physical node. Re-executing it
  on the first node exactly reproduced panel gains
  `+.5166667/+.5000000/+.4833333`, ASCENT accuracies
  `.7166667/.6833333/.6666667`, wins `64/67/61`, and regressions `2/7/3`.
  Every panel hash and executed model-file record matched. More stringently,
  every normalized Foundation/ASCENT prediction, score, retained fact list,
  row order, and Foundation/ASCENT token count was byte-for-byte equal to the
  archived reference signature. The cross-node replication gate passed.
- In the latency audit, all six new runs reproduced accuracy and predictions
  exactly. Retrieval times were `963.87`, `952.79`, `952.42`, `952.41`,
  `952.71`, and `949.58` microseconds/sample; none exceeded 1 ms. The combined
  old-plus-new set has median `952.57` microseconds/sample and exactly one of
  twelve measurements above 1 ms: the previously reported 1.075 ms run.
  Median decode-time ratio in the six new runs was `.42434`.
- The frozen isolated-tail audit passes. This supports interpreting the 1.075
  ms observation as an isolated scheduling/runtime tail, but it does not
  retroactively convert the original strict every-run-below-1-ms gate into a
  pass.
- Raw archives are
  `artifacts/remote_results/cross_node_replication_c87fa8c.tar.gz` (SHA-256
  `81184e8d6cd8a2c6edc4cfdd478da88b8f4db1ee1ca532fc1c65222c7d0ba97e`)
  and `artifacts/remote_results/latency_tail_c87fa8c.tar.gz` (SHA-256
  `380f49daec955c27d9bb1eeeecfeb9d2580f80da6cbd40da31759a7630428302`).
- A consolidated `reproduction/` package now records the exact core runtime,
  the two GPU-driver variants, immutable config/data/archive hashes, frozen
  commands, and a read-only integrity verifier. The verifier passes every file
  hash, JSON parse, and embedded BABILong/same-FLOP primary gate. This closes
  the local packaging and physical-node replication gaps; it is not a claim of
  independent external-team reproduction.
- The self-contained source/data/result/environment release archive is
  `artifacts/remote_results/ascent_core_reproduction_ffe6c78.tar.gz`, SHA-256
  `b67d521e7c5e6153395d35225b168ba6d5b9babffa227f279c76727da271069c`.
  Its embedded `reproduction/verification.json` records all package gates as
  passed.

## 2026-08-14 — exact-generate supported-operator FLOP profile

- Clean commit `e49eae913e52ab7966d87e1bdf23223627f8ab1f` froze the
  profiler, panel, three SmolLM2 endpoints, exact model artifacts, state law,
  measurement boundary, and gates before any profiled decoder run. The config
  SHA-256 is
  `8c088dfb1ab6947354be0fadced6dcce36ecaade832ef7198448a32bb05e743d`.
  The 135M and 1.7B endpoints ran on the first GPU node; 360M ran concurrently
  on the second.
- On the untouched official BABILong 8K panel, Foundation-to-ASCENT accuracy
  was `.1417 -> .3500`, `.2000 -> .7167`, and `.2083 -> .9167`. Thus the
  ASCENT gain curve was `+.2083 -> +.5167 -> +.7083`, strictly increasing with
  model scale and exactly reproducing the previously archived normalized
  prediction vectors sample by sample.
- Across all 120 exact greedy `model.generate` calls, ASCENT/Foundation
  supported-operator FLOP ratios were `.01195/.01259/.01279`, corresponding to
  `83.67x/79.45x/78.20x` reductions. Padded prompt-token ratios were
  `.01081/.01156/.01199`. All three preregistered below-0.10 FLOP-ratio gates
  passed.
- Profiler-enclosed wall-time ratios were `.9008/.8651/.7262`, or measured
  speedups of `1.11x/1.16x/1.38x`. All are faster than the long-context arm,
  and the speedup grows with scale, but none passes the deliberately strict
  below-0.10 wall-ratio gate (at least `10x`). Consequently the frozen combined
  profile gate is retained as failed rather than weakened after seeing the
  scores.
- The FLOP values are explicitly a supported-operator lower bound from
  `torch.profiler(with_flops=True)`. Fused attention, operators without FLOP
  formulas, CPU retrieval, model loading, and state construction are not
  declared zero. Repeated per-batch profiler startup and synchronization also
  make its enclosed wall time a diagnostic rather than a deployment-latency
  estimate; the separate fresh-process systems panel remains the appropriate
  latency evidence.
- All config/data/model/commit/clean-tree and exact-prediction gates passed.
  Raw node archives have SHA-256
  `eddcdc08fa9fadb0960fe0144971681594eae54079e6f09ff13c2cfeeeb213f4`
  and
  `2358338835174d070121172ac8a4cef5cd7019a10e93127706f9c300029606a8`.
  The consolidated config/code/analysis/raw bundle is
  `artifacts/remote_results/babilong_flops_profile_e49eae9.tar.gz`, SHA-256
  `6ce48c08baff7dfd5e53f880ded7833f8525f27ad429ef83f8e067436c208d40`.
  Decision: the supported-operator FLOP evidence strongly improves the
  generative systems case while transparently leaving the preregistered 10x
  profiled-wall target and complete end-to-end operator accounting open.

## 2026-08-14 — official RULER FWE aggregation development at 4K

- The official NVIDIA RULER repository at generator commit
  `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` was used to generate 60
  new FWE aggregation rows at seed 377804. Clean commit
  `a386ec0eb730a16bdf20f13f6e8468158fa86395` froze the data, exact model
  hashes, target-blind bounded counter, `1/2/3` certified-word scale law,
  fixed one-word control, metric, fusion, and gates before any model score.
- The counter reads only six-letter codes before the final question; it never
  reads the official `outputs` field while writing or ranking. Its top three
  exactly matched all 60 official references. The persistent table contained
  30--33 codes and used 660--726 bytes under a frozen 256-entry bound.
- Foundation official string-match scores were `.2722/.2833/.7056` at
  135M/360M/1.7B. Scale-ASCENT gains were `+.1333/+.4167/+.2944`, with
  adjacent differences `+.2833/-.1222`; all scale arms had zero regressions.
  Fixed one-word gains were `+.1333/+.1500/+.0056`, so the expanded state beat
  the fixed control at both larger endpoints and relative state cost strictly
  decreased.
- The frozen **absolute-gain** development gate failed solely because the
  middle-to-large difference was negative; this is not an ASCENT performance
  failure. The cause is visible metric saturation at 1.7B:
  with three certified words, ASCENT reaches full official credit and can gain
  at most `1-.7056=.2944`.
- Raw node archives are
  `artifacts/remote_results/fwe_dev_a386ec0/fwe_dev_a386ec0_node1.tar.gz`
  (SHA-256
  `976970e376071ed25886a982448facccbc8e97c9e77810acc3775364ff6b65fd`)
  and
  `artifacts/remote_results/fwe_dev_a386ec0/fwe_dev_a386ec0_node2.tar.gz`
  (SHA-256
  `6a1b3f8d6a2ab847c5b3be87ed658762a110f4bc4e0798ad8373a406d9b1e3b6`).
  Decision: retain the 4K absolute-gain gate miss as a saturation boundary and
  test one disclosed difficulty repair
  at the models' native 8K context using a new seed, without changing the
  parser, state law, model ladder, metric, fusion, or gates.

## 2026-08-14 — official RULER FWE aggregation at 8K

- A disclosed difficulty repair changed only context length and data seed after
  the 4K saturation-boundary result. Commit
  `26cbf9dc82156c28856f76f8a1af30f646559e11` froze seed 378803 at the
  models' native 8K limit while retaining the same parser, `1/2/3` state law,
  model ladder, official metric, fusion, and development gates.
- Development scale gains were `+.1833/+.3278/+.4333`; adjacent differences
  were `+.1444/+.1056`, all cells had zero regressions, parser/provenance/state
  gates passed, and the point-estimate development gate passed. The raw node
  archives have SHA-256
  `ee01e108f63b466b9633911aa0a89551df3b1e28f19b9fba85ff14f98a5b3a77`
  and
  `d8d8178cd35ca2d2682428203eb02c9719a5d443fe821858cf7569f955b9e2d6`.
- This development run emitted a Hugging Face warning because the official
  base-template length generator did not reserve the later SmolLM2 chat
  wrapper. It was not promoted. Three entirely new confirmation seeds were
  generated with a 192-token reserve. Before any score, all 180 rows were
  tokenized by all three endpoint tokenizers: the maximum chat prompt was 7979
  tokens and prompt plus the frozen 50-token output budget was 8029, below the
  native 8192 limit.
- Commit `f416297554e0c8028a25d8dd899fe3e1af6fcdcc` froze the corrected
  three-seed confirmation panel before any confirmation score. Per-seed gain
  curves were `+.2056/+.3444/+.4611`, `+.2111/+.3889/+.2611`, and
  `+.1778/+.2611/+.4000`. Every endpoint gain was positive, all nine cells had
  zero regressions, all endpoint gain t(2) lower bounds were positive, and all
  parser/provenance/native-length/state/fixed-control gates passed.
- The equal-seed mean curve increased `+.1981 -> +.3315 -> +.3741`, and the
  low-to-middle increment passed at `+.1333 [+.0154,+.2512]`. The
  middle-to-large increment was only `+.0426 [-.3250,+.4102]`; its lower bound
  and the slope lower bound `-.0533` failed. Seed 380202 reversed because its
  1.7B Foundation reached `.7389` while scale ASCENT reached the metric ceiling
  of `1.0000` on every confirmation seed.
- After the user identified that an already-perfect ASCENT arm cannot produce
  an impossible score above `1.0`, a clearly labeled **post-hoc ceiling-aware**
  analysis was added without changing the original frozen confirmation
  verdict. The fraction of Foundation's remaining error eliminated by ASCENT,
  `(ASCENT-Foundation)/(1-Foundation)`, increased
  `.2508 -> .5171 -> 1.0000`. Its paired adjacent increases were
  `+.2663 [+.1725,+.3601]` and `+.4829 [+.3254,+.6403]` across the three
  untouched seeds. Thus FWE supplies strong scale-complementarity evidence
  under a ceiling-correct estimand, while the original absolute-gain gate is
  retained verbatim as a metric-design miss rather than rewritten post hoc.
- The confirmation archives are
  `artifacts/remote_results/fwe_confirm_f416297/fwe_confirm_f416297_node1.tar.gz`
  (SHA-256
  `627f5ad014a3de271ed38ecadaf49b3428d4ca4b1a222c697b6ac06d9fc9f328`)
  and
  `artifacts/remote_results/fwe_confirm_f416297/fwe_confirm_f416297_node2.tar.gz`
  (SHA-256
  `de24a117feae624597147f7458f33b67d645dcf72c70cf9b8a7285d855896bc1`).
  Decision: the official FWE category shows positive absolute mean growth and
  strongly increasing remaining-error elimination, culminating in perfect
  ASCENT accuracy. The preregistered absolute-gain confirmation gate remains
  formally missed because it did not model the three-answer ceiling. Retain
  both facts and move to official ten-answer CWE for a less-censored primary
  test, not to post-hoc FWE seed or slot tuning.

## 2026-08-14 — official RULER CWE ten-answer aggregation confirmation

- The official RULER CWE task was selected before development scores because
  its ten-reference target provides a less-censored test than FWE's
  three-reference ceiling. Official generator commit
  `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` and its exact 8,564,991-byte
  LFS word asset (SHA-256
  `affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca`)
  generated 60 length-safe development rows at seed 381803. Commit
  `9cd84c32565ff66610f7d67cca4c8b0ccf6ad06b` froze the target-blind exact
  counter, fixed three-word control, width-linear capped `3/5/10` certified
  evidence rule, models, data, hashes, metrics, and development gate before
  any CWE model score.
- Development gains were `+.2917 -> +.4400 -> +.9833`; adjacent differences
  were `+.1483/+.5433`. All 180 endpoint-sample pairs benefited, none
  regressed, the parser exactly recovered all 60 official top-ten sets, scale
  expansion beat the fixed three-word control at both larger endpoints, and
  total state bytes per model parameter strictly decreased. The frozen
  development gate passed and authorized only three entirely new official
  seeds.
- Seeds 382101/382202/382303 were generated at the same safe 8000-token base
  length and had SHA-256
  `eaea849fadee495ca8d06d52f86f6418502447a2ea828dc5d7e26f4ded7178c8`,
  `1b5221180468e40cc096b9e32a89e376398b550ac34af279e63a2f6d77238887`,
  and `3f263649c56850bc7dddc319141910391c8fccf44ad1887aa832fe1f8335b167`.
  Before scoring, the causal parser matched all 180 top-ten sets, unique
  entries stayed within 277--298 under the 512-slot bound, and every one of
  the 540 seed/tokenizer rows fit natively: the maximum chat prompt was 7910
  tokens and prompt plus the frozen generation budget was 8030, below 8192.
- Commit `3771662e795d144df56c29e31c21152c07d07b88` froze the confirmation
  panel. The first launch stopped at config parsing before loading a model or
  producing any score because the runner required the legacy singular
  `evaluation_samples` field. Commit
  `c749f246d8aca49f3c54e1d3d4d0240e9bc0e6ce` added that value-preserving
  compatibility field and a fixed greedy seed before any confirmation score;
  data, models, scale law, metrics, and gates were unchanged.
- Per-seed absolute-gain curves were
  `+.2817/+.4467/+.9733`, `+.2667/+.4417/+.9567`, and
  `+.2400/+.4250/+.9383`. The equal-seed means were
  `+.2628 -> +.4378 -> +.9561`. Paired adjacent increments were
  `+.1750 [+.1502,+.1998]` and `+.5183 [+.5003,+.5364]`; the log-parameter
  slope was `+.2780 [+.2744,+.2816]`. Endpoint gain lower bounds were
  `+.2104/+.4096/+.9126`. All intervals are seed-clustered t(2) 95%
  intervals.
- The original absolute-gain confirmation gate passed in full: every seed had
  positive endpoint gains and positive adjacent differences; all nine cells
  had zero regressions; all parser, hash, clean-tree, native-length,
  fixed-control, and relative-state gates passed. The secondary ceiling-aware
  estimand was also frozen before confirmation scores and increased
  `.2777 -> .4952 -> 1.0000`; adjacent lower bounds were `+.1897` and
  `+.4923`. Thus this result does not depend on substituting a post-hoc metric.
- Raw node archives are
  `artifacts/remote_results/cwe_confirm_c749f24/cwe_confirm_c749f24_node1.tar.gz`
  (SHA-256
  `81f064678bf65af3c916cd7552744e8e919e2f97ad6101115eb0d52662d1a591`)
  and
  `artifacts/remote_results/cwe_confirm_c749f24/cwe_confirm_c749f24_node2.tar.gz`
  (SHA-256
  `35e0b04a202a04fcf478273ac41c6e4fd0a79f54bb36652ef4385bb6e14ac8e8`).
  Decision: promote official CWE aggregation as a primary scale-
  complementarity result. It materially strengthens, but does not alone
  complete, the multi-task evidence required for the final TNNLS claim.

- A fresh-process repeat on the second physical node independently reran the
  360M endpoint for all three confirmation seeds. Foundation, fixed/scale
  ASCENT summaries, parser results, win/regression counts, and every per-row
  prediction reproduced exactly for gains `+.4467/+.4417/+.4250`. The repeat
  archive is
  `artifacts/remote_results/cwe_confirm_c749f24/cwe_confirm_repeat_c749f24_node2.tar.gz`,
  SHA-256
  `1e36dcad26303161ce74be9dfa1ca79ce6a98393fceb3f4518a21f26d38399b7`.

## 2026-08-14 — Qwen2.5 cross-architecture official CWE at 16K

- Cross-architecture development used a newly generated official CWE seed
  383803 at 16K, exact data SHA-256
  `73e0014a4cfb4d2f589d6cae15f22fc599a75d62516d62d6fe04cb6170270fa1`.
  Commit `985713575f93d1a3277a089ab779ca7d3588d508` froze Qwen2.5
  0.5B/1.5B/3B, the 1024-entry causal counter, fixed three-word control,
  hidden-width-derived capped `3/7/10` rule, exact model shards, metrics, and
  gates before any Qwen CWE score. The parser matched 60/60 top-ten sets;
  observed unique entries were 638--655 and maximum prompt plus output budget
  was 16029 under the native 32768 limit.
- Development absolute gains were `+.2733 -> +.6117 -> +.7400`, with adjacent
  differences `+.3383/+.1283`, 180/180 endpoint-sample wins, zero regressions,
  and all provenance/fixed-control/relative-state gates passing. This
  authorized only three new confirmation seeds.
- Commit `cd8352f46fe2001739cc63096b8d11f024bd228a` froze entirely new seeds
  384101/384202/384303 before any confirmation score. Their data SHA-256 were
  `b311f57c4e176bdbb3d4597ef432fcf3d18c4dbcb1efdb445586bc3b29e975e5`,
  `561290b4481bf18dff5b897acc5aefccc682fab67ef40c33391ec2fe81fb2c12`,
  and `a1055db59cea4103705da52b76e9594ab2a3d148067e7113169a39de69fa230f`.
  All 180 top-ten parser sets were exact, unique entries remained 635--661,
  and all 540 seed/tokenizer rows were native-length safe with a maximum
  prompt-plus-output budget of 16028/32768.
- Per-seed absolute-gain curves were `+.2200/+.6067/+.7400`,
  `+.2567/+.6167/+.7167`, and `+.2483/+.6200/+.6733`. Thus every one of the
  six adjacent differences was positive and all nine cells had zero
  regressions. Equal-seed means were `+.2417 -> +.6144 -> +.7100`; endpoint
  gain lower bounds were `+.1939/+.5972/+.6260`, and the log-parameter slope
  was `+.2627 [+.2020,+.3234]`.
- The paired low-to-middle absolute increment strongly passed at
  `+.3728 [+.3396,+.4060]`. The middle-to-large mean was still positive at
  `+.0956`, but its t(2) interval `[-.0043,+.1954]` narrowly crossed zero.
  Therefore the original all-adjacent-absolute-LCB gate is formally retained
  as **not passed**, despite every seed increasing and the overall slope LCB
  being positive.
- This miss was anticipated by the ceiling issue and not repaired after
  scores: the remaining-error-elimination secondary was explicitly frozen in
  `cd8352f` before confirmation. It rose
  `.2581 -> .6917 -> 1.0000`; paired increments were
  `+.4336 [+.3988,+.4683]` and `+.3083 [+.2973,+.3193]`. Scale ASCENT reached
  exactly 1.0 at 3B for every seed, so demanding a continuing absolute gain
  with zero variance would require accuracy above the metric ceiling. The
  pre-score ceiling-aware confirmation gate passed strongly.
- Raw development and confirmation archives are
  `artifacts/remote_results/qwen_cwe_dev_9857135/qwen_cwe_dev_9857135_node1.tar.gz`
  (SHA-256
  `f28c2596ddb5259e3cf6a400459db9d249fd2367825b6e2cfc4f3864b87c0d53`)
  and
  `artifacts/remote_results/qwen_cwe_confirm_cd8352f/qwen_cwe_confirm_cd8352f_node1.tar.gz`
  (SHA-256
  `48b3eb6c3c8c07193ac4cb1832306df4caec1d80ebea399643f2ba61b27b22e9`).
  Decision: cross-architecture scale complementarity is confirmed under the
  prospectively frozen ceiling-aware estimand and supported by positive
  per-seed absolute curves and a positive absolute slope interval. Do not
  claim that the stricter absolute-adjacent-LCB gate passed.

## 2026-08-14 — posterior-mean overwrite redesign and falsification

- The replicated negative overwrite argmax result was not retuned. A
  substantive representation redesign instead preserved the exact
  current-version posterior as a posterior-weighted latent codebook mean,
  rather than collapsing it prematurely to one MAP label. Latest-version-only
  masking remained mandatory; posterior-mean unmasked/reversed history and the
  old argmax read were frozen as ablations. Commit
  `8f8cbed2c3feb3c5fdebad0814b9d4d1fc942f2d` and config SHA-256
  `cbee43fa6f571e2d4cace8615a5ff4eefeb61a31c3667b85216c072a6600c18f`
  froze seed 20261011, interface, Pythia endpoints, one/four-round signals,
  2,048 calibration plus 8,192 test episodes, and the original positive
  foundation-by-signal interaction gate before either endpoint score.
- Posterior-mean replay materially improved capability over its same-run old
  argmax ablation. At 410M, fixed/rich gains were
  `+.7885 [+.7568,+.8201]` and `+2.0211 [+1.9916,+2.0506]`, versus argmax
  point gains `+.7179/+1.4923`. At 2.8B, posterior-mean gains were
  `+.7500 [+.7213,+.7788]` and `+1.7596 [+1.7352,+1.7841]`, versus argmax
  `+.7401/+1.4291`.
- The scale interaction nevertheless became decisively negative:
  `-.2230 [-.2350,-.2110]`. All current-label/signal/query/config/commit,
  clean-tree, and latest-version invariants passed. Thus the redesign improves
  both endpoint capabilities but does not make the larger foundation extract
  more marginal value from the rich overwrite signal.
- Raw archives are
  `artifacts/remote_results/overwrite_posterior_mean_8f8cbed/overwrite_posterior_mean_8f8cbed_node1.tar.gz`
  (SHA-256
  `1b7bfccdd360184f3912f3d452eea68fafeb89a8043051a3196aa8a767113305`)
  and
  `artifacts/remote_results/overwrite_posterior_mean_8f8cbed/overwrite_posterior_mean_8f8cbed_node2.tar.gz`
  (SHA-256
  `235846585012060eec1bf496a09461a724cbc0e9801be81aa19b4ebc05edf656`).
  Decision: stop depth/round/template tuning for this interface. Retain
  overwrite as a stable external-validity boundary; any further attempt must
  change the version-address/value architecture or explicitly narrow the
  final claim away from overwrite.

## 2026-08-14 — separated version-address/value overwrite development

- A third, substantive overwrite interface separated address selection from
  value replay. Each current event bound a mean-pooled key/version address
  latent to a posterior-mean value latent; target/distractor order was
  randomized. The frozen controls were unmasked four-event history and
  shuffled address-value binding. Commit
  `9ecefb6ebd6c44474bcbe1c19108c9e201de8e11` and config SHA-256
  `97257111e66e771515b0b9282834b696d08d69e2e2a3a7f4defd55f9cd0d6896`
  fixed seed 20261021, Pythia 410M/2.8B, one/four signal rounds, 512
  calibration plus 4,096 paired test episodes, metrics, and gates before any
  episode score. The only post-freeze compatibility repair occurred before
  scoring and replaced a false equal-token-length assertion with masked mean
  pooling over all valid address tokens; nine tests passed.
- Primary ASCENT gains were `+.4786 [+.4513,+.5060]` and
  `+1.1783 [+1.1478,+1.2088]` at 410M, versus
  `+.4373 [+.4078,+.4669]` and `+1.1883 [+1.1586,+1.2180]` at 2.8B.
  Consequently the frozen foundation-by-signal interaction was positive for
  the first time in the overwrite series:
  `+.05131 [+.03573,+.06689]`. All four gain lower bounds and all provenance,
  clean-tree, query, current-version, address-order, and pooling invariants
  passed.
- Latest-only replay also beat unmasked history at all four cells, with paired
  advantages `+.1283 [+.1120,+.1447]` and
  `+.3954 [+.3791,+.4117]` at 410M, and
  `+.1253 [+.1086,+.1420]` and `+.3520 [+.3354,+.3685]` at 2.8B.
- The pre-frozen binding-specificity gate failed. Correct address-value
  binding differed from shuffled binding by only
  `+.00136 [-.00592,+.00864]` / `-.00149 [-.00985,+.00688]` at 410M and
  `+.00039 [-.00397,+.00475]` / `+.00229 [-.00333,+.00791]` at 2.8B.
  Thus the positive scale interaction is real for the registered interface,
  but cannot yet be attributed to address-value binding. Development is
  retained as a partial positive/falsified-mechanism result; no confirmation
  seeds or post-hoc tuning are authorized.
- Raw archives are
  `artifacts/remote_results/overwrite_address_value_9ecefb6/overwrite_address_value_9ecefb6_node1.tar.gz`
  (SHA-256
  `a098765b1c592f3aa7e68b4bbfd66f01ea75fc35ec2aa7b5270d855df32c1729`)
  and
  `artifacts/remote_results/overwrite_address_value_9ecefb6/overwrite_address_value_9ecefb6_node2.tar.gz`
  (SHA-256
  `de4f4a075ca37c39f9f8c2d0f8a11d22b65d94d16e9fcd45e27aed2d7415114a`).

## 2026-08-14 — binding-dependent affine overwrite scale unlock

- A fourth overwrite screen removed the accuracy ceiling and made correct
  role binding necessary for the registered task. Two distinct random keys
  each received obsolete/current noisy values; an exact certified address
  decoder selected the latest versions, and the latent suffix had to predict
  the noncommutative target `2 * first_current + second_current`. Stored event
  order was randomized. Paired NLL gain, not accuracy, was the primary metric.
  Commit `61b8f94b008ee25ff1d0a3e6045f3318af342d97` and config SHA-256
  `56af905ad1842ce6bee3ed3bc6dfe0fe6311c88a0c3a8eadc2ffbcf69bc7ff35`
  froze seed 20261031, Pythia 410M/2.8B, one/four signal rounds, 512
  calibration plus 4,096 test episodes, the affine target, prompts, codebooks,
  metrics, and all gates before any model score. Candidate/tokenizer, model
  hash, config, and clean-tree audits passed on both physical nodes.
- At 410M the calibration-safe fusion gate rejected every replay arm, yielding
  exactly zero fixed/rich gain. At 2.8B the gate enabled the latent replay and
  gains were `+.15110 [+.14438,+.15782]` and
  `+.17785 [+.17070,+.18499]` nats. Thus the task exposed a clear scale
  capability threshold: the larger foundation used replay that the smaller
  foundation could not safely use.
- The richer signal's marginal gain was zero at 410M but
  `+.02675 [+.02242,+.03108]` at 2.8B. The identical paired interval therefore
  gives a strictly positive frozen foundation-by-signal interaction. At 2.8B,
  latest-only replay also beat obsolete unmasked history by
  `+.03317 [+.03025,+.03608]` and `+.04455 [+.04135,+.04775]` for fixed/rich.
- The development gate nevertheless failed. Both 410M gain/expansion lower
  bounds were zero, and at 2.8B shuffled role binding was slightly better than
  correct binding: correct-minus-shuffled differences were
  `-.00408 [-.00546,-.00270]` and `-.00700 [-.00854,-.00546]` nats. The result
  therefore supports a scale-unlock observation for role-specific replay but
  falsifies the claimed causal address-value binding mechanism. No seed,
  target-function, depth, round, or template tuning and no confirmation are
  authorized for this frozen interface.
- Raw archives are
  `artifacts/remote_results/overwrite_affine_binding_61b8f94/overwrite_affine_binding_61b8f94_node1.tar.gz`
  (SHA-256
  `35373a7429c89d91c8103e83e8f48ddac637d51f62b36ff142496620da9b990c`)
  and
  `artifacts/remote_results/overwrite_affine_binding_61b8f94/overwrite_affine_binding_61b8f94_node2.tar.gz`
  (SHA-256
  `08462d1b81d9a610a120465903732521efa83e3c2cb212c6ee86afa433c9ecb2`).

## 2026-08-14 — official 16K RULER multikey headroom screen

- The previously frozen official NVIDIA RULER `niah_multikey_1` full-context
  greedy screen was executed without changing its task, seed, model, decoder,
  prompt, or gate. Config
  `ruler_niah_qwen2p5_full_context_16k_multikey_greedy_development.json`
  (original freeze commit `1698cd5`, SHA-256
  `0754849ad3ae76becd8f9c7a9c49c257907ed3befdf0f55e8236ec34d39bf551`)
  used the exact official seed-370804 task file (SHA-256
  `3fa39607aef7e52b74a96ab3a49970975729f9beee737f521f2388fbfafe8075`),
  complete 16K essay contexts, repetition-free greedy decoding, 20 calibration
  and 40 test samples, and Qwen2.5-0.5B.
- Foundation scored `.975` (39/40), leaving only one error. The calibration
  latent gain was `+.000773` nats but its 95% lower bound was `-.001242`, so
  the frozen safe gate correctly set its effective weight to zero. ASCENT also
  scored `.975`, with zero wins and zero regressions.
- The development rule required measurable small-endpoint headroom before
  larger endpoints. It failed because the official task is already nearly
  saturated at 0.5B, not because ASCENT reduced performance. Running 1.5B/3B
  would make an increasing absolute-accuracy claim progressively less
  identifiable and could implicitly demand accuracy above 1.0. The panel is
  therefore stopped as a prospective headroom exclusion, with no task or
  decoder substitution after the score.
- The raw archive is
  `artifacts/remote_results/qwen_multikey_greedy_1698cd5/qwen_multikey_greedy_1698cd5_node1.tar.gz`,
  SHA-256
  `5f7fba5fc518b2b5d120a27ed507ca293a3157216c28c59bd51b1e02705099c0`.

## 2026-08-14 — official 16K RULER HotpotQA prospective headroom pass

- NVIDIA RULER's official task taxonomy and limitations were inspected before
  any QA model score. HotpotQA (`qa_2`) was selected because it adds the
  missing realistic multi-hop QA category and is structurally less prone to
  the single-step extraction saturation observed on `niah_multikey_1`.
  Foundation-only task-selection code was committed as `9ae7f6e`; no ASCENT
  QA read state or score existed at selection time.
- The exact RULER generator commit
  `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` and `qa.py` SHA-256
  `f00c0b59cf8698e90831c2c4f40f2a30f2469534c27455808c717281d0` used the
  7,405-row HotpotQA distractor validation v1 file, 61,065,698 bytes, SHA-256
  `e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8`.
  An initial generation call omitted the official answer-prefix concatenation;
  pre-score auditing caught malformed prefixes and invalidated all of those
  rows before any model run. The correct NVIDIA pipeline contract regenerated
  a new directory with seed 385803.
- The final 60-row official task SHA-256 is
  `7c46359f17dbddb236ac97a779a51c46bda0074346d70d0b4913cace4e9dfca1`.
  All indices and inputs are unique; every row has exactly one nonempty answer
  and the exact prefix ` Answer:`. Generator lengths were 13,010--15,976
  tokens; Qwen chat prompts were 13,007--15,973, all natively below 32,768
  with zero truncation.
- Commit `052cd2a6ca1adbe86e3139d7c97bc7d11bff10b7` and config SHA-256
  `36d563feb511645793f4664c4f07854354666964b8309a17b11e11f1a408e2da`
  froze Qwen2.5-0.5B, full-context greedy decoding, the official
  `string_match_part` metric, and a two-sided 10%--85% headroom gate before
  the first QA score.
- Foundation scored 19/60, `.3167 [.1980,.4354]`, leaving 41 errors. All
  task/config/model/source/clean-tree/native-context checks passed, so the
  prospective headroom gate passed. This result authorizes designing an ASCENT
  QA architecture and multi-scale gate without having observed any ASCENT QA
  score; it is not itself reported as an ASCENT gain.
- The immutable headroom archive, including model output, final task file, and
  official generator, is
  `artifacts/remote_results/ruler_qa2_headroom_052cd2a/ruler_qa2_headroom_052cd2a_node1.tar.gz`,
  SHA-256
  `5dfa237aa864b2e4efad8a96fdf93764a52065ae738e4c5dbc465f29962d5e0f`.

## Metric-ceiling interpretation rule

- Accuracy is bounded by 1.0. Once scale ASCENT reaches 1.0, requiring its
  absolute gain to keep increasing can demand an impossible accuracy above
  1.0; this is right-censoring by the metric ceiling, not evidence of model
  failure.
- Absolute gain remains reported unchanged. For panels whose ceiling-aware
  estimand was frozen before confirmation, also report remaining-error
  elimination `(A-F)/(1-F)`. A perfect ASCENT cell therefore scores 1.0 on
  error elimination without inventing an impossible 150% accuracy.
- A ceiling-aware result may support a ceiling-qualified scale claim only when
  the metric and gate were frozen before the evaluated scores. Original
  absolute-gain gates and their misses remain visible; post-hoc ceiling
  explanations cannot retroactively turn a failed preregistered gate into a
  pass.

## 2026-08-14 — official HotpotQA ASCENT evidence design freeze

- After the Foundation-only headroom pass and before any ASCENT QA score, the
  query-conditioned document-memory interface was frozen. The reader receives
  only the original official 16K prompt and question, ranks documents with
  deterministic BM25, then expands exact title links. It never receives the
  answer or HotpotQA supporting-fact labels. Fixed ASCENT replays one document;
  Scale ASCENT uses a nested 2/3/4-document law for Qwen2.5-0.5B/1.5B/3B.
- The same-count controls are query-only BM25 without graph expansion and an
  irrelevant bottom-ranked replay. A safe calibration gate can only enable a
  path whose paired answer-NLL gain has a positive 95% lower bound. All paths
  retain the complete unchanged Foundation prompt; evidence is appended near
  the answer and no prompt may be truncated.
- Score-free parsing of all 60 hash-locked rows found 102--122 numbered
  documents per prompt, 60 nonempty questions, and 60 unique complete
  rankings. Graph expansion changes the top-three prefix on 33 rows and the
  top-four prefix on 34 rows, so the graph-vs-query-only control is nontrivial
  before model evaluation. The complete-ranking SHA-256 is
  `11bf9982361a96498a224882da4f6a1406ec75ed7c32ffe58cab98f7bca196c9`.
- The continuous primary is paired teacher-forced answer NLL, which is not
  capped by 100% accuracy. Official RULER accuracy remains a secondary and is
  prospectively reported both as absolute gain and remaining-error elimination
  `(A-F)/(1-F)`. ASCENT accuracy of 1.0 therefore means it eliminated 100% of
  Foundation's remaining errors; the protocol never demands accuracy above
  1.0. A perfect Foundation cell is labeled saturated/undefined for this
  secondary, not an ASCENT failure.
- This panel is development-only. Promotion requires every scale, adjacent,
  fixed-control, query-only-control, irrelevant-control, native-context,
  provenance, and relative-state gate to pass exactly as frozen. Only then may
  three untouched official HotpotQA seeds be generated for confirmation.

## 2026-08-14 — official HotpotQA ASCENT evidence development failure

- Frozen commit `80037a15b6c3f40a3b12d92bab8651bcd4975ded`, config SHA-256
  `9d8e9206a92f6261f3f2e37e0e331b53fb7e67e24b2eb8b9961aaf83d4a11991`,
  and task SHA-256
  `7c46359f17dbddb236ac97a779a51c46bda0074346d70d0b4913cace4e9dfca1`
  were executed on Qwen2.5-0.5B/1.5B/3B. The mid endpoint ran independently
  on the second physical GPU node. All native-context, model/task/config,
  identical-ranking, clean-commit, no-label-reader, nested-prefix, and
  decreasing-relative-state checks passed. Longest prompts were
  16,870/16,950/17,065 tokens under the frozen 32,768-token limit.
- Scale ASCENT's paired test answer-NLL gains over Foundation were
  `-.19146 [-.69703,+.31411]`, `-.07948 [-.60067,+.44171]`, and
  `-.17698 [-1.48157,+1.12762]` nats. All three calibration safety gates
  closed. Official Scale predictions therefore copied Foundation exactly:
  accuracy was `.35/.40/.45`, with zero gains, zero regressions, and zero
  remaining-error elimination. This is not a ceiling event; the path simply
  failed to establish safe utility.
- The low-to-mid point difference was positive (`+.11198`) but uncertain
  (`[-.33894,+.56289]`), while mid-to-large reversed (`-.09750
  [-1.12253,+.92753]`). The log-parameter slope was only `+.01686
  [-.47310,+.50683]`. Thus neither adjacent monotonicity nor a positive slope
  was established.
- Mechanism controls contain limited positive evidence. At 3B, Scale beat
  Fixed by `+1.69543 [.35452,3.03635]` nats. Graph Scale beat query-only BM25
  by point estimates `+.17519` at 1.5B and `+.44247` at 3B, but the lower
  bounds were `-.02601/-.02230`; only the 0.5B irrelevant control passed its
  lower-bound gate. These comparisons cannot rescue a path that loses to
  Foundation.
- The first 0.5B run wrote outputs inside the repository, making its runtime
  clean-tree flag false. Its complete files were retained, outputs were moved
  outside the repository, and the identical frozen command was rerun without
  changing code/config/data/model. Every saved array hash reproduced exactly
  and the repeat recorded `git_dirty=false`; only the clean repeat enters the
  analyzer.
- The frozen development gate failed and does not authorize confirmation or
  result-driven tuning on this panel. The immutable combined archive is
  `artifacts/remote_results/ruler_qa2_hotpot_evidence_dev_80037a1/ruler_qa2_hotpot_evidence_dev_80037a1_node1.tar.gz`,
  SHA-256
  `b362eed88cfc93896d735c4d85022b24b768a98c9cf9999ccc08482991206a4a`.

## 2026-08-14 — proposal-aligned HotpotQA dual-path factorial freeze

- A substantively different interface was designed without tuning the failed
  text-replay panel. It implements the proposal's two complementary paths from
  one charged query-conditioned state: a compact evidence distribution and
  Qwen2 latent replay at the previously registered relative depth `0.875`.
  Their sample-clustered safe simplex includes the exact Foundation corner;
  target tokens within one answer are not treated as independent confidence
  samples.
- The replay budget `38/69/92` was reused from the earlier official Qwen-RULER
  program rather than searched on HotpotQA. Graph Small/Mid/Large contain
  nested `2/3/4` document sketches. Each sketch is independently computed as
  title plus the highest question/title-overlap sentence, so appending a
  document cannot modify an existing state prefix.
- Every 0.5B/1.5B/3B Foundation evaluates all three Small/Mid/Large states,
  forming the proposal-required `3 x 3` factorial panel. This directly tests
  whether a larger Foundation extracts more benefit from the same memory
  expansion. Co-scaled ASCENT is the diagonal; Graph Small is the fixed-state
  control. Co-scale-matched query-only BM25, irrelevant-state, certified-only,
  and latent-only controls are also frozen.
- A new untouched official HotpotQA task was generated at seed `386803` from
  the same pinned NVIDIA generator and source dataset. Its SHA-256 is
  `57e26bdba9f922fa34cc4abafad01fc90771ea1f0dbfae5c95a9aa7c458ac706`.
  Score-free inspection found 60 rows, 96--116 documents per row, exact nested
  sketch bytes, 31 top-three and 32 top-four graph prefixes differing from
  query-only BM25, and complete-ranking SHA-256
  `d98ad33d3acb377eae13af79c6f5173ce11f7398b65fee75bc46c4d3e3565038`.
  No Foundation or ASCENT NLL/accuracy on this seed existed at freeze time.
- Charged co-scaled state bytes are 68,400/212,520/377,568; their
  bytes-per-parameter ratios are
  `.000138452/.000137668/.000122351`, strictly decreasing. The Fixed Graph
  Small ratios also decrease. Promotion is deliberately strict: positive
  co-scaled endpoint/adjacent/slope lower bounds, positive adjacent and global
  Foundation-state interaction lower bounds, fixed/query-only/irrelevant
  control passes, selection of both dual paths somewhere on the diagonal, and
  every provenance/system gate. Only a complete development pass authorizes
  official generation or new confirmation seeds.

## 2026-08-14 — HotpotQA dual-path 3B latent success, factorial failure

- Frozen commit `8566dae34c7ea6f8d4214abbfc27fe2ece65f344`, config SHA-256
  `2adfcb59edb21f17d42c23111dcef6aa6b2a7327bd440f16a8080ff2157ff8ad`,
  and untouched seed-386803 task SHA-256
  `57e26bdba9f922fa34cc4abafad01fc90771ea1f0dbfae5c95a9aa7c458ac706`
  were executed on two physical GPU nodes. All task/model/config/commit,
  clean-tree, no-label reader, native-context, shared-state, exact nested token
  prefix, cross-Foundation token identity, and decreasing fixed/co-scaled
  relative-state gates passed.
- The co-scaled effective NLL gains were `+.12866 [-.15874,+.41607]`, exactly
  zero after the closed 1.5B gate, and `+.25342
  [+.11298,+.39387]` nats. The 3B safe simplex was strongly mechanistic:
  `5%` compact evidence and `95%` Qwen2 latent replay, with calibration gain
  `+.12002` and LCB `+.02193`. This is the first reliable official HotpotQA
  result for the proposal-aligned latent interface, and the large graph state
  beat query-only BM25 by the same `+.25342 [+.11298,+.39387]` interval.
- The full scale claim nevertheless failed. Low-to-mid co-scaled change was
  `-.12866 [-.41607,+.15874]`; mid-to-large passed at `+.25342
  [+.11298,+.39387]`; the log-parameter slope was only `+.05016
  [-.13008,+.23039]`. The 1.5B calibration means were slightly positive but
  every LCB remained below zero, so its gates correctly copied Foundation.
- The `3 x 3` factorial prevents overclaiming the 3B success. At 3B, Graph
  Small had a larger point gain (`+.55322`, wide interval) than Graph Large,
  so co-scaled-minus-Fixed was `-.29979 [-1.00322,+.40363]`. The adjacent
  Foundation-state interactions were `-.05388 [-.24361,+.13586]` and
  `+.25342 [+.11298,+.39387]`; the global small-large interaction was
  `-.47350 [-1.08195,+.13496]`. Thus the positive large endpoint does not yet
  prove that the larger Foundation extracts more value from state expansion.
- The irrelevant-state control also exposed a generic replay-prefix effect:
  3B irrelevant state gained `+.28241 [+.03354,+.53127]`, leaving
  graph-minus-irrelevant at `-.02899 [-.26058,+.20261]`. This motivates a
  content-complete or contrastive read design on untouched data; it cannot be
  repaired or confirmed on seed 386803.
- `development_pass=false`; official autoregressive decoding and confirmation
  are not authorized. The immutable archive is
  `artifacts/remote_results/ruler_qa2_hotpot_dualpath_factorial_8566dae/ruler_qa2_hotpot_dualpath_factorial_8566dae_node1.tar.gz`,
  SHA-256
  `2e5582a06f52e3f259dc98d76d49b0180d17426bee935074ccc0f9cfb67839f7`.

## 2026-08-14 — complete-document dual-path factorial freeze

- Before any model score on a new official seed, the dual-path interface was
  changed from truncated single-sentence sketches to the complete selected
  documents. This addresses the preceding panel's content-identifiability
  failure without changing fusion, depth, ranking, model family, calibration
  size, test size, or factorial/control requirements.
- The proposal's width-linear law with `alpha=1` is instantiated as
  `B_s = round_half_up(512*d_s/896)`, giving maximum budgets
  `512/878/1170` tokens. A score-free audit proved that the selected complete
  2/3/4-document states require at most `447/731/914` tokens, so every state is
  complete and the larger state contains the exact smaller prefix. Even the
  maximum-budget charged bytes per parameter decrease
  `.00186546 -> .00175177 -> .00155599`; observed complete-state ratios are
  smaller and also decreasing.
- A new official HotpotQA seed `387803` was generated from the unchanged
  NVIDIA source. Its 60-row task SHA-256 is
  `0fe214ad0c7d2e85fbad5162aeaac2844e4fbb5331ddb8b485d79289a2f510c1`.
  It contains 93--113 documents per row; graph expansion changes 33 top-three
  and 34 top-four prefixes versus query-only BM25. Complete-ranking SHA-256 is
  `4f92240fc77c675592bbe2d00259dddf81e797da4cb913dac6489a1817641f0d`.
  No Foundation or ASCENT score on this seed existed at freeze time.
- The full `3 x 3` Foundation/state panel and all gates from the previous
  design are retained unchanged. Complete documents are not a post-hoc
  relaxation of the gate: positive endpoint/adjacent/slope, adjacent/global
  interaction, fixed/query-only/irrelevant, dual-path selection, provenance,
  and system gates must all pass before any generation or confirmation.
- The first score-free tokenizer audit stopped execution before any model NLL:
  although complete-document bytes were prefix-nested, whole-string BPE changed
  the last token at an append boundary. The implementation was therefore
  repaired pre-score to tokenize every document independently and append a
  fixed independently tokenized delimiter before concatenating token IDs. A
  final score-free audit and new config commit are required before execution;
  the failed audit produced no Foundation or ASCENT score.
- The repaired final audit passed on all three endpoint tokenizers. Graph-state
  hashes are identical across models; actual Small/Mid/Large ranges are
  `134--449`, `160--734`, and `268--918` tokens, below every frozen budget and
  therefore complete. Mean actual co-scaled bytes-per-parameter decrease
  `.000877594 -> .000719598 -> .000654046`; fixed-small ratios decrease more
  steeply. The longest Foundation/compact prompts are 15,975/1,156 tokens.
  This final audit still read no answer output or support label.

## 2026-08-14 — complete-document HotpotQA interaction success, curve failure

- Frozen commit `59cf4e69e77f12eecd5fb409cbdb556ebe0e1cf1`, config
  SHA-256
  `3799d9917975d069b0646dfb767e6fb42447d7de90241a4f7a126914156c47a6`,
  and untouched seed-387803 task SHA-256
  `0fe214ad0c7d2e85fbad5162aeaac2844e4fbb5331ddb8b485d79289a2f510c1`
  were evaluated on two physical GPU nodes. All task/config/commit,
  clean-tree, no-label-reader, native-context, shared-state, exact nested token
  prefix, cross-Foundation token identity, and decreasing relative-state gates
  passed.
- Co-scaled paired answer-NLL gains were `+.78571
  [+.23719,+1.33422]`, exactly zero after the closed 1.5B safety gate, and
  `+.78185 [+.24727,+1.31644]` nats. The 0.5B gate selected equal Foundation
  and certified evidence; the 3B gate selected `20%` certified evidence and
  `80%` latent replay. At 3B, co-scaled ASCENT beat Fixed Small, query-only,
  and irrelevant controls by the same `+.78185 [+.24727,+1.31644]`
  interval. This removes the generic irrelevant-prefix ambiguity observed in
  the truncated-sketch panel.
- The complete factorial produced the first statistically positive global
  Foundation-by-state interaction on official HotpotQA: Small-to-Large was
  `+.60582 [+.00401,+1.20762]`. State expansion at 3B was also strong:
  Small-to-Mid `+.82657 [+.19363,+1.45951]` and Small-to-Large
  `+.78185 [+.24727,+1.31644]`. At 1.5B, Mid-to-Large was smaller but positive,
  `+.23519 [+.00558,+.46480]`.
- The preregistered complete development gate nevertheless failed. The
  diagonal curve was `+.78571 -> 0 -> +.78185`; its adjacent changes were
  `-.78571 [-1.33422,-.23719]` and `+.78185
  [+.24727,+1.31644]`, and its log-parameter slope was `-.07025
  [-.30945,+.16894]`. Adjacent factorial interactions were `-.14851
  [-.39831,+.10130]` and `-.27990 [-.54183,-.01797]`. Thus the global
  interaction and the large endpoint are positive, but the 1.5B checkpoint
  breaks the required monotone co-scaled curve and adjacent interaction claim.
- `development_pass=false`; no generation or confirmation was authorized.
  The result is not a metric-ceiling failure because answer NLL is uncapped.
  It motivates a prospectively frozen wider same-family scale extension while
  preserving, not replacing, the 1.5B failure. The immutable archive is
  `artifacts/remote_results/ruler_qa2_hotpot_fullstate_factorial_59cf4e6/ruler_qa2_hotpot_fullstate_factorial_59cf4e6_node1.tar.gz`,
  SHA-256
  `2e3b23d87eaf48fa905bd8786b8c121a9f4f2d230c5212d20e86868a547928e3`.

## 2026-08-14 — prospective 0.5B/3B/7B HotpotQA extension design

- The 1.5B failure remains part of the primary record. To distinguish a
  checkpoint-local anomaly from a wider scaling boundary, a supplementary
  same-family 0.5B/3B/7B development was designed before any score on a newly
  generated official HotpotQA seed `388803`. Its 60-row task SHA-256 is
  `5ac6fa1e9e9e055a03ab34f4ca22eed1b090743f8b32e5448e5411ee79298dfc`.
- The complete-document dual path, relative replay depth, calibration/test
  split, safe simplex, 3x3 factorial, and controls are unchanged. Only the
  prospectively declared scale ladder and capacity law change. Co-scaled
  states contain `2/3/4` complete documents with growing `512/774/1024` token
  capacities under
  `B_s = round_half_up(512*(d_s/896)^0.5)`. Registered capacity bytes per
  parameter decrease `.00186546 -> .00102935 -> .00096489` while absolute
  capacity grows.
- This is not confirmation and cannot erase seed 387803. The design is not
  executable until the 7B checkpoint finishes downloading, every shard is
  hashed, and a score-free audit proves complete nested state, cross-tokenizer
  identity, native prompts, and decreasing observed relative cost. Only a new
  clean commit after that audit may authorize the first model score.

## 2026-08-14 — replay code audit and exact 1.5B diagnostic reproduction

- A full local test run exposed a genuine environment-compatibility defect in
  the Llama replay helper: Transformers `4.52.1` predates the masking API used
  on the pinned GPU runtime (`4.57.6`). This did not execute in the Qwen2.5
  HotpotQA path, so it does not invalidate its arrays, but it was fixed rather
  than dismissed. Llama and Qwen2 replay now support both APIs, including the
  old tuple/new tensor decoder-layer return contracts.
- A new Qwen2 invariant test requires zero-prefix replay to reproduce the
  frozen model suffix exactly, plus rejection of width mismatch. The complete
  local suite passes `128/128` on Transformers `4.52.1`; the focused
  Llama/Qwen2 replay suite passes `4/4` independently on Transformers `4.57.6`
  on both GPU nodes.
- Before the 7B score, the dual-path runner was changed to retain only the
  registered Foundation-layer tensor after computing Foundation probabilities,
  rather than retaining every hidden layer and full-sequence logits during all
  replay arms. This is intended to change memory only, not values.
- The exact seed-387803 Qwen2.5-1.5B command was rerun in a fresh process on
  node 2 at current commit
  `76a4ec16d22d7bc2d4631c33f6ef1f378680a36d`. All 16 saved test arrays
  (Foundation; fused/certified/latent for three graph states, query-only, and
  irrelevant) were bitwise identical to frozen commit `59cf4e6`; every fusion
  gate was identical. Peak allocated GPU memory fell from `11,056,310,272` to
  `9,383,235,072` bytes and wall time remained `45.56/45.31` seconds.
- Therefore the observed closed 1.5B Graph Mid gate is a reproducible outcome
  of this frozen state/gate pairing, not a memory-optimization or stochastic
  code regression. It is still not attributed to an intrinsically deficient
  model: Graph Large opens and gains `+.23519 [.00558,.46480]`. The correct
  label remains a scale/state boundary pending the untouched 7B extension.
  Diagnostic archive:
  `artifacts/remote_results/ruler_qa2_fullstate_1p5_diagnostic_76a4ec1/ruler_qa2_fullstate_1p5_diagnostic_76a4ec1_node2.tar.gz`,
  SHA-256
  `87883c2390ada549966f31ef0126f7e28335a022806e630566a9d13a0532d0b6`.

## 2026-08-14 — audited 0.5B/3B/7B HotpotQA extension boundary

- The first score on seed `388803` was executed only after final commit
  `124c00458f402f0e8a939cba1d8278d16354e00f`, config SHA-256
  `4e4a81414e0972ef0ee3c944b2f595d970e778ea1f3ca91d307bc4fd9ed1c72e`,
  and task SHA-256
  `5ac6fa1e9e9e055a03ab34f4ca22eed1b090743f8b32e5448e5411ee79298dfc`
  were identical on both nodes. All four official Qwen2.5-7B-Instruct
  safetensors shards were locked by SHA-256; all `339/339` indexed tensors
  were present exactly once, and their `15,231,233,024` payload bytes exactly
  matched the model index. The no-answer/no-support-label audit was identical
  across endpoint tokenizers and nodes. Graph Small/Mid/Large states were
  complete, exact nested prefixes with ranges `135--449`, `161--733`, and
  `267--917` tokens; mean absolute state grew `240.12 -> 368.75 -> 498.33`
  tokens while state bytes per parameter decreased.
- Co-scaled paired answer-NLL gains were `+.72640
  [+.24985,+1.20295]` at 0.5B, `+.81056 [+.29084,+1.33027]` at 3B, and
  exactly zero after the closed 7B safety gate. Thus the low-to-mid point
  change was positive (`+.08416`) but uncertain (`[-.38826,+.55657]`), while
  mid-to-large was negative (`-.81056 [-1.33027,-.29084]`); the
  log-parameter slope was `-.22009 [-.38753,-.05265]`. The frozen monotone
  scale claim fails.
- The factorial still contains useful positive structure. The 0.5B-to-3B
  adjacent Foundation-state interaction was `+.61549
  [+.04449,+1.18649]`, but the 3B-to-7B interaction was only `+.04165
  [-.11978,+.20308]`, and the global interaction was `-.10724
  [-.38895,+.17448]`. At 3B, Graph Mid beat Fixed Small, query-only, and
  irrelevant by `+.81056 [+.29084,+1.33027]`; the closed 7B gates make all
  effective 7B controls tie Foundation and cannot support mechanism claims.
- The 7B event was audited as a possible implementation failure before any
  model interpretation. Two physical nodes with byte-identical model shards
  independently produced identical SHA-256 values for all 16 saved arrays and
  identical gates. Both GPU environments passed the complete `128/128` test
  suite. On the real 7B checkpoint, empty-prefix replay from registered layer
  `24/28` reproduced both the frozen suffix hidden tensor and final logits
  bitwise, with maximum absolute delta exactly zero. No OOM, NaN, truncation,
  dirty tree, hash mismatch, or node-dependent result was observed.
- The diagnostic raw 7B latent path improved 27/40 test examples and had mean
  gain `+.04219`, but its interval narrowly crossed zero
  `[-.00443,+.08882]`. Complete-document certified-only replay was unstable
  (`-1.25372 [-2.58546,+.07802]`). Its calibration-selected candidate had
  positive mean but high sample-level variance, so the preregistered LCB gate
  correctly returned exact Foundation. The defensible label is therefore a
  reproducible statistical/interface boundary under the frozen mapping, not
  “the 7B model is bad.”
- `development_pass=false`; generation and confirmation remain unauthorized.
  The immutable archive is
  `artifacts/remote_results/ruler_qa2_7bscale_dev_124c004/ruler_qa2_7bscale_dev_124c004_node1.tar.gz`,
  SHA-256
  `3ea1a96bdee99172dd0f7813ff77806e77704aac692a39c3c53ae8934ea521e5`.

## 2026-08-14 — fresh official BABILong generative factorial and focused follow-up freeze

- A complete `3 models x 3 state sizes x 5 untouched panels` factorial ran at
  clean commit `e84a3540e25879ef537cde32bb8f79f7326fcc0b` on 600 official
  BABILong 8K QA1--QA3 rows disjoint from every earlier panel. All 45 result
  cells passed model/config/data hashes, clean-tree provenance, exact parser,
  row alignment, and Foundation-invariance checks. The full local suite then
  passed `160/160`, and an independent Student-t implementation reproduced all
  six registered intervals exactly.
- The co-scaled diagonal generative gains were `.29167 -> .53000 -> .65333`.
  Both adjacent increases passed strongly: `+.23833
  [+.21038,+.26629]` and `+.12333 [+.09029,+.15638]`. The first registered
  model-by-state interaction passed: `+.09500 [+.02691,+.16309]`. The second
  was positive but uncertain: `+.01667 [-.00971,+.04305]`. The first
  co-scale excess was `+.04167 [-.02254,+.10587]`, while the second passed at
  `+.10500 [+.04758,+.16242]`.
- Consequently the prospectively frozen fresh-five-panel primary gate failed
  and remains failed. The supporting eight-panel analysis passed all six
  lower-bound gates, including interaction LCBs `+.02939/+.00595` and
  co-scale-excess LCBs `+.00038/+.08507`. This is strong supporting evidence,
  not a retroactive fresh-primary pass.
- Before any further decoder score, a focused independent confirmation was
  frozen on all ten remaining disjoint 40-row/task hash-ordered panels
  (stable-order starts `580..940`). It contains 1,200 unique official rows,
  zero overlap with all prior local panels, and target-blind parser
  reconstruction `1200/1200`. Only five necessary cells per panel will run:
  135M/K2, 360M/K2/K3, and 1.7B/K2/K3. The four preregistered df=9 contrasts
  are the first co-scale excess, second interaction, upper diagonal increment,
  and upper co-scale excess; all four two-sided 95% LCBs must be positive.
- This 50-cell follow-up independently tests the two marginal estimates and
  cannot erase or relabel the five-panel result. No panel/task/outlier deletion
  is permitted. The frozen design SHA-256 is
  `2a267d018ac998b8d1695755448ac54bc870dbe40b17c5b10c87089f092b08df`;
  the generated config SHA-256 is
  `ab8b7eb4f281599b9eb6bb50a42fc97adba4e23afcb7168bcb3d26cbe9cd8b29`.
- All 50 focused cells then ran from clean commit
  `d448f6b4d24150d10b8fb4227e0e04975fe55fa3`. Every registered provenance,
  model/config/data hash, row alignment, and repeated-state Foundation
  invariance check passed. The four df=9 primary contrasts all passed:
  first co-scale excess `+.05917 [+.03380,+.08453]`, second registered
  interaction `+.04333 [+.02311,+.06356]`, upper diagonal gain increment
  `+.15583 [+.12211,+.18956]`, and upper co-scale excess
  `+.16417 [+.12974,+.19859]`. The focused primary gate is therefore true.
- A separately invoked SciPy Student-t calculation reproduced every mean and
  interval endpoint within maximum absolute difference `1.721e-12`. That tiny
  difference comes from the frozen df=9 critical constant
  `2.2621571627409915` versus SciPy's `2.2621571628540993`; it is retained as
  an explicit numerical audit and cannot affect any LCB sign. The complete
  result archive SHA-256 is
  `a5dfc76978c3d5ec35b7be83b183fd08e69e79009e479fbc34983725c198a144`.

## 2026-08-14 — Qwen2.5 same-task BABILong generative replication freeze

- Before any Qwen BABILong decoder score, a complete
  `3 Qwen2.5 models x 3 state sizes x 10 panels` replication was frozen on
  the exact ten official QA1--QA3 panels used for the independent SmolLM2
  focused confirmation. Reusing the same panels supplies paired
  cross-architecture evidence; panel selection is score-independent for Qwen
  and cannot change either earlier SmolLM2 outcome.
- The Qwen ladder is 0.5B/1.5B/3B with the unchanged exact 1/2/3 fact-slot
  state factorial, target-blind parser, prompts, greedy decoding, and eight-token
  output cap. The design contains all 90 cells and forbids panel, task,
  prediction, or outlier deletion. The 1,200 official rows are unique across
  panels and balanced at 40 rows per task per panel.
- The primary gate requires all provenance, row alignment, model/config/data
  hashes, and repeated-state Foundation invariance checks plus positive
  two-sided df=9 95% lower bounds for both diagonal gain increments, both
  registered model-by-state interactions, and both co-scale-excess contrasts.
  A separately frozen Bonferroni directional familywise audit requires all six
  simultaneous lower bounds to be positive using `t(9)=2.9333240883739897`.
- A pre-score infrastructure audit added strict support for both single-file
  and multi-shard frozen model manifests because Qwen2.5-3B has two weight
  shards. The complete local suite passed `167/167`. The design SHA-256 is
  `05feab75b99d5c68cb32fd4c825c4562605d7e4861557e8dcdcca0736f88edaa`;
  the generated config SHA-256 is
  `2e569aaee0df840ef581ace1e0fa33eb2da508e7d6d62d94677a1478ce8c133f`.

## 2026-08-15 — canonical Qwen confirmations and paper-gate decision

- The unchanged raw Qwen2.5 `3 models x 3 states x 10 panels` factorial
  completed first and retained a genuine upper-step failure. Model, data,
  alignment, parser, and code-path audits passed. A target-blind canonical,
  deictic-free event representation was then developed on the already-visible
  official 8K panels; its development-only gains were `.0500 -> .7675 ->
  .8225`.
- Before any official 16K score, a ten-panel QA2+QA3 canonical confirmation was
  frozen at commit `4fd25e9`. Its gains were `.1050 [.0680,.1420] -> .78875
  [.76161,.81589] -> .82625 [.80968,.84282]`. The two registered directional
  adjacent contrasts passed Holm familywise correction: `+.68375`, adjusted
  `p=3.43e-11`, and `+.03750`, adjusted `p=.02550`. The latter two-sided 95%
  interval was `[-.00020,.07520]`; the one-sided family test was frozen before
  scores and remains the primary test.
- The prospectively frozen 3B raw control reached `.83375` versus `.9875` for
  canonical state, a `+.15375 [.13686,.17064]` canonical advantage. QA2 and
  QA3 differences were both positive, and Foundation predictions matched
  exactly row-by-row. This isolates state representation rather than model or
  dataset drift.
- The official 8K and 16K BABILong releases reuse the same finite semantic
  stories with different distractor instantiations. A separate supportive
  validation was therefore frozen at commit `b7480e6` before any selected-row
  decoder score. It selected 800 public training-generation rows after
  excluding all task, normalized-question, and target-blind event fingerprints
  shared with the full official 8K test pool. Selected semantic overlap was
  zero and parser coverage was `800/800`.
- On this semantic holdout, gains were `.0350 [.00647,.06353] -> .7725
  [.74187,.80313] -> .8075 [.76681,.84819]`. Both adjacent directional
  contrasts passed Holm: `+.7375`, adjusted `p=1.65e-13`, and `+.0350`,
  adjusted `p=.02897`. All 30 panels were retained, the full suite passed
  `193/193`, and every archived result passed its SHA-256 manifest.
- Scope guardrail: the public source split was generated for training
  long-context systems. This project did not train or tune on the selected
  rows, but foundation pretraining contamination is unknown. The result is
  supportive independent-semantic evidence, not an official BABILong test
  score, a leakage-proof benchmark, or a fair Q-RAG test comparison.
- Final decision: the project-defined internal gate of evidence being far above
  ordinary TNNLS experimental packages is open for the scoped long-context
  scale-complementarity claim. This is not an acceptance guarantee and does not
  erase the 0.5B Q-RAG loss, raw Qwen upper-step failure, failed systems wall
  target, or other null results. Formal double-anonymous IEEEtran manuscript
  drafting is authorized.

## 2026-08-15 — prospective Qwen 3B/7B QA7/QA8 headroom factorial freeze

- The around-7B/8B QA2+QA3 extension retained all five public model families
  and all ten panels. Every model had a large positive ASCENT gain after Holm
  correction, but the frozen Qwen 3B-to-7B absolute-gain increment failed:
  Qwen 7B Foundation improved while ASCENT rose from `.9875` to the exact
  metric ceiling of `1.0000`. The negative absolute increment remains part of
  the record and is not relabeled as a pass.
- Before any Qwen QA7/QA8 decoder score, a new headroom development was frozen
  on public causal aggregation rows requiring two to seven events. The rows
  and panels were selected earlier for SmolLM2 without any Qwen score; all are
  retained. The complete neural generative factorial is Qwen2.5 3B/7B by
  State-4/State-8. State-8 is a nested extension of State-4, while relative
  slots per parameter decrease `1.2962020e-9 -> 1.0504731e-9` on the co-scaled
  `(3B,4) -> (7B,8)` diagonal.
- The registered development gate requires positive gain in all four cells,
  positive co-scaled absolute-gain and remaining-error-elimination increments,
  a positive model-by-state interaction, and a positive State-8 minus State-4
  gain at 7B. Failure forbids confirmation and forbids retuning the 4/8 law on
  visible rows. Only a pass authorizes the five fixed confirmation panels,
  whose registered t(4) gate requires positive two-sided 95% lower bounds for
  all four cell gains and all four primary contrasts.
- The readout is deliberately neural generation, not the deterministic
  certified-evidence arm. Target-blind causal projection supplies the
  inventory/count prompt; Qwen must produce the scored public answer. The
  config, analyzer, and focused tests were added before scoring, and the full
  local suite passed `231/231`.

## 2026-08-15 — Qwen QA7/QA8 headroom development falsification and audit

- Both nodes independently passed `231/231` tests, exact commit/config/data/
  runtime/model-artifact checks, and target-blind parser reconstruction on all
  192 frozen rows before the first Qwen QA7/QA8 score. The four development
  cells then ran on two physical nodes. All provenance, row-alignment,
  Foundation-invariance, loaded-parameter, and nested-event checks passed.
- State-4 and State-8 gains were identical within each model. Qwen2.5-3B was
  `.21875 -> .96875` (`+.7500`) and Qwen2.5-7B was
  `.21875 -> .90625` (`+.6875`). The co-scaled increment was `-.0625`, the
  remaining-error increment was `-.08`, and both the state-expansion contrast
  and model-by-state interaction were exactly zero. The frozen development
  gate failed, so the five confirmation panels are permanently unauthorized;
  the 4/8 law will not be tuned on these visible rows.
- The required failure audit found a design-level nonredundancy violation that
  the original preflight did not test. Five rows retained additional State-8
  events, but later events overwrote their effect; the final inventory/count
  projection was identical between State-4 and State-8 on all 32 rows. Thus
  the registered refinement added no semantic information and cannot test the
  proposition's nonredundant-refinement condition. The analyzer now reports
  this condition explicitly rather than attributing the zero contrast to the
  model.
- The three 7B errors were also inspected directly. In every case the frozen
  state contained the correct count or complete inventory, but neural
  generation returned the wrong count or omitted one object. Model identity,
  official shards, parser state, and scoring were correct. This is retained as
  an interface result, not evidence that the public 7B checkpoint is invalid.
- A new score-blind data audit counted eligible public source rows under every
  parameter-sublinear discrete law before any new model score. State-4/State-8
  had only 29 eligible QA7 rows and could not support six balanced panels.
  State-3/State-7 retained decreasing slots per parameter and supplied 77 QA7
  and 101 QA8 rows after all exclusions. The new builder therefore freezes 12
  rows per task in each of six panels, requires the State-3 projected answer to
  differ from State-7 and State-7 to equal the full-event projection, and
  excludes all previously materialized source rows. This is a new design on
  entirely new rows, not a retuning or confirmation of the failed 4/8 panel.

## 2026-08-15 — nonredundant State-3/State-7 Qwen headroom freeze

- Six new public QA7/QA8 panels were materialized from the official bAbI train
  split after excluding all 512 previously materialized source rows. Every
  panel contains 24 rows balanced 12/12 by task; all 144 source rows are
  unique. The target-blind audit proves on every row that the State-3 projected
  answer differs from State-7 and State-7 equals the full-event projection.
  The data manifest SHA-256 is
  `984a966641e8a78dad109361f9acd0fa503ae40b85cc80fe2b75133c9b141043`.
- The complete Qwen2.5 3B/7B by State-3/State-7 neural-generation factorial,
  official model shards, exact panel hashes, runtime, estimator definitions,
  and development/confirmation gates were frozen before any new-panel decoder
  score. The co-scaled state/parameter ratios decrease
  `9.721515e-10 -> 9.191639e-10`. Config SHA-256 is
  `649afb6404b5cb848cedb43fa23931bf61562ee157b410d933fc4c20df92e781`.
- A final score-free tokenizer audit on both nodes found identical prompt
  lengths for the two official Qwen checkpoints. Exactly one of 144 Foundation
  prompts is 8,223 tokens before the frozen 8,180-token input cap, so 43
  leading background tokens will be left-truncated; all other prompts fit.
  The same row and truncation apply to both models and both state conditions,
  and the terminal question/answer boundary plus every ASCENT state prompt fit.
  This disclosure was added before the first new-panel score and produced the
  final config hash above.
- The new development gate again requires all four cell gains, the co-scaled
  absolute and remaining-error increments, the model-by-state interaction,
  and the 7B state-expansion contrast to be positive. It additionally requires
  24/24 semantic state changes at both model endpoints. A failure permanently
  forbids the five confirmation panels and any 3/7 tuning on visible rows.

## 2026-08-15 — State-3/State-7 Qwen development boundary

- The four frozen development cells completed at commit
  `2912dbc56c04bbf12749cf9ebf5b6101108de7d2`. Every result passed config,
  model-shard, data, clean-tree, row-alignment, exact-parser, and repeated-arm
  Foundation checks. Qwen2.5-3B scored `.16667 -> 0` with State-3 and
  `.16667 -> .87500` with State-7; Qwen2.5-7B scored `.29167 -> 0` with
  State-3 and `.29167 -> .58333` with State-7.
- The co-scaled absolute-gain increment (`+.45833`), remaining-error
  increment (`+.61176`), and 7B State-7 expansion (`+.58333`) were positive,
  but the model-by-state interaction was `-.29167` and both State-3 gains were
  negative. The complete registered gate therefore failed and all five
  confirmation panels remain forbidden.
- A cross-node audit reran the 7B/State-7 cell on the second physical node.
  Every generated string, normalized answer, score, prompt-token count, and
  retained-state field reproduced exactly, with the same official model and
  dataset hashes. No OOM, NaN, truncation of the ASCENT prompt, parser drift,
  or node-specific difference was found.
- The failure has a clear task-design interpretation rather than a software
  fault: selection required State-3's projected answer to differ from the
  full target. State-3 therefore supplies a confidently wrong incomplete
  world state on every row, making the all-cell-positive-gain gate logically
  incompatible with the data construction. This panel is retained as a failed
  development and will not be repaired by relabeling or retuning its visible
  rows. Future refinement panels must add conditionally informative noisy
  evidence without making smaller states deterministically false.

## 2026-08-15 — Falcon3 1B/3B/7B factorial development and nonredundancy audit

- A same-family public Falcon3-Instruct ladder was frozen at commit
  `17395ba4fd8c275149a0dc7825562d39ab5defd8` using exact official 1B, 3B,
  and 7B revisions, a complete `3 models x 3 states` development factorial,
  and ten whole official BABILong QA1--QA3 panels selected only by score-free
  native-token safety. The frozen config SHA-256 is
  `837c0383462fe283bf3663a71d8b3dc67de2971617f843f4f25af517e594b9c4`.
  All 1,200 rows fit the 8,184-token input cap (`8178` maximum Foundation
  length), all ASCENT prompts fit, and the target-blind parser was exact.
- Development scores on the first 120-row panel were: 1B Foundation `.15833`
  and ASCENT `.92500/.92500/.92500` for K2/K3/K5; 3B Foundation `.38333`
  and ASCENT `.83333/.85000/.85000`; 7B Foundation `.30833` and ASCENT
  `.99167/.96667/.96667`. The registered first interaction was `+.01667`,
  first diagonal increment and co-scale excess were both `-.30000`; the
  second interaction was `0`, while the second diagonal increment and
  co-scale excess were `+.19167`. The complete development gate failed, so
  none of the nine confirmation panels is authorized.
- The mandatory negative-result audit found that K2-to-K3 changes the retained
  event sequence on exactly 40/120 QA3 rows, but K3-to-K5 changes zero rows at
  every endpoint. QA1--QA3 require at most three supporting facts and the
  parser stores relevant facts only; K5 was therefore byte-for-byte identical
  to K3. The second interaction was structurally forced to zero and cannot be
  interpreted as a model failure. Preflight and analysis were corrected at
  commit `b5c3f4bb6ef67c34da2124c8cb41b82026e4a55c` to require every registered
  transition to change at least one row. Both GPU nodes then passed `237/237`
  tests.
- The valid K2-to-K3 transition itself was model-dependent: it raised 3B QA3
  accuracy from `.80` to `.85`, tied overall at 1B, and reduced 7B QA3 from
  `.975` to `.90`. Direct output inspection found the extra pickup fact could
  act as a distractor (for example, inducing deictic fragments such as
  “the house”). This is retained as an interface boundary; exact cross-node
  reruns are required before any stronger causal attribution.

## 2026-08-15 — post-training noisy-composition free-generation boundary

- A proposition-matched synthetic mechanism panel was frozen at commit
  `c0e2114101ae2e8bf6f60577622331a15d0ef5ab` before generation. Two fresh
  post-training values in `0..7` were observed through independent categorical
  noise and composed as `(2*A+B) mod 8`. The 2/3/5 observation prefixes were
  exact nested refinements on all rows; the co-scaled state/parameter ratios
  decreased across official Falcon3 1B/3B/7B endpoints. Exact posterior NLL on
  development decreased `1.04786 -> .67653 -> .24878`, so both transitions
  contained genuine conditional target information.
- All nine neural generation cells passed config, data, model, parameter,
  prompt-length, alignment, nesting, and clean-tree checks. The registered
  accuracy gate nevertheless failed. The co-scaled gains were `-.14167`,
  `-.14167`, and `+.01667`; only the upper diagonal increment (`+.15833`) and
  7B K3-to-K5 increment (`+.00833`) were positive. Confirmation is forbidden.
- Mandatory output inspection identified a metric/interface problem rather
  than a silent execution error. Every 1B K5 and 3B K5 generation exhausted
  the frozen eight-token budget with the identical unfinished prefix “To solve
  this problem, we need to”; their ASCENT answer parser therefore returned
  `None` on all 120 rows. The 7B endpoint usually emitted a digit immediately,
  but was strongly anchored to the worked example's answer `4`. These facts
  explain the failure without blaming model identity or fabricating a pass.

## 2026-08-15 — proper-scoring noisy-composition candidate-head boundaries

- A fresh 5,120-row candidate-normalized NLL design removed the worked example
  and eliminated output-length censoring. Its first score-free preflight
  correctly failed because Falcon tokenized each originally registered
  leading-space digit into two tokens. No score had been produced. Bare digits
  were then verified as eight distinct single tokens at every endpoint,
  re-frozen at commit `6a0d0c3f563c1c83b943b5fb06dac8dce2671cac`, and all `243/243`
  tests passed on both nodes.
- Falcon3 candidate-NLL development had clean provenance and exact nonredundant
  transitions on every row. Co-scaled gains were `-1.02922 -> +.26607 ->
  +.15146` nats, so the first diagonal increment was strongly positive
  (`+1.29529`) but the second was negative (`-.11461`). Importantly, 7B gain
  increased from K3 to K5 by `+.12029`, and both model-by-state interactions
  were positive (`+.16977/+.56831`). The complete registered gate failed and
  confirmation remains forbidden; the positive interactions are retained only
  as development evidence.
- A separately frozen Qwen2.5 0.5B/3B/7B candidate-head development used
  entirely fresh rows at commit
  `d4ae0dcdfe12c2e0dee85ee9fbb0dc7a7dd2f6b2`. Its co-scaled gains were
  `-.27307 -> -.81119 -> +1.52595` nats. The upper diagonal increment was
  `+2.33715`, the 7B K3-to-K5 increment was `+.05514`, and both interactions
  were positive (`+.23827/+.49688`), but negative small/mid gains and the
  negative first diagonal increment failed the gate. Candidate-next-token
  scoring exposes strong large-model sensitivity to ASCENT evidence but does
  not yield the required complete curve; its confirmation is forbidden.
- A third, substantively different Qwen readout was frozen at commit
  `b3fa398fb00fbcf2fbd775dd6a407acd27b6225f` on another 2,560 fresh rows.
  It permits 96 deterministic tokens of brief reasoning, removes all worked
  examples, requires a `FINAL:` marker, and refuses promotion below 95% marker
  coverage. All three endpoints passed score-free model, data, prompt, channel,
  and 2/3/5 nesting audits before its development generation began.

## 2026-08-15 — compact certified-MAP sum confirmation

- A direct dual-path implementation was frozen at commit
  `443265eb6c515a94feceaca94a95075cb7ec8d5b`. The certified path retains exact
  append-only 2/3/5 noisy observations and computes the registered posterior
  MAP values; the neural path receives only the two compact decoded integers
  and performs ordinary addition. Across Qwen2.5 1.5B/3B/7B, the external-state
  to parameter ratio strictly decreases.
- Development passed with co-scaled gains `.41016 -> .62109 -> .83203`. All 81
  untouched confirmation cells then ran as nine panels by three endpoints by
  three state sizes. Mean co-scaled accuracy gains were `.39800 -> .62066 ->
  .83811`, with adjacent increments `+.22266 [.19142,.25390]` and `+.21745
  [.18877,.24613]`. The 7B K5-minus-K3 gain was `+.18967
  [.16700,.21234]`.
- All six registered two-sided t(8) lower bounds passed. A stricter audit fixed
  before reading confirmation scores also passed all six Bonferroni simultaneous
  lower bounds; their minimum was `+.15546`. All model/data/config/commit checks,
  6,912 nonredundant transitions, nested prefixes, exact-information curves,
  row alignment, answer coverage, Foundation invariance, and decreasing-relative-
  state checks passed.
- Both model-by-state interactions were exactly zero. This experiment therefore
  supports the certified co-scaled absolute-gain claim but is not used to claim
  that a larger neural foundation extracts more benefit from the same state
  increment. A 7B K5 audit rerun on the second physical node reproduced all 256
  predictions, prompts, decoded MAP values, metrics, and scientific metadata
  exactly after independent weight validation.

## 2026-08-15 — ordinary-sum full-posterior 7B completion and boundary

- The previously incomplete frozen ordinary-sum posterior-replay matrix was
  completed at commit `bb17a1ba05a6b2893ae06ac4c63e6f9bfeabb5b2` after the
  second node independently passed 247 tests, exact four-shard validation,
  7,615,616,512 loaded parameters, and all three score-free 7B preflights.
- The complete development matrix produced co-scaled gains `.11719 -> .14844
  -> .80859`, adjacent increments `+.03125/+.66016`, positive interactions
  `+.09375/+.15625`, and a strictly increasing 7B K2/K3/K5 curve `.45703 ->
  .61719 -> .80859`. These are positive development diagnostics only.
- The registered gate failed because the full-posterior free-generation readout
  had sub-99% answer coverage: the 3B ASCENT cells parsed only `.75391/.69531/
  .73047`, and 1.5B K5 parsed `.97656`. No confirmation is authorized. In
  contrast, every 7B arm had 100% coverage, and all three 7B cells reproduced
  every scientific field and all 256 predictions exactly on the other node.
- Before any new score, a fresh 10-panel candidate-normalized successor was
  frozen. It retains the full posterior interface but replaces parse-sensitive
  generation with the equal-length single-token answer code `A=0,...,O=14`,
  verified identical at all three Qwen2.5 tokenizers. It requires both positive
  model-by-state interactions in addition to the six co-scale gain estimands.

## 2026-08-15 — arbitrary letter-candidate audit and numeric-sequence freeze

- The fresh letter-code development ran from clean commit
  `b18134712948c9bc050a2e4cb82697b3fdb39256` only after 251 tests and all
  nine score-free preflights passed on both nodes. It failed decisively: the
  co-scaled candidate-NLL gains were `-4.92784 -> +1.88865 -> -5.14383`, the
  upper diagonal increment was `-7.03248`, and the 7B K5-minus-K3 increment
  was `-.66943`. Confirmation is forbidden.
- Mandatory logit inspection localized an interface confound. On an audited 7B
  K5 row, the actual next-token distribution assigned about `.48685/.48685`
  to “Based” and “Given”. Among the bare `A,...,O` candidates, `A` alone had
  `.00477` probability while most other letters were around `1e-9` or below.
  Candidate normalization therefore measured arbitrary token-initial frequency,
  not the registered numeric composition. This is not attributed to the model.
- Before another model score, the scorer was generalized to exact joint
  likelihood for equal-length two-token candidate sequences. A CPU toy model
  test independently verifies that its output equals the normalized sum of the
  first-token and conditional second-token log probabilities. All 15 numeric
  candidates `00,...,14` are distinct length-two sequences with identical IDs
  at every Qwen2.5 endpoint. The scorer additionally requires exact boundary
  tokenization after the fixed assistant prefix “The ordinary integer sum as
  two digits is ”.
- Ten further panels with seed rule `2026085000 + panel_index` were frozen with
  zero panel-hash overlap against every prior noisy-composition config. The new
  eight-estimand gate again requires both interactions, and its nine-panel
  confirmation remains inaccessible until the single development panel passes.

## 2026-08-15 — numeric-sequence sum composition confirmation

- The two-token numeric scorer and all fresh data were frozen at commit
  `f133a9689adb870680d462049c0dd4e0bfc41e9a`. Both nodes passed 253 tests.
  Every one of nine score-free preflights verified the exact candidate sequences
  `00,...,14`, their common two-token length, distinctness, complete scoring-
  boundary stability, model/data/config hashes, and maximum prompt length.
- The one-panel development gate passed all eight estimands: co-scaled NLL gains
  `.44126 -> 2.36864 -> 7.51195`, diagonal increments `+1.92737/+5.14332`,
  7B K5-minus-K3 `+1.57297`, and model-by-state interactions
  `+1.04480/+1.24804`. Confirmation was therefore authorized without changing
  the prompt, candidates, state law, metric, or gate.
- All 81 untouched confirmation cells completed. Mean co-scaled NLL gains were
  `.38817 -> 2.09929 -> 7.40613`. The two adjacent increments were `+1.71112
  [1.60373,1.81850]` and `+5.30684 [5.14770,5.46599]`; 7B K5-minus-K3 was
  `+1.45571 [1.20974,1.70168]`. Crucially, both registered model-by-state
  interactions passed: `+.78114 [.62936,.93291]` and `+.96782
  [.78180,1.15384]`.
- All eight ordinary two-sided t(8) lower bounds passed. The supplementary
  eight-comparison Bonferroni simultaneous family audit also passed every
  estimand; its smallest lower bound was `+.31110`, and the interaction lower
  bounds were `+.53915/+.67124`. All 13,824 endpoint--transition row checks,
  exact-information curves, nesting, row alignment, Foundation-probability
  identity, relative-state, coverage, clean-commit, and artifact checks passed.
- A 7B K5 audit on the second physical node reproduced all 256 fifteen-way
  probability vectors, joint sequence NLLs, answers, gains, preflight fields,
  and scientific metadata exactly. This is a teacher-forced synthetic mechanism
  test with a fixed assistant prefix; it supports the proposition's neural
  composition/interaction mechanism and complements, rather than replaces,
  the official RULER and BABILong generative evidence.

## 2026-08-15 — affine posterior replay provenance closure

- The affine posterior 7B cells were rerun from the exact detached freeze
  `73cf38bcc134472ef418140268bd7b68cd4c4ce4`, replacing an earlier diagnostic
  whose code checkout had advanced before execution. Combined with the original
  same-commit node-2 cells, all nine provenance checks now pass.
- The correctly sourced development still fails: co-scaled gains are
  `-.01563 -> +.05078 -> +.07031`; confirmation remains forbidden. The unique
  negative 1.5B/K2 cell then reproduced every prediction and metric exactly on
  the other physical node. The boundary is therefore not explained by source-
  commit drift, weight mismatch, nondeterminism, or row misalignment.

## 2026-08-23 — complete third-node Qwen3-8B deterministic audit

- A fresh NVIDIA RTX PRO 6000 Blackwell Server Edition node validated the
  pinned Qwen3-8B checkpoint: all five shard SHA-256 values, the tokenizer and
  model configuration, Transformers 4.57.6, Tokenizers 0.22.2, SentencePiece
  0.2.2, Safetensors 0.8.0, and PyTorch 2.8.0+cu128 matched the frozen record.
- The node reran all ten official 16K BABILong QA2/QA3 panels with the archived
  four-slot ASCENT condition. The audit reproduced all 800 paired prediction
  rows and every scientific field exactly. The aggregate remained .21625
  Foundation accuracy, .93625 ASCENT accuracy, and .72000 absolute gain, with
  577 improvements and one regression.
- Runtime, hostname, repository-state, and the explicit
  `diagnostic_no_promotion` audit tag are intentionally excluded from
  scientific-field identity. The rerun is posthoc and therefore supports only
  deterministic reproducibility; it is not added to the prespecified
  inferential family or used as independent promotion evidence.

## 2026-09-20 — Array revision 1: Phase 0 resolutions

- Four open factual questions about the retained cross-node evidence were
  resolved from records only (`reports/PHASE0_OPEN_QUESTIONS_RESOLVED.md`).
  The third-instance Qwen3-8B status label is the runner's generic
  `--audit-rerun` constant introduced in commit `693c414` for the 2026-08-14
  state-accounting fix, which predates every formal 7--8B run; all 800
  third-instance rows and every scientific field are identical to the formal
  records (`reports/THIRD_NODE_AUDIT_ROW_DIFF_2026-09-20.json`). The
  third-instance `git_dirty=true` reflects concurrent manuscript-packaging
  edits on a one-commit snapshot repository; the diff is retained in the
  frozen backup and no execution-path file differs from commit `65e1491`.
  GPU identity for all three instances was recovered from the frozen
  archives: three distinct serial numbers and UUIDs, and distinct host driver
  versions for instances 1 and 2 captured at the same second
  (`reports/NODE_IDENTITY_EVIDENCE_2026-09-20.json`). The three files under
  `artifacts/around7b_formal/diagnostics/` are decode-order checks that
  reproduce their formal counterparts exactly and enter no reported number.

## 2026-09-20 — Array revision 1: cross-architecture reproduction audit (Phase 2A)

- Five preregistered cells (`artifacts_revision/crossnode_2026-09/PREREGISTRATION.json`,
  committed as `9c15400` before execution) were rerun on a Vast.ai NVIDIA
  A100-SXM4-80GB (UUID `GPU-5b892f6f-ac8e-8c99-71ed-7efad31c6765`) with the
  pinned stack (Python 3.12.3, torch 2.8.0+cu128, transformers 4.57.6,
  tokenizers 0.22.2, safetensors 0.8.0, sentencepiece 0.2.2, numpy 2.3.2,
  cuDNN 9.10.2) from a clean worktree, with environment snapshots before and
  after. Reproduction holds at the level of scientific conclusions but not
  row-for-row or bit-for-bit: 1--2 argmax candidates per arm flipped on
  factorial panels 2 and 3 (gain NLL moved by 0.008 and 0.011 nats), one
  scored output per arm changed on official-16K panel 1 for each reader
  (those three panels each moved by exactly 1.25 points; the complete
  sixty-cell figures are in the Phase 2B entry below), and 8,440 of 8,448
  stored floats per factorial cell differ in their low-order bits
  (`CROSSARCH_REPRODUCTION_REPORT.md`, `CROSSARCH_COMPARISON.json`).
- The first execution used Python 3.12.14 by mistake; it is retained in full
  (`artifacts_revision/crossnode_2026-09_run1_python3.12.14/`) and is
  bit-identical to the preregistered 3.12.3 execution at every level
  (`SAME_GPU_RUN1_VS_RUN2_CONTROL.json`), so the A100 is run-to-run
  deterministic and every difference from the Blackwell references is
  attributable to the architecture change. The comparator was amended after
  run 1 to report the runner output field `model_identity` (added in commit
  `0d60263`, after the reference runs) as skipped rather than mismatched;
  both deviations are recorded in `DEVIATIONS.md`. No hardware was
  substituted and no cell was rerun with different settings.

## 2026-09-20 — Array revision 1: per-row target-blindness instrumentation (Phase 2B)

- `ascent/target_blindness.py` adds a blinded row view (reading the answer
  raises until `reveal()`) and a three-link state-provenance record; the
  BABILong runner routes its write/readout/prompt path through the view and
  records a `target_blindness` block on every row. The writer module is
  unchanged. All 60 official-16K and semantic-holdout cells (4,800 rows, both
  arms) were rerun on the A100 from commit `ec95bb1`: every row records zero
  blocked target accesses, no read of the answer before the reveal, only
  `input`/`question`/`task` read before it, and passing provenance. Six
  same-instance instrumented-vs-uninstrumented controls are bit-identical
  (`artifacts_revision/target_blindness_2026-09/TARGET_BLINDNESS_AUDIT.json`).
- The 60 A100 cells also extend the cross-architecture comparison: 124 of
  9,600 scored outputs changed (103 Foundation, 21 ASCENT); 20 cells were
  unchanged; single panels moved by up to 5.00 points (four rows); ten-panel
  means moved by at most 1.00 point (official 16K 10.50/78.88/82.63 to
  9.75/78.88/83.25; semantic holdout 3.50/77.25/80.75 to 4.50/77.00/80.62);
  every registered directional contrast still passes on the A100 (second
  official increment 4.38 points, `p_Holm=0.0028`; semantic 3.62,
  `p_Holm=0.022`). The manuscript reports the retained Blackwell values and
  presents the A100 results as a sensitivity analysis.

## 2026-09-21 — Array revision 1: schema-blind writer ablation (Phase 2C)

- Preregistered before any score (`artifacts_revision/schema_blind_2026-09/PREREGISTRATION.json`,
  commit `b13062d`): the identical official-16K pipeline (ten panels, slot
  budget 1/2/3, serialisation, prompt, readers, greedy decoding, scorer) with
  the structured-fact writer replaced by three schema-free state
  constructions — a sentence-recency window (new `sentence_window` memory
  source), the existing BM25-style lexical chain, and BGE-M3 hybrid retrieval
  with the official reranker (protocol re-pinned to the 16K panels; caches
  built on the A100 with shard-verified BAAI/bge-m3 and bge-reranker-v2-m3).
  Ninety runner jobs (3 writers × 3 readers × 10 panels, both arms) executed
  from clean worktree `c84f587` on the A100 with environment snapshots
  before and after; 225 files hash-verified on transfer.
- Outcome under the fixed interpretation rule: all three writers collapse.
  Ten-panel gains (points, 0.5B/1.5B/3B): sentence window
  −5.25/−17.50/−16.00; lexical chain −5.00/−16.38/−11.62; BGE-M3
  −5.50/−11.00/−5.38 — negative at every scale (the generic state is worse
  than no state at a one-to-three-slot budget) and no adjacent-increment
  family passes with increasing gains. The structured-fact writer's advantage
  in ASCENT accuracy, paired over panels on the same instance, is 15 points at
  0.5B and 89–99 points at 1.5B/3B. Foundation outputs are identical across
  all three conditions and the same-instance canonical run.
- Consequence: Option B. Section 7 scopes the claim to external state
  constructed against the evaluated structured-fact schema; Appendix D
  reports the ablation (`GENERIC_WRITER_ABLATION.json`;
  `paper/generated/generic_writer_ablation*.tex` are generated from the record
  and compared by `make reproduce`). Phase 2D (non-templated benchmark) was
  not attempted and is not mentioned in the manuscript.

## 2026-09-20 — Array revision 1: rounding convention made uniform

- The manuscript rounds every reported statistic half up on its exact decimal
  value (`render_paper_additional_tables.decimal`). `reproduce_all_tables.py`,
  `render_appendix_tables.py`, `plot_manuscript_evidence.py`,
  `analyze_panel_power.py` and the cross-architecture check in
  `audit_manuscript_consistency.py` used Python `f"{x:.Nf}"`, which rounds the
  binary double (half to even when the tie is exactly representable). All five
  now use the half-up rule. Values on a decimal tie whose printed form changed:
  Granite-3.3-8B gain 137/160 = 0.85625, printed 85.62 by `make reproduce`
  and in `VERIFICATION.md` §2, now 85.63 as in the manuscript; A100
  semantic-holdout 3B ten-panel mean 645/8 = 80.625 and its second increment
  3.625, previously 80.62 and 3.62 in §8, `VERIFICATION.md` §5 and the Phase 2A
  report, now 80.63 and 3.63 (the ledger entry above keeps the earlier
  digits); SmolLM2 certified-factorial appendix cells .44375/.69375/.94375/
  .40625/.65625/.90625, previously .4437/.6937/.9437/.4062/.6562/.9062, now
  .4438/.6938/.9438/.4063/.6563/.9063; Fig. 2 labels 77.25 and 26.25,
  previously 77.2 and 26.2, now 77.3 and 26.3. No underlying value changed.
  Appendix D (`analyze_generic_writer_ablation.py`) was left as generated;
  its nine tie values are listed in `VERIFICATION.md` §2.
