# ASCENT preregistered experiment protocol

Status: working preregistration, endpoint results not yet observed.

## Active baseline scope update — 2026-08-13

The user explicitly removed FPVM from the active experiment matrix and does
not want it used as a baseline. All new ASCENT experiments therefore compare
against the frozen foundation, matched-byte raw/clean-KV storage, same-signal
exact decoding, `ASCENT-Fixed`, `ASCENT-Scale`, and ASCENT path/mechanism
ablations. FPVM is neither run nor imported into future result tables.

Earlier preregistered configurations and completed artifacts still contain
FPVM as an unimplemented blocker because that was the protocol in force when
their hashes were frozen. They remain immutable historical records and are not
edited retroactively. This update supersedes that blocker for future runs.

## Claim separation

1. **Certified claim:** on a post-training fresh-information distribution, the
   exact ASCENT gain increases strictly with append-only refinement.
2. **Foundation interaction claim:** in a two-factor experiment, the co-scaled
   foundation/state improvement exceeds memory-only growth with a positive
   interaction interval.
3. **External-validity claim:** utility transfers to natural text and established
   long-context tasks without a stable negative model-family effect.

No natural-language theorem is claimed. A positive synthetic result alone does
not promote the method to a paper.

## Locked Stage-A protocol

- Uniform 16-way fresh answer (`log(16)` exact no-memory NLL).
- Four fixed code bits and independent BSC observations with crossover 0.18.
- Registered scale prefixes contain 1, 2, 3, and 4 observation rounds.
- The decoder is the exact posterior under the registered channel.
- Exact enumeration is primary; 200,000 Monte Carlo episodes are an
  implementation cross-check.
- Strictness requires every exact conditional-information increment to exceed
  `1e-9` nats and the gain-difference identity to agree within `1e-10` nats.
- Causal tests cover read-before-write, future-prefix invariance, overwrite
  versioning, and projection of a large signal to a small signal.

## Promotion ladder

1. Stage A unit tests and exact identity.
2. Small/large endpoint screen with foundation, matched raw/clean-KV cache,
   ASCENT-Fixed, and ASCENT-Scale.
3. Three-scale factorial curve with at least three projection seeds and three
   data seeds.
4. Required ablations, corrupt-memory controls, same-byte and same-FLOP controls.
5. WikiText-103, PG19, PALOMA clean/no-overlap, held-out C4/Dolma, RULER, and
   BABILong; Transformer and recurrent/linear model-family replication.
6. Latency, throughput, peak memory, exact external-state bytes, and generation
   safety.

Endpoint configurations and test splits are frozen before endpoint evaluation.
Development failures are retained in the ledger.

The matched-raw arm is fail-closed: a deliberately weak heuristic decoder is
not acceptable. See `docs/DESIGN_RISKS.md` for the exact same-signal equality
control and the byte-matched capacity-pressure protocol.

## Submission-quality evidence gate

The work can be described as competitive for a TNNLS Full Paper only after:

- all headline values are regenerated from immutable raw results;
- confidence intervals are paired at the episode/document level;
- multiplicity control is stated for families of confirmatory hypotheses;
- the three-scale interaction slope has a positive 95% lower bound;
- improvements are not explained solely by extra state bytes or decoder FLOPs;
- external validity is positive across most preregistered cells and not stably
  negative in either backbone family;
- causal-leakage, checkpoint, tokenizer, split, seed, and byte-accounting audits
  pass independently;
- limitations and all failed preregistered branches are reported.

Acceptance cannot be guaranteed; this gate means that the evidence is strong
enough to justify submission and adversarial peer review.

## Current TNNLS format facts (official page checked 2026-08-13)

- Full Papers must be non-incremental, well-founded, conclusive, and archival.
- Initial manuscript: PDF, single-spaced, IEEE double-column format, numbered
  pages, informative abstract, and 4–5 index terms.
- Regular-paper over-length charges start after 10 printed pages; maximum main
  manuscript length is 15 pages excluding supplementary material.
- Double-anonymous review and ORCID for all authors are required.
- AI-generated text and the sections using it must be disclosed and cited in
  the acknowledgements.

Official source: https://cis.ieee.org/publications/t-neural-networks-and-learning-systems/tnnls
