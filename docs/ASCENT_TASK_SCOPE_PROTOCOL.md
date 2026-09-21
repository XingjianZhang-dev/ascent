# ASCENT task-scope and benchmark-selection protocol

**Frozen date:** 2026-08-14  
**Status:** prospective protocol before any ASCENT score on BABILong QA7/QA8

## Claim scope

ASCENT is not claimed to improve every pretrained model on every distribution.
The paper-level empirical claim is conditional: when the target depends on
causal test-time events that can be represented as a nested refinement state,
co-scaling the frozen Foundation and ASCENT should produce positive and,
preferably, increasing utility while external-state bytes per Foundation
parameter decrease.

Results outside this information-bearing regime are boundaries, not universal
falsifiers. They remain reported, but they do not silently redefine the scoped
claim into distribution-free dominance, which Proposition 0 proves impossible.

## Prospective inclusion criteria

A benchmark family is eligible for the primary empirical scope only when all
of the following can be established without reading ASCENT model scores:

1. **Causal test-time information.** The query depends on events, facts, or
   assignments available in the evaluated episode. For the certified theorem
   regime these are fresh after pretraining; established natural benchmarks
   may use ordinary language, but the evaluated state must still contain
   query-relevant episode information.
2. **Target-blind state construction.** Writing, parsing, ranking, and retention
   cannot inspect the reference answer. Reference fields may be read only by a
   separate audit/scoring path.
3. **Nested refinements.** Every larger state preserves the smaller signal
   exactly and appends additional causal evidence, events, or computation.
4. **Decreasing relative cost.** Absolute ASCENT capacity may grow with model
   scale, but charged persistent bytes per Foundation parameter must decrease.
5. **Usable headroom.** The primary metric must not already be saturated at the
   smallest Foundation. If saturation is structurally possible, a
   remaining-error-elimination estimand must be frozen before the relevant
   scores and the original absolute metric must still be retained.
6. **Mechanism match.** The task must exercise at least one declared ASCENT
   operation: repeated retrieval, multi-event aggregation, causal overwrite,
   or composition/relational reasoning over stored events.
7. **External legitimacy.** At least one established public benchmark must
   support each natural-task claim. Synthetic tasks are used for theorem and
   controlled-mechanism validation, not as the sole empirical evidence.

## Selected primary families

Before any new QA7/QA8 score, the eligible primary families are frozen as:

- official NVIDIA RULER multiquery retrieval;
- official NVIDIA RULER FWE/CWE aggregation;
- official RULER Variable Tracking for teacher-forced information gain, with
  its failed greedy-generation bridge retained as a decoding boundary;
- official BABILong event-state retrieval and composition;
- controlled fresh-information composition for direct theorem/mechanism tests.

Official BABILong QA7 and QA8 are selected prospectively for the next
development because they extend the already confirmed QA1--QA3 event-memory
family to counting and set/list aggregation over a variable number of relevant
events. The official task specification reports 1--10 supporting facts for QA7
and 1--8 for QA8, so nested larger states can append nonredundant causal
evidence rather than merely repeat a one-fact solution. They were selected from
task semantics before an ASCENT score screen, and no QA7/QA8 ASCENT model score
existed at this revision.

QA4 and QA5 were named in the first frozen revision of this document, then
rejected before any model score after a target-free semantic audit showed that
the released examples normally reduce to one query-relevant relation or
transfer fact. That makes them legitimate BABILong tasks but a poor test of the
specific scale-expanding-state hypothesis. Their pre-score rejection is kept in
the Git history; they were not replaced because of an unfavorable model result.

## Boundary families retained outside the primary scope

- Open-domain multi-hop QA is not excluded from reporting. The frozen
  HotpotQA panels show that utility can fail when the charged state omits a
  support document; the audited 7B failure is an interface/retrieval-coverage
  boundary.
- Exact raw key-value recall remains a boundary where clean exact storage can
  dominate neural memory.
- Natural repeated-context language modeling has positive utility but a
  contracting absolute scale curve and is not used for the strict scaling
  claim.
- Overwrite experiments with failed role-binding controls remain limitations
  unless a substantively new, prospectively frozen binding operator succeeds.

These failures cannot be removed from the ledger. Conversely, they are not
used to imply a stronger universal claim than the architecture or theorem
makes.

## Development and confirmation rule

For each newly selected public task family:

1. freeze parser/state semantics, model artifacts, data hashes, scale law,
   metrics, controls, and gates before development scores;
2. retain a failed development verbatim and do not turn that seed into
   confirmation;
3. a passed development may authorize only new untouched confirmation panels;
4. primary promotion requires three untouched panels or seeds, positive
   same-scale gains, an increasing scale estimand with a positive
   seed-clustered lower bound, decreasing relative state cost, and the frozen
   task-specific controls;
5. one task-family failure does not fail the scoped paper claim, but the final
   manuscript must report the selection protocol, the number of attempted
   families, and every stable negative family.

## Paper-level sufficiency rule

The empirical package is considered comfortably beyond a single-task result
only if the scoped scale effect is confirmed across multiple public task
families, at least two Transformer architecture families, multiple untouched
seeds, strong fixed/state/content/compute baselines, and immutable
reproduction artifacts. Formal manuscript writing remains blocked until the
separate TNNLS readiness audit judges this whole package, not any one favorable
cell, to be compelling and conclusive.
