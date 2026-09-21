# ASCENT Around-7B/8B Experiment Completion Audit

**Audit date:** 2026-08-15  
**Decision:** The frozen around-7B/8B model roster and every subsequently
authorized 7B development/confirmation path are complete. No confirmation is
missing behind a passed development gate. Failed development interfaces remain
closed and are preserved rather than silently retried on their untouched panels.

## Public-model 16K BABILong roster

The prospective ten-panel QA2+QA3 evaluation is complete for all five official
public checkpoints. Each endpoint has exactly ten immutable panel results.

| Endpoint | Foundation | ASCENT | Gain | Gain 95% LCB |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | .3325 | 1.0000 | .6675 | .6291 |
| Mistral-7B-Instruct-v0.3 | .1600 | .9675 | .8075 | .7732 |
| Falcon3-7B-Instruct | .1900 | .9988 | .8088 | .7735 |
| Granite-3.3-8B-Instruct | .0850 | .9413 | .8563 | .8312 |
| Qwen3-8B nonthinking | .2163 | .9363 | .7200 | .6768 |

All five gains and lower bounds are positive. Qwen2.5 additionally has frozen
fixed-state and raw-representation controls, and its 7B result has an exact
physical-node rerun.

## Qwen2.5 7B mechanism/readout paths

| Frozen path | 7B execution | Decision |
|---|---|---|
| Compact certified MAP ordinary sum | Development plus 9-panel 3x3 confirmation complete; K5 gain `.8381`, K5-minus-K3 `+.1897` | Passed the six-estimand and Bonferroni family gates; interaction is exactly zero and is not claimed |
| Full-posterior two-token numeric sum | Development plus 9-panel 3x3 confirmation complete; K5 gain `7.4061` nats, K5-minus-K3 `+1.4557` | Passed all eight gain/interaction intervals and all eight Bonferroni simultaneous lower bounds |
| Full-posterior free-generation sum | All 3x3 development cells complete; 7B K2/K3/K5 gains `.4570/.6172/.8086` with 100% 7B coverage | Overall gate failed smaller-endpoint parse coverage; confirmation correctly forbidden |
| Full-posterior affine replay | All 3x3 development cells rerun at the exact frozen commit | Gate failed the negative 1.5B/K2 gain; confirmation forbidden |
| Qwen candidate-NLL affine readout | All 3x3 development cells complete | Gate failed negative small/middle cells; confirmation forbidden |
| Qwen 96-token CoT readout | All 3x3 development cells complete | `FINAL:` coverage gate failed; confirmation forbidden |
| QA7/QA8 headroom and nonredundancy screens | All registered 3B/7B state cells complete | Retained as structural/readout boundaries; no unauthorized confirmation |

The numeric-sum 7B K5 probability vectors and all scientific fields reproduce
exactly across the two physical nodes. The free-generation sum 7B K2/K3/K5
cells also reproduce exactly. The affine path's unique negative 1.5B/K2 cell
reproduces exactly on the other node, so the retained failure is not explained
by a node, weight, commit, or nondeterminism mismatch.

## Falcon3 7B same-family paths

The Falcon3 1B/3B/7B free-generation and candidate-NLL 3x3 developments are
complete. Both fail their full registered scale gates and therefore have no
authorized confirmation. The exact channel, model identities, prompt budgets,
candidate tokens, nested prefixes, and nonredundancy checks pass. A separate
raw BABILong K2/K3/K5 screen identified one semantically redundant transition;
its corrected analyzer prevents that transition from being counted as scale
evidence.

## Completion and claim boundary

The around-7B/8B requirement is satisfied in two distinct senses:

1. all five planned public 7B/8B checkpoints have ten-panel official BABILong
   results with positive confidence lower bounds; and
2. every frozen Qwen2.5/Falcon3 7B mechanism cell has been executed, with
   confirmation run only where its prospective development gate passed.

This audit does not convert failed interfaces into successes and does not imply
universal dominance. It establishes that the large-model roster is complete,
that the central scale claim has both official generative support and a direct
three-scale interaction confirmation, and that no passed 7B experiment remains
unrun before manuscript preparation.
