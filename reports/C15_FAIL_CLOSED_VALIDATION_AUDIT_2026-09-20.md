# C-15 fail-closed validation audit

Date: 2026-09-20  
Scope: the current working tree, the frozen GPU backup, retained formal result files, retained log files, all 53 `tar.gz`/`tgz` result archives, and the recorded Git history of the relevant runners.

Repeatable audit:

- `experiments/audit_formal_validation_records.py`
- `reports/FORMAL_VALIDATION_RECORD_AUDIT_2026-09-20.json`
- result SHA-256: `83c8c05b90259b65b9a025075237c414363107102b7245d08532ee637278effe`

## Conclusion

The implementation does **not** implement the manuscript's stated behavior that a failed state-validity check sets `E_{s,Q}=empty` and silently returns the Foundation path. The relevant runners are fail-closed: a missing, malformed, mismatched, or unsupported input raises an exception and stops that invocation before an output record is promoted.

No second runner implementing the manuscript's fallback rule was found. No retained promoted result contains a fallback marker or a state-validation failure that was converted into a Foundation prediction.

The manuscript statements in Section 4.1 and Section 4.3 are therefore implementation-description errors and should be replaced with a fail-closed description. This is not evidence that a Foundation output was counted as an ASCENT output; the opposite is true—the code refuses to create such a replacement output.

## Runner-by-runner result

| Runner | Validation behavior | Representative checks |
|---|---|---|
| `experiments/run_babilong_prompt.py` | fail-closed | frozen panel hash, row count, task balance, model artifact, retrieval-cache presence/hash/alignment, supported state/readout, parser agreement; failures raise `RuntimeError`/`ValueError` |
| `experiments/run_noisy_composition_candidate.py` | fail-closed | registered panel/state, panel and manifest hashes, row count, target construction, exact posterior, candidate support, token sequence distinctness, scoring-boundary stability, context length; failures raise `RuntimeError` |
| `experiments/run_ruler_aggregation_certified.py` | fail-closed | task/seed hash, data hash, row count, registered reader and parser agreement; failures raise `RuntimeError` |
| `experiments/run_ruler_niah.py` | fail-closed | model-weight presence/hash, registered reader, manifest/task hashes and row counts; failures raise `RuntimeError`/`ValueError` |

Current source hashes:

| File | SHA-256 |
|---|---|
| `experiments/run_babilong_prompt.py` | `ba981a01d235df34906a704acd3bb3736a8fce41401138d66d2a7cd1c5cdd401` |
| `experiments/run_noisy_composition_candidate.py` | `e6d0c0724ff7dad95fafcd3586c6a5b8de221a148ddfc3a479725e4ac9724f50` |
| `experiments/run_ruler_aggregation_certified.py` | `81d840208aeb0ae62e723eefccba5216db559f115a6599ad9d459621f806cea0` |
| `experiments/run_ruler_niah.py` | `6d4ff6334c8ad2588fba6ba98933e8920d3de2d9b5f23fd55cb894772049fb62` |

The copies of all four files inside
`backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst`
have the same four SHA-256 values. The frozen backup therefore confirms the
behavior observed in the current tree; this is not a later local-code drift.

## Formal-result checks

The following retained result families were parsed directly:

| Result family | Result files | Prediction rows | Recorded gate |
|---|---:|---:|---|
| official 16K canonical Qwen2.5 | 30 | 2,400 | every file has `parser_accuracy = 1.0` |
| official 16K raw 3B representation control | 10 | 800 | every file has `parser_accuracy = 1.0` |
| semantic holdout | 30 | 2,400 | every file has `parser_accuracy = 1.0` |
| formal 7–8B breadth (excluding three diagnostics) | 70 | 5,600 | every file has `parser_accuracy = 1.0` |
| isolated systems runs | 18 | 2,160 | every file has `parser_accuracy = 1.0` |
| third-node audit | 10 | 800 | every file has `parser_accuracy = 1.0` |
| numeric factorial confirmation | 81 | 20,736 | every file has `preflight.pass = true`; all preflight subchecks agree |

The official 16K SHA manifest verifies 41/41 files, and the semantic-holdout
manifest verifies 31/31 files. A recursive key scan across 221 retained formal
result files found no `fallback`, malformed-state, or validation-failure field;
the only matching validation key was the positive
`candidate_support_valid` factorial preflight field.

## Failure-log audit

- 127 unpacked `.log`/`.out`/`.err` files under `artifacts/` were searched for tracebacks, runtime/value/assertion failures, missing files, CUDA OOM, and failure markers. No match was found.
- All 53 retained `tar.gz`/`tgz` result archives were opened. Their 284 text/log members (165,959 bytes) were searched with the same failure classes. Exactly one traceback was found:
  - archive: `artifacts/remote_results/cwe_confirm_c749f24/cwe_confirm_c749f24_node2.tar.gz`
  - member: `cwe_confirm_3771662/seed382303_smollm2-1p7b-instruct.log`
  - cause: the node-2 path did not contain the SmolLM2-1.7B model weights.
  - interpretation: this invocation stopped before scoring. The retained 1.7B confirmation was completed on node 1. It is an infrastructure/model-availability interruption, not a state-validity failure and not a fallback.

The experiment ledger records one additional pre-score interruption for the
same RULER CWE confirmation: the first launch stopped during config parsing
because the legacy singular `evaluation_samples` compatibility field was
absent. The compatibility field was added before any confirmation score, with
data, models, scale law, metrics, and gates unchanged. This also produced no
fallback output.

Consequently, it would be inaccurate to write that *no formal attempt ever
failed*. The supported, narrower statement is:

> No retained promoted result was produced by silently replacing invalid ASCENT state with a Foundation output. The formal promoted result files passed their recorded parser/hash/preflight gates. Two documented RULER CWE attempts stopped before scoring—one at config parsing and one because a node lacked a model checkpoint—and neither contributed a score.

## Git-history audit

The recorded histories of all four runners were searched for `fallback`,
`foundation path`, the execution-contract notation, and missing/malformed-state
handling. No historical fallback implementation was found. The relevant
validation checks enter history as raising checks (for example, the BABILong
panel-hash check at commit `d53b21f...` and the numeric-candidate panel-hash
check at commit `95764cf...`).

## Manuscript corrections required

Current inaccurate statements:

1. Section 4.1: “A failed validity check sets `E_{s,Q}=empty` and therefore returns the foundation path.”
2. Section 4.3: “Deterministic validity checks route missing or malformed state to the foundation path.”

Evidence-faithful replacement:

> Each endpoint executes the same five operations: (i) write state from X before Q and Y are revealed; (ii) validate registered types, ordering, hashes, alignment, and budget constraints; (iii) select at most s query-relevant entries; (iv) serialize those entries with the fixed canonical template; and (v) decode with the unchanged model and scorer. Validation is fail-closed: a missing, malformed, or mismatched state aborts the invocation rather than silently substituting a Foundation output. All retained promoted result files passed their recorded parser, hash, alignment, and preflight gates.

And in the co-scaled-budget paragraph:

> Deterministic validity checks are fail-closed; missing or malformed state is not scored as an ASCENT observation.

These corrections describe the code without claiming that every attempted job
completed successfully.
