# Deviations from the pre-registration — Phase 2A

Recorded so that nothing about this audit has to be discovered by a reader.

## 1. First execution used Python 3.12.14 instead of the pre-registered 3.12.3

`PREREGISTRATION.json` lists Python 3.12.3 under `software_parity`. The
instance's default virtual environment (`/venv/main`) is Python 3.12.14; the
system interpreter, which I had checked, is 3.12.3. The first execution of
`experiments/run_crossarch_reproduction.sh` (2026-09-20T21:04–21:09Z)
therefore ran under 3.12.14 (and numpy 2.5.3 instead of 2.3.2). This was
noticed when the rerun's `environment.python` field was compared with the
reference records.

Handling:

- The 3.12.14 run is **retained in full**, unmodified, as
  `../crossnode_2026-09_run1_python3.12.14/` (its own environment
  snapshots, logs, results, `SHA256SUMS`, and the comparison output produced
  by the comparator version then in force). It was moved to that directory,
  not deleted or relabelled inside its files.
- A second virtual environment was built from the system interpreter
  (`/usr/bin/python3.12` = 3.12.3) with the identical pinned stack (torch
  2.8.0+cu128, transformers 4.57.6, tokenizers 0.22.2, safetensors 0.8.0,
  sentencepiece 0.2.2, numpy 2.3.2), and the pre-registered configuration was
  executed at 2026-09-20T21:12–21:18Z from the same clean worktree
  (`9c15400`). That execution is the one in this directory.
- Both runs were compared against the retained references and against each
  other. They are identical to each other at every level, including
  binary64 identity of all stored floats
  (`SAME_GPU_RUN1_VS_RUN2_CONTROL.json`). The deviation therefore had no
  effect on any output, and the second execution was not a search for
  agreement: it restored the pre-registered environment and produced the
  same numbers.

## 2. `model_identity` is absent from the official-16K reference records

The BABILong runner gained a `model_identity` output field in commit
`0d60263` (2026-08-15 02:49 +0800), after the official-16K reference runs
(2026-08-14, commit `4fd25e9`). The reruns therefore carry the field and the
references do not. The comparator initially reported this as a top-level
mismatch. It was changed (after run 1, before run 2's comparison) to report
such fields as `fields_absent_from_reference_schema_not_compared`, with an
explicit `fields_skipped_explanation` string in the output, and to exclude
them from every verdict. The comparator's pre-registered SHA-256 was
`f2b0c2fc82c9f5b80a10062dba2dbc61720dd901cc1f9ba23e6e42d2eb91455e`; the
version used for `CROSSARCH_COMPARISON.json` is the one committed after this
change (`git log -- experiments/compare_crossarch_reproduction.py`). The
run-1 comparison output produced by the original version is retained in
`../crossnode_2026-09_run1_python3.12.14/CROSSARCH_COMPARISON.json`; its
row-level and panel-level findings are the same as those reported here.

## 3. Nothing else

No cell was rerun with different settings, no seed, batch size, dtype or
decode order was changed, no hardware was substituted, and no output path
under `artifacts/` was written.
