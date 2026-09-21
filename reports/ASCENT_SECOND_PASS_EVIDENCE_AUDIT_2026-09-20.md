# ASCENT second-pass evidence audit

Manuscript: ARRAY-D-26-05300  
Date: 2026-09-20  
Purpose: re-search the working tree, Git history, frozen backup, retained result archives, and submission archives after the first 128-item location audit, with special attention to C-15 and R-02/R-06.

## Preservation rule

No file under `artifacts/`, `data/`, `configs/`, or the backup manifests was
edited. Provenance fields, hostnames, paths, timestamps, and serialized floats
were left byte-for-byte unchanged. The new work consists only of an independent
comparison script and reports outside those immutable result locations.

The principal frozen backup remains:

- `backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst`
- SHA-256: `431049c38a3c91e7b0fb1f8a3d5a52484aeae609a81a12af0c9c5272a5e00821`

The historical Array supplementary archive remains:

- `paper/submission/ASCENT_Array_Supplementary_Material_2026-08-23.zip`
- SHA-256: `b6d14bf7dee8aa7162ba18a2a6856c758e506aaab5b87c1430ed4bd54617866b`

## Revised 128-item status

| Status | Count | Meaning |
|---|---:|---|
| `FOUND` | 100 | direct implementation, original artifact, or now-saved audit evidence exists |
| `DISTRIBUTED` | 3 | complete functionality/evidence exists across multiple files or archives |
| `ARCHIVED_ONLY` | 1 | preserved only inside the verified disaster-recovery archive |
| `PARTIAL` | 17 | substantial evidence exists, but a requested standalone deliverable or full environment capture is missing |
| `MISMATCH` | 2 | retained material contradicts the checklist/manuscript formulation |
| `ABSENT` | 5 | no current-tree, Git-history, backup-name, or submission-archive evidence was found |

The machine-readable row-by-row locations remain in
`reports/ASCENT_REPRO_MATERIAL_LOCATIONS_2026-09-20.tsv`. The second pass changes
five rows to `FOUND`: C-04, R-02, R-06, S-02, and S-03.

The final location-integrity check confirms that the TSV contains exactly 128
unique expected identifiers, with no duplicate, missing, or unexpected ID. Its
301 current-tree or outer-archive location references all resolve; its four
inner paths into the frozen zstd archive were also listed and verified. The
only non-path evidence token is the explicit `git history` reference for W-09.

## 1. C-15: implementation is fail-closed, not fallback-to-Foundation

The second pass inspected all relevant runner implementations, their Git
histories, their byte-identical copies in the frozen GPU backup, the formal
result files, unpacked logs, and all retained result archives.

No runner implementing `E_{s,Q}=empty` fallback was found. The following all
raise and stop on invalid frozen inputs or state:

- `experiments/run_babilong_prompt.py`
- `experiments/run_noisy_composition_candidate.py`
- `experiments/run_ruler_aggregation_certified.py`
- `experiments/run_ruler_niah.py`

The four current runner hashes exactly match their copies inside the final GPU
backup. Git-history searches found raising validation checks but no historical
fallback implementation.

Formal-result validation counts:

| Family | Files | Rows | Result |
|---|---:|---:|---|
| official 16K canonical | 30 | 2,400 | all `parser_accuracy=1.0` |
| official 16K raw control | 10 | 800 | all `parser_accuracy=1.0` |
| semantic holdout | 30 | 2,400 | all `parser_accuracy=1.0` |
| formal 7–8B breadth | 70 | 5,600 | all `parser_accuracy=1.0` |
| isolated systems | 18 | 2,160 | all `parser_accuracy=1.0` |
| third-node audit | 10 | 800 | all `parser_accuracy=1.0` |
| numeric factorial confirmation | 81 | 20,736 | all `preflight.pass=true` |

The official 16K SHA manifest verifies 41/41 files; the semantic-holdout SHA
manifest verifies 31/31 files.

The counts, manifest verification, unpacked-log search, and archive-log search
are repeatable with `experiments/audit_formal_validation_records.py`; its saved
machine result is `reports/FORMAL_VALIDATION_RECORD_AUDIT_2026-09-20.json`
(SHA-256 `83c8c05b90259b65b9a025075237c414363107102b7245d08532ee637278effe`).

Failure-log search:

- 127 unpacked log/output/error files: no matching exception/failure.
- 53 retained `tar.gz`/`tgz` archives, 284 text/log members: one traceback.
- The one traceback is a node-2 RULER CWE attempt that lacked the local
  SmolLM2-1.7B checkpoint. It stopped before scoring. It is not a state
  validation failure and created no fallback output.
- The ledger separately records one first-launch config-parsing stop before any
  score because the legacy singular `evaluation_samples` field was absent.

Therefore the manuscript must not claim either fallback behavior or “zero
failed attempts.” It can accurately claim that validation is fail-closed, that
no invalid state was scored as ASCENT, and that retained promoted files passed
their recorded gates. Full evidence and replacement wording are in
`reports/C15_FAIL_CLOSED_VALIDATION_AUDIT_2026-09-20.md`.

## 2. R-02/R-06: correct pair found and exactness established

The first audit cited mismatched panel numbers. That citation was wrong. The
matched cross-node pair is:

- node 1: `artifacts/sum_numeric_candidate/development/node1/rounds_5_sum_numeric_candidate_panel_1_qwen2p5-7b-instruct.json`
- node 2: `artifacts/sum_numeric_candidate/audit_node2/rounds_5_sum_numeric_candidate_panel_1_qwen2p5-7b-instruct.json`

Both records identify:

- panel: `sum_numeric_candidate_panel_1`
- rounds/state size: 5
- endpoint: Qwen2.5-7B-Instruct
- config SHA-256: `ea488327c9f29d7d551c2d02205bd03471cf1927c0fb16ba2f02cbf0b181c510`
- data SHA-256: `74e580b8485d97955f07d3b0760175532a65250f5816c332815d76de46949c81`
- Git commit: `f133a9689adb870680d462049c0dd4e0bfc41e9a`
- 256 rows in the same order, on two distinct recorded hostnames

The new independent comparison is:

- script: `experiments/compare_factorial_cross_node.py`
- saved result: `reports/FACTORIAL_CROSS_NODE_EXACT_AUDIT_2026-09-20.json`
- result SHA-256: `1a91bc33ae59a8d27cdf66f154b0269dd50ee979709a66eaba3f79b963937816`
- status: `passed`

Exactness result:

- every listed scientific top-level field is equal;
- all 256 prediction records are equal;
- those records contain 256 Foundation and 256 ASCENT 15-way vectors, i.e.
  **512 arm-level vectors and 7,680 probability values**;
- all 768 row-level Foundation/ASCENT/gain NLL values are equal;
- after JSON parsing, every stored probability and row NLL has the same
  IEEE-754 binary64 bytes;
- timestamp, hostname, audit-rerun flag, and wall time are intentionally not
  scientific equality fields.

The cross-node audit is the one preregistered **development panel**, not all
nine untouched confirmation panels. The manuscript's current phrase “all 256
probability vectors” is technically ambiguous because each of 256 records has
two vectors. An evidence-exact formulation is:

> On the preregistered development panel, an audit rerun on a second physical node reproduced all 256 prediction records, including the 512 arm-level 15-way probability vectors (7,680 stored probabilities), all row-level joint NLLs, and all prespecified scientific fields exactly in their saved binary64 representation.

This supports retaining “exactly,” but only with the panel and equality scope
made explicit.

One qualification is required for the broader reproducibility paragraph. Each Qwen3 third-node JSON labels
itself `posthoc_code_audit_correction_rerun_no_independence_claim` and records
`git_dirty=true`. The physical node is distinct and the saved scientific fields
compare exactly, but the repository's own status forbids describing this as an
*independent* replication. Use “separate third-node audit” for Qwen3; reserve
“independent” for reruns whose retained provenance supports that word. A fully
safe combined sentence is therefore:

> Cross-node audit reruns reproduced the selected audit cells: all 80 Qwen2.5-7B official-panel predictions, all 256 factorial development-panel prediction records with their stored probabilities and NLLs, and all 800 Qwen3-8B official-panel paired predictions and prespecified scientific fields. The Qwen3 run is retained as a post-hoc third-node code audit rather than an independent replication, and these audits do not assert that every formal experimental cell was rerun on another node.

## 3. C-04 was present

`experiments/audit_babilong_qwen_preflight.py` is a genuine exhaustive,
score-free parser preflight. For every selected row it verifies the panel hash,
registered row count, task balance, unique task/row identity, and parser answer;
it raises before scoring unless parser correctness equals the total row count.
It also checks pinned model artifacts and refuses to run if result files already
exist or the worktree is dirty. C-04 is therefore `FOUND`, not `PARTIAL`.

This does not by itself complete C-08: the writer is structurally called with
input and question rather than target, and factorial preflights record target
omission, but there is still no single repository-wide per-row assertion that
instruments every writer family for target-field access.

## 4. Security and path audit

### Git-history credential scan

All 1,535 unique reachable Git blobs were covered:

- 1,527 ordinary blobs were read directly from Git object storage;
- eight blobs larger than 10 MB were archive files;
- each of the eight current archive files exactly matches its recorded Git
  object ID;
- the three zstd backups and five historical/current submission zip archives
  were decompressed-stream scanned.

No private-key header, OpenAI/Hugging Face/GitHub/AWS/Google/Slack credential
pattern, or credential-style `password`/`api_key`/`access_token` assignment was
found. The current Array supplementary zip contains 531 members and produced
zero high-signal secret matches. No secret value was printed or saved during
the audit.

This is a content-pattern audit, not a claim that a third-party scanner was
used; `gitleaks` and `trufflehog` were not installed. The complete reachable
history was nevertheless scanned rather than sampling only the current tree.

### Local paths and hostnames

- No `/Users/...`, project-root string, cloud login hostname, `root@` login, or
  local username occurs in the historical Array supplementary zip.
- Opaque `autodl-container-*` hostnames occur in 82 packaged result JSON files.
  These are immutable provenance fields and must not be rewritten in place.
- No `/Users/xingjianzhang0816` or `ChatGPT/Ascent -- TNNLS` string was found in
  current `artifacts/`, `configs/`, `reproduction/`, or the unpacked backup
  manifests.
- One local project path occurs in the frozen backup's
  `PROJECT_VENV_PIP_FREEZE.txt`, as part of the original environment capture.
  It is preserved inside the hashed backup rather than rewritten.
- Current absolute-path occurrences outside immutable results are limited to
  disaster-recovery verification metadata and this audit report.

The correct public explanation is that opaque runtime hostnames and original
environment paths are retained only where they form part of the provenance and
hash chain; they are not credentials and are not required for rerunning the
portable commands.

### “AI trace” search

The manuscript already contains the required generative-AI disclosure. No
`AI_ASSISTANCE.md` exists yet. The Array supplement's only matches to common AI
vendor-name patterns are five BABILong-derived JSONL files containing the
ordinary personal name “Claude” in task text; they are not AI-tool metadata.

## 5. Submission-history consistency

The local record supports the following chronology:

1. TNNLS was a project target/readiness standard. Local files named TNNLS are
   planning, gap-audit, and protocol documents; no local evidence of an actual
   TNNLS manuscript submission was found.
2. A Neural Networks submission package and dated author declarations exist.
3. The Array cover letter says the manuscript was transferred through
   Elsevier's manuscript-transfer service and is not under consideration
   elsewhere. This is consistent with the known Neural Networks rejection and
   subsequent Array transfer.

The directory suffix `Ascent -- TNNLS` and the old builder filename
`build_neural_networks_supplement.py` are historical development labels, not
evidence of concurrent submission. The public timeline should say this
explicitly rather than delete or rewrite hashed provenance.

## 6. Items confirmed genuinely absent

After searching the current tree, 271-commit reachable Git history, backup
member names, and submission archive member names, five requested deliverables
remain absent:

1. `C-39`: a power/sample-size/MDE script.
2. `W-03`: `AI_ASSISTANCE.md` with per-file assistance and independent checks.
3. `W-05`: a repository code license.
4. `W-06`: an artifact/data license.
5. `W-08`: a Zenodo version DOI record.

These are not “hidden somewhere else”; they still need to be created or, for
the DOI, minted after a release.

## 7. Important partial/mismatch items still open

Highest-priority open deliverables:

- `C-08`: repository-wide per-row target-blindness instrumentation/assertion.
- `C-26/D-17`: a flat 13,824-row transition audit file; current analysis checks
  and summarizes all rows but saves only aggregate `6 x 2,304` counts.
- `C-38/W-09`: standalone freeze/preregistration timeline tied to commits and
  timestamps.
- `W-02`: public `VERIFICATION.md` combining numeric provenance, exactness,
  13,824-row construction, and freeze evidence.
- `W-04`: reader-facing writer-schema scope and limitations document.
- `C-60`: one wrapper command around the existing three CPU regeneration
  commands.
- `E-04`: full conda/pip exports exist only inside the final frozen backup and
  should be extracted into a public `reproduction/` directory without altering
  the backup.
- `E-06`: explicit determinism policy. Existing code/configs record greedy
  decoding, dtype, revisions, and seeds, but no repository-wide statement of
  TF32, deterministic-algorithm, and attention-kernel settings exists.
- `S-06`: saved transcript from a fresh clone/environment rerun.
- `E-03/R-05`: the third node records Python, Torch, Transformers, tokenizer,
  SentencePiece, safetensors, GPU model, peak CUDA bytes, commit, and dirty flag,
  but no full CPU/RAM/pip snapshot was retained. This cannot be reconstructed
  retrospectively and must be disclosed as a limitation.

`S-05` remains a mismatch for the historical supplement/publication plan: the
Array supplementary zip contains 49 data files, including selected BABILong and
RULER row bodies. A future public repository package should be rebuilt to carry
constructors, selected IDs, hashes, source revisions/links, and redistribution
notices according to upstream licenses. Original local data and immutable
artifacts should remain untouched.

### BABILong redistribution boundary

The retained source manifest identifies the upstream dataset as
`RMT-team/babilong-1k-samples` at revision
`fc4d1a584dfc498c37578753bee4cdd91b987ae2`. The upstream dataset card and
official BABILong repository describe mixed component licensing: BABILong code
under Apache-2.0, PG-19 material under Apache-2.0, and bAbI material under BSD.
This means redistribution is not categorically forbidden, but a public package
must preserve the applicable attribution, license, and notice obligations for
each component. It would be inaccurate to label all packaged rows simply as
“Apache-2.0 data.”

Relevant upstream records:

- <https://huggingface.co/datasets/RMT-team/babilong-1k-samples/blob/main/README.md>
- <https://github.com/booydar/babilong>
- <https://proceedings.neurips.cc/paper_files/paper/2024/file/c0d62e70dbc659cc9bd44cbcf1cb652f-Paper-Datasets_and_Benchmarks_Track.pdf>

Because W-05/W-06 and the corresponding third-party notices are still absent,
the lowest-risk future public release remains: publish constructors, selected
row/task identifiers, exact hashes, pinned upstream revision and retrieval
instructions, rather than republishing row bodies. If row bodies are retained,
add the component-specific notices before release. This recommendation applies
only to a newly built public package; it does not authorize rewriting the
historical supplementary zip or any hashed artifact.

## 8. Bottom line

The two most integrity-sensitive questions now have direct answers:

- C-15 is a manuscript-description bug: the implementation is consistently
  fail-closed, and no invalid state was silently scored via Foundation.
- R-02/R-06 pass after correcting the panel reference: the matched development
  panel reproduces every saved scientific value exactly across two physical
  nodes, with binary64 equality explicitly defined and independently checked.

The repository is materially stronger than the first audit suggested, but it
is not yet publication-ready as a public reproducibility repository until the
five absent deliverables and the high-priority partial items above are handled.
