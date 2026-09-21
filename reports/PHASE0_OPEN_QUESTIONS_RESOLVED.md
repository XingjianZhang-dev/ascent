# Phase 0 — four open factual questions, resolved

Manuscript: ARRAY-D-26-05300 (Array, major revision, decision 2026-09-17)
Date: 2026-09-20
Scope: only reproducibility wording depends on these answers (§6.2, §6.3, §8,
§9, abstract, cover letter). Every claim below cites a file path and a SHA-256.

Preservation rule: no file under `artifacts/`, `configs/`, `data/`, or
`backups/` was modified. The three frozen archives were streamed read-only and
their recorded SHA-256 values re-verified before extraction:

| Archive | SHA-256 (verified 2026-09-20) |
|---|---|
| `backups/ASCENT_GPU_FINAL_20260823/ASCENT_GPU_FINAL_20260823.tar.zst` | `431049c38a3c91e7b0fb1f8a3d5a52484aeae609a81a12af0c9c5272a5e00821` |
| `GPU_NODES_DISASTER_RECOVERY_20260815/ASCENT_GPU_node1_port32784_20260815_dr.tar.zst` | `2791b1d0b8876652bb857aae3bb0b2fa8da7b39a420b1add775902ecc897e5e0` |
| `GPU_NODES_DISASTER_RECOVERY_20260815/ASCENT_GPU_node2_port10826_20260815_dr.tar.zst` | `87142b7b910dc2fc5ec2fc75add1a5d19cc7d3943dff0ecb98e818162dd63df5` |

## Summary of terminal states

| Q | Question | Terminal state | Consequence for the manuscript |
|---|---|---|---|
| Q1 | What does "correction" mean in the third-node status string? | **RESOLVED_FROM_RECORDS** (case c, with the historical correction identified) | The string is a flag-level constant unrelated to the Qwen3-8B run. §8 states this in one sentence. "Reproduction" may be kept; "independent" may not. |
| Q2 | What was uncommitted when the third node ran? | **RESOLVED_FROM_RECORDS** + **RESOLVED_BY_DIFF** | Concurrent manuscript-packaging edits on the same container; no execution-path file was modified; full diff retained. The "diff not preserved" default wording is not needed. |
| Q3 | Can "physical node" be substantiated? | **RESOLVED_FROM_RECORDS** | Three distinct GPU UUIDs and serial numbers, and three distinct NVIDIA driver versions. "Physical" may stay, attached to what is proven: physically distinct GPUs and (for nodes 1–2) hosts. |
| Q4 | What are the three excluded diagnostics? | **RESOLVED_FROM_RECORDS** + **RESOLVED_BY_DIFF** | Three decode-order diagnostics on panel 1, each reproducing its formal record exactly. §6.3 may keep "complete ten-panel results"; one supplement paragraph suffices. |

No question reached L3 (rerun) or L4 (conservative default). No existing
artifact hash failed verification. The user does **not** need to be consulted
under §11 for Q1 (case a did not occur).

---

## Q1 — "correction" in `posthoc_code_audit_correction_rerun_no_independence_claim`

### Evidence

1. **Where the string comes from.** It is a hard-coded constant in
   `experiments/run_babilong_prompt.py` (line 382 at HEAD), set whenever the
   runner is invoked with `--audit-rerun`:

   ```python
   if audit_rerun:
       diagnostic_status = "posthoc_code_audit_correction_rerun_no_independence_claim"
   ```

2. **When it was introduced.** `git log -S 'posthoc_code_audit_correction'`
   finds two commits. The introducing commit is
   `693c41455d3a431a6da6880d64eb791ec56a6592` (2026-08-14 22:13:14 +0800,
   "Isolate BABILong state accounting by task set"). That commit does two
   things at once:
   - adds the `--audit-rerun` flag and the status constant to
     `experiments/run_babilong_prompt.py`;
   - changes `ascent/babilong_memory.py::read_babilong` to take
     `enabled_tasks` and to allocate inventory-event state (QA7/QA8) only
     when those tasks are enabled, plus tests in `tests/test_babilong_memory.py`.

   The second change is the "code audit correction" the label refers to: a
   **state-byte accounting** fix so that panels containing only QA1–QA3 do not
   accrue inventory-state bytes for tasks they do not contain.

3. **What that correction was applied to, and how it was audited.** The
   SmolLM2 8K generative factorial was rerun at clean commit `693c414` into
   `artifacts/remote_results/babilong_8k_generative_factorial_audit_693c414/`
   (27 files, all carrying the same status string, all `git_dirty=false`).
   `experiments/audit_babilong_generative_factorial.py` (added in
   `c4f7f30bc049205487e05bf644c69ba8242c1c4f`, 2026-08-14 22:31 +0800) audits
   that rerun against two frozen references: decode/score predictions must be
   exact against the immediately preceding factorial (commit `e1843cd`), and
   state bytes are compared against the older diagonal archive (commit
   `072adbd`). Its docstring: *"Audit the corrected generative factorial across
   two frozen references. The immediately preceding factorial is the
   decode/score reference. The older diagonal archive predates inventory-task
   support and is the state-byte reference."* So the 2026-08-14 correction
   changed state-byte accounting and was verified not to change any decode or
   score output.

4. **The formal Qwen3-8B runs already include that correction.** All ten
   formal records under `artifacts/around7b_formal/qwen3-8b/canonical_slots_4/`
   carry `git_commit=65e1491cfac1a766b107415779915cbcbd09a986`,
   `git_dirty=false`, started 2026-08-14T19:54–20:03 UTC. `693c414` is an
   ancestor of `65e1491` (`git merge-base --is-ancestor` → yes; 44 commits
   between them, all on 2026-08-14/15). Therefore the reported Qwen3-8B number
   (72.00 [67.7, 76.3]) was produced by code that already contained the
   accounting fix; nothing was corrected *after* the formal run.

5. **The third-node rerun simply reused the flag.** The ten records in
   `artifacts/array_third_node_audit/` were produced on 2026-08-23
   01:30–01:51 UTC with `--audit-rerun`, so they inherited the constant. The
   ledger entry for that day (`docs/EXPERIMENT_LEDGER.md` §"2026-08-23 —
   complete third-node Qwen3-8B deterministic audit") describes a
   deterministic reproduction and mentions no code change.

6. **L2 diff — nothing changed.** `experiments/compare_third_node_audit.py`
   (SHA-256 `dfd89596f6aff106990d98f604d793f8136f684f41d992925115a3de37626ba2`)
   compares all ten third-node records against all ten formal records,
   field by field and row by row. Result
   (`reports/THIRD_NODE_AUDIT_ROW_DIFF_2026-09-20.json`, SHA-256
   `81d8b46f0b2c636646c263bc13ec839da0f5990ae6114a4423382102dade8d3e`):
   - 10 panels, 800 rows, `status: identical`;
   - every scientific top-level field equal (`endpoint`, `config`, `data`,
     `model_files`, `model_identity`, `parser_accuracy`, `foundation`,
     `ascent`, `gain`, `remaining_error_elimination`, `wins`, `regressions`,
     `by_task`, `state`, `condition` minus the bookkeeping flag);
   - every prediction row equal in every field, including the state-byte
     fields `retained_read_state_bytes` and `persistent_payload_bytes`;
   - only provenance differs: `status`, timestamps, wall time, `systems`
     timing, `git_commit`, `git_dirty`, and
     `condition.diagnostic_no_promotion` (`false` formal / `true` audit).

### Decision

**Case (c) for the third-node run, with the historical correction identified.**
The status string is a generic label attached to every `--audit-rerun`
invocation; the "correction" it names is the 2026-08-14 state-accounting fix
in commit `693c414`, which predates the formal Qwen3-8B runs and was itself
audited as decode/score-exact. No correction was made in connection with the
2026-08-23 third-instance rerun, and that rerun changed no output.

Wording consequences:
- §6.3 may describe the third-instance run as a *reproduction* (it is) but
  not as an *independent replication* (the repository's own label forbids it;
  see Q2 for the reason the label was warranted).
- §8, one sentence: *"The third-instance rerun carries the repository's generic
  audit-rerun label; the 'correction' it names is the 2026-08-14 state-byte
  accounting fix (commit 693c414), which predates every formal 7–8B run and was
  audited as decode-exact."*
- Any **new** audit record produced in this revision uses a neutral label
  (`posthoc_audit_rerun_no_independence_claim`) rather than the legacy string.
  The existing constant in `run_babilong_prompt.py` is left in place so the
  retained records remain traceable to the code that wrote them.

Record hashes (formal, `artifacts/around7b_formal/qwen3-8b/canonical_slots_4/`):

| panel | SHA-256 |
|---|---|
| 1 | `73dd9df55a1edc4609848955ae2907b7769a39d7bdbeb9e45bde543e3eade6fe` |
| 2 | `96e346299602a8fed217f290d1b0098f31f7412b2e8fab116b20ad7d044d7ee8` |
| 3 | `6bce2c4be8889a5f7a174dc91878322989941112de5c23d8ebbe4e8f08602aa7` |
| 4 | `5082369d07729233adc93239a061f96628ef7f7a5a529d53fb7a7f197b0d67ec` |
| 5 | `fc89c09db850b4375ff4cc074003185867b35952513a9b1e374c75b5edcfebba` |
| 6 | `5078bf645a10031ad9e843d51a9d4d8ce64d501fd154e518cdd0c349e6e3d321` |
| 7 | `fc8315df9f0e8126d19d1b8a08c08491c7ae3977cb9a69b6bc309ccb9d2923ab` |
| 8 | `898464c282e2b20dced5dbaf34c5c2f9271539562c3fe9177a277eac34941ecc` |
| 9 | `c0765a71cee996b905843654e8115385c19fb6d25904a113ee4f3d33cde2e354` |
| 10 | `7a2684037e0ff0222537d7bf7a5e65722108df40c69c89b0b7c51ca80ce83392` |

Record hashes (third instance, `artifacts/array_third_node_audit/qwen3-8b-panel{N}-audit-rerun.json`):

| panel | SHA-256 |
|---|---|
| 1 | `e7b3688a49df132d6819a0d22758597aafeb903deb72dd7c647a8964c2fd3eaa` |
| 2 | `fc0060604f8c3ac7c4f53200780a17c1e3bb32a2da552c11ee71b7758fe02e13` |
| 3 | `ba3cea03d44e5bd14780a19f876d5540ca34c5513b25252ee0e3270b70612ffc` |
| 4 | `8dfd9a231edeab8a197702db77fef76e7c689253d58b8d1a38dd54dead0de611` |
| 5 | `50ed4ce70a2109c66e62c98ce07a03abdfa2fe1a7a7f3f7969aafab2cfcdd64b` |
| 6 | `11f02e72dbbe2752b356ab43d93feaeabb2f222cd710e1b2e22e7f9b1f38fdc4` |
| 7 | `6a3df7e56f1b985258f169cebeb24480014798d810c6aaddc40623f1099a7a58` |
| 8 | `94c20961e69ceb7b3f574492f3acb774a0ac2282c39cccf898d9394a329568ef` |
| 9 | `421193cc6c8407ba4e707cb3fe07cb1dff39acc48d2bc092f6b1d27bc02d7613` |
| 10 | `24f2d5413ad510f6ffdf9f478a58bccac8f54c64303671d530c85b0b549e513a` |
| `analysis.json` | `b6ab71487b9dff470a1dd9dd0c1d71ef020640a97004ab255a12c89bf37716dd` |

---

## Q2 — what was uncommitted when the third instance ran (`git_dirty=true`)

### Evidence

1. **The third instance ran from a one-commit snapshot repository, not the
   development history.** The final backup contains the instance's full
   project tree with `.git/`
   (`root/autodl-tmp/ascent_array_20260823/`, listed in
   `backups/ASCENT_GPU_FINAL_20260823/archive_contents.txt`). Its history is a
   single commit:

   ```
   db2407c8508652403f5d94ccedf196e4c407b817
   ASCENT Audit <ascent-audit@invalid.local>
   2026-08-23 09:21:29 +0800  (= 01:21:29 UTC)
   Snapshot for Array audit and reproduction
   ```

   That is nine minutes before panel 1 started (01:30:51 UTC). `db2407c` does
   not exist in the development repository (`git cat-file -t` fails locally);
   it is a snapshot of the project as rsynced to the instance. The recorded
   `git_commit` in the ten audit JSONs is therefore a snapshot identifier, and
   `git_dirty=true` means the working tree diverged from that snapshot at run
   time.

2. **The dirty state was retained.** The backup's metadata directory
   (`ASCENT_GPU_FINAL_20260823_META/BACKUP_METADATA/`, captured
   2026-08-23T14:59:29Z on the same hostname) contains `GIT_HEAD.txt`
   (= `db2407c…`), `GIT_STATUS.txt`, and the full `GIT_DIFF.patch`. They are
   copied out unmodified to `reports/phase0_archive_evidence/node3/`:

   | File | SHA-256 |
   |---|---|
   | `GIT_STATUS.txt` | `71e36855de86d01cc8f0f0aa1bc0411446e38c512c9f736d328bb5f4bf41adb9` |
   | `GIT_DIFF.patch` (1,270,624 bytes, 24 files) | `15eaa337a9a94b3e0a599ef8d75bfedd08d358b7b5f40faf88f709cf600d1fb3` |

   The 24 modified tracked files are, in full:
   `experiments/audit_neural_networks_submission_package.py`,
   `experiments/audit_paper_claims.py`,
   `experiments/build_neural_networks_submission_bundle.py`,
   `experiments/build_neural_networks_supplement.py`,
   `experiments/build_neural_networks_upload_folder.py`,
   `experiments/create_declaration_of_interest.py`,
   `experiments/plot_neural_networks_graphical_abstract.py`,
   `paper/README.md`, `paper/main.tex`, `paper/references.bib`,
   `paper/supplement.tex`, and 13 files under `paper/submission/`
   (cover letter, declarations, graphical abstract, highlights, metadata,
   bundle readme). Untracked files are submission bundles, flat copies of
   scripts placed in the project root for upload, and
   `experiments/analyze_array_third_node_audit.py` (byte-identical to the
   local copy, SHA-256
   `bdd7a629870df46bf40eeb7a90ca419b6adbec47c2be0fdda2e2d5b967237c0f`).

   **None of these files is on the execution path.**
   `experiments/run_babilong_prompt.py` imports only
   `ascent.babilong_memory`, `ascent.babilong_controls`,
   `ascent.generic_retrieval`, `experiments.run_natural_repeat`
   (`git_value`, `sha256_file`) and `experiments.run_ruler_niah`
   (`verify_model_artifact`); none appears in the modified or untracked list.

3. **The execution path was byte-identical everywhere.** For every file in
   `ascent/*.py` (22 files), `experiments/run_babilong_prompt.py`, and
   `configs/babilong_around7b_16k_extension_confirmatory.json`, the SHA-256 of
   the blob is identical across four states: local commit `65e1491` (the
   formal run), local HEAD `98462d9`, the snapshot commit `db2407c`, and the
   instance's working tree at backup time. Their mtimes on the instance
   (preserved by the archive) are 2026-08-13/14, i.e. before the snapshot —
   they were never touched on the instance.

4. **The dirty flag was caused by concurrent manuscript-packaging edits during
   the run.** Panels ran 01:30:51–01:51:21 UTC. The archive preserves mtimes
   of the modified files; several fall inside that window:
   `paper/README.md` 01:37:23Z, `paper/submission/SUBMISSION_METADATA.md`
   01:38:32Z, `BUNDLE_README.txt` 01:38:59Z,
   `plot_neural_networks_graphical_abstract.py` 01:39:20Z,
   `audit_paper_claims.py` 01:39:45Z,
   `build_neural_networks_submission_bundle.py` 01:40:35Z,
   `create_declaration_of_interest.py` 01:43:54Z,
   `Graphical_Abstract.svg` 01:46:25Z. The Array submission package (dated
   2026-08-23) was being assembled on the same instance while the GPU audit
   ran; that is why the runner's dirty-tree check fired.

5. **Same answer as Q1.** The dirty modifications are not the Q1
   "correction" (which was committed nine days earlier), and the L2 diff
   (Q1, item 6) shows the outputs are identical to the formal run, which bounds
   the effect of any working-tree state at zero on every retained field.

### Decision

**RESOLVED_FROM_RECORDS and RESOLVED_BY_DIFF.** The uncommitted modifications
were manuscript and submission-packaging files edited concurrently on the
same instance; the diff is preserved in the frozen backup; no execution-path
file differed from the formal-run commit; all 800 outputs are identical.

The L4 default sentence ("diff was not preserved") is **not** accurate and
must not be used. Replacement for §8:

> The Qwen3-8B rerun on the third instance executed from a one-commit
> snapshot of the repository whose working tree also held uncommitted edits to
> manuscript and submission-packaging files made concurrently on that
> instance; the full diff is retained in the frozen backup, none of the
> modified files is imported by the runner, and every file on the execution
> path is byte-identical to the formal-run commit. The rerun is nonetheless
> retained as a post-hoc code audit rather than an independent replication.

---

## Q3 — can "physical node" be substantiated?

### Evidence

Each of the three instances produced a `nvidia-smi -q` capture inside a
frozen archive. `experiments/extract_phase0_archive_evidence.py` (SHA-256
`4e435bb3678ba29fe37aca242ddc2a2ea7d17fbfcbafad49f65fd9a564aaf983`)
re-verifies each archive hash, streams it read-only, copies the capture files
to `reports/phase0_archive_evidence/<node>/`, and writes
`reports/NODE_IDENTITY_EVIDENCE_2026-09-20.json` (SHA-256
`38e3cc1434cfca3c4300968abaa5efcd795565572983a519b1772f2bcd32bbc6`).

| | node 1 | node 2 | node 3 |
|---|---|---|---|
| hostname | `autodl-container-d91lagjqwt-00a4d7fe` | `autodl-container-d9584ab6ae-3c8a8e5b` | `autodl-container-ggd2kdt87v-ce17c290` |
| role | primary factorial / official-panel node | second node, audit reruns (`audit_node2`) | third instance, Qwen3-8B post-hoc audit |
| GPU | RTX PRO 6000 Blackwell Server Edition | same | same |
| **GPU UUID** | `GPU-342cbca6-b6cd-1081-aa90-5bd70b126178` | `GPU-c9b75175-9fde-b4b2-1430-309151c450f2` | `GPU-71ce435a-24f3-075a-2351-c4154bebd55f` |
| **GPU serial** | `1322925025279` | `1322825091979` | `1322825113993` |
| PCI bus id | `00000000:C8:00.0` | `00000000:D8:00.0` | `00000000:A8:00.0` |
| **NVIDIA driver** | `580.95.05` | `595.71.05` | `580.119.02` |
| kernel | 5.15.0-78-generic | same | same |
| CPU / RAM | Xeon Platinum 8470Q, 208 vCPU, 1.0 TiB | same | same |
| capture (UTC) | 2026-08-15T09:28:07Z | 2026-08-15T09:28:07Z | 2026-08-23T14:59:29Z |
| `NVIDIA_SMI.txt` SHA-256 | `1c42170127533777525fbfb5cc2f20dc554a6c107fc3ee2566dc9cfd7031c4bc` | `0414eaad8fe2d06e149e177415bd91ceb526ab8895ea09773d76bdb698704092` | `e78822a8446a6b388431cec7cdf2d26a41825399a1fe6297e3cf7b4bc456e1ef` |

Retained result records with these hostnames: 82 records on node 1
(2026-08-14T22:33Z → 2026-08-15T00:18Z) and 149 on node 2
(2026-08-14T22:34Z → 2026-08-15T00:15Z); the matched factorial development
pair (`artifacts/sum_numeric_candidate/development/node1/…` and
`…/audit_node2/…`) records `hostname` fields equal to the node 1 and node 2
values above. The ten third-instance records do not store a hostname; the
backup that contains them was taken on node 3's hostname and its `.git`
HEAD equals their recorded `git_commit`.

Two independent lines of evidence:

1. **Distinct GPU devices.** GPU serial numbers are burned into the board and
   UUIDs are derived from the device; three distinct values are three distinct
   physical GPUs.
2. **Distinct hosts for nodes 1 and 2.** The NVIDIA kernel driver is loaded on
   the host and reported identically by every container on that host. Nodes 1
   and 2 were captured at the same second (`09:28:07Z`) with different driver
   versions (`580.95.05` vs `595.71.05`) and different host memory usage;
   they cannot have been containers on one machine. Node 3 was captured eight
   days later with a third driver version; distinct-host inference for node 3
   rests on its distinct GPU serial, since a driver upgrade on a reused host
   cannot be excluded by the driver alone.

Caveat to disclose: the result records store hostname and GPU product name,
not UUID. The hostname→GPU mapping comes from captures taken on the same
container hostnames roughly 9 h (nodes 1–2) and 13 h (node 3) after the last
relevant run. New runs in this revision record the GPU UUID directly in each
result file so this indirection does not recur.

### Decision

**RESOLVED_FROM_RECORDS.** The brief's criterion ("distinct GPU UUIDs
available → 'physical node' may stay, and the UUIDs go into the supplement")
is met. Adopt wording that names what is proven:

- Replace "three independent physical nodes" (and each occurrence of
  "physical node") with **"three separately provisioned compute instances,
  each on a physically distinct GPU (distinct serial number and UUID; Table
  S-x)"**. Where the sentence concerns only nodes 1 and 2, "on distinct hosts"
  may be added, citing the concurrent driver-version evidence.
- Never pair "independent" with the third instance (Q1/Q2).
- The table above goes into the supplement verbatim, with the archive hashes.
- Apply consistently across the abstract, §6.2, §6.3, §8, §9, and the cover
  letter.

---

## Q4 — the three diagnostics excluded from the 70-file formal family

### Evidence

`experiments/audit_around7b_diagnostics.py` (SHA-256
`a09cc13394b1cefecc67fed588b27080ec7c6bb05b97070ae45d2be8febc7b39`) writes
`reports/AROUND7B_DIAGNOSTICS_EXCLUSION_AUDIT_2026-09-20.json` (SHA-256
`c04598aa1e4f3415e6f1d68e61b32206b7cfe6e306512c86fab0fa8c51a28854`).

`artifacts/around7b_formal/` holds 73 result JSONs. The exclusion rule in
`experiments/audit_formal_validation_records.py` (line 61) skips any path
with a `diagnostics` component, leaving 70 = 7 endpoints × 10 panels. The
three excluded files are all in
`artifacts/around7b_formal/diagnostics/ascent_first_batch1/`:

| File | Endpoint | SHA-256 |
|---|---|---|
| `qwen2p5-7b-panel1.json` | Qwen2.5-7B-Instruct | `664175a490f866249fc67a8d98d72d134905c67bad8c9c149c3982752c436534` |
| `mistral-7b-panel1.json` | Mistral-7B-Instruct-v0.3 | `cb4f81b5e0ed66bd16af6bfcf1d70a1854250c38cbabbd7195bcf1efbe0d98db` |
| `qwen3-8b-panel1.json` | Qwen3-8B (non-thinking) | `ee33e76b2c274923fcb7e9cf68ce944ecfb0473e663d0ef48c27e3c95f4ceb41` |

What each one is:

- status `posthoc_decode_batch_diagnostic_no_promotion`,
  `condition.diagnostic_no_promotion = true`;
- same frozen config (`fbb372df…`), same panel
  (`canonical_16k_confirmation_panel_1.jsonl`, `f9aa249c…`, 80 rows), same
  checkpoint hashes, same commit `65e1491` (clean), batch size 1;
- the single deliberate difference: the two arms were decoded in the order
  `ascent_first` instead of the frozen `foundation_first`;
- run 2026-08-14T21:02Z, about one hour **after** the formal panel-1 records
  for the same three endpoints (19:54–19:58Z); they are post-hoc
  decode-order invariance checks, not partial or abandoned formal runs.

Comparison against each file's formal counterpart
(`artifacts/around7b_formal/<endpoint>/canonical_slots_4/canonical_16k_confirmation_panel_1.json`):
every scientific field and all 80 prediction rows are identical for all
three; the only differences are `status`, timestamps, `systems` timing, the
`diagnostic_no_promotion` flag, and `systems.decode_order`.

No analysis script reads the `diagnostics` directory
(`experiments/analyze_babilong_around7b_extension.py` contains no reference
to it), so no reported number depends on the exclusion.

### Decision

**RESOLVED_FROM_RECORDS and RESOLVED_BY_DIFF.** None of the three is a partial
or abandoned run of a reported endpoint; each is a complete 80-row
decode-order check that reproduces the formal record exactly. §6.3 may keep
"complete ten-panel results" for every planned reader. Supplement paragraph:

> The public `artifacts/around7b_formal/` tree contains three result files
> beyond the 70 that form the formal 7–8B family. They sit under
> `diagnostics/ascent_first_batch1/` and are post-hoc decode-order checks on
> confirmation panel 1 for Qwen2.5-7B, Mistral-7B, and Qwen3-8B: identical
> frozen configuration, panel, checkpoint, and batch size, with the two arms
> decoded ASCENT-first instead of Foundation-first. Each reproduces every
> scientific field and all 80 prediction rows of its formal counterpart
> exactly. They are labelled `diagnostic_no_promotion`, are excluded from
> every analysis by path, and contribute to no reported number.

---

## Acceptance checklist (Phase 0, §4.6)

- [x] This file exists; every answer cites a path and SHA-256.
- [x] Q1 RESOLVED_FROM_RECORDS · Q2 RESOLVED_FROM_RECORDS/BY_DIFF ·
      Q3 RESOLVED_FROM_RECORDS · Q4 RESOLVED_FROM_RECORDS/BY_DIFF.
- [x] No UNRESOLVED_CONSERVATIVE outcome; no L4 wording is required.
- [x] No file under `artifacts/`, `configs/`, `data/`, or `backups/` was
      modified: the three archive hashes were re-verified; the official-16K
      and semantic-holdout SHA manifests are re-run in the final acceptance
      pass (Phase 5).
- [x] Q1's L2 diff came back identical; the §11 stop condition did not occur.

New files produced by Phase 0 (all outside the immutable trees):

| Path | SHA-256 |
|---|---|
| `experiments/compare_third_node_audit.py` | `dfd89596f6aff106990d98f604d793f8136f684f41d992925115a3de37626ba2` |
| `experiments/audit_around7b_diagnostics.py` | `a09cc13394b1cefecc67fed588b27080ec7c6bb05b97070ae45d2be8febc7b39` |
| `experiments/extract_phase0_archive_evidence.py` | `4e435bb3678ba29fe37aca242ddc2a2ea7d17fbfcbafad49f65fd9a564aaf983` |
| `reports/THIRD_NODE_AUDIT_ROW_DIFF_2026-09-20.json` | `81d8b46f0b2c636646c263bc13ec839da0f5990ae6114a4423382102dade8d3e` |
| `reports/AROUND7B_DIAGNOSTICS_EXCLUSION_AUDIT_2026-09-20.json` | `c04598aa1e4f3415e6f1d68e61b32206b7cfe6e306512c86fab0fa8c51a28854` |
| `reports/NODE_IDENTITY_EVIDENCE_2026-09-20.json` | `38e3cc1434cfca3c4300968abaa5efcd795565572983a519b1772f2bcd32bbc6` |
| `reports/phase0_archive_evidence/{node1,node2,node3}/` | per-file hashes inside `NODE_IDENTITY_EVIDENCE_2026-09-20.json` |
