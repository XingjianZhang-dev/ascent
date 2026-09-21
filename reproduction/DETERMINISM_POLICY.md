# Determinism policy — what was fixed, and what was not

This document states, from the code and configs as they were executed, which
sources of run-to-run variation were controlled in the retained experiments
and which were not. It was written for the Array revision (2026-09) by
reading the frozen runners; it does not retroactively introduce any setting
that the retained runs did not use. Statements about defaults refer to
PyTorch 2.8.0+cu128 and Transformers 4.57.6, the pinned versions.

## Fixed by code or config

| Source of variation | Setting | Where |
|---|---|---|
| Model weights | Pinned Hugging Face revision per endpoint; every `.safetensors` shard is hashed before loading and must match the frozen `model_artifact(s)` byte size and SHA-256, otherwise the run aborts | `configs/*.json`; `verify_model_artifact` in `experiments/run_ruler_niah.py` (used by every runner) |
| Weight dtype | `torch.bfloat16` for all decoder forward passes | `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16)` in `experiments/run_babilong_prompt.py` and `experiments/run_noisy_composition_candidate.py` |
| Decoding | Greedy (`do_sample=False`), fixed `max_new_tokens` from the config, fixed `eos`/`pad` ids, left padding | `model.generate(...)` in `run_babilong_prompt.py`; `tokenizer.padding_side = "left"` |
| Candidate scoring | Exact joint log-likelihood of the 15 fixed two-token candidates after one fixed assistant prefix; final logits are cast to float32 before `log_softmax`; probabilities stored as float64 | `score_candidates` in `run_noisy_composition_candidate.py` |
| Batch size | Frozen per endpoint in the config (`endpoint.batch_size`), recorded as `frozen_batch_size`/`effective_batch_size` in every result; a non-frozen batch size can only be used with `--diagnostic-batch-size`, which tags the result `no_promotion` | both runners |
| Decode order | `foundation_first` is the only allowed order in the confirmatory configs; the other order can only be used as a tagged diagnostic | `allowed_decode_orders` in the config; `run_babilong_prompt.py` |
| Panel construction | Every panel is a hashed JSONL file; the runner re-hashes the panel and its `MANIFEST.json` and aborts on mismatch. Panel sampling seeds are frozen: `seed = 20260815` in the official-16K config; `2026085000 + panel_index` for the numeric factorial (recorded in the data manifest `seed_rule`) | `configs/*.json`, `data/*/MANIFEST.json`, `experiments/prepare_*.py` |
| Torch RNG | `torch.manual_seed(config["seed"])` is called before decoding in the BABILong runner. With greedy decoding and no dropout this seed does not influence any output; it is set so that any accidental sampling path would be reproducible | `run_babilong_prompt.py` |
| Prompt truncation | Left truncation to `context_tokens - max_new_tokens`; the number of rows at the cap is recorded per arm (`samples_at_prompt_token_cap`) | `run_babilong_prompt.py` |
| Chat template | The rendered chat-template probe is hashed (`chat_template_probe_sha256`) and compared to the frozen value | endpoint block in each config |
| Runtime library versions | `transformers`, `tokenizers`, `safetensors`, `sentencepiece` versions are validated against the config's `runtime_dependencies` and the run aborts on mismatch | `validate_runtime_dependencies` in both runners |
| Repository state | Commit hash and dirty flag recorded in every result (`environment.git_commit`, `environment.git_dirty`) | both runners |

## Not fixed — stated explicitly

| Source of variation | State in the retained runs | Consequence |
|---|---|---|
| `torch.use_deterministic_algorithms` | **Never called.** Default `False` | Non-deterministic CUDA kernels (e.g. some scatter/index and attention paths) are not forbidden. On matched hardware the retained cross-node reruns nevertheless reproduced every stored float bit-for-bit (development panel, node 1 vs node 2), so the executed paths were deterministic in practice, but this was observed, not enforced. |
| `torch.backends.cudnn.deterministic` / `benchmark` | Defaults (`False` / `False`) | cuDNN autotuning was not enabled, so kernel selection did not vary with timing; determinism was not requested. |
| TF32 | `torch.backends.cuda.matmul.allow_tf32` default `False`; `torch.backends.cudnn.allow_tf32` default `True`; `torch.set_float32_matmul_precision` **not called** in either formal runner (it is called with `"high"` only in `experiments/run_babilong_qrag_retrieval.py`, the Q-RAG retrieval baseline) | The decoder forward passes are bf16, so TF32 affects at most the float32 `log_softmax` epilogue in the factorial scorer, which does not use cuDNN. No TF32 policy was pinned; the defaults above are what ran. |
| Attention kernel | Transformers default (`sdpa`); not pinned to a specific backend | The SDPA backend chosen by PyTorch may differ by GPU architecture. |
| `CUBLAS_WORKSPACE_CONFIG` | Not set | cuBLAS reduction order is not constrained. |
| GPU architecture | All retained formal runs: NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0) | **Bit-level identity is a matched-architecture property.** The revision-stage rerun on an A100 (compute capability 8.0) with identical software pins reproduced panel statistics to within one row per arm but flipped a small number of individual greedy outputs and argmax candidates and did not reproduce any stored float bit-for-bit; see `artifacts_revision/crossnode_2026-09/CROSSARCH_COMPARISON.json`. |
| Python patch version | 3.12.3 on all retained nodes | The first revision-stage A100 run used 3.12.14 by mistake and is retained as a disclosed deviation (`artifacts_revision/crossnode_2026-09_run1_python3.12.14/`); the pre-registered run used 3.12.3. |
| `numpy` | 2.3.2 on all retained nodes (`reproduction/environment_captures/`) | Used only for float64 aggregation of stored probabilities; not validated by the runner. |
| Tokenizer parallelism / thread counts | Not pinned | Affects timing only. |

## What "exact" means in this repository

Three levels are used, always named explicitly:

1. **Scientific-field identity** — every top-level field that enters an analysis
   (accuracies, NLL summaries, wins/regressions, state accounting, hashes) is
   equal after JSON parsing.
2. **Prediction identity** — every per-row discrete output (generated string,
   parsed answer, score, argmax candidate) is equal.
3. **IEEE-754 binary64 identity** — every stored float (probability vectors,
   row NLLs) has the same 8-byte representation.

Retained matched-architecture reruns satisfy all three on the audited cells
(`reports/FACTORIAL_CROSS_NODE_EXACT_AUDIT_2026-09-20.json`,
`reports/THIRD_NODE_AUDIT_ROW_DIFF_2026-09-20.json`,
`artifacts/around7b_crossnode/qwen7b-panel1-exact-audit.json`). The
cross-architecture rerun satisfies none of the three row-for-row; it is
reported at the level of panel statistics with the row-level differences
enumerated. The manuscript uses "exact" only for level 3 on matched hardware
and names the panel and node pair each time.
