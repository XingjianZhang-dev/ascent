# Reproducing ASCENT — from the released records, and from scratch

Three levels, in increasing cost. Level 1 needs no GPU and no downloads.

## Level 1 — every reported number from the released records (CPU, seconds)

```sh
# Python 3.10 or newer (exact NumPy/SciPy pins on 3.12+)
python3 -m venv .venv && source .venv/bin/activate
pip install -r reproduction/requirements-reproduce.txt      # numpy, scipy
make reproduce                                               # prints every number; non-zero exit on mismatch
make test                                                    # unit tests; GPU-only and row-body tests skip with a reason
```

`experiments/reproduce_all_tables.py` re-verifies the SHA-256 manifests,
recomputes the four authoritative analysis files from the per-row records,
re-renders every generated manuscript table byte for byte, regenerates the
power analysis and the 13,824-row transition audit (compared column-wise:
structure and non-float columns exact, float columns to 1e-12, since NumPy's
SIMD dispatch differs across CPUs), and prints every reported number.
`VERIFICATION.md` §2 maps each value to its records and command.

## Level 2 — re-running any experimental cell on a GPU

Environment (identical to every retained run; `DETERMINISM_POLICY.md` states
what is and is not pinned):

```sh
python3.12 -m venv .venv-gpu && source .venv-gpu/bin/activate     # Python 3.12.3 was used
pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
pip install "transformers==4.57.6" "tokenizers==0.22.2" "safetensors==0.8.0" "sentencepiece==0.2.2" "numpy==2.3.2" scipy accelerate huggingface_hub
```

Weights: download each checkpoint at the revision pinned in `configs/`
(Appendix E of the paper lists them); the runner hashes every shard before
loading and aborts on mismatch. Example:

```sh
hf download Qwen/Qwen2.5-3B-Instruct --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 --local-dir models/qwen2p5-3b-instruct
```

Panels: BABILong and RULER row bodies are not redistributed (see
`THIRD_PARTY_NOTICES.md`). Rebuild them from the pinned upstream revision;
the constructors are deterministic and the runner checks the panel SHA-256
recorded in the config:

```sh
# official BABILong 16K QA2/QA3 source at the pinned dataset revision
python -m experiments.fetch_babilong_official_source --output-root artifacts/babilong_official_16k_qa23_source --dataset-config 16k
# ten disjoint 80-row confirmation panels in stable hash order (see data/babilong_16k_canonical_confirmation/MANIFEST.json)
python -m experiments.prepare_babilong_panel --source-root artifacts/babilong_official_16k_qa23_source --output-root data/babilong_16k_canonical_confirmation
sha256sum data/babilong_16k_canonical_confirmation/canonical_16k_confirmation_panel_1.jsonl   # must equal the hash in MANIFEST.json and *.ids.json
```

The semantic holdout is built by `experiments/prepare_babilong_semantic_holdout.py`
from the BABILong training-generation split with the official 8K pool as the
exclusion set; RULER task files are generated with the pinned `NVIDIA/RULER`
commit and the seeds recorded in each `configs/ruler_*.json`. The synthetic
factorial panels under `data/posttraining_*` are included and hash-checked.

Run a cell exactly as the retained records were produced (the record embeds
the config hash, panel hash, model shard hashes, commit and dirty flag):

```sh
# official 16K BABILong, one reader, one panel, both arms
python experiments/run_babilong_prompt.py \
  --config configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json \
  --data data/babilong_16k_canonical_confirmation/canonical_16k_confirmation_panel_1.jsonl \
  --endpoint qwen2p5-3b-instruct --model-dir models/qwen2p5-3b-instruct \
  --condition canonical_coscale --output out/canonical_16k_confirmation_panel_1.json

# numeric factorial, one reader, one panel, one state size
python experiments/run_noisy_composition_candidate.py \
  --config configs/posttraining_noisy_composition_sum_numeric_candidate.json \
  --data-root data --panel sum_numeric_candidate_panel_2 --endpoint qwen2p5-7b-instruct \
  --model-dir models/qwen2p5-7b-instruct --rounds 5 --output out/rounds_5_sum_numeric_candidate_panel_2_qwen2p5-7b-instruct.json
```

Compare a rerun with its retained reference at the four ladder levels
(discrete, continuous, printed precision, binary64):

```sh
python experiments/compare_crossarch_reproduction.py --pairs pairs.json --output comparison.json
```

Expect prediction-level and bit-level identity on matched hardware (RTX PRO
6000 Blackwell) and small row-level differences on other architectures
(`artifacts_revision/crossnode_2026-09/CROSSARCH_REPRODUCTION_REPORT.md`).

## Level 3 — the revision-stage audits

Each has a committed pre-registration, a job script, environment snapshots
before and after, and an analysis script:

| Audit | Pre-registration | Job | Analysis |
|---|---|---|---|
| Cross-architecture reproduction (2A) | `artifacts_revision/crossnode_2026-09/PREREGISTRATION.json` | `experiments/run_crossarch_reproduction.sh MODELS OUT` | `experiments/compare_crossarch_reproduction.py` |
| Per-row target blindness (2B) | `artifacts_revision/target_blindness_2026-09/PREREGISTRATION.json` | `experiments/run_target_blindness_audit.sh MODELS OUT` | `experiments/analyze_target_blindness_audit.py` |
| Schema-blind writer ablation (2C) | `artifacts_revision/schema_blind_2026-09/PREREGISTRATION.json` | `experiments/run_generic_writer_ablation.sh MODELS OUT all` | `experiments/analyze_generic_writer_ablation.py` |

`experiments/record_run_environment.py` writes the environment snapshot
(GPU UUID/serial/driver, `nvidia-smi -q`, `lscpu`, `pip freeze`, git commit and
dirty flag, pre-registration hash) that every revision run carries.
