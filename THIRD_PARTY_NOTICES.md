# Third-party notices

This repository redistributes **no benchmark row bodies and no model weights**.
It ships the identifiers, hashes, pinned upstream revisions, and construction
scripts needed to rebuild every evaluation panel, together with the per-row
outputs of our own runs. Short strings that appear inside our result records
(generated answers, the parsed event sentences retained in state, questions
and target words) derive from the sources below and remain under their
original licences; they are reproduced only to the extent needed to audit the
results.

## BABILong

- Dataset: `RMT-team/babilong-1k-samples`, Hugging Face Hub, pinned revision
  `fc4d1a584dfc498c37578753bee4cdd91b987ae2`
  (<https://huggingface.co/datasets/RMT-team/babilong-1k-samples>).
- Code and paper: <https://github.com/booydar/babilong>; Kuratov et al.,
  *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack*,
  NeurIPS 2024 Datasets and Benchmarks.
- Licence split, quoted from the dataset card: "Our code is released under
  the Apache 2.0 License. We use data from the PG-19 corpora (Rae et al.,
  2020) (Apache 2.0 License) and the bAbI dataset (Weston et al., 2016) (BSD
  License)."
  - BABILong code and the PG-19-derived background text: **Apache-2.0**
    (<https://github.com/google-deepmind/pg19/blob/master/LICENSE>).
  - bAbI-derived facts, questions and answers: **BSD**
    (<https://github.com/facebookarchive/bAbI-tasks/blob/master/LICENSE.md>).
- What this repository contains: `data/*/MANIFEST.json` (panel SHA-256,
  source-file SHA-256, stable-hash-order offsets), per-panel row-identifier
  lists (`row_id` = SHA-256 of the canonical source row), and the constructors
  `experiments/fetch_babilong_official_source.py`,
  `experiments/prepare_babilong_panel.py`, and
  `experiments/prepare_babilong_semantic_holdout.py`, which rebuild the panel
  files byte-for-byte from the pinned revision. The 16K-token inputs are not
  redistributed.

## RULER

- Generator: `NVIDIA/RULER` (<https://github.com/NVIDIA/RULER>), **Apache-2.0**,
  pinned commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`; the official word
  asset `scripts/data/synthetic/json/english_words.json` (SHA-256
  `affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca`).
- What this repository contains: the frozen configurations with generation
  seeds, task file SHA-256 values, and result records. The generated task
  files are not redistributed; they are rebuilt from the pinned generator
  commit and seeds.

## Model weights

Third-party model weights are not redistributed. The configurations pin each
checkpoint's Hugging Face repository, revision, shard byte sizes and SHA-256
values; the runners abort if a loaded shard does not match. Each model remains
under its own licence: Qwen2.5-0.5B/1.5B/3B/7B-Instruct and Qwen3-8B (Apache-2.0),
Mistral-7B-Instruct-v0.3 (Apache-2.0), Falcon3-7B-Instruct (Falcon-LLM licence),
Granite-3.3-8B-Instruct (Apache-2.0), SmolLM2-135M/360M/1.7B-Instruct (Apache-2.0),
BGE-M3 and its reranker (MIT), and Pythia (Apache-2.0). Consult each repository
for the authoritative terms.

## Baseline code

- Q-RAG (<https://github.com/griver/Q-RAG>) is used unmodified as a retrieval
  baseline through the adapter in `experiments/run_babilong_qrag_*.py`; it is
  not vendored here. See its repository for licence terms.

## Python dependencies

PyTorch, Transformers, tokenizers, safetensors, sentencepiece, NumPy and SciPy
are used under their respective open-source licences; exact versions are
recorded in `reproduction/`.
