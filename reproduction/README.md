# ASCENT core experiment reproduction package

This package verifies the two strongest current empirical controls without
changing their frozen protocols:

1. official BABILong 8K scale complementarity against frozen BGE-M3 hybrid
   retrieval/reranking and an exact ASCENT-byte-capacity raw FIFO; and
2. the exact same-suffix-FLOP static replay control on Pythia-410M and
   Pythia-2.8B.

It also includes the later cross-node prediction replication and the frozen
repeat of the isolated retrieval-latency tail.  These later runs are
reproducibility/system audits, not additional independent statistical panels.

## Integrity check

Place the four raw result archives at the paths recorded in
`core_manifest.json`, then run:

```bash
python experiments/verify_reproduction_package.py \
  --manifest reproduction/core_manifest.json
```

The verifier checks every config, official panel, and raw archive hash; parses
all JSON configs; and checks the frozen pass flags stored inside the two main
archives.  It never regenerates or overwrites results.

## Runtime lock

Both GPU nodes used Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.57.6,
CUDA runtime 12.8, cuDNN 9.10.2, and an NVIDIA RTX PRO 6000 Blackwell Server
Edition with 97,887 MiB.  The only hardware-environment difference relevant to
the audit was the installed driver: 580.95.05 on node 1 and 595.71.05 on node
2.  `requirements-core.txt` records the portable core Python lock; the raw
node captures and their hashes are recorded in `environment.json`.

## Frozen execution commands

The exact commands are in `commands.md`.  Model directories are not bundled.
Each config pins repository/model revisions, artifact byte sizes, and SHA-256
hashes, and every runner verifies the model artifact before scoring.

This package does not claim that the paper is submission-ready.  It closes the
local cross-node/environment packaging gap while broader generative-task FLOP
accounting, overwrite redesign/boundary analysis, and additional adversarial
coverage remain open.

The frozen self-contained release built from this package is
`artifacts/remote_results/ascent_core_reproduction_ffe6c78.tar.gz` with
SHA-256 `b67d521e7c5e6153395d35225b168ba6d5b9babffa227f279c76727da271069c`.
