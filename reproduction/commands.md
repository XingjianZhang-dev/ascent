# Frozen commands

All commands run from the repository root.  Replace only the absolute model
and result-root prefixes; artifact contents must match the hashes in the
configs.

## BABILong 8K scale-relevant confirmation arm

```bash
python experiments/run_babilong_prompt.py \
  --config configs/babilong_smollm2_8k_neural_controls_confirmatory.json \
  --data data/babilong_1k_8k_neural_control/neural_control_panel_1.jsonl \
  --endpoint smollm2-360m-instruct \
  --model-dir /ABSOLUTE/MODEL/smollm2-360m-instruct \
  --condition scale_relevant \
  --output /ABSOLUTE/RESULT/panel1_smollm2-360m-instruct_scale_relevant.json
```

Repeat for panels 2 and 3 and for each endpoint.  The same frozen runner is
used for `scale_matched_raw` and `scale_bge_m3`; BGE runs additionally require
the hash-bound caches named by the config.

## Same-FLOP static replay confirmation

```bash
python experiments/run_latent_replay_factorial.py \
  --config configs/latent_replay_same_flop_static_confirmatory.json \
  --endpoint pythia-410m \
  --model-dir /ABSOLUTE/MODEL/pythia-410m \
  --data-seed 20260941 \
  --output /ABSOLUTE/RESULT/seed20260941_pythia-410m.json
```

Repeat for seeds 20260951 and 20260961 and endpoint `pythia-2.8b`, then run
`experiments/analyze_static_replay_panel.py` on the six JSON results.

## Cross-node replication

```bash
python experiments/run_babilong_prompt.py \
  --config configs/babilong_smollm2_8k_cross_node_replication.json \
  --data data/babilong_1k_8k_neural_control/neural_control_panel_1.jsonl \
  --endpoint smollm2-360m-instruct \
  --model-dir /ABSOLUTE/MODEL/smollm2-360m-instruct \
  --condition scale_relevant \
  --output /ABSOLUTE/RESULT/panel1_smollm2-360m-instruct_scale_relevant.json
```

Use `experiments/analyze_cross_node_replication.py` to compare the three new
files with the archived reference.

## Latency-tail repeat

Run three fresh processes per `--decode-order foundation_first` and
`--decode-order ascent_first` with
`configs/babilong_smollm2_4k_latency_tail_audit.json`, then use
`experiments/analyze_latency_tail_audit.py`.  The original 1.075 ms result must
remain in the analysis.
