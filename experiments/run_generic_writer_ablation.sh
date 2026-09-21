#!/bin/bash
# Phase 2C — schema-blind writer ablation (Array revision 1).
# Stage 1 builds the BGE-M3 retrieval caches for the ten official-16K panels (GPU, no reader scores).
# Stage 2 fills the cache hashes into the ablation config (mechanical), records the environment, and
# runs the three generic writers x three readers x ten panels with the unchanged runner.
# Usage (GPU host, repository root, clean worktree):
#   bash experiments/run_generic_writer_ablation.sh /path/to/models /path/to/output_root [caches|readers|all]
set -euo pipefail
MODELS="${1:?model root}"; RUN="${2:?output root outside the repository}"; STAGE="${3:-all}"
CFGDIR=artifacts_revision/schema_blind_2026-09/config
PREREG=artifacts_revision/schema_blind_2026-09/PREREGISTRATION.json
case "$RUN" in "$PWD"/*) echo "output root must be outside the repository" >&2; exit 2;; esac
[ -n "$(git status --porcelain)" ] && { echo "refusing to run: worktree is dirty" >&2; git status --short >&2; exit 2; }
mkdir -p "$RUN/logs" "$RUN/bge_cache"
echo "commit $(git rev-parse HEAD)  $(date -u +%FT%TZ)"
PANELS="1 2 3 4 5 6 7 8 9 10"

if [ "$STAGE" = "caches" ] || [ "$STAGE" = "all" ]; then
  python experiments/record_run_environment.py --run-dir "$RUN" --preregistration "$PREREG" --phase before
  for n in $PANELS; do
    OUT="$RUN/bge_cache/canonical_16k_confirmation_panel_${n}.json"
    [ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 3; }
    python experiments/prepare_babilong_bge_cache.py --protocol "$CFGDIR/babilong_bge_m3_retrieval_protocol_16k.json" \
      --data "data/babilong_16k_canonical_confirmation/canonical_16k_confirmation_panel_${n}.jsonl" \
      --embedding-model-dir "$MODELS/bge-m3" --reranker-model-dir "$MODELS/bge-reranker-v2-m3" \
      --output "$OUT" > "$RUN/logs/bge_cache_panel_${n}.log" 2>&1
    echo "cache panel $n $(date -u +%FT%TZ)"
  done
fi

if [ "$STAGE" = "readers" ] || [ "$STAGE" = "all" ]; then
  # Fill the cache paths and hashes into a run copy of the config (the committed config keeps the empty map).
  CFG="$RUN/babilong_qwen2p5_generic_writer_16k_ablation.run.json"
  python - "$CFGDIR/babilong_qwen2p5_generic_writer_16k_ablation.json" "$RUN" "$CFG" <<'PY'
import hashlib, json, sys
src, run, dst = sys.argv[1:4]
cfg = json.load(open(src))
caches = {}
for n in range(1, 11):
    p = f"{run}/bge_cache/canonical_16k_confirmation_panel_{n}.json"
    caches[f"canonical_16k_confirmation_panel_{n}"] = {"path": p, "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
cfg["conditions"]["bge_m3_coscale"]["retrieval_cache_by_panel"] = caches
cfg["committed_config_sha256_before_cache_fill"] = hashlib.sha256(open(src, "rb").read()).hexdigest()
json.dump(cfg, open(dst, "w"), indent=2); open(dst, "a").write("\n")
print("run config written", dst)
PY
  for COND in sentence_window_coscale generic_bm25_coscale bge_m3_coscale; do
    for n in $PANELS; do for EP in 0p5b 1p5b 3b; do
      OUT="$RUN/$COND/qwen2p5-${EP}-instruct/canonical_16k_confirmation_panel_${n}.json"
      [ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 3; }
      mkdir -p "$(dirname "$OUT")"
      python experiments/run_babilong_prompt.py --config "$CFG" \
        --data "data/babilong_16k_canonical_confirmation/canonical_16k_confirmation_panel_${n}.jsonl" \
        --endpoint "qwen2p5-${EP}-instruct" --model-dir "$MODELS/qwen2p5-${EP}-instruct" --condition "$COND" \
        --output "$OUT" > "$RUN/logs/${COND}_panel_${n}_${EP}.log" 2>&1
      echo "done $COND panel $n $EP $(date -u +%FT%TZ)"
    done; done
  done
  python experiments/record_run_environment.py --run-dir "$RUN" --preregistration "$PREREG" --phase after
  ( cd "$RUN" && find . -type f ! -name SHA256SUMS.remote.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.remote.txt )
  echo "DONE $(date -u +%FT%TZ)"
fi
