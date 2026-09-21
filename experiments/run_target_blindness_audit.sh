#!/bin/bash
# Phase 2B — instrumented target-blindness rerun (Array revision 1).
# Usage (GPU host, repository root, clean worktree at the commit holding the instrumentation):
#   bash experiments/run_target_blindness_audit.sh /path/to/models /path/to/output_root
set -euo pipefail
MODELS="${1:?model root}"; RUN="${2:?output root outside the repository}"
PREREG=artifacts_revision/target_blindness_2026-09/PREREGISTRATION.json
case "$RUN" in "$PWD"/*) echo "output root must be outside the repository" >&2; exit 2;; esac
[ -n "$(git status --porcelain)" ] && { echo "refusing to run: worktree is dirty" >&2; git status --short >&2; exit 2; }
mkdir -p "$RUN/logs"
echo "commit $(git rev-parse HEAD)  $(date -u +%FT%TZ)"
python experiments/record_run_environment.py --run-dir "$RUN" --preregistration "$PREREG" --phase before
run_cell() { # study config data_dir stem ep panel
  local study=$1 config=$2 ddir=$3 stem=$4 ep=$5 n=$6
  local out="$RUN/$study/$ep/${stem}${n}.json"
  [ -e "$out" ] && { echo "refusing to overwrite $out" >&2; exit 3; }
  mkdir -p "$(dirname "$out")"
  python experiments/run_babilong_prompt.py --config "$config" --data "$ddir/${stem}${n}.jsonl" \
    --endpoint "qwen2p5-${ep}-instruct" --model-dir "$MODELS/qwen2p5-${ep}-instruct" --condition canonical_coscale \
    --audit-rerun --audit-label posthoc_target_blindness_instrumented_rerun_no_independence_claim \
    --output "$out" > "$RUN/logs/${study}_panel_${n}_${ep}.log" 2>&1
  echo "done $study panel $n $ep $(date -u +%FT%TZ)"
}
for n in 1 2 3 4 5 6 7 8 9 10; do for ep in 0p5b 1p5b 3b; do
  run_cell official16k configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json data/babilong_16k_canonical_confirmation canonical_16k_confirmation_panel_ $ep $n
done; done
for n in 1 2 3 4 5 6 7 8 9 10; do for ep in 0p5b 1p5b 3b; do
  run_cell semantic_holdout configs/babilong_qwen2p5_canonical_semantic_holdout_8k_confirmatory.json data/babilong_train_8k_semantic_holdout_confirmation canonical_train_8k_semantic_panel_ $ep $n
done; done
python experiments/record_run_environment.py --run-dir "$RUN" --preregistration "$PREREG" --phase after
( cd "$RUN" && find . -type f ! -name SHA256SUMS.remote.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.remote.txt )
echo "DONE $(date -u +%FT%TZ)"
