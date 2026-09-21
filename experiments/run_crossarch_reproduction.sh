#!/bin/bash
# Phase 2A — cross-architecture reproduction audit (Array revision 1).
# Executes exactly the cells listed in artifacts_revision/crossnode_2026-09/PREREGISTRATION.json,
# from a clean git worktree, recording an environment snapshot before and after.
# Usage (on the GPU host, from the repository root):
#   bash experiments/run_crossarch_reproduction.sh /path/to/models /path/to/output_root
# The output root must lie OUTSIDE the repository so that result files do not
# make the worktree dirty while later cells run; the results are moved into
# artifacts_revision/crossnode_2026-09/ on the local machine afterwards.
set -euo pipefail
MODELS="${1:?model root}"
RUN="${2:?output root outside the repository}"
PREREG=artifacts_revision/crossnode_2026-09/PREREGISTRATION.json
case "$RUN" in "$PWD"/*) echo "output root must be outside the repository" >&2; exit 2;; esac
mkdir -p "$RUN/logs" "$RUN/factorial" "$RUN/official16k/canonical/0p5b" "$RUN/official16k/canonical/1p5b" "$RUN/official16k/canonical/3b"

if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to run: worktree is dirty" >&2; git status --short >&2; exit 2
fi
echo "commit $(git rev-parse HEAD)  $(date -u +%FT%TZ)"
python experiments/record_run_environment.py --run-dir "$RUN" --preregistration "$PREREG" --phase before

# Group 1: factorial confirmation panels, Qwen2.5-7B, K=5
for P in 2 3; do
  OUT="$RUN/factorial/rounds_5_sum_numeric_candidate_panel_${P}_qwen2p5-7b-instruct.json"
  [ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 3; }
  python experiments/run_noisy_composition_candidate.py \
    --config configs/posttraining_noisy_composition_sum_numeric_candidate.json \
    --data-root data \
    --panel "sum_numeric_candidate_panel_${P}" \
    --endpoint qwen2p5-7b-instruct \
    --model-dir "$MODELS/qwen2p5-7b-instruct" \
    --rounds 5 \
    --audit-rerun \
    --output "$OUT" 2>&1 | tee "$RUN/logs/factorial_panel_${P}.log"
done

# Group 2: official 16K BABILong confirmation panel 1, both arms, co-scaled readers
for EP in 0p5b 1p5b 3b; do
  OUT="$RUN/official16k/canonical/${EP}/canonical_16k_confirmation_panel_1.json"
  [ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 3; }
  python experiments/run_babilong_prompt.py \
    --config configs/babilong_qwen2p5_canonical_coscale_16k_confirmatory.json \
    --data data/babilong_16k_canonical_confirmation/canonical_16k_confirmation_panel_1.jsonl \
    --endpoint "qwen2p5-${EP}-instruct" \
    --model-dir "$MODELS/qwen2p5-${EP}-instruct" \
    --condition canonical_coscale \
    --audit-rerun \
    --audit-label posthoc_cross_architecture_audit_rerun_no_independence_claim \
    --output "$OUT" 2>&1 | tee "$RUN/logs/official16k_panel_1_${EP}.log"
done

python experiments/record_run_environment.py --run-dir "$RUN" --preregistration "$PREREG" --phase after
( cd "$RUN" && find . -type f ! -name SHA256SUMS.remote.txt -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.remote.txt )
echo "DONE $(date -u +%FT%TZ)"
