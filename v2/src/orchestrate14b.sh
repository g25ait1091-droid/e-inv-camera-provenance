#!/usr/bin/env bash
# Chain 14b (Entry 87) - lane 2 resumed: G5's generation died of CUDA OOM after one adapter of six.
# The generate stage skips any adapter already complete, so this finishes the remaining five at the
# primary 500 images, the budget alt_A_s0 already used. Only then is G5 measured.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
ADMIT="$SRC"/gpu_admit.sh
until [ -f out/t1/train_png/altA_a0 ]; do sleep 120; done
echo "[orch14b] $(date) start"
bash $ADMIT 31000 env T1_ARMSET=alt python -u "$SRC"/t1_ladder.py generate > logs/t1_alt_generate.log 2>&1
echo "[orch14b] $(date) alt generated"
CUDA_VISIBLE_DEVICES= T1_ARMSET=alt T1_W=3 python -u "$SRC"/t1_measure.py > logs/t1_alt_measure.log 2>&1
echo "[orch14b] $(date) chain done"
