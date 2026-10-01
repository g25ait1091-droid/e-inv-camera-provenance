#!/usr/bin/env bash
# Chain 15 (Entries 80, 84, 85, 86, 87) - lane 3: the iPhone 5c pair. GPU stages go through
# gpu_admit.sh so this lane cannot start into memory another lane is about to take; the measurement
# is CPU work and runs with the card hidden.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
ADMIT="$SRC"/gpu_admit.sh
until [ -f out/fp_5c/manifest.json ]; do sleep 120; done
echo "[orch15] $(date) start"
bash $ADMIT 14000 env T1_ARMSET=p5c T1_GENS=250 python -u "$SRC"/t1_ladder.py train    > logs/t1_p5c_train.log 2>&1
echo "[orch15] $(date) p5c trained"
bash $ADMIT 31000 env T1_ARMSET=p5c T1_GENS=250 python -u "$SRC"/t1_ladder.py generate > logs/t1_p5c_generate.log 2>&1
echo "[orch15] $(date) p5c generated"
CUDA_VISIBLE_DEVICES= G6_W=4 python -u "$SRC"/g6_measure.py > logs/g6_measure.log 2>&1
echo "[orch15] $(date) chain done"
