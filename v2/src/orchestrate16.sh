#!/usr/bin/env bash
# Chain 16 (Entries 85, 86, 87) - lane 4: the modern-smartphone paired design retried on the Daxing
# P20 pair after G4 closed at its fingerprint gate. Same admission control as lane 3.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
ADMIT="$SRC"/gpu_admit.sh
until [ -f out/fp_p20b/manifest.json ]; do sleep 120; done
echo "[orch16] $(date) start"
bash $ADMIT 14000 env T1_ARMSET=p20b T1_GENS=250 python -u "$SRC"/t1_ladder.py train    > logs/t1_p20b_train.log 2>&1
echo "[orch16] $(date) p20b trained"
bash $ADMIT 31000 env T1_ARMSET=p20b T1_GENS=250 python -u "$SRC"/t1_ladder.py generate > logs/t1_p20b_generate.log 2>&1
echo "[orch16] $(date) p20b generated"
CUDA_VISIBLE_DEVICES= G4B_W=4 python -u "$SRC"/g4b_measure.py > logs/g4b_measure.log 2>&1
echo "[orch16] $(date) chain done"
