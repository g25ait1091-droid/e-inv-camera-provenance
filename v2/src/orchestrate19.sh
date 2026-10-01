#!/usr/bin/env bash
# Chain 19 (Entry 99) - G1 extended to eight adapters per arm, queued behind the chain-17 queue so it does
# not starve G6 and G4b. Ten adapters at 16000 steps, about 190 GPU-hours. Every stage resumes.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
ADMIT="$SRC"/gpu_admit2.sh
# wait for the G6/G4b queue to finish before taking the card
until grep -q "queue done" logs/orchestrate17.log 2>/dev/null; do sleep 600; done
echo "[orch19] $(date +%H:%M) chain 17 finished; starting the G1 extension"
bash $ADMIT 12000 env T1_ARMSET=inv16kext python -u "$SRC"/t1_ladder.py train    > logs/t1_inv16kext_train.log 2>&1
echo "[orch19] $(date +%H:%M) trained"
bash $ADMIT 30000 env T1_ARMSET=inv16kext python -u "$SRC"/t1_ladder.py generate > logs/t1_inv16kext_generate.log 2>&1
echo "[orch19] $(date +%H:%M) generated"
CUDA_VISIBLE_DEVICES= T1_ARMSET=inv16kext T1_W=3 python -u "$SRC"/t1_measure.py > logs/t1_inv16kext_measure.log 2>&1
echo "[orch19] $(date +%H:%M) measured"
