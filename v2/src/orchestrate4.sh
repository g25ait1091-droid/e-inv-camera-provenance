#!/usr/bin/env bash
# Fourth chain (Entry 31, F6): v2 stack on v1's Colab-decoded training crops. Waits for chain 3.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch4] $(date) waiting for chain 3"
while ! grep -aq "chain done" logs/orchestrate3.log 2>/dev/null; do sleep 180; done
echo "[orch4] $(date) chain 3 done; starting colab arms"
T1_ARMSET=colab python -u "$SRC"/t1_ladder.py train > logs/t1_colab_train.log 2>&1
echo "[orch4] $(date) colab training finished"
T1_ARMSET=colab python -u "$SRC"/t1_ladder.py generate > logs/t1_colab_generate.log 2>&1
echo "[orch4] $(date) colab generation finished"
T1_ARMSET=colab T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_colab_measure.log 2>&1
echo "[orch4] $(date) colab measure finished; chain done"
