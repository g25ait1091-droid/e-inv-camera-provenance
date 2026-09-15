#!/usr/bin/env bash
# Fifth chain (Entry 35, F7): unmarked B-body adapters in the v2 environment, then the symmetric statistic.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch5] $(date) start"
T1_ARMSET=nomarkB python -u "$SRC"/t1_ladder.py materialise > logs/t1_nomarkB_materialise.log 2>&1
T1_ARMSET=nomarkB python -u "$SRC"/t1_ladder.py train > logs/t1_nomarkB_train.log 2>&1
echo "[orch5] $(date) nomarkB training finished"
T1_ARMSET=nomarkB python -u "$SRC"/t1_ladder.py generate > logs/t1_nomarkB_generate.log 2>&1
echo "[orch5] $(date) nomarkB generation finished"
T1_ARMSET=nomarkB T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_nomarkB_measure.log 2>&1
python -u "$SRC"/f7_stats.py > logs/f7_stats.log 2>&1
echo "[orch5] $(date) F7 finished; chain done"
