#!/usr/bin/env bash
# Chain 8 (Entry 48 F9): two more 16000-step adapters per body, then measurement, dose statistics, F8 re-run.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch8] $(date) start"
T1_ARMSET=dose16krep python -u "$SRC"/t1_ladder.py train > logs/t1_dose16krep_train.log 2>&1
echo "[orch8] $(date) training finished"
T1_ARMSET=dose16krep python -u "$SRC"/t1_ladder.py generate > logs/t1_dose16krep_generate.log 2>&1
echo "[orch8] $(date) generation finished"
T1_ARMSET=dose16krep T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_dose16krep_measure.log 2>&1
python -u "$SRC"/dose_stats.py > logs/dose_stats_final.log 2>&1
python -u "$SRC"/f8_memorization.py > logs/f8_memorization_final.log 2>&1
echo "[orch8] $(date) chain done"
