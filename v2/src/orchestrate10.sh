#!/usr/bin/env bash
# Chain 10 (Entry 55): second 16000-step replication, three more adapters per body (seeds 3-5).
# Waits for chain 9 to finish, then trains, generates (250 each), measures, and re-runs the dose
# statistics (out/t1/dose_stats.json: rows "16000_new" = seeds 3-5, the pre-specified primary reading,
# and "16000" = seeds 0-5 pooled) and the F8 memorization test over every v2 adapter.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
until grep -q "chain done" logs/orchestrate9.log; do sleep 300; done
echo "[orch10] $(date) start"
T1_ARMSET=dose16krep2 python -u "$SRC"/t1_ladder.py train > logs/t1_dose16krep2_train.log 2>&1
echo "[orch10] $(date) training finished"
T1_ARMSET=dose16krep2 python -u "$SRC"/t1_ladder.py generate > logs/t1_dose16krep2_generate.log 2>&1
echo "[orch10] $(date) generation finished"
T1_ARMSET=dose16krep2 T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_dose16krep2_measure.log 2>&1
python -u "$SRC"/dose_stats.py > logs/dose_stats_rep2.log 2>&1
python -u "$SRC"/f8_memorization.py > logs/f8_memorization_rep2.log 2>&1
echo "[orch10] $(date) chain done"
