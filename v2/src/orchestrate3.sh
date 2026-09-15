#!/usr/bin/env bash
# Third chain (Entries 26-27). Waits for the F5 measurement, then:
#   wm: materialise -> train (3) -> generate (3)      DiffusionShield arm
#   rcrop: train (3) -> generate (3)                   random-crop arm
#   measure both; stats.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch3] $(date) waiting for F5 measurement"
while [ ! -f out/t1/summary_f5.json ]; do sleep 120; done
echo "[orch3] $(date) F5 done; starting watermark arm"
T1_ARMSET=wm python -u "$SRC"/t1_ladder.py materialise > logs/t1_wm_materialise.log 2>&1
T1_ARMSET=wm python -u "$SRC"/t1_ladder.py train > logs/t1_wm_train.log 2>&1
echo "[orch3] $(date) wm training finished"
T1_ARMSET=wm python -u "$SRC"/t1_ladder.py generate > logs/t1_wm_generate.log 2>&1
echo "[orch3] $(date) wm generation finished"
( T1_W=6 python -u "$SRC"/t1_wm_measure.py > logs/t1_wm_measure.log 2>&1; echo "[orch3] $(date) wm measure finished" ) &
T1_ARMSET=rcrop python -u "$SRC"/t1_ladder.py train > logs/t1_rcrop_train.log 2>&1
echo "[orch3] $(date) rcrop training finished"
T1_ARMSET=rcrop python -u "$SRC"/t1_ladder.py generate > logs/t1_rcrop_generate.log 2>&1
echo "[orch3] $(date) rcrop generation finished"
wait
T1_ARMSET=rcrop T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_rcrop_measure.log 2>&1
echo "[orch3] $(date) rcrop measure finished; chain done"
