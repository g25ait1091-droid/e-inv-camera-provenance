#!/usr/bin/env bash
# Sixth chain (GPU): Entry 37 band transfer function (A1 then A2), then Entry 39 dose arms.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch6] $(date) start"
python -u "$SRC"/t1_band.py fields > logs/band_fields.log 2>&1
python -u "$SRC"/t1_band.py materialise > logs/band_materialise.log 2>&1
python -u "$SRC"/t1_band.py vae > logs/band_vae.log 2>&1
echo "[orch6] $(date) A1 (autoencoder) done"
T1_ARMSET=band python -u "$SRC"/t1_ladder.py train > logs/t1_band_train.log 2>&1
echo "[orch6] $(date) band training finished"
T1_ARMSET=band python -u "$SRC"/t1_ladder.py generate > logs/t1_band_generate.log 2>&1
echo "[orch6] $(date) band generation finished"
( python -u "$SRC"/t1_band.py measure > logs/band_measure.log 2>&1; echo "[orch6] $(date) band measure finished" ) &
T1_ARMSET=dose8k python -u "$SRC"/t1_ladder.py train > logs/t1_dose8k_train.log 2>&1
T1_ARMSET=dose8k python -u "$SRC"/t1_ladder.py generate > logs/t1_dose8k_generate.log 2>&1
echo "[orch6] $(date) dose8k finished"
T1_ARMSET=dose8k T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_dose8k_measure.log 2>&1
python -u "$SRC"/dose_stats.py > logs/dose_stats_8k.log 2>&1
echo "[orch6] $(date) dose8k measured (interim stats)"
T1_ARMSET=dose16k python -u "$SRC"/t1_ladder.py train > logs/t1_dose16k_train.log 2>&1
T1_ARMSET=dose16k python -u "$SRC"/t1_ladder.py generate > logs/t1_dose16k_generate.log 2>&1
wait
T1_ARMSET=dose16k T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_dose16k_measure.log 2>&1
python -u "$SRC"/dose_stats.py > logs/dose_stats.log 2>&1
echo "[orch6] $(date) chain done"
