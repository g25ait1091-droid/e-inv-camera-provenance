#!/usr/bin/env bash
# Chain 9 (Entry 51, review follow-up): waits for chain 8 (F9) to finish, then trains and generates the
# tile replicates and grid-separating tiles (periodic2) and the finest-band replicates (band2), measures
# both, and writes periodic2_summary.json and band2_summary.json. Fields and crops are materialised
# beforehand (t1_periodic2.py fields / materialise), outside this chain.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
until grep -q "chain done" logs/orchestrate8.log; do sleep 300; done
echo "[orch9] $(date) start"
python -u "$SRC"/t1_periodic2.py vae > logs/t1_periodic2_vae.log 2>&1
T1_ARMSET=periodic2 python -u "$SRC"/t1_ladder.py train > logs/t1_periodic2_train.log 2>&1
T1_ARMSET=periodic2 python -u "$SRC"/t1_ladder.py generate > logs/t1_periodic2_generate.log 2>&1
echo "[orch9] $(date) periodic2 generated"
T1_ARMSET=band2 python -u "$SRC"/t1_ladder.py train > logs/t1_band2_train.log 2>&1
T1_ARMSET=band2 python -u "$SRC"/t1_ladder.py generate > logs/t1_band2_generate.log 2>&1
echo "[orch9] $(date) band2 generated"
python -u "$SRC"/t1_periodic2.py measure > logs/t1_periodic2_measure.log 2>&1
python -u "$SRC"/t1_band2.py > logs/t1_band2_measure.log 2>&1
echo "[orch9] $(date) chain done"
