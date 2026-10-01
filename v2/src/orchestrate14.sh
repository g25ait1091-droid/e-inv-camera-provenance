#!/usr/bin/env bash
# Chain 14 (Entry 78) - the second GPU lane, run concurrently with chain 12's G1 arms. Training takes
# about 16 GB of the 46 GB card, so two lanes fit; each writes its own adapters, generations and results.
# G5 (a second training set for the D200 pair) first because it is cheap, then G4 (the P10 Plus pair).
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
echo "[orch14] $(date) start (lane 2, concurrent with chain 12)"
run_set () {
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py train    > logs/t1_$1_train.log 2>&1
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py generate > logs/t1_$1_generate.log 2>&1
  echo "[orch14] $(date) $1 generated"
}
run_set alt
T1_ARMSET=alt T1_W=3 python -u "$SRC"/t1_measure.py > logs/t1_alt_measure.log 2>&1 &
run_set p10
wait
G4_W=4 python -u "$SRC"/g4_measure.py > logs/g4_measure.log 2>&1
echo "[orch14] $(date) chain done"
