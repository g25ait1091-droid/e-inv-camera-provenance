#!/usr/bin/env bash
# Chain 13 (Entry 78): waits for chain 12, then G5 (a second training set for the D200 pair, six adapters)
# and G4 (the Huawei P10 Plus pair, twenty-four adapters). Each study keeps its own crops, adapters,
# generations, fingerprints and result files; G4 is measured only against the P10 Plus fingerprints.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
until grep -q "chain done" logs/orchestrate12.log 2>/dev/null; do sleep 300; done
echo "[orch13] $(date) start"
run_set () {
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py train    > logs/t1_$1_train.log 2>&1
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py generate > logs/t1_$1_generate.log 2>&1
  echo "[orch13] $(date) $1 generated"
}
[ -d out/t1/train_png/altA_a0 ] || { echo "[orch13] G5 crops missing"; exit 1; }
[ -d out/t1/train_png/p10A_a0 ] || { echo "[orch13] G4 crops missing; run src/g4_prep.py"; exit 1; }
run_set alt
T1_ARMSET=alt T1_W=4 python -u "$SRC"/t1_measure.py > logs/t1_alt_measure.log 2>&1 &
run_set p10
wait
G4_W=6 python -u "$SRC"/g4_measure.py > logs/g4_measure.log 2>&1
echo "[orch13] $(date) chain done"
