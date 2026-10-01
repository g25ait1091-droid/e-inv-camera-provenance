#!/usr/bin/env bash
# Chain 12 (Entry 76): G2 first (six 2000-step unmarked adapters in the v2 environment, ~6 h), then
# G1 (six 16000-step adapters on fingerprint-suppressed crops, ~34 h). Each set is measured as soon as
# its generations are done, while the next set trains. Suppressed crops come from src/g1_suppress.py.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
echo "[orch12] $(date) start"
run_set () {
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py train    > logs/t1_$1_train.log 2>&1
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py generate > logs/t1_$1_generate.log 2>&1
  echo "[orch12] $(date) $1 generated"
}
[ -d out/t1/train_png/invA_a0 ] || { echo "[orch12] suppressed crops missing; run src/g1_suppress.py"; exit 1; }
run_set nomarkrep
T1_ARMSET=nomarkrep T1_W=4 python -u "$SRC"/t1_measure.py > logs/t1_nomarkrep_measure.log 2>&1 &
run_set inv16k
wait
T1_ARMSET=inv16k T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_inv16k_measure.log 2>&1
python -u "$SRC"/g_analyze.py > logs/g_analyze.log 2>&1
echo "[orch12] $(date) chain done"
