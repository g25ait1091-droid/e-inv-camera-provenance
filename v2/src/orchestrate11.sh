#!/usr/bin/env bash
# Chain 11 (Entry 59): waits for chain 10, then trains and generates E1 (kinj), E3 (cm), E2 (gk), E4 (periodic3)
# in that order. Each set is measured in the background while the next one trains. Training crops are
# materialised beforehand (t1_kfield.py fields/materialise, t1_content_match.py).
#
# The generations are tens of GB. Set EINV_GENS to a directory on a disk with room and this script links
# out/t1/gens to it; the default keeps them in place, under the working root. (Here EINV_GENS pointed at a
# second disk and out/t1/gens was a Windows directory junction, created beforehand with mklink /J.)
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
GENS_ROOT="${EINV_GENS:-$PWD/out/t1/gens}"
mkdir -p "$GENS_ROOT" out/t1
if [ ! -e out/t1/gens ]; then
  ln -s "$GENS_ROOT" out/t1/gens || {
    echo "[orch11] could not link out/t1/gens -> $GENS_ROOT; create it yourself (Windows: mklink /J)"; exit 1; }
fi
until grep -q "chain done" logs/orchestrate10.log; do sleep 300; done
echo "[orch11] $(date) start"
run_set () {
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py train > logs/t1_$1_train.log 2>&1
  T1_ARMSET=$1 python -u "$SRC"/t1_ladder.py generate > logs/t1_$1_generate.log 2>&1
  echo "[orch11] $(date) $1 generated"
}
run_set kinj
T1_W=4 python -u "$SRC"/t1_kfield.py measure > logs/t1_kfield_measure_1.log 2>&1 &
run_set cm
( T1_ARMSET=cm T1_W=4 python -u "$SRC"/t1_measure.py > logs/t1_cm_measure.log 2>&1
  python -u "$SRC"/t2_learned_arms.py --A out/t1/gens/cm_A_s0 out/t1/gens/cm_A_s1 out/t1/gens/cm_A_s2 \
         --B out/t1/gens/cm_B_s0 out/t1/gens/cm_B_s1 out/t1/gens/cm_B_s2 --out out/t2_learned_arms_cm.json > logs/t2_learned_arms_cm.log 2>&1 ) &
run_set gk
T1_W=4 python -u "$SRC"/t1_kfield.py measure > logs/t1_kfield_measure_2.log 2>&1 &
run_set periodic3
wait
[ -f out/t1/periodic2_summary_entry51.json ] || cp out/t1/periodic2_summary.json out/t1/periodic2_summary_entry51.json
[ -f out/t1/periodic2_rows_entry51.npz ] || cp out/t1/periodic2_rows.npz out/t1/periodic2_rows_entry51.npz
T1_W=6 python -u "$SRC"/t1_periodic2.py measure > logs/t1_periodic3_measure.log 2>&1
T1_W=6 python -u "$SRC"/t1_kfield.py measure > logs/t1_kfield_measure_final.log 2>&1
echo "[orch11] $(date) chain done"
