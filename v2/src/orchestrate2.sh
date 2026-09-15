#!/usr/bin/env bash
# Second chain (Entry 18 F2, Entry 20 F4). Waits for E-PROMPT generation to finish, then:
#   (GPU) t2_learned_f3.py prompt       -- learned detector on the prompt-bank generations (F2)
#   (GPU) T1_ARMSET=nomark t1_ladder.py materialise/train/generate -- three unmarked control arms (F4)
#   (CPU) T1_ARMSET=nomark t1_measure.py ; f4_stats.py
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch2] $(date) waiting for E-PROMPT generation"
while ! grep -aq "t5 generate finished" logs/orchestrate.log 2>/dev/null; do sleep 120; done
echo "[orch2] $(date) E-PROMPT generation complete"
python -u "$SRC"/t2_learned_f3.py prompt > logs/t2_learned_f2.log 2>&1
echo "[orch2] $(date) F2 finished"
T1_ARMSET=nomark python -u "$SRC"/t1_ladder.py materialise > logs/t1_nomark_materialise.log 2>&1
T1_ARMSET=nomark python -u "$SRC"/t1_ladder.py train > logs/t1_nomark_train.log 2>&1
echo "[orch2] $(date) nomark training finished"
T1_ARMSET=nomark python -u "$SRC"/t1_ladder.py generate > logs/t1_nomark_generate.log 2>&1
echo "[orch2] $(date) nomark generation finished"
T1_ARMSET=nomark T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_nomark_measure.log 2>&1
python -u "$SRC"/f4_stats.py > logs/f4_stats.log 2>&1
echo "[orch2] $(date) F4 finished; chain done"
