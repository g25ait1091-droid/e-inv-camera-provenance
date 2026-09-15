#!/usr/bin/env bash
# Waits for the Tier 1 train+generate chain to finish, then:
#   (CPU)  t1_measure.py  -- registered Tier 1 statistics
#   (GPU)  t5_prompt.py   -- E-PROMPT generate + measure   (Entry 11)
#   (GPU)  t2_learned.py  -- learned detector             (Entry 09)
# Everything logs under logs/; each step is resumable and idempotent.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch] $(date) waiting for Tier 1 generation to finish"
while ! grep -aq "done generate" logs/t1_generate.log 2>/dev/null; do sleep 120; done
echo "[orch] $(date) Tier 1 generation complete"
( T1_W=6 python -u "$SRC"/t1_measure.py > logs/t1_measure.log 2>&1; echo "[orch] $(date) t1_measure finished" ) &
python -u "$SRC"/t5_prompt.py generate > logs/t5_generate.log 2>&1
echo "[orch] $(date) t5 generate finished"
python -u "$SRC"/t2_learned.py > logs/t2_learned.log 2>&1
echo "[orch] $(date) t2_learned finished"
wait
python -u "$SRC"/t5_prompt.py measure > logs/t5_measure.log 2>&1
echo "[orch] $(date) t5 measure finished; all queued work done"
