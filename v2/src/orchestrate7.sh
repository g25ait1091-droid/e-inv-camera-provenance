#!/usr/bin/env bash
# Seventh chain (CPU + light GPU): Entry 38 — five-body learned attribution (archived generations
# already local), then the PRNU-projected ablation once the seed 3-11 re-fetch has finished.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
echo "[orch7] $(date) start"
python -u "$SRC"/t2_learned_5body.py kodak > logs/t2_5body_kodak.log 2>&1
echo "[orch7] $(date) kodak finished"
python -u "$SRC"/t2_learned_5body.py p20 > logs/t2_5body_p20.log 2>&1
echo "[orch7] $(date) p20 finished"
while ! grep -aq "\[fetch_ext\] done" logs/fetch_ext2.log 2>/dev/null; do sleep 120; done
python -u "$SRC"/t2_learned_noprnu.py > logs/t2_learned_noprnu.log 2>&1
echo "[orch7] $(date) noprnu finished; chain done"
