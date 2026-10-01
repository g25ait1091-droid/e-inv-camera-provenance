#!/usr/bin/env bash
# Chain 18 (Entries 88, 93) - CPU lane: re-runs H4 with the training-set component pinned, as soon as G5's
# six adapters are measured. Until then H4's training rows are a sensitivity sweep; this replaces them with
# a measurement without anyone having to remember to do it. Costs the GPU queue nothing.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
until [ -f out/t1/summary_alt.json ]; do sleep 300; done
# the measure writes the file once all six arms are in; guard against reading it half-written
sleep 60
echo "[orch18] $(date +%H:%M) G5 measured; re-running H4 with the training-set component"
python -u "$SRC"/h4_coverage2.py > logs/h4_with_g5.log 2>&1
python -u "$SRC"/findings.py > logs/findings.log 2>&1
echo "[orch18] $(date +%H:%M) done"
