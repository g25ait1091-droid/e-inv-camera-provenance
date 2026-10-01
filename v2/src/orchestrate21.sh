#!/usr/bin/env bash
# Chain 21 (Entries 99, 106) - after the G1 extension: score it, then P1 (diverse prompts on the G6 adapters).
# Started by Task Scheduler through restart_queue.sh (D13), never from the app's shell. Every stage's exit code
# and image count is checked (D12). Output folders for P1 are junctions to C: (D12).
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
say () { echo "[orch21] $(date +%H:%M) $*"; }
say "waiting for chain 19 (G1 extension) to finish"
until grep -a "^\[orch19\]" logs/orchestrate19.log | tail -1 | grep -q "measured$"; do sleep 300; done

if [ ! -f out/t1/summary_inv16kext.json ]; then
  # chain 19's measurement ran with an armset t1_measure.py did not know and scored other arms (D14)
  say "measuring the G1 extension (2,500 images, CPU)"
  if ! CUDA_VISIBLE_DEVICES= T1_ARMSET=inv16kext T1_W=4 python -u "$SRC"/t1_measure.py > logs/t1_inv16kext_measure.log 2>&1 \
     || [ ! -f out/t1/summary_inv16kext.json ]; then
    say "G1-ext measurement FAILED - see logs/t1_inv16kext_measure.log"; exit 1
  fi
fi
say "scoring the G1 extension"
if ! python -u "$SRC"/g1_ext_score.py > logs/g1_ext_score.log 2>&1; then say "G1-ext scoring FAILED - see logs/g1_ext_score.log"; exit 1; fi
say "G1-ext scored: $(grep READING logs/g1_ext_score.log)"

say "P1: diverse-prompt generation on the 24 G6 adapters"
if ! bash "$SRC"/gpu_admit2.sh 30000 env T1_ARMSET=p5c T1_GENS=250 T1_PROMPTBANK=diverse T1_GEN_SUFFIX=_div python -u "$SRC"/t1_ladder.py generate >> logs/t1_p5c_div_generate.log 2>&1; then
  say "P1 generation FAILED - see logs/t1_p5c_div_generate.log"; exit 1
fi
n=$(ls out/t1/gens/p5c_*_div/*.png 2>/dev/null | wc -l)
if [ "$n" -lt 6000 ]; then say "P1 generation incomplete: $n of 6000 images"; exit 1; fi
say "P1 generated ($n images)"
if ! CUDA_VISIBLE_DEVICES= G6_SUFFIX=_div G6_W=4 python -u "$SRC"/g6_measure.py > logs/g6_measure_div.log 2>&1; then
  say "P1 measurement FAILED - see logs/g6_measure_div.log"; exit 1
fi
if ! python -u "$SRC"/p1_score.py > logs/p1_score.log 2>&1; then say "P1 scoring FAILED - see logs/p1_score.log"; exit 1; fi
say "P1 measured"
