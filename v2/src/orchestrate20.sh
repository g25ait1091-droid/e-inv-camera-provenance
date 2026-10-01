#!/usr/bin/env bash
# Chain 20 (Entry 102) - finish G4b after D: filled and killed its generation at p20b_B_s1 image 49.
# Runs beside chain 19's G1-extension training: one generation and one training fit on the card.
# Unlike chains 12-19, every stage's exit code is checked, so a failure is reported as a failure
# instead of being logged as "generated" and handed to the next stage (defect D12).
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
say () { echo "[orch20] $(date +%H:%M) $*"; }
say "start: resume G4b generation"
if ! bash "$SRC"/gpu_admit2.sh 30000 env T1_ARMSET=p20b T1_GENS=250 python -u "$SRC"/t1_ladder.py generate >> logs/t1_p20b_generate.log 2>&1; then
  say "G4b generation FAILED - see logs/t1_p20b_generate.log"; exit 1
fi
n=$(ls out/t1/gens/p20b_*/*.png 2>/dev/null | wc -l)
if [ "$n" -lt 6000 ]; then say "G4b generation incomplete: $n of 6000 images"; exit 1; fi
say "G4b generated ($n images)"
if ! CUDA_VISIBLE_DEVICES= G4B_W=4 python -u "$SRC"/g4b_measure.py > logs/g4b_measure.log 2>&1; then
  say "G4b measurement FAILED - see logs/g4b_measure.log"; exit 1
fi
say "G4b measured"
