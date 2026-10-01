#!/usr/bin/env bash
# Chain 17 (Entry 87) - one priority queue replacing lanes 2, 3 and 4.
#
# A single training or generation already saturates this card's compute, so running three lanes at once
# did not finish sooner - it only fragmented memory until a 29 GB generation could never be admitted, which
# is how G5 lost five adapters. This queue runs one stage at a time beside lane 1's 16000-step training,
# so a generation and a training coexist (about 39 GB of 46) and generation never starves.
#
# Order is by how close each arm is to being a result: G5 needs only its remaining five adapters generated,
# then G6 and G4b each need training and generation. Every stage resumes, so a restart costs nothing done.
SRC="$(cd "$(dirname "$0")" && pwd)"
cd "${EINV_V2:-$SRC/../workspace}" || exit 1
mkdir -p logs
# EINV_V2 and the other roots come from the environment (v2/README.md)
ADMIT="$SRC"/gpu_admit.sh
say () { echo "[orch17] $(date +%H:%M) $*"; }

say "start: G5 generation, then G6, then G4b"

# --- G5: the second training set. alt_A_s0 already holds 500 images and is skipped on resume.
bash $ADMIT 30000 env T1_ARMSET=alt python -u "$SRC"/t1_ladder.py generate > logs/t1_alt_generate.log 2>&1
say "G5 generated"
CUDA_VISIBLE_DEVICES= T1_ARMSET=alt T1_W=3 python -u "$SRC"/t1_measure.py > logs/t1_alt_measure.log 2>&1
say "G5 measured"

# --- G6: the iPhone 5c pair, both estimators.
bash $ADMIT 12000 env T1_ARMSET=p5c T1_GENS=250 python -u "$SRC"/t1_ladder.py train > logs/t1_p5c_train.log 2>&1
say "G6 trained"
bash $ADMIT 30000 env T1_ARMSET=p5c T1_GENS=250 python -u "$SRC"/t1_ladder.py generate > logs/t1_p5c_generate.log 2>&1
say "G6 generated"
CUDA_VISIBLE_DEVICES= G6_W=4 python -u "$SRC"/g6_measure.py > logs/g6_measure.log 2>&1
say "G6 measured"

# --- G4b: the modern-smartphone pair.
bash $ADMIT 12000 env T1_ARMSET=p20b T1_GENS=250 python -u "$SRC"/t1_ladder.py train > logs/t1_p20b_train.log 2>&1
say "G4b trained"
bash $ADMIT 30000 env T1_ARMSET=p20b T1_GENS=250 python -u "$SRC"/t1_ladder.py generate > logs/t1_p20b_generate.log 2>&1
say "G4b generated"
CUDA_VISIBLE_DEVICES= G4B_W=4 python -u "$SRC"/g4b_measure.py > logs/g4b_measure.log 2>&1
say "G4b measured"

say "queue done"
