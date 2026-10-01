#!/usr/bin/env bash
# Entry 87, corrected in Entry 100: admission control for GPU stages.
#
#   gpu_admit.sh <needed_MiB> <command...>
#
# Holds a lock while it waits for the card to have room AND while the job allocates, so the next lane only
# measures memory that is already committed. The lock is a directory, not flock: Git Bash on Windows ships
# no flock, and the first version of this script called it, failed with "command not found", and admitted
# everything unconditionally. mkdir is atomic on every filesystem this runs on.
need=$1; shift
LOCK="${EINV_V2:-.}"/locks/admit.d
mkdir -p "${EINV_V2:-.}"/locks
# take the lock, clearing one left behind by a killed holder
while ! mkdir "$LOCK" 2>/dev/null; do
  if [ -f "$LOCK/pid" ] && ! kill -0 "$(cat "$LOCK/pid" 2>/dev/null)" 2>/dev/null; then
    rm -rf "$LOCK"
    continue
  fi
  sleep 30
done
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge "$need" ] && break
  sleep 180
done
"$@" &
job=$!
sleep 240          # long enough for the pipeline to load and reach its steady-state allocation
rm -rf "$LOCK"; trap - EXIT INT TERM
wait $job
