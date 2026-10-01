#!/usr/bin/env bash
# Entry 87: admission control for GPU stages. Lane scripts race for one card; before this, two lanes could
# each see the same free memory and both start, which is how G5's generation died with 1.2 GB left.
#
#   gpu_admit.sh <needed_MiB> <command...>
#
# Holds an exclusive lock while it waits for the card to have room AND while the job allocates, so the next
# lane only ever measures memory that is already committed. The lock is released once the job is resident;
# the job itself keeps running under the caller.
need=$1; shift
exec 9>"${EINV_V2:-.}/locks/admit"
flock 9
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge "$need" ] && break
  sleep 180
done
"$@" &
job=$!
sleep 240          # long enough for the pipeline to load and reach its steady-state allocation
flock -u 9
wait $job
