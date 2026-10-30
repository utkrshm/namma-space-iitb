#!/usr/bin/env bash
# Run GPU jobs one at a time. Each line of $QUEUE/jobs.txt is "<name> <command...>".
# A job runs once; its output goes to $QUEUE/<name>.log and its exit code to <name>.done.
# Append lines to jobs.txt at any time; the runner exits after 2 idle hours.
QUEUE=${QUEUE:-/home/utkarsh/namma-space-iitb-vyas/outputs/spacegen/queue}
cd "$(dirname "$0")/.." || exit 1
source scripts/env.sh
idle=0
while [ $idle -lt 720 ]; do
  job=""
  while read -r name command; do
    [ -z "$name" ] && continue
    if [ ! -e "$QUEUE/$name.done" ] && [ ! -e "$QUEUE/$name.running" ]; then job="$name"; cmd="$command"; break; fi
  done < "$QUEUE/jobs.txt"
  if [ -z "$job" ]; then idle=$((idle + 1)); sleep 10; continue; fi
  idle=0
  touch "$QUEUE/$job.running"
  echo "[$(date '+%F %T')] start $job" >> "$QUEUE/queue.log"
  bash -c "$cmd" > "$QUEUE/$job.log" 2>&1
  code=$?
  echo "$code" > "$QUEUE/$job.done"
  rm -f "$QUEUE/$job.running"
  echo "[$(date '+%F %T')] end $job exit $code" >> "$QUEUE/queue.log"
done
