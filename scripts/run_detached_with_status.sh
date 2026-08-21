#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--worker" ]]; then
  shift
  status_dir=$1
  shift
  log="$status_dir/worker.log"
  exit_file="$status_dir/exit_code"
  finished_file="$status_dir/finished_at"
  set +e
  "$@" > "$log" 2>&1
  code=$?
  printf '%s\n' "$code" > "$exit_file"
  date -Iseconds > "$finished_file"
  exit "$code"
fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 status-dir command [args...]" >&2
  exit 2
fi
status_dir=$1
shift
mkdir -p "$status_dir"
exit_file="$status_dir/exit_code"
started_file="$status_dir/started_at"
finished_file="$status_dir/finished_at"
rm -f "$exit_file" "$finished_file"
date -Iseconds > "$started_file"
nohup "$0" --worker "$status_dir" "$@" </dev/null >/dev/null 2>&1 &
printf '%s\n' "$!" > "$status_dir/pid"
printf 'launched pid=%s log=%s\n' "$!" "$status_dir/worker.log"
