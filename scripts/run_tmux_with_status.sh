#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "usage: $0 session-name status-dir command [args...]" >&2
  exit 2
fi
session_name=$1
status_dir=$2
shift 2
mkdir -p "$status_dir"
rm -f "$status_dir/exit_code" "$status_dir/finished_at"
date -Iseconds > "$status_dir/started_at"
worker=(bash scripts/run_detached_with_status.sh --worker "$status_dir" "$@")
printf -v worker_command '%q ' "${worker[@]}"
tmux new-session -d -s "$session_name" "$worker_command"
if tmux has-session -t "$session_name" 2>/dev/null; then
  tmux list-panes -t "$session_name" -F '#{pane_pid}' > "$status_dir/pid"
else
  printf 'completed-before-pid-capture\n' > "$status_dir/pid"
fi
printf '%s\n' "$session_name" > "$status_dir/tmux_session"
printf 'launched session=%s pid=%s log=%s\n' "$session_name" "$(cat "$status_dir/pid")" "$status_dir/worker.log"
