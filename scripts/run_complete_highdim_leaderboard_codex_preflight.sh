#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ARTIFACT_DIR="$ROOT/docs/plans/artifacts/complete-highdim-leaderboard"
EVENTS="$ARTIFACT_DIR/launch-codex-preflight-events-2026-07-11.jsonl"
FINAL_MESSAGE="$ARTIFACT_DIR/launch-codex-preflight-final-2026-07-11.txt"
STDERR="$ARTIFACT_DIR/launch-codex-preflight-stderr-2026-07-11.log"
OUTPUT="$ARTIFACT_DIR/launch-codex-preflight-2026-07-11.json"

mkdir -p "$ARTIFACT_DIR"
: >"$EVENTS"
: >"$FINAL_MESSAGE"
: >"$STDERR"

timeout --signal=TERM --kill-after=15s 120s \
  codex exec \
    --cd "$ROOT" \
    --ephemeral \
    --sandbox read-only \
    --json \
    --output-last-message "$FINAL_MESSAGE" \
    "Return exactly CODEX_PROBE_OK. Do not run tools." \
    >"$EVENTS" 2>"$STDERR"
probe_exit=$?

python "$ROOT/scripts/write_complete_highdim_leaderboard_codex_preflight.py" \
  --output "$OUTPUT" \
  --events "$EVENTS" \
  --final-message "$FINAL_MESSAGE" \
  --stderr "$STDERR" \
  --exit-code "$probe_exit" \
  --runner-script "$ROOT/scripts/run_complete_highdim_leaderboard_codex_preflight.sh"
exit $?
