#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE' >&2
Usage: complete_highdim_leaderboard_claude_audit_worker.sh [options] PROMPT...

Run one Claude print-mode review while preserving the raw stream and observed
tool-use records inside the isolated workspace for the post-run audit.
USAGE
}

cwd="${CLAUDE_WORKER_CWD:-$PWD}"
name="${CLAUDE_WORKER_NAME:-complete-highdim-review}"
model="${CLAUDE_WORKER_MODEL:-opus}"
effort="${CLAUDE_WORKER_EFFORT:-max}"
permission_mode="${CLAUDE_WORKER_PERMISSION_MODE:-plan}"
settings_file="${CLAUDE_WORKER_SETTINGS:-${HOME}/.claude/settings.codex-worker.json}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cwd)
      cwd="${2:?missing cwd}"
      shift 2
      ;;
    --name)
      name="${2:?missing name}"
      shift 2
      ;;
    --model)
      model="${2:?missing model}"
      shift 2
      ;;
    --effort)
      effort="${2:?missing effort}"
      shift 2
      ;;
    --permission-mode)
      permission_mode="${2:?missing permission mode}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

[[ $# -gt 0 ]] || { usage; exit 2; }
[[ -d "$cwd" && -f "$settings_file" ]] || {
  echo "audited Claude worker cwd or settings is missing" >&2
  exit 2
}
[[ "$permission_mode" == "plan" ]] || {
  echo "audited Claude worker requires permission mode plan" >&2
  exit 2
}
[[ "$name" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "audited Claude worker name is unsafe" >&2
  exit 2
}
cwd="$(cd "$cwd" && pwd)"

if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" && -n "${ANTHROPIC_API_KEY:-}" ]]; then
  case "${CLAUDE_WORKER_AUTH_METHOD:-token}" in
    token) unset ANTHROPIC_API_KEY ;;
    api_key) unset ANTHROPIC_AUTH_TOKEN ;;
    *)
      echo "invalid CLAUDE_WORKER_AUTH_METHOD" >&2
      exit 2
      ;;
  esac
fi

audit_dir="$cwd/.complete_highdim_claude_audit"
/usr/bin/mkdir -p -m 700 "$audit_dir"
stamp="$(/usr/bin/date -u +%Y%m%dT%H%M%S)-$$"
prefix="$audit_dir/${stamp}-${name}"
raw="$prefix-stream.jsonl"
stderr_file="$prefix-stderr.log"
metadata="$prefix-metadata.json"
prompt="$*"

exit_code=125
finalized=0
claude_pid=""

handle_signal() {
  local code="$1"
  exit_code="$code"
  if [[ -n "$claude_pid" ]] && kill -0 "$claude_pid" 2>/dev/null; then
    kill -TERM -- "-$claude_pid" 2>/dev/null || true
    wait "$claude_pid" 2>/dev/null || true
  fi
  exit "$code"
}

finalize_audit() {
  local shell_exit=$?
  local parser_exit=0
  trap - EXIT TERM INT HUP
  if [[ "$finalized" -eq 1 ]]; then
    exit "$exit_code"
  fi
  finalized=1
  if [[ "$exit_code" -eq 125 ]]; then
    exit_code="$shell_exit"
  fi

  set +e
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python - \
    "$cwd" "$raw" "$stderr_file" "$metadata" "$name" "$model" "$effort" \
    "$permission_mode" "$exit_code" <<'PY'
import json
import sys
from pathlib import Path

cwd, raw, stderr_path, metadata_path = map(Path, sys.argv[1:5])
name, model, effort, permission_mode = sys.argv[5:9]
exit_code = int(sys.argv[9])
events = []
invalid_lines = 0
for line in raw.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        invalid_lines += 1
        continue
    if not isinstance(value, dict):
        invalid_lines += 1
        continue
    events.append(value)

tool_uses = []
seen = []


def walk(value):
    if isinstance(value, dict):
        if value.get("type") == "tool_use" and isinstance(value.get("name"), str):
            key = (str(value.get("id", "")), value["name"])
            if key not in seen:
                seen.append(key)
                tool_uses.append({"id": key[0], "name": key[1]})
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)


for event in events:
    walk(event)

allowed = {"Read", "Glob", "Grep", "LS"}
disallowed = [record for record in tool_uses if record["name"] not in allowed]
result_text = ""
result_event_present = False
for event in events:
    if event.get("type") == "result" and isinstance(event.get("result"), str):
        result_text = event["result"]
        result_event_present = True
if not result_text:
    chunks = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        content = event.get("message", {}).get("content", [])
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text", "")))
    result_text = "\n".join(chunks)

stream_parse_complete = invalid_lines == 0
observed_read_only = stream_parse_complete and not disallowed
payload = {
    "schema_version": (
        "bayesfilter.complete_highdim_leaderboard.claude_tool_audit.v1"
    ),
    "worker_name": name,
    "model": model,
    "effort": effort,
    "permission_mode": permission_mode,
    "raw_stream_path": raw.relative_to(cwd).as_posix(),
    "stderr_path": stderr_path.relative_to(cwd).as_posix(),
    "claude_exit_code": exit_code,
    "metadata_generated_after_worker_exit_or_signal": True,
    "parsed_event_count": len(events),
    "invalid_stream_line_count": invalid_lines,
    "stream_parse_complete": stream_parse_complete,
    "result_event_present": result_event_present,
    "observed_tool_uses": tool_uses,
    "disallowed_tool_uses": disallowed,
    "read_only_instruction_contract_satisfied_by_observed_tools": observed_read_only,
    "technical_tool_capability_absent": False,
    "prompt_or_credential_value_recorded_by_wrapper": False,
}
with metadata_path.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if result_text:
    print(result_text)
if invalid_lines:
    raise SystemExit(43)
if disallowed:
    raise SystemExit(42)
if exit_code == 0 and not result_text:
    raise SystemExit(44)
PY
  parser_exit=$?
  set -e
  if [[ "$exit_code" -eq 0 && "$parser_exit" -ne 0 ]]; then
    exit_code="$parser_exit"
  fi
  exit "$exit_code"
}

trap finalize_audit EXIT
trap 'handle_signal 143' TERM
trap 'handle_signal 130' INT
trap 'handle_signal 129' HUP

cd "$cwd"
set +e
/usr/bin/setsid --wait claude \
  --print "$prompt" \
  --bare \
  --no-session-persistence \
  --disable-slash-commands \
  --output-format stream-json \
  --verbose \
  --permission-mode plan \
  --settings "$settings_file" \
  --setting-sources project \
  --name "$name" \
  --model "$model" \
  --effort "$effort" \
  >"$raw" 2>"$stderr_file" &
claude_pid=$!
wait "$claude_pid"
exit_code=$?
claude_pid=""
set -e
exit "$exit_code"
