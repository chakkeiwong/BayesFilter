#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?ROOT is required}"
OUTER_HANDOFF_DIR="${OUTER_HANDOFF_DIR:-${OUTER_LOG_DIR:?OUTER_LOG_DIR is required}}"
CODEX_BIN="${CODEX_BIN:?CODEX_BIN is required}"
CODEX_FINAL_MESSAGE_IN_COPY="${CODEX_FINAL_MESSAGE_IN_COPY:?CODEX_FINAL_MESSAGE_IN_COPY is required}"
CODEX_SUPPORT_DIR="${CODEX_SUPPORT_DIR:?CODEX_SUPPORT_DIR is required}"
: "${CODEX_SUPPORT_BINDINGS_JSON:?CODEX_SUPPORT_BINDINGS_JSON is required}"
TF_ENV=/home/chakwong/anaconda3/envs/tf-gpu
NODE_RUNTIME=/home/chakwong/.nvm/versions/node/v22.23.1
VISIBLE=/tmp/complete-highdim-model-visible
SUPPORT_STAGING=/tmp/complete-highdim-support-staging

# Stage the workspace without recursive submounts. In particular, the live
# handoff submount must never follow the copied workspace into a /tmp alias.
/usr/bin/mount --bind "$OUTER_HANDOFF_DIR" "$OUTER_HANDOFF_DIR"
/usr/bin/mount -o remount,bind,ro "$OUTER_HANDOFF_DIR"
if /usr/bin/touch "$OUTER_HANDOFF_DIR/.forbidden-codex-write" 2>/dev/null; then
  echo "Codex handoff mount remained writable" >&2
  exit 73
fi
probe="$ROOT/.complete_highdim_codex_copy_probe"
/usr/bin/touch "$probe"
/usr/bin/rm "$probe"

private_home=/tmp/complete-highdim-model-home
private_codex_home=/tmp/complete-highdim-codex-home
/usr/bin/mkdir -m 700 "$private_home" "$private_codex_home"
[[ -f /home/chakwong/.codex/auth.json && ! -L /home/chakwong/.codex/auth.json ]] || {
  echo "approved Codex authentication source is missing or unsafe" >&2
  exit 73
}
/usr/bin/cp /home/chakwong/.codex/auth.json "$private_codex_home/auth.json"
/usr/bin/chmod 600 "$private_codex_home/auth.json"

/usr/bin/mkdir -m 700 "$VISIBLE" "$SUPPORT_STAGING"
/usr/bin/mkdir -m 755 "$VISIBLE/workspace" "$VISIBLE/tf-gpu" "$VISIBLE/node"
/usr/bin/mount --bind "$ROOT" "$VISIBLE/workspace"
/usr/bin/mount --bind "$TF_ENV" "$VISIBLE/tf-gpu"
/usr/bin/mount --bind "$NODE_RUNTIME" "$VISIBLE/node"
/usr/bin/mount -o remount,bind,ro "$VISIBLE/tf-gpu"
/usr/bin/mount -o remount,bind,ro "$VISIBLE/node"

# Copy and verify exactly the four disclosed support files before hiding the
# host home. The Claude worker is the run-specific stream-auditing wrapper.
TF_CPP_MIN_LOG_LEVEL=3 "$TF_ENV/bin/python" - "$CODEX_SUPPORT_DIR" "$SUPPORT_STAGING" <<'PY'
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

source_dir = Path(sys.argv[1])
staging = Path(sys.argv[2])
bindings = json.loads(os.environ["CODEX_SUPPORT_BINDINGS_JSON"])
expected_names = {
    "trusted-review-verifier.py",
    "trusted-claude-review-gate.sh",
    "trusted-claude-worker.sh",
    "trusted-claude-worker-settings.json",
}
if not isinstance(bindings, list) or {item.get("path") for item in bindings} != expected_names:
    raise SystemExit("support binding allowlist is invalid")
if source_dir.is_symlink() or not source_dir.is_dir():
    raise SystemExit("support source directory is missing or unsafe")
if {path.name for path in source_dir.iterdir()} != expected_names:
    raise SystemExit("support source directory contains an undisclosed entry")
for binding in bindings:
    source = source_dir / binding["path"]
    info = source.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size != binding["size"]
        or hashlib.sha256(source.read_bytes()).hexdigest() != binding["sha256"]
    ):
        raise SystemExit(f"support source drifted: {source}")
    destination = staging / binding["path"]
    with source.open("rb") as source_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(source_stream, output_stream)
    destination.chmod(0o444)
PY

# Hide all host-home data, then re-expose the approved copy and the two
# explicitly disclosed read-only runtime trees.
/usr/bin/mount -t tmpfs -o mode=0755,nosuid,nodev tmpfs /home/chakwong
/usr/bin/mkdir -p "$ROOT" "$TF_ENV" "$NODE_RUNTIME"
/usr/bin/mount --bind "$VISIBLE/workspace" "$ROOT"
/usr/bin/mount --bind "$VISIBLE/tf-gpu" "$TF_ENV"
/usr/bin/mount --bind "$VISIBLE/node" "$NODE_RUNTIME"
/usr/bin/mount -o remount,bind,ro "$TF_ENV"
/usr/bin/mount -o remount,bind,ro "$NODE_RUNTIME"

# Cover the copied support directory and expose only four hash-bound files.
/usr/bin/mkdir -p "$CODEX_SUPPORT_DIR"
/usr/bin/mount -t tmpfs -o mode=0755,nosuid,nodev,noexec tmpfs "$CODEX_SUPPORT_DIR"
for support_source in "$SUPPORT_STAGING"/*; do
  support_name="$(/usr/bin/basename "$support_source")"
  support_target="$CODEX_SUPPORT_DIR/$support_name"
  /usr/bin/touch "$support_target"
  /usr/bin/mount --bind "$support_source" "$support_target"
  /usr/bin/mount -o remount,bind,ro,nosuid,nodev,noexec "$support_target"
done
/usr/bin/mount -o remount,ro,nosuid,nodev,noexec tmpfs "$CODEX_SUPPORT_DIR"

# The canonical dynamic handoff and every temporary staging alias are hidden.
/usr/bin/mount -t tmpfs -o mode=0555,ro,nosuid,nodev,noexec tmpfs "$OUTER_HANDOFF_DIR"
/usr/bin/umount "$VISIBLE/workspace"
/usr/bin/umount "$VISIBLE/tf-gpu"
/usr/bin/umount "$VISIBLE/node"
/usr/bin/mount -t tmpfs -o mode=0555,ro,nosuid,nodev,noexec tmpfs "$VISIBLE"
/usr/bin/mount -t tmpfs -o mode=0555,ro,nosuid,nodev,noexec tmpfs "$SUPPORT_STAGING"
/usr/bin/mount -o remount,ro,mode=0755,nosuid,nodev tmpfs /home/chakwong

for hidden in /home/chakwong/python /home/chakwong/.codex /home/chakwong/.claude; do
  if [[ -e "$hidden" ]]; then
    echo "external model can still read a hidden sibling-home path: $hidden" >&2
    exit 73
  fi
done
for runtime in "$ROOT" "$TF_ENV/bin/python" "$NODE_RUNTIME/bin/codex" "$NODE_RUNTIME/bin/claude"; do
  if [[ ! -e "$runtime" ]]; then
    echo "required selected model runtime is missing: $runtime" >&2
    exit 73
  fi
done

export HOME="$private_home"
export CODEX_HOME="$private_codex_home"
export CLAUDE_WORKER_SETTINGS="$CODEX_SUPPORT_DIR/trusted-claude-worker-settings.json"
export CLAUDE_CONFIG_DIR="$private_home/.claude"
/usr/bin/mkdir -m 700 "$CLAUDE_CONFIG_DIR"

exec /usr/bin/setpriv \
  --no-new-privs \
  --bounding-set=-all \
  --inh-caps=-all \
  --ambient-caps=-all \
  "$TF_ENV/bin/python" \
    "$ROOT/scripts/complete_highdim_leaderboard_exec_codex_after_boundary_check.py" \
    --root "$ROOT" \
    --handoff-dir "$OUTER_HANDOFF_DIR" \
    --support-dir "$CODEX_SUPPORT_DIR" \
    --staging-alias "$VISIBLE" \
    --staging-alias "$SUPPORT_STAGING" \
    --codex-bin "$CODEX_BIN" \
    --final-message "$CODEX_FINAL_MESSAGE_IN_COPY"
