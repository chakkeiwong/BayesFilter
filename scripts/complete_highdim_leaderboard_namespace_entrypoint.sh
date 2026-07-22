#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:?ROOT is required}"
SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
LAUNCH_ROOT="${LAUNCH_ROOT:?LAUNCH_ROOT is required}"
RUN_ID="${RUN_ID:?RUN_ID is required}"
OUTER_HANDOFF_DIR="${OUTER_HANDOFF_DIR:-${OUTER_LOG_DIR:?OUTER_LOG_DIR is required}}"
COPY_SENTINEL_NONCE="${COPY_SENTINEL_NONCE:?COPY_SENTINEL_NONCE is required}"
export OUTER_HANDOFF_DIR

if [[ "$ROOT" == "$LAUNCH_ROOT" || "$ROOT" != "$SOURCE_ROOT" ]]; then
  echo "invalid namespace path contract" >&2
  exit 73
fi
if [[ "$(/usr/bin/stat -Lc '%d:%i' "$ROOT")" != "$(/usr/bin/stat -Lc '%d:%i' "$LAUNCH_ROOT")" ]]; then
  echo "source path is not the bind-mounted launch copy" >&2
  exit 73
fi
expected_handoff="$SOURCE_ROOT/docs/plans/logs/$RUN_ID"
if [[ "$OUTER_HANDOFF_DIR" != "$expected_handoff" ]]; then
  echo "handoff path is not the approved per-run directory" >&2
  exit 73
fi

remount_read_only() {
  local target="$1"
  /usr/bin/mount --bind "$target" "$target"
  /usr/bin/mount -o remount,bind,ro "$target"
}

# Close nested writable host mounts before closing their parents. The copied
# repo and per-run handoff are the only deliberate writable home submounts.
/usr/bin/mount --rbind /home/chakwong /home/chakwong
home_mount_list="$(/usr/bin/findmnt -Rrno TARGET /home/chakwong)" || {
  echo "failed to enumerate home-tree mounts" >&2
  exit 73
}
[[ -n "$home_mount_list" ]] || {
  echo "home-tree mount enumeration was empty" >&2
  exit 73
}
home_mountinfo_list="$(/home/chakwong/anaconda3/envs/tf-gpu/bin/python - /home/chakwong <<'PY'
import sys

root = sys.argv[1]

def decode(value: str) -> str:
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value

targets = []
with open("/proc/self/mountinfo", encoding="utf-8") as stream:
    for line in stream:
        fields = line.split()
        if len(fields) < 5:
            raise SystemExit("malformed /proc/self/mountinfo")
        target = decode(fields[4])
        if target == root or target.startswith(root + "/"):
            targets.append(target)
if not targets:
    raise SystemExit("mountinfo home inventory was empty")
print("\n".join(sorted(set(targets))))
PY
)" || {
  echo "failed to enumerate home-tree mounts from mountinfo" >&2
  exit 73
}
home_findmnt_sorted="$(printf '%s\n' "$home_mount_list" | /usr/bin/sort -u)"
[[ "$home_findmnt_sorted" == "$home_mountinfo_list" ]] || {
  echo "findmnt home-tree inventory was partial or inconsistent" >&2
  exit 73
}
while IFS= read -r target; do
  [[ -n "$target" ]] || continue
  if [[ "$target" != "/home/chakwong" && "$target" != "$ROOT" && "$target" != "$OUTER_HANDOFF_DIR" ]]; then
    remount_read_only "$target"
  fi
done < <(printf '%s\n' "$home_mount_list" | /usr/bin/awk '{ print length, $0 }' | /usr/bin/sort -rn | /usr/bin/cut -d' ' -f2-)
/usr/bin/mount -o remount,bind,ro /home/chakwong
home_readonly_check="$(/usr/bin/findmnt -Rrno TARGET,OPTIONS /home/chakwong)" || {
  echo "failed to verify home-tree mount options" >&2
  exit 73
}
home_final_mountinfo_list="$(/home/chakwong/anaconda3/envs/tf-gpu/bin/python - /home/chakwong <<'PY'
import sys

root = sys.argv[1]

def decode(value: str) -> str:
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value

targets = []
with open("/proc/self/mountinfo", encoding="utf-8") as stream:
    for line in stream:
        fields = line.split()
        if len(fields) < 5:
            raise SystemExit("malformed /proc/self/mountinfo")
        target = decode(fields[4])
        if target == root or target.startswith(root + "/"):
            targets.append(target)
if not targets:
    raise SystemExit("final mountinfo home inventory was empty")
print("\n".join(sorted(set(targets))))
PY
)" || {
  echo "failed to enumerate final home-tree mounts from mountinfo" >&2
  exit 73
}
home_readonly_targets="$(printf '%s\n' "$home_readonly_check" | /usr/bin/awk '{print $1}' | /usr/bin/sort -u)"
[[ "$home_readonly_targets" == "$home_final_mountinfo_list" ]] || {
  echo "home read-only verification inventory was partial or inconsistent" >&2
  exit 73
}
while read -r target options; do
  [[ -n "$target" ]] || continue
  if [[ "$target" == "$ROOT" || "$target" == "$OUTER_HANDOFF_DIR" ]]; then
    continue
  fi
  case ",$options," in
    *,ro,*) ;;
    *)
      echo "unapproved home-tree mount remained writable: $target" >&2
      exit 73
      ;;
  esac
done <<< "$home_readonly_check"

/usr/bin/mount -t tmpfs -o mode=0555,ro,nosuid,nodev,noexec tmpfs /mnt

# The copied repository and per-run handoff are existing submounts and must
# remain writable after the enclosing home tree becomes read-only.
root_probe="$ROOT/.complete_highdim_namespace_root_probe"
handoff_probe="$OUTER_HANDOFF_DIR/${RUN_ID}-namespace-handoff-probe.tmp"
/usr/bin/touch "$root_probe"
/usr/bin/rm "$root_probe"
/usr/bin/touch "$handoff_probe"
/usr/bin/rm "$handoff_probe"
if /usr/bin/touch "/home/chakwong/.${RUN_ID}-forbidden-write" 2>/dev/null; then
  /usr/bin/rm -f "/home/chakwong/.${RUN_ID}-forbidden-write"
  echo "home tree remained writable" >&2
  exit 73
fi
if /usr/bin/touch "/mnt/.${RUN_ID}-forbidden-write" 2>/dev/null; then
  /usr/bin/rm -f "/mnt/.${RUN_ID}-forbidden-write"
  echo "mounted host drives remained writable" >&2
  exit 73
fi

for private_dir in /tmp /var/tmp /dev/shm; do
  /usr/bin/mount -t tmpfs -o mode=1777,nosuid,nodev tmpfs "$private_dir"
done
if [[ -d /run/user/1000 ]]; then
  /usr/bin/mount -t tmpfs -o mode=700,nosuid,nodev tmpfs /run/user/1000
fi

boundary_receipt="$OUTER_HANDOFF_DIR/${RUN_ID}-namespace-boundary.json"
/home/chakwong/anaconda3/envs/tf-gpu/bin/python - "$boundary_receipt" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "bayesfilter.complete_highdim_leaderboard.namespace_boundary.v1",
    "run_id": os.environ["RUN_ID"],
    "nonce": os.environ["COPY_SENTINEL_NONCE"],
    "root": os.environ["ROOT"],
    "launch_root": os.environ["LAUNCH_ROOT"],
    "handoff_dir": os.environ["OUTER_HANDOFF_DIR"],
    "root_mount_matches_launch_copy": True,
    "home_tree_read_only": True,
    "mounted_host_drives_hidden_by_private_read_only_tmpfs": True,
    "private_tmpfs": True,
    "trusted_supervisor_retains_namespace_mount_capability": True,
    "codex_child_requires_empty_capability_sets": True,
    "codex_child_requires_no_new_privs": True,
    "codex_child_handoff_read_only": True,
    "codex_child_private_pid_namespace": True,
}
with path.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

export TMPDIR=/tmp

preparation="$OUTER_HANDOFF_DIR/${RUN_ID}-launch-preparation.json"
remaining_seconds="$(/home/chakwong/anaconda3/envs/tf-gpu/bin/python - "$preparation" <<'PY'
import json
import math
import sys
import time
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if time.time() >= float(payload["approval_not_after_epoch"]):
    raise SystemExit("one-time approval expired before detached supervisor launch")
remaining = math.floor(
    float(payload["started_monotonic"]) + 27600 - time.monotonic()
)
if remaining < 1:
    raise SystemExit("absolute namespace deadline already expired")
print(remaining)
PY
)"

exec /usr/bin/timeout --signal=TERM --kill-after=60s "${remaining_seconds}s" \
  /usr/bin/bash "$ROOT/scripts/complete_highdim_leaderboard_overnight_supervisor.sh"
