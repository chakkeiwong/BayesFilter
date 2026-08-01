#!/usr/bin/env bash
set -euo pipefail

LIVE_ROOT=/home/chakwong/BayesFilter
PYTHON=/home/chakwong/anaconda3/envs/tf-gpu/bin/python

: "${LAUNCH_STARTED_EPOCH:?exact wrapper wall-clock origin is required}"
: "${LAUNCH_STARTED_MONOTONIC:?exact wrapper monotonic origin is required}"

run_id=""
source_snapshot_root=""
approval_not_after_epoch=""
approval_instance_id=""
args=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      run_id="${2:?missing run id}"
      shift 2
      ;;
    --source-snapshot-root)
      source_snapshot_root="${2:?missing source snapshot root}"
      shift 2
      ;;
    --approval-not-after-epoch)
      approval_not_after_epoch="${2:?missing approval expiry}"
      shift 2
      ;;
    --approval-instance-id)
      approval_instance_id="${2:?missing approval instance id}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

[[ -n "$run_id" && -n "$source_snapshot_root" && -n "$approval_not_after_epoch" && -n "$approval_instance_id" ]] || {
  echo "outer launch boundary is missing required identity arguments" >&2
  exit 73
}
[[ "$source_snapshot_root" == "$LIVE_ROOT/.complete-highdim-source-snapshot-$run_id" ]] || {
  echo "source snapshot root does not match the concrete run" >&2
  exit 73
}
[[ -d "$source_snapshot_root" && ! -L "$source_snapshot_root" ]] || {
  echo "source snapshot root is missing or unsafe" >&2
  exit 73
}
LAUNCHER="$source_snapshot_root/scripts/launch_complete_highdim_leaderboard.py"
[[ -f "$LAUNCHER" && ! -L "$LAUNCHER" ]] || {
  echo "frozen Python launcher is missing or unsafe" >&2
  exit 73
}
[[ "$approval_not_after_epoch" =~ ^[0-9]+$ ]] || {
  echo "approval expiry is invalid" >&2
  exit 73
}
if (( $(/usr/bin/date +%s) >= approval_not_after_epoch )); then
  echo "one-time launch approval has expired" >&2
  exit 73
fi

# Verify the exact manifest and review receipt before creating any live handoff.
"$PYTHON" "$LAUNCHER" "${args[@]}" --verify-only-before-handoff

handoff="$LIVE_ROOT/docs/plans/logs/$run_id"
snapshot_handoff="$source_snapshot_root/docs/plans/logs/$run_id"
handoff_staging="/tmp/${run_id}-live-handoff"
[[ ! -e "$handoff" && ! -L "$handoff" ]] || {
  echo "live handoff already exists" >&2
  exit 73
}
[[ -d "$snapshot_handoff" && ! -L "$snapshot_handoff" ]] || {
  echo "frozen snapshot handoff mountpoint is missing" >&2
  exit 73
}
[[ ! -e "$handoff_staging" && ! -L "$handoff_staging" ]] || {
  echo "per-run handoff staging path already exists" >&2
  exit 73
}
/usr/bin/mkdir -m 700 "$handoff"
/usr/bin/mkdir -m 700 "$handoff_staging"
/usr/bin/mount --bind "$handoff" "$handoff_staging"

/usr/bin/mount --rbind "$LIVE_ROOT" "$LIVE_ROOT"
# A read-only remount of only the top-level bind does not close nested mounts.
# Remount every descendant first, then the repository root itself.
mount_list="$(/usr/bin/findmnt -Rrno TARGET "$LIVE_ROOT")" || {
  echo "failed to enumerate live-source mounts" >&2
  exit 73
}
[[ -n "$mount_list" ]] || {
  echo "live-source mount enumeration was empty" >&2
  exit 73
}
mountinfo_list="$("$PYTHON" - "$LIVE_ROOT" <<'PY'
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
    raise SystemExit("mountinfo repository inventory was empty")
print("\n".join(sorted(set(targets))))
PY
)" || {
  echo "failed to enumerate repository mounts from mountinfo" >&2
  exit 73
}
findmnt_sorted="$(printf '%s\n' "$mount_list" | /usr/bin/sort -u)"
[[ "$findmnt_sorted" == "$mountinfo_list" ]] || {
  echo "findmnt repository inventory was partial or inconsistent" >&2
  exit 73
}
while IFS= read -r target; do
  [[ -n "$target" && "$target" != "$LIVE_ROOT" ]] || continue
  /usr/bin/mount -o remount,bind,ro "$target"
done < <(printf '%s\n' "$mount_list" | /usr/bin/awk '{ print length, $0 }' | /usr/bin/sort -rn | /usr/bin/cut -d' ' -f2-)
/usr/bin/mount -o remount,bind,ro "$LIVE_ROOT"
readonly_check="$(/usr/bin/findmnt -Rrno TARGET,OPTIONS "$LIVE_ROOT")" || {
  echo "failed to verify live-source mount options" >&2
  exit 73
}
readonly_targets="$(printf '%s\n' "$readonly_check" | /usr/bin/awk '{print $1}' | /usr/bin/sort -u)"
[[ "$readonly_targets" == "$mountinfo_list" ]] || {
  echo "read-only verification mount inventory was partial or inconsistent" >&2
  exit 73
}
while read -r target options; do
  [[ -n "$target" ]] || continue
  case ",$options," in
    *,ro,*) ;;
    *)
      echo "live-source descendant remained writable: $target" >&2
      exit 73
      ;;
  esac
done <<< "$readonly_check"
/usr/bin/mount --bind "$handoff_staging" "$snapshot_handoff"

if /usr/bin/touch "$LIVE_ROOT/.${run_id}-forbidden-write" 2>/dev/null; then
  /usr/bin/rm -f "$LIVE_ROOT/.${run_id}-forbidden-write"
  echo "live source remained writable in the outer boundary" >&2
  exit 73
fi
probe="$snapshot_handoff/${run_id}-outer-boundary-probe.tmp"
/usr/bin/touch "$probe"
/usr/bin/rm "$probe"

export COMPLETE_HIGHDIM_OUTER_BOUNDARY_ACTIVE=1
unset PYTHONPATH PYTHONHOME LD_PRELOAD

set +e
"$PYTHON" "$LAUNCHER" "${args[@]}"
launcher_exit_code=$?
set -e

finalizer="$source_snapshot_root/scripts/finalize_complete_highdim_leaderboard_handoff.py"
seal_deadline="$($PYTHON -c 'import os; print(float(os.environ["LAUNCH_STARTED_MONOTONIC"]) + 28720)')"
seal_timeout="$($PYTHON -c '
import os
import time

remaining = float(os.environ["LAUNCH_STARTED_MONOTONIC"]) + 28720 - time.monotonic()
if remaining <= 0.0:
    raise SystemExit(124)
print(f"{remaining:.6f}")
')" || exit 124
set +e
/usr/bin/timeout --signal=TERM --kill-after=5s "${seal_timeout}s" \
  "$PYTHON" "$finalizer" \
  --handoff-dir "$snapshot_handoff" \
  --run-id "$run_id" \
  --launcher-exit-code "$launcher_exit_code" \
  --approval-instance-id "$approval_instance_id" \
  --deadline-monotonic "$seal_deadline" \
  --required-file "$snapshot_handoff/${run_id}-post-export-verification.json" \
  --required-file "$snapshot_handoff/${run_id}-foreground-outcome.json" \
  --required-file "$snapshot_handoff/${run_id}-watchdog-status.json" \
  --producer-descriptor "$snapshot_handoff/${run_id}-supervisor-producer.json" \
  --producer-descriptor "$snapshot_handoff/${run_id}-watchdog-producer.json"
seal_exit_code=$?
set -e
if [[ "$seal_exit_code" -ne 0 ]]; then
  exit 95
fi
for path in "$snapshot_handoff"/*; do
  [[ -f "$path" && ! -L "$path" ]] || continue
  /usr/bin/chmod a-w "$path"
done
/usr/bin/chmod a-w "$snapshot_handoff"
/usr/bin/mount -o remount,bind,ro "$snapshot_handoff"
# The unique staging alias was the original writable bind. Close it before
# claiming the handoff is immutable, then prove both remaining aliases reject.
/usr/bin/mount -o remount,bind,ro "$handoff_staging"
if /usr/bin/touch "$snapshot_handoff/${run_id}-forbidden-post-seal-write" 2>/dev/null; then
  echo "handoff remained writable after final sealing" >&2
  exit 95
fi
if /usr/bin/touch "$handoff_staging/${run_id}-forbidden-staging-post-seal-write" 2>/dev/null; then
  echo "handoff staging alias remained writable after final sealing" >&2
  exit 95
fi
post_lock_writer="$source_snapshot_root/scripts/write_complete_highdim_leaderboard_post_lock_receipt.py"
post_lock_receipt="/tmp/${run_id}-post-lock-integrity.json"
[[ -f "$post_lock_writer" && ! -L "$post_lock_writer" ]] || {
  echo "frozen post-lock receipt writer is missing or unsafe" >&2
  exit 95
}
set +e
"$PYTHON" "$post_lock_writer" \
  --run-id "$run_id" \
  --canonical-alias "$handoff" \
  --snapshot-alias "$snapshot_handoff" \
  --staging-alias "$handoff_staging" \
  --seal "$snapshot_handoff/${run_id}-final-handoff-seal.json" \
  --output "$post_lock_receipt"
post_lock_exit_code=$?
set -e
if [[ "$post_lock_exit_code" -ne 0 ]]; then
  echo "post-lock alias and seal receipt failed" >&2
  exit 95
fi
exit "$launcher_exit_code"
