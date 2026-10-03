"""Bound the already-running Phase 0 diagnostic; never launch a GPU workload."""

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path


OUTPUT = Path(__file__).resolve().parent
PID = 203299
GRACE_SECONDS = 30.0
run_start = json.loads((OUTPUT / "run_start.json").read_text())
ledger = json.loads((OUTPUT.parent / "campaign_budget_ledger.json").read_text())
attempt = next(row for row in ledger["attempts"] if row["attempt_id"] == OUTPUT.name)
registration_delay = (
    datetime.fromisoformat(attempt["started_at_utc"])
    - datetime.fromisoformat(run_start["started_at_utc"])
).total_seconds()
started_monotonic = attempt["started_monotonic"] - registration_delay
deadline = started_monotonic + run_start["max_seconds"]
command = Path(f"/proc/{PID}/cmdline").read_bytes().replace(b"\0", b" ").decode()
if OUTPUT.name not in command or "diagnose_ssl_lstm_q20_phase9b" not in command:
    raise RuntimeError("PID does not belong to this diagnostic")
def still_running():
    try:
        current = Path(f"/proc/{PID}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return current == command
    except FileNotFoundError:
        return False


def wait_for_exit(wait_seconds):
    end = time.monotonic() + max(0.0, wait_seconds)
    while still_running():
        remaining = end - time.monotonic()
        if remaining <= 0.0:
            return False
        time.sleep(min(1.0, remaining))
    return True


receipt = {
    "schema": "bayesfilter.phase9b_deadline_supervision.v1",
    "pid": PID,
    "command": command,
    "attached_at_utc": datetime.now(timezone.utc).isoformat(),
    "elapsed_seconds_at_attachment": time.monotonic() - started_monotonic,
    "workload_deadline_seconds": run_start["max_seconds"],
    "termination_grace_seconds": GRACE_SECONDS,
    "reason": "repair_missing_external_timeout_before_deadline",
    "signals": [],
}
with (OUTPUT / "deadline-supervision-start.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
exited = wait_for_exit(deadline - time.monotonic())
if not exited:
    for signal_number, wait_seconds in ((signal.SIGTERM, GRACE_SECONDS), (signal.SIGKILL, 5.0)):
        if not still_running():
            exited = True
            break
        try:
            os.kill(PID, signal_number)
        except ProcessLookupError:
            exited = True
            break
        receipt["signals"].append({
            "signal": signal_number.name,
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
        })
        exited = wait_for_exit(wait_seconds)
        if exited:
            break
receipt["process_exited"] = exited
receipt["elapsed_seconds_at_exit_check"] = time.monotonic() - started_monotonic
receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
with (OUTPUT / "deadline-supervision-result.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
