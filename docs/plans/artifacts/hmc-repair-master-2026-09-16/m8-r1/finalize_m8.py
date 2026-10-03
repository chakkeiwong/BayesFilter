"""Terminal integrity audit with its own measured process cost included exactly once."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    raw = ROOT/"reconciliation-terminal-before-self-charge.json"
    final = ROOT/"reconciliation-terminal.json"
    if raw.exists() or final.exists():
        raise FileExistsError("terminal audit outputs already exist")
    subprocess.run([sys.executable, str(ROOT/"run_cpu.py"), "--seconds", "180",
        "terminal-reconciliation", "--", sys.executable, str(ROOT/"reconcile_m8.py"),
        "--terminal", "--output", str(raw)], check=True)
    result = json.loads(raw.read_text())
    record_path = ROOT/"terminal-reconciliation-run.json"
    record = json.loads(record_path.read_text())
    assert record["returncode"] == 0 and record["gpu_intentionally_hidden"]
    assert not result["outstanding"] and not result["invalid_artifacts"]
    assert not any(result["indexed_audit"]["reserved_seconds"].values())
    path = str(record_path.relative_to(ROOT.parents[4]))
    assert all(row["path"] != path for row in result["other_attempts"])
    seconds = record["elapsed_seconds"]
    budget = result["budget_seconds"]
    budget["m8_measured_or_junit_seconds"]["cpu_reference"] += seconds
    budget["cumulative_charged"]["cpu_reference"] += seconds
    budget["remaining"]["cpu_reference"] -= seconds
    budget["exceeded"] = any(v < 0 for v in budget["remaining"].values())
    assert not budget["exceeded"]
    result["other_attempts"].append({"path": path,
        "sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(), "seconds": seconds,
        "device": "cpu_reference", "returncode": 0, "command": record["command"]})
    result["finalization"] = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "environment": sys.executable,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "precharge_audit_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "self_cost_seconds": seconds,
        "remaining_wrapper_bookkeeping": "covered by declared 600-second CPU overhead allowance"}
    with final.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"terminal": result["terminal"], "invalid_artifacts": result["invalid_artifacts"],
        "outstanding": result["outstanding"], "budget_seconds": budget,
        "source_snapshots": len(result["source_snapshots"]),
        "candidate_inventories": len(result["candidate_inventories"]),
        "indexed_results": result["indexed_audit"]["indexed_results_checked"],
        "tensor_checksums": result["indexed_audit"]["tensor_checksums_checked"]}, indent=2))


if __name__ == "__main__":
    main()
