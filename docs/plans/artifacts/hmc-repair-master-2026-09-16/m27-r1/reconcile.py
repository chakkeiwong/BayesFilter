"""Reconcile outer M27 attempts exactly once; preserve failed and duplicate work."""
from pathlib import Path
import argparse
import datetime
import json
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--terminal", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parent
opening_file = root.parent / "m26-r1/reconciliation-terminal.json"
opening = json.loads(opening_file.read_text())["remaining_budget_seconds"]
rows = []
for path in sorted(root.glob("*/execution.json")):
    value = json.loads(path.read_text())
    rows.append({"attempt": path.parent.name, "receipt": str(path),
                 "cpu_reference": value.get("cpu_worker_seconds", 0),
                 "gpu": value.get("gpu_worker_seconds", 0), "exit_code": value["exit_code"]})
# The one unmetered focused test has a JUnit clock; reserve ten seconds for import/shutdown.
xml = ET.parse(root / "sibling-tests-r3.xml").getroot()
test_seconds = sum(float(s.attrib.get("time", 0)) for s in xml.findall("testsuite"))
guide = json.loads((root / "guide-build.json").read_text())
collection = json.loads((root / "test-continuation.json").read_text())
extra = {"unmetered_focused_test_junit_plus_ten": test_seconds + 10,
         "test_collection": collection["collection_seconds"],
         "guide_build": guide["wall_seconds"] + guide["bibliography_rebuild"]["wall_seconds"],
         "short_planning_and_summary_diagnostics_allowance": 30}
charged = {key: sum(row[key] for row in rows) for key in ("cpu_reference", "gpu")}
charged["cpu_reference"] += sum(extra.values())
charged["gpu"] += 60  # Conservative allowance for the two earlier readiness probes and preflight queries.
limits = {"cpu_reference": 3600, "gpu": 4800}
assert all(charged[k] <= limits[k] for k in limits), (charged, limits)
missing = [str(p.parent) for p in root.glob("*/manifest.json") if not (p.parent / "execution.json").exists()]
if args.terminal:
    assert not missing, missing
value = {"phase": "M27", "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "opening_ledger": str(opening_file), "opening_remaining_seconds": opening,
         "phase_budget_seconds": limits, "attempts": rows, "additional_cpu_seconds": extra,
         "additional_gpu_seconds": {"readiness_and_preflight_conservative_allowance": 60},
         "charged_seconds": charged,
         "remaining_phase_seconds": {k: limits[k] - charged[k] for k in limits},
         "remaining_budget_seconds": {k: opening[k] - charged[k] for k in limits},
         "pending_attempt_receipts": missing, "all_phase_workers_terminal": args.terminal and not missing,
         "double_count_rule": "Outer execution.json only; nested sequential child receipts already included. Duplicate r3 is charged but not an independent fit.",
         "failures_included": True}
path = root / ("reconciliation-terminal.json" if args.terminal else "reconciliation-live.json")
path.write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps({k: value[k] for k in ("charged_seconds", "remaining_phase_seconds", "remaining_budget_seconds", "pending_attempt_receipts")}))
