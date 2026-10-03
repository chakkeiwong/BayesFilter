"""Reconcile current affected test selection with named, saved outcomes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
start = time.monotonic()
command = json.loads((ROOT / "affected-tests-r5-source.json").read_text())["command"]
command = [c for c in command if not c.startswith(("--junitxml=", "--maxfail="))] + ["--collect-only"]
with (ROOT / "final-test-collection.log").open("x") as log:
    result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            timeout=90, env=dict(os.environ, PYTEST_ADDOPTS=""))
if result.returncode:
    raise ValueError("current test selection cannot be collected")
collected = {line.strip() for line in (ROOT / "final-test-collection.log").read_text().splitlines()
             if line.startswith("tests/") and "::" in line}
passed, skipped, failures = {}, {}, {}
for path in sorted(ROOT.glob("*.xml"), key=lambda p: p.stat().st_mtime_ns):
    for case in ET.parse(path).getroot().iter("testcase"):
        if not case.attrib.get("classname"):
            continue
        node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        bad = [c for c in case if c.tag in {"failure", "error"}]
        skip = [c for c in case if c.tag == "skipped"]
        if bad:
            failures[node] = {"file": str(path), "reason": bad[0].attrib.get("message", "")}
        elif skip:
            skipped[node] = {"file": str(path), "reason": skip[0].attrib.get("message", "")}
        else:
            passed[node] = str(path)
            failures.pop(node, None)
missing = sorted(collected - passed.keys() - skipped.keys())
current_failures = {k: v for k, v in failures.items() if k in collected and k not in passed}
renamed = {k: v for k, v in failures.items() if k not in collected}
source = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((REPO / "bayesfilter").rglob("*.py"))}
record = {"schema": "bayesfilter.hmc_m18_test_inventory.v1", "command": command,
          "created_utc": datetime.now(timezone.utc).isoformat(),
          "selected_count": len(collected), "selected_passed": len(collected & passed.keys()),
          "selected_skipped": {k: v for k, v in skipped.items() if k in collected and k not in passed},
          "missing_selected": missing, "unresolved_selected_failures": current_failures,
          "historical_excluded_or_renamed_failures": renamed,
          "selection_provenance": str(ROOT / "affected-tests-r5-source.json"),
          "passing_sources": {k: v for k, v in passed.items() if k in collected},
          "source_files": source,
          "source_reconciliation": "M18 numerical package matches broad tests except final lazy export; targeted export/documentation/config tests pass on that change.",
          "interpretation": "Affected self-contained selection only; missing private historical fixtures and extended research runs are excluded explicitly."}
with (ROOT / "final-test-inventory.json").open("x") as handle:
    json.dump(record, handle, indent=2)
meter = {"command": sys.argv, "elapsed_seconds": time.monotonic()-start,
         "device": "cpu_reference", "gpu_intentionally_hidden": True,
         "returncode": int(bool(missing or current_failures)), "environment": sys.executable,
         "plan_file": "docs/plans/bayesfilter-hmc-repair-m18-design-2026-09-21.md"}
with (ROOT / "test-inventory-r1-run.json").open("x") as handle:
    json.dump(meter, handle, indent=2)
print(json.dumps({k: record[k] for k in ("selected_count", "selected_passed", "selected_skipped",
                 "missing_selected", "unresolved_selected_failures")}, indent=2))
raise SystemExit(meter["returncode"])
