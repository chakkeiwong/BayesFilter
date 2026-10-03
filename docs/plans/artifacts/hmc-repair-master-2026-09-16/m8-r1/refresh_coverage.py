"""Refresh the official coverage table from preserved campaign reports."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
table = REPO / "docs/generated/inference_validation_coverage.md"
reports = {Path(p) for p in re.findall(r"Source: `([^`]+)`", table.read_text())}
reports.update(ROOT.glob("*/report.json"))
reports = sorted({(p if p.is_absolute() else REPO / p).resolve() for p in reports})
for p in reports:
    if not p.exists():
        raise FileNotFoundError(p)
command = [sys.executable, str(REPO / "scripts/render_inference_validation_coverage.py"),
           *map(str, reports)]
subprocess.run(command, cwd=REPO, check=True)
payload = {"command": command, "report_count": len(reports),
           "reports": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in reports},
           "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
           "scope": "Common-suite reports only; case-specific posteriordb outcomes remain in the M8 result."}
with (ROOT / "coverage-refresh.json").open("x") as handle:
    json.dump(payload, handle, indent=2)
print("coverage reports", len(reports))
