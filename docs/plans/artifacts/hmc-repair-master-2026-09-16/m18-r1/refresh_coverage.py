"""Add terminal M13--M17 reports without changing prior numerical evidence."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from bayesfilter.testing.inference_validation.reporting import report
from render_inference_validation_coverage import render

assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
start = time.monotonic()
markdown = REPO / "docs/generated/inference_validation_coverage.md"
reports = {Path(p) for p in re.findall(r"Source: `([^`]+)`", markdown.read_text())}
created = []
for phase in ("m13-r1", "m14-r1", "m15-r1", "m16-r1", "m17-r1"):
    for index_path in sorted((ROOT.parent / phase).glob("*/run_index.json")):
        index = json.loads(index_path.read_text())
        if any(j["status"] not in {"complete", "failed", "timed_out", "unfunded"}
               for j in index["jobs"].values()):
            raise ValueError("nonterminal suite: " + str(index_path))
        destination = index_path.parent / "report.json"
        if not destination.exists():
            report(index_path.parent)
            created.append(str(destination))
        reports.add(destination)
render(sorted(reports), markdown, REPO / "docs/generated/inference_validation_coverage.tex")
record = {"command": sys.argv, "created_utc": datetime.now(timezone.utc).isoformat(),
          "elapsed_seconds": time.monotonic()-start, "returncode": 0,
          "device": "cpu_reference", "gpu_intentionally_hidden": True, "environment": sys.executable,
          "plan_file": "docs/plans/bayesfilter-hmc-repair-m18-design-2026-09-21.md",
          "reports": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(reports)},
          "new_reports": created, "generated_markdown": str(markdown),
          "source_classification": "Exact whole-package identity; historical reports are never relabeled current."}
with (ROOT / "coverage-refresh-r1-run.json").open("x") as handle:
    json.dump(record, handle, indent=2)
print(json.dumps({"report_count": len(reports), "new_reports": len(created)}))
