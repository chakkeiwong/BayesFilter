"""Read-only diagnostic reassessment of saved mode-report denominators."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This saved-output reassessment is explicitly CPU-only")

from bayesfilter.testing.inference_validation.designs import ValidationDesign
from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


started = time.monotonic()
index = json.loads((ROOT / "modes-gpu-r1/run_index.json").read_text())
rows = []
for planned in index["plan"]["jobs"]:
    design = ValidationDesign.from_payload(planned["design"])
    job = index["jobs"][design.design_id]
    if job["status"] != "complete":
        raise RuntimeError("Saved mode experiment is not complete")
    source = Path(job["result"])
    result = json.loads(source.read_text())
    summary = summarize_replications(design, result["assessment"]["replications"])
    summary.pop("replications")  # Preserve the full records in the original file.
    rows.append({"design_id": design.design_id, "source": str(source),
        "source_sha256": sha(source), "assessment": summary,
        "original_interval_coverage": result["assessment"]["interval_coverage_at_stop"]})

source_files = ["bayesfilter/testing/inference_validation/engines/pipeline.py",
                "bayesfilter/testing/inference_validation/engines/statistics.py",
                "bayesfilter/testing/inference_validation/designs.py",
                "bayesfilter/testing/inference_validation/catalog.py"]
manifest = {"schema": "bayesfilter.hmc_saved_mode_reassessment.v1", "rows": rows,
    "command": [sys.executable, str(Path(__file__).resolve())],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    "python_version": platform.python_version(), "environment": "tfgpu",
    "device": "CPU diagnostic; GPU deliberately hidden", "CUDA_VISIBLE_DEVICES": "-1",
    "runtime_source_identity": index["source"]["identity"],
    "assessor_source_sha256": {name: sha(REPO / name) for name in source_files},
    "script_sha256": sha(Path(__file__).resolve()), "elapsed_seconds": time.monotonic() - started,
    "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
    "sampler_executed": False, "original_artifacts_changed": False,
    "interpretation": "Corrected missing-quantity denominators; no new posterior evidence"}
with (ROOT / "saved-mode-denominators-r2.json").open("x") as out:
    json.dump(manifest, out, indent=2, sort_keys=True, allow_nan=False)
    out.write("\n")
for row in rows:
    print(row["design_id"], row["assessment"]["interval_coverage_at_stop"]["left_mode_probability:mean"])
