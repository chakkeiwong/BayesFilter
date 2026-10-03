"""Tiny GPU/XLA training-to-tuning canary, not a learned-map quality study."""
import argparse
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
from bayesfilter.testing.inference_validation.designs import ValidationDesign, ScenarioSpec
from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
from bayesfilter.testing.inference_validation.storage import write_json, file_hash

args.output.mkdir(parents=True, exist_ok=False)
design = ValidationDesign(design_id="m37-gpu-composition", engine="accuracy",
    scenario=ScenarioSpec("banana", "fixed_transport"), replications=1, draws=64,
    seed=2026092337, budget_seconds=600, purpose="tiny target-specific GPU graph/batch/freeze/retune canary",
    numerical_provenance="M37 composition test sizes; no quality or adequate-training-price claim")
runtime = configure_worker(design)
import inspect
from tests.inference_validation.test_training_pipeline_composition import training_pipeline_composition
write_json(args.output / "manifest.json", {"runtime": runtime, "source": source_state(),
    "design": design.payload(), "worker_sha256": file_hash(__file__),
    "composition_helper_sha256": file_hash(inspect.getsourcefile(training_pipeline_composition)),
    "seed": [20260923,37], "plan": "docs/plans/bayesfilter-hmc-m37-training-protocol-2026-09-23.md"})
rows = []
for target in ("banana", "mixture"):
    start = time.monotonic()
    row = training_pipeline_composition(args.output / target, target, device="gpu")
    row["elapsed_seconds"] = time.monotonic()-start
    rows.append(row)
    write_json(args.output / "result.json", {"rows": rows, "complete": len(rows) == 2})
print(rows)
