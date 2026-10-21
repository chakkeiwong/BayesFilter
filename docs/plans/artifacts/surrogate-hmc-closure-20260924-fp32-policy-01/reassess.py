"""CPU-only decision reassessment of immutable saved M13 numerical evidence."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from bayesfilter.inference.surrogate_hmc_program import ROOT, write_json
from bayesfilter.testing.surrogate_hmc_m13_consumer import assess_full_parity

started = time.monotonic()
destination = Path(__file__).resolve().parent / "assessment"
source = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repair-01/m13-consumer-repair/report.json"
source_bytes = source.read_bytes()
saved = json.loads(source_bytes)
decision = assess_full_parity(saved)
dependencies = saved["fp32_replay"]["manifest"]["source_sha256"]
dependency_checks = {
    path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    for path, expected in dependencies.items()
}
report = dict(
    schema="m13_saved_evidence_reassessment_v1",
    assessment_kind="owner_authorized_decision_policy_change_no_numerical_rerun",
    source_path=str(source.relative_to(ROOT)),
    source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    original_status=saved["status"],
    original_qualified=saved["qualified"],
    original_checks=saved["checks"],
    numerical_dependency_checks=dependency_checks,
    all_numerical_dependencies_unchanged=all(dependency_checks.values()),
    numerical_reexecution=False,
    **decision,
)
write_json(destination / "report.json", report)
write_json(destination / "reassessment-manifest.json", dict(
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    command=[sys.executable, str(Path(__file__).resolve())],
    environment=sys.executable,
    cpu_gpu_status="CPU-only postprocessing; GPUs intentionally hidden before TensorFlow import",
    data_version=report["source_sha256"],
    random_seeds="N/A: no new random draws; source manifest preserves original seeds",
    numerical_reexecution=False,
    source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    decision_source_sha256=hashlib.sha256((ROOT / "bayesfilter/testing/surrogate_hmc_m13_consumer.py").read_bytes()).hexdigest(),
    plan_file=decision["decision_plan"],
    result_file=str((destination / "report.json").relative_to(ROOT)),
    wall_seconds=time.monotonic() - started,
    wall_seconds_scope="post-import reassessment; job manifest records complete process runtime",
))
assert source.read_bytes() == source_bytes, "Saved evidence changed during reassessment"
assert report["all_numerical_dependencies_unchanged"], "Numerical source changed since saved replay"
assert report["qualified"] and not report["failed_required_checks"]
assert len(report["warnings"]) == 1 and not report["warnings"][0]["blocks_continuation"]
print(json.dumps({key: report[key] for key in (
    "status", "qualified", "failed_required_checks", "warnings",
    "all_numerical_dependencies_unchanged", "posterior_qualified")}, indent=2))
