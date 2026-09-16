"""Read-only full-search replay and persistence fan-out inspection."""
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU diagnostic requires hidden GPUs")
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
started = time.monotonic()
from tests.test_hmc_candidate_set_execution import GaussianTarget
from bayesfilter.inference import load_numerical_tuning_checkpoint
from bayesfilter.inference.hmc_candidate_set_checkpoint import write_numerical_tuning_checkpoint
from bayesfilter.inference.hmc_candidate_set_retained import build_retained_bound_hmc_archive_runner_from_candidate_set_result

original = ROOT / "docs/plans/artifacts/hmc-consistency-gap-repair-2026-09-15/automatic-regressions-r2/test_automatic_preparation_and0/isotropic/tuning"
binding, controller = load_numerical_tuning_checkpoint(original / "tuning_checkpoint.json", adapter=GaussianTarget())
result = controller.result()
load_seconds = time.monotonic()-started
calls = []
with patch("bayesfilter.inference.hmc_candidate_set_checkpoint._write",
           side_effect=lambda payload, path, **kwargs: calls.append(str(path))):
    write_numerical_tuning_checkpoint(binding, result, Path("/unused-review-dry-call"))
analyses = []
original_analyze = binding.analyze
def counted(*args):
    analyses.append(1)
    return original_analyze(*args)
with patch.object(binding, "analyze", counted):
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=result, candidate_id=result.verified_candidate_ids[0], retained_binding=binding)
report = {"role": "engineering replay; no new numerical transitions; no target-scale throughput claim",
    "original_path": str(original), "completion": result.completion_status,
    "verified_count": len(result.verified_candidate_ids), "evidence_count": len(binding._evidence),
    "checkpoint_load_seconds_including_imports": load_seconds,
    "one_checkpoint_evidence_visits": sum("numerical_evidence" in p for p in calls),
    "one_member_builder_analysis_count": len(analyses),
    "evidence_json_bytes": sum((original / "numerical_evidence" / (k + ".json")).stat().st_size for k in binding._evidence),
    "member_id": member.candidate.candidate_id, "wall_seconds": time.monotonic()-started}
Path(__file__).with_name("replay-checks.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
