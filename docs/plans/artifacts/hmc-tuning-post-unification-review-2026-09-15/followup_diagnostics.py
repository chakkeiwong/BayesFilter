"""CPU engineering diagnostics for proposal and persistence failures.

Uses private binding construction only to reproduce the exact saved numerical
scope. This script is diagnostic evidence, not a new supported tuning entry point.
"""
from __future__ import annotations

import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Hide GPUs before imports for this CPU-only diagnostic")

from dataclasses import replace
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
started = time.monotonic()
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=False)
baseline_dir = Path(__file__).parent / "automatic-ordinary-r1" / "tuning"

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController
from bayesfilter.inference.hmc_candidate_set_artifacts import (
    write_candidate_set_result, load_candidate_set_result_payload,
)
from bayesfilter.inference.hmc_candidate_set_execution import (
    _issue_binding, _tensor_from_payload, HMCCandidateExecutionConfig,
)
from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
from tests.test_hmc_candidate_set_tuning import _scope, _config, _pass
from tests.test_hmc_candidate_set_execution import GaussianTarget

rows = {}

# A monotone fixture need not assume real HMC acceptance is globally monotone.
# It asks whether the current repair sequence even preserves this simple bracket.
scope = replace(_scope(), epsilon_domain=(.01, 2.2), max_repairs_per_family=3)
config = _config(grid=(3,), epsilons=((3, (1.,)),), evidence_rungs=(1,))
def monotone_outcome(work, candidate):
    if candidate.epsilon < 1.6:
        return {"decision": "repair_step_higher", "acceptance": .9}
    if candidate.epsilon > 1.8:
        return {"decision": "repair_step_lower", "acceptance": .4}
    return _pass(work, candidate)
result = HMCTuningCandidateSetController(scope, config).run(monotone_outcome)
rows["monotone_proposal_fixture"] = {
    "valid_epsilon_interval": [1.6, 1.8],
    "proposals": [c.epsilon for c in result.candidates],
    "verified": result.verified_candidate_ids,
    "completion": result.completion_status,
    "budget_used": result.budget_used_units,
    "remaining_budget": result.remaining_budget_units,
}

# An acceptance decision and a promotion veto are separate fields by design.
def veto_one(work, candidate):
    if candidate.leapfrog_steps == 3:
        return {"decision": "passed", "acceptance": .7,
                "promotion_vetoes": ("native_divergence_positive",),
                "evidence_validity": "valid", "repair_eligible": False}
    return _pass(work, candidate)
result = HMCTuningCandidateSetController(_scope(), _config()).run(veto_one)
archive = destination / "promotion-veto-result.json"
write_candidate_set_result(result, archive)
try:
    load_candidate_set_result_payload(archive)
    reload_error = None
except ValueError as exc:
    reload_error = str(exc)
rows["promotion_veto_roundtrip"] = {
    "candidate_states": result.candidate_states,
    "verified": result.verified_candidate_ids,
    "write_succeeded": True, "reload_error": reload_error,
}

# Reissue the exact prepared scope under a fresh search identity. The complete
# parameterization is in the saved execution spec, not in new numerical defaults.
spec = json.loads((baseline_dir / "execution_spec.json").read_text())["execution"]
prior = json.loads((baseline_dir / "candidate_set_result.json").read_text())
prior_controller = HMCTuningCandidateSetController.from_result_payload(prior)
initial = prior["config"]["initial_epsilon"]
lo, hi = initial * math.sqrt(2), initial * 2
interiors = tuple(math.exp(math.log(lo) + fraction * math.log(hi / lo))
                  for fraction in (.25, .5, .75))
search = replace(prior_controller.config,
    epsilon_by_l=tuple((l, interiors) for l in prior_controller.config.primary_l_grid),
    pilot_enabled=False, refinement_rounds=0)
binding = _issue_binding(adapter=GaussianTarget(), layers=spec["layers"],
    initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
    target_scope=spec["target_scope"], target_lineage=spec["target_lineage"],
    preparation=spec["preparation"],
    config=HMCCandidateExecutionConfig.from_payload(spec["config"]),
    source_paths=[ROOT / "tests/test_hmc_candidate_set_execution.py"],
    scope_id=spec["scope"]["scope_id"], search_id="review-bracket-interiors-v1",
    epsilon_domain=tuple(spec["scope"]["epsilon_domain"]),
    repair_factor=spec["scope"]["repair_factor"],
    max_repairs_per_family=spec["scope"]["max_repairs_per_family"])
identity_fields = ("target_signature", "mass_signature", "coordinate_system",
    "start_bank_signature", "warmup_protocol", "adapter_signature",
    "source_dependency_hash", "target_preparation_identity", "transition_identity")
identities = {k: binding.scope.payload()[k] == spec["scope"][k] for k in identity_fields}
if not all(identities.values()):
    raise RuntimeError("The follow-up scope differs beyond its declared search identity")
run = run_typed_hmc_candidate_set(binding.typed_adapter, search,
    output_dir=destination / "interior-search")
rows["frozen_scope_interior_search"] = {
    "same_scope_fields": identities, "interior_proposals": interiors,
    "completion": run.result.completion_status,
    "candidate_count": len(run.result.candidates),
    "budget_used": run.result.budget_used_units,
    "verified": [{"id": c.candidate_id, "L": c.leapfrog_steps, "epsilon": c.epsilon}
                 for c in run.result.candidates if c.candidate_id in run.result.verified_candidate_ids],
    "candidate_states": run.result.candidate_states,
    "role": "Existence check under the existing screen, no candidate ranking",
}
payload = {
    "baseline": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout.strip(),
    "command": sys.argv,
    "environment": {k: os.environ.get(k) for k in (
        "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_TEST_DEVICE_SCOPE")},
    "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                 "tensorflow_probability": tfp.__version__},
    "wall_seconds_including_imports": time.monotonic() - started,
    "results": rows,
}
(destination / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
