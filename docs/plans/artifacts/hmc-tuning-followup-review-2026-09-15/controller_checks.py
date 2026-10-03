"""Deterministic review counterexamples; no sampler or posterior claims."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig, HMCCandidateSetScope, HMCTuningCandidateSetController,
)
from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload, _validate_result_payload


def scope():
    return HMCCandidateSetScope(scope_id="review", search_id="released-reserve",
        target_signature="synthetic", mass_signature="identity",
        coordinate_system="ordinary", start_bank_signature="synthetic-bank",
        warmup_protocol="none", epsilon_domain=(.05, 2.),
        repair_factor=2., max_repairs_per_family=2)


def observe(work, candidate):
    if candidate.leapfrog_steps == 3:
        if candidate.parent_candidate_id is None:
            return {"decision": "repair_step_lower", "acceptance": .2}
        if work.stage == "measurement" and work.evidence_rung < 2:
            return {"decision": "inconclusive_evidence", "acceptance": .7}
    return {"decision": "passed", "acceptance": .7}


def compact(result):
    return {"completion": result.completion_status, "budget_used": result.budget_used_units,
        "budget_remaining": result.remaining_budget_units,
        "verified_pairs": [(c.leapfrog_steps, c.epsilon) for c in result.candidates
                           if c.candidate_id in result.verified_candidate_ids],
        "pending": [{"L": next(c.leapfrog_steps for c in result.candidates if c.candidate_id == w.candidate_id),
                     "stage": w.stage, "reservation_units": w.reservation_units}
                    for w in result.work_items if w.status != "completed"],
        "events": [dict(e) for e in result.accounting_events
                   if e["event"] in {"reserve_released", "work_budget_deferred"}]}


started = time.monotonic()
config = HMCControllerConfig(primary_l_grid=(3, 5),
    epsilon_by_l=((3, (.5,)), (5, (.5,))), total_budget_units=7,
    repair_reserve_units=3, candidate_reserve_units=3, evidence_rungs=(1, 2, 4))
controller = HMCTuningCandidateSetController(scope(), config)
first = controller.run(observe)
resumed = HMCTuningCandidateSetController.from_result_payload(candidate_set_result_payload(first)).run(observe)
result = {"first": compact(first), "unchanged_resume": compact(resumed),
          "wall_seconds": time.monotonic()-started,
          "role": "deterministic scheduler diagnostic; no numerical authority"}

def shared_observe(work, candidate):
    if candidate.leapfrog_steps == 3 and candidate.parent_candidate_id is None:
        return {"decision": "repair_step_lower", "acceptance": .2}
    if candidate.leapfrog_steps == 5 and work.stage == "verification":
        return {"decision": "failed", "evidence_validity": "shared_execution_invalid"}
    return {"decision": "passed", "acceptance": .7}

shared = HMCTuningCandidateSetController(scope(), HMCControllerConfig(
    primary_l_grid=(3, 5), epsilon_by_l=((3, (.5,)), (5, (.5,))),
    total_budget_units=15, repair_reserve_units=3)).run(shared_observe)
try:
    _validate_result_payload(candidate_set_result_payload(shared))
    validation = "accepted"
except ValueError as exc:
    validation = str(exc)
result["shared_invalidity_after_verified_repair"] = {
    "completion": shared.completion_status, "verified_ids": shared.verified_candidate_ids,
    "repair_statuses": [a.qualified_repair_status for a in shared.repair_actions],
    "artifact_validation": validation,
}
destination = Path(__file__).with_name("controller-checks.json")
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
