"""Bounded CPU engineering diagnostics. Synthetic evidence is not HMC qualification."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_TEST_DEVICE_SCOPE"] = "cpu"
REPOSITORY = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY))

started = time.perf_counter()
import tensorflow as tf
from bayesfilter.inference import HMCAcceptancePolicy, HMCKernelTuningConfig, tune_hmc_kernel
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCControllerConfig, HMCCandidateSetScope, HMCTuningCandidateSetController,
)
from bayesfilter.inference.hmc_candidate_set_artifacts import (
    candidate_set_result_payload, resume_hmc_candidate_set, write_candidate_set_result,
)
from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
from tests.test_hmc_candidate_set_execution import execution_config, make_binding


def scope(**kwargs):
    return HMCCandidateSetScope(**(dict(scope_id="audit", search_id="one",
        target_signature="synthetic", mass_signature="identity", coordinate_system="ordinary",
        start_bank_signature="bank", warmup_protocol="fixture", epsilon_domain=(.01, 2.),
        repair_factor=2., max_repairs_per_family=2) | kwargs))


def config(**kwargs):
    return HMCControllerConfig(**(dict(primary_l_grid=(3,), epsilon_by_l=((3, (.4,)),),
        total_budget_units=20, repair_reserve_units=3) | kwargs))


def summarize(result):
    return dict(completion_status=result.completion_status, final_status=result.final_status,
        states=dict(result.candidate_states), verified=list(result.verified_candidate_ids),
        viable=list(result.viable_candidate_ids), remaining=result.remaining_budget_units,
        spent=result.budget_used_units, pending=list(result.resume_pending_work_item_ids),
        candidate_settings=[(c.leapfrog_steps, c.epsilon) for c in result.candidates],
        repairs=len(result.repair_actions))


results = {}
for decision in ("inconclusive_evidence", "inconclusive_conflict"):
    controller = HMCTuningCandidateSetController(scope(), config())
    result = controller.run(lambda work, candidate: {"decision": decision})
    calls = []
    resumed = resume_hmc_candidate_set(candidate_set_result_payload(result),
        lambda work, candidate: calls.append(work.work_item_id) or {"decision": "passed"})
    assert result.completion_status == "complete" and result.viable_candidate_ids
    assert not result.verified_candidate_ids and not result.resume_pending_work_item_ids and not calls
    results[decision] = dict(initial=summarize(result), resume_calls=len(calls), resumed=summarize(resumed))

controller = HMCTuningCandidateSetController(scope(repair_factor=.5), config())
result = controller.run(lambda work, candidate: {"decision": "repair_step_higher"
    if work.stage == "verification" and candidate.parent_candidate_id is None else "passed"})
action = result.repair_actions[0]
assert action.new_epsilon < action.old_epsilon and result.verified_candidate_ids
try:
    result.replay_candidate(result.verified_candidate_ids[0])
except ValueError as error:
    replay_error = str(error)
else:
    raise AssertionError("expected the direction validator to reject the issued result")
results["repair_factor_below_one"] = dict(action=action.payload(), replay_error=replay_error)

controller = HMCTuningCandidateSetController(scope(), config(candidate_reserve_units=1))
result = controller.run(lambda *_: {"decision": "passed"})
calls = []
resumed = resume_hmc_candidate_set(candidate_set_result_payload(result),
    lambda *args: calls.append(args) or {"decision": "passed"})
assert result.budget_used_units == 1
assert result.remaining_budget_units == 19 and not calls
results["insufficient_candidate_reservation"] = dict(initial=summarize(result),
    resume_calls=len(calls), resumed=summarize(resumed))

controller = HMCTuningCandidateSetController(scope(), config())
result = controller.run(lambda work, candidate: {"decision": "passed" if work.stage == "measurement"
    else ("repair_step_higher" if candidate.epsilon < .6 else "repair_step_lower")})
assert [c.epsilon for c in result.candidates] == [.4, .8, .4]
results["revisits_failed_epsilon"] = summarize(result)

controller = HMCTuningCandidateSetController(scope(), config())
controller.add_exploration_candidate(3, .4)
result = controller.run(lambda *_: {"decision": "passed"})
assert len(result.verified_candidate_ids) == 2
results["duplicate_exploration_settings"] = summarize(result)

# An actual repository binding analyzes a finite, internally consistent all-reject
# fixture. These tensors are fabricated telemetry; no sampling or target claim.
binding = make_binding(config=execution_config(measurement_num_results=64,
    verification_num_results=64, num_warmup_steps=0, acceptance_policy=HMCAcceptancePolicy(),
    target_status_trace_policy="none"), max_repairs_per_family=2)
initial = binding.initial_active_state
samples = tf.repeat(initial[None], 64, axis=0)
proposed = samples + 1.
trace = dict(is_accepted=tf.zeros([64, 4], tf.bool),
    log_accept_ratio=tf.fill([64, 4], tf.math.log(tf.constant(.01, tf.float64))),
    target_log_prob=-.5 * tf.reduce_sum(samples**2, axis=-1),
    proposed_target_log_prob=-.5 * tf.reduce_sum(proposed**2, axis=-1),
    target_score_finite=tf.ones([64, 4], tf.bool), proposed_state=proposed,
    initial_momentum=tf.ones_like(samples), final_momentum=tf.ones_like(samples))
analysis = binding.analyze(initial, samples, trace)
assert binding.health_failures(initial, samples, trace) == ()
assert analysis["acceptance_evidence"]["acceptance_decision"] == "repair_step_lower"
assert analysis["decision"] == "failed" and "movement_gate_failed" in analysis["hard_vetoes"]
controller = HMCTuningCandidateSetController(binding.scope, config())
result = controller.run(lambda *_: analysis)
assert not result.repair_actions
results["lower_step_repair_suppressed"] = dict(analysis=analysis, result=summarize(result))

# Measurement observations are not retained in the controller artifact.
result_text = json.dumps(candidate_set_result_payload(result))
assert "movement_gate_failed" not in result_text and "acceptance_evidence" not in result_text
results["lost_measurement_explanation"] = dict(has_failure_reason=False,
    has_acceptance_evidence=False, verification_receipts=len(result.verification_receipts))

# The evidence evaluator already labels corrupted accepted draws shared-invalid.
# The execution adapter currently loses that scope when normalizing the decision.
corrupt_samples = tf.fill(samples.shape, tf.constant(float("nan"), tf.float64))
shared_analysis = binding.analyze(initial, corrupt_samples, trace)
assert shared_analysis["acceptance_evidence"]["evidence_validity"] == "shared_execution_invalid"
controller = HMCTuningCandidateSetController(binding.scope, config(
    primary_l_grid=(3, 5), epsilon_by_l=((3, (.4,)), (5, (.4,)))))
result = controller.run(lambda work, candidate: shared_analysis if candidate.leapfrog_steps == 3
    else {"decision": "passed"})
assert result.completion_status == "complete" and result.verified_candidate_ids
results["shared_invalidity_loses_scope"] = dict(analysis=shared_analysis, result=summarize(result))

# Fault injection tests the boundary, not actual allocator pressure or R-hat math.
artifact_dir = Path(__file__).parent / "fault-output"
fake_run = SimpleNamespace(samples=samples, trace=trace)
def resource_failure(*args, **kwargs):
    raise tf.errors.ResourceExhaustedError(None, None, "injected CPU audit resource failure")

for label, run_side_effect, diagnostic_side_effect in (
    ("framework_resource_failure", resource_failure, None),
    ("reporting_rhat_failure", None, RuntimeError("injected reporting diagnostic failure")),
):
    out = artifact_dir / label
    with patch.object(binding, "_run", side_effect=run_side_effect, return_value=fake_run), patch(
        "bayesfilter.inference.hmc._rhat_summary_from_retained_samples", side_effect=diagnostic_side_effect,
        return_value={"role": "reporting_only"}):
        try:
            run_typed_hmc_candidate_set(binding.typed_adapter, config(), output_dir=out)
        except Exception as error:
            recorded = dict(error_type=type(error).__name__, error=str(error),
                result_written=(out / "candidate_set_result.json").exists(),
                saved_evidence=len(binding._evidence))
        else:
            raise AssertionError("expected an uncaught boundary failure")
    assert not recorded["result_written"] and recorded["saved_evidence"] == 0
    results[label] = recorded

# Persisted final-result files can be silently replaced on a second call.
path = artifact_dir / "overwrite" / "candidate_set_result.json"
first = HMCTuningCandidateSetController(scope(), config()).run(lambda *_: {"decision": "passed"})
second = HMCTuningCandidateSetController(scope(search_id="two"), config()).run(lambda *_: {"decision": "failed"})
write_candidate_set_result(first, path)
first_bytes = path.read_bytes()
write_candidate_set_result(second, path)
assert path.read_bytes() != first_bytes
results["existing_result_overwritten"] = dict(overwritten=True, path=str(path))

# The public no-config/ordinary-config call still bypasses the shared controller.
for cfg in (None, HMCKernelTuningConfig.standard()):
    sentinel = object()
    with patch("bayesfilter.inference.hmc_kernel_tuning._run_canonical_hmc_tuning", return_value=sentinel) as legacy:
        assert tune_hmc_kernel(adapter=object(), initial_position=[0., 0.], config=cfg) is sentinel
        assert legacy.call_count == 1
results["ordinary_public_dispatch"] = {"omitted_config": "legacy_executor", "ordinary_config": "legacy_executor"}

payload = dict(question="Remaining engineering gaps in the full HMC tuning procedure",
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True).strip(),
    command=[sys.executable, *sys.argv], environment=sys.executable,
    cpu_gpu_status="CPU diagnostics; GPU devices intentionally hidden", data="synthetic fixtures only",
    seeds="No numerical HMC execution; binding fixture seed (20260914, 11)",
    plan="docs/plans/bayesfilter-hmc-whole-procedure-gap-review-2026-09-14.md",
    elapsed_seconds=time.perf_counter()-started, results=results,
    limitations="Fault injection/controller fixtures prove boundary behavior, not real-target incidence or sampler validity")
destination = Path(__file__).parent / "reproduction-results.md"
destination.write_text("# Bounded reproduction results\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n")
print(json.dumps({"artifact": str(destination), "cases": len(results), "elapsed_seconds": payload["elapsed_seconds"]}))
