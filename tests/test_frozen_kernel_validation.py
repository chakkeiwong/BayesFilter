from __future__ import annotations

import pytest

from bayesfilter.inference.frozen_kernel_validation import (
    FrozenTuningArtifactBinding,
    FrozenValidationCandidate,
    FrozenValidationPolicy,
    FrozenValidationScope,
    run_frozen_kernel_validation,
)


def _scope() -> FrozenValidationScope:
    return FrozenValidationScope(
        model_id="fixture_model",
        target_signature="target-v1",
        tuning_scope_signature="scope-v1",
        calibration_partition_signature="calibration-v1",
        validation_partition_signature="validation-v1",
        validation_data_signature="data-v1",
        dtype="tf.float64",
        backend="tensorflow_xla_cpu_smoke",
    )


def _artifact() -> FrozenTuningArtifactBinding:
    return FrozenTuningArtifactBinding(
        artifact_signature="artifact-v1",
        model_id="fixture_model",
        target_signature="target-v1",
        tuning_scope_signature="scope-v1",
    )


def _candidate(
    name: str = "primary",
    *,
    parent: str | None = None,
    step: float = 0.2,
) -> FrozenValidationCandidate:
    return FrozenValidationCandidate(
        candidate_id=name,
        model_id="fixture_model",
        target_signature="target-v1",
        tuning_scope_signature="scope-v1",
        controls={"step_size": step, "num_leapfrog_steps": 5},
        control_provenance="independently_tuned" if parent is None else "inherited_exact",
        execution_seed=(11, len(name)),
        parent_candidate_id=parent,
        inherited_control_keys=("step_size",) if parent is not None else (),
    )


def test_generic_executor_returns_unranked_viable_candidates_and_preserves_provenance():
    primary = _candidate()
    coverage = _candidate("coverage", parent="primary", step=0.2)
    result = run_frozen_kernel_validation(
        candidates=(primary, coverage),
        tuning_artifact=_artifact(),
        scope=_scope(),
        policy=FrozenValidationPolicy(required_diagnostics=("finite", "status")),
        runner=lambda candidate, scope, seed: {
            "controls": candidate.controls,
            "status": "complete",
            "diagnostics": {"finite": True, "status": "healthy"},
        },
    )
    assert tuple(item.candidate_id for item in result.next_round_candidates) == (
        "primary",
        "coverage",
    )
    assert result.payload()["next_round_ranking_performed"] is False
    assert coverage.parent_candidate_id == "primary"
    assert coverage.controls["step_size"] == primary.controls["step_size"]


def test_scope_and_partition_mismatches_fail_before_runner():
    with pytest.raises(ValueError, match="calibration and validation"):
        FrozenValidationScope(
            model_id="fixture_model",
            target_signature="target-v1",
            tuning_scope_signature="scope-v1",
            calibration_partition_signature="same",
            validation_partition_signature="same",
            validation_data_signature="data-v1",
            dtype="tf.float64",
            backend="cpu",
        )
    mismatched = FrozenValidationCandidate(
        candidate_id="wrong-model",
        model_id="other_model",
        target_signature="target-v1",
        tuning_scope_signature="scope-v1",
        controls={"step_size": 0.2},
        control_provenance="independently_tuned",
        execution_seed=(1, 2),
    )
    result = run_frozen_kernel_validation(
        candidates=(mismatched,),
        tuning_artifact=_artifact(),
        scope=_scope(),
        policy=FrozenValidationPolicy(),
        runner=lambda *_: pytest.fail("runner must not run"),
    )
    assert "candidate_scope_mismatch:wrong-model" in result.contract_vetoes


def test_missing_diagnostic_and_runner_control_mutation_are_hard_vetoes():
    candidate = _candidate()
    missing = run_frozen_kernel_validation(
        candidates=(candidate,),
        tuning_artifact=_artifact(),
        scope=_scope(),
        policy=FrozenValidationPolicy(required_diagnostics=("finite", "rhat")),
        runner=lambda *_: {"diagnostics": {"finite": True}, "status": "complete"},
    )
    assert missing.observations[0].viable is False
    assert "missing_required_diagnostic:rhat" in missing.observations[0].hard_vetoes
    mutated = run_frozen_kernel_validation(
        candidates=(candidate,),
        tuning_artifact=_artifact(),
        scope=_scope(),
        policy=FrozenValidationPolicy(),
        runner=lambda *_: {
            "controls": {"step_size": 0.3, "num_leapfrog_steps": 5},
            "diagnostics": {"finite": True, "status": "healthy"},
        },
    )
    assert mutated.observations[0].exception is not None
    assert "candidate_execution_exception" in mutated.observations[0].hard_vetoes


def test_candidate_local_failure_does_not_erase_other_candidates():
    good = _candidate("good")
    bad = _candidate("bad", step=0.3)

    def runner(candidate, scope, seed):
        if candidate.candidate_id == "bad":
            raise RuntimeError("fixture failure")
        return {"diagnostics": {"finite": True, "status": "healthy"}}

    result = run_frozen_kernel_validation(
        candidates=(good, bad),
        tuning_artifact=_artifact(),
        scope=_scope(),
        policy=FrozenValidationPolicy(),
        runner=runner,
    )
    assert tuple(item.candidate_id for item in result.next_round_candidates) == ("good",)
    assert result.observations[1].exception == "RuntimeError: fixture failure"
