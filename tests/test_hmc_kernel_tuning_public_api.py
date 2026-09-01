from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

import bayesfilter
from bayesfilter.inference import (
    FixedMassHMCTuningBudgetCallbackResult,
    HMCGeometryScaledBudgetTimingPolicy,
    HMCFrozenStepTrajectoryStageConfig,
    HMCFrozenStepTrajectoryStageResult,
    HMCKernelTuningConfig,
    HMCKernelTuningResult,
    HMCTuneVerifyRepairAttempt,
    HMCTuneVerifyRepairLoopConfig,
    HMCTuneVerifyRepairLoopResult,
    HMCWindowedMassStageConfig,
    HMCWindowedMassStageResult,
    HMC_KERNEL_TUNING_PUBLIC_NONCLAIMS,
    prepare_operational_windowed_mass_handoff,
    tune_hmc_kernel,
)
from bayesfilter.runtime import stable_config_hash
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    bind_neural_force_hmc_tuning_runner,
)

import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning_module
from tests.test_hmc_kernel_tuning_fixed_mass_step import _ToyGaussianAdapter
from tests.test_hmc_kernel_tuning_fixed_mass_step import _bootstrap as _passed_bootstrap
from tests.test_hmc_kernel_tuning_fixed_mass_step import _geometry
from tests.test_hmc_kernel_tuning_outer_loop import _loop_config
from tests.test_hmc_kernel_tuning_bootstrap import _config as _bootstrap_config
from tests.test_hmc_kernel_tuning_bootstrap import _fake_result as _bootstrap_fake_result
from tests.test_hmc_kernel_tuning_bootstrap import run_hmc_bootstrap_screen


def _bootstrap_passed():
    return _passed_bootstrap()


def test_public_tuner_threads_typed_runner_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_scope = "kernel_fixed_mass_step_toy_gaussian"
    binding = bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(
            lambda position: tf.convert_to_tensor(position, tf.float64),
            identity="public-api-binding-force",
        ),
        target=FrozenTargetPotential(
            lambda position: 0.5
            * tf.reduce_sum(tf.square(position), axis=-1),
            identity="public-api-binding-target",
        ),
        target_scope=target_scope,
    )
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result(passed=False)
    seen: list[str] = []

    monkeypatch.setattr(
        hmc_kernel_tuning_module,
        "initialize_hmc_kernel_geometry",
        lambda **_kwargs: geometry,
    )

    def bootstrap_runner(**kwargs: Any):
        assert kwargs["run_full_chain"] is binding
        seen.append("bootstrap")
        return bootstrap

    def loop_runner(**kwargs: Any):
        assert kwargs["run_full_chain"] is binding
        seen.append("loop")
        return loop

    monkeypatch.setattr(
        hmc_kernel_tuning_module, "run_hmc_bootstrap_screen", bootstrap_runner
    )
    monkeypatch.setattr(
        hmc_kernel_tuning_module, "run_hmc_tune_verify_repair_loop", loop_runner
    )

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(target_scope=target_scope),
        runner_binding=binding,
    )

    assert seen == ["bootstrap", "loop"]
    assert result.runner_binding_payload is not None
    assert result.runner_binding_payload["binding_hash"] == binding.binding_hash
    assert result.final_kernel_payload is None
    with pytest.raises(TypeError, match="HMCTuningRunnerBinding"):
        tune_hmc_kernel(
            adapter=_ToyGaussianAdapter(),
            initial_position=[0.0, 0.0],
            config=HMCKernelTuningConfig.smoke(target_scope=target_scope),
            runner_binding=lambda *_args: None,
        )


def test_public_mass_preparation_owns_adequate_standard_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = SimpleNamespace(
        target_dimension=9,
        artifact_hash="geometry-hash",
        mass_artifact=None,
    )
    bootstrap = SimpleNamespace(
        artifact_hash="bootstrap-hash",
        final_status="passed",
    )
    operational = SimpleNamespace(operational_metric_update_count=1)
    windowed = SimpleNamespace(
        passed=True,
        final_status="passed",
        operational_warmup_result=operational,
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        hmc_kernel_tuning_module,
        "initialize_hmc_kernel_geometry",
        lambda **_kwargs: geometry,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning_module,
        "run_hmc_bootstrap_screen",
        lambda **_kwargs: bootstrap,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning_module,
        "_bootstrap_preflight_passed",
        lambda _bootstrap: True,
    )

    def run_windowed(**kwargs: Any) -> Any:
        observed["warmup_steps"] = kwargs[
            "_attempt_budget_policy"
        ].phase4_warmup_steps
        observed["metric_update_requirement"] = kwargs[
            "config"
        ].metric_update_requirement
        return windowed

    monkeypatch.setattr(
        hmc_kernel_tuning_module,
        "run_hmc_windowed_mass_stage",
        run_windowed,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning_module,
        "build_operational_fixed_mass_hmc_adapter",
        lambda **_kwargs: {
            "adapted_mass_artifact": object(),
            "adapted_mass_artifact_signature": "mass-hash",
            "phase4_adapter": object(),
            "final_adapter": object(),
            "final_adapter_signature": "adapter-hash",
            "initial_position": [[0.0] * 9] * 4,
            "start_lineage": {"count": 4},
            "target_scope": "mass-preparation-fixture",
            "hmc_or_tuning_invoked": False,
            "raw_samples_retained": False,
        },
    )

    result = prepare_operational_windowed_mass_handoff(
        adapter=SimpleNamespace(target_scope="mass-preparation-fixture"),
        initial_position=[0.0] * 9,
        config=HMCKernelTuningConfig.standard(
            target_scope="mass-preparation-fixture",
            use_xla=False,
            metric_update_requirement="require_operational_update",
        ),
    )

    assert observed == {
        "warmup_steps": 225,
        "metric_update_requirement": "require_operational_update",
    }
    assert result["windowed_stage"] is windowed
    assert result["warmup_draws_discarded"] is True
    assert result["hmc_or_tuning_invoked"] is False
    assert result["warmup_invoked"] is True
    assert result["reports_posterior_convergence"] is False


def _loop_result(*, passed: bool = True) -> HMCTuneVerifyRepairLoopResult:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    config = _loop_config(max_attempts=1)
    attempt = HMCTuneVerifyRepairAttempt(
        attempt_index=0,
        budget_policy_payload={"budget": 8, "internal_policy_only": True},
        incoming_state_payload=None,
        windowed_stage=None,
        fixed_mass_step_stage=None,
        frozen_step_trajectory_stage=None,
        verification_config_payload={"trace_policy": "standard"},
        verification_diagnostics={
            "acceptance_rate": 0.70 if passed else 0.82,
            "reports_posterior_convergence": False,
        },
        verification_callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        final_status="passed" if passed else "repair_or_retry",
        diagnostic_role="fresh_fixed_kernel_verification_passed"
        if passed
        else "verification_acceptance_repair_trigger",
        hard_vetoes=(),
        repair_triggers=()
        if passed
        else ("verification_acceptance_outside_pass_band",),
        handoff_state_payload={
            "required_private_handoff_complete": True,
            "mass_artifact_signature": geometry.mass_artifact_signature,
            "selected_step_size": 0.2,
            "selected_step_hash": "step-hash",
            "selected_num_leapfrog_steps": 8,
            "selected_trajectory_hash": "trajectory-hash",
        },
    )
    final_kernel_payload = None
    final_kernel_hash = None
    if passed:
        final_kernel_payload = {
            "runtime": "bayesfilter.inference.run_hmc_tune_verify_repair_loop",
            "schema": "bayesfilter.hmc_frozen_kernel_handoff.v1",
            "target_scope": "kernel_fixed_mass_step_toy_gaussian",
            "target_dimension": geometry.target_dimension,
            "adapted_mass_artifact_payload": geometry.mass_artifact.to_payload(
                include_arrays=True
            ),
            "adapted_mass_artifact_signature": geometry.mass_artifact_signature,
            "step_size": 0.2,
            "num_leapfrog_steps": 8,
            "trajectory_length": 1.6,
            "target_accept_prob": config.target_accept_prob,
            "acceptance_band": config.acceptance_band,
            "geometry_artifact_hash": geometry.artifact_hash,
            "bootstrap_artifact_hash": bootstrap.artifact_hash,
            "windowed_stage_artifact_hash": "windowed-hash",
            "fixed_mass_step_stage_artifact_hash": "fixed-step-hash",
            "frozen_step_trajectory_stage_artifact_hash": "trajectory-hash",
            "selected_step_hash": "step-hash",
            "selected_trajectory_hash": "trajectory-hash",
            "verification_config_payload": {"num_results": 4},
            "verification_acceptance_rate": 0.70,
            "budget_policy": {"budget": 8, "internal_policy_only": True},
            "fresh_fixed_kernel_verification_passed": True,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
            "reports_external_client_scientific_claim": False,
            "reports_gpu_or_xla_readiness": False,
            "nonclaims": ("synthetic phase7 result only",),
        }
        final_kernel_hash = stable_config_hash(final_kernel_payload)
    return HMCTuneVerifyRepairLoopResult(
        config=config,
        geometry_artifact_hash=geometry.artifact_hash,
        bootstrap_artifact_hash=bootstrap.artifact_hash,
        adapter_signature=geometry.adapter_signature,
        target_dimension=geometry.target_dimension,
        attempts=(attempt,),
        final_status="passed" if passed else "budget_exhausted",
        diagnostic_role="fresh_fixed_kernel_verification_passed"
        if passed
        else "budget_exhausted_non_promoting",
        hard_vetoes=(),
        repair_triggers=()
        if passed
        else ("phase7_budget_exhausted",),
        final_kernel_payload=final_kernel_payload,
        final_kernel_hash=final_kernel_hash,
        seed_report={"seed_owner": "BayesFilter"},
        diagnostic_roles={
            "fresh_fixed_kernel_verification": "promotion_or_repair_or_hard_veto"
        },
    )


def _loop_result_for_bootstrap(
    bootstrap_hash: str,
    *,
    passed: bool = True,
) -> HMCTuneVerifyRepairLoopResult:
    base = _loop_result(passed=passed)
    final_kernel_payload = base.final_kernel_payload
    final_kernel_hash = base.final_kernel_hash
    if final_kernel_payload is not None:
        final_kernel_payload = {
            **dict(final_kernel_payload),
            "bootstrap_artifact_hash": bootstrap_hash,
        }
        final_kernel_hash = stable_config_hash(final_kernel_payload)
    return HMCTuneVerifyRepairLoopResult(
        config=base.config,
        geometry_artifact_hash=base.geometry_artifact_hash,
        bootstrap_artifact_hash=bootstrap_hash,
        adapter_signature=base.adapter_signature,
        target_dimension=base.target_dimension,
        attempts=base.attempts,
        final_status=base.final_status,
        diagnostic_role=base.diagnostic_role,
        hard_vetoes=base.hard_vetoes,
        repair_triggers=base.repair_triggers,
        final_kernel_payload=final_kernel_payload,
        final_kernel_hash=final_kernel_hash,
        seed_report=base.seed_report,
        diagnostic_roles=base.diagnostic_roles,
    )


def _loop_result_with_rhat_cap_public_summary() -> HMCTuneVerifyRepairLoopResult:
    base = _loop_result(passed=False)
    attempt = base.attempts[0]
    attempt = HMCTuneVerifyRepairAttempt(
        attempt_index=attempt.attempt_index,
        budget_policy_payload={
            "target_dimension": 2,
            "attempt_index": 0,
            "budget": 8,
            "verification_num_results": 64,
            "verification_num_burnin_steps": 16,
            "phase5_tune_budgets": (2, 4, 8),
            "internal_policy_only": True,
            "public_budget_class": "bounded_public_diagnostic_plus",
            "public_budget_cap": 256,
            "public_max_attempts": 3,
            "public_diagnostic_preset": "diagnostic_plus",
        },
        incoming_state_payload=None,
        windowed_stage=None,
        fixed_mass_step_stage=None,
        frozen_step_trajectory_stage=None,
        verification_config_payload={
            "verification_policy": "sequential_rhat",
            "acceptance_band": (0.65, 0.75),
            "max_results": 64,
            "num_burnin_steps": 16,
            "chain_count": 4,
            "rhat_threshold_role": "fixed_kernel_tuning_handoff_gate_not_posterior_proof",
            "step_size": 0.2,
            "num_leapfrog_steps": 8,
        },
        verification_diagnostics={
            "sequential_rhat_verification": True,
            "passed": False,
            "rhat_threshold": 1.01,
            "check_interval": 64,
            "max_results": 64,
            "all_finite_rhat_at_or_below_threshold": False,
            "cap_hit": True,
            "acceptance_rate": 0.70,
            "runtime_finite": True,
            "log_accept_ratio_finite": True,
            "samples_all_finite": True,
            "target_log_prob_finite": True,
            "runner_route_summary": {
                "active_route": "phase7_sequential_rhat_fixed_size_chunk_verifier",
                "single_use_build_count": 0,
                "fallback_status": "none",
                "semantic_source": "_run_phase7_sequential_rhat_final_verification",
                "step_size": 0.2,
            },
            "samples": [[0.0, 0.1]],
            "trace": {"target_log_prob": [-1.0]},
            "target_status_telemetry": {"target_status_trace": [True]},
            "reports_posterior_convergence": False,
        },
        verification_callback_result=attempt.verification_callback_result,
        final_status="repair_or_retry",
        diagnostic_role="verification_rhat_repair_trigger",
        hard_vetoes=(),
        repair_triggers=(
            "verification_rhat_above_threshold_or_cap_hit",
            "verification_rhat_cap_hit",
        ),
        handoff_state_payload=attempt.handoff_state_payload,
    )
    return HMCTuneVerifyRepairLoopResult(
        config=base.config,
        geometry_artifact_hash=base.geometry_artifact_hash,
        bootstrap_artifact_hash=base.bootstrap_artifact_hash,
        adapter_signature=base.adapter_signature,
        target_dimension=base.target_dimension,
        attempts=(attempt,),
        final_status="repair_or_retry",
        diagnostic_role="verification_rhat_repair_trigger",
        hard_vetoes=(),
        repair_triggers=(
            "verification_rhat_above_threshold_or_cap_hit",
            "verification_rhat_cap_hit",
        ),
        final_kernel_payload=None,
        final_kernel_hash=None,
        seed_report=base.seed_report,
        diagnostic_roles=base.diagnostic_roles,
    )


def _phase6_public_summary_stage() -> HMCFrozenStepTrajectoryStageResult:
    geometry = _geometry()
    return HMCFrozenStepTrajectoryStageResult(
        config=HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
            seed=(20260630, 41),
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        geometry_artifact_hash=geometry.artifact_hash,
        bootstrap_artifact_hash="bootstrap-hash",
        windowed_stage_artifact_hash="windowed-hash",
        fixed_mass_step_stage_artifact_hash="fixed-step-hash",
        selected_bootstrap_kernel_hash="bootstrap-kernel-hash",
        selected_step_hash="step-hash",
        adapter_signature=geometry.adapter_signature,
        phase4_hmc_adapter_signature="phase4-hmc-signature",
        phase5_ladder_hmc_adapter_signature="phase5-hmc-signature",
        trajectory_hmc_adapter_signature="phase6-hmc-signature",
        adapted_mass_artifact_payload=geometry.mass_artifact.to_payload(
            include_arrays=True
        ),
        adapted_mass_artifact_signature=geometry.mass_artifact_signature,
        frozen_step_size=0.2,
        fixed_bootstrap_num_leapfrog_steps=8,
        target_dimension=geometry.target_dimension,
        candidate_generation={
            "candidate_l_values": (3, 4, 5),
            "verification_repair_neighborhood_applied": True,
        },
        candidate_results=(
            {
                "candidate_index": 0,
                "classification": "repair_or_retry",
                "diagnostics": {"acceptance_rate": 0.80},
            },
            {
                "candidate_index": 1,
                "classification": "repair_or_retry",
                "diagnostics": {},
            },
        ),
        selected_candidate_index=None,
        selected_trajectory_payload=None,
        selected_trajectory_hash=None,
        final_status="repair_or_retry",
        diagnostic_role="repair_trigger",
        hard_vetoes=(),
        repair_triggers=("acceptance_outside_pass_band",),
        diagnostics={
            "expected_candidate_count": 3,
            "completed_candidate_count": 2,
            "skipped_candidate_count": 1,
            "passed_candidate_count": 0,
            "public_timeout_closeout": {
                "enabled": True,
                "timeout_budget_s": 10.0,
                "reserve_s": 5.0,
                "elapsed_s": 9.2,
                "remaining_s": 0.8,
                "within_closeout_window": True,
                "deadline_clock_scope": "public_one_call_global",
                "stage_elapsed_s": 1.2,
                "stage_remaining_s": 8.8,
                "estimated_next_candidate_s": 1.25,
                "completed_candidate_elapsed_count": 2,
                "closeout_required_before_next_candidate": True,
                "diagnostic_role": "public_timeout_closeout_hard_veto",
                "candidate_index": 2,
                "candidate_count": 3,
                "completed_candidate_count": 2,
                "progress_only": True,
                "public_closeout_artifact_expected": True,
                "reason": "phase6_public_timeout_soft_deadline_before_next_candidate",
                "hmc_mechanics_exposed": False,
                "reports_posterior_convergence": False,
                "reports_sampler_superiority": False,
                "reports_default_readiness": False,
                "reports_gpu_or_xla_readiness": False,
                "step_size": 0.2,
                "num_leapfrog_steps": 5,
            },
            "runner_route_summary": {
                "active_route": "single_use_or_injected_runner",
                "reusable_runner_build_count": 0,
                "distinct_static_runner_contract_count": 0,
                "single_use_build_count": 1,
                "injected_runner_call_count": 0,
                "fallback_status": "inactive_reusable_route",
                "round_route_events": ({"num_leapfrog_steps": 3},),
            },
        },
        frozen_mass_invariant={"passed": True},
        frozen_step_invariant={"passed": True},
        seed_report={"seed_owner": "BayesFilter"},
        diagnostic_roles={"trajectory": "handoff_screen_only"},
    )


def _loop_result_with_phase6_public_summary() -> HMCTuneVerifyRepairLoopResult:
    base = _loop_result(passed=False)
    attempt = base.attempts[0]
    attempt = HMCTuneVerifyRepairAttempt(
        attempt_index=attempt.attempt_index,
        budget_policy_payload=attempt.budget_policy_payload,
        incoming_state_payload=attempt.incoming_state_payload,
        windowed_stage=attempt.windowed_stage,
        fixed_mass_step_stage=attempt.fixed_mass_step_stage,
        frozen_step_trajectory_stage=_phase6_public_summary_stage(),
        verification_config_payload=None,
        verification_diagnostics={
            "not_run": "phase6_repair_or_retry",
            "reports_posterior_convergence": False,
        },
        verification_callback_result=attempt.verification_callback_result,
        final_status="repair_or_retry",
        diagnostic_role="repair_trigger",
        hard_vetoes=(),
        repair_triggers=("phase6_trajectory_status:repair_or_retry",),
        handoff_state_payload=attempt.handoff_state_payload,
    )
    return HMCTuneVerifyRepairLoopResult(
        config=base.config,
        geometry_artifact_hash=base.geometry_artifact_hash,
        bootstrap_artifact_hash=base.bootstrap_artifact_hash,
        adapter_signature=base.adapter_signature,
        target_dimension=base.target_dimension,
        attempts=(attempt,),
        final_status="budget_exhausted",
        diagnostic_role="budget_exhausted_non_promoting",
        hard_vetoes=(),
        repair_triggers=("phase6_trajectory_status:repair_or_retry",),
        final_kernel_payload=None,
        final_kernel_hash=None,
        seed_report=base.seed_report,
        diagnostic_roles=base.diagnostic_roles,
    )


def _loop_result_with_windowed_mass_timeout_closeout() -> HMCTuneVerifyRepairLoopResult:
    base = _loop_result(passed=False)
    attempt = base.attempts[0]
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    closeout = {
        "schema": "bayesfilter.windowed_mass_public_timeout_closeout.v1",
        "stage": "windowed_mass_segment_start",
        "attempt_index": 0,
        "enabled": True,
        "timeout_budget_s": 900.0,
        "reserve_s": 30.0,
        "elapsed_s": 872.0,
        "remaining_s": 28.0,
        "within_closeout_window": True,
        "deadline_clock_scope": "public_one_call_global",
        "closeout_required_before_hmc_call": True,
        "diagnostic_role": "public_timeout_closeout_hard_veto",
        "hard_veto": "windowed_mass_public_timeout_soft_deadline",
        "repair_trigger": "windowed_mass_public_timeout_closeout_before_hmc_call",
        "progress_only": True,
        "public_closeout_artifact_expected": True,
        "completed_segment_count": 5,
        "planned_segment_count": 16,
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_external_client_scientific_claim": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": ("windowed mass timeout closeout only",),
        "step_size": 0.2,
        "num_leapfrog_steps": 8,
        "mass_artifact_payload": {"forbidden": True},
        "samples": [[0.0, 0.1]],
        "trace": {"target_log_prob": [-1.0]},
        "target_log_prob": [-1.0],
        "final_state": [0.0, 0.1],
    }
    windowed_stage = HMCWindowedMassStageResult(
        config=HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            public_timeout_budget_s=900.0,
        ),
        geometry_artifact_hash=geometry.artifact_hash,
        bootstrap_artifact_hash=bootstrap.artifact_hash,
        selected_bootstrap_kernel_hash=str(bootstrap.selected_kernel_hash),
        adapter_signature=geometry.adapter_signature,
        hmc_adapter_signature=bootstrap.hmc_adapter_signature,
        initial_mass_artifact_signature=geometry.mass_artifact_signature,
        target_dimension=geometry.target_dimension,
        final_status="hard_veto",
        diagnostic_role="hard_veto",
        hard_vetoes=("windowed_mass_public_timeout_soft_deadline",),
        diagnostics={
            "passed": False,
            "hard_vetoes": ("windowed_mass_public_timeout_soft_deadline",),
            "public_timeout_closeout": closeout,
            "reports_posterior_convergence": False,
        },
        draw_capture_policy={"diagnostic_role": "adaptation_input_only"},
        warmup_draw_provenance={"adaptation_input_only": True},
        acceptance_telemetry_provenance={"finite_and_aligned": False},
        diagnostic_run_config_payload=None,
        windowed_config_payload={"hmc_mechanics_exposed": False},
        windowed_mass_result=None,
        seed_report={"seed_owner": "BayesFilter"},
        diagnostic_roles={"runtime": "hard_veto"},
    )
    attempt = HMCTuneVerifyRepairAttempt(
        attempt_index=attempt.attempt_index,
        budget_policy_payload=attempt.budget_policy_payload,
        incoming_state_payload=attempt.incoming_state_payload,
        windowed_stage=windowed_stage,
        fixed_mass_step_stage=None,
        frozen_step_trajectory_stage=None,
        verification_config_payload=None,
        verification_diagnostics={
            "not_run": "windowed_mass_public_timeout_soft_deadline",
            "reports_posterior_convergence": False,
        },
        verification_callback_result=attempt.verification_callback_result,
        final_status="hard_veto",
        diagnostic_role="hard_veto",
        hard_vetoes=("windowed_mass_public_timeout_soft_deadline",),
        repair_triggers=(),
        handoff_state_payload=attempt.handoff_state_payload,
    )
    return HMCTuneVerifyRepairLoopResult(
        config=base.config,
        geometry_artifact_hash=base.geometry_artifact_hash,
        bootstrap_artifact_hash=base.bootstrap_artifact_hash,
        adapter_signature=base.adapter_signature,
        target_dimension=base.target_dimension,
        attempts=(attempt,),
        final_status="hard_veto",
        diagnostic_role="hard_veto",
        hard_vetoes=("windowed_mass_public_timeout_soft_deadline",),
        repair_triggers=(),
        final_kernel_payload=None,
        final_kernel_hash=None,
        seed_report=base.seed_report,
        diagnostic_roles=base.diagnostic_roles,
    )


def _loop_result_with_phase7_pre_windowed_timeout_closeout() -> HMCTuneVerifyRepairLoopResult:
    base = _loop_result(passed=False)
    attempt = base.attempts[0]
    closeout = {
        "schema": "bayesfilter.phase7_public_timeout_before_windowed_mass.v1",
        "stage": "phase7_loop_attempt_before_windowed_mass",
        "attempt_index": 0,
        "enabled": True,
        "timeout_budget_s": 900.0,
        "reserve_s": 30.0,
        "elapsed_s": 872.0,
        "remaining_s": 28.0,
        "within_closeout_window": True,
        "deadline_clock_scope": "public_one_call_global",
        "closeout_required_before_windowed_mass_runner_build": True,
        "diagnostic_role": "phase7_pre_windowed_public_timeout_hard_veto",
        "hard_veto": "phase7_public_timeout_before_windowed_mass",
        "repair_trigger": "phase7_public_timeout_before_windowed_mass",
        "progress_only": True,
        "public_closeout_artifact_expected": True,
        "windowed_mass_runner_called": False,
        "target_dimension": 2,
        "public_budget_class": "bounded_public_diagnostic",
        "public_budget_cap": 64,
        "budget_is_public_policy": True,
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_external_client_scientific_claim": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": ("phase7 pre-windowed timeout closeout only",),
        "step_size": 0.2,
        "num_leapfrog_steps": 8,
        "mass_artifact_payload": {"forbidden": True},
        "samples": [[0.0, 0.1]],
        "trace": {"target_log_prob": [-1.0]},
        "target_log_prob": [-1.0],
        "final_state": [0.0, 0.1],
    }
    attempt = HMCTuneVerifyRepairAttempt(
        attempt_index=attempt.attempt_index,
        budget_policy_payload={
            "target_dimension": 2,
            "attempt_index": 0,
            "budget": 32,
            "serious_policy": False,
            "public_budget_class": "bounded_public_diagnostic",
            "public_budget_cap": 64,
            "public_max_attempts": 2,
            "public_diagnostic_preset": "diagnostic",
        },
        incoming_state_payload=attempt.incoming_state_payload,
        windowed_stage=None,
        fixed_mass_step_stage=None,
        frozen_step_trajectory_stage=None,
        verification_config_payload=None,
        verification_diagnostics={
            "attempt_index": 0,
            "not_run": "phase7_public_timeout_before_windowed_mass",
            "public_timeout_closeout": closeout,
            "windowed_stage_runner_called": False,
            "reports_posterior_convergence": False,
        },
        verification_callback_result=attempt.verification_callback_result,
        final_status="hard_veto",
        diagnostic_role="hard_veto",
        hard_vetoes=("phase7_public_timeout_before_windowed_mass",),
        repair_triggers=("phase7_public_timeout_before_windowed_mass",),
        handoff_state_payload=None,
    )
    return HMCTuneVerifyRepairLoopResult(
        config=base.config,
        geometry_artifact_hash=base.geometry_artifact_hash,
        bootstrap_artifact_hash=base.bootstrap_artifact_hash,
        adapter_signature=base.adapter_signature,
        target_dimension=base.target_dimension,
        attempts=(attempt,),
        final_status="hard_veto",
        diagnostic_role="hard_veto",
        hard_vetoes=("phase7_public_timeout_before_windowed_mass",),
        repair_triggers=("phase7_public_timeout_before_windowed_mass",),
        final_kernel_payload=None,
        final_kernel_hash=None,
        seed_report=base.seed_report,
        diagnostic_roles=base.diagnostic_roles,
    )


def test_public_config_hides_raw_hmc_mechanics() -> None:
    parameters = set(inspect.signature(HMCKernelTuningConfig).parameters)
    forbidden = {
        "step_size",
        "initial_step_size",
        "num_leapfrog_steps",
        "leapfrog_count",
        "min_leapfrog",
        "candidate_l_grid",
        "trajectory_grid",
        "mass_window_schedule",
        "warmup_budget_schedule",
        "tuning_budget_schedule",
        "tune_num_results",
        "screen_num_results",
        "verification_num_results",
        "verification_num_burnin_steps",
        "operational_evidence_extension_checkpoints",
    }

    assert parameters.isdisjoint(forbidden)
    assert "max_leapfrog_steps" in parameters
    assert set(HMCKernelTuningConfig.smoke().payload()["forbidden_public_fields"]) >= forbidden


def test_public_config_exposes_named_operational_evidence_policy_only() -> None:
    default = HMCKernelTuningConfig.serious()
    opted_in = HMCKernelTuningConfig.serious(
        operational_evidence_policy="one_doubling"
    )

    assert default.operational_evidence_policy == "initial_only"
    assert default.payload()["operational_evidence_policy"] == "initial_only"
    assert opted_in.payload()["operational_evidence_policy"] == "one_doubling"
    assert "operational_evidence_extension_checkpoints" not in inspect.signature(
        HMCKernelTuningConfig
    ).parameters

    with pytest.raises(ValueError, match="operational_evidence_policy"):
        HMCKernelTuningConfig.serious(operational_evidence_policy="unknown")
    with pytest.raises(ValueError, match="requires preset='serious'"):
        HMCKernelTuningConfig.standard(operational_evidence_policy="one_doubling")


def test_public_config_accepts_diagnostic_bootstrap_sizing_without_raw_mechanics() -> None:
    config = HMCKernelTuningConfig.diagnostic(
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        bootstrap_diagnostic_screen_num_results=1,
        bootstrap_diagnostic_screen_num_burnin_steps=1,
        bootstrap_max_repairs=0,
    )

    payload = config.payload()
    bootstrap = hmc_kernel_tuning_module._public_bootstrap_config(config)

    assert payload["bootstrap_diagnostic_screen_num_results"] == 1
    assert payload["bootstrap_diagnostic_screen_num_burnin_steps"] == 1
    assert payload["bootstrap_max_repairs"] == 0
    assert "observability diagnostic only" in payload["bootstrap_diagnostic_sizing_claim"]
    assert bootstrap.screen_num_results == 1
    assert bootstrap.screen_num_burnin_steps == 1
    assert bootstrap.max_repairs == 0
    assert "screen_num_results" not in inspect.signature(HMCKernelTuningConfig).parameters


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"bootstrap_diagnostic_screen_num_results": 0}, "bootstrap_diagnostic_screen_num_results"),
        ({"bootstrap_diagnostic_screen_num_results": -1}, "bootstrap_diagnostic_screen_num_results"),
        ({"bootstrap_diagnostic_screen_num_burnin_steps": 0}, "bootstrap_diagnostic_screen_num_burnin_steps"),
        ({"bootstrap_diagnostic_screen_num_burnin_steps": -1}, "bootstrap_diagnostic_screen_num_burnin_steps"),
    ],
)
def test_public_config_rejects_nonpositive_diagnostic_bootstrap_sizing(
    overrides: Mapping[str, Any],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        HMCKernelTuningConfig.diagnostic(**overrides)


@pytest.mark.parametrize("factory,preset", [
    (HMCKernelTuningConfig.smoke, "smoke"),
    (HMCKernelTuningConfig.standard, "standard"),
    (HMCKernelTuningConfig.serious, "serious"),
])
def test_public_presets_construct_nonclaim_configs(factory: Any, preset: str) -> None:
    config = factory(target_scope="kernel_fixed_mass_step_toy_gaussian")

    assert config.preset == preset
    assert config.use_xla is False
    assert config.trajectory_window_lower_multiplier == pytest.approx(0.3)
    assert config.trajectory_window_upper_multiplier == pytest.approx(3.0)
    assert config.payload()["reports_posterior_convergence"] is False
    assert config.payload()["reports_gpu_or_xla_readiness"] is False
    assert config.payload()["hmc_mechanics_owned_by_bayesfilter"] is True
    assert (
        config.payload()["geometry_scaled_budget_timing_policy"]["schema"]
        == "bayesfilter.hmc_geometry_scaled_budget_timing_policy.v1"
    )


def test_geometry_scaled_budget_timing_policy_is_public_export() -> None:
    assert bayesfilter.HMCGeometryScaledBudgetTimingPolicy is HMCGeometryScaledBudgetTimingPolicy
    assert "HMCGeometryScaledBudgetTimingPolicy" in bayesfilter.__all__
    assert HMCGeometryScaledBudgetTimingPolicy().payload()["public_safe"] is True


def test_public_config_caps_attempts_at_ten() -> None:
    assert (
        HMCKernelTuningConfig.serious(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            max_attempts=10,
        ).max_attempts
        == 10
    )
    with pytest.raises(ValueError, match="hard-capped"):
        HMCKernelTuningConfig.serious(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            max_attempts=11,
        )


def test_public_xla_runtime_parameter_propagates_to_internal_stage_configs() -> None:
    config = HMCKernelTuningConfig.standard(
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        metric_update_requirement="require_operational_update",
        use_xla=True,
        public_timeout_budget_s=810.0,
        max_leapfrog_steps=40,
        step_repair_factor=2.5,
        step_repair_high_acceptance_directional_factor=3.0,
        step_repair_high_acceptance_ladder_max_factor=4.0,
        trajectory_window_lower_multiplier=0.75,
        trajectory_window_upper_multiplier=1.5,
        handoff_screen_policy="phase23_nomination_only",
        terminal_phase6_repair_extra_attempts=1,
        verification_chunk_max_results=250,
        verification_min_retained_results_for_pass=1000,
    )
    geometry = hmc_kernel_tuning_module._public_geometry_config(config)
    bootstrap = hmc_kernel_tuning_module._public_bootstrap_config(config)
    loop = hmc_kernel_tuning_module._public_loop_config(config)

    assert config.payload()["use_xla"] is True
    assert config.payload()["metric_update_requirement"] == (
        "require_operational_update"
    )
    assert config.payload()["public_timeout_budget_s"] == pytest.approx(810.0)
    assert config.payload()["max_leapfrog_steps"] == 40
    assert config.payload()["step_repair_high_acceptance_directional_factor"] == (
        pytest.approx(3.0)
    )
    assert config.payload()["step_repair_high_acceptance_ladder_max_factor"] == (
        pytest.approx(4.0)
    )
    assert geometry.max_leapfrog_steps == 40
    assert bootstrap.max_leapfrog_steps == 40
    assert bootstrap.step_repair_factor == pytest.approx(2.5)
    assert bootstrap.use_xla is True
    assert bootstrap.payload()["use_xla"] is True
    assert loop.use_xla is True
    assert loop.metric_update_requirement == "require_operational_update"
    assert loop.payload()["use_xla"] is True
    assert loop.payload()["metric_update_requirement"] == (
        "require_operational_update"
    )
    assert config.payload()["terminal_phase6_repair_extra_attempts"] == 1
    assert config.payload()["handoff_screen_policy"] == "phase23_nomination_only"
    assert config.payload()["verification_chunk_max_results"] == 250
    assert config.payload()["verification_min_retained_results_for_pass"] == 1000
    assert loop.terminal_phase6_repair_extra_attempts == 1
    assert loop.payload()["terminal_phase6_repair_extra_attempts"] == 1
    assert loop.handoff_screen_policy == "phase23_nomination_only"
    assert loop.payload()["handoff_screen_policy"] == "phase23_nomination_only"
    assert loop.verification_chunk_max_results == 250
    assert loop.verification_min_retained_results_for_pass == 1000
    assert loop.payload()["verification_chunk_max_results"] == 250
    assert loop.payload()["verification_min_retained_results_for_pass"] == 1000
    assert loop.max_leapfrog_steps == 40
    assert loop.step_repair_factor == pytest.approx(2.5)
    assert loop.step_repair_high_acceptance_directional_factor == pytest.approx(3.0)
    assert loop.step_repair_high_acceptance_ladder_max_factor == pytest.approx(4.0)
    assert loop.trajectory_window_lower_multiplier == pytest.approx(0.75)
    assert loop.trajectory_window_upper_multiplier == pytest.approx(1.5)
    assert loop.public_timeout_budget_s == pytest.approx(810.0)
    assert loop.public_timeout_started_perf_counter_s is None

    loop_with_anchor = hmc_kernel_tuning_module._public_loop_config(
        config,
        public_timeout_started_perf_counter_s=12.5,
    )
    assert loop_with_anchor.public_timeout_started_perf_counter_s == pytest.approx(12.5)

    windowed = hmc_kernel_tuning_module._phase7_windowed_stage_config(
        loop_with_anchor,
        attempt_index=0,
    )
    fixed_step = hmc_kernel_tuning_module._phase7_fixed_step_stage_config(
        loop_with_anchor,
        attempt_index=0,
    )
    trajectory = hmc_kernel_tuning_module._phase7_trajectory_stage_config(
        loop_with_anchor,
        attempt_index=0,
    )

    assert windowed.use_xla is True
    assert windowed.metric_update_requirement == "require_operational_update"
    assert windowed.payload()["metric_update_requirement"] == (
        "require_operational_update"
    )
    assert fixed_step.use_xla is True
    assert fixed_step.handoff_screen_policy == "phase23_nomination_only"
    assert fixed_step.step_repair_factor == pytest.approx(2.5)
    assert fixed_step.step_repair_high_acceptance_directional_factor == (
        pytest.approx(3.0)
    )
    assert fixed_step.step_repair_high_acceptance_ladder_max_factor == pytest.approx(4.0)
    assert trajectory.use_xla is True
    assert trajectory.handoff_screen_policy == "phase23_nomination_only"
    assert trajectory.max_leapfrog_steps == 40
    assert trajectory.trajectory_window_lower_multiplier == pytest.approx(0.75)
    assert trajectory.trajectory_window_upper_multiplier == pytest.approx(1.5)
    assert trajectory.public_timeout_budget_s == pytest.approx(810.0)
    assert trajectory.public_timeout_started_perf_counter_s == pytest.approx(12.5)


def test_public_handoff_screen_policy_defaults_and_rejects_unknown_value() -> None:
    config = HMCKernelTuningConfig.standard(
        target_scope="kernel_fixed_mass_step_toy_gaussian"
    )
    loop = hmc_kernel_tuning_module._public_loop_config(config)

    assert config.handoff_screen_policy == "phase22_heuristic_viability_gate"
    assert config.payload()["handoff_screen_policy"] == (
        "phase22_heuristic_viability_gate"
    )
    assert loop.handoff_screen_policy == "phase22_heuristic_viability_gate"
    assert loop.payload()["handoff_screen_policy"] == (
        "phase22_heuristic_viability_gate"
    )

    with pytest.raises(ValueError, match="handoff_screen_policy"):
        HMCKernelTuningConfig.standard(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            handoff_screen_policy="unknown-policy",
        )


def test_public_xla_runtime_parameter_rejects_eager_mode() -> None:
    with pytest.raises(ValueError, match="XLA HMC requires"):
        HMCKernelTuningConfig(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            chain_execution_mode="eager",
            use_xla=True,
        )

    with pytest.raises(ValueError, match="XLA HMC requires"):
        HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            use_xla=True,
        )


class _XLAReadyToyGaussianAdapter(_ToyGaussianAdapter):
    def value_score_capability(self) -> Any:
        return hmc_kernel_tuning_module.ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tests.xla_ready_toy",
            evidence_path="tests/test_hmc_kernel_tuning_public_api.py",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            nonclaims=("tiny xla authority fixture only",),
        )


class _MismatchedScopeXLAReadyToyGaussianAdapter(_ToyGaussianAdapter):
    def value_score_capability(self) -> Any:
        return hmc_kernel_tuning_module.ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tests.xla_ready_mismatched_scope_toy",
            evidence_path="tests/test_hmc_kernel_tuning_public_api.py",
            target_scope="different_scope",
            nonclaims=("tiny xla authority fixture only",),
        )


class _Rank2RequiredToyGaussianAdapter(_ToyGaussianAdapter):
    batch_rank_policy = "rank2_required"

    def __init__(self) -> None:
        self.value_score_shapes: list[tuple[int, ...]] = []

    def log_prob_and_grad(self, theta: Any) -> tuple[Any, Any]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        if value.shape.rank != 2:
            raise ValueError("rank2 bootstrap fixture requires [batch, parameter]")
        self.value_score_shapes.append(tuple(int(item) for item in value.shape))
        return super().log_prob_and_grad(value)


def test_latent_fixed_mass_wrapper_preserves_only_accepted_xla_authority() -> None:
    geometry = _geometry(adapter=_XLAReadyToyGaussianAdapter())
    wrapper = hmc_kernel_tuning_module._build_bootstrap_fixed_mass_adapter(
        adapter=_XLAReadyToyGaussianAdapter(),
        mass_artifact=geometry.mass_artifact,
        mass_signature=geometry.mass_artifact_signature,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )
    capability = wrapper.value_score_capability()

    assert capability.xla_hmc_ready is True
    assert capability.full_chain_xla_diagnostic_ready is True
    assert capability.is_accepted_full_chain_xla_diagnostic_authority is True
    assert capability.target_scope == "kernel_fixed_mass_step_toy_gaussian"
    assert capability.evidence_path == "tests/test_hmc_kernel_tuning_public_api.py"

    non_xla_geometry = _geometry()
    non_xla_wrapper = hmc_kernel_tuning_module._build_bootstrap_fixed_mass_adapter(
        adapter=_ToyGaussianAdapter(),
        mass_artifact=non_xla_geometry.mass_artifact,
        mass_signature=non_xla_geometry.mass_artifact_signature,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )
    non_xla_capability = non_xla_wrapper.value_score_capability()
    assert non_xla_capability.xla_hmc_ready is False
    assert non_xla_capability.full_chain_xla_diagnostic_ready is False

    mismatched_geometry = _geometry(adapter=_MismatchedScopeXLAReadyToyGaussianAdapter())
    mismatched_wrapper = hmc_kernel_tuning_module._build_bootstrap_fixed_mass_adapter(
        adapter=_MismatchedScopeXLAReadyToyGaussianAdapter(),
        mass_artifact=mismatched_geometry.mass_artifact,
        mass_signature=mismatched_geometry.mass_artifact_signature,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )
    mismatched_capability = mismatched_wrapper.value_score_capability()
    assert mismatched_capability.xla_hmc_ready is False
    assert mismatched_capability.full_chain_xla_diagnostic_ready is False


def test_bootstrap_fixed_mass_wrapper_bridges_scalar_state_to_rank2_base() -> None:
    base = _Rank2RequiredToyGaussianAdapter()
    geometry = _geometry(adapter=base)
    wrapper = hmc_kernel_tuning_module._build_bootstrap_fixed_mass_adapter(
        adapter=base,
        mass_artifact=geometry.mass_artifact,
        mass_signature=geometry.mass_artifact_signature,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )

    value, score = wrapper.log_prob_and_grad(wrapper.initial_position())

    assert value.shape == ()
    assert score.shape == (2,)
    assert bool(tf.math.is_finite(value).numpy()) is True
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy()) is True
    assert base.value_score_shapes == [(1, 2)]


def test_tune_hmc_kernel_runs_phase2_phase3_phase7_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result(passed=True)

    def geometry_runner(**_kwargs: Any):
        calls.append("geometry")
        return geometry

    def bootstrap_runner(**_kwargs: Any):
        calls.append("bootstrap")
        return bootstrap

    def loop_runner(**_kwargs: Any):
        calls.append("loop")
        return loop

    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", geometry_runner)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", bootstrap_runner)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", loop_runner)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
    )

    assert calls == ["geometry", "bootstrap", "loop"]
    assert isinstance(result, HMCKernelTuningResult)
    assert result.passed is True
    assert result.final_kernel_payload is not None
    assert result.final_kernel_payload["runtime"] == "bayesfilter.inference.tune_hmc_kernel"
    assert result.final_kernel_payload["internal_tuning_controls_exposed"] is False
    assert result.final_kernel_payload["hmc_mechanics_exposed"] is False
    assert result.final_kernel_payload["mass_arrays_exposed"] is False
    assert "step_size" not in result.final_kernel_payload
    assert "num_leapfrog_steps" not in result.final_kernel_payload
    assert "trajectory_length" not in result.final_kernel_payload
    assert "adapted_mass_artifact_payload" not in result.final_kernel_payload
    assert result.final_kernel_hash == stable_config_hash(result.final_kernel_payload)


def test_public_progress_artifact_records_bootstrap_callback_before_bootstrap_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result(passed=True)
    progress_path = tmp_path / "hmc_kernel_tuning_progress.json"
    calls: list[str] = []

    def bootstrap_runner(**kwargs: Any):
        calls.append("bootstrap_entered")
        callback = kwargs["progress_callback"]
        callback(
            "bootstrap_round_hmc_call_start",
            {
                "schema": "bayesfilter.hmc_bootstrap_public_progress.v1",
                "stage": "bootstrap_round_hmc_call_start",
                "round_index": 0,
                "bootstrap_diagnostic_screen_num_results": 1,
                "bootstrap_diagnostic_screen_num_burnin_steps": 1,
                "chain_execution_mode": "eager",
                "use_xla": False,
                "target_scope": "kernel_fixed_mass_step_toy_gaussian",
                "route_category": "single_use_or_injected_runner",
                "runner_reused": False,
                "elapsed_s": 0.0,
                "classification": None,
                "diagnostic_role": None,
                "hard_veto_categories": (),
                "acceptance_relation_to_band": None,
                "runtime_finite": None,
                "round_timing_available": None,
                "hmc_mechanics_exposed": False,
                "reports_posterior_convergence": False,
                "reports_sampler_superiority": False,
                "reports_default_readiness": False,
                "reports_external_client_scientific_claim": False,
                "reports_gpu_or_xla_readiness": False,
            },
        )
        assert progress_path.exists()
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        event = progress["extra"]["bootstrap_progress_event"]
        assert progress["current_stage"] == "bootstrap_round_hmc_call_start"
        assert progress["last_started_stage"] == "bootstrap_round_hmc_call_start"
        assert event["bootstrap_diagnostic_screen_num_results"] == 1
        assert event["bootstrap_diagnostic_screen_num_burnin_steps"] == 1
        assert event["hmc_mechanics_exposed"] is False
        assert "screen_num_results" not in event
        assert "screen_num_burnin_steps" not in event
        text = json.dumps(progress, sort_keys=True)
        for forbidden in (
            "step_size",
            "num_leapfrog_steps",
            "mass_artifact_payload",
            "samples",
            "trace",
            "target_log_prob",
            "final_state",
        ):
            assert forbidden not in text
        return bootstrap

    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", bootstrap_runner)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.diagnostic(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            chain_execution_mode="eager",
            bootstrap_diagnostic_screen_num_results=1,
            bootstrap_diagnostic_screen_num_burnin_steps=1,
            bootstrap_max_repairs=0,
        ),
        output_dir=tmp_path,
    )

    assert calls == ["bootstrap_entered"]
    assert result.passed is True


def test_phase3_non_promoting_preflight_calls_phase7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_bootstrap_config(
            max_repairs=0,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=lambda _adapter, _initial_state, _config: _bootstrap_fake_result(
            acceptance=0.95
        ),
    )
    calls: list[str] = []

    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def loop_runner(**kwargs: Any):
        calls.append("loop")
        assert kwargs["bootstrap"] is bootstrap
        assert bootstrap.passed is False
        assert kwargs["geometry"] is geometry
        return _loop_result_for_bootstrap(bootstrap.artifact_hash, passed=True)

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", loop_runner)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
    )

    assert calls == ["loop"]
    assert result.passed is True
    assert result.tune_verify_repair_loop is not None
    assert result.final_kernel_payload is not None
    assert result.final_kernel_hash is not None


def test_serious_tuner_fails_closed_without_acceptance_promoted_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_bootstrap_config(
            max_repairs=0,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=lambda _adapter, _initial_state, _config: _bootstrap_fake_result(
            acceptance=0.95
        ),
    )
    assert bootstrap.passed is False
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def forbidden_loop(**_kwargs: Any):
        raise AssertionError("serious tuning must not run after non-promoting bootstrap")

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", forbidden_loop)
    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.serious(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
    )

    assert result.final_status == "hard_veto"
    assert result.diagnostic_role == "bootstrap_acceptance_promotion_required"
    assert result.hard_vetoes == ("bootstrap_acceptance_promoted_kernel_missing",)
    assert result.final_kernel_payload is None


def test_phase3_hard_veto_does_not_call_phase7(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = _geometry()
    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_bootstrap_config(
            max_repairs=0,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=lambda _adapter, _initial_state, _config: _bootstrap_fake_result(
            finite_log_accept=False
        ),
    )
    calls: list[str] = []

    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def forbidden_loop(**_kwargs: Any):
        calls.append("loop")
        raise AssertionError("Phase 7 must not run after bootstrap hard veto")

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", forbidden_loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
    )

    assert calls == []
    assert result.passed is False
    assert result.tune_verify_repair_loop is None
    assert result.final_kernel_payload is None
    assert result.final_kernel_hash is None
    assert result.diagnostic_role == "bootstrap_screen_hard_veto"


def test_bootstrap_hard_veto_retains_round_diagnostics_and_skips_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = _geometry()

    def failing_chain(*_args: Any, **_kwargs: Any):
        raise ValueError("PP-UKF target call failed: sentinel")

    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_bootstrap_config(
            max_repairs=0,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=failing_chain,
    )
    assert bootstrap.rounds[-1].diagnostics["error_type"] == "ValueError"

    calls: list[str] = []
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def forbidden_loop(**_kwargs: Any):
        calls.append("loop")
        raise AssertionError("Phase 7 must not run after bootstrap hard veto")

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", forbidden_loop)
    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    assert calls == []
    assert result.bootstrap is bootstrap
    assert result.failure_diagnostics is not None
    assert result.failure_diagnostics["round_index"] == 0
    assert result.failure_diagnostics["exception_type"] == "ValueError"
    assert result.failure_diagnostics["exception_message"] == (
        "PP-UKF target call failed: sentinel"
    )
    artifact = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    assert artifact["stage_hashes"]["bootstrap"] == bootstrap.artifact_hash
    assert artifact["bootstrap_public_summary"]["hard_veto_present"] is True
    assert artifact["failure_diagnostics"]["exception_message"] == (
        result.failure_diagnostics["exception_message"]
    )
    assert artifact["failure_diagnostics"]["hard_vetoes"] == list(
        result.failure_diagnostics["hard_vetoes"]
    )
    assert artifact["failure_diagnostics"]["repair_triggers"] == list(
        result.failure_diagnostics["repair_triggers"]
    )


def test_final_kernel_emitted_only_after_phase7_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result(passed=False)
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
    )

    assert result.passed is False
    assert result.final_kernel_payload is None
    assert result.final_kernel_hash is None
    assert "phase7_budget_exhausted" in result.repair_triggers


def test_bootstrap_exception_provenance_is_preserved_in_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = _geometry()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])

    def failing_bootstrap(**_kwargs: Any):
        raise ValueError("PP-UKF bootstrap target call failed: sentinel")

    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", failing_bootstrap)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    assert result.final_status == "hard_veto"
    assert result.hard_vetoes == ("bootstrap_screen_error",)
    assert result.failure_diagnostics == {
        "schema": "bayesfilter.hmc_tuning_failure_diagnostics.v1",
        "stage": "bootstrap",
        "exception_type": "ValueError",
        "exception_message": "PP-UKF bootstrap target call failed: sentinel",
        "role": "diagnostic_exception_provenance_not_scientific_evidence",
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_gpu_or_xla_readiness": False,
    }
    artifact = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    assert artifact["failure_diagnostics"] == result.failure_diagnostics


def test_phase23_final_kernel_emitted_only_after_phase7_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result(passed=False)
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            handoff_screen_policy="phase23_nomination_only",
        ),
    )

    assert result.passed is False
    assert result.final_kernel_payload is None
    assert result.final_kernel_hash is None
    assert result.config.payload()["handoff_screen_policy"] == (
        "phase23_nomination_only"
    )
    assert "phase7_budget_exhausted" in result.repair_triggers


def test_output_artifact_is_sanitized_public_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result(passed=True)
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    artifact_path = tmp_path / "hmc_kernel_tuning_result.json"
    assert result.artifact_path == str(artifact_path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["final_kernel_hash"] == result.final_kernel_hash
    assert payload["repair_triggers"] == []
    assert payload["active_repair_triggers"] == []
    assert payload["historical_repair_triggers"] == []
    assert payload["artifact_policy"]["posterior_samples_written"] is False
    assert payload["artifact_policy"]["internal_tuning_controls_exposed"] is False
    assert payload["artifact_policy"]["public_reconstruction_api"] is False
    assert payload["reports_posterior_convergence"] is False
    assert payload["final_kernel_payload"]["verification_acceptance_rate"] == 0.70
    assert payload["final_kernel_payload"]["fresh_fixed_kernel_verification_passed"] is True
    assert payload["final_kernel_payload"]["hmc_mechanics_exposed"] is False
    assert payload["final_kernel_payload"]["mass_arrays_exposed"] is False
    text = json.dumps(payload, sort_keys=True)
    assert "_HMCAttemptBudgetPolicy" not in text
    assert "_HMCPhaseAttemptState" not in text
    assert "candidate_results" not in text
    assert "verification_config_payload" not in text
    final_text = json.dumps(payload["final_kernel_payload"], sort_keys=True)
    assert '"step_size"' not in final_text
    assert '"num_leapfrog_steps"' not in final_text
    assert '"trajectory_length"' not in final_text
    assert '"adapted_mass_artifact_payload"' not in final_text
    assert '"position"' not in final_text
    assert '"covariance"' not in final_text
    assert '"factor"' not in final_text
    assert "budget_policy" not in payload["final_kernel_payload"]


def test_passed_public_artifact_separates_active_and_historical_repair_triggers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    base_loop = _loop_result(passed=True)
    loop = HMCTuneVerifyRepairLoopResult(
        config=base_loop.config,
        geometry_artifact_hash=base_loop.geometry_artifact_hash,
        bootstrap_artifact_hash=base_loop.bootstrap_artifact_hash,
        adapter_signature=base_loop.adapter_signature,
        target_dimension=base_loop.target_dimension,
        attempts=base_loop.attempts,
        final_status=base_loop.final_status,
        diagnostic_role=base_loop.diagnostic_role,
        hard_vetoes=base_loop.hard_vetoes,
        repair_triggers=("verification_acceptance_outside_pass_band",),
        final_kernel_payload=base_loop.final_kernel_payload,
        final_kernel_hash=base_loop.final_kernel_hash,
        seed_report=base_loop.seed_report,
        diagnostic_roles=base_loop.diagnostic_roles,
    )
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    assert result.passed is True
    assert result.repair_triggers == ("verification_acceptance_outside_pass_band",)
    assert payload["status"] == "passed"
    assert payload["repair_triggers"] == []
    assert payload["active_repair_triggers"] == []
    assert payload["historical_repair_triggers"] == [
        "verification_acceptance_outside_pass_band"
    ]
    assert payload["final_kernel_payload"]["verification_acceptance_rate"] == 0.70
    assert payload["final_kernel_payload"]["fresh_fixed_kernel_verification_passed"] is True
    final_text = json.dumps(payload["final_kernel_payload"], sort_keys=True)
    assert '"step_size"' not in final_text
    assert '"num_leapfrog_steps"' not in final_text
    assert '"adapted_mass_artifact_payload"' not in final_text


def test_public_artifact_exposes_phase6_summary_without_candidate_grid_or_mechanics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result_with_phase6_public_summary()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    phase7 = payload["phase7_public_summary"]
    phase6 = phase7["latest_phase6_public_summary"]
    latest_resume = phase7["latest_resume_split_public_summary"]
    assert result.final_kernel_hash is None
    assert payload["artifact_policy"]["candidate_grid_exposed"] is False
    assert payload["artifact_policy"]["phase7_public_summary_exposed"] is True
    assert phase7["final_status"] == "budget_exhausted"
    assert latest_resume["schema"] == "bayesfilter.phase7_resume_split_public_summary.v1"
    assert latest_resume["availability_status"] == "unavailable"
    assert latest_resume["unavailable_reason"] == "unsupported_private_handoff_stage"
    assert latest_resume["private_resume_payload_available"] is False
    assert latest_resume["private_resume_payload_exposed"] is False
    assert latest_resume["verifier_entry_manifest"] is False
    assert latest_resume["final_kernel_handoff"] is False
    assert latest_resume["actual_target_runtime_executed"] is False
    assert latest_resume["contract_hash"] is None
    assert latest_resume["private_resume_payload_hash"]
    assert latest_resume["loop_artifact_hash"] == loop.artifact_hash
    assert phase6["schema"] == "bayesfilter.hmc_frozen_step_trajectory_public_summary.v1"
    assert phase6["candidate_count"] == 3
    assert phase6["completed_candidate_count"] == 2
    assert phase6["skipped_candidate_count"] == 1
    assert phase6["passed_candidate_count"] == 0
    assert (
        phase6["phase6_handoff_screen_policy_role"]
        == "handoff_screen_repair_trigger_non_promoting"
    )
    assert phase6["phase6_handoff_screen_is_fresh_final_verification"] is False
    assert (
        phase6["phase6_handoff_screen_is_posterior_or_sampler_validity_evidence"]
        is False
    )
    assert (
        phase6["trajectory_window_policy_role"]
        == "engineering_viability_gate_non_scientific"
    )
    assert phase6["trajectory_window_viability_gate_active"] is True
    assert phase6["trajectory_window_relations_seen"] == ["unavailable"]
    assert phase6["candidate_acceptance_relation_counts"] == {
        "above_acceptance_band": 1,
        "below_acceptance_band": 0,
        "inside_acceptance_band": 0,
        "unavailable": 1,
    }
    assert phase6["public_timeout_closeout"]["enabled"] is True
    assert phase6["public_timeout_closeout"]["completed_candidate_count"] == 2
    assert phase6["public_timeout_closeout"]["public_closeout_artifact_expected"] is True
    assert (
        phase6["public_timeout_closeout"]["deadline_clock_scope"]
        == "public_one_call_global"
    )
    assert phase6["public_timeout_closeout"]["stage_elapsed_s"] == pytest.approx(1.2)
    assert phase6["public_timeout_closeout"]["hmc_mechanics_exposed"] is False
    assert phase6["candidate_grid_exposed"] is False
    assert phase6["hmc_mechanics_exposed"] is False
    text = json.dumps(phase7, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "candidate_results",
        "num_leapfrog_steps",
        "step_size",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "private_handoff_state",
        "round_route_events",
    ):
        assert forbidden not in text


def test_public_progress_exposes_resume_split_summary_without_mechanics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result_with_phase6_public_summary()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def loop_runner(*, _progress_callback: Any, **_kwargs: Any):
        attempt = loop.attempts[0]
        _progress_callback(
            "loop_attempt_complete",
            {
                "stage": "loop_attempt_complete",
                "attempt_index": attempt.attempt_index,
                "completed": True,
                "bounded_public_budget_payload": attempt.budget_policy_payload,
                "progress_only": True,
                "hmc_mechanics_exposed": False,
                "extra": {
                    "resume_split_public_summary": hmc_kernel_tuning_module._phase7_attempt_resume_split_public_summary(
                        attempt
                    ),
                },
            },
        )
        return loop

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", loop_runner)

    tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    progress = json.loads(
        (tmp_path / "hmc_kernel_tuning_progress.json").read_text(encoding="utf-8")
    )
    event = progress["extra"]["phase7_loop_event"]
    summary = progress["extra"]["phase7_resume_split_public_summary"]
    assert progress["current_stage"] == "result_written"
    assert progress["phase7_last_attempt_index"] == 0
    assert progress["extra"]["phase7_progress_contract"]["progress_only"] is True
    assert event["stage"] == "loop_attempt_complete"
    assert summary["schema"] == "bayesfilter.phase7_resume_split_public_summary.v1"
    assert summary["availability_status"] == "unavailable"
    assert summary["unavailable_reason"] == "unsupported_private_handoff_stage"
    assert summary["private_resume_payload_available"] is False
    assert summary["private_resume_payload_exposed"] is False
    assert summary["verifier_entry_manifest"] is False
    assert summary["final_kernel_handoff"] is False
    assert summary["actual_target_runtime_executed"] is False
    text = json.dumps(progress, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "private/",
        "_manifest.json",
        ".tfs",
        "private_handoff_state",
    ):
        assert forbidden not in text


def test_one_call_private_tuning_ledger_records_mechanics_without_public_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result(passed=False)
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)

    def bootstrap_runner(*, _private_diagnostic_callback: Any, **_kwargs: Any):
        _private_diagnostic_callback(
            "bootstrap_kernel_repair",
            {
                "stage": "bootstrap_repair",
                "round_index": 0,
                "previous_step_size": 0.1,
                "step_size": 0.2,
                "previous_num_leapfrog_steps": 4,
                "num_leapfrog_steps": 8,
                "private_hmc_mechanics": True,
            },
        )
        return bootstrap

    def loop_runner(*, _private_diagnostic_callback: Any, **_kwargs: Any):
        _private_diagnostic_callback(
            "windowed_mass_matrix_change",
            {
                "stage": "windowed_mass_complete",
                "attempt_index": 0,
                "final_status": "passed",
                "diagnostic_role": "windowed_mass_stage_handoff_only",
                "mass_artifact": geometry.mass_artifact,
                "candidate_step_size": 0.2,
                "step_size_trace": (0.2,),
                "acceptance_trace": (0.70,),
                "private_hmc_mechanics": True,
            },
        )
        _private_diagnostic_callback(
            "joint_l_epsilon_candidate_complete",
            {
                "stage": "joint_l_epsilon_candidate_complete",
                "round_index": 0,
                "grid_stage": "initial",
                "candidate_index": 0,
                "candidate_count": 2,
                "candidate_completed_count": 1,
                "candidate_pass_count": 1,
                "candidate_hard_veto_count": 0,
                "num_leapfrog_steps": 8,
                "step_size": 0.2,
                "viable": True,
                "private_hmc_mechanics": True,
            },
        )
        _private_diagnostic_callback(
            "joint_l_epsilon_round_selected",
            {
                "stage": "joint_l_epsilon_round_selected",
                "round_index": 0,
                "grid_stage": "initial",
                "candidate_count": 2,
                "candidate_completed_count": 2,
                "candidate_pass_count": 1,
                "candidate_hard_veto_count": 1,
                "selected_pair_exists": True,
                "selected_candidate_index": 0,
                "num_leapfrog_steps": 8,
                "step_size": 0.2,
                "edge_direction": None,
                "private_hmc_mechanics": True,
            },
        )
        _private_diagnostic_callback(
            "frozen_step_trajectory_candidate_complete",
            {
                "stage": "frozen_step_trajectory_candidate_complete",
                "attempt_index": 0,
                "candidate_index": 0,
                "candidate_count": 1,
                "candidate_completed_count": 1,
                "candidate_pass_count": 1,
                "candidate_hard_veto_count": 0,
                "num_leapfrog_steps": 8,
                "step_size": 0.2,
                "trajectory_length": 1.6,
                "target_trajectory_length": 1.57,
                "screen_acceptance_rate": 0.70,
                "classification": "passed_screen",
                "private_hmc_mechanics": True,
            },
        )
        _private_diagnostic_callback(
            "frozen_step_trajectory_round_selected",
            {
                "stage": "frozen_step_trajectory_round_selected",
                "attempt_index": 0,
                "candidate_count": 1,
                "candidate_completed_count": 1,
                "candidate_pass_count": 1,
                "candidate_hard_veto_count": 0,
                "selected_pair_exists": True,
                "selected_candidate_index": 0,
                "num_leapfrog_steps": 8,
                "step_size": 0.2,
                "trajectory_length": 1.6,
                "target_trajectory_length": 1.57,
                "selected_trajectory_hash": "trajectory-hash",
                "private_hmc_mechanics": True,
            },
        )
        return loop

    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", bootstrap_runner)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", loop_runner)

    tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    private_dir = tmp_path / "private_diagnostics"
    events_path = private_dir / "hmc_tuning_events.jsonl"
    assert events_path.exists()
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_types = [event["event_type"] for event in events]
    assert "bootstrap_kernel_initial" in event_types
    assert "bootstrap_kernel_repair" in event_types
    assert "windowed_mass_matrix_change" in event_types
    assert "joint_l_epsilon_candidate_complete" in event_types
    assert "joint_l_epsilon_round_selected" in event_types
    assert "frozen_step_trajectory_candidate_complete" in event_types
    assert "frozen_step_trajectory_round_selected" in event_types
    assert any(event.get("step_size") == 0.2 for event in events)
    assert any(event.get("num_leapfrog_steps") == 8 for event in events)
    assert any(event.get("trajectory_length") == 1.6 for event in events)
    mass_files = list(private_dir.glob("mass_*.npz"))
    assert mass_files

    progress = json.loads(
        (tmp_path / "hmc_kernel_tuning_progress.json").read_text(encoding="utf-8")
    )
    summary = progress["extra"]["private_tuning_diagnostics"]
    assert summary["private_event_count"] == len(events)
    assert summary["last_private_event_hash"] == events[-1]["event_hash"]
    assert summary["mass_hash"]
    assert summary["candidate_count"] == 2
    assert summary["candidate_completed_count"] == 2
    assert summary["candidate_pass_count"] == 1
    assert summary["candidate_hard_veto_count"] == 1
    assert summary["selected_pair_exists"] is True
    assert summary["private_paths_publicized"] is False
    assert summary["hmc_mechanics_exposed"] is False
    public_text = json.dumps(progress, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "private/",
        "private_diagnostics/",
        ".npz",
    ):
        assert forbidden not in public_text


def test_public_progress_exposes_pre_windowed_resume_split_unavailable_without_mechanics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    clock = {"values": [0.0, 1.0, 100.0]}

    def fake_perf_counter() -> float:
        if clock["values"]:
            return clock["values"].pop(0)
        return 100.0

    monkeypatch.setattr(hmc_kernel_tuning_module.time, "perf_counter", fake_perf_counter)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.diagnostic(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            public_timeout_budget_s=10.0,
        ),
        output_dir=tmp_path,
        diagnostic_callback=None,
    )

    progress = json.loads(
        (tmp_path / "hmc_kernel_tuning_progress.json").read_text(encoding="utf-8")
    )
    artifact = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    event = progress["extra"]["phase7_loop_event"]
    summary = progress["extra"]["phase7_resume_split_public_summary"]
    artifact_summary = artifact["phase7_public_summary"][
        "latest_resume_split_public_summary"
    ]
    assert result.final_status == "hard_veto"
    assert progress["current_stage"] == "result_written"
    assert progress["phase7_last_attempt_index"] == 0
    assert event["stage"] == "loop_complete"
    assert event["extra"]["final_status"] == "hard_veto"
    assert artifact["status"] == "hard_veto"
    assert artifact["repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert artifact["active_repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert artifact["phase7_public_summary"]["repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert artifact["phase7_public_summary"]["latest_attempt_repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert artifact_summary["availability_status"] == "unavailable"
    assert artifact_summary["unavailable_reason"] == (
        "no_private_handoff_before_resume_split"
    )
    assert summary["schema"] == "bayesfilter.phase7_resume_split_public_summary.v1"
    assert summary["availability_status"] == "unavailable"
    assert summary["unavailable_reason"] == "no_private_handoff_before_resume_split"
    assert summary["private_resume_payload_available"] is False
    assert summary["private_resume_payload_exposed"] is False
    assert summary["verifier_entry_manifest"] is False
    assert summary["final_kernel_handoff"] is False
    assert summary["actual_target_runtime_executed"] is False
    assert progress["extra"]["phase7_progress_contract"]["progress_only"] is True
    text = json.dumps(progress, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "private/",
        "_manifest.json",
        ".tfs",
        "private_handoff_state",
    ):
        assert forbidden not in text


def test_public_artifact_exposes_early_global_timeout_closeout_without_phase7_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    calls: list[str] = []
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def forbidden_loop(**_kwargs: Any):
        calls.append("loop")
        raise AssertionError("Phase 7 loop must not run after early timeout closeout")

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", forbidden_loop)
    clock = {"values": [0.0, 100.0]}

    def fake_perf_counter() -> float:
        if clock["values"]:
            return clock["values"].pop(0)
        return 100.0

    monkeypatch.setattr(hmc_kernel_tuning_module.time, "perf_counter", fake_perf_counter)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.diagnostic(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            public_timeout_budget_s=10.0,
        ),
        output_dir=tmp_path,
    )

    progress = json.loads(
        (tmp_path / "hmc_kernel_tuning_progress.json").read_text(encoding="utf-8")
    )
    artifact = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    closeout = artifact["phase7_early_closeout_public_summary"]
    progress_closeout = progress["extra"]["phase7_early_closeout_public_summary"]

    assert calls == []
    assert result.final_status == "hard_veto"
    assert result.tune_verify_repair_loop is None
    assert result.repair_triggers == ("phase7_public_timeout_before_windowed_mass",)
    assert artifact["status"] == "hard_veto"
    assert artifact["attempt_count"] is None
    assert artifact["phase7_public_summary"] is None
    assert artifact["repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert artifact["active_repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert artifact["artifact_policy"][
        "phase7_early_closeout_public_summary_exposed"
    ] is True
    assert closeout["schema"] == "bayesfilter.phase7_early_global_timeout_closeout.v1"
    assert closeout["stage"] == "phase7_before_tune_verify_repair_loop"
    assert closeout["deadline_clock_scope"] == "public_one_call_global"
    assert closeout["timeout_budget_s"] == 10.0
    assert closeout["elapsed_s"] == 100.0
    assert closeout["remaining_s"] == -90.0
    assert closeout["reserve_s"] == 5.0
    assert closeout["bootstrap_passed"] is True
    assert closeout["phase7_loop_entered"] is False
    assert closeout["windowed_mass_runner_called"] is False
    assert closeout["closeout_required_before_phase7_loop"] is True
    assert closeout["hard_veto"] == "phase7_public_timeout_before_windowed_mass"
    assert closeout["repair_trigger"] == "phase7_public_timeout_before_windowed_mass"
    assert closeout["hmc_mechanics_exposed"] is False
    assert closeout["reports_posterior_convergence"] is False
    assert closeout["bootstrap_public_summary"]["final_status"] == "passed"
    assert progress["current_stage"] == "result_written"
    assert progress["last_completed_substage"] == (
        "phase7_public_timeout_before_windowed_mass"
    )
    assert progress_closeout["phase7_loop_entered"] is False
    assert progress_closeout["windowed_mass_runner_called"] is False

    text = json.dumps(
        {
            "closeout": closeout,
            "progress_closeout": progress_closeout,
            "progress_extra": progress["extra"],
        },
        sort_keys=True,
    )
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "private/",
        "_manifest.json",
        ".tfs",
        "private_handoff_state",
    ):
        assert forbidden not in text


def test_public_artifact_exposes_windowed_mass_timeout_closeout_without_mechanics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result_with_windowed_mass_timeout_closeout()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    phase7 = payload["phase7_public_summary"]
    attempt = phase7["attempt_summaries"][0]
    windowed_mass = attempt["stage_statuses"]["windowed_mass"]
    closeout = windowed_mass["public_timeout_closeout"]
    assert result.final_kernel_hash is None
    assert payload["status"] == "hard_veto"
    assert phase7["final_status"] == "hard_veto"
    assert windowed_mass["final_status"] == "hard_veto"
    assert closeout["stage"] == "windowed_mass_segment_start"
    assert closeout["completed_segment_count"] == 5
    assert closeout["planned_segment_count"] == 16
    assert closeout["hard_veto"] == "windowed_mass_public_timeout_soft_deadline"
    assert closeout["public_closeout_artifact_expected"] is True
    assert closeout["hmc_mechanics_exposed"] is False
    assert closeout["reports_posterior_convergence"] is False
    assert "step_size" not in closeout
    assert "num_leapfrog_steps" not in closeout

    text = json.dumps(phase7, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_public_artifact_exposes_phase7_public_timeout_before_windowed_mass_without_mechanics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result_with_phase7_pre_windowed_timeout_closeout()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.diagnostic(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    phase7 = payload["phase7_public_summary"]
    attempt = phase7["attempt_summaries"][0]
    verification = attempt["stage_statuses"]["verification"]
    closeout = verification["public_timeout_closeout"]
    assert result.final_kernel_hash is None
    assert payload["status"] == "hard_veto"
    assert payload["repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert payload["active_repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert payload["historical_repair_triggers"] == []
    assert phase7["final_status"] == "hard_veto"
    assert phase7["repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert phase7["latest_attempt_repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert attempt["repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert attempt["stage_statuses"]["windowed_mass"] is None
    assert verification["verification_ran"] is False
    assert verification["repair_triggers"] == [
        "phase7_public_timeout_before_windowed_mass",
    ]
    assert closeout["stage"] == "phase7_loop_attempt_before_windowed_mass"
    assert closeout["hard_veto"] == "phase7_public_timeout_before_windowed_mass"
    assert closeout["windowed_mass_runner_called"] is False
    assert closeout["closeout_required_before_windowed_mass_runner_build"] is True
    assert closeout["public_closeout_artifact_expected"] is True
    assert closeout["hmc_mechanics_exposed"] is False
    assert closeout["reports_posterior_convergence"] is False
    assert "step_size" not in closeout
    assert "num_leapfrog_steps" not in closeout

    text = json.dumps(phase7, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "private/",
        "_manifest.json",
        ".tfs",
    ):
        assert forbidden not in text


def test_public_tuner_rejects_failed_sequential_rhat_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    loop = _loop_result_with_rhat_cap_public_summary()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)
    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", lambda **_kwargs: loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    payload = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    phase7 = payload["phase7_public_summary"]
    attempt = phase7["attempt_summaries"][0]
    verification = attempt["stage_statuses"]["verification"]
    assert result.final_status == "repair_or_retry"
    assert result.final_kernel_hash is None
    assert result.final_kernel_payload is None
    assert phase7["attempt_count"] == 1
    assert attempt["attempt_index"] == 0
    assert attempt["budget_public_summary"]["public_budget_class"] == (
        "bounded_public_diagnostic_plus"
    )
    assert attempt["budget_public_summary"]["substage_budget_details_exposed"] is False
    assert verification["verification_policy"] == "sequential_rhat"
    assert verification["sequential_rhat_verification"] is True
    assert verification["cap_hit"] is True
    assert verification["all_finite_rhat_at_or_below_threshold"] is False
    assert (
        verification["rhat_threshold_role"]
        == "fixed_kernel_tuning_handoff_gate_not_posterior_proof"
    )
    assert verification["acceptance_relation"] == "inside_acceptance_band"
    assert verification["acceptance_band_from_payload"] is True
    assert verification["acceptance_band_fallback_used"] is False
    assert verification["max_results"] == 64
    assert verification["runner_route_public_summary"]["active_route"] == (
        "phase7_sequential_rhat_fixed_size_chunk_verifier"
    )
    assert verification["hmc_mechanics_exposed"] is False
    assert attempt["private_handoff_payload_exposed"] is False

    text = json.dumps(phase7, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "phase5_tune_budgets",
        "samples",
        "trace",
        "target_log_prob",
        "target_status_trace",
        "target_status_telemetry",
        "mass_artifact_payload",
        "final_state",
        "selected_kernel_payload",
    ):
        assert forbidden not in text


def test_public_smoke_budget_is_capped_contract_policy() -> None:
    factory = hmc_kernel_tuning_module._public_budget_policy_factory(
        HMCKernelTuningConfig.smoke(target_scope="kernel_fixed_mass_step_toy_gaussian")
    )
    assert factory is not None

    policy = factory(314, 0)

    assert policy.serious_policy is False
    assert policy.budget == 64
    assert policy.phase4_warmup_steps == 64
    assert policy.phase5_tune_budgets == (16, 32, 64)
    assert policy.phase5_screen_num_results == 16
    assert policy.phase5_screen_burnin_steps == 4
    assert policy.phase6_screen_num_results == 16
    assert policy.phase6_screen_burnin_steps == 4
    assert policy.verification_num_results == 32
    assert policy.verification_num_burnin_steps == 8


def test_public_progress_artifact_survives_loop_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def failing_loop(**_kwargs: Any) -> None:
        raise RuntimeError("synthetic loop stop")

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", failing_loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    progress_path = tmp_path / "hmc_kernel_tuning_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert result.final_status == "hard_veto"
    assert progress["schema"] == "bayesfilter.hmc_kernel_tuning_progress.v1"
    assert progress["current_stage"] == "result_written"
    assert progress["last_started_stage"] == "loop_start"
    assert progress["last_completed_stage"] == "bootstrap_complete"
    assert progress["last_started_artifact_stage"] == "artifact_finalization_start"
    assert progress["last_completed_artifact_stage"] == "artifact_finalization_complete"
    assert progress["adapter_signature"] == geometry.adapter_signature
    assert progress["target_dimension"] == 2
    assert progress["reports_posterior_convergence"] is False
    assert "no posterior convergence claim" in progress["nonclaims"]


def test_public_progress_artifact_survives_internal_loop_substage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap_passed()
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    def failing_loop(**kwargs: Any) -> None:
        callback = kwargs["_progress_callback"]
        budget_policy = kwargs["_budget_policy_factory"](geometry.target_dimension, 0)
        hmc_kernel_tuning_module._emit_phase7_progress(
            callback,
            "loop_attempt_start",
            attempt_index=0,
            budget_policy=budget_policy,
            started=True,
        )
        hmc_kernel_tuning_module._emit_phase7_progress(
            callback,
            "windowed_mass_start",
            attempt_index=0,
            budget_policy=budget_policy,
            started=True,
        )
        hmc_kernel_tuning_module._emit_phase7_progress(
            callback,
            "windowed_mass_complete",
            attempt_index=0,
            budget_policy=budget_policy,
            completed=True,
        )
        hmc_kernel_tuning_module._emit_phase7_progress(
            callback,
            "fixed_mass_step_start",
            attempt_index=0,
            budget_policy=budget_policy,
            started=True,
        )
        raise RuntimeError("synthetic fixed stage stop")

    monkeypatch.setattr(module, "run_hmc_tune_verify_repair_loop", failing_loop)

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.smoke(
            target_scope="kernel_fixed_mass_step_toy_gaussian"
        ),
        output_dir=tmp_path,
    )

    progress = json.loads(
        (tmp_path / "hmc_kernel_tuning_progress.json").read_text(encoding="utf-8")
    )
    assert result.final_status == "hard_veto"
    assert progress["current_stage"] == "result_written"
    assert progress["last_started_stage"] == "fixed_mass_step_start"
    assert progress["last_completed_stage"] == "windowed_mass_complete"
    assert progress["last_started_artifact_stage"] == "artifact_finalization_start"
    assert progress["last_completed_artifact_stage"] == "artifact_finalization_complete"
    assert progress["last_started_substage"] == "fixed_mass_step_start"
    assert progress["last_completed_substage"] == "windowed_mass_complete"
    assert progress["phase7_last_attempt_index"] == 0
    assert progress["phase7_last_budget_payload"]["target_dimension"] == 2
    assert progress["phase7_last_budget_payload"]["budget"] == 8
    assert progress["phase7_last_budget_payload"]["substage_budget_details_exposed"] is False
    assert progress["phase7_last_budget_payload"]["hmc_mechanics_exposed"] is False
    assert progress["reports_posterior_convergence"] is False


def test_public_exports_are_available() -> None:
    assert bayesfilter.HMCKernelTuningConfig is HMCKernelTuningConfig
    assert bayesfilter.HMCKernelTuningResult is HMCKernelTuningResult
    assert bayesfilter.tune_hmc_kernel is tune_hmc_kernel
    assert bayesfilter.HMC_KERNEL_TUNING_PUBLIC_NONCLAIMS is HMC_KERNEL_TUNING_PUBLIC_NONCLAIMS
    assert "tune_hmc_kernel" in bayesfilter.__all__
    assert "no posterior convergence claim" in HMC_KERNEL_TUNING_PUBLIC_NONCLAIMS
    private_phase7_names = {
        "_HMCPhase7FixedKernelVerificationInput",
        "_HMCPhase7FixedKernelVerificationOutcome",
        "_HMCPhase7FixedKernelVerificationExecution",
        "_phase7_direct_candidate_seed",
        "_phase7_direct_candidate_verification_input",
        "_run_phase7_direct_candidate_verification",
        "_run_phase7_fixed_kernel_verification",
        "KillableChildSpec",
        "build_hmc_tuning_engineering_artifact",
        "run_killable_child",
        "validate_hmc_tuning_engineering_artifact",
    }
    assert private_phase7_names.isdisjoint(bayesfilter.__all__)
    assert not any(hasattr(bayesfilter, name) for name in private_phase7_names)
    loop_parameters = inspect.signature(
        bayesfilter.run_hmc_tune_verify_repair_loop
    ).parameters
    assert "_phase7_direct_verification_runner" not in loop_parameters


def test_public_artifacts_include_sanitized_bootstrap_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    acceptances = [0.95, 0.20]
    runtimes = [0.2, 0.4]
    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_bootstrap_config(
            max_repairs=1,
            chain_execution_mode="tf_function",
            use_xla=True,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=lambda _adapter, _initial_state, config: _bootstrap_fake_result(
            acceptance=acceptances.pop(0),
            runtime_s=runtimes.pop(0),
            use_xla=bool(config.use_xla),
            timing_scope="synthetic_bootstrap_timing_scope",
        ),
    )
    module = __import__("bayesfilter.inference.hmc_kernel_tuning", fromlist=[""])
    monkeypatch.setattr(module, "initialize_hmc_kernel_geometry", lambda **_kwargs: geometry)
    monkeypatch.setattr(module, "run_hmc_bootstrap_screen", lambda **_kwargs: bootstrap)

    monkeypatch.setattr(
        module,
        "run_hmc_tune_verify_repair_loop",
        lambda **_kwargs: _loop_result_for_bootstrap(
            bootstrap.artifact_hash,
            passed=False,
        ),
    )

    result = tune_hmc_kernel(
        adapter=_ToyGaussianAdapter(),
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.standard(
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            bootstrap_max_repairs=1,
            use_xla=True,
        ),
        output_dir=tmp_path,
    )

    assert result.final_status == "budget_exhausted"
    result_payload = json.loads(
        (tmp_path / "hmc_kernel_tuning_result.json").read_text(encoding="utf-8")
    )
    assert result_payload["bootstrap_public_summary"]["final_status"] == (
        "repair_budget_exhausted"
    )
    assert result_payload["bootstrap_public_summary"]["preflight_passed"] is True
    assert result_payload["bootstrap_public_summary"]["acceptance_promoted_kernel"] is False
    assert result_payload["bootstrap_public_summary"]["handoff_kernel_source"] == (
        "geometry_preflight_fallback"
    )
    progress_payload = json.loads(
        (tmp_path / "hmc_kernel_tuning_progress.json").read_text(encoding="utf-8")
    )
    allowed = {
        "round_count",
        "final_status",
        "preflight_passed",
        "acceptance_promoted_kernel",
        "handoff_kernel_source",
        "last_round_index",
        "last_classification",
        "last_diagnostic_role",
        "last_acceptance_relation_to_band",
        "observed_acceptance_relations",
        "oscillatory_acceptance_repair_observed",
        "hard_veto_present",
        "hard_veto_categories",
        "leapfrog_cap_saturation_observed",
        "leapfrog_cap_saturation_direction",
        "xla_requested",
        "jit_compile_metadata",
        "round_timing_available",
        "max_round_runtime_s",
        "timing_scope",
        "metadata_limitations",
    }
    forbidden = {
        "step_size",
        "num_leapfrog_steps",
        "unclamped_num_leapfrog_steps",
        "target_trajectory_length",
        "mass_artifact_payload",
        "screen_config_payload",
        "acceptance_rate",
        "repair_band",
        "budget_schedule",
    }
    for payload in (result_payload, progress_payload):
        summary = payload["bootstrap_public_summary"]
        assert set(summary) == allowed
        assert set(summary).isdisjoint(forbidden)
        assert summary["round_count"] == 2
        assert summary["final_status"] == "repair_budget_exhausted"
        assert summary["last_round_index"] == 1
        assert summary["last_classification"] == "repair"
        assert summary["last_diagnostic_role"] == "bootstrap_acceptance_repair_trigger"
        assert summary["last_acceptance_relation_to_band"] == "below"
        assert summary["observed_acceptance_relations"] == ["above", "below"]
        assert summary["oscillatory_acceptance_repair_observed"] is True
        assert summary["hard_veto_present"] is False
        assert summary["hard_veto_categories"] == []
        assert summary["xla_requested"] is True
        assert summary["jit_compile_metadata"] == "true"
        assert summary["round_timing_available"] is True
        assert summary["max_round_runtime_s"] == pytest.approx(0.4)
        assert summary["timing_scope"] == "synthetic_bootstrap_timing_scope"
        summary_text = json.dumps(summary, sort_keys=True)
        assert all(name not in summary_text for name in forbidden)
