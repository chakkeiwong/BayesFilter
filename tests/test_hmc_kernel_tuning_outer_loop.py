from __future__ import annotations

import inspect
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

import bayesfilter
import bayesfilter.inference.hmc_budget_ladder as hmc_budget_ladder
import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning
from bayesfilter.inference.hmc_coordinates import WarmupTrajectoryPolicy
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)
from bayesfilter.inference import (
    FixedMassHMCTuningBudgetCallbackResult,
    HMCTuneVerifyRepairAttempt,
    HMCTuneVerifyRepairLoopConfig,
    HMCTuneVerifyRepairLoopResult,
    SequentialRHatCheckpointWriterConfig,
    TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS,
    build_retained_frozen_kernel_hmc_adapter_from_tuning_payload,
    run_hmc_tune_verify_repair_loop,
    stable_adapter_signature,
)
from bayesfilter.inference.hmc_kernel_tuning import (
    _HMCAttemptBudgetPolicy,
    _HMCPhaseAttemptState,
    _default_attempt_budget_policy,
    _fixed_mass_step_initial_step,
    _phase6_trajectory_repair_handoff_payload,
    _phase7_extended_attempt_stall_blocker,
    _phase7_progress_budget_payload,
    _phase7_verification_repair_handoff_payload,
    _public_budget_policy_factory,
)

from tests.test_hmc_kernel_tuning_fixed_mass_step import (
    _ToyGaussianAdapter,
    _bootstrap,
    _fake_result,
    _geometry,
    _scripted_step_runner,
    _windowed_stage,
)
from tests.test_hmc_kernel_tuning_windowed_mass import (
    _operational_budget,
    _operational_inputs,
)


def _loop_config(**overrides: Any) -> HMCTuneVerifyRepairLoopConfig:
    payload = {
        "target_accept_prob": 0.70,
        "acceptance_band": (0.65, 0.75),
        "repair_band": (0.55, 0.85),
        "max_attempts": 5,
        "seed": (20260621, 70),
        "chain_execution_mode": "eager",
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
    }
    payload.update(overrides)
    return HMCTuneVerifyRepairLoopConfig(**payload)


def _tiny_budget_factory(_dimension: int, attempt_index: int) -> _HMCAttemptBudgetPolicy:
    budget = 8 * (2 ** int(attempt_index))
    screen = max(4, budget // 2)
    verification = max(4, budget // 2)
    return _HMCAttemptBudgetPolicy(
        target_dimension=2,
        attempt_index=int(attempt_index),
        budget=budget,
        phase4_warmup_steps=12,
        phase5_tune_budgets=(2 * (2 ** int(attempt_index)), 4 * (2 ** int(attempt_index)), budget),
        phase5_screen_num_results=screen,
        phase5_screen_burnin_steps=1,
        phase6_screen_num_results=screen,
        phase6_screen_burnin_steps=1,
        verification_num_results=verification,
        verification_num_burnin_steps=1,
        serious_policy=False,
    )


def _verification_budget_factory(
    _dimension: int,
    attempt_index: int,
) -> _HMCAttemptBudgetPolicy:
    return replace(
        _tiny_budget_factory(_dimension, attempt_index),
        verification_num_results=64,
    )


def _acceptance_evidence_payload(
    acceptance: float,
    *,
    draw_count: int = 64,
    cost_stop_reasons: tuple[str, ...] = (),
    native_divergence_count: int | None = None,
) -> Mapping[str, Any]:
    draw = np.arange(draw_count, dtype=float)[:, None, None]
    chain = np.arange(4, dtype=float)[None, :, None]
    samples = draw * np.array([1.0, -0.5])[None, None, :] + chain
    probability = np.full((draw_count, 4), float(acceptance))
    return evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(probability),
        is_accepted=np.ones((draw_count, 4), dtype=bool),
        policy=HMCAcceptancePolicy(
            allowed_cost_stop_reasons=("persistent_candidate_cost_stop",)
            if cost_stop_reasons
            else ()
        ),
        native_divergence_status=(
            "not_exposed_by_kernel"
            if native_divergence_count is None
            else "available"
        ),
        native_divergence_count=native_divergence_count,
        cost_stop_reasons=cost_stop_reasons,
    ).payload()


def _sequential_verification_diagnostics(
    acceptance: float,
    *,
    draw_count: int = 64,
    rhat_passed: bool = False,
    cost_stop_reasons: tuple[str, ...] = (),
    native_divergence_count: int | None = None,
) -> Mapping[str, Any]:
    evidence = _acceptance_evidence_payload(
        acceptance,
        draw_count=draw_count,
        cost_stop_reasons=cost_stop_reasons,
        native_divergence_count=native_divergence_count,
    )
    return {
        "sequential_rhat_verification": True,
        "all_finite_rhat_at_or_below_threshold": bool(rhat_passed),
        "cap_hit": evidence["decision"] == "inconclusive_evidence",
        "rhat_threshold": 1.01,
        "check_interval": int(draw_count),
        "max_results": int(draw_count),
        "retained_sample_count": int(draw_count),
        "verification_min_retained_results_for_pass": int(draw_count),
        "verification_min_retained_pass_gate_satisfied": True,
        "runtime_finite": True,
        "samples_all_finite": True,
        "target_value_health_passed": True,
        "acceptance_log_health_passed": True,
        "acceptance_rate": evidence["pooled_mean"],
        "acceptance_evidence": evidence,
    }


def _operational_exact_l_retune_result(
    *,
    num_results: int,
    step_size: float,
    dimension: int = 2,
):
    samples = np.zeros((int(num_results), 4, int(dimension)))
    probability = np.full((int(num_results), 4), 0.70)
    return replace(
        _fake_result(
            num_results=int(num_results),
            acceptance=0.70,
            step_size=float(step_size),
            samples=samples,
        ),
        trace={
            "log_accept_ratio": np.log(probability),
            "is_accepted": np.ones_like(probability, dtype=bool),
            "target_log_prob": np.zeros_like(probability),
            "step_size": np.full(int(num_results), float(step_size)),
        },
        diagnostics={"final_step_size": float(step_size)},
    )


def _passed_phase7_stage_fixtures():
    windowed = _windowed_stage()
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=_scripted_full_chain_runner(verification_acceptances=[0.70])[0],
    )

    def trajectory_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        run_full_chain=trajectory_run,
    )
    assert windowed.passed is True
    assert fixed.passed is True
    assert trajectory.passed is True
    return windowed, fixed, trajectory


def _scripted_full_chain_runner(
    *,
    verification_acceptances: list[float],
    phase5_screen_acceptances: list[float] | None = None,
    trajectory_acceptance: float | None = None,
):
    calls: list[Mapping[str, Any]] = []
    phase = "windowed"
    phase6_seed_first: int | None = None

    def run(adapter: Any, initial_state: Any, config: Any):
        nonlocal phase, phase6_seed_first
        uses_tuning = bool(config.tuning_policy.uses_dual_averaging)
        call = {
            "adapter_signature": adapter.adapter_signature(),
            "initial_state": np.asarray(initial_state, dtype=float),
            "num_results": int(config.num_results),
            "num_burnin_steps": int(config.num_burnin_steps),
            "step_size": float(config.step_size),
            "num_leapfrog_steps": int(config.num_leapfrog_steps),
            "seed": tuple(config.seed),
            "trace_policy": config.trace_policy,
            "adaptation_policy": config.adaptation_policy,
            "uses_dual_averaging": uses_tuning,
            "target_scope": config.target_scope,
            "use_xla": bool(config.use_xla),
        }
        calls.append(call)
        if uses_tuning:
            phase = "phase5_screen"
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                step_size=0.20 + 0.01 * len(
                    [item for item in calls if item["uses_dual_averaging"]]
                ),
                num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
                samples=np.zeros((int(config.num_results), 2)),
            )
        if int(config.num_results) == 12:
            phase = "phase5_tune"
            phase6_seed_first = None
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                samples=np.zeros((int(config.num_results), 2)),
            )
        if phase == "phase5_tune" and int(config.num_burnin_steps) == 1:
            phase = "phase6"
            acceptance = (
                0.70
                if not phase5_screen_acceptances
                else phase5_screen_acceptances.pop(0)
            )
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=acceptance,
                samples=np.zeros((int(config.num_results), 2)),
            )
        if phase == "phase5_screen":
            phase = "phase6"
            acceptance = (
                0.70
                if not phase5_screen_acceptances
                else phase5_screen_acceptances.pop(0)
            )
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=acceptance,
                samples=np.zeros((int(config.num_results), 2)),
            )
        if trajectory_acceptance is not None and phase == "phase6" and (
            phase6_seed_first is None or int(config.seed[0]) == phase6_seed_first
        ):
            phase6_seed_first = int(config.seed[0])
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=trajectory_acceptance,
                samples=np.zeros((int(config.num_results), 2)),
            )
        phase = "windowed"
        phase6_seed_first = None
        acceptance = verification_acceptances.pop(0)
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=acceptance,
            samples=np.zeros((int(config.num_results), 2)),
        )

    return run, calls


def _replay_tuning_payload() -> Mapping[str, Any]:
    geometry = _geometry()
    bootstrap = _bootstrap()
    run, _calls = _scripted_full_chain_runner(verification_acceptances=[0.70])
    loop = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=1),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )
    assert loop.passed is True
    public_final = hmc_kernel_tuning._public_final_kernel_handoff_payload(loop)
    return {
        "schema": "bayesfilter.hmc_kernel_tuning_result.v1",
        "config": _loop_config(max_attempts=1).payload(),
        "adapter_signature": stable_adapter_signature(_ToyGaussianAdapter()),
        "target_dimension": 2,
        "geometry_artifact_hash": geometry.artifact_hash,
        "bootstrap_artifact_hash": bootstrap.artifact_hash,
        "loop_artifact_hash": loop.artifact_hash,
        "geometry": geometry.payload(include_mass_arrays=False),
        "bootstrap": bootstrap.payload(),
        "tune_verify_repair_loop": loop.payload(include_final_mass_arrays=True),
        "final_status": "passed",
        "diagnostic_role": "fresh_fixed_kernel_verified",
        "hard_vetoes": (),
        "repair_triggers": (),
        "final_kernel_payload": public_final,
        "final_kernel_hash": hmc_kernel_tuning.stable_config_hash(public_final),
        "artifact_path": None,
        "diagnostic_roles": {},
        "passed": True,
    }


def _phase7_direct_fixture() -> tuple[
    Any,
    Any,
    Any,
    Any,
    Any,
    tuple[int, int, str, int],
    tuple[int, int, str, int],
]:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.66, 5: 0.70, 7: 0.73})
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(fixed)
    assert handoff is not None
    selected = handoff.candidate_records[handoff.selected_batch_ordinal]
    alternative = next(
        record
        for record in handoff.candidate_records
        if record.handoff_eligible and record.batch_ordinal != selected.batch_ordinal
    )

    def identity(record: Any) -> tuple[int, int, str, int]:
        return (
            record.batch_ordinal,
            record.source_round_index,
            record.source_grid_stage,
            record.source_round_candidate_index,
        )

    return (
        geometry,
        bootstrap,
        windowed,
        fixed,
        handoff,
        identity(selected),
        identity(alternative),
    )


def _historical_phase5_stage(
    stage: hmc_kernel_tuning.HMCFixedMassStepStageResult,
) -> hmc_kernel_tuning.HMCFixedMassStepStageResult:
    """Build the only reviewed no-batch fixture allowed to enter Phase 6."""

    return replace(
        stage,
        diagnostics={
            **dict(stage.diagnostics),
            "algorithm": "historical_fixed_mass_step_fixture",
            "candidate_count": 0,
            "historical_phase5_compatibility_fixture": True,
        },
        _candidate_batch_handoff=None,
    )


def _run_historical_phase5_stage(
    **kwargs: Any,
) -> hmc_kernel_tuning.HMCFixedMassStepStageResult:
    return _historical_phase5_stage(
        hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(**kwargs)
    )


def test_tune_verify_repair_config_does_not_expose_hmc_mechanics_or_budgets() -> None:
    parameters = set(inspect.signature(HMCTuneVerifyRepairLoopConfig).parameters)
    forbidden = {
        "step_size",
        "num_leapfrog_steps",
        "candidate_l_values",
        "mass_window_schedule",
        "warmup_steps",
        "budget_schedule",
        "tune_num_results",
        "screen_num_results",
        "screen_num_burnin_steps",
        "verification_num_results",
        "verification_num_burnin_steps",
        "trajectory_grid",
    }

    assert parameters.isdisjoint(forbidden)
    payload = HMCTuneVerifyRepairLoopConfig(
        step_repair_factor=2.5,
        step_repair_high_acceptance_directional_factor=3.0,
        verification_chunk_max_results=250,
        verification_min_retained_results_for_pass=1000,
    ).payload()
    assert payload["step_repair_factor"] == pytest.approx(2.5)
    assert payload["step_repair_high_acceptance_directional_factor"] == pytest.approx(3.0)
    assert payload["trajectory_window_lower_multiplier"] == pytest.approx(0.3)
    assert payload["trajectory_window_upper_multiplier"] == pytest.approx(3.0)
    assert payload["verification_chunk_max_results"] == 250
    assert payload["verification_min_retained_results_for_pass"] == 1000
    assert HMCTuneVerifyRepairLoopConfig(max_attempts=10).max_attempts == 10
    with pytest.raises(ValueError, match="hard-capped"):
        HMCTuneVerifyRepairLoopConfig(max_attempts=11)
    with pytest.raises(ValueError, match="verification_chunk_max_results"):
        HMCTuneVerifyRepairLoopConfig(verification_chunk_max_results=0)
    with pytest.raises(ValueError, match="verification_min_retained_results_for_pass"):
        HMCTuneVerifyRepairLoopConfig(verification_min_retained_results_for_pass=0)


def test_incall_progress_heartbeat_propagates_to_fixed_mass_budget_ladder() -> None:
    loop_config = HMCTuneVerifyRepairLoopConfig(incall_progress_heartbeat_s=60.0)

    fixed_stage_config = hmc_kernel_tuning._phase7_fixed_step_stage_config(
        loop_config,
        attempt_index=0,
    )
    ladder_config = hmc_kernel_tuning._fixed_mass_step_stage_ladder_config(
        fixed_stage_config,
        initial_step=0.1,
        num_leapfrog_steps=8,
        target_scope="test_target",
    )

    assert fixed_stage_config.incall_progress_heartbeat_s == pytest.approx(60.0)
    assert ladder_config.incall_progress_heartbeat_s == pytest.approx(60.0)


def test_default_budget_policy_matches_reviewed_phase7_mapping() -> None:
    policy0 = _default_attempt_budget_policy(17, 0)
    policy1 = _default_attempt_budget_policy(17, 1)

    assert policy0.budget == 1000
    assert policy0.phase4_warmup_steps == 1000
    assert policy0.phase5_tune_budgets == (250, 500, 1000)
    assert policy0.phase5_screen_num_results == 250
    assert policy0.phase5_screen_burnin_steps == 63
    assert policy0.phase6_screen_num_results == 250
    assert policy0.phase6_screen_burnin_steps == 63
    assert policy0.verification_num_results == 500
    assert policy0.verification_num_burnin_steps == 125
    assert policy1.budget == 2000
    assert policy1.verification_num_results == 1000
    assert policy1.phase5_tune_budgets == (500, 1000, 2000)


def test_default_budget_policy_caps_ccma_scale_serious_attempts() -> None:
    policy0 = _default_attempt_budget_policy(314, 0)
    policy1 = _default_attempt_budget_policy(314, 1)
    policy2 = _default_attempt_budget_policy(314, 2)

    assert policy0.budget == 5000
    assert policy0.phase5_tune_budgets == (1250, 2500, 5000)
    assert policy0.phase5_screen_num_results == 1250
    assert policy0.phase5_screen_burnin_steps == 313
    assert policy0.phase6_screen_num_results == 1250
    assert policy0.phase6_screen_burnin_steps == 313
    assert policy0.verification_num_results == 2500
    assert policy0.verification_num_burnin_steps == 625
    assert policy1.budget == 10000
    assert policy1.phase5_tune_budgets == (2500, 5000, 10000)
    assert policy1.phase6_screen_num_results == 2500
    assert policy1.phase6_screen_burnin_steps == 625
    assert policy1.verification_num_results == 5000
    assert policy2.budget == 10000


def test_geometry_scaled_budget_policy_increases_with_condition_number() -> None:
    policy = hmc_kernel_tuning.HMCGeometryScaledBudgetTimingPolicy(
        dimension_factor=2.0,
        min_initial_budget=10,
        max_initial_budget=10_000,
        max_tune_budget=20_000,
    )
    low_mass = hmc_kernel_tuning.PrecomputedMassArtifact.from_covariance(
        position=[0.0, 0.0, 0.0, 0.0],
        covariance=np.eye(4),
        adapter_signature="test_geometry_budget_condition_adapter",
        position_role="test",
        covariance_source="test_identity",
        jitter=0.0,
        regularization_report={"method": "none"},
    )
    high_mass = hmc_kernel_tuning.PrecomputedMassArtifact.from_covariance(
        position=[0.0, 0.0, 0.0, 0.0],
        covariance=np.diag([1.0, 10.0, 100.0, 1000.0]),
        adapter_signature="test_geometry_budget_condition_adapter",
        position_role="test",
        covariance_source="test_ill_conditioned",
        jitter=0.0,
        regularization_report={"method": "none"},
    )

    low_payload = policy.attempt_budget_payload(
        target_dimension=4,
        attempt_index=0,
        mass_artifact=low_mass,
    )
    high_payload = policy.attempt_budget_payload(
        target_dimension=4,
        attempt_index=0,
        mass_artifact=high_mass,
    )

    assert high_payload["budget"] > low_payload["budget"]
    assert (
        high_payload["geometry_budget_summary"]["condition_number"]
        > low_payload["geometry_budget_summary"]["condition_number"]
    )
    assert high_payload["geometry_budget_summary"]["raw_eigenvalues_exposed"] is False
    assert high_payload["geometry_budget_summary"]["mass_arrays_exposed"] is False


def test_geometry_scaled_budget_policy_increases_with_low_effective_dimension() -> None:
    policy = hmc_kernel_tuning.HMCGeometryScaledBudgetTimingPolicy(
        dimension_factor=2.0,
        min_initial_budget=10,
        max_initial_budget=10_000,
        max_tune_budget=20_000,
        condition_log10_weight=0.0,
    )
    isotropic = hmc_kernel_tuning.PrecomputedMassArtifact.from_covariance(
        position=[0.0] * 8,
        covariance=np.eye(8),
        adapter_signature="test_geometry_budget_effective_dimension_adapter",
        position_role="test",
        covariance_source="test_isotropic",
        jitter=0.0,
        regularization_report={"method": "none"},
    )
    anisotropic = hmc_kernel_tuning.PrecomputedMassArtifact.from_covariance(
        position=[0.0] * 8,
        covariance=np.diag([1000.0] + [1.0] * 7),
        adapter_signature="test_geometry_budget_effective_dimension_adapter",
        position_role="test",
        covariance_source="test_low_effective_dimension",
        jitter=0.0,
        regularization_report={"method": "none"},
    )

    base = policy.attempt_budget_payload(
        target_dimension=8,
        attempt_index=0,
        mass_artifact=isotropic,
    )
    stressed = policy.attempt_budget_payload(
        target_dimension=8,
        attempt_index=0,
        mass_artifact=anisotropic,
    )

    assert stressed["budget"] > base["budget"]
    assert (
        stressed["geometry_budget_summary"]["effective_dimension"]
        < base["geometry_budget_summary"]["effective_dimension"]
    )


def test_geometry_scaled_budget_policy_increases_with_regularization_pressure() -> None:
    policy = hmc_kernel_tuning.HMCGeometryScaledBudgetTimingPolicy(
        dimension_factor=2.0,
        min_initial_budget=10,
        max_initial_budget=10_000,
        max_tune_budget=20_000,
        condition_log10_weight=0.0,
        anisotropy_sqrt_weight=0.0,
    )
    base_mass = hmc_kernel_tuning.PrecomputedMassArtifact.from_covariance(
        position=[0.0] * 4,
        covariance=np.eye(4),
        adapter_signature="test_geometry_budget_regularization_adapter",
        position_role="test",
        covariance_source="test_base",
        jitter=0.0,
        regularization_report={"method": "none"},
    )
    repaired_mass = hmc_kernel_tuning.PrecomputedMassArtifact.from_covariance(
        position=[0.0] * 4,
        covariance=np.eye(4),
        adapter_signature="test_geometry_budget_regularization_adapter",
        position_role="test",
        covariance_source="test_regularized",
        jitter=0.0,
        regularization_report={
            "clipped_eigenvalue_count": 2,
            "raw_nonpositive_eigenvalue_count": 1,
            "diagonal_fallback_used": True,
        },
    )

    base = policy.attempt_budget_payload(
        target_dimension=4,
        attempt_index=0,
        mass_artifact=base_mass,
    )
    repaired = policy.attempt_budget_payload(
        target_dimension=4,
        attempt_index=0,
        mass_artifact=repaired_mass,
    )

    assert repaired["budget"] > base["budget"]
    assert repaired["geometry_budget_summary"]["regularization_pressure"][
        "diagonal_fallback_used"
    ] is True


def test_geometry_scaled_policy_payload_is_public_safe() -> None:
    policy = hmc_kernel_tuning.HMCGeometryScaledBudgetTimingPolicy()
    payload = policy.payload()
    text = json.dumps(payload, sort_keys=True)

    assert payload["emergency_timing_policy"]["role"] == (
        "machine_protection_only_not_scientific_stop_rule"
    )
    assert payload["raw_eigenvalues_exposed"] is False
    assert payload["mass_arrays_exposed"] is False
    for forbidden in (
        '"step_size"',
        '"num_leapfrog_steps"',
        '"covariance"',
        '"factor"',
        '"samples"',
    ):
        assert forbidden not in text


def test_public_diagnostic_budget_policy_is_bounded_and_non_serious() -> None:
    config = hmc_kernel_tuning.HMCKernelTuningConfig.diagnostic(use_xla=True)
    factory = _public_budget_policy_factory(config)

    assert factory is not None
    assert config.preset == "diagnostic"
    assert config.max_attempts == 2
    assert config.uses_serious_budget_policy is False

    policy0 = factory(4, 0)
    policy1 = factory(4, 1)
    payload0 = policy0.payload()
    progress_payload = _phase7_progress_budget_payload(policy0)

    assert policy0.budget == 32
    assert policy1.budget == 64
    assert policy0.phase5_tune_budgets == (8, 16, 32)
    assert policy0.phase5_screen_num_results == 8
    assert policy0.phase5_screen_burnin_steps == 2
    assert policy0.phase6_screen_num_results == 8
    assert policy0.phase6_screen_burnin_steps == 2
    assert policy0.verification_num_results == 16
    assert policy0.verification_num_burnin_steps == 4
    assert payload0["serious_policy"] is False
    assert payload0["public_budget_class"] == "bounded_public_diagnostic"
    assert payload0["public_budget_cap"] == 64
    assert payload0["public_max_attempts"] == 2
    assert payload0["public_diagnostic_preset"] == "diagnostic"
    assert payload0["diagnostic_role"] == "public_bounded_timeout_diagnostic"
    assert payload0["nonclaims"] == TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS
    assert progress_payload["serious_policy"] is False
    assert progress_payload["public_budget_class"] == "bounded_public_diagnostic"
    assert progress_payload["public_budget_cap"] == 64
    assert progress_payload["public_max_attempts"] == 2
    assert progress_payload["public_diagnostic_preset"] == "diagnostic"
    assert progress_payload["diagnostic_role"] == "public_bounded_timeout_diagnostic"
    assert progress_payload["hmc_mechanics_exposed"] is False
    for forbidden in ("step_size", "num_leapfrog_steps", "mass_artifact_payload", "final_state"):
        assert forbidden not in progress_payload


def test_public_diagnostic_plus_budget_policy_is_bounded_and_non_serious() -> None:
    config = hmc_kernel_tuning.HMCKernelTuningConfig.diagnostic_plus(use_xla=True)
    factory = _public_budget_policy_factory(config)

    assert factory is not None
    assert config.preset == "diagnostic_plus"
    assert config.max_attempts == 2
    assert config.uses_serious_budget_policy is False

    policy0 = factory(4, 0)
    policy1 = factory(4, 1)
    policy2 = factory(4, 2)
    payload0 = policy0.payload()
    progress_payload = _phase7_progress_budget_payload(policy1)

    assert policy0.budget == 128
    assert policy1.budget == 256
    assert policy2.budget == 256
    assert policy0.phase5_tune_budgets == (32, 64, 128)
    assert policy1.phase5_tune_budgets == (64, 128, 256)
    assert policy0.phase5_screen_num_results == 32
    assert policy1.phase5_screen_num_results == 64
    assert policy0.phase6_screen_num_results == 32
    assert policy1.phase6_screen_num_results == 64
    assert policy0.verification_num_results == 64
    assert policy1.verification_num_results == 128
    assert payload0["serious_policy"] is False
    assert payload0["public_budget_class"] == "bounded_public_diagnostic_plus"
    assert payload0["public_budget_cap"] == 256
    assert payload0["public_max_attempts"] == 2
    assert payload0["public_diagnostic_preset"] == "diagnostic_plus"
    assert payload0["diagnostic_role"] == "public_bounded_verification_diagnostic_plus"
    assert payload0["nonclaims"] == TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS
    assert progress_payload["serious_policy"] is False
    assert progress_payload["public_budget_class"] == "bounded_public_diagnostic_plus"
    assert progress_payload["public_budget_cap"] == 256
    assert progress_payload["public_max_attempts"] == 2
    assert progress_payload["public_diagnostic_preset"] == "diagnostic_plus"
    assert progress_payload["diagnostic_role"] == "public_bounded_verification_diagnostic_plus"
    assert progress_payload["hmc_mechanics_exposed"] is False
    for forbidden in ("step_size", "num_leapfrog_steps", "mass_artifact_payload", "final_state"):
        assert forbidden not in progress_payload


def test_public_diagnostic_plus_passed_result_writes_public_artifact_payload() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    run, _calls = _scripted_full_chain_runner(verification_acceptances=[0.70])
    loop = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=1),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )
    final_kernel_payload = hmc_kernel_tuning._public_final_kernel_handoff_payload(loop)
    result = hmc_kernel_tuning.HMCKernelTuningResult(
        config=hmc_kernel_tuning.HMCKernelTuningConfig.diagnostic_plus(),
        adapter_signature=stable_adapter_signature(_ToyGaussianAdapter()),
        target_dimension=2,
        geometry=geometry,
        bootstrap=bootstrap,
        tune_verify_repair_loop=loop,
        final_status="passed",
        diagnostic_role=loop.diagnostic_role,
        hard_vetoes=(),
        repair_triggers=(),
        final_kernel_payload=final_kernel_payload,
        final_kernel_hash=hmc_kernel_tuning.stable_config_hash(final_kernel_payload),
        artifact_path="hmc_kernel_tuning_result.json",
        diagnostic_roles={},
    )

    payload = result.payload()
    artifact = hmc_kernel_tuning._public_tuning_artifact_payload(result)

    assert payload["config"]["preset"] == "diagnostic_plus"
    assert payload["config"]["preset_role"] == "bounded_public_verification_diagnostic_only"
    assert artifact["config"]["preset"] == "diagnostic_plus"
    assert artifact["config"]["preset_role"] == "bounded_public_verification_diagnostic_only"
    assert artifact["status"] == "passed"
    assert artifact["final_kernel_hash"] == result.final_kernel_hash
    public_text = json.dumps(artifact["final_kernel_payload"], sort_keys=True)
    for forbidden in (
        '"step_size"',
        '"num_leapfrog_steps"',
        '"trajectory_length"',
        '"adapted_mass_artifact_payload"',
        '"position"',
        '"covariance"',
        '"factor"',
    ):
        assert forbidden not in public_text
    loop_public = result.payload()["tune_verify_repair_loop"]
    assert loop_public["final_kernel_payload"]["private_replay_payload"] is False
    loop_text = json.dumps(loop_public["final_kernel_payload"], sort_keys=True)
    assert '"step_size"' not in loop_text
    assert '"num_leapfrog_steps"' not in loop_text
    assert '"adapted_mass_artifact_payload"' not in loop_text


def test_phase7_extended_attempt_gate_requires_meaningful_progress() -> None:
    config = _loop_config(max_attempts=10)
    next_policy = _tiny_budget_factory(2, 5)
    last_attempt = HMCTuneVerifyRepairAttempt(
        attempt_index=4,
        budget_policy_payload=_tiny_budget_factory(2, 4).payload(),
        incoming_state_payload=None,
        windowed_stage=None,
        fixed_mass_step_stage=None,
        frozen_step_trajectory_stage=None,
        verification_config_payload=None,
        verification_diagnostics={
            "attempt_index": 4,
            "not_run": True,
            "reports_posterior_convergence": False,
        },
        verification_callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        final_status="repair_or_retry",
        diagnostic_role="repair_trigger",
        hard_vetoes=(),
        repair_triggers=(),
        handoff_state_payload=None,
    )
    stalled_state = _HMCPhaseAttemptState(
        mass_artifact_payload={"schema": "test.mass"},
        mass_artifact_signature="mass-hash",
        selected_step_size=0.2,
        selected_step_hash="step-hash",
        handoff_stage="phase5_selected",
    )

    blocker = _phase7_extended_attempt_stall_blocker(
        config=config,
        attempt_state=stalled_state,
        last_attempt=last_attempt,
        next_attempt_policy=next_policy,
    )

    assert blocker is not None
    assert blocker["classification"] == (
        "phase7_extended_attempt_stalled_no_meaningful_progress"
    )
    assert blocker["base_max_attempts"] == 5
    assert blocker["configured_max_attempts"] == 10
    assert blocker["next_attempt_index"] == 5
    assert blocker["stalled_reason"] == (
        "previous_attempt_left_no_effective_repair_progress"
    )
    assert blocker["hmc_mechanics_exposed"] is False

    repairing_attempt = HMCTuneVerifyRepairAttempt(
        attempt_index=4,
        budget_policy_payload=_tiny_budget_factory(2, 4).payload(),
        incoming_state_payload=None,
        windowed_stage=None,
        fixed_mass_step_stage=None,
        frozen_step_trajectory_stage=None,
        verification_config_payload=None,
        verification_diagnostics={
            "attempt_index": 4,
            "not_run": True,
            "reports_posterior_convergence": False,
        },
        verification_callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        final_status="repair_or_retry",
        diagnostic_role="repair_trigger",
        hard_vetoes=(),
        repair_triggers=("phase6_trajectory_acceptance_outside_pass_band",),
        handoff_state_payload=None,
    )
    repairing_state = _HMCPhaseAttemptState(
        mass_artifact_payload={"schema": "test.mass"},
        mass_artifact_signature="mass-hash",
        selected_step_size=0.2,
        selected_step_hash="step-hash",
        selected_num_leapfrog_steps=8,
        selected_trajectory_hash="trajectory-hash",
        verification_acceptance_rate=0.89,
        verification_acceptance_relation="above_acceptance_band",
        verification_repair_trigger="phase6_trajectory_acceptance_outside_pass_band",
        verification_repair_source="phase6_frozen_step_trajectory_acceptance",
        verification_repair_step_size=0.4,
        verification_repair_step_hash="repair-step-hash",
        verification_repair_applied=True,
        handoff_stage="phase6",
    )

    assert (
        _phase7_extended_attempt_stall_blocker(
            config=config,
            attempt_state=repairing_state,
            last_attempt=repairing_attempt,
            next_attempt_policy=next_policy,
        )
        is None
    )


def test_public_standard_and_serious_budget_policies_remain_unchanged() -> None:
    standard = hmc_kernel_tuning.HMCKernelTuningConfig.standard()
    standard_factory = _public_budget_policy_factory(standard)
    serious = hmc_kernel_tuning.HMCKernelTuningConfig.serious()
    serious_factory = _public_budget_policy_factory(serious)

    assert standard_factory is not None
    standard_policy0 = standard_factory(4, 0)
    standard_policy1 = standard_factory(4, 1)
    standard_policy2 = standard_factory(4, 2)

    assert standard.uses_serious_budget_policy is False
    assert standard_policy0.budget == 128
    assert standard_policy1.budget == 256
    assert standard_policy2.budget == 512
    assert standard_policy0.phase5_tune_budgets == (32, 64, 128)
    assert standard_policy0.phase5_screen_num_results == 32
    assert standard_policy0.phase6_screen_num_results == 32
    assert standard_policy0.verification_num_results == 64
    assert standard_policy0.serious_policy is False
    assert serious.max_attempts == 5
    assert serious.uses_serious_budget_policy is True
    assert serious_factory is not None
    serious_policy0 = serious_factory(4, 0)
    assert serious_policy0.serious_policy is True
    assert serious_policy0.payload()["geometry_budget_summary"]["dimension"] == 4
    assert "geometry_multiplier" in serious_policy0.payload()["geometry_budget_summary"]


def test_public_standard_terminal_phase6_extra_attempt_caps_phase6_screen_only() -> None:
    config = hmc_kernel_tuning.HMCKernelTuningConfig.standard(
        max_attempts=4,
        terminal_phase6_repair_extra_attempts=1,
    )
    factory = _public_budget_policy_factory(config)

    assert factory is not None
    policy3 = factory(4, 3)
    terminal_policy = factory(4, 4)

    assert policy3.budget == 1024
    assert policy3.phase6_screen_num_results == 256
    assert terminal_policy.budget == 2048
    assert terminal_policy.phase5_tune_budgets == (512, 1024, 2048)
    assert terminal_policy.phase5_screen_num_results == 512
    assert terminal_policy.phase6_screen_num_results == 128
    assert terminal_policy.phase6_screen_burnin_steps == 32
    assert terminal_policy.verification_num_results == 1024
    assert terminal_policy.public_budget_class == "standard_public_diagnostic"
    assert terminal_policy.public_diagnostic_preset == "standard"


def test_outer_loop_passes_only_after_fresh_fixed_kernel_verification() -> None:
    run, calls = _scripted_full_chain_runner(verification_acceptances=[0.70])

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert isinstance(result, HMCTuneVerifyRepairLoopResult)
    assert result.passed is True
    assert result.final_status == "passed"
    assert result.final_kernel_payload is not None
    assert result.final_kernel_hash
    assert result.final_kernel_payload["fresh_fixed_kernel_verification_passed"] is True
    assert result.final_kernel_payload["reports_posterior_convergence"] is False
    assert result.attempts[0].verification_config_payload["trace_policy"] == "standard"
    assert result.attempts[0].verification_config_payload["adaptation_policy"] == "fixed_kernel_no_adaptation"
    assert result.attempts[0].verification_config_payload["use_xla"] is False
    assert result.attempts[0].verification_config_payload["num_results"] == 4
    assert result.attempts[0].verification_config_payload["num_burnin_steps"] == 1
    assert result.attempts[0].verification_diagnostics["acceptance_rate"] == pytest.approx(0.70)
    assert result.nonclaims == TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS
    verification_calls = [
        call
        for call in calls
        if call["num_results"] == 4
        and call["num_burnin_steps"] == 1
        and call["uses_dual_averaging"] is False
    ]
    assert verification_calls
    assert all(call["trace_policy"] == "standard" for call in verification_calls)
    assert all(call["use_xla"] is False for call in verification_calls)


def test_phase7_public_timeout_before_windowed_mass_skips_windowed_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", lambda: 100.0)
    events: list[tuple[str, Mapping[str, Any]]] = []

    def forbidden_windowed_runner(**_kwargs: Any):
        raise AssertionError("windowed mass runner must not be called after timeout")

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            public_timeout_budget_s=10.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=lambda *_args, **_kwargs: _fake_result(),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=forbidden_windowed_runner,
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    assert result.final_status == "hard_veto"
    assert result.passed is False
    assert result.hard_vetoes == ("phase7_public_timeout_before_windowed_mass",)
    assert result.repair_triggers == ("phase7_public_timeout_before_windowed_mass",)
    assert result.final_kernel_payload is None
    assert result.attempts[0].windowed_stage is None
    assert result.attempts[0].repair_triggers == (
        "phase7_public_timeout_before_windowed_mass",
    )
    assert result.attempts[0].verification_diagnostics["windowed_stage_runner_called"] is False
    closeout = result.attempts[0].verification_diagnostics["public_timeout_closeout"]
    assert closeout["schema"] == "bayesfilter.phase7_public_timeout_before_windowed_mass.v1"
    assert closeout["stage"] == "phase7_loop_attempt_before_windowed_mass"
    assert closeout["hard_veto"] == "phase7_public_timeout_before_windowed_mass"
    assert closeout["windowed_mass_runner_called"] is False
    assert closeout["closeout_required_before_windowed_mass_runner_build"] is True
    assert closeout["hmc_mechanics_exposed"] is False
    assert [stage for stage, _payload in events] == [
        "loop_attempt_start",
        "phase7_public_timeout_before_windowed_mass",
        "loop_attempt_complete",
        "loop_complete",
    ]
    timeout_event = events[1][1]
    timeout_summary = timeout_event["extra"]["resume_split_public_summary"]
    assert timeout_summary["schema"] == "bayesfilter.phase7_resume_split_public_summary.v1"
    assert timeout_summary["availability_status"] == "unavailable"
    assert timeout_summary["unavailable_reason"] == "no_private_handoff_before_resume_split"
    assert timeout_summary["private_resume_payload_available"] is False
    assert timeout_summary["private_resume_payload_exposed"] is False
    assert timeout_summary["verifier_entry_manifest"] is False
    assert timeout_summary["final_kernel_handoff"] is False
    assert timeout_event["extra"]["repair_trigger"] == (
        "phase7_public_timeout_before_windowed_mass"
    )

    complete_event = events[2][1]
    complete_summary = complete_event["extra"]["resume_split_public_summary"]
    assert complete_summary["schema"] == "bayesfilter.phase7_resume_split_public_summary.v1"
    assert complete_summary["availability_status"] == "unavailable"
    assert complete_summary["private_resume_payload_available"] is False
    assert complete_event["extra"]["repair_trigger_count"] == 1
    progress_text = json.dumps([timeout_event, complete_event], sort_keys=True)
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
        assert forbidden not in progress_text


def test_staged_timeout_stage_entry_anchor_is_fresh_for_pre_windowed_closeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_clock = {"value": 0.0}

    def fake_perf_counter() -> float:
        return base_clock["value"]

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)
    stage_entry_times: list[float] = []
    stage_elapsed: list[float] = []

    def record_timeout(
        *,
        config: Any,
        attempt_index: int,
        target_dimension: int,
        budget_policy: Any,
    ) -> Mapping[str, Any] | None:
        del attempt_index, target_dimension, budget_policy
        stage_entry_times.append(float(config.staged_timeout_stage_started_perf_counter_s))
        base_clock["value"] = 12.5
        stage_elapsed.append(
            float(
                hmc_kernel_tuning.time.perf_counter()
                - float(config.staged_timeout_stage_started_perf_counter_s)
            )
        )
        return {
            "schema": "bayesfilter.phase7_public_timeout_before_windowed_mass.v1",
            "stage": "phase7_loop_attempt_before_windowed_mass",
            "attempt_index": 0,
            "enabled": True,
            "timeout_budget_s": 10.0,
            "reserve_s": 5.0,
            "elapsed_s": 12.5,
            "remaining_s": -2.5,
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
        }

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_phase7_public_timeout_before_windowed_mass",
        record_timeout,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_bootstrap_screen",
        lambda **_kwargs: _bootstrap(),
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "initialize_hmc_kernel_geometry",
        lambda **_kwargs: _geometry(),
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            public_timeout_budget_s=10.0,
            public_timeout_started_perf_counter_s=0.0,
            staged_timeout_policy=hmc_kernel_tuning.HMCStagedTimeoutPolicy(),
        ),
        run_full_chain=lambda *_args, **_kwargs: _fake_result(),
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "hard_veto"
    assert stage_entry_times == [0.0]
    assert stage_elapsed == [12.5]
    assert result.attempts[0].verification_diagnostics["public_timeout_closeout"]["elapsed_s"] == 12.5
    assert result.attempts[0].verification_diagnostics["public_timeout_closeout"]["stage"] == "phase7_loop_attempt_before_windowed_mass"


def test_terminal_phase6_repair_slot_does_not_bypass_pre_windowed_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windowed = _windowed_stage()
    fixed_pass_run, _ = _scripted_full_chain_runner(
        phase5_screen_acceptances=[0.70],
        verification_acceptances=[0.70],
    )
    fixed_pass = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=fixed_pass_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )
    fixed_pass = _historical_phase5_stage(fixed_pass)

    def trajectory_fail_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.90,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory_fail = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed_pass,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        run_full_chain=trajectory_fail_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )
    mixed_overreach_candidates = []
    for index, candidate in enumerate(trajectory_fail.candidate_results):
        payload = dict(candidate)
        diagnostics = dict(payload.get("diagnostics", {}))
        diagnostics["acceptance_rate"] = 0.60 if index % 2 == 0 else 0.90
        payload["diagnostics"] = diagnostics
        payload["trajectory_window_relation"] = "above_trajectory_window"
        payload["repair_triggers"] = tuple(
            dict.fromkeys(
                tuple(payload.get("repair_triggers", ()))
                + ("trajectory_length_above_window",)
            )
        )
        mixed_overreach_candidates.append(payload)
    trajectory_fail = replace(
        trajectory_fail,
        candidate_results=tuple(mixed_overreach_candidates),
    )
    clock = {"now": 0.0}
    windowed_calls: list[int] = []

    def fake_perf_counter() -> float:
        return float(clock["now"])

    def windowed_runner(**kwargs: Any):
        attempt_index = int(kwargs["_attempt_index"])
        windowed_calls.append(attempt_index)
        return windowed

    def trajectory_runner(**kwargs: Any):
        if int(kwargs["_attempt_index"]) == 0:
            clock["now"] = 100.0
        return trajectory_fail

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            terminal_phase6_repair_extra_attempts=1,
            public_timeout_budget_s=10.0,
            public_timeout_started_perf_counter_s=0.0,
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_runner,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed_pass,
        _frozen_step_trajectory_stage_runner=trajectory_runner,
        _phase7_final_verification_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("verification must not run when Phase 6 fails")
        ),
    )

    assert result.final_status == "hard_veto"
    assert "phase7_public_timeout_before_windowed_mass" in result.hard_vetoes
    assert (
        result.diagnostic_roles["phase6_handoff_screen"]
        == "handoff_screen_repair_trigger_non_promoting"
    )
    assert (
        result.diagnostic_roles["trajectory_window"]
        == "engineering_viability_gate_non_scientific"
    )
    assert windowed_calls == [0]
    assert len(result.attempts) == 2
    assert result.attempts[0].frozen_step_trajectory_stage is trajectory_fail
    assert result.attempts[1].windowed_stage is None

    summary = hmc_kernel_tuning._phase7_public_summary(result)
    latest_resume = summary["latest_resume_split_public_summary"]
    assert latest_resume["schema"] == "bayesfilter.phase7_resume_split_public_summary.v1"
    assert latest_resume["loop_artifact_hash"] == result.artifact_hash
    if "availability_status" in latest_resume:
        assert latest_resume["availability_status"] in {"available", "unavailable"}
    if "private_resume_payload_available" in latest_resume:
        assert latest_resume["private_resume_payload_available"] in {True, False}
    assert latest_resume["verifier_entry_manifest"] is False
    verification = summary["attempt_summaries"][0]["stage_statuses"]["verification"]
    assert verification["final_status"] == "repair_or_retry"
    assert "phase7_public_timeout_before_windowed_mass" in summary["hard_vetoes"]
    text = json.dumps(summary, sort_keys=True)
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


def test_retained_frozen_kernel_adapter_replay_uses_private_loop_payload() -> None:
    payload = _replay_tuning_payload()

    result = build_retained_frozen_kernel_hmc_adapter_from_tuning_payload(
        adapter=_ToyGaussianAdapter(),
        tuning_payload=payload,
        initial_position=np.zeros(2),
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )

    assert result.contract["replay_owned_by_bayesfilter"] is True
    assert result.contract["hmc_or_tuning_invoked"] is False
    assert result.contract["final_hmc_adapter_signature"] == stable_adapter_signature(
        result.adapter
    )
    assert (
        result.contract["adapted_mass_parent_adapter_signature"]
        == result.contract["phase4_hmc_adapter_signature"]
    )
    assert result.final_kernel_payload["schema"] == "bayesfilter.hmc_frozen_kernel_handoff.v1"
    assert result.payload()["final_kernel_payload"][
        "public_handoff_schema"
    ] is None


def test_retained_frozen_kernel_adapter_replay_rejects_public_only_handoff() -> None:
    payload = dict(_replay_tuning_payload())
    payload["tune_verify_repair_loop"] = {
        "schema": "bayesfilter.hmc_tune_verify_repair_loop.v1"
    }

    with pytest.raises(ValueError, match="private final kernel payload"):
        build_retained_frozen_kernel_hmc_adapter_from_tuning_payload(
            adapter=_ToyGaussianAdapter(),
            tuning_payload=payload,
            initial_position=np.zeros(2),
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        )


def test_retained_frozen_kernel_adapter_replay_rejects_mass_parent_mismatch() -> None:
    payload = dict(_replay_tuning_payload())
    loop = dict(payload["tune_verify_repair_loop"])
    final_kernel = dict(loop["final_kernel_payload"])
    mass_payload = dict(final_kernel["adapted_mass_artifact_payload"])
    mass_payload["adapter_signature"] = "wrong-phase4-parent-signature"
    final_kernel["adapted_mass_artifact_payload"] = mass_payload
    loop["final_kernel_payload"] = final_kernel
    payload["tune_verify_repair_loop"] = loop

    with pytest.raises(ValueError, match="adapter signature"):
        build_retained_frozen_kernel_hmc_adapter_from_tuning_payload(
            adapter=_ToyGaussianAdapter(),
            tuning_payload=payload,
            initial_position=np.zeros(2),
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        )


def test_retained_frozen_kernel_adapter_replay_rejects_final_signature_mismatch() -> None:
    payload = dict(_replay_tuning_payload())
    loop = dict(payload["tune_verify_repair_loop"])
    final_kernel = dict(loop["final_kernel_payload"])
    final_kernel["verification_hmc_adapter_signature"] = (
        "wrong-final-adapter-signature"
    )
    loop["final_kernel_payload"] = final_kernel
    payload["tune_verify_repair_loop"] = loop

    with pytest.raises(ValueError, match="final HMC adapter signature"):
        build_retained_frozen_kernel_hmc_adapter_from_tuning_payload(
            adapter=_ToyGaussianAdapter(),
            tuning_payload=payload,
            initial_position=np.zeros(2),
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        )


def test_outer_loop_config_use_xla_propagates_to_all_full_chain_calls() -> None:
    run, calls = _scripted_full_chain_runner(verification_acceptances=[0.70])

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            chain_execution_mode="tf_function",
            use_xla=True,
        ),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.passed is True
    assert result.config.payload()["use_xla"] is True
    assert result.attempts[0].verification_config_payload["use_xla"] is True
    assert calls
    assert all(call["use_xla"] is True for call in calls)


def test_outer_loop_default_tf_function_verification_uses_sequential_rhat_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, Any]] = []

    class _FakeReusableRunner:
        def __init__(
            self,
            config: Any,
            *,
            dynamic_num_leapfrog_steps: bool = False,
        ) -> None:
            self.config = config
            self.dynamic_num_leapfrog_steps = bool(dynamic_num_leapfrog_steps)

        def run(
            self,
            *,
            current_state: Any,
            seed: Any,
            step_size: Any,
            num_leapfrog_steps: Any | None = None,
        ):
            calls.append(
                {
                    "role": "run",
                    "num_results": int(self.config.num_results),
                    "num_burnin_steps": int(self.config.num_burnin_steps),
                    "uses_dual_averaging": bool(
                        self.config.tuning_policy.uses_dual_averaging
                    ),
                    "seed": tuple(int(item) for item in seed),
                    "step_size": float(step_size),
                    "num_leapfrog_steps": int(self.config.num_leapfrog_steps)
                    if num_leapfrog_steps is None
                    else int(num_leapfrog_steps),
                    "dynamic_num_leapfrog_steps": self.dynamic_num_leapfrog_steps,
                    "initial_state": np.asarray(current_state, dtype=float),
                }
            )
            if self.config.tuning_policy.uses_dual_averaging:
                return _fake_result(
                    num_results=int(self.config.num_results),
                    acceptance=0.70,
                    step_size=0.20,
                    num_adaptation_steps=self.config.tuning_policy.num_adaptation_steps,
                )
            return _fake_result(
                num_results=int(self.config.num_results),
                acceptance=0.70,
            )

    def fake_builder(
        adapter: Any,
        initial_state_template: Any,
        config: Any,
        *,
        dynamic_num_leapfrog_steps: bool = False,
    ) -> _FakeReusableRunner:
        calls.append(
            {
                "role": "build",
                "adapter_signature": adapter.adapter_signature(),
                "num_results": int(config.num_results),
                "num_burnin_steps": int(config.num_burnin_steps),
                "uses_dual_averaging": bool(config.tuning_policy.uses_dual_averaging),
                "dynamic_num_leapfrog_steps": bool(dynamic_num_leapfrog_steps),
                "initial_state_template": np.asarray(initial_state_template, dtype=float),
            }
        )
        return _FakeReusableRunner(
            config,
            dynamic_num_leapfrog_steps=dynamic_num_leapfrog_steps,
        )

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_reusable_full_chain_tfp_hmc_runner",
        fake_builder,
    )
    monkeypatch.setattr(
        hmc_budget_ladder,
        "build_reusable_full_chain_tfp_hmc_runner",
        fake_builder,
    )
    sequential_configs: list[Mapping[str, Any]] = []

    class _FakeSequentialVerifier:
        def __init__(self, config: Any) -> None:
            self.config = config

        def _configure_retained_target_health_policy(self, policy: str) -> None:
            assert policy == "none"

        def run(
            self,
            *,
            checkpoint_writer_config: Any = None,
            checkpoint_reference_callback: Any = None,
        ):
            if checkpoint_writer_config is not None:
                reference = {
                    "artifact_type": "bayesfilter_sequential_rhat_checkpoint_public_reference",
                    "schema_version": 1,
                    "checkpoint_kind": "verification_chunk",
                    "checkpoint_id": "srhat-v1-11111111111111111111111111111111",
                    "checkpoint_sha256": "2" * 64,
                    "contract_sha256": (
                        bayesfilter.sequential_rhat_verification_checkpoint_contract()[
                            "contract_sha256"
                        ]
                    ),
                    "private_paths_publicized": False,
                    "public_summary_contains_paths": False,
                    "public_summary_contains_raw_values": False,
                    "public_summary_contains_tensor_descriptors": False,
                    "public_summary_contains_kernel_payload": False,
                    "nonclaims": bayesfilter.SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
                }
                if checkpoint_reference_callback is not None:
                    checkpoint_reference_callback(reference)
                checkpoint_count = 1
                checkpoint_references = (reference,)
            else:
                checkpoint_count = 0
                checkpoint_references = ()
            return type(
                "_SequentialResult",
                (),
                {
                    "diagnostics": {
                        **_sequential_verification_diagnostics(
                            0.70,
                            draw_count=int(self.config.max_results),
                            rhat_passed=True,
                        ),
                        "sequential_rhat_verification": True,
                        "passed": True,
                        "cap_hit": False,
                        "retained_sample_count": int(self.config.max_results),
                        "check_interval": int(self.config.check_interval),
                        "max_results": int(self.config.max_results),
                        "chunk_count": 1,
                        "rhat_threshold": float(self.config.rhat_threshold),
                        "max_finite_rhat": 1.0,
                        "finite_rhat_count": 2,
                        "nonfinite_rhat_count": 0,
                        "all_finite_rhat_at_or_below_threshold": True,
                        "samples_all_finite": True,
                        "target_log_prob_finite": True,
                        "log_accept_ratio_finite": True,
                        "runtime_s": 0.01,
                        "runtime_finite": True,
                        "divergence_status": "not_exposed_by_kernel",
                        "divergence_count": None,
                        "hard_vetoes": (),
                        "checkpointing_enabled": checkpoint_writer_config is not None,
                        "checkpoint_count": checkpoint_count,
                        "checkpoint_references": checkpoint_references,
                        "privacy_contract": {
                            "public_summary_contains_raw_values": False,
                            "public_summary_contains_chain_states": False,
                            "public_summary_contains_step_size": False,
                            "public_summary_contains_leapfrog_count": False,
                            "public_summary_contains_mass_matrix": False,
                        },
                    },
                },
            )()

    def fake_sequential_builder(
        _adapter: Any,
        _initial_state_template: Any,
        config: Any,
    ) -> _FakeSequentialVerifier:
        sequential_configs.append(
            {
                "check_interval": int(config.check_interval),
                "max_results": int(config.max_results),
                "num_burnin_steps": int(config.num_burnin_steps),
                "use_xla": bool(config.use_xla),
                "chain_execution_mode": config.chain_execution_mode,
                "target_scope": config.target_scope,
            }
        )
        return _FakeSequentialVerifier(config)

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_sequential_rhat_hmc_verifier",
        fake_sequential_builder,
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=1, chain_execution_mode="tf_function"),
        _budget_policy_factory=_verification_budget_factory,
    )

    assert result.passed is True
    verification = result.attempts[0].verification_diagnostics
    route = verification["runner_route_summary"]
    assert route["active_route"] == "phase7_sequential_rhat_fixed_size_chunk_verifier"
    assert route["single_use_build_count"] == 0
    assert route["fallback_status"] == "none"
    assert "not a stopping rule" in route["route_nonclaims"][1]
    assert verification["sequential_rhat_verification"] is True
    assert verification["all_finite_rhat_at_or_below_threshold"] is True
    assert result.attempts[0].verification_config_payload["verification_policy"] == (
        "sequential_rhat"
    )
    assert result.attempts[0].verification_config_payload["check_interval"] == 64
    assert result.attempts[0].verification_config_payload["max_results"] == 64
    assert (
        result.attempts[0].verification_config_payload["max_results"]
        == result.attempts[0].budget_policy_payload["verification_num_results"]
    )
    assert verification["sequential_rhat_policy"]["max_results"] == 64
    assert verification["sequential_rhat_policy"]["check_interval"] == 64
    assert (
        verification["sequential_rhat_policy"]["rhat_threshold_role"]
        == "historical_explanatory_only_not_stopping_or_admission"
    )
    assert (
        verification["sequential_rhat_policy"]["cap_rule"]
        == "stop_inconclusive_at_budget_policy_verification_num_results"
    )
    assert sequential_configs == [
        {
            "check_interval": 64,
            "max_results": 64,
            "num_burnin_steps": 1,
            "use_xla": False,
            "chain_execution_mode": "tf_function",
            "target_scope": "kernel_fixed_mass_step_toy_gaussian",
        }
    ]
    forbidden_keys: list[str] = []

    def collect_forbidden_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_text = str(key)
                if key_text not in {
                    "public_summary_contains_step_size",
                    "public_summary_contains_leapfrog_count",
                    "public_summary_contains_mass_matrix",
                } and any(
                    token in key_text
                    for token in ("step_size", "num_leapfrog", "mass_matrix")
                ):
                    forbidden_keys.append(key_text)
                collect_forbidden_keys(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect_forbidden_keys(item)

    collect_forbidden_keys(verification)
    collect_forbidden_keys(result.attempts[0].verification_config_payload)
    assert forbidden_keys == []
    assert any(
        call["role"] == "build"
        and call["uses_dual_averaging"] is False
        for call in calls
    )


def test_sequential_verification_uses_configured_compile_chunk_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windowed, fixed, trajectory = _passed_phase7_stage_fixtures()
    sequential_configs: list[Mapping[str, Any]] = []

    class _FakeSequentialVerifier:
        def __init__(self, config: Any) -> None:
            self.config = config

        def _configure_retained_target_health_policy(self, policy: str) -> None:
            assert policy == "none"

        def run(self, **_kwargs: Any):
            return type(
                "_SequentialResult",
                (),
                    {
                        "diagnostics": {
                            **_sequential_verification_diagnostics(
                                0.70,
                                draw_count=int(self.config.max_results),
                                rhat_passed=True,
                            ),
                            "sequential_rhat_verification": True,
                        "passed": True,
                        "cap_hit": False,
                        "retained_sample_count": int(self.config.max_results),
                        "check_interval": int(self.config.check_interval),
                        "max_results": int(self.config.max_results),
                        "chunk_count": 4,
                        "rhat_threshold": float(self.config.rhat_threshold),
                        "max_finite_rhat": 1.0,
                        "finite_rhat_count": 2,
                        "nonfinite_rhat_count": 0,
                        "all_finite_rhat_at_or_below_threshold": True,
                        "samples_all_finite": True,
                        "target_log_prob_finite": True,
                        "log_accept_ratio_finite": True,
                        "runtime_s": 0.01,
                        "runtime_finite": True,
                        "acceptance_rate": 0.70,
                        "divergence_status": "not_exposed_by_kernel",
                        "divergence_count": None,
                        "hard_vetoes": (),
                    },
                },
            )()

    def fake_sequential_builder(
        _adapter: Any,
        _initial_state_template: Any,
        config: Any,
    ) -> _FakeSequentialVerifier:
        sequential_configs.append(
            {
                "check_interval": int(config.check_interval),
                "max_results": int(config.max_results),
                "min_retained_results_for_pass": int(
                    config.min_retained_results_for_pass
                ),
            }
        )
        return _FakeSequentialVerifier(config)

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_sequential_rhat_hmc_verifier",
        fake_sequential_builder,
    )

    def budget_factory(_dimension: int, attempt_index: int) -> _HMCAttemptBudgetPolicy:
        base = _tiny_budget_factory(_dimension, attempt_index)
        return replace(
            base,
            verification_num_results=1000,
            verification_num_burnin_steps=250,
        )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            chain_execution_mode="tf_function",
            verification_chunk_max_results=250,
            verification_min_retained_results_for_pass=1000,
        ),
        _budget_policy_factory=budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: trajectory,
    )

    assert result.passed is True
    verification = result.attempts[0].verification_diagnostics
    assert verification["sequential_rhat_policy"]["check_interval"] == 250
    assert verification["sequential_rhat_policy"]["max_results"] == 1000
    assert (
        verification["sequential_rhat_policy"]["minimum_retained_results_for_pass"]
        == 1000
    )
    assert verification["verification_min_retained_pass_gate_satisfied"] is True
    assert sequential_configs == [
        {
            "check_interval": 250,
            "max_results": 1000,
            "min_retained_results_for_pass": 1000,
        }
    ]


def test_sequential_verification_blocks_pass_before_minimum_retained_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windowed, fixed, trajectory = _passed_phase7_stage_fixtures()

    class _EarlyPassSequentialVerifier:
        def __init__(self, config: Any) -> None:
            self.config = config

        def _configure_retained_target_health_policy(self, policy: str) -> None:
            assert policy == "none"

        def run(self, **_kwargs: Any):
            return type(
                "_SequentialResult",
                (),
                    {
                        "diagnostics": {
                            **_sequential_verification_diagnostics(
                                0.70,
                                draw_count=250,
                                rhat_passed=True,
                            ),
                            "sequential_rhat_verification": True,
                        "passed": True,
                        "cap_hit": False,
                        "retained_sample_count": 250,
                        "check_interval": int(self.config.check_interval),
                        "max_results": int(self.config.max_results),
                        "chunk_count": 1,
                        "rhat_threshold": float(self.config.rhat_threshold),
                        "max_finite_rhat": 1.0,
                        "finite_rhat_count": 2,
                        "nonfinite_rhat_count": 0,
                        "all_finite_rhat_at_or_below_threshold": True,
                        "samples_all_finite": True,
                        "target_log_prob_finite": True,
                        "log_accept_ratio_finite": True,
                        "runtime_s": 0.01,
                        "runtime_finite": True,
                        "acceptance_rate": 0.70,
                        "divergence_status": "not_exposed_by_kernel",
                        "divergence_count": None,
                        "hard_vetoes": (),
                    },
                },
            )()

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_sequential_rhat_hmc_verifier",
        lambda _adapter, _initial_state_template, config: _EarlyPassSequentialVerifier(
            config
        ),
    )

    def budget_factory(_dimension: int, attempt_index: int) -> _HMCAttemptBudgetPolicy:
        base = _tiny_budget_factory(_dimension, attempt_index)
        return replace(
            base,
            verification_num_results=1000,
            verification_num_burnin_steps=250,
        )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            chain_execution_mode="tf_function",
            verification_chunk_max_results=250,
            verification_min_retained_results_for_pass=1000,
        ),
        _budget_policy_factory=budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: trajectory,
    )

    assert result.passed is False
    attempt = result.attempts[0]
    assert attempt.final_status == "repair_or_retry"
    assert attempt.diagnostic_role == "verification_minimum_retained_repair_trigger"
    assert (
        "verification_minimum_retained_results_not_reached"
        in attempt.repair_triggers
    )
    assert (
        attempt.verification_diagnostics[
            "rhat_passed_before_minimum_retained_count"
        ]
        is True
    )
    assert (
        attempt.verification_diagnostics[
            "verification_min_retained_pass_gate_satisfied"
        ]
        is False
    )


def test_phase7_checkpoint_writer_emits_pre_verification_handoff_before_verification_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    writer_config = SequentialRHatCheckpointWriterConfig(
        checkpoint_dir=tmp_path,
        checkpoint_label="phase3",
    )
    events: list[tuple[str, Mapping[str, Any]]] = []
    references: list[Mapping[str, Any]] = []
    contract_sha = bayesfilter.sequential_rhat_verification_checkpoint_contract()[
        "contract_sha256"
    ]

    def make_reference(kind: str, digit: str) -> Mapping[str, Any]:
        reference = {
            "artifact_type": "bayesfilter_sequential_rhat_checkpoint_public_reference",
            "schema_version": 1,
            "checkpoint_kind": kind,
            "checkpoint_id": f"srhat-v1-{digit * 32}",
            "checkpoint_sha256": digit * 64,
            "contract_sha256": contract_sha,
            "private_paths_publicized": False,
            "public_summary_contains_paths": False,
            "public_summary_contains_raw_values": False,
            "public_summary_contains_tensor_descriptors": False,
            "public_summary_contains_kernel_payload": False,
            "nonclaims": bayesfilter.SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
        }
        bayesfilter.assert_sequential_rhat_checkpoint_public_reference_safe(reference)
        return reference

    def fake_handoff_writer(**kwargs: Any) -> Mapping[str, Any]:
        assert kwargs["writer_config"] is writer_config
        assert kwargs["selected_kernel_private_payload"]["step_size"] > 0.0
        assert kwargs["selected_kernel_private_payload"]["num_leapfrog_steps"] > 0
        assert kwargs["selected_kernel_private_payload"]["private_handoff_only"] is True
        assert kwargs["mass_payload"]["dimension"] == 2
        reference = make_reference("pre_verification_handoff", "1")
        references.append(reference)
        return reference

    class _FakeSequentialVerifier:
        def _configure_retained_target_health_policy(self, policy: str) -> None:
            assert policy == "none"

        def run(
            self,
            *,
            checkpoint_writer_config: Any = None,
            checkpoint_reference_callback: Any = None,
        ):
            assert checkpoint_writer_config is writer_config
            reference = make_reference("verification_chunk", "2")
            references.append(reference)
            assert checkpoint_reference_callback is not None
            checkpoint_reference_callback(reference)
            return type(
                "_SequentialResult",
                (),
                {
                    "diagnostics": {
                        **_sequential_verification_diagnostics(
                            0.70,
                            draw_count=64,
                            rhat_passed=True,
                        ),
                        "sequential_rhat_verification": True,
                        "passed": True,
                        "cap_hit": False,
                        "retained_sample_count": 64,
                        "check_interval": 64,
                        "max_results": 64,
                        "chunk_count": 1,
                        "rhat_threshold": 1.01,
                        "max_finite_rhat": 1.0,
                        "finite_rhat_count": 2,
                        "nonfinite_rhat_count": 0,
                        "all_finite_rhat_at_or_below_threshold": True,
                        "samples_all_finite": True,
                        "target_log_prob_finite": True,
                        "log_accept_ratio_finite": True,
                        "runtime_s": 0.01,
                        "runtime_finite": True,
                        "divergence_status": "not_exposed_by_kernel",
                        "divergence_count": None,
                        "hard_vetoes": (),
                        "checkpointing_enabled": True,
                        "checkpoint_count": 1,
                        "checkpoint_references": (reference,),
                    },
                },
            )()

    class _FakeReusableRunner:
        def __init__(
            self,
            config: Any,
            *,
            dynamic_num_leapfrog_steps: bool = False,
        ) -> None:
            self.config = config
            self.dynamic_num_leapfrog_steps = bool(dynamic_num_leapfrog_steps)

        def run(
            self,
            *,
            current_state: Any,
            seed: Any,
            step_size: Any,
            num_leapfrog_steps: Any | None = None,
        ):
            del current_state, seed, step_size
            if (
                num_leapfrog_steps is not None
                and not self.dynamic_num_leapfrog_steps
            ):
                raise AssertionError(
                    "static reusable runner must not receive dynamic L"
                )
            if self.config.tuning_policy.uses_dual_averaging:
                return _fake_result(
                    num_results=int(self.config.num_results),
                    acceptance=0.70,
                    step_size=0.20,
                    num_adaptation_steps=self.config.tuning_policy.num_adaptation_steps,
                )
            return _fake_result(
                num_results=int(self.config.num_results),
                acceptance=0.70,
            )

    def fake_reusable_builder(
        _adapter: Any,
        _initial_state_template: Any,
        config: Any,
        *,
        dynamic_num_leapfrog_steps: bool = False,
    ) -> _FakeReusableRunner:
        return _FakeReusableRunner(
            config,
            dynamic_num_leapfrog_steps=dynamic_num_leapfrog_steps,
        )

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_reusable_full_chain_tfp_hmc_runner",
        fake_reusable_builder,
    )
    monkeypatch.setattr(
        hmc_budget_ladder,
        "build_reusable_full_chain_tfp_hmc_runner",
        fake_reusable_builder,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "write_sequential_rhat_pre_verification_handoff_checkpoint",
        fake_handoff_writer,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_sequential_rhat_hmc_verifier",
        lambda *_args, **_kwargs: _FakeSequentialVerifier(),
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=1, chain_execution_mode="tf_function"),
        verification_checkpoint_writer_config=writer_config,
        _budget_policy_factory=_verification_budget_factory,
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    stage_names = [stage for stage, _payload in events]
    assert result.passed is True
    assert references[0]["checkpoint_kind"] == "pre_verification_handoff"
    assert references[1]["checkpoint_kind"] == "verification_chunk"
    assert "trajectory_complete" not in stage_names
    assert "trajectory_candidate_call_start" not in stage_names
    assert stage_names.index("direct_verification_queue_start") < stage_names.index(
        "verification_checkpoint_written"
    )
    assert stage_names.index("verification_checkpoint_written") < stage_names.index(
        "verification_start"
    )
    assert stage_names.index("verification_start") < stage_names.index(
        "verification_complete"
    )
    checkpoint_events = [
        payload["extra"]
        for stage, payload in events
        if stage == "verification_checkpoint_written"
    ]
    assert len(checkpoint_events) == 2
    for extra in checkpoint_events:
        reference = extra["checkpoint_reference"]
        bayesfilter.assert_sequential_rhat_checkpoint_public_reference_safe(reference)
        public_text = json.dumps(extra, sort_keys=True)
        for forbidden in (
            str(tmp_path),
            "/",
            "\\",
            "step_size",
            "num_leapfrog_steps",
            "mass_payload",
            "selected_kernel",
            "final_state",
            "samples",
            ".tftensor",
        ):
            assert forbidden not in public_text
        assert extra["private_paths_publicized"] is False
        assert extra["hmc_mechanics_exposed"] is False
    verification = result.attempts[0].verification_diagnostics
    assert verification["phase7_checkpointing_enabled"] is True
    assert verification["phase7_checkpoint_count"] == 2
    assert [item["checkpoint_kind"] for item in verification["phase7_checkpoint_references"]] == [
        "pre_verification_handoff",
        "verification_chunk",
    ]
    summary = hmc_kernel_tuning._phase7_verification_public_summary(
        result.attempts[0]
    )
    assert summary["checkpointing_enabled"] is True
    assert summary["checkpoint_count"] == 2
    assert summary["checkpoint_references_public_safe"] is True
    summary_text = json.dumps(summary, sort_keys=True)
    assert "log_accept" not in summary_text
    assert "target_log_prob" not in summary_text
    assert summary["private_acceptance_log_health_passed"] is True
    assert summary["private_target_value_health_passed"] is True


def test_phase7_direct_checkpoint_uses_candidate_lineage_without_phase6(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (
        geometry,
        _bootstrap_result,
        windowed,
        fixed,
        handoff,
        selected_identity,
        _alternative_identity,
    ) = _phase7_direct_fixture()
    writer_config = SequentialRHatCheckpointWriterConfig(
        checkpoint_dir=tmp_path,
        checkpoint_label="phase7-direct",
    )
    captured_payloads: list[Mapping[str, Any]] = []
    contract_sha = bayesfilter.sequential_rhat_verification_checkpoint_contract()[
        "contract_sha256"
    ]
    reference = {
        "artifact_type": "bayesfilter_sequential_rhat_checkpoint_public_reference",
        "schema_version": 1,
        "checkpoint_kind": "pre_verification_handoff",
        "checkpoint_id": f"srhat-v1-{'3' * 32}",
        "checkpoint_sha256": "3" * 64,
        "contract_sha256": contract_sha,
        "private_paths_publicized": False,
        "public_summary_contains_paths": False,
        "public_summary_contains_raw_values": False,
        "public_summary_contains_tensor_descriptors": False,
        "public_summary_contains_kernel_payload": False,
        "nonclaims": bayesfilter.SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
    }
    bayesfilter.assert_sequential_rhat_checkpoint_public_reference_safe(reference)

    def fake_writer(**kwargs: Any) -> Mapping[str, Any]:
        captured_payloads.append(dict(kwargs["selected_kernel_private_payload"]))
        return reference

    class _FakeSequentialVerifier:
        def _configure_retained_target_health_policy(self, policy: str) -> None:
            assert policy == "none"

        def run(self, **_kwargs: Any):
            return type(
                "_SequentialResult",
                (),
                {
                    "diagnostics": {
                        **_sequential_verification_diagnostics(
                            0.70,
                            draw_count=64,
                            rhat_passed=True,
                        ),
                        "sequential_rhat_verification": True,
                        "all_finite_rhat_at_or_below_threshold": True,
                    }
                },
            )()

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "write_sequential_rhat_pre_verification_handoff_checkpoint",
        fake_writer,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_sequential_rhat_hmc_verifier",
        lambda *_args, **_kwargs: _FakeSequentialVerifier(),
    )
    outcome = hmc_kernel_tuning._run_phase7_direct_candidate_verification(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        candidate_identity=selected_identity,
        config=_loop_config(max_attempts=1, chain_execution_mode="tf_function"),
        budget_policy=_verification_budget_factory(2, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=writer_config,
        verification_start_callback=None,
        checkpoint_reference_callback=None,
        run_full_chain=hmc_kernel_tuning.run_full_chain_tfp_hmc,
    )

    assert outcome.final_status == "passed"
    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["phase5_candidate_batch_hash"] == handoff.handoff_hash
    assert payload["phase5_candidate_identity"] == selected_identity
    assert payload["phase5_candidate_record_hash"] == (
        handoff.candidate_records[selected_identity[0]].record_hash
    )
    assert "frozen_step_trajectory_stage_artifact_hash" not in payload
    assert outcome.diagnostics["phase7_checkpoint_references"] == (reference,)
    assert "phase5_candidate_batch_hash" not in repr(outcome)


def test_operational_checkpoint_binds_complete_v2_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()
    windowed = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=hmc_kernel_tuning.HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            seed=(20260711, 640),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        _attempt_budget_policy=_operational_budget(),
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_full_chain_tfp_hmc",
        lambda *_args, **_kwargs: None,
    )

    operational = windowed.operational_warmup_result
    assert operational is not None
    (
        _adapted_mass,
        _mass_signature,
        phase4_adapter,
        verification_adapter,
        verification_signature,
    ) = hmc_kernel_tuning._phase7_verification_runtime_context(
        adapter=adapter,
        geometry=geometry,
        windowed_stage=windowed,
        target_scope="kernel_windowed_mass_toy_gaussian",
    )
    _active_bank, active_bank_summary = (
        hmc_kernel_tuning._phase7_verification_initial_state(
            windowed_stage=windowed,
            phase4_adapter=phase4_adapter,
            verification_adapter=verification_adapter,
            verification_hmc_signature=verification_signature,
        )
    )
    active_bank_signature = active_bank_summary["active_signature"]
    from bayesfilter.inference.hmc_kernel_selection import (
        FixedTrajectoryCandidate,
        FixedTrajectoryCandidateResult,
        FixedTrajectoryReplication,
        FixedTrajectorySelection,
    )

    candidate = FixedTrajectoryCandidate(
        anchor_l=10,
        num_leapfrog_steps=10,
        max_leapfrog_steps=25,
        coordinate_signature=operational.final_kernel_state.transform.signature,
        metric_signature=operational.final_kernel_state.momentum_metric.signature,
        start_bank_signature=active_bank_signature,
    )
    replications = tuple(
        FixedTrajectoryReplication(
            candidate=candidate,
            replication_index=index,
            seed=(100 + index, 200 + index),
            acceptance_evidence_payload=_acceptance_evidence_payload(0.70),
        )
        for index in range(3)
    )
    candidate_result = FixedTrajectoryCandidateResult(
        candidate=candidate,
        replications=replications,
        exact_l_retuned_step_size=0.125,
        exact_l_retune_signature="exact-l-retune",
    )
    selection = FixedTrajectorySelection(
        anchor_l=10,
        candidate_results=(candidate_result,),
        representative_signature=candidate_result.signature,
        disposition="representative_selected",
    )
    selected_payload = {
        "schema": "bayesfilter.hmc_operational_exact_l_step.v2",
        "step_size": 0.125,
        "num_leapfrog_steps": 10,
        "selection_signature": selection.signature,
        "candidate_signature": candidate.signature,
        "exact_l_retune_signature": "exact-l-retune",
        "coordinate_signature": candidate.coordinate_signature,
        "metric_signature": candidate.metric_signature,
        "start_bank_signature": candidate.start_bank_signature,
        "private_handoff_only": True,
    }
    fixed = hmc_kernel_tuning.HMCFixedMassStepStageResult(
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_scope="kernel_windowed_mass_toy_gaussian",
            chain_execution_mode="tf_function",
        ),
        windowed_stage_artifact_hash=windowed.artifact_hash,
        selected_bootstrap_kernel_hash=windowed.selected_bootstrap_kernel_hash,
        adapter_signature=windowed.adapter_signature,
        phase4_hmc_adapter_signature=windowed.hmc_adapter_signature,
        ladder_adapter_signature="phase4-adapter",
        ladder_hmc_adapter_signature=verification_signature,
        adapted_mass_artifact_payload=windowed.adapted_mass_artifact_payload,
        adapted_mass_artifact_signature=windowed.adapted_mass_artifact_signature,
        initial_step_size=0.1,
        fixed_num_leapfrog_steps=10,
        target_dimension=2,
        final_status="passed",
        diagnostic_role="operational_fixed_trajectory_handoff_only",
        hard_vetoes=(),
        repair_triggers=(),
        diagnostics={
            "algorithm": "operational_paired_fixed_trajectory_selection_v3",
            "passed": True,
        },
        budget_ladder_config_payload=None,
        budget_ladder_result=None,
        selected_step_payload=selected_payload,
        selected_step_hash=hmc_kernel_tuning.stable_config_hash(selected_payload),
        repair_step_payload=None,
        repair_step_hash=None,
        frozen_mass_invariant={"passed": True},
        seed_report={"seed_owner": "BayesFilter"},
        diagnostic_roles={"acceptance_evidence": "promotion"},
        _operational_selection=selection,
    )
    writer_config = SequentialRHatCheckpointWriterConfig(
        checkpoint_dir=tmp_path,
        checkpoint_label="operational-v2",
    )
    captured: list[Mapping[str, Any]] = []

    def fake_writer(**kwargs: Any):
        captured.append(dict(kwargs["selected_kernel_private_payload"]))
        return {
            "artifact_type": "bayesfilter_sequential_rhat_checkpoint_public_reference",
            "schema_version": 1,
            "checkpoint_kind": "pre_verification_handoff",
            "checkpoint_id": "srhat-v1-" + "5" * 32,
            "checkpoint_sha256": "5" * 64,
            "contract_sha256": bayesfilter.sequential_rhat_verification_checkpoint_contract()[
                "contract_sha256"
            ],
            "private_paths_publicized": False,
            "public_summary_contains_paths": False,
            "public_summary_contains_raw_values": False,
            "public_summary_contains_tensor_descriptors": False,
            "public_summary_contains_kernel_payload": False,
            "nonclaims": bayesfilter.SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
        }

    class _Verifier:
        def _configure_retained_target_health_policy(self, policy: str) -> None:
            assert policy == "none"

        def run(self, **_kwargs: Any):
            return type(
                "_Result",
                (),
                {"diagnostics": _sequential_verification_diagnostics(0.70, draw_count=64)},
            )()

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "write_sequential_rhat_pre_verification_handoff_checkpoint",
        fake_writer,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_sequential_rhat_hmc_verifier",
        lambda *_args, **_kwargs: _Verifier(),
    )
    outcome = hmc_kernel_tuning._run_phase7_operational_selection_verification(
        adapter=adapter,
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=HMCTuneVerifyRepairLoopConfig(
            target_scope="kernel_windowed_mass_toy_gaussian",
            chain_execution_mode="tf_function",
        ),
        budget_policy=_operational_budget(),
        attempt_index=0,
        target_scope="kernel_windowed_mass_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=writer_config,
        verification_start_callback=None,
        checkpoint_reference_callback=None,
        run_full_chain=hmc_kernel_tuning.run_full_chain_tfp_hmc,
    )

    assert outcome.final_status == "passed"
    assert len(captured) == 1
    payload = captured[0]
    assert payload["verification_source_kind"] == "operational_selection_v2"
    assert payload["operational_selection_signature"] == selection.signature
    assert payload["operational_candidate_signature"] == candidate.signature
    assert payload["coordinate_signature"] == candidate.coordinate_signature
    assert payload["metric_signature"] == candidate.metric_signature
    assert payload["trajectory_signature"] == WarmupTrajectoryPolicy(
        candidate.num_leapfrog_steps,
        operational.final_kernel_state.trajectory_policy.max_leapfrog_steps,
    ).signature
    assert payload["start_bank_signature"] == active_bank_signature
    assert active_bank_summary["source_signature"] == (
        operational.private_start_bank_signature
    )
    assert payload["verification_input_hash"] == outcome.verification_input.input_hash


def test_phase7_checkpoint_writer_emits_boundary_before_windowed_execute_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    writer_config = SequentialRHatCheckpointWriterConfig(
        checkpoint_dir=tmp_path,
        checkpoint_label="boundary",
    )
    events: list[tuple[str, Mapping[str, Any]]] = []
    writer_calls: list[Mapping[str, Any]] = []
    contract_sha = bayesfilter.sequential_rhat_verification_checkpoint_contract()[
        "contract_sha256"
    ]
    boundary_reference = {
        "artifact_type": "bayesfilter_sequential_rhat_checkpoint_public_reference",
        "schema_version": 1,
        "checkpoint_kind": "phase7_boundary_handoff",
        "checkpoint_id": "srhat-v1-" + "3" * 32,
        "checkpoint_sha256": "4" * 64,
        "contract_sha256": contract_sha,
        "private_paths_publicized": False,
        "public_summary_contains_paths": False,
        "public_summary_contains_raw_values": False,
        "public_summary_contains_tensor_descriptors": False,
        "public_summary_contains_kernel_payload": False,
        "nonclaims": bayesfilter.SEQUENTIAL_RHAT_CHECKPOINT_PUBLIC_NONCLAIMS,
    }
    bayesfilter.assert_sequential_rhat_checkpoint_public_reference_safe(
        boundary_reference
    )

    def fake_boundary_writer(**kwargs: Any) -> Mapping[str, Any]:
        assert kwargs["writer_config"] is writer_config
        assert kwargs["boundary_private_payload"]["stage"] == (
            "windowed_mass_runner_execute_start"
        )
        assert kwargs["boundary_private_payload"]["private_raw_state_allowed"] is False
        assert kwargs["state_summary_private_payload"]["raw_state_included"] is False
        writer_calls.append(kwargs)
        return boundary_reference

    class _FailingReusableRunner:
        def run(self, **_kwargs: Any) -> Any:
            raise RuntimeError("windowed execute blocked after boundary checkpoint")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_reusable_full_chain_tfp_hmc_runner",
        lambda *_args, **_kwargs: _FailingReusableRunner(),
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "write_sequential_rhat_boundary_handoff_checkpoint",
        fake_boundary_writer,
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=1, chain_execution_mode="tf_function"),
        verification_checkpoint_writer_config=writer_config,
        _budget_policy_factory=_tiny_budget_factory,
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    stage_names = [stage for stage, _payload in events]
    assert result.passed is False
    assert writer_calls
    assert "windowed_mass_runner_execute_start" in stage_names
    assert "windowed_mass_runner_execute_complete" not in stage_names
    execute_payload = dict(
        events[stage_names.index("windowed_mass_runner_execute_start")][1]["extra"]
    )
    assert execute_payload["checkpoint_reference"] == boundary_reference
    assert execute_payload["checkpoint_reference_public_safe"] is True
    public_text = json.dumps(execute_payload, sort_keys=True)
    for forbidden in (
        str(tmp_path),
        "/",
        "\\",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "mass_matrix",
        "inverse_mass",
        "selected_kernel",
        "final_state",
    ):
        assert forbidden not in public_text


def test_sequential_verification_supported_high_acceptance_requests_repair() -> None:
    diagnostics = _sequential_verification_diagnostics(0.82, rhat_passed=True)
    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_final_verification(
            _loop_config(acceptance_band=(0.65, 0.75), repair_band=(0.55, 0.85)),
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "repair_or_retry"
    assert role == "verification_acceptance_repair_trigger"
    assert hard_vetoes == ()
    assert repair_triggers == ("verification_acceptance_outside_pass_band",)


def test_sequential_rhat_is_explanatory_when_acceptance_evidence_passes() -> None:
    diagnostics = _sequential_verification_diagnostics(0.70, rhat_passed=False)
    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_final_verification(
            _loop_config(acceptance_band=(0.65, 0.75), repair_band=(0.55, 0.85)),
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "passed"
    assert role == "dependence_aware_fixed_kernel_verification_passed"
    assert hard_vetoes == ()
    assert repair_triggers == ()


@pytest.mark.parametrize("acceptance", [0.82, 0.60])
def test_sequential_rhat_does_not_override_directional_acceptance_repair(
    acceptance: float,
) -> None:
    diagnostics = _sequential_verification_diagnostics(acceptance, rhat_passed=False)
    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_final_verification(
            _loop_config(acceptance_band=(0.65, 0.75), repair_band=(0.55, 0.85)),
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "repair_or_retry"
    assert role == "verification_acceptance_repair_trigger"
    assert hard_vetoes == ()
    assert repair_triggers == ("verification_acceptance_outside_pass_band",)


def test_sequential_verification_passes_with_valid_in_band_evidence() -> None:
    diagnostics = _sequential_verification_diagnostics(0.70, rhat_passed=True)
    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_final_verification(
            _loop_config(acceptance_band=(0.65, 0.75), repair_band=(0.55, 0.85)),
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "passed"
    assert role == "dependence_aware_fixed_kernel_verification_passed"
    assert hard_vetoes == ()
    assert repair_triggers == ()


def test_sequential_verification_four_results_are_inconclusive() -> None:
    diagnostics = _sequential_verification_diagnostics(
        0.70,
        draw_count=4,
        rhat_passed=True,
    )

    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_final_verification(
            _loop_config(),
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "repair_or_retry"
    assert role == "verification_acceptance_inconclusive"
    assert hard_vetoes == ()
    assert repair_triggers == ("verification_acceptance_evidence_inconclusive",)


def test_sequential_verification_rejects_fractional_retained_count() -> None:
    diagnostics = dict(_sequential_verification_diagnostics(0.70))
    diagnostics["retained_sample_count"] = 64.5

    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_final_verification(
            _loop_config(),
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "hard_veto"
    assert role == "shared_invalidity"
    assert hard_vetoes == ("verification_retained_sample_count_invalid",)
    assert repair_triggers == ()


def test_phase7_direct_input_uses_selected_and_nonselected_candidate_lineage() -> None:
    (
        geometry,
        _bootstrap_result,
        windowed,
        fixed,
        handoff,
        selected_identity,
        alternative_identity,
    ) = _phase7_direct_fixture()
    common = {
        "adapter": _ToyGaussianAdapter(),
        "geometry": geometry,
        "windowed_stage": windowed,
        "fixed_mass_step_stage": fixed,
        "config": _loop_config(max_attempts=1),
        "attempt_index": 0,
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
    }
    selected_input = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        **common,
        candidate_identity=selected_identity,
    )
    alternative_input = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        **common,
        candidate_identity=alternative_identity,
    )
    selected_record = handoff.candidate_records[selected_identity[0]]
    alternative_record = handoff.candidate_records[alternative_identity[0]]

    assert selected_input.source_kind == "direct_phase5_candidate"
    assert selected_input.trajectory_stage_artifact_hash is None
    assert selected_input.candidate_batch_hash == handoff.handoff_hash
    assert selected_input.candidate_record_hash == selected_record.record_hash
    assert selected_input.step_size == selected_record.selected_step_size
    assert selected_input.num_leapfrog_steps == selected_record.num_leapfrog_steps
    assert alternative_input.candidate_record_hash == alternative_record.record_hash
    assert alternative_input.selected_step_hash == alternative_record.selected_step_hash
    assert alternative_input.step_size == alternative_record.selected_step_size
    assert alternative_input.num_leapfrog_steps == alternative_record.num_leapfrog_steps
    assert alternative_record.ladder_artifact_hash != fixed.budget_ladder_result.artifact_hash
    assert alternative_input.input_hash != selected_input.input_hash
    assert "step_size" not in repr(selected_input)
    assert "candidate_record_hash" not in repr(selected_input)


def test_phase7_direct_input_fails_closed_on_source_mismatch_before_runner() -> None:
    (
        geometry,
        _bootstrap_result,
        windowed,
        fixed,
        handoff,
        selected_identity,
        _alternative_identity,
    ) = _phase7_direct_fixture()
    common = {
        "adapter": _ToyGaussianAdapter(),
        "geometry": geometry,
        "windowed_stage": windowed,
        "fixed_mass_step_stage": fixed,
        "config": _loop_config(max_attempts=1),
        "attempt_index": 0,
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
    }
    with pytest.raises(ValueError, match="missing or ambiguous"):
        hmc_kernel_tuning._phase7_direct_candidate_verification_input(
            **common,
            candidate_identity=(999, 0, "initial", 0),
        )
    ineligible = next(
        record for record in handoff.candidate_records if not record.handoff_eligible
    )
    with pytest.raises(ValueError, match="not handoff eligible"):
        hmc_kernel_tuning._phase7_direct_candidate_verification_input(
            **common,
            candidate_identity=(
                ineligible.batch_ordinal,
                ineligible.source_round_index,
                ineligible.source_grid_stage,
                ineligible.source_round_candidate_index,
            ),
        )
    selected_input = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        **common,
        candidate_identity=selected_identity,
    )
    with pytest.raises(ValueError, match="input hash mismatch"):
        replace(selected_input, step_size=selected_input.step_size * 2.0)
    with pytest.raises(ValueError, match="forbids Phase 6 lineage"):
        replace(
            selected_input,
            trajectory_stage_artifact_hash="fabricated-phase6",
            input_hash="",
        )


def test_phase7_direct_seed_policy_locks_literal_vectors_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = (7, 1, "final_local", 2)
    assert hmc_kernel_tuning._phase7_direct_candidate_seed(
        (20260621, 70), 0, identity
    ) == (832440578, 1678530773)
    vectors = {
        hmc_kernel_tuning._phase7_direct_candidate_seed((20260621, 70), 1, identity),
        hmc_kernel_tuning._phase7_direct_candidate_seed(
            (20260621, 70), 0, (8, 1, "final_local", 2)
        ),
        hmc_kernel_tuning._phase7_direct_candidate_seed(
            (20260621, 70), 0, (7, 2, "final_local", 2)
        ),
        hmc_kernel_tuning._phase7_direct_candidate_seed(
            (20260621, 70), 0, (7, 1, "final_localx", 2)
        ),
        hmc_kernel_tuning._phase7_direct_candidate_seed(
            (20260621, 70), 0, (7, 1, "final_local", 3)
        ),
    }
    assert vectors == {
        (1793697093, 1935094610),
        (1569281479, 648569285),
        (1388670300, 2034319312),
        (1776738629, 931412166),
        (649492816, 1126868618),
    }
    assert (832440578, 1678530773) not in vectors
    modulus = 2**31 - 1
    attempt_seed = hmc_kernel_tuning._phase7_attempt_seed((20260621, 70), 0)
    forced_digest = (
        f"{(-attempt_seed[0]) % modulus:08x}"
        f"{(-attempt_seed[1]) % modulus:08x}"
        + "0" * 48
    )
    monkeypatch.setattr(hmc_kernel_tuning, "stable_config_hash", lambda _payload: forced_digest)
    assert hmc_kernel_tuning._phase7_direct_candidate_seed(
        (20260621, 70), 0, identity
    ) == (0, 1)


def test_phase7_historical_and_direct_inputs_share_mechanics_but_preserve_seed_policy() -> None:
    (
        geometry,
        bootstrap,
        windowed,
        fixed,
        _handoff,
        selected_identity,
        _alternative_identity,
    ) = _phase7_direct_fixture()

    def phase6_run(_adapter: Any, _state: Any, run_config: Any):
        return _fake_result(
            num_results=int(run_config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(run_config.num_results), 2)),
        )

    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        run_full_chain=phase6_run,
    )
    common = {
        "adapter": _ToyGaussianAdapter(),
        "geometry": geometry,
        "windowed_stage": windowed,
        "fixed_mass_step_stage": fixed,
        "config": _loop_config(max_attempts=1),
        "attempt_index": 0,
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
    }
    historical = hmc_kernel_tuning._phase7_historical_verification_input(
        **common,
        trajectory_stage=trajectory,
    )
    direct = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        **common,
        candidate_identity=selected_identity,
    )

    assert historical.source_kind == "historical_phase6"
    assert historical.candidate_identity is None
    assert historical.trajectory_stage_artifact_hash == trajectory.artifact_hash
    assert historical.verification_seed == (20274747, 18422)
    assert historical.verification_seed == hmc_kernel_tuning._derive_seed(
        hmc_kernel_tuning._phase7_attempt_seed((20260621, 70), 0),
        stage_index=4,
    )
    assert direct.verification_seed == (832440578, 1678530773)
    for field in (
        "target_scope",
        "target_dimension",
        "windowed_stage_artifact_hash",
        "adapted_mass_artifact_signature",
        "fixed_mass_step_stage_artifact_hash",
        "selected_step_hash",
        "step_size",
        "num_leapfrog_steps",
        "adapter_signature",
        "phase4_adapter_signature",
        "verification_hmc_adapter_signature",
    ):
        assert getattr(historical, field) == getattr(direct, field)


@pytest.mark.parametrize(
    ("acceptance", "expected_status", "expected_role"),
    [
        (0.70, "passed", "fresh_fixed_kernel_verification_passed"),
        (0.60, "repair_or_retry", "verification_acceptance_repair_trigger"),
        (0.82, "repair_or_retry", "verification_acceptance_repair_trigger"),
    ],
)
def test_phase7_direct_and_historical_injected_verifiers_share_classification_and_budget(
    acceptance: float,
    expected_status: str,
    expected_role: str,
) -> None:
    (
        geometry,
        bootstrap,
        windowed,
        fixed,
        _handoff,
        selected_identity,
        _alternative_identity,
    ) = _phase7_direct_fixture()

    def phase6_run(_adapter: Any, _state: Any, run_config: Any):
        return _fake_result(
            num_results=int(run_config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(run_config.num_results), 2)),
        )

    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        run_full_chain=phase6_run,
    )
    calls: list[Mapping[str, Any]] = []

    def verification_runner(_adapter: Any, _state: Any, run_config: Any):
        calls.append(
            {
                "num_results": int(run_config.num_results),
                "num_burnin_steps": int(run_config.num_burnin_steps),
                "seed": tuple(run_config.seed),
                "step_size": float(run_config.step_size),
                "num_leapfrog_steps": int(run_config.num_leapfrog_steps),
            }
        )
        return _fake_result(
            num_results=int(run_config.num_results),
            acceptance=acceptance,
            samples=np.zeros((int(run_config.num_results), 2)),
        )

    common = {
        "adapter": _ToyGaussianAdapter(),
        "geometry": geometry,
        "windowed_stage": windowed,
        "fixed_mass_step_stage": fixed,
        "config": _loop_config(max_attempts=1),
        "budget_policy": _tiny_budget_factory(2, 0),
        "attempt_index": 0,
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
        "verification_callback": None,
        "checkpoint_writer_config": None,
        "verification_start_callback": None,
        "checkpoint_reference_callback": None,
        "run_full_chain": verification_runner,
    }
    historical = hmc_kernel_tuning._run_phase7_final_verification(
        **common,
        trajectory_stage=trajectory,
    )
    direct = hmc_kernel_tuning._run_phase7_direct_candidate_verification(
        **common,
        candidate_identity=selected_identity,
    )

    assert len(calls) == 2
    assert [call["num_results"] for call in calls] == [4, 4]
    assert [call["num_burnin_steps"] for call in calls] == [1, 1]
    assert calls[0]["seed"] == (20274747, 18422)
    assert calls[1]["seed"] == (832440578, 1678530773)
    assert calls[0]["step_size"] == calls[1]["step_size"]
    assert calls[0]["num_leapfrog_steps"] == calls[1]["num_leapfrog_steps"]
    assert historical[3:] == (
        expected_status,
        expected_role,
        (),
        () if expected_status == "passed" else ("verification_acceptance_outside_pass_band",),
    )
    assert direct.historical_tuple()[3:] == historical[3:]
    assert direct.verification_input.candidate_identity == selected_identity
    assert direct.continuation_scope == (
        "passed" if expected_status == "passed" else "repair_or_retry"
    )


def _phase7_healthy_diagnostics(acceptance: float = 0.70) -> Mapping[str, Any]:
    return {
        "acceptance_rate": float(acceptance),
        "runtime_finite": True,
        "acceptance_log_health_passed": True,
        "samples_all_finite": True,
        "target_value_health_passed": True,
    }


@pytest.mark.parametrize(
    ("case", "expected_scope", "expected_status"),
    [
        ("runner_exception", "shared_continuation_veto", "hard_veto"),
        ("checkpoint_exception", "shared_continuation_veto", "hard_veto"),
        ("callback_exception", "shared_continuation_veto", "hard_veto"),
        ("callback_continuation", "shared_continuation_veto", "hard_veto"),
        ("candidate_health", "candidate_local_hard_veto", "hard_veto"),
        ("callback_hard_veto", "candidate_local_hard_veto", "hard_veto"),
        ("callback_repair", "repair_or_retry", "repair_or_retry"),
        ("out_of_band", "repair_or_retry", "repair_or_retry"),
        ("passed", "passed", "passed"),
    ],
)
def test_phase7_shared_finalizer_assigns_failure_scope_by_origin(
    case: str,
    expected_scope: str,
    expected_status: str,
) -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, identity, _ = (
        _phase7_direct_fixture()
    )
    verification_input = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        candidate_identity=identity,
    )
    diagnostics = dict(_phase7_healthy_diagnostics())
    callback_result = FixedMassHMCTuningBudgetCallbackResult()
    runner_error = None
    runner_error_origin = None
    callback_error = None
    if case in {"runner_exception", "checkpoint_exception"}:
        runner_error = RuntimeError(case)
        runner_error_origin = "checkpoint" if case == "checkpoint_exception" else "runner"
        diagnostics = dict(hmc_kernel_tuning._bootstrap_error_diagnostics(runner_error))
    elif case == "callback_exception":
        callback_error = RuntimeError(case)
        callback_result = FixedMassHMCTuningBudgetCallbackResult(
            hard_vetoes=("callback_error",)
        )
    elif case == "callback_continuation":
        callback_result = FixedMassHMCTuningBudgetCallbackResult(
            continuation_vetoes=("client_stop",)
        )
    elif case == "candidate_health":
        diagnostics["samples_all_finite"] = False
    elif case == "callback_hard_veto":
        callback_result = FixedMassHMCTuningBudgetCallbackResult(
            hard_vetoes=("client_candidate_veto",)
        )
    elif case == "callback_repair":
        callback_result = FixedMassHMCTuningBudgetCallbackResult(
            repair_triggers=("client_repair",)
        )
    elif case == "out_of_band":
        diagnostics["acceptance_rate"] = 0.82
    execution = hmc_kernel_tuning._HMCPhase7FixedKernelVerificationExecution(
        verification_config_payload={"num_results": 4, "num_burnin_steps": 1},
        diagnostics=diagnostics,
        callback_result=callback_result,
        runner_error=runner_error,
        runner_error_origin=runner_error_origin,
        callback_error=callback_error,
        observed_step_size=verification_input.step_size,
        observed_num_leapfrog_steps=verification_input.num_leapfrog_steps,
    )
    outcome = hmc_kernel_tuning._finalize_phase7_fixed_kernel_verification(
        verification_input=verification_input,
        adapted_mass=hmc_kernel_tuning._phase4_adapted_mass_artifact(windowed),
        config=_loop_config(max_attempts=1),
        execution=execution,
    )

    assert outcome.final_status == expected_status
    assert outcome.continuation_scope == expected_scope
    assert outcome.verification_input.source_identity["candidate_identity"] == identity
    if case == "out_of_band":
        assert outcome.repair_evidence["observed_acceptance_rate"] == pytest.approx(0.82)
        assert outcome.repair_evidence["acceptance_relation"] == "above_acceptance_band"
        assert outcome.repair_evidence["candidate_record_hash"] == (
            verification_input.candidate_record_hash
        )
    elif expected_status != "repair_or_retry":
        assert outcome.repair_evidence is None


def _phase4_direct_outcome(
    *,
    fixed: Any,
    windowed: Any,
    identity: tuple[int, int, str, int],
    acceptance: float = 0.70,
    callback_result: FixedMassHMCTuningBudgetCallbackResult | None = None,
    shared_error: Exception | None = None,
    sequential_diagnostics: Mapping[str, Any] | None = None,
) -> Any:
    verification_input = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        candidate_identity=identity,
    )
    execution = hmc_kernel_tuning._HMCPhase7FixedKernelVerificationExecution(
        verification_config_payload={
            "verification_policy": "fixed_kernel",
            "num_results": 4,
            "num_burnin_steps": 1,
            "acceptance_band": (0.65, 0.75),
        },
        diagnostics=(
            hmc_kernel_tuning._bootstrap_error_diagnostics(shared_error)
            if shared_error is not None
            else (
                _phase7_healthy_diagnostics(acceptance)
                if sequential_diagnostics is None
                else sequential_diagnostics
            )
        ),
        callback_result=(
            FixedMassHMCTuningBudgetCallbackResult()
            if callback_result is None
            else callback_result
        ),
        runner_error=shared_error,
        runner_error_origin=None if shared_error is None else "runner",
        callback_error=None,
        observed_step_size=verification_input.step_size,
        observed_num_leapfrog_steps=verification_input.num_leapfrog_steps,
    )
    return hmc_kernel_tuning._finalize_phase7_fixed_kernel_verification(
        verification_input=verification_input,
        adapted_mass=hmc_kernel_tuning._phase4_adapted_mass_artifact(windowed),
        config=_loop_config(max_attempts=1),
        execution=execution,
    )


def _phase4_stage_with_status(
    fixed: Any,
    *,
    status: str,
) -> Any:
    """Create a source-valid joint status fixture from a completed batch."""

    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(fixed)
    assert handoff is not None
    eligible_order = tuple(
        record.batch_ordinal
        for record in handoff.candidate_records
        if record.handoff_eligible
    )
    status_handoff = replace(
        handoff,
        final_status=status,
        selected_batch_ordinal=None,
        selected_record_hash=None,
        repair_batch_ordinal=None,
        repair_record_hash=None,
        verification_order_seed=eligible_order,
        handoff_hash="",
    )
    status_fixed = replace(
        fixed,
        final_status=status,
        diagnostic_role="hard_veto" if status == "hard_veto" else "repair_trigger",
        hard_vetoes=("phase5_fixture_hard_veto",) if status == "hard_veto" else (),
        repair_triggers=(f"phase5_fixture_{status}",),
        selected_step_payload=None,
        selected_step_hash=None,
        repair_step_payload=None,
        repair_step_hash=None,
        _candidate_batch_handoff=status_handoff,
    )
    assert hmc_kernel_tuning._phase5_candidate_batch_handoff(status_fixed) is not None
    return status_fixed


def _phase4_repair_with_eligible_stage(fixed: Any) -> Any:
    return _phase4_stage_with_status(fixed, status="repair_or_retry")


def test_phase4_direct_queue_plan_precomputes_complete_unique_seed_map() -> None:
    geometry, _bootstrap_result, windowed, fixed, handoff, selected, alternative = (
        _phase7_direct_fixture()
    )
    plan = hmc_kernel_tuning._phase7_direct_candidate_queue_plan(
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
    )

    assert plan.candidate_count == handoff.handoff_eligible_count
    assert plan.candidate_identities[0] == selected
    assert alternative in plan.candidate_identities
    assert tuple(identity[0] for identity in plan.candidate_identities[1:]) == tuple(
        sorted(identity[0] for identity in plan.candidate_identities[1:])
    )
    assert len(set(plan.verification_seeds)) == plan.candidate_count
    assert plan.maximum_candidate_starts == 2
    assert plan.allocated_start_count == 2
    assert plan.verification_num_results == 4
    assert plan.verification_num_burnin_steps == 1
    assert plan.payload()["total_result_cap"] == 8
    assert plan.payload()["total_burnin_cap"] == 2
    assert "candidate_identities" not in repr(plan)


def test_phase4_direct_queue_rejects_seed_collision_before_any_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, _bootstrap_result, _windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_phase7_direct_candidate_seed",
        lambda *_args, **_kwargs: (17, 19),
    )

    with pytest.raises(ValueError, match="seed collision"):
        hmc_kernel_tuning._phase7_direct_candidate_queue_plan(
            fixed_mass_step_stage=fixed,
            config=_loop_config(max_attempts=1),
            budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
            attempt_index=0,
        )


def test_phase4_direct_queue_continues_candidate_local_then_admits_second() -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, alternative = (
        _phase7_direct_fixture()
    )
    calls: list[tuple[int, int, str, int]] = []

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        if len(calls) == 1:
            return _phase4_direct_outcome(
                fixed=fixed,
                windowed=windowed,
                identity=candidate_identity,
                callback_result=FixedMassHMCTuningBudgetCallbackResult(
                    hard_vetoes=("candidate_fixture_veto",)
                ),
            )
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
        )

    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fake verifier must not call HMC")
        ),
        verification_runner=verifier,
    )

    assert calls == [selected, alternative]
    assert result.final_status == "passed"
    assert result.started_count == 2
    assert result.candidate_results[0]["state"] == "candidate_local_hard_veto"
    assert result.candidate_results[1]["state"] == "passed"
    assert all(
        item["state"] == "not_run" and item["not_run_reason"] == "first_admission"
        for item in result.candidate_results[2:]
    )


def test_phase7_v3_cost_stop_preserves_valid_pass_evidence_without_admission() -> None:
    diagnostics = _sequential_verification_diagnostics(
        0.70,
        cost_stop_reasons=("persistent_candidate_cost_stop",),
    )

    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_acceptance_evidence_verification(
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "budget_exhausted"
    assert role == "candidate_cost_stop_nonpromoting"
    assert hard_vetoes == ()
    assert repair_triggers == ()
    evidence = diagnostics["acceptance_evidence"]
    assert evidence["evidence_validity"] == "valid"
    assert evidence["acceptance_decision"] == "passed"
    assert evidence["promotion_eligible"] is False
    assert evidence["cost_stop_scope"] == "exact_candidate_replication"


def test_phase7_v3_native_divergence_is_veto_only_when_acceptance_passes() -> None:
    diagnostics = _sequential_verification_diagnostics(
        0.70,
        native_divergence_count=1,
    )

    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_acceptance_evidence_verification(
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "budget_exhausted"
    assert role == "candidate_promotion_veto_nonpromoting"
    assert hard_vetoes == ()
    assert repair_triggers == ()
    evidence = diagnostics["acceptance_evidence"]
    assert evidence["acceptance_decision"] == "passed"
    assert evidence["candidate_promotion_vetoes"] == (
        "native_divergence_positive",
    )
    assert evidence["promotion_eligible"] is False


def test_phase7_v3_direction_survives_native_divergence_veto() -> None:
    diagnostics = _sequential_verification_diagnostics(
        0.40,
        native_divergence_count=1,
    )

    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_acceptance_evidence_verification(
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )
    )

    assert status == "repair_or_retry"
    assert role == "verification_acceptance_repair_trigger"
    assert hard_vetoes == ()
    assert repair_triggers == ("verification_acceptance_outside_pass_band",)


def test_phase7_v3_callback_promotion_veto_is_not_a_repair_trigger() -> None:
    diagnostics = _sequential_verification_diagnostics(0.70)

    status, role, hard_vetoes, repair_triggers = (
        hmc_kernel_tuning._classify_phase7_acceptance_evidence_verification(
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(
                promotion_vetoes=("domain_screen_not_yet_passed",),
            ),
        )
    )

    assert status == "budget_exhausted"
    assert role == "candidate_promotion_veto_nonpromoting"
    assert hard_vetoes == ()
    assert repair_triggers == ()


def test_phase4_direct_queue_continues_after_cost_stop_then_admits_peer() -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, alternative = (
        _phase7_direct_fixture()
    )
    calls: list[tuple[int, int, str, int]] = []

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        if len(calls) == 1:
            return _phase4_direct_outcome(
                fixed=fixed,
                windowed=windowed,
                identity=candidate_identity,
                sequential_diagnostics=_sequential_verification_diagnostics(
                    0.70,
                    cost_stop_reasons=("persistent_candidate_cost_stop",),
                ),
            )
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
        )

    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )

    assert calls == [selected, alternative]
    assert result.final_status == "passed"
    assert result.candidate_results[0]["state"] == "candidate_local_cost_stop"
    assert result.candidate_results[0]["cost_stop_reasons"] == (
        "persistent_candidate_cost_stop",
    )
    assert result.candidate_results[0]["cost_stop_scope"] == (
        "exact_candidate_replication"
    )
    assert result.candidate_results[0]["acceptance_evidence"][
        "acceptance_decision"
    ] == "passed"
    assert result.candidate_results[0]["acceptance_evidence"][
        "promotion_eligible"
    ] is False
    assert result.candidate_results[1]["state"] == "passed"


def test_phase4_direct_queue_continues_after_promotion_veto_then_admits_peer() -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, alternative = (
        _phase7_direct_fixture()
    )
    calls: list[tuple[int, int, str, int]] = []

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        if len(calls) == 1:
            return _phase4_direct_outcome(
                fixed=fixed,
                windowed=windowed,
                identity=candidate_identity,
                sequential_diagnostics=_sequential_verification_diagnostics(
                    0.70,
                    native_divergence_count=1,
                ),
            )
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
        )

    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )

    assert calls == [selected, alternative]
    assert result.final_status == "passed"
    assert result.candidate_results[0]["state"] == (
        "candidate_local_promotion_veto"
    )
    assert result.candidate_results[0]["candidate_promotion_vetoes"] == (
        "native_divergence_positive",
    )
    assert result.candidate_results[1]["state"] == "passed"


def test_phase4_direct_queue_all_cost_stopped_is_terminal_nonpromotion() -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, alternative = (
        _phase7_direct_fixture()
    )
    calls: list[tuple[int, int, str, int]] = []

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
            sequential_diagnostics=_sequential_verification_diagnostics(
                0.70,
                cost_stop_reasons=("persistent_candidate_cost_stop",),
            ),
        )

    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=2),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )

    assert calls == [selected, alternative]
    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "candidate_cost_stop_nonpromoting"
    assert result.repair_outcome is None
    assert result.cost_stopped_count == 2
    assert result.repair_triggers == ()
    assert all(
        item["state"] == "candidate_local_cost_stop"
        for item in result.candidate_results[:2]
    )


def test_phase4_direct_queue_stops_shared_veto_and_preserves_not_run() -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, _alternative = (
        _phase7_direct_fixture()
    )
    calls: list[tuple[int, int, str, int]] = []

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
            shared_error=RuntimeError("shared fixture failure"),
        )

    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )

    assert calls == [selected]
    assert result.final_status == "hard_veto"
    assert result.candidate_results[0]["state"] == "shared_continuation_veto"
    assert all(
        item["state"] == "not_run"
        and item["not_run_reason"] == "shared_continuation_veto"
        for item in result.candidate_results[1:]
    )


def test_phase4_direct_queue_conflicting_repairs_do_not_mutate_epsilon() -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, alternative = (
        _phase7_direct_fixture()
    )
    acceptances = iter((0.82, 0.60))
    calls: list[tuple[int, int, str, int]] = []

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
            sequential_diagnostics=_sequential_verification_diagnostics(
                next(acceptances)
            ),
        )

    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )

    assert calls == [selected, alternative]
    assert result.final_status == "repair_or_retry"
    assert result.diagnostic_role == "verification_acceptance_conflict"
    assert result.started_count == 2
    assert result.repair_outcome.verification_input.candidate_identity == selected
    assert result.repair_directions == ("higher_epsilon", "lower_epsilon")
    assert result.repair_direction_conflict is True
    queue = result.private_diagnostics()["phase7_direct_candidate_queue"]
    assert queue["schema"] == (
        "bayesfilter.hmc_phase7_direct_candidate_queue_result.v2"
    )
    assert queue["repair_direction_conflict"] is True
    assert "verification_acceptance_inconclusive_conflict" in result.repair_triggers
    assert all(
        item["state"] == "not_run"
        and item["not_run_reason"] == "start_quota_exhausted"
        for item in result.candidate_results[2:]
    )


def test_phase4_joint_outer_loop_bypasses_phase6_and_uses_direct_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, bootstrap, windowed, fixed, _handoff, selected, _alternative = (
        _phase7_direct_fixture()
    )
    direct_calls: list[tuple[int, int, str, int]] = []
    phase6_calls: list[str] = []

    def direct_verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        direct_calls.append(candidate_identity)
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
        )

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_phase7_direct_candidate_verification",
        direct_verifier,
    )
    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=1),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: phase6_calls.append(
            "unexpected"
        ),
        run_full_chain=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fake direct verifier must not call HMC")
        ),
    )

    assert result.passed is True
    assert direct_calls == [selected]
    assert phase6_calls == []
    assert result.attempts[0].frozen_step_trajectory_stage is None
    assert result.attempts[0].handoff_state_payload["handoff_stage"] == (
        "phase7_direct"
    )


def test_phase4_repair_status_with_eligible_records_runs_direct_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, bootstrap, windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    fixed = _phase4_repair_with_eligible_stage(fixed)
    calls: list[tuple[int, int, str, int]] = []

    def direct_verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
        )

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_phase7_direct_candidate_verification",
        direct_verifier,
    )
    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=1),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Phase 6 must remain bypassed")
        ),
    )

    assert result.passed is True
    assert len(calls) == 1
    assert result.attempts[0].verification_diagnostics[
        "phase7_direct_candidate_queue"
    ]["started_count"] == 1


def test_phase4_repair_without_eligible_records_preserves_phase5_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.83, 5: 0.84, 7: 0.84})
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(fixed)
    assert handoff is not None and handoff.handoff_eligible_count == 0
    direct_calls: list[str] = []
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_phase7_direct_candidate_verification",
        lambda **_kwargs: direct_calls.append("unexpected"),
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=1),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Phase 6 must remain bypassed")
        ),
    )

    assert direct_calls == []
    attempt = result.attempts[0]
    assert attempt.final_status == "repair_or_retry"
    assert attempt.handoff_state_payload["handoff_stage"] == "phase5_repair"
    assert attempt.handoff_state_payload["selected_step_hash"] == fixed.repair_step_hash
    queue = attempt.verification_diagnostics["phase7_direct_candidate_queue"]
    assert queue["started_count"] == 0
    assert all(
        item["state"] == "not_run"
        and item["not_run_reason"] == "phase5_repair_handoff_without_eligible_candidate"
        for item in queue["candidate_results"]
    )


@pytest.mark.parametrize("status", ["budget_exhausted", "hard_veto"])
def test_phase4_terminal_phase5_status_marks_all_candidates_not_run(
    status: str,
) -> None:
    geometry, bootstrap, windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    fixed = _phase4_stage_with_status(fixed, status=status)
    phase6_calls: list[str] = []
    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=1),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: phase6_calls.append(
            "unexpected"
        ),
    )

    assert result.final_status == status
    assert phase6_calls == []
    queue = result.attempts[0].verification_diagnostics[
        "phase7_direct_candidate_queue"
    ]
    assert queue["started_count"] == 0
    assert queue["not_run_count"] == queue["candidate_count"]
    assert all(item["state"] == "not_run" for item in queue["candidate_results"])


def test_phase4_direct_queue_without_finite_repair_is_architecture_blocked() -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, alternative = (
        _phase7_direct_fixture()
    )
    calls: list[tuple[int, int, str, int]] = []

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(
                hard_vetoes=("candidate_fixture_veto",)
            ),
        )

    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )

    assert calls == [selected, alternative]
    assert result.final_status == "architecture_blocked"
    assert result.repair_outcome is None
    assert "phase7_direct_queue_no_finite_repair_handoff" in result.repair_triggers


@pytest.mark.parametrize("timeout_position", ["before_first", "before_second"])
def test_phase4_direct_queue_timeout_closeout_marks_unstarted_candidates(
    timeout_position: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, selected, _alternative = (
        _phase7_direct_fixture()
    )
    clock = {"now": 0.0}
    calls: list[tuple[int, int, str, int]] = []
    monkeypatch.setattr(
        hmc_kernel_tuning.time,
        "perf_counter",
        lambda: float(clock["now"]),
    )

    def verifier(*, candidate_identity: tuple[int, int, str, int], **_kwargs: Any):
        calls.append(candidate_identity)
        clock["now"] = 100.0
        return _phase4_direct_outcome(
            fixed=fixed,
            windowed=windowed,
            identity=candidate_identity,
            acceptance=0.82,
        )

    timeout_budget = 100.0 if timeout_position == "before_first" else 200.0
    result = hmc_kernel_tuning._run_phase7_direct_candidate_queue(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(
            max_attempts=1,
            public_timeout_budget_s=timeout_budget,
            public_timeout_started_perf_counter_s=0.0,
        ),
        budget_policy=_tiny_budget_factory(geometry.target_dimension, 0),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        verification_callback=None,
        checkpoint_writer_config=None,
        progress_callback=None,
        run_full_chain=lambda *_args, **_kwargs: None,
        verification_runner=verifier,
    )

    expected_calls = [] if timeout_position == "before_first" else [selected]
    assert calls == expected_calls
    assert result.final_status == (
        "budget_exhausted"
        if timeout_position == "before_first"
        else "repair_or_retry"
    )
    assert (result.repair_outcome is not None) == (
        timeout_position == "before_second"
    )
    assert result.timeout_closeout["closeout_required_before_next_candidate"] is True
    assert result.started_count == len(expected_calls)
    assert all(
        item["state"] == "not_run"
        for item in result.candidate_results[len(expected_calls):]
    )


def test_phase4_private_seed_report_is_complete_and_public_summaries_redact_it() -> None:
    run, _calls = _scripted_full_chain_runner(verification_acceptances=[0.70])
    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=1),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    reports = result.seed_report["direct_candidate_verification_seed_maps"]
    assert len(reports) == 1
    report = reports[0]
    queue = result.attempts[0].verification_diagnostics[
        "phase7_direct_candidate_queue"
    ]
    assert report["seed_map_precomputed"] is True
    assert report["verification_seed_policy"] == (
        "bayesfilter.phase7_direct_candidate_seed.v1"
    )
    assert len(report["candidate_seed_map"]) == queue["candidate_count"]
    assert tuple(
        (item["candidate_identity"], item["verification_seed"])
        for item in report["candidate_seed_map"]
    ) == tuple(
        (item["candidate_identity"], item["verification_seed"])
        for item in queue["candidate_results"]
    )

    public_text = json.dumps(
        {
            "phase7": hmc_kernel_tuning._phase7_public_summary(result),
            "final": hmc_kernel_tuning._public_final_kernel_handoff_payload(result),
        },
        sort_keys=True,
    )
    for forbidden in (
        "candidate_identity",
        "candidate_seed_map",
        "verification_seed",
        "candidate_batch_hash",
        "bayesfilter.phase7_direct_candidate_seed.v1",
    ):
        assert forbidden not in public_text


@pytest.mark.parametrize(
    ("field", "expected_veto"),
    [
        ("mass", "verification_mass_signature_mutated"),
        ("step", "verification_step_size_mutated"),
        ("leapfrog", "verification_leapfrog_count_mutated"),
    ],
)
def test_phase7_shared_finalizer_treats_fixed_kernel_mutation_as_shared(
    field: str,
    expected_veto: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, identity, _ = (
        _phase7_direct_fixture()
    )
    verification_input = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=_loop_config(max_attempts=1),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        candidate_identity=identity,
    )
    adapted_mass = hmc_kernel_tuning._phase4_adapted_mass_artifact(windowed)
    if field == "mass":
        monkeypatch.setattr(
            hmc_kernel_tuning,
            "_mass_artifact_signature",
            lambda _mass: "mutated-mass-signature",
        )
    execution = hmc_kernel_tuning._HMCPhase7FixedKernelVerificationExecution(
        verification_config_payload={"num_results": 4, "num_burnin_steps": 1},
        diagnostics=_phase7_healthy_diagnostics(),
        callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        runner_error=None,
        runner_error_origin=None,
        callback_error=None,
        observed_step_size=(
            verification_input.step_size * 2.0
            if field == "step"
            else verification_input.step_size
        ),
        observed_num_leapfrog_steps=(
            verification_input.num_leapfrog_steps + 1
            if field == "leapfrog"
            else verification_input.num_leapfrog_steps
        ),
    )
    outcome = hmc_kernel_tuning._finalize_phase7_fixed_kernel_verification(
        verification_input=verification_input,
        adapted_mass=adapted_mass,
        config=_loop_config(max_attempts=1),
        execution=execution,
    )
    assert outcome.final_status == "hard_veto"
    assert outcome.continuation_scope == "shared_continuation_veto"
    assert expected_veto in outcome.hard_vetoes


@pytest.mark.parametrize(
    ("field", "value", "expected_veto"),
    [
        ("step", np.array([0.1, 0.1]), "verification_step_size_mutated"),
        ("step", True, "verification_step_size_mutated"),
        ("leapfrog", 4.5, "verification_leapfrog_count_mutated"),
        ("leapfrog", np.array([4, 4]), "verification_leapfrog_count_mutated"),
        ("leapfrog", True, "verification_leapfrog_count_mutated"),
    ],
)
def test_phase7_shared_finalizer_rejects_nonscalar_or_noninteger_mechanics(
    field: str,
    value: object,
    expected_veto: str,
) -> None:
    geometry, _bootstrap_result, windowed, fixed, _handoff, identity, _ = (
        _phase7_direct_fixture()
    )
    verification_input = hmc_kernel_tuning._phase7_direct_candidate_verification_input(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        candidate_identity=identity,
        config=_loop_config(max_attempts=1),
        attempt_index=0,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )
    execution = hmc_kernel_tuning._HMCPhase7FixedKernelVerificationExecution(
        verification_config_payload={"num_results": 64, "num_burnin_steps": 1},
        diagnostics=_sequential_verification_diagnostics(0.70),
        callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        runner_error=None,
        runner_error_origin=None,
        callback_error=None,
        observed_step_size=(
            value if field == "step" else verification_input.step_size
        ),
        observed_num_leapfrog_steps=(
            value if field == "leapfrog" else verification_input.num_leapfrog_steps
        ),
    )

    outcome = hmc_kernel_tuning._finalize_phase7_fixed_kernel_verification(
        verification_input=verification_input,
        adapted_mass=hmc_kernel_tuning._phase4_adapted_mass_artifact(windowed),
        config=_loop_config(max_attempts=1),
        execution=execution,
    )

    assert outcome.final_status == "hard_veto"
    assert outcome.continuation_scope == "shared_continuation_veto"
    assert expected_veto in outcome.hard_vetoes


def test_verification_high_acceptance_handoff_supplies_private_repair_step() -> None:
    repair = _phase7_verification_repair_handoff_payload(
        config=_loop_config(acceptance_band=(0.65, 0.75), repair_band=(0.55, 0.85)),
        selected_step_size=0.125,
        selected_step_hash="selected-step-hash",
        verification_config_payload={"verification_policy": "sequential_rhat"},
        verification_diagnostics={
            "acceptance_rate": 0.82,
            "sequential_rhat_verification": True,
            "acceptance_evidence": _acceptance_evidence_payload(0.82),
        },
        verification_final_status="repair_or_retry",
        verification_diagnostic_role="verification_acceptance_repair_trigger",
        verification_repair_triggers=("verification_acceptance_outside_pass_band",),
        verification_reserved=True,
        enforce_reservation=True,
        use_directional_trust_region=True,
    )
    state = _HMCPhaseAttemptState(
        mass_artifact_payload={"dimension": 2},
        mass_artifact_signature="mass-signature",
        selected_step_size=0.125,
        selected_step_hash="selected-step-hash",
        selected_num_leapfrog_steps=9,
        selected_trajectory_hash="trajectory-hash",
        handoff_stage="phase6",
        **repair,
    )

    assert state.verification_acceptance_rate == pytest.approx(0.82)
    assert state.verification_acceptance_relation == "above_acceptance_band"
    assert state.verification_repair_trigger == "verification_acceptance_outside_pass_band"
    assert state.verification_repair_source == "phase7_final_verification_acceptance"
    assert state.verification_repair_step_size == pytest.approx(0.25)
    assert state.verification_repair_step_hash is not None
    assert state.verification_repair_applied is True
    assert state.verification_repair_direction == "higher_epsilon"
    assert state.repair_direction_history == ("higher_epsilon",)
    assert state.repaired_step_history == pytest.approx((0.25,))
    assert state.repair_verification_reserved is True
    assert state.fixed_mass_bracket_state["bracket_role"] == (
        "local_directional_trust_region_not_empirical_acceptance_bracket"
    )
    assert state.payload()["verification_repair_applied"] is True
    assert _fixed_mass_step_initial_step(_windowed_stage(), attempt_state=state) == pytest.approx(0.25)


def test_verification_mixed_rhat_acceptance_handoff_supplies_private_repair_step() -> None:
    repair = _phase7_verification_repair_handoff_payload(
        config=_loop_config(acceptance_band=(0.65, 0.75), repair_band=(0.55, 0.85)),
        selected_step_size=0.125,
        selected_step_hash="selected-step-hash",
        verification_config_payload={"verification_policy": "sequential_rhat"},
        verification_diagnostics={
            "acceptance_rate": 0.82,
            "sequential_rhat_verification": True,
            "acceptance_evidence": _acceptance_evidence_payload(0.82),
            "all_finite_rhat_at_or_below_threshold": False,
            "cap_hit": True,
        },
        verification_final_status="repair_or_retry",
        verification_diagnostic_role="verification_rhat_repair_trigger",
        verification_repair_triggers=(
            "verification_rhat_above_threshold_or_cap_hit",
            "verification_rhat_cap_hit",
            "verification_acceptance_outside_pass_band",
        ),
        verification_reserved=True,
        enforce_reservation=True,
        use_directional_trust_region=True,
    )

    assert repair["verification_acceptance_relation"] == "above_acceptance_band"
    assert repair["verification_repair_trigger"] == "verification_acceptance_outside_pass_band"
    assert repair["verification_repair_source"] == "phase7_final_verification_acceptance"
    assert repair["verification_repair_step_size"] == pytest.approx(0.25)
    assert repair["verification_repair_step_hash"] is not None
    assert repair["verification_repair_applied"] is True


def test_verification_in_band_handoff_does_not_create_repair_step() -> None:
    repair = _phase7_verification_repair_handoff_payload(
        config=_loop_config(acceptance_band=(0.65, 0.75), repair_band=(0.55, 0.85)),
        selected_step_size=0.125,
        selected_step_hash="selected-step-hash",
        verification_config_payload={"verification_policy": "sequential_rhat"},
        verification_diagnostics={"acceptance_rate": 0.70},
        verification_final_status="passed",
        verification_diagnostic_role="sequential_rhat_fixed_kernel_verification_passed",
        verification_repair_triggers=(),
    )

    assert repair["verification_acceptance_relation"] == "inside_acceptance_band"
    assert repair["verification_repair_step_size"] is None
    assert repair["verification_repair_step_hash"] is None
    assert repair["verification_repair_applied"] is False


def test_outer_loop_progress_callback_marks_internal_substages() -> None:
    run, _calls = _scripted_full_chain_runner(verification_acceptances=[0.70])
    events: list[tuple[str, Mapping[str, Any]]] = []

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=1),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    stage_names = [stage for stage, _payload in events]
    assert result.passed is True
    expected_prefix = [
        "loop_attempt_start",
        "windowed_mass_start",
        "windowed_mass_runner_build_start",
        "windowed_mass_runner_build_complete",
        "windowed_mass_runner_execute_start",
        "windowed_mass_runner_execute_complete",
        "windowed_mass_capture_start",
        "windowed_mass_capture_complete",
        "windowed_mass_semantic_diagnostic_start",
        "windowed_mass_semantic_diagnostic_complete",
        "windowed_mass_complete",
        "fixed_mass_step_start",
    ]
    assert stage_names[: len(expected_prefix)] == expected_prefix
    assert stage_names.count("fixed_mass_ladder_tune_call_start") > 1
    assert stage_names.count("fixed_mass_ladder_screen_call_start") > 1
    assert "fixed_mass_step_complete" in stage_names
    assert stage_names.count("trajectory_candidate_call_start") == 0
    assert stage_names.count("trajectory_candidate_call_complete") == 0
    assert stage_names.count("direct_verification_queue_start") == 1
    assert stage_names.count("direct_verification_queue_complete") == 1
    assert stage_names[-5:] == [
        "verification_start",
        "verification_complete",
        "direct_verification_queue_complete",
        "loop_attempt_complete",
        "loop_complete",
    ]
    first_budget = events[0][1]["bounded_public_budget_payload"]
    assert first_budget["target_dimension"] == 2
    assert first_budget["budget"] == 8
    assert first_budget["substage_budget_details_exposed"] is False
    assert first_budget["hmc_mechanics_exposed"] is False
    forbidden_progress_keys = {
        "step_size",
        "num_leapfrog_steps",
        "acceptance_rate",
        "runtime_metadata",
        "raw_diagnostics",
        "trace",
        "samples",
        "mass_artifact_payload",
        "diagnostic_config",
    }
    inner_events = [
        payload
        for stage, payload in events
        if stage.startswith("windowed_mass_") and stage not in {"windowed_mass_start", "windowed_mass_complete"}
    ]
    assert inner_events
    for payload in inner_events:
        assert payload["hmc_mechanics_exposed"] is False
        assert payload["reports_posterior_convergence"] is False
        extra = payload["extra"]
        assert set(extra).isdisjoint(forbidden_progress_keys)
        assert extra["route_category"] == "injected_runner"
        assert extra["hmc_mechanics_exposed"] is False
        assert extra["progress_only"] is True
        if payload["started"] is True:
            assert extra["elapsed_s"] == pytest.approx(0.0)
            assert extra["started_perf_counter_s"] >= 0.0
            assert extra["timing_anchor_role"] == "process_local_monotonic_debug_only"
    boundary_events = [
        (stage, payload)
        for stage, payload in events
        if stage.startswith("fixed_mass_ladder_")
        or stage.startswith("trajectory_candidate_")
    ]
    assert boundary_events
    for stage, payload in boundary_events:
        assert payload["hmc_mechanics_exposed"] is False
        assert payload["reports_posterior_convergence"] is False
        assert payload["bounded_public_budget_payload"][
            "substage_budget_details_exposed"
        ] is True
        extra = payload["extra"]
        assert set(extra).isdisjoint(forbidden_progress_keys)
        assert extra["hmc_mechanics_exposed"] is False
        assert extra["progress_only"] is True
        assert extra["substage_budget_details_exposed"] is True
        assert "call_config_hash" in extra
        assert "num_results" in extra
        assert "num_burnin_steps" in extra
        if stage.startswith("fixed_mass_ladder_"):
            assert extra["round_index"] == 0
            assert extra["budget"] in {2}
            assert extra["role"] in {"tune", "screen"}
        if payload["started"] is True:
            assert extra["elapsed_s"] == pytest.approx(0.0)
            assert extra["started_perf_counter_s"] >= 0.0
            assert extra["timing_anchor_role"] == "process_local_monotonic_debug_only"
        else:
            assert "started_perf_counter_s" not in extra
            assert "timing_anchor_role" not in extra
    assert events[-1][1]["extra"]["final_status"] == "passed"
    assert all(payload["reports_posterior_convergence"] is False for _stage, payload in events)


def test_outer_loop_progress_helper_allowlists_timing_anchors_without_private_mechanics() -> None:
    budget_extra = hmc_kernel_tuning._budget_ladder_progress_extra(
        {
            "stage": "fixed_mass_ladder_tune_call_start",
            "round_index": 0,
            "budget": 8,
            "role": "tune",
            "started": True,
            "completed": False,
            "route_category": "reusable_runner",
            "call_config_hash": "public-call-hash",
            "num_results": 4,
            "num_burnin_steps": 2,
            "substage_budget_details_exposed": True,
            "uses_dual_averaging": True,
            "runner_reused": False,
            "static_contract_hash": "public-static-contract-hash",
            "elapsed_s": 0.0,
            "started_perf_counter_s": 123.0,
            "timing_anchor_role": "process_local_monotonic_debug_only",
            "progress_only": True,
            "hmc_mechanics_exposed": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
            "reports_external_client_scientific_claim": False,
            "reports_gpu_or_xla_readiness": False,
            "nonclaims": ("progress only",),
            "step_size": 0.1,
            "num_leapfrog_steps": 5,
            "mass_artifact_payload": {"private": True},
            "samples": [[0.0]],
            "trace": {"target_log_prob": [0.0]},
            "final_state": [0.0],
        }
    )
    assert budget_extra["started_perf_counter_s"] == pytest.approx(123.0)
    assert budget_extra["timing_anchor_role"] == "process_local_monotonic_debug_only"
    assert budget_extra["elapsed_s"] == pytest.approx(0.0)

    trajectory_config = hmc_budget_ladder.FullChainHMCConfig(
        num_results=4,
        num_burnin_steps=2,
        step_size=0.1,
        num_leapfrog_steps=5,
        seed=(20260630, 1),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
    )
    trajectory_extra = hmc_kernel_tuning._trajectory_candidate_progress_extra(
        stage="trajectory_candidate_call_start",
        candidate_index=0,
        candidate_count=3,
        config=trajectory_config,
        runner_event=None,
        elapsed_s=0.0,
        started_perf_counter_s=456.0,
    )
    assert trajectory_extra["started_perf_counter_s"] == pytest.approx(456.0)
    assert trajectory_extra["timing_anchor_role"] == "process_local_monotonic_debug_only"
    assert trajectory_extra["elapsed_s"] == pytest.approx(0.0)

    forbidden = {
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    }
    assert set(budget_extra).isdisjoint(forbidden)
    assert set(trajectory_extra).isdisjoint(forbidden)


def test_outer_loop_direct_queue_advances_and_emits_truthful_private_handoff() -> None:
    run, calls = _scripted_full_chain_runner(verification_acceptances=[0.82, 0.70])
    private_events: list[tuple[str, Mapping[str, Any]]] = []

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=3),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
        _private_diagnostic_callback=lambda event_type, payload: private_events.append(
            (event_type, dict(payload))
        ),
    )

    assert result.final_status == "passed"
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.final_status == "passed"
    queue = attempt.verification_diagnostics["phase7_direct_candidate_queue"]
    assert queue["started_count"] == 2
    assert queue["candidate_results"][0]["state"] == "repair_or_retry"
    assert queue["candidate_results"][1]["state"] == "passed"
    assert attempt.frozen_step_trajectory_stage is None
    assert attempt.handoff_state_payload["handoff_stage"] == "phase7_direct"
    assert result.attempts[0].handoff_state_payload["required_private_handoff_complete"] is True
    assert result.attempts[0].handoff_state_payload["final_kernel_handoff_complete"] is True
    assert (
        result.attempts[0].handoff_state_payload["verification_acceptance_relation"]
        == "inside_acceptance_band"
    )
    assert result.attempts[0].handoff_state_payload["verification_repair_applied"] is False
    assert result.attempts[0].budget_policy_payload["budget"] == 8
    assert result.attempts[0].verification_config_payload["num_results"] == 4
    assert result.final_kernel_payload["attempt_index"] == 0
    assert result.final_kernel_payload["selected_trajectory_hash"] is None
    assert result.final_kernel_payload[
        "frozen_step_trajectory_stage_artifact_hash"
    ] is None
    assert len([call for call in calls if not call["uses_dual_averaging"]]) > 1
    handoff_events = [
        payload
        for event_type, payload in private_events
        if event_type == "phase7_handoff_kernel_change"
    ]
    assert handoff_events
    assert handoff_events[0]["handoff_stage"] == "phase7_direct"
    assert handoff_events[0]["verification_repair_applied"] is False
    assert handoff_events[0]["selected_trajectory_hash"] is None
    assert handoff_events[0]["num_leapfrog_steps"] > 0
    assert handoff_events[0]["private_hmc_mechanics"] is True


def test_phase7_resume_split_contract_public_summary_is_sanitized() -> None:
    run, _calls = _scripted_full_chain_runner(verification_acceptances=[0.82, 0.70])

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    contract = hmc_kernel_tuning._phase7_private_resume_split_contract(
        loop=result,
        attempt_index=0,
    )
    summary = hmc_kernel_tuning._phase7_resume_split_public_summary(contract)

    assert contract["schema"] == "bayesfilter.phase7_private_resume_split_contract.v1"
    assert contract["private_resume_payload_only"] is True
    assert contract["private_handoff_state"]["handoff_stage"] == "phase7_direct"
    assert contract["private_handoff_state_hash"]
    assert contract["contract_hash"]
    assert contract["verifier_entry_manifest"] is False
    assert contract["final_kernel_handoff"] is False
    assert summary["schema"] == "bayesfilter.phase7_resume_split_public_summary.v1"
    assert summary["contract_hash"] == contract["contract_hash"]
    assert summary["private_resume_payload_hash"] == contract["private_handoff_state_hash"]
    assert summary["handoff_stage"] == "phase7_direct"
    assert (
        summary["resume_entry_stage"]
        == "phase7_direct_verification_or_verification_only_retry"
    )
    assert summary["private_resume_payload_exposed"] is False
    assert summary["verifier_entry_manifest"] is False
    assert summary["final_kernel_handoff"] is False
    assert summary["actual_target_runtime_executed"] is False
    assert summary["reports_posterior_convergence"] is False
    assert summary["reports_sampler_superiority"] is False
    assert summary["reports_default_readiness"] is False
    assert summary["reports_gpu_or_xla_readiness"] is False
    text = json.dumps(summary, sort_keys=True)
    for forbidden in (
        "private/",
        "private_checkpoints",
        "_manifest.json",
        ".tfs",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "private_handoff_state",
        "private_handoff_payload",
    ):
        assert forbidden not in text


def test_phase7_resume_split_contract_rejects_missing_private_handoff() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    attempt = hmc_kernel_tuning.HMCTuneVerifyRepairAttempt(
        attempt_index=0,
        budget_policy_payload=_tiny_budget_factory(2, 0).payload(),
        incoming_state_payload=None,
        windowed_stage=None,
        fixed_mass_step_stage=None,
        frozen_step_trajectory_stage=None,
        verification_config_payload=None,
        verification_diagnostics={"not_run": True},
        verification_callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        final_status="architecture_blocked",
        diagnostic_role="architecture_blocked",
        hard_vetoes=(),
        repair_triggers=("phase7_required_private_handoff_missing",),
        handoff_state_payload=None,
    )
    result = HMCTuneVerifyRepairLoopResult(
        config=_loop_config(max_attempts=1),
        geometry_artifact_hash=geometry.artifact_hash,
        bootstrap_artifact_hash=bootstrap.artifact_hash,
        adapter_signature=geometry.adapter_signature,
        target_dimension=geometry.target_dimension,
        attempts=(attempt,),
        final_status="architecture_blocked",
        diagnostic_role="architecture_blocked",
        hard_vetoes=(),
        repair_triggers=("phase7_required_private_handoff_missing",),
        final_kernel_payload=None,
        final_kernel_hash=None,
        seed_report={"phase7_root_seed": (20260621, 70)},
        diagnostic_roles={"architecture_blocked": "test"},
    )

    assert result.final_status == "architecture_blocked"
    with pytest.raises(ValueError, match="private handoff state"):
        hmc_kernel_tuning._phase7_private_resume_split_contract(loop=result)


def test_outer_loop_classifies_verification_acceptance_retry_when_public_budget_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0, "verification_count": 0}
    base_run, _calls = _scripted_full_chain_runner(
        verification_acceptances=[0.82, 0.83, 0.70]
    )

    def fake_perf_counter() -> float:
        return float(clock["now"])

    def run(adapter: Any, initial_state: Any, config: Any):
        result = base_run(adapter, initial_state, config)
        acceptance = float(np.asarray(result.diagnostics["acceptance_rate"]))
        if acceptance in {0.82, 0.83}:
            clock["verification_count"] += 1
            clock["now"] = 100.0 if clock["verification_count"] == 1 else 760.0
        return result

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=3,
            public_timeout_budget_s=810.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "verification_acceptance_budget_blocked"
    assert "verification_acceptance_budget_blocked" in result.repair_triggers
    assert len(result.attempts) == 1
    assert result.attempts[-1].attempt_index == 0
    assert result.attempts[-1].handoff_state_payload[
        "verification_acceptance_relation"
    ] == "above_acceptance_band"
    assert result.attempts[-1].handoff_state_payload[
        "verification_repair_applied"
    ] is True
    assert all(
        "phase6_public_timeout_soft_deadline" not in attempt.hard_vetoes
        for attempt in result.attempts
    )
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    assert public_summary["diagnostic_role"] == "verification_acceptance_budget_blocked"
    assert public_summary["attempt_count"] == 1
    guard = public_summary["terminal_budget_guard"]
    assert guard["classification"] == "verification_acceptance_budget_blocked"
    assert guard["previous_verification_acceptance_relation"] == "above_acceptance_band"
    assert guard["closeout_required_before_next_attempt"] is True
    assert guard["hmc_mechanics_exposed"] is False
    assert guard["next_attempt_budget_is_public_policy"] is False
    assert guard["remaining_s"] == pytest.approx(50.0)
    text = json.dumps(public_summary, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_terminal_phase6_repair_slot_does_not_bypass_verification_acceptance_budget_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    retry_stage_calls = {"windowed": 0, "fixed": 0, "trajectory": 0}
    run, _calls = _scripted_full_chain_runner(
        verification_acceptances=[0.70],
        trajectory_acceptance=0.90,
    )

    def fake_perf_counter() -> float:
        return float(clock["now"])

    def trajectory_stage_runner(**kwargs: Any):
        result = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(**kwargs)
        if int(kwargs["_attempt_index"]) == 0:
            clock["now"] = 680.0
        elif int(kwargs["_attempt_index"]) == 1:
            retry_stage_calls["trajectory"] += 1
        return result

    def windowed_stage_runner(**kwargs: Any):
        if int(kwargs["_attempt_index"]) == 1:
            retry_stage_calls["windowed"] += 1
        return hmc_kernel_tuning.run_hmc_windowed_mass_stage(**kwargs)

    def fixed_step_stage_runner(**kwargs: Any):
        if int(kwargs["_attempt_index"]) == 1:
            retry_stage_calls["fixed"] += 1
        return _run_historical_phase5_stage(**kwargs)

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            terminal_phase6_repair_extra_attempts=1,
            public_timeout_budget_s=810.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_stage_runner,
        _fixed_mass_step_stage_runner=fixed_step_stage_runner,
        _frozen_step_trajectory_stage_runner=trajectory_stage_runner,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "verification_acceptance_budget_blocked"
    assert len(result.attempts) == 1
    assert retry_stage_calls == {"windowed": 0, "fixed": 0, "trajectory": 0}
    guard = hmc_kernel_tuning._phase7_public_summary(result)["terminal_budget_guard"]
    assert guard["classification"] == "verification_acceptance_budget_blocked"
    assert (
        guard["verification_repair_trigger"]
        == "phase6_trajectory_acceptance_outside_pass_band"
    )
    assert guard["hmc_mechanics_exposed"] is False


def test_phase7_fixed_mass_stage_config_threads_public_timeout_fields() -> None:
    config = _loop_config(
        chain_execution_mode="tf_function",
        use_xla=True,
        public_timeout_budget_s=810.0,
        public_timeout_started_perf_counter_s=12.5,
    )

    fixed_config = hmc_kernel_tuning._phase7_fixed_step_stage_config(
        config,
        attempt_index=2,
    )
    ladder_config = hmc_kernel_tuning._fixed_mass_step_stage_ladder_config(
        fixed_config,
        initial_step=0.1,
        num_leapfrog_steps=5,
        target_scope="kernel_fixed_mass_step_toy_gaussian",
        attempt_budget_policy=_tiny_budget_factory(2, 2),
    )

    assert fixed_config.public_timeout_budget_s == pytest.approx(810.0)
    assert fixed_config.public_timeout_started_perf_counter_s == pytest.approx(12.5)
    assert ladder_config.public_timeout_budget_s == pytest.approx(810.0)
    assert ladder_config.public_timeout_started_perf_counter_s == pytest.approx(12.5)
    assert ladder_config.chain_execution_mode == "tf_function"
    assert ladder_config.use_xla is True



def test_windowed_mass_public_timeout_closeout_before_runner_skips_hmc_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", lambda: 9.0)
    calls: list[str] = []
    events: list[tuple[str, Mapping[str, Any]]] = []

    def run(_adapter: Any, _initial_state: Any, _config: Any):
        calls.append("unexpected")
        raise AssertionError("windowed mass HMC runner must not run after closeout")

    result = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=hmc_kernel_tuning.HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            public_timeout_budget_s=10.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=run,
        _attempt_budget_policy=_tiny_budget_factory(2, 0),
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
        _attempt_index=0,
    )

    assert calls == []
    assert result.passed is False
    assert result.final_status == "hard_veto"
    assert result.diagnostic_role == "hard_veto"
    assert result.hard_vetoes == ("windowed_mass_public_timeout_soft_deadline",)
    closeout = result.diagnostics["public_timeout_closeout"]
    assert closeout["remaining_s"] == pytest.approx(1.0)
    assert closeout["closeout_required_before_hmc_call"] is True
    assert closeout["deadline_clock_scope"] == "public_one_call_global"
    assert closeout["hmc_mechanics_exposed"] is False
    assert [stage for stage, _payload in events] == [
        "windowed_mass_public_timeout_closeout"
    ]
    event_payload = events[0][1]
    assert event_payload["public_timeout_closeout"]["hard_veto"] == (
        "windowed_mass_public_timeout_soft_deadline"
    )
    assert event_payload["hmc_mechanics_exposed"] is False
    forbidden_fields = {
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    }
    assert forbidden_fields.isdisjoint(event_payload)
    assert forbidden_fields.isdisjoint(event_payload["public_timeout_closeout"])
    public_text = json.dumps(event_payload, sort_keys=True)
    for forbidden in (
        "mass_artifact_payload",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in public_text

def test_outer_loop_blocks_verification_acceptance_retry_before_stage_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0, "verification_count": 0}
    retry_stage_calls = {"windowed": 0, "fixed": 0, "trajectory": 0}
    base_run, _calls = _scripted_full_chain_runner(
        verification_acceptances=[0.82, 0.83, 0.70]
    )

    def fake_perf_counter() -> float:
        return float(clock["now"])

    def run(adapter: Any, initial_state: Any, config: Any):
        result = base_run(adapter, initial_state, config)
        acceptance = float(np.asarray(result.diagnostics["acceptance_rate"]))
        if acceptance in {0.82, 0.83}:
            clock["verification_count"] += 1
            # At t=680 the Phase 4v guard's old reserve+first-candidate check
            # still looked affordable for an 810s public budget. Attempt 2
            # then paid retry-stage overhead and reached Phase 6 with too
            # little remaining budget for candidate 0.
            clock["now"] = 100.0 if clock["verification_count"] == 1 else 680.0
        return result

    def windowed_stage_runner(**kwargs: Any):
        result = hmc_kernel_tuning.run_hmc_windowed_mass_stage(**kwargs)
        if int(kwargs["_attempt_index"]) == 2:
            retry_stage_calls["windowed"] += 1
            clock["now"] += 35.0
        return result

    def fixed_step_stage_runner(**kwargs: Any):
        result = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(**kwargs)
        if int(kwargs["_attempt_index"]) == 2:
            retry_stage_calls["fixed"] += 1
            clock["now"] += 45.0
        return result

    def trajectory_stage_runner(**kwargs: Any):
        if int(kwargs["_attempt_index"]) == 2:
            retry_stage_calls["trajectory"] += 1
        return hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(**kwargs)

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=3,
            public_timeout_budget_s=810.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_stage_runner,
        _fixed_mass_step_stage_runner=fixed_step_stage_runner,
        _frozen_step_trajectory_stage_runner=trajectory_stage_runner,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "verification_acceptance_budget_blocked"
    assert len(result.attempts) == 1
    assert retry_stage_calls == {"windowed": 0, "fixed": 0, "trajectory": 0}
    assert all(
        "phase6_public_timeout_soft_deadline" not in attempt.hard_vetoes
        for attempt in result.attempts
    )
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    guard = public_summary["terminal_budget_guard"]
    assert guard["classification"] == "verification_acceptance_budget_blocked"
    assert guard["remaining_s"] == pytest.approx(130.0)
    assert guard["reserve_s"] == pytest.approx(60.0)
    assert guard["estimated_next_candidate_s"] == pytest.approx(60.0)
    assert guard["estimated_pre_phase6_retry_overhead_s"] == pytest.approx(60.0)
    assert guard["estimated_minimum_next_attempt_s"] == pytest.approx(120.0)
    assert guard["hmc_mechanics_exposed"] is False
    text = json.dumps(guard, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_outer_loop_blocks_phase6_repair_retry_before_stage_overhead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    retry_stage_calls = {"windowed": 0, "fixed": 0, "trajectory": 0}
    run, _calls = _scripted_full_chain_runner(
        verification_acceptances=[0.70],
        trajectory_acceptance=0.90,
    )

    def fake_perf_counter() -> float:
        return float(clock["now"])

    def trajectory_stage_runner(**kwargs: Any):
        result = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(**kwargs)
        if int(kwargs["_attempt_index"]) == 0:
            clock["now"] = 680.0
        elif int(kwargs["_attempt_index"]) == 1:
            retry_stage_calls["trajectory"] += 1
        return result

    def windowed_stage_runner(**kwargs: Any):
        if int(kwargs["_attempt_index"]) == 1:
            retry_stage_calls["windowed"] += 1
        return hmc_kernel_tuning.run_hmc_windowed_mass_stage(**kwargs)

    def fixed_step_stage_runner(**kwargs: Any):
        if int(kwargs["_attempt_index"]) == 1:
            retry_stage_calls["fixed"] += 1
        return _run_historical_phase5_stage(**kwargs)

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=3,
            public_timeout_budget_s=810.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_stage_runner,
        _fixed_mass_step_stage_runner=fixed_step_stage_runner,
        _frozen_step_trajectory_stage_runner=trajectory_stage_runner,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "verification_acceptance_budget_blocked"
    assert "verification_acceptance_budget_blocked" in result.repair_triggers
    assert len(result.attempts) == 1
    assert result.attempts[0].handoff_state_payload[
        "verification_repair_trigger"
    ] == "phase6_trajectory_acceptance_outside_pass_band"
    assert result.attempts[0].handoff_state_payload[
        "verification_acceptance_relation"
    ] == "above_acceptance_band"
    assert result.attempts[0].handoff_state_payload[
        "verification_repair_applied"
    ] is True
    assert retry_stage_calls == {"windowed": 0, "fixed": 0, "trajectory": 0}
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    guard = public_summary["terminal_budget_guard"]
    assert guard["classification"] == "verification_acceptance_budget_blocked"
    assert (
        guard["verification_repair_trigger"]
        == "phase6_trajectory_acceptance_outside_pass_band"
    )
    assert guard["previous_verification_acceptance_relation"] == (
        "above_acceptance_band"
    )
    assert guard["closeout_required_before_next_attempt"] is True
    assert guard["hmc_mechanics_exposed"] is False
    assert guard["remaining_s"] == pytest.approx(130.0)
    text = json.dumps(public_summary, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_outer_loop_phase5_acceptance_repair_uses_private_step_handoff() -> None:
    run, calls = _scripted_full_chain_runner(
        phase5_screen_acceptances=[0.90, 0.91, 0.92, 0.70],
        verification_acceptances=[0.70] * 20,
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "passed"
    assert len(result.attempts) == 1
    first = result.attempts[0]
    assert first.final_status == "passed"
    assert first.fixed_mass_step_stage.selected_step_payload is not None
    assert first.fixed_mass_step_stage.passed is True
    assert first.fixed_mass_step_stage.diagnostics["algorithm"] == (
        "joint_l_epsilon_grid_fixed_mass_hmc"
    )
    assert first.fixed_mass_step_stage.repair_step_payload is None
    assert first.frozen_step_trajectory_stage is None
    assert first.handoff_state_payload["handoff_stage"] == "phase7_direct"
    assert first.verification_config_payload is not None
    dual_averaging_calls = [call for call in calls if call["uses_dual_averaging"]]
    assert len(dual_averaging_calls) > 1


def test_phase6_high_acceptance_handoff_supplies_private_repair_step() -> None:
    config = _loop_config(
        acceptance_band=(0.65, 0.75),
        repair_band=(0.55, 0.85),
        trajectory_window_lower_multiplier=0.01,
        trajectory_window_upper_multiplier=100.0,
    )
    geometry, bootstrap, windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    fixed = _historical_phase5_stage(fixed)
    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        run_full_chain=lambda _adapter, _state, run_config: _fake_result(
            num_results=int(run_config.num_results),
            acceptance=0.90,
            samples=np.zeros((int(run_config.num_results), 2)),
        ),
        _attempt_budget_policy=_tiny_budget_factory(2, 0),
    )
    repair = _phase6_trajectory_repair_handoff_payload(
        config=config,
        selected_step_size=fixed.selected_step_size,
        selected_step_hash=fixed.selected_step_hash,
        frozen_step_trajectory_stage=trajectory,
    )
    state = _HMCPhaseAttemptState(
        mass_artifact_payload={"dimension": 2},
        mass_artifact_signature="mass-signature",
        selected_step_size=fixed.selected_step_size,
        selected_step_hash=fixed.selected_step_hash,
        handoff_stage="phase5_selected",
        **repair,
    )

    assert trajectory.final_status == "repair_or_retry"
    assert {
        candidate["trajectory_window_relation"]
        for candidate in trajectory.candidate_results
    } == {"inside_trajectory_window"}
    assert repair["verification_acceptance_relation"] == "above_acceptance_band"
    assert (
        repair["verification_repair_trigger"]
        == "phase6_trajectory_acceptance_outside_pass_band"
    )
    assert (
        repair["verification_repair_source"]
        == "phase6_frozen_step_trajectory_acceptance"
    )
    assert repair["verification_repair_step_size"] == pytest.approx(
        2.0 * fixed.selected_step_size
    )
    assert repair["verification_repair_step_hash"] is not None
    assert repair["verification_repair_applied"] is True
    assert _fixed_mass_step_initial_step(_windowed_stage(), attempt_state=state) == (
        pytest.approx(repair["verification_repair_step_size"])
    )


def test_phase6_low_acceptance_handoff_supplies_private_repair_step() -> None:
    config = _loop_config(
        acceptance_band=(0.65, 0.75),
        repair_band=(0.55, 0.85),
        trajectory_window_lower_multiplier=0.01,
        trajectory_window_upper_multiplier=100.0,
    )

    geometry, bootstrap, windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    fixed = _historical_phase5_stage(fixed)
    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        run_full_chain=lambda _adapter, _state, run_config: _fake_result(
            num_results=int(run_config.num_results),
            acceptance=0.50,
            samples=np.zeros((int(run_config.num_results), 2)),
        ),
        _attempt_budget_policy=_tiny_budget_factory(2, 0),
    )
    repair = _phase6_trajectory_repair_handoff_payload(
        config=config,
        selected_step_size=fixed.selected_step_size,
        selected_step_hash=fixed.selected_step_hash,
        frozen_step_trajectory_stage=trajectory,
    )

    assert trajectory.final_status == "repair_or_retry"
    assert {
        candidate["trajectory_window_relation"]
        for candidate in trajectory.candidate_results
    } == {"inside_trajectory_window"}
    assert repair["verification_acceptance_relation"] == "below_acceptance_band"
    assert (
        repair["verification_repair_trigger"]
        == "phase6_trajectory_acceptance_outside_pass_band"
    )
    assert repair["verification_repair_step_size"] == pytest.approx(
        0.5 * fixed.selected_step_size
    )
    assert repair["verification_repair_applied"] is True


def test_outer_loop_phase6_repair_uses_directional_private_step_handoff() -> None:
    run, calls = _scripted_full_chain_runner(
        verification_acceptances=[0.70],
        trajectory_acceptance=0.90,
    )
    config = _loop_config(
        max_attempts=2,
        trajectory_window_lower_multiplier=0.01,
        trajectory_window_upper_multiplier=100.0,
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=config,
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
        _fixed_mass_step_stage_runner=_run_historical_phase5_stage,
    )

    assert result.final_status == "budget_exhausted"
    assert result.final_kernel_payload is None
    assert len(result.attempts) == 2
    first = result.attempts[0]
    assert first.final_status == "repair_or_retry"
    assert first.fixed_mass_step_stage.passed is True
    assert first.frozen_step_trajectory_stage.passed is False
    assert first.frozen_step_trajectory_stage.selected_trajectory_hash is None
    assert {
        candidate["trajectory_window_relation"]
        for candidate in first.frozen_step_trajectory_stage.candidate_results
    } == {"inside_trajectory_window"}
    assert "phase6_trajectory_status:repair_or_retry" in first.repair_triggers
    assert first.handoff_state_payload["handoff_stage"] == "phase5_selected"
    assert first.handoff_state_payload["step_handoff_complete"] is True
    assert first.handoff_state_payload["stage_repair_handoff_complete"] is True
    assert first.handoff_state_payload["final_kernel_handoff_complete"] is False
    assert (
        first.handoff_state_payload["verification_repair_trigger"]
        == "phase6_trajectory_acceptance_outside_pass_band"
    )
    assert (
        first.handoff_state_payload["verification_repair_source"]
        == "phase6_frozen_step_trajectory_acceptance"
    )
    assert first.handoff_state_payload["verification_acceptance_relation"] == (
        "above_acceptance_band"
    )
    assert first.handoff_state_payload["verification_repair_applied"] is True
    assert first.handoff_state_payload["verification_repair_step_size"] == pytest.approx(
        2.0 * first.fixed_mass_step_stage.selected_step_size
    )
    bracket_state = first.handoff_state_payload["fixed_mass_bracket_state"]
    assert bracket_state["next_step_size"] == pytest.approx(
        first.handoff_state_payload["verification_repair_step_size"]
    )
    assert bracket_state["high_acceptance_step_lower_bound"] == pytest.approx(
        first.handoff_state_payload["verification_repair_step_size"]
    )
    assert bracket_state["private_handoff_only"] is True
    assert bracket_state["public_progress_exposes_step"] is False
    assert first.handoff_state_payload["selected_num_leapfrog_steps"] is None
    assert first.handoff_state_payload["phase6_retry_num_leapfrog_steps"] is not None
    assert first.handoff_state_payload["phase6_retry_anchor_source"] == (
        "phase6_failed_candidate_nearest_tau"
    )
    assert result.attempts[1].incoming_state_payload["handoff_stage"] == "phase5_selected"
    assert result.attempts[1].incoming_state_payload["selected_step_hash"] is not None
    assert result.attempts[1].incoming_state_payload["phase6_retry_num_leapfrog_steps"] == (
        first.handoff_state_payload["phase6_retry_num_leapfrog_steps"]
    )
    assert result.attempts[1].incoming_state_payload["phase6_retry_anchor_source"] == (
        "phase6_failed_candidate_nearest_tau"
    )
    assert result.attempts[1].incoming_state_payload[
        "verification_repair_applied"
    ] is True
    assert result.attempts[1].incoming_state_payload[
        "fixed_mass_bracket_state_available"
    ] is True
    assert result.attempts[1].fixed_mass_step_stage.initial_step_size == pytest.approx(
        first.handoff_state_payload["verification_repair_step_size"]
    )
    assert result.attempts[1].budget_policy_payload["budget"] == 16
    retry_step = first.handoff_state_payload["verification_repair_step_size"]
    retry_fixed_mass_calls = [
        call
        for call in calls
        if abs(call["step_size"] - retry_step) < 1.0e-12
        and call["num_burnin_steps"] == 1
    ]
    assert retry_fixed_mass_calls
    assert retry_fixed_mass_calls[0]["uses_dual_averaging"] is False


def test_outer_loop_budget_exhausted_emits_no_final_kernel() -> None:
    run, _calls = _scripted_full_chain_runner(
        verification_acceptances=[0.82, 0.83, 0.82, 0.83]
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "budget_exhausted"
    assert result.passed is False
    assert result.final_kernel_payload is None
    assert result.final_kernel_hash is None
    assert result.payload()["budget_exhausted_is_non_promoting"] is True
    assert "phase7_budget_exhausted" in result.repair_triggers


def test_operational_outer_loop_accepts_fallback_and_applies_reserved_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()
    windowed = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=hmc_kernel_tuning.HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            seed=(20260711, 640),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        _attempt_budget_policy=_operational_budget(),
    )
    assert windowed.operational_warmup_result is not None

    selector_calls: list[Mapping[str, Any]] = []
    exact_l_retune_count = 0

    def operational_runner(_adapter: Any, initial_state: Any, config: Any):
        nonlocal exact_l_retune_count
        bank = np.asarray(initial_state, dtype=float)
        selector_calls.append(
            {
                "bank": bank.copy(),
                "seed": tuple(config.seed),
                "leapfrog": int(config.num_leapfrog_steps),
                "uses_dual_averaging": bool(
                    config.tuning_policy.uses_dual_averaging
                ),
            }
        )
        if config.tuning_policy.uses_dual_averaging:
            exact_l_retune_count += 1
            result = _operational_exact_l_retune_result(
                num_results=int(config.num_results),
                step_size=0.125,
            )
            if exact_l_retune_count == 1:
                samples = np.asarray(result.samples, dtype=float).copy()
                samples[0, 0, 0] = np.nan
                return replace(result, samples=samples)
            return result
        draw = np.arange(int(config.num_results), dtype=float)[:, None, None]
        samples = draw + bank[None, :, :]
        probability = np.full((int(config.num_results), 4), 0.70)
        result = _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=samples,
        )
        return replace(
            result,
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            },
        )

    verification_inputs = []

    def scripted_verification_core(*, verification_input: Any, **_kwargs: Any):
        verification_inputs.append(verification_input)
        repaired = verification_input.seed_policy == (
            "bayesfilter.phase7_operational_repair_verification_seed.v2"
        )
        diagnostics = _sequential_verification_diagnostics(
            0.70 if repaired else 0.90,
            draw_count=64,
        )
        return hmc_kernel_tuning._HMCPhase7FixedKernelVerificationOutcome(
            verification_input=verification_input,
            verification_config_payload={
                "verification_policy": "dependence_aware_acceptance_v2",
                "max_results": 64,
                "num_results": 64,
            },
            diagnostics=diagnostics,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
            final_status="passed" if repaired else "repair_or_retry",
            diagnostic_role=(
                "dependence_aware_fixed_kernel_verification_passed"
                if repaired
                else "verification_acceptance_repair_trigger"
            ),
            hard_vetoes=(),
            continuation_scope="passed" if repaired else "repair_or_retry",
            repair_triggers=(
                ()
                if repaired
                else ("verification_acceptance_outside_pass_band",)
            ),
            repair_evidence=None,
        )

    def forbidden(*_args: Any, **_kwargs: Any):
        raise AssertionError("historical Phase 5/6 route was called")

    windowed_calls = 0

    def windowed_runner(**_kwargs: Any):
        nonlocal windowed_calls
        windowed_calls += 1
        return windowed

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_full_chain_tfp_hmc",
        operational_runner,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_phase7_fixed_kernel_verification",
        scripted_verification_core,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_fixed_mass_hmc_tuning_budget_ladder",
        forbidden,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_fixed_mass_step_initial_state_factory",
        forbidden,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_phase7_direct_candidate_queue",
        forbidden,
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=HMCTuneVerifyRepairLoopConfig(
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
            repair_band=(0.55, 0.85),
            max_attempts=2,
            seed=(20260711, 700),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        run_full_chain=operational_runner,
        _budget_policy_factory=lambda _dimension, attempt: _operational_budget(
            attempt
        ),
        _windowed_stage_runner=windowed_runner,
        _frozen_step_trajectory_stage_runner=forbidden,
    )

    assert result.final_status == "passed"
    assert windowed_calls == 1
    assert len(selector_calls) == 11
    assert all(
        np.array_equal(call["bank"], selector_calls[0]["bank"])
        for call in selector_calls
    )
    assert all(not call["uses_dual_averaging"] for call in selector_calls[:9])
    assert all(call["uses_dual_averaging"] for call in selector_calls[-2:])
    assert tuple(call["leapfrog"] for call in selector_calls[-2:]) == (4, 2)
    assert len({call["seed"] for call in selector_calls}) == 11
    assert len(verification_inputs) == 2
    first_input, repaired_input = verification_inputs
    assert first_input.source_kind == repaired_input.source_kind == (
        "operational_selection_v2"
    )
    assert first_input.coordinate_signature == repaired_input.coordinate_signature
    assert first_input.metric_signature == repaired_input.metric_signature
    assert first_input.trajectory_signature == repaired_input.trajectory_signature
    assert first_input.start_bank_signature == repaired_input.start_bank_signature
    assert first_input.num_leapfrog_steps == repaired_input.num_leapfrog_steps
    assert first_input.num_leapfrog_steps == 2
    assert repaired_input.step_size == pytest.approx(2.0 * first_input.step_size)
    assert first_input.verification_seed != repaired_input.verification_seed

    fixed = result.attempts[0].fixed_mass_step_stage
    selection = fixed._operational_selection
    assert selection is not None
    assert selection.disposition == "representative_selected"
    assert selection.representative.candidate.num_leapfrog_steps == 2
    assert len(selection.candidate_retune_failures) == 1
    assert selection.candidate_retune_failures[0].nomination_ordinal == 0
    summary = fixed.payload()["operational_selection_summary"]
    assert summary["schema"] == "bayesfilter.hmc_fixed_trajectory_selection_summary.v4"
    assert summary["candidate_retune_failure_count"] == 1
    assert "operational_candidate_retune_failed" not in fixed.repair_triggers

    first_state = result.attempts[0].handoff_state_payload
    assert first_state["verification_repair_applied"] is True
    assert first_state["verification_repair_direction"] == "higher_epsilon"
    assert first_state["repair_direction_history"] == ("higher_epsilon",)
    assert first_state["repair_verification_reserved"] is True
    assert first_state["fixed_mass_bracket_state"]["bracket_role"] == (
        "local_directional_trust_region_not_empirical_acceptance_bracket"
    )
    second = result.attempts[1]
    assert second.windowed_stage is windowed
    assert second.fixed_mass_step_stage is result.attempts[0].fixed_mass_step_stage
    assert second.frozen_step_trajectory_stage is None
    assert second.verification_diagnostics[
        "phase7_operational_repair_verification"
    ] is True
    reports = result.seed_report["direct_candidate_verification_seed_maps"]
    assert tuple(report["verification_seed_domain"] for report in reports) == (
        "independent_final_verification",
        "repair_verification",
    )
    assert result.final_kernel_hash is not None


def test_operational_phase5_selection_repairs_through_empirical_midpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()
    windowed = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=hmc_kernel_tuning.HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            seed=(20260711, 641),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        _attempt_budget_policy=_operational_budget(),
    )
    screen_steps: list[float] = []

    def operational_runner(_adapter: Any, initial_state: Any, config: Any):
        bank = np.asarray(initial_state, dtype=float)
        if config.tuning_policy.uses_dual_averaging:
            return _operational_exact_l_retune_result(
                num_results=int(config.num_results),
                step_size=0.125,
            )
        step = float(config.step_size)
        screen_steps.append(step)
        first = screen_steps[0]
        probability = (
            0.90
            if np.isclose(step, first, rtol=1.0e-12, atol=0.0)
            else (
                0.40
                if np.isclose(step, 2.0 * first, rtol=1.0e-12, atol=0.0)
                else 0.70
            )
        )
        draw = np.arange(int(config.num_results), dtype=float)[:, None, None]
        samples = draw + bank[None, :, :]
        probabilities = np.full((int(config.num_results), 4), probability)
        return replace(
            _fake_result(
                num_results=int(config.num_results),
                acceptance=probability,
                samples=samples,
            ),
            trace={
                "log_accept_ratio": np.log(probabilities),
                "is_accepted": np.ones_like(probabilities, dtype=bool),
                "target_log_prob": np.zeros_like(probabilities),
            },
            diagnostics={
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            },
        )

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_full_chain_tfp_hmc",
        operational_runner,
    )
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260711, 642),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        run_full_chain=operational_runner,
        _attempt_budget_policy=_operational_budget(),
        _repair_verification_reserved=True,
        _selection_max_attempts=5,
    )

    assert fixed.passed is True
    assert len(screen_steps) == 27
    first = screen_steps[0]
    assert screen_steps[:9] == pytest.approx([first] * 9)
    assert screen_steps[9:18] == pytest.approx([2.0 * first] * 9)
    assert screen_steps[18:] == pytest.approx(
        [np.sqrt(first * 2.0 * first)] * 9
    )
    loop = fixed._operational_selection_loop
    assert loop is not None
    assert loop.repair_direction_history == (
        "higher_epsilon",
        "lower_epsilon",
    )
    assert loop.final_bracket == pytest.approx((first, 2.0 * first))
    summary = fixed.payload()["operational_selection_repair_summary"]
    assert summary["attempt_count"] == 3
    assert summary["repair_count"] == 2
    assert summary["empirical_bracket_available"] is True
    assert summary["selection_repair_loop_exercised"] is True
    assert summary["repair_loop_validated"] is False
    operational = windowed.operational_warmup_result
    assert operational is not None
    (
        _adapted_mass,
        _mass_signature,
        phase4_adapter,
        verification_adapter,
        verification_signature,
    ) = hmc_kernel_tuning._phase7_verification_runtime_context(
        adapter=adapter,
        geometry=geometry,
        windowed_stage=windowed,
        target_scope="kernel_windowed_mass_toy_gaussian",
    )
    _active_bank, active_bank_summary = (
        hmc_kernel_tuning._phase7_verification_initial_state(
            windowed_stage=windowed,
            phase4_adapter=phase4_adapter,
            verification_adapter=verification_adapter,
            verification_hmc_signature=verification_signature,
        )
    )
    active_signature = active_bank_summary["active_signature"]
    assert fixed.diagnostics["frozen_start_bank_signature"] == active_signature
    assert fixed.diagnostics["frozen_start_bank_source_signature"] == (
        operational.private_start_bank_signature
    )
    assert active_bank_summary["source_signature"] == (
        operational.private_start_bank_signature
    )
    assert active_signature != operational.private_start_bank_signature
    assert fixed.selected_step_payload["start_bank_signature"] == active_signature
    assert all(
        result.candidate.start_bank_signature == active_signature
        for attempt in loop.attempts
        for result in attempt.selection.candidate_results
    )
    assert summary["raw_step_history_exposed"] is False
    assert "repaired_step_history" not in summary


def test_operational_exact_l_candidate_failure_is_budget_exhausted_not_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()
    windowed = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=hmc_kernel_tuning.HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            seed=(20260711, 643),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        _attempt_budget_policy=_operational_budget(),
    )

    def operational_runner(_adapter: Any, initial_state: Any, config: Any):
        if config.tuning_policy.uses_dual_averaging:
            result = _operational_exact_l_retune_result(
                num_results=int(config.num_results),
                step_size=0.125,
            )
            samples = np.asarray(result.samples, dtype=float).copy()
            samples[0, 0, 0] = np.nan
            return replace(result, samples=samples)
        bank = np.asarray(initial_state, dtype=float)
        draw = np.arange(int(config.num_results), dtype=float)[:, None, None]
        samples = draw + bank[None, :, :]
        probability = np.full((int(config.num_results), 4), 0.70)
        return replace(
            _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                samples=samples,
            ),
            trace={
                "log_accept_ratio": np.log(probability),
                "is_accepted": np.ones_like(probability, dtype=bool),
                "target_log_prob": np.zeros_like(probability),
            },
            diagnostics={
                "native_divergence_status": "not_exposed_by_kernel",
                "divergence_count": None,
            },
        )

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_full_chain_tfp_hmc",
        operational_runner,
    )
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260711, 644),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        run_full_chain=operational_runner,
        _attempt_budget_policy=_operational_budget(),
        _selection_max_attempts=1,
    )

    assert fixed.final_status == "budget_exhausted"
    assert fixed.diagnostic_role == "candidate_retune_failed_non_promoting"
    assert fixed.hard_vetoes == ()
    assert fixed.repair_triggers == ("operational_candidate_retune_failed",)
    assert fixed.selected_step_payload is None
    summary = fixed.payload()["operational_selection_summary"]
    assert summary["schema"] == "bayesfilter.hmc_fixed_trajectory_selection_summary.v4"
    assert summary["disposition"] == "candidate_retune_failed"
    assert summary["retune_failure_scope"] == "candidate_data_invalid"
    assert summary["retune_failure_reasons"] == ("nonfinite_candidate_state",)
    assert summary["retune_candidate_signature"] is not None
    assert summary["representative_signature"] is None
    assert summary["candidate_retune_failure_count"] == 3
    failures = summary["candidate_retune_failures"]
    candidates_by_signature = {
        item.candidate.signature: item.candidate.num_leapfrog_steps
        for item in fixed._operational_selection.candidate_results
    }
    assert tuple(
        candidates_by_signature[item["candidate_signature"]]
        for item in failures
    ) == (4, 2, 8)
    assert tuple(item["nomination_ordinal"] for item in failures) == (0, 1, 2)
    assert len({tuple(item["seed"]) for item in failures}) == 3

    phase6_calls: list[str] = []
    result = run_hmc_tune_verify_repair_loop(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=HMCTuneVerifyRepairLoopConfig(
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
            repair_band=(0.55, 0.85),
            max_attempts=1,
            seed=(20260711, 645),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
        ),
        run_full_chain=operational_runner,
        _budget_policy_factory=lambda _dimension, attempt: _operational_budget(
            attempt
        ),
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: phase6_calls.append(
            "unexpected"
        ),
    )

    assert result.final_status == "budget_exhausted"
    assert result.final_kernel_payload is None
    assert phase6_calls == []
    attempt = result.attempts[0]
    assert attempt.final_status == "budget_exhausted"
    assert "phase7_runtime_error" not in attempt.hard_vetoes
    assert attempt.verification_diagnostics["phase7_direct_candidate_queue"][
        "started_count"
    ] == 0


def test_outer_loop_propagates_fixed_mass_budget_incomplete_without_final_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windowed = _windowed_stage()
    clock = {"now": 0.0}
    timed_screen_l: set[int] = set()
    fixed_run, _calls = _scripted_step_runner(
        {3: 0.82, 4: 0.82, 5: 0.82, 6: 0.82, 8: 0.70}
    )

    def fake_perf_counter() -> float:
        return float(clock["now"])

    def timed_fixed_run(adapter: Any, initial_state: Any, config: Any):
        result = fixed_run(adapter, initial_state, config)
        leapfrog = int(config.num_leapfrog_steps)
        if (
            not bool(config.tuning_policy.uses_dual_averaging)
            and leapfrog not in timed_screen_l
        ):
            timed_screen_l.add(leapfrog)
            clock["now"] += 12.0
        return result

    def windowed_runner(**_kwargs: Any):
        return windowed

    def fixed_runner(**kwargs: Any):
        return hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
            **{
                **kwargs,
                "run_full_chain": timed_fixed_run,
            }
        )

    def forbidden_trajectory_runner(**_kwargs: Any):
        raise AssertionError("trajectory stage must not run after fixed-mass budget closeout")

    def forbidden_verification_runner(**_kwargs: Any):
        raise AssertionError("verification must not run after fixed-mass budget closeout")

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=3,
            public_timeout_budget_s=130.0,
            public_timeout_started_perf_counter_s=0.0,
            max_leapfrog_steps=12,
        ),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_runner,
        _fixed_mass_step_stage_runner=fixed_runner,
        _frozen_step_trajectory_stage_runner=forbidden_trajectory_runner,
        _phase7_final_verification_runner=forbidden_verification_runner,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_BUDGET_INCOMPLETE_ROLE
    )
    assert result.hard_vetoes == ()
    assert result.final_kernel_payload is None
    assert result.final_kernel_hash is None
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.final_status == "budget_exhausted"
    assert attempt.frozen_step_trajectory_stage is None
    assert attempt.verification_diagnostics["not_run"] is True
    assert attempt.hard_vetoes == ()
    assert (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_BUDGET_INCOMPLETE_REPAIR_TRIGGER
        in attempt.repair_triggers
    )
    fixed_stage = attempt.fixed_mass_step_stage
    assert fixed_stage.final_status == "budget_exhausted"
    assert fixed_stage.selected_step_payload is None
    assert fixed_stage.repair_step_payload is None
    assert fixed_stage.diagnostics["public_timeout_budget_incomplete"] is True
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    fixed_summary = public_summary["attempt_summaries"][0]["stage_statuses"][
        "fixed_mass_step"
    ]
    assert fixed_summary["final_status"] == "budget_exhausted"
    assert fixed_summary["public_timeout_closeout"]["budget_incomplete"] is True
    assert "hard_veto" not in fixed_summary["public_timeout_closeout"]


def test_outer_loop_rhat_cap_does_not_block_healthy_tuning_handoff() -> None:
    windowed = _windowed_stage()
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=_scripted_full_chain_runner(verification_acceptances=[0.70])[0],
    )
    fixed = _historical_phase5_stage(fixed)
    def trajectory_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=trajectory_run,
    )
    verification_calls: list[int] = []
    stage_calls: list[str] = []

    def windowed_runner(**_kwargs: Any):
        stage_calls.append("windowed")
        return windowed

    def fixed_runner(**_kwargs: Any):
        stage_calls.append("fixed")
        return fixed

    def trajectory_runner(**_kwargs: Any):
        stage_calls.append("trajectory")
        return trajectory

    def verification_runner(
        *,
        budget_policy: Any,
        attempt_index: int,
        verification_start_callback: Any,
        **_kwargs: Any,
    ):
        verification_calls.append(int(attempt_index))
        if verification_start_callback is not None:
            verification_start_callback()
        diagnostics = _sequential_verification_diagnostics(
            0.70,
            draw_count=int(budget_policy.verification_num_results),
            rhat_passed=False,
        )
        return hmc_kernel_tuning._classify_phase7_final_verification(
            _loop_config(max_attempts=2),
            diagnostics=diagnostics,
            screen_error=None,
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
        )

    def verification_wrapper(**kwargs: Any):
        (
            status,
            role,
            hard_vetoes,
            repair_triggers,
        ) = verification_runner(**kwargs)
        budget_policy = kwargs["budget_policy"]
        diagnostics = _sequential_verification_diagnostics(
            0.70,
            draw_count=int(budget_policy.verification_num_results),
            rhat_passed=False,
        )
        return (
            {
                "verification_policy": "sequential_rhat",
                "max_results": int(budget_policy.verification_num_results),
                "acceptance_band": (0.65, 0.75),
                "rhat_threshold_role": "historical_explanatory_only_not_stopping_or_admission",
            },
            diagnostics,
            FixedMassHMCTuningBudgetCallbackResult(),
            status,
            role,
            hard_vetoes,
            repair_triggers,
        )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        _budget_policy_factory=_verification_budget_factory,
        _windowed_stage_runner=windowed_runner,
        _fixed_mass_step_stage_runner=fixed_runner,
        _frozen_step_trajectory_stage_runner=trajectory_runner,
        _phase7_final_verification_runner=verification_wrapper,
    )

    assert result.final_status == "passed"
    assert verification_calls == [0]
    assert stage_calls == ["windowed", "fixed", "trajectory"]
    assert result.attempts[0].windowed_stage is windowed
    assert result.attempts[0].fixed_mass_step_stage is fixed
    assert result.attempts[0].frozen_step_trajectory_stage is trajectory
    assert result.attempts[0].verification_diagnostics["cap_hit"] is False
    assert (
        result.attempts[0].verification_diagnostics[
            "all_finite_rhat_at_or_below_threshold"
        ]
        is False
    )
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    verification = public_summary["attempt_summaries"][0]["stage_statuses"][
        "verification"
    ]
    assert verification["verification_only_retry"] is False
    assert verification["reused_frozen_kernel_handoff"] is False
    assert verification["hmc_mechanics_exposed"] is False
    assert verification["cap_hit"] is False
    assert verification["all_finite_rhat_at_or_below_threshold"] is False
    assert (
        verification["rhat_threshold_role"]
        == "historical_explanatory_only_not_stopping_or_admission"
    )


def test_outer_loop_rhat_cap_with_out_of_band_acceptance_still_reenters_stage_repair() -> None:
    windowed = _windowed_stage()
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=_scripted_full_chain_runner(verification_acceptances=[0.70])[0],
    )
    fixed = _historical_phase5_stage(fixed)

    def trajectory_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=trajectory_run,
    )
    verification_calls: list[int] = []
    stage_calls: list[str] = []

    def capped_budget_factory(_dimension: int, attempt_index: int) -> _HMCAttemptBudgetPolicy:
        base = _tiny_budget_factory(_dimension, attempt_index)
        return replace(
            base,
            budget=128,
            verification_num_results=64,
            verification_num_burnin_steps=2,
            public_budget_cap=128,
            public_max_attempts=3,
            public_diagnostic_preset="diagnostic_plus",
        )

    def windowed_runner(**_kwargs: Any):
        stage_calls.append("windowed")
        return windowed

    def fixed_runner(**_kwargs: Any):
        stage_calls.append("fixed")
        return fixed

    def trajectory_runner(**_kwargs: Any):
        stage_calls.append("trajectory")
        return trajectory

    def verification_wrapper(
        *,
        budget_policy: Any,
        attempt_index: int,
        verification_start_callback: Any,
        **_kwargs: Any,
    ):
        verification_calls.append(int(attempt_index))
        if verification_start_callback is not None:
            verification_start_callback()
        diagnostics = _sequential_verification_diagnostics(
            0.82,
            draw_count=int(budget_policy.verification_num_results),
            rhat_passed=False,
        )
        status, role, hard_vetoes, repair_triggers = (
            hmc_kernel_tuning._classify_phase7_final_verification(
                _loop_config(max_attempts=3),
                diagnostics=diagnostics,
                screen_error=None,
                callback_result=FixedMassHMCTuningBudgetCallbackResult(),
            )
        )
        return (
            {
                "verification_policy": "sequential_rhat",
                "max_results": int(budget_policy.verification_num_results),
                "acceptance_band": (0.65, 0.75),
            },
            diagnostics,
            FixedMassHMCTuningBudgetCallbackResult(),
            status,
            role,
            hard_vetoes,
            repair_triggers,
        )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=3),
        _budget_policy_factory=capped_budget_factory,
        _windowed_stage_runner=windowed_runner,
        _fixed_mass_step_stage_runner=fixed_runner,
        _frozen_step_trajectory_stage_runner=trajectory_runner,
        _phase7_final_verification_runner=verification_wrapper,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "budget_exhausted_non_promoting"
    assert result.repair_triggers == (
        "verification_acceptance_outside_pass_band",
        "phase7_budget_exhausted",
    )
    assert verification_calls == [0, 1, 2]
    assert stage_calls == ["windowed", "fixed", "trajectory"] * 3
    assert result.terminal_budget_guard_payload is None
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    verification = public_summary["attempt_summaries"][0]["stage_statuses"][
        "verification"
    ]
    assert verification["acceptance_relation"] == "above_acceptance_band"
    assert verification["cap_hit"] is False
    assert verification["all_finite_rhat_at_or_below_threshold"] is False


def test_terminal_phase6_repair_slot_no_longer_depends_on_rhat_saturation() -> None:
    windowed = _windowed_stage()
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=_scripted_full_chain_runner(verification_acceptances=[0.70])[0],
    )
    fixed = _historical_phase5_stage(fixed)

    def trajectory_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=trajectory_run,
    )
    verification_calls: list[int] = []
    stage_calls: list[str] = []

    def capped_budget_factory(_dimension: int, attempt_index: int) -> _HMCAttemptBudgetPolicy:
        base = _tiny_budget_factory(_dimension, attempt_index)
        return replace(
            base,
            budget=128,
            verification_num_results=64,
            verification_num_burnin_steps=2,
            public_budget_cap=128,
            public_max_attempts=3,
            public_diagnostic_preset="diagnostic_plus",
        )

    def verification_wrapper(
        *,
        budget_policy: Any,
        attempt_index: int,
        verification_start_callback: Any,
        **_kwargs: Any,
    ):
        verification_calls.append(int(attempt_index))
        if verification_start_callback is not None:
            verification_start_callback()
        diagnostics = _sequential_verification_diagnostics(
            0.70,
            draw_count=int(budget_policy.verification_num_results),
            rhat_passed=False,
        )
        status, role, hard_vetoes, repair_triggers = (
            hmc_kernel_tuning._classify_phase7_final_verification(
                _loop_config(max_attempts=3),
                diagnostics=diagnostics,
                screen_error=None,
                callback_result=FixedMassHMCTuningBudgetCallbackResult(),
            )
        )
        return (
            {
                "verification_policy": "sequential_rhat",
                "max_results": int(budget_policy.verification_num_results),
                "acceptance_band": (0.65, 0.75),
            },
            diagnostics,
            FixedMassHMCTuningBudgetCallbackResult(),
            status,
            role,
            hard_vetoes,
            repair_triggers,
        )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=3, terminal_phase6_repair_extra_attempts=1),
        _budget_policy_factory=capped_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: stage_calls.append("windowed") or windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: stage_calls.append("fixed") or fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: stage_calls.append("trajectory") or trajectory,
        _phase7_final_verification_runner=verification_wrapper,
    )

    assert result.final_status == "passed"
    assert result.diagnostic_role == "fresh_fixed_kernel_verification_passed"
    assert verification_calls == [0]
    assert stage_calls == ["windowed", "fixed", "trajectory"]
    assert result.terminal_budget_guard_payload is None
    assert result.attempts[0].verification_diagnostics["cap_hit"] is False
    assert (
        result.attempts[0].verification_diagnostics[
            "all_finite_rhat_at_or_below_threshold"
        ]
        is False
    )


def test_outer_loop_out_of_band_direct_candidate_advances_before_stage_repair() -> None:
    run, calls = _scripted_full_chain_runner(verification_acceptances=[0.82, 0.70])

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "passed"
    dual_averaging_calls = [call for call in calls if call["uses_dual_averaging"]]
    assert len(dual_averaging_calls) > 2
    assert len(result.attempts) == 1
    queue = result.attempts[0].verification_diagnostics[
        "phase7_direct_candidate_queue"
    ]
    assert queue["started_count"] == 2
    assert queue["candidate_results"][0]["state"] == "repair_or_retry"
    assert queue["candidate_results"][1]["state"] == "passed"
    assert result.attempts[0].handoff_state_payload[
        "verification_repair_trigger"
    ] is None


def test_outer_loop_conflicting_direct_repairs_rerun_without_scalar_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, bootstrap, windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    acceptances = iter((0.82, 0.60, 0.70))
    stage_calls: list[str] = []

    def verifier(
        *,
        candidate_identity: tuple[int, int, str, int],
        attempt_index: int,
        config: Any,
        target_scope: str,
        **_kwargs: Any,
    ):
        verification_input = (
            hmc_kernel_tuning._phase7_direct_candidate_verification_input(
                adapter=_ToyGaussianAdapter(),
                geometry=geometry,
                windowed_stage=windowed,
                fixed_mass_step_stage=fixed,
                config=config,
                attempt_index=attempt_index,
                target_scope=target_scope,
                candidate_identity=candidate_identity,
            )
        )
        acceptance = next(acceptances)
        execution = hmc_kernel_tuning._HMCPhase7FixedKernelVerificationExecution(
            verification_config_payload={"num_results": 64, "num_burnin_steps": 1},
            diagnostics=_sequential_verification_diagnostics(acceptance),
            callback_result=FixedMassHMCTuningBudgetCallbackResult(),
            runner_error=None,
            runner_error_origin=None,
            callback_error=None,
            observed_step_size=verification_input.step_size,
            observed_num_leapfrog_steps=verification_input.num_leapfrog_steps,
        )
        return hmc_kernel_tuning._finalize_phase7_fixed_kernel_verification(
            verification_input=verification_input,
            adapted_mass=hmc_kernel_tuning._phase4_adapted_mass_artifact(windowed),
            config=config,
            execution=execution,
        )

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_phase7_direct_candidate_verification",
        verifier,
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=2),
        _budget_policy_factory=_verification_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: stage_calls.append("windowed")
        or windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: stage_calls.append("fixed")
        or fixed,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("current joint route must bypass Phase 6")
        ),
    )

    assert result.final_status == "passed"
    assert len(result.attempts) == 2
    first, second = result.attempts
    queue = first.verification_diagnostics["phase7_direct_candidate_queue"]
    assert queue["repair_direction_conflict"] is True
    assert first.handoff_state_payload["verification_repair_applied"] is False
    assert first.handoff_state_payload["verification_repair_step_size"] is None
    assert first.handoff_state_payload["verification_repair_direction"] is None
    assert first.handoff_state_payload["verification_repair_disposition"] == (
        "inconclusive_conflict"
    )
    assert first.handoff_state_payload["repair_verification_reserved"] is False
    assert stage_calls == ["windowed", "fixed", "windowed", "fixed"]
    assert second.incoming_state_payload["selected_step_size"] == pytest.approx(
        first.handoff_state_payload["selected_step_size"]
    )
    assert second.verification_diagnostics.get(
        "phase7_operational_repair_verification"
    ) is not True


def test_outer_loop_out_of_band_historical_acceptance_reenters_full_stages() -> None:
    windowed = _windowed_stage()
    fixed = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=_scripted_full_chain_runner(verification_acceptances=[0.70])[0],
    )
    fixed = _historical_phase5_stage(fixed)

    def trajectory_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=trajectory_run,
    )
    verification_acceptances = [0.82, 0.70]
    verification_calls: list[int] = []
    stage_calls: list[str] = []

    def windowed_runner(**_kwargs: Any):
        stage_calls.append("windowed")
        return windowed

    def fixed_runner(**_kwargs: Any):
        stage_calls.append("fixed")
        return fixed

    def trajectory_runner(**_kwargs: Any):
        stage_calls.append("trajectory")
        return trajectory

    def verification_wrapper(
        *,
        budget_policy: Any,
        attempt_index: int,
        verification_start_callback: Any,
        **_kwargs: Any,
    ):
        verification_calls.append(int(attempt_index))
        if verification_start_callback is not None:
            verification_start_callback()
        acceptance = verification_acceptances[len(verification_calls) - 1]
        diagnostics = _sequential_verification_diagnostics(
            acceptance,
            draw_count=max(64, int(budget_policy.verification_num_results)),
            rhat_passed=False,
        )
        status, role, hard_vetoes, repair_triggers = (
            hmc_kernel_tuning._classify_phase7_final_verification(
                _loop_config(max_attempts=3),
                diagnostics=diagnostics,
                screen_error=None,
                callback_result=FixedMassHMCTuningBudgetCallbackResult(),
            )
        )
        return (
            {
                "verification_policy": "sequential_rhat",
                "max_results": int(budget_policy.verification_num_results),
                "acceptance_band": (0.65, 0.75),
            },
            diagnostics,
            FixedMassHMCTuningBudgetCallbackResult(),
            status,
            role,
            hard_vetoes,
            repair_triggers,
        )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=3),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_runner,
        _fixed_mass_step_stage_runner=fixed_runner,
        _frozen_step_trajectory_stage_runner=trajectory_runner,
        _phase7_final_verification_runner=verification_wrapper,
    )

    assert result.final_status == "passed"
    assert verification_calls == [0, 1]
    assert stage_calls == [
        "windowed",
        "fixed",
        "trajectory",
        "windowed",
        "fixed",
        "trajectory",
    ]
    first_attempt, second_attempt = result.attempts
    assert first_attempt.handoff_state_payload[
        "verification_repair_trigger"
    ] == "verification_acceptance_outside_pass_band"
    assert first_attempt.handoff_state_payload["verification_repair_applied"] is True
    assert "phase7_retry_class" not in second_attempt.verification_diagnostics
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    second_verification = public_summary["attempt_summaries"][1]["stage_statuses"][
        "verification"
    ]
    assert second_verification["acceptance_relation"] == "inside_acceptance_band"
    assert second_verification["verification_only_retry"] is False
    assert second_verification["reused_frozen_kernel_handoff"] is False



def test_outer_loop_callback_roles_are_classified_for_verification() -> None:
    def callback(
        _round_payload: Mapping[str, Any],
        _samples: Any,
        _diagnostics: Mapping[str, Any],
    ) -> FixedMassHMCTuningBudgetCallbackResult:
        return FixedMassHMCTuningBudgetCallbackResult(
            promotion_vetoes=("domain_screen_not_yet_passed",),
        )

    run, _calls = _scripted_full_chain_runner(
        verification_acceptances=[0.70, 0.70]
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=1),
        verification_callback=callback,
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "budget_exhausted"
    assert result.attempts[0].final_status == "repair_or_retry"
    assert "domain_screen_not_yet_passed" in result.attempts[0].repair_triggers
    assert result.final_kernel_payload is None


def test_outer_loop_callback_repair_trigger_is_retry_without_final_kernel() -> None:
    def callback(
        _round_payload: Mapping[str, Any],
        _samples: Any,
        _diagnostics: Mapping[str, Any],
    ) -> FixedMassHMCTuningBudgetCallbackResult:
        return FixedMassHMCTuningBudgetCallbackResult(
            repair_triggers=("domain_repair_requested",),
        )

    run, _calls = _scripted_full_chain_runner(
        verification_acceptances=[0.70, 0.70]
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=1),
        verification_callback=callback,
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "budget_exhausted"
    assert result.attempts[0].final_status == "repair_or_retry"
    assert "domain_repair_requested" in result.attempts[0].repair_triggers
    assert result.final_kernel_payload is None


def test_outer_loop_hard_veto_for_continuation_veto_callback() -> None:
    def callback(
        _round_payload: Mapping[str, Any],
        _samples: Any,
        _diagnostics: Mapping[str, Any],
    ) -> FixedMassHMCTuningBudgetCallbackResult:
        return FixedMassHMCTuningBudgetCallbackResult(
            continuation_vetoes=("artifact_cannot_answer_question",),
        )

    run, _calls = _scripted_full_chain_runner(verification_acceptances=[0.70])

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        verification_callback=callback,
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "hard_veto"
    assert "verification_callback_continuation_veto" in result.hard_vetoes
    assert result.final_kernel_payload is None


def test_outer_loop_hard_veto_for_hard_veto_callback() -> None:
    def callback(
        _round_payload: Mapping[str, Any],
        _samples: Any,
        _diagnostics: Mapping[str, Any],
    ) -> FixedMassHMCTuningBudgetCallbackResult:
        return FixedMassHMCTuningBudgetCallbackResult(
            hard_vetoes=("domain_hard_veto",),
        )

    run, _calls = _scripted_full_chain_runner(verification_acceptances=[0.70])

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(max_attempts=2),
        verification_callback=callback,
        run_full_chain=run,
        _budget_policy_factory=_tiny_budget_factory,
    )

    assert result.final_status == "hard_veto"
    assert "domain_hard_veto" in result.hard_vetoes
    assert result.final_kernel_payload is None


@pytest.mark.parametrize("corruption", ["missing", "count_mismatch"])
def test_outer_loop_joint_invalid_candidate_handoff_fails_closed_before_phase6(
    corruption: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, bootstrap, windowed, fixed, _handoff, _selected, _alternative = (
        _phase7_direct_fixture()
    )
    if corruption == "missing":
        invalid = replace(fixed, _candidate_batch_handoff=None)
    else:
        invalid = replace(
            fixed,
            diagnostics={
                **dict(fixed.diagnostics),
                "candidate_count": int(fixed.diagnostics["candidate_count"]) + 1,
            },
        )
    phase6_calls: list[str] = []
    direct_calls: list[str] = []

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_phase7_direct_candidate_verification",
        lambda **_kwargs: direct_calls.append("unexpected"),
    )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_loop_config(max_attempts=1),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: invalid,
        _frozen_step_trajectory_stage_runner=lambda **_kwargs: phase6_calls.append(
            "unexpected"
        ),
    )

    assert result.final_status == "hard_veto"
    assert result.hard_vetoes == ("phase7_runtime_error",)
    assert result.attempts[0].verification_diagnostics["error_type"] == "ValueError"
    assert phase6_calls == []
    assert direct_calls == []
    assert result.final_kernel_payload is None
    assert result.final_kernel_hash is None


def test_outer_loop_labels_phase6_repair_handoff_when_final_attempt_has_no_slot() -> None:
    windowed = _windowed_stage()
    fixed_pass_run, _ = _scripted_full_chain_runner(
        phase5_screen_acceptances=[0.70],
        verification_acceptances=[0.70],
    )
    fixed_pass = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=fixed_pass_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )
    fixed_pass = _historical_phase5_stage(fixed_pass)

    def trajectory_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.90,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory_fail = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed_pass,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        run_full_chain=trajectory_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )

    assert fixed_pass.passed is True
    assert trajectory_fail.passed is False
    assert {
        candidate["trajectory_window_relation"]
        for candidate in trajectory_fail.candidate_results
    } == {"inside_trajectory_window"}

    def windowed_stage_runner(**_kwargs: Any):
        return windowed

    fixed_stage_calls: list[int] = []

    def fixed_step_stage_runner(**kwargs: Any):
        attempt_index = int(kwargs["_attempt_index"])
        fixed_stage_calls.append(attempt_index)
        if attempt_index < 2:
            repair_payload = {
                "runtime": "bayesfilter.test.phase5_repair_fixture",
                "step_size": 0.1,
                "private_handoff_only": True,
            }
            return replace(
                fixed_pass,
                final_status="repair_or_retry",
                diagnostic_role="repair_trigger",
                repair_triggers=("fixed_mass_fixture_repair",),
                selected_step_payload=None,
                selected_step_hash=None,
                repair_step_payload=repair_payload,
                repair_step_hash=hmc_kernel_tuning.stable_config_hash(repair_payload),
            )
        return fixed_pass

    def trajectory_stage_runner(**_kwargs: Any):
        return trajectory_fail

    def verification_runner(**_kwargs: Any):
        raise AssertionError("verification must not run without a Phase 6 pass")

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=3,
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_stage_runner,
        _fixed_mass_step_stage_runner=fixed_step_stage_runner,
        _frozen_step_trajectory_stage_runner=trajectory_stage_runner,
        _phase7_final_verification_runner=verification_runner,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == (
        "phase7_repair_handoff_budget_exhausted_no_attempt_slot"
    )
    assert len(result.attempts) == 3
    assert fixed_stage_calls == [0, 1, 2]
    assert result.attempts[0].fixed_mass_step_stage.passed is False
    assert result.attempts[0].frozen_step_trajectory_stage is None
    assert result.attempts[1].fixed_mass_step_stage.passed is False
    assert result.attempts[1].frozen_step_trajectory_stage is None
    final_attempt = result.attempts[2]
    assert final_attempt.fixed_mass_step_stage.passed is True
    assert final_attempt.frozen_step_trajectory_stage.passed is False
    assert final_attempt.handoff_state_payload["verification_repair_applied"] is True
    assert (
        final_attempt.handoff_state_payload["verification_repair_trigger"]
        == "phase6_trajectory_acceptance_outside_pass_band"
    )
    assert (
        final_attempt.handoff_state_payload["verification_repair_source"]
        == "phase6_frozen_step_trajectory_acceptance"
    )
    assert (
        "phase7_repair_handoff_budget_exhausted_no_attempt_slot"
        in result.repair_triggers
    )
    assert "phase7_budget_exhausted" not in result.repair_triggers
    guard = result.terminal_budget_guard_payload
    assert guard["classification"] == (
        "phase7_repair_handoff_budget_exhausted_no_attempt_slot"
    )
    assert guard["last_attempt_index"] == 2
    assert guard["configured_max_attempts"] == 3
    assert guard["remaining_attempt_slots"] == 0
    assert guard["last_handoff_stage"] == "phase5_selected"
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    public_guard = public_summary["terminal_budget_guard"]
    assert public_guard["classification"] == guard["classification"]
    assert public_guard["hmc_mechanics_exposed"] is False
    text = json.dumps(public_summary, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_outer_loop_terminal_phase6_repair_slot_can_pass_once() -> None:
    windowed = _windowed_stage()
    fixed_pass_run, _ = _scripted_full_chain_runner(
        phase5_screen_acceptances=[0.70],
        verification_acceptances=[0.70],
    )
    fixed_pass = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=fixed_pass_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )
    fixed_pass = _historical_phase5_stage(fixed_pass)

    def trajectory_fail_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.90,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory_fail = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed_pass,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        run_full_chain=trajectory_fail_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )
    fixed_stage_calls: list[int] = []
    trajectory_stage_calls: list[int] = []
    verification_calls: list[int] = []

    def windowed_stage_runner(**_kwargs: Any):
        return windowed

    def fixed_step_stage_runner(**kwargs: Any):
        attempt_index = int(kwargs["_attempt_index"])
        fixed_stage_calls.append(attempt_index)
        return fixed_pass

    def trajectory_stage_runner(**kwargs: Any):
        attempt_index = int(kwargs["_attempt_index"])
        trajectory_stage_calls.append(attempt_index)
        if attempt_index == 0:
            return trajectory_fail

        def trajectory_pass_run(_adapter: Any, _initial_state: Any, config: Any):
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                samples=np.zeros((int(config.num_results), 2)),
            )

        return hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=fixed_pass,
            config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
                target_accept_prob=0.70,
                seed=(20260621, 61),
                chain_execution_mode="eager",
                target_scope="kernel_fixed_mass_step_toy_gaussian",
                trajectory_window_lower_multiplier=0.01,
                trajectory_window_upper_multiplier=100.0,
            ),
            run_full_chain=trajectory_pass_run,
            _attempt_budget_policy=kwargs["_attempt_budget_policy"],
            _attempt_state=kwargs.get("_attempt_state"),
        )

    def verification_runner(
        *,
        attempt_index: int,
        budget_policy: Any,
        verification_start_callback: Any,
        **_kwargs: Any,
    ):
        verification_calls.append(int(attempt_index))
        if verification_start_callback is not None:
            verification_start_callback()
        return (
            {
                "verification_policy": "fixed_kernel",
                "num_results": int(budget_policy.verification_num_results),
                "acceptance_band": (0.65, 0.75),
            },
            {
                "acceptance_rate": 0.70,
                "runtime_finite": True,
                "samples_all_finite": True,
                "target_log_prob_finite": True,
                "log_accept_ratio_finite": True,
            },
            FixedMassHMCTuningBudgetCallbackResult(),
            "passed",
            "fresh_fixed_kernel_verification_passed",
            (),
            (),
        )

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            terminal_phase6_repair_extra_attempts=1,
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=windowed_stage_runner,
        _fixed_mass_step_stage_runner=fixed_step_stage_runner,
        _frozen_step_trajectory_stage_runner=trajectory_stage_runner,
        _phase7_final_verification_runner=verification_runner,
    )

    assert result.final_status == "passed"
    assert len(result.attempts) == 2
    assert [attempt.attempt_index for attempt in result.attempts] == [0, 1]
    assert fixed_stage_calls == [0, 1]
    assert trajectory_stage_calls == [0, 1]
    assert verification_calls == [1]
    first, extra = result.attempts
    assert first.final_status == "repair_or_retry"
    assert first.handoff_state_payload["verification_repair_applied"] is True
    assert (
        first.handoff_state_payload["verification_repair_trigger"]
        == "phase6_trajectory_acceptance_outside_pass_band"
    )
    budget_summary = hmc_kernel_tuning._phase7_attempt_budget_public_summary(
        extra.budget_policy_payload
    )
    assert budget_summary["terminal_phase6_repair_extra_attempt"] is True
    assert budget_summary["terminal_phase6_repair_extra_attempt_index"] == 1
    assert budget_summary["terminal_phase6_repair_extra_attempts"] == 1
    public_summary = hmc_kernel_tuning._phase7_public_summary(result)
    assert public_summary["attempt_count"] == 2
    public_text = json.dumps(public_summary, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in public_text


def test_outer_loop_terminal_phase6_repair_slot_closes_after_one_extra_attempt() -> None:
    windowed = _windowed_stage()
    fixed_pass_run, _ = _scripted_full_chain_runner(
        phase5_screen_acceptances=[0.70],
        verification_acceptances=[0.70],
    )
    fixed_pass = hmc_kernel_tuning.run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        config=hmc_kernel_tuning.HMCFixedMassStepStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 50),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        run_full_chain=fixed_pass_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )
    fixed_pass = _historical_phase5_stage(fixed_pass)

    def trajectory_fail_run(_adapter: Any, _initial_state: Any, config: Any):
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.90,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory_fail = hmc_kernel_tuning.run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed_pass,
        config=hmc_kernel_tuning.HMCFrozenStepTrajectoryStageConfig(
            target_accept_prob=0.70,
            seed=(20260621, 60),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        run_full_chain=trajectory_fail_run,
        _attempt_budget_policy=_tiny_budget_factory(2, 2),
    )
    trajectory_stage_calls: list[int] = []

    def trajectory_stage_runner(**kwargs: Any):
        attempt_index = int(kwargs["_attempt_index"])
        trajectory_stage_calls.append(attempt_index)
        return trajectory_fail

    result = run_hmc_tune_verify_repair_loop(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_loop_config(
            max_attempts=1,
            terminal_phase6_repair_extra_attempts=1,
            trajectory_window_lower_multiplier=0.01,
            trajectory_window_upper_multiplier=100.0,
        ),
        _budget_policy_factory=_tiny_budget_factory,
        _windowed_stage_runner=lambda **_kwargs: windowed,
        _fixed_mass_step_stage_runner=lambda **_kwargs: fixed_pass,
        _frozen_step_trajectory_stage_runner=trajectory_stage_runner,
        _phase7_final_verification_runner=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("verification must not run when Phase 6 fails")
        ),
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "phase7_terminal_phase6_repair_slot_exhausted"
    assert len(result.attempts) == 2
    assert trajectory_stage_calls == [0, 1]
    guard = result.terminal_budget_guard_payload
    assert guard["classification"] == "phase7_terminal_phase6_repair_slot_exhausted"
    assert guard["terminal_phase6_repair_extra_attempts"] == 1
    assert guard["terminal_phase6_repair_extra_attempts_consumed"] == 1
    public_guard = hmc_kernel_tuning._phase7_public_summary(result)[
        "terminal_budget_guard"
    ]
    assert public_guard["classification"] == guard["classification"]
    assert public_guard["hmc_mechanics_exposed"] is False


def test_outer_loop_terminal_phase6_repair_slot_config_default_and_public_threading() -> None:
    kernel_config = hmc_kernel_tuning.HMCKernelTuningConfig.standard()
    assert kernel_config.terminal_phase6_repair_extra_attempts == 0
    assert (
        kernel_config.payload()["terminal_phase6_repair_extra_attempts"]
        == 0
    )
    loop_config = hmc_kernel_tuning._public_loop_config(kernel_config)
    assert loop_config.terminal_phase6_repair_extra_attempts == 0
    assert (
        loop_config.payload()["terminal_phase6_repair_extra_attempts"]
        == 0
    )
    opted_in = hmc_kernel_tuning.HMCKernelTuningConfig.standard(
        terminal_phase6_repair_extra_attempts=1
    )
    assert (
        hmc_kernel_tuning._public_loop_config(opted_in)
        .terminal_phase6_repair_extra_attempts
        == 1
    )
    with pytest.raises(ValueError, match="terminal_phase6_repair_extra_attempts"):
        hmc_kernel_tuning.HMCKernelTuningConfig.standard(
            terminal_phase6_repair_extra_attempts=2
        )
    with pytest.raises(ValueError, match="terminal_phase6_repair_extra_attempts"):
        _loop_config(terminal_phase6_repair_extra_attempts=-1)


def test_outer_loop_public_exports_are_scoped_without_final_tuner() -> None:
    assert bayesfilter.HMCTuneVerifyRepairLoopConfig is HMCTuneVerifyRepairLoopConfig
    assert bayesfilter.HMCTuneVerifyRepairLoopResult is HMCTuneVerifyRepairLoopResult
    assert bayesfilter.run_hmc_tune_verify_repair_loop is run_hmc_tune_verify_repair_loop
    assert bayesfilter.TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS is TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS
    assert hasattr(bayesfilter, "tune_hmc_kernel")
    assert "tune_hmc_kernel" in bayesfilter.__all__


def test_public_progress_writes_are_atomic_under_concurrent_reader(
    tmp_path: Any,
) -> None:
    progress_path = tmp_path / "hmc_kernel_tuning_progress.json"
    config = hmc_kernel_tuning.HMCKernelTuningConfig.smoke()
    parse_errors: list[Exception] = []
    stop = threading.Event()

    def writer(writer_index: int) -> None:
        for sequence in range(40):
            hmc_kernel_tuning._write_public_tuning_progress_if_requested(
                progress_path=progress_path,
                config=config,
                artifact_path=None,
                current_stage=f"writer_{writer_index}_{sequence}",
                last_started_stage="test",
                last_completed_stage=None,
                adapter_signature="test-adapter",
                target_dimension=2,
            )

    def reader() -> None:
        while not stop.is_set():
            try:
                json.loads(progress_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except Exception as exc:  # noqa: BLE001 - capture torn reader evidence.
                parse_errors.append(exc)
                return

    reader_thread = threading.Thread(target=reader)
    writers = [threading.Thread(target=writer, args=(index,)) for index in range(3)]
    reader_thread.start()
    for thread in writers:
        thread.start()
    for thread in writers:
        thread.join()
    stop.set()
    reader_thread.join()

    assert parse_errors == []
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "bayesfilter.hmc_kernel_tuning_progress.v1"
    assert not tuple(tmp_path.glob("*.tmp"))
