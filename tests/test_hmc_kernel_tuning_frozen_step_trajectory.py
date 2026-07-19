from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning
from bayesfilter.inference import (
    FROZEN_STEP_TRAJECTORY_STAGE_NONCLAIMS,
    FixedMassHMCTuningBudgetCallbackResult,
    HMCFrozenStepTrajectoryStageConfig,
    HMCFrozenStepTrajectoryStageResult,
    run_hmc_frozen_step_trajectory_stage,
)
from bayesfilter.inference.hmc_kernel_tuning import (
    _HMCPhaseAttemptState,
    _frozen_step_trajectory_candidate_generation,
    _phase6_next_candidate_soft_deadline_veto,
)
from bayesfilter.runtime import stable_config_hash

from tests.test_hmc_kernel_tuning_fixed_mass_step import (
    _FakeRunResult,
    _MismatchedAdapter,
    _ToyGaussianAdapter,
    _bootstrap,
    _fake_result,
    _geometry,
    _runtime_metadata,
    _stage_config,
    _windowed_config,
    _windowed_stage,
    run_hmc_fixed_mass_step_stage,
)


def _trajectory_config(**overrides: Any) -> HMCFrozenStepTrajectoryStageConfig:
    payload = {
        "target_accept_prob": 0.70,
        "acceptance_band": (0.65, 0.75),
        "seed": (20260621, 60),
        "chain_execution_mode": "eager",
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
    }
    payload.update(overrides)
    return HMCFrozenStepTrajectoryStageConfig(**payload)


def _fixed_mass_step_stage(
    *,
    selected_acceptance: float = 0.70,
    windowed_stage: Any | None = None,
):
    stage_windowed = _windowed_stage() if windowed_stage is None else windowed_stage

    def run(adapter: Any, initial_state: Any, config: Any) -> _FakeRunResult:
        uses_tuning = bool(config.tuning_policy.uses_dual_averaging)
        if uses_tuning:
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                step_size=(np.pi / 2.0) / float(config.num_leapfrog_steps),
                num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
                samples=np.zeros((int(config.num_results), 2)),
            )
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=selected_acceptance,
            samples=np.zeros((int(config.num_results), 2)),
        )

    return run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=stage_windowed,
        config=_stage_config(),
        run_full_chain=run,
    )


def _windowed_and_step_stage(*, selected_acceptance: float = 0.70):
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(
        selected_acceptance=selected_acceptance,
        windowed_stage=windowed,
    )
    return windowed, step_stage


def _consistent_chain_for_geometry(
    geometry: Any,
    *,
    fixed_step_size: float | None = None,
    fixed_stage_config_overrides: Mapping[str, Any] | None = None,
):
    def bootstrap_run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    bootstrap = hmc_kernel_tuning.run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        run_full_chain=bootstrap_run,
    )

    def windowed_run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    windowed = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_windowed_config(),
        run_full_chain=windowed_run,
    )

    def fixed_step_run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        uses_tuning = bool(config.tuning_policy.uses_dual_averaging)
        if uses_tuning:
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                step_size=(
                    float(geometry.target_trajectory_length)
                    / float(config.num_leapfrog_steps)
                    if fixed_step_size is None
                    else float(fixed_step_size)
                ),
                num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
                samples=np.zeros((int(config.num_results), 2)),
            )
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    step_stage = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(**dict(fixed_stage_config_overrides or {})),
        run_full_chain=fixed_step_run,
    )
    return bootstrap, windowed, step_stage


def _expect_frozen_step_validation_error(
    match: str,
    *,
    adapter: Any | None = None,
    geometry: Any | None = None,
    bootstrap: Any | None = None,
    windowed_stage: Any | None = None,
    fixed_mass_step_stage: Any | None = None,
    config: HMCFrozenStepTrajectoryStageConfig | None = None,
) -> None:
    if windowed_stage is None:
        default_windowed = _windowed_stage()
        default_step_stage = (
            _fixed_mass_step_stage(windowed_stage=default_windowed)
            if fixed_mass_step_stage is None
            else fixed_mass_step_stage
        )
    else:
        default_windowed = windowed_stage
        default_step_stage = (
            _fixed_mass_step_stage()
            if fixed_mass_step_stage is None
            else fixed_mass_step_stage
        )
        if fixed_mass_step_stage is None:
            object.__setattr__(
                default_step_stage,
                "windowed_stage_artifact_hash",
                default_windowed.artifact_hash,
            )
    with pytest.raises(ValueError, match=match):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter() if adapter is None else adapter,
            geometry=_geometry() if geometry is None else geometry,
            bootstrap=_bootstrap() if bootstrap is None else bootstrap,
            windowed_stage=default_windowed,
            fixed_mass_step_stage=default_step_stage,
            config=_trajectory_config() if config is None else config,
            run_full_chain=_scripted_trajectory_runner({})[0],
        )


def _scripted_trajectory_runner(acceptance_by_l: Mapping[int, float]):
    calls: list[Mapping[str, Any]] = []

    def run(adapter: Any, initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append(
            {
                "adapter_signature": adapter.adapter_signature(),
                "initial_state": np.asarray(initial_state, dtype=float),
                "step_size": float(config.step_size),
                "num_leapfrog_steps": int(config.num_leapfrog_steps),
                "num_results": int(config.num_results),
                "num_burnin_steps": int(config.num_burnin_steps),
                "seed": tuple(config.seed),
                "trace_policy": config.trace_policy,
                "adaptation_policy": config.adaptation_policy,
                "uses_dual_averaging": bool(config.tuning_policy.uses_dual_averaging),
                "target_scope": config.target_scope,
                "use_xla": bool(config.use_xla),
            }
        )
        np.testing.assert_allclose(np.asarray(initial_state, dtype=float), np.zeros(2))
        acceptance = acceptance_by_l.get(int(config.num_leapfrog_steps), 0.82)
        metadata = dict(_runtime_metadata())
        metadata["nonclaims"] = (
            "deterministic hmc contract plumbing result",
            "no sampler convergence claim",
            "no posterior validity claim",
        )
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=acceptance,
            samples=np.zeros((int(config.num_results), 2)),
            metadata_overrides=metadata,
        )

    return run, calls


def test_frozen_step_trajectory_config_does_not_expose_hmc_mechanics() -> None:
    parameters = set(inspect.signature(HMCFrozenStepTrajectoryStageConfig).parameters)
    forbidden = {
        "candidate_l_values",
        "num_leapfrog_steps",
        "min_leapfrog",
        "leapfrog_bounds",
        "trajectory_grid",
        "target_trajectory_length",
        "step_size",
        "mass_window_schedule",
        "warmup_steps",
        "budget_schedule",
        "screen_num_results",
        "screen_num_burnin_steps",
    }

    assert parameters.isdisjoint(forbidden)
    assert "max_leapfrog_steps" in parameters


def test_frozen_step_trajectory_stage_passes_with_internal_candidates_and_real_hook() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    expected_candidates = (step_stage.fixed_num_leapfrog_steps,)
    run, calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.70}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(),
        run_full_chain=run,
    )

    assert isinstance(result, HMCFrozenStepTrajectoryStageResult)
    assert result.passed is True
    assert result.final_status == "passed"
    assert result.candidate_generation["candidate_l_values"] == expected_candidates
    assert result.candidate_generation["phase5_joint_l_epsilon_algorithm"] is True
    assert result.candidate_generation["no_second_frozen_epsilon_l_search"] is True
    assert result.candidate_generation["internal_min_leapfrog"] == 3
    assert result.candidate_generation["internal_max_leapfrog"] == 25
    assert result.selected_num_leapfrog_steps == step_stage.fixed_num_leapfrog_steps
    assert result.frozen_step_size == pytest.approx(step_stage.selected_step_size)
    assert result.fixed_bootstrap_num_leapfrog_steps == step_stage.fixed_num_leapfrog_steps
    assert result.selected_step_hash == step_stage.selected_step_hash
    assert result.selected_bootstrap_kernel_hash == bootstrap.selected_kernel_hash
    assert result.trajectory_hmc_adapter_signature == step_stage.ladder_hmc_adapter_signature
    assert result.phase5_ladder_hmc_adapter_signature == step_stage.ladder_hmc_adapter_signature
    assert result.frozen_mass_invariant["passed"] is True
    assert result.frozen_step_invariant["passed"] is True
    assert result.selected_trajectory_payload is not None
    assert result.selected_trajectory_hash == stable_config_hash(result.selected_trajectory_payload)
    assert result.selected_trajectory_payload["not_fresh_final_verification"] is True
    assert result.payload()["reports_fresh_final_verification"] is False
    assert result.payload()["candidate_mechanics_are_diagnostic_telemetry_only"] is True
    assert "not fresh final verification" in result.nonclaims
    assert FROZEN_STEP_TRAJECTORY_STAGE_NONCLAIMS == result.nonclaims
    assert [call["num_leapfrog_steps"] for call in calls] == list(expected_candidates)
    assert all(call["step_size"] == pytest.approx(step_stage.selected_step_size) for call in calls)
    assert all(call["trace_policy"] == "standard" for call in calls)
    assert all(call["adaptation_policy"] == "fixed_kernel_no_adaptation" for call in calls)
    assert all(call["uses_dual_averaging"] is False for call in calls)
    assert all(call["adapter_signature"] == step_stage.ladder_hmc_adapter_signature for call in calls)
    assert all(call["use_xla"] is False for call in calls)


def test_frozen_step_trajectory_private_callback_records_candidate_and_selection() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, _calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.70}
    )
    events: list[tuple[str, Mapping[str, Any]]] = []

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(),
        run_full_chain=run,
        _private_diagnostic_callback=lambda event_type, payload: events.append(
            (event_type, dict(payload))
        ),
    )

    candidate_events = [
        payload
        for event_type, payload in events
        if event_type == "frozen_step_trajectory_candidate_complete"
    ]
    selected_events = [
        payload
        for event_type, payload in events
        if event_type == "frozen_step_trajectory_round_selected"
    ]

    assert result.passed is True
    assert candidate_events
    assert selected_events
    candidate = candidate_events[-1]
    selected = selected_events[-1]
    assert candidate["step_size"] == pytest.approx(step_stage.selected_step_size)
    assert candidate["num_leapfrog_steps"] == step_stage.fixed_num_leapfrog_steps
    assert candidate["trajectory_length"] == pytest.approx(
        step_stage.selected_step_size * step_stage.fixed_num_leapfrog_steps
    )
    assert candidate["screen_acceptance_rate"] == pytest.approx(0.70)
    assert candidate["private_hmc_mechanics"] is True
    assert selected["selected_pair_exists"] is True
    assert selected["selected_candidate_index"] == result.selected_candidate_index
    assert selected["step_size"] == pytest.approx(result.frozen_step_size)
    assert selected["num_leapfrog_steps"] == result.selected_num_leapfrog_steps
    assert selected["trajectory_length"] == pytest.approx(
        result.selected_candidate["trajectory_length"]
    )
    assert selected["selected_trajectory_hash"] == result.selected_trajectory_hash
    assert selected["private_hmc_mechanics"] is True


def test_frozen_step_trajectory_config_use_xla_propagates_to_candidate_configs() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.70}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(chain_execution_mode="tf_function", use_xla=True),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.config.payload()["use_xla"] is True
    assert calls
    assert all(call["use_xla"] is True for call in calls)


def test_frozen_step_trajectory_injected_tf_function_bypasses_reusable_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_built(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("injected run_full_chain must bypass reusable runner")

    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.70}
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_reusable_full_chain_tfp_hmc_runner",
        fail_if_built,
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(chain_execution_mode="tf_function"),
        run_full_chain=run,
    )

    assert result.passed is True
    assert calls
    route = result.diagnostics["runner_route_summary"]
    assert route["active_route"] == "single_use_or_injected_runner"
    assert route["reusable_runner_build_count"] == 0
    assert route["injected_runner_call_count"] == len(calls)


def test_frozen_step_trajectory_default_tf_function_route_uses_reusable_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Mapping[str, Any]] = []

    class _FakeReusableRunner:
        def __init__(self, config: Any, *, dynamic_num_leapfrog_steps: bool) -> None:
            self.config = config
            self.dynamic_num_leapfrog_steps = bool(dynamic_num_leapfrog_steps)

        def run(
            self,
            *,
            current_state: Any,
            seed: Any,
            step_size: Any,
            num_leapfrog_steps: Any | None = None,
        ) -> _FakeRunResult:
            leapfrogs = (
                int(self.config.num_leapfrog_steps)
                if num_leapfrog_steps is None
                else int(num_leapfrog_steps)
            )
            calls.append(
                {
                    "role": "run",
                    "num_leapfrog_steps": leapfrogs,
                    "dynamic_num_leapfrog_steps": self.dynamic_num_leapfrog_steps,
                    "seed": tuple(int(item) for item in seed),
                    "step_size": float(step_size),
                    "initial_state": np.asarray(current_state, dtype=float),
                }
            )
            return _fake_result(
                num_results=int(self.config.num_results),
                acceptance=0.70,
                samples=np.zeros((int(self.config.num_results), 2)),
                metadata_overrides=_runtime_metadata(),
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
                "num_leapfrog_steps": int(config.num_leapfrog_steps),
                "chain_execution_mode": config.chain_execution_mode,
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
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(chain_execution_mode="tf_function"),
    )

    assert result.passed is True
    build_calls = [call for call in calls if call["role"] == "build"]
    run_calls = [call for call in calls if call["role"] == "run"]
    assert build_calls
    assert len(build_calls) == 1
    assert len(run_calls) == 1
    assert all(call["chain_execution_mode"] == "tf_function" for call in build_calls)
    assert build_calls[0]["dynamic_num_leapfrog_steps"] is True
    assert all(call["dynamic_num_leapfrog_steps"] is True for call in run_calls)
    assert all(
        call["adapter_signature"] == step_stage.ladder_hmc_adapter_signature
        for call in build_calls
    )
    route = result.diagnostics["runner_route_summary"]
    assert route["active_route"] == "frozen_step_trajectory_scoped_reusable_runner"
    assert route["single_use_build_count"] == 0
    assert route["reusable_runner_build_count"] == 1
    assert route["distinct_static_runner_contract_count"] == 1
    assert route["stage_runner_build_count"] == 1
    assert route["stage_runner_reuse_count"] == 0
    assert route["stage_new_distinct_contract_count"] == 1
    assert route["final_distinct_contract_count"] == 1
    assert route["contract_count_invariant_satisfied"] is True
    assert not any(event["runner_reused"] is True for event in route["round_route_events"])
    assert route["fallback_status"] == "none"
    assert route["fallback_to_single_use_runner"] is False
    assert route["runner_cache_handoff_source"] == "none"
    assert route["initial_handoff_contract_count"] == 0
    assert result.candidate_results[0]["runner_route_event"]["route"] == (
        "frozen_step_trajectory_scoped_reusable_runner"
    )
    assert result.candidate_results[0]["diagnostics"]["runtime_metadata"][
        "kernel_stage_route"
    ] == "frozen_step_trajectory_scoped_reusable_runner"


def test_runner_route_summary_does_not_count_inherited_contract_as_stage_build() -> None:
    summary = hmc_kernel_tuning._kernel_stage_runner_route_summary(
        active_route="frozen_step_trajectory_scoped_reusable_runner",
        events=(
            {
                "route": "frozen_step_trajectory_scoped_reusable_runner",
                "runner_reused": True,
                "used_single_use_runner": False,
            },
        ),
        contract_payloads={"inherited-contract": {"static": "payload"}},
        semantic_source="test_inherited_contract_arithmetic",
        reuse_nonclaim="engineering telemetry only",
        handoff_source="fixed_mass_step",
        initial_handoff_contract_count=1,
    )

    assert summary["initial_handoff_contract_count"] == 1
    assert summary["stage_runner_build_count"] == 0
    assert summary["stage_runner_reuse_count"] == 1
    assert summary["stage_new_distinct_contract_count"] == 0
    assert summary["final_distinct_contract_count"] == 1
    assert summary["reusable_runner_build_count"] == 0
    assert summary["contract_count_invariant_satisfied"] is True


def test_frozen_step_trajectory_stage_returns_repair_without_selected_l_when_no_candidate_passes() -> None:
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, _calls = _scripted_trajectory_runner({})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "repair_or_retry"
    assert result.selected_candidate_index is None
    assert result.selected_trajectory_payload is None
    assert result.selected_trajectory_hash is None
    assert "acceptance_outside_pass_band" in result.repair_triggers


def test_frozen_step_trajectory_high_acceptance_underreach_cannot_pass() -> None:
    geometry = replace(_geometry(), target_trajectory_length=100.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(
        geometry,
        fixed_step_size=0.2,
        fixed_stage_config_overrides={"trajectory_window_lower_multiplier": 0.04},
    )
    run, _calls = _scripted_trajectory_runner({})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "repair_or_retry"
    assert all(
        candidate["trajectory_window_relation"] == "below_trajectory_window"
        for candidate in result.candidate_results
    )
    assert (
        result.diagnostics["phase6_handoff_screen_policy_role"]
        == "handoff_screen_repair_trigger_non_promoting"
    )
    assert result.diagnostics["phase6_handoff_screen_is_fresh_final_verification"] is False
    assert (
        result.diagnostics[
            "phase6_handoff_screen_is_posterior_or_sampler_validity_evidence"
        ]
        is False
    )
    assert (
        result.diagnostics["trajectory_window_policy_role"]
        == "engineering_viability_gate_non_scientific"
    )
    assert result.diagnostics["trajectory_window_viability_gate_active"] is True
    assert result.diagnostics["trajectory_window_relations_seen"] == (
        "below_trajectory_window",
    )
    assert result.diagnostics["reports_sampler_superiority"] is False
    assert result.diagnostics["reports_default_readiness"] is False
    assert "high_acceptance_trajectory_underreach" in result.repair_triggers
    assert "trajectory_length_below_window" in result.repair_triggers


def test_frozen_step_trajectory_high_acceptance_underreach_nominates_under_phase23() -> None:
    geometry = replace(_geometry(), target_trajectory_length=100.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(
        geometry,
        fixed_step_size=0.2,
        fixed_stage_config_overrides={
            "trajectory_window_lower_multiplier": 0.04,
            "handoff_screen_policy": "phase23_nomination_only",
        },
    )
    run, _calls = _scripted_trajectory_runner({})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(
            max_leapfrog_steps=25,
            handoff_screen_policy="phase23_nomination_only",
        ),
        run_full_chain=run,
    )
    public_summary = hmc_kernel_tuning._frozen_step_trajectory_public_summary(result)

    assert result.passed is True
    assert result.final_status == "passed"
    assert result.selected_candidate is not None
    assert result.selected_candidate["classification"] == "nomination_screen"
    assert result.selected_candidate["nomination_eligible"] is True
    assert result.selected_candidate["trajectory_window_relation"] == (
        "below_trajectory_window"
    )
    assert result.selected_trajectory_payload is not None
    assert result.selected_trajectory_payload["not_fresh_final_verification"] is True
    assert result.diagnostics["handoff_screen_policy"] == "phase23_nomination_only"
    assert result.diagnostics["trajectory_window_viability_gate_active"] is False
    assert result.diagnostics["trajectory_window_nomination_only"] is True
    assert public_summary["handoff_screen_policy"] == "phase23_nomination_only"
    assert public_summary["nomination_candidate_count"] >= 1
    assert public_summary["reports_fresh_final_verification"] is False
    assert public_summary["reports_posterior_convergence"] is False
    assert public_summary["reports_sampler_superiority"] is False


def test_frozen_step_trajectory_acceptance_pass_underreach_cannot_pass() -> None:
    geometry = replace(_geometry(), target_trajectory_length=100.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(
        geometry,
        fixed_step_size=0.2,
        fixed_stage_config_overrides={"trajectory_window_lower_multiplier": 0.04},
    )
    run, _calls = _scripted_trajectory_runner({25: 0.70})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "repair_or_retry"
    assert "acceptance_pass_but_trajectory_underreach" in result.repair_triggers
    assert result.candidate_results[-1]["num_leapfrog_steps"] == 25
    assert result.candidate_results[-1]["diagnostics"]["acceptance_rate"] == pytest.approx(0.70)


def test_frozen_step_trajectory_acceptance_pass_underreach_nominates_under_phase23() -> None:
    geometry = replace(_geometry(), target_trajectory_length=100.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(
        geometry,
        fixed_step_size=0.2,
        fixed_stage_config_overrides={
            "trajectory_window_lower_multiplier": 0.04,
            "handoff_screen_policy": "phase23_nomination_only",
        },
    )
    run, _calls = _scripted_trajectory_runner({25: 0.70})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(
            max_leapfrog_steps=25,
            handoff_screen_policy="phase23_nomination_only",
        ),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.final_status == "passed"
    assert result.selected_candidate is not None
    assert result.selected_candidate["classification"] == "nomination_screen"
    assert result.selected_candidate["num_leapfrog_steps"] == 25
    assert result.selected_candidate["diagnostics"]["acceptance_rate"] == pytest.approx(
        0.70
    )
    assert "acceptance_pass_but_trajectory_underreach" in result.repair_triggers
    assert result.diagnostics["trajectory_window_viability_gate_active"] is False
    assert result.diagnostics["reports_sampler_superiority"] is False
    assert result.payload()["reports_fresh_final_verification"] is False


def test_frozen_step_trajectory_phase23_preserves_invalid_trajectory_hard_veto() -> None:
    (
        classification,
        diagnostic_role,
        hard_vetoes,
        continuation_vetoes,
        promotion_vetoes,
        repair_triggers,
    ) = hmc_kernel_tuning._classify_frozen_step_trajectory_candidate(
        _trajectory_config(handoff_screen_policy="phase23_nomination_only"),
        diagnostics={
            "acceptance_rate": 0.70,
            "runtime_finite": True,
            "log_accept_ratio_finite": True,
            "samples_all_finite": True,
            "target_log_prob_finite": True,
        },
        trajectory_window={
            "trajectory_window_relation": "invalid_trajectory_length",
            "leapfrog_exceeds_max": False,
        },
        screen_error=None,
        callback_result=FixedMassHMCTuningBudgetCallbackResult(),
    )

    assert classification == "hard_veto"
    assert diagnostic_role == "hard_veto"
    assert "trajectory_length_invalid" in hard_vetoes
    assert continuation_vetoes == ()
    assert promotion_vetoes == ()
    assert repair_triggers == ()


def test_phase6_underreach_repair_preserves_in_band_acceptance_relation() -> None:
    geometry = replace(_geometry(), target_trajectory_length=100.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(
        geometry,
        fixed_step_size=0.2,
        fixed_stage_config_overrides={"trajectory_window_lower_multiplier": 0.04},
    )
    candidates = _frozen_step_trajectory_candidate_generation(
        geometry=geometry,
        selected_step_size=step_stage.selected_step_size,
        fixed_bootstrap_l=step_stage.fixed_num_leapfrog_steps,
        max_leapfrog_steps=25,
    )["candidate_l_values"]
    run, _calls = _scripted_trajectory_runner({int(l): 0.70 for l in candidates})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase6_trajectory_repair_handoff_payload(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        frozen_step_trajectory_stage=result,
    )
    state = _HMCPhaseAttemptState(
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        selected_num_leapfrog_steps=25,
        selected_trajectory_hash="previous-trajectory-hash",
        handoff_stage="phase6",
        **handoff,
    )

    assert handoff["verification_repair_applied"] is True
    assert handoff["verification_repair_source"] == "phase6_frozen_step_trajectory_underreach"
    assert handoff["verification_acceptance_relation"] == "inside_acceptance_band"
    assert state.verification_repair_applied is True
    assert state.verification_acceptance_relation == "inside_acceptance_band"


def test_frozen_step_trajectory_boundary_acceptance_and_tau_pass() -> None:
    geometry = replace(_geometry(), target_trajectory_length=5.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(geometry)
    # Phase 5 already selected the joint L/epsilon pair. Phase 6 only screens
    # that selected pair; it does not perform a second frozen-epsilon L search.
    run, calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.75}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.selected_num_leapfrog_steps == step_stage.fixed_num_leapfrog_steps
    assert result.selected_candidate["trajectory_window_relation"] == "inside_trajectory_window"
    assert result.selected_candidate["trajectory_length"] == pytest.approx(5.0)
    assert [call["num_leapfrog_steps"] for call in calls] == [
        step_stage.fixed_num_leapfrog_steps
    ]


def test_frozen_step_trajectory_configured_cap_changes_candidates_and_floor() -> None:
    geometry = replace(_geometry(), target_trajectory_length=100.0)
    default = _frozen_step_trajectory_candidate_generation(
        geometry=geometry,
        selected_step_size=0.2,
        fixed_bootstrap_l=3,
        max_leapfrog_steps=25,
    )
    raised = _frozen_step_trajectory_candidate_generation(
        geometry=geometry,
        selected_step_size=0.2,
        fixed_bootstrap_l=3,
        max_leapfrog_steps=80,
    )

    assert default["internal_max_leapfrog"] == 25
    assert raised["internal_max_leapfrog"] == 80
    assert max(default["candidate_l_values"]) == 25
    assert max(raised["candidate_l_values"]) == 80
    assert default["minimum_step_size_for_tau_floor"] == pytest.approx(30.0 / 25.0)
    assert raised["minimum_step_size_for_tau_floor"] == pytest.approx(30.0 / 80.0)


def test_phase6_underreach_repair_handoff_floors_step_and_candidate_generation_realizes_floor() -> None:
    geometry = replace(_geometry(), target_trajectory_length=100.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(
        geometry,
        fixed_step_size=0.2,
        fixed_stage_config_overrides={"trajectory_window_lower_multiplier": 0.04},
    )
    run, _calls = _scripted_trajectory_runner({})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase6_trajectory_repair_handoff_payload(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        frozen_step_trajectory_stage=result,
    )

    assert handoff["verification_repair_applied"] is True
    assert handoff["verification_repair_source"] == "phase6_frozen_step_trajectory_underreach"
    assert handoff["verification_repair_step_size"] >= 30.0 / 25.0

    repaired = _frozen_step_trajectory_candidate_generation(
        geometry=geometry,
        selected_step_size=handoff["verification_repair_step_size"],
        fixed_bootstrap_l=3,
        max_leapfrog_steps=25,
        attempt_state=_HMCPhaseAttemptState(
            selected_step_size=step_stage.selected_step_size,
            selected_step_hash=step_stage.selected_step_hash,
            selected_num_leapfrog_steps=25,
            selected_trajectory_hash="previous-trajectory-hash",
            verification_acceptance_relation="above_acceptance_band",
            verification_repair_trigger="phase6_trajectory_acceptance_outside_pass_band",
            verification_repair_source="phase6_frozen_step_trajectory_underreach",
            verification_repair_step_size=handoff["verification_repair_step_size"],
            verification_repair_step_hash=handoff["verification_repair_step_hash"],
            verification_repair_applied=True,
            handoff_stage="phase6",
        ),
    )

    assert 25 in repaired["candidate_l_values"]
    assert repaired["tau_floor_feasible_at_selected_step"] is True


def test_frozen_step_trajectory_stage_public_counts_all_above_without_mechanics() -> None:
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, calls = _scripted_trajectory_runner({})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(),
        run_full_chain=run,
    )
    public_summary = hmc_kernel_tuning._frozen_step_trajectory_public_summary(result)

    assert calls
    assert result.final_status == "repair_or_retry"
    assert result.selected_trajectory_payload is None
    assert public_summary["candidate_acceptance_relation_counts"] == {
        "below_acceptance_band": 0,
        "inside_acceptance_band": 0,
        "above_acceptance_band": len(calls),
        "unavailable": 0,
    }
    assert public_summary["candidate_grid_exposed"] is False
    assert public_summary["hmc_mechanics_exposed"] is False
    text = json.dumps(public_summary, sort_keys=True)
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
    ):
        assert forbidden not in text


def test_phase6_mixed_window_all_high_acceptance_handoff_increases_step() -> None:
    geometry = replace(_geometry(), target_trajectory_length=1.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(geometry)
    run, _calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.90}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase6_trajectory_repair_handoff_payload(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        frozen_step_trajectory_stage=result,
    )

    assert result.final_status == "repair_or_retry"
    assert result.candidate_generation["no_second_frozen_epsilon_l_search"] is True
    assert len(result.candidate_results) == 1
    assert result.candidate_results[0]["trajectory_window_relation"] == (
        "inside_trajectory_window"
    )
    assert handoff["verification_acceptance_relation"] == "above_acceptance_band"
    assert handoff["verification_repair_source"] == (
        "phase6_frozen_step_trajectory_acceptance"
    )
    tau_max = result.candidate_generation["trajectory_window"][1]
    feasible_ceiling = tau_max / result.candidate_generation["internal_min_leapfrog"]
    assert 2.0 * step_stage.selected_step_size <= feasible_ceiling
    assert handoff["verification_repair_step_size"] == pytest.approx(
        2.0 * step_stage.selected_step_size
    )
    assert handoff["verification_repair_max_step_size"] is None
    assert handoff["verification_repair_applied"] is True


def test_phase6_high_acceptance_handoff_caps_step_to_tau_feasible_ceiling() -> None:
    geometry = replace(_geometry(), target_trajectory_length=0.9)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(geometry)
    run, _calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.90}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(
            max_leapfrog_steps=25,
            trajectory_window_lower_multiplier=0.8,
            trajectory_window_upper_multiplier=1.25,
        ),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase6_trajectory_repair_handoff_payload(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        frozen_step_trajectory_stage=result,
    )

    tau_min, tau_max = result.candidate_generation["trajectory_window"]
    feasible_ceiling = tau_max / result.candidate_generation["internal_min_leapfrog"]
    bracket_state = handoff["fixed_mass_bracket_state"]

    assert tau_min == pytest.approx(0.72)
    assert tau_max == pytest.approx(1.125)
    assert 2.0 * step_stage.selected_step_size > feasible_ceiling
    assert handoff["verification_repair_step_size"] == pytest.approx(
        feasible_ceiling
    )
    assert handoff["verification_repair_max_step_size"] == pytest.approx(
        feasible_ceiling
    )
    assert bracket_state["next_step_size"] == pytest.approx(feasible_ceiling)
    assert bracket_state["high_acceptance_step_lower_bound"] == pytest.approx(
        feasible_ceiling
    )
    assert bracket_state["private_handoff_only"] is True
    assert bracket_state["public_progress_exposes_step"] is False
    assert bracket_state["tau_feasible_handoff"]["step_ceiling_applied"] is True


def test_phase6_mixed_window_inside_tau_high_acceptance_repairs_step() -> None:
    geometry = replace(_geometry(), target_trajectory_length=1.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(geometry)
    run, _calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.88}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase6_trajectory_repair_handoff_payload(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        frozen_step_trajectory_stage=result,
    )

    inside = [
        candidate
        for candidate in result.candidate_results
        if candidate["trajectory_window_relation"] == "inside_trajectory_window"
    ]
    assert result.final_status == "repair_or_retry"
    assert inside
    assert all(
        candidate["diagnostics"]["acceptance_rate"] > 0.75 for candidate in inside
    )
    assert len(result.candidate_results) == 1
    assert result.candidate_generation["no_second_frozen_epsilon_l_search"] is True
    assert handoff["verification_acceptance_relation"] == "above_acceptance_band"
    assert handoff["verification_repair_source"] == (
        "phase6_frozen_step_trajectory_acceptance"
    )
    tau_max = result.candidate_generation["trajectory_window"][1]
    feasible_ceiling = tau_max / result.candidate_generation["internal_min_leapfrog"]
    assert 2.0 * step_stage.selected_step_size <= feasible_ceiling
    assert handoff["verification_repair_step_size"] == pytest.approx(
        2.0 * step_stage.selected_step_size
    )
    assert handoff["verification_repair_max_step_size"] is None
    assert handoff["fixed_mass_bracket_state"]["next_step_size"] == pytest.approx(
        handoff["verification_repair_step_size"]
    )
    assert handoff["verification_repair_applied"] is True


def test_phase6_singleton_inside_tau_high_acceptance_repairs_step() -> None:
    geometry = replace(_geometry(), target_trajectory_length=0.5)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(geometry)
    run, _calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.88}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase6_trajectory_repair_handoff_payload(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        frozen_step_trajectory_stage=result,
    )

    inside = [
        candidate
        for candidate in result.candidate_results
        if candidate["trajectory_window_relation"] == "inside_trajectory_window"
    ]
    assert len(inside) == 1
    assert inside[0]["num_leapfrog_steps"] == step_stage.fixed_num_leapfrog_steps
    assert inside[0]["diagnostics"]["acceptance_rate"] == pytest.approx(0.88)
    assert handoff["verification_repair_source"] == (
        "phase6_frozen_step_trajectory_acceptance"
    )
    tau_max = result.candidate_generation["trajectory_window"][1]
    feasible_ceiling = tau_max / result.candidate_generation["internal_min_leapfrog"]
    assert 2.0 * step_stage.selected_step_size <= feasible_ceiling
    assert handoff["verification_repair_step_size"] == pytest.approx(
        2.0 * step_stage.selected_step_size
    )
    assert handoff["verification_repair_max_step_size"] is None
    assert handoff["fixed_mass_bracket_state"]["next_step_size"] == pytest.approx(
        handoff["verification_repair_step_size"]
    )
    assert handoff["verification_repair_applied"] is True


def test_phase6_mixed_inside_tau_acceptance_sides_do_not_repair_step() -> None:
    geometry = replace(_geometry(), target_trajectory_length=1.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(geometry)
    run, _calls = _scripted_trajectory_runner(
        {step_stage.fixed_num_leapfrog_steps: 0.50}
    )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase6_trajectory_repair_handoff_payload(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        selected_step_size=step_stage.selected_step_size,
        selected_step_hash=step_stage.selected_step_hash,
        frozen_step_trajectory_stage=result,
    )

    assert len(result.candidate_results) == 1
    assert result.candidate_results[0]["trajectory_window_relation"] == (
        "inside_trajectory_window"
    )
    assert result.candidate_results[0]["diagnostics"]["acceptance_rate"] < 0.65
    assert handoff["verification_acceptance_relation"] == "below_acceptance_band"
    assert handoff["verification_repair_source"] == (
        "phase6_frozen_step_trajectory_acceptance"
    )
    assert handoff["verification_repair_step_size"] == pytest.approx(
        0.5 * step_stage.selected_step_size
    )
    assert handoff["verification_repair_applied"] is True


def test_phase6_failed_mixed_window_handoff_preserves_private_retry_l_anchor() -> None:
    geometry = replace(_geometry(), target_trajectory_length=1.0)
    bootstrap, windowed, step_stage = _consistent_chain_for_geometry(geometry)
    candidates = _frozen_step_trajectory_candidate_generation(
        geometry=geometry,
        selected_step_size=step_stage.selected_step_size,
        fixed_bootstrap_l=step_stage.fixed_num_leapfrog_steps,
        max_leapfrog_steps=25,
    )["candidate_l_values"]
    run, _calls = _scripted_trajectory_runner({int(l): 0.90 for l in candidates})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
    )
    state = hmc_kernel_tuning._phase7_attempt_state_from_stages(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            max_leapfrog_steps=25,
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        frozen_step_trajectory_stage=result,
    )

    assert result.final_status == "repair_or_retry"
    assert state.selected_num_leapfrog_steps is None
    assert state.phase6_retry_num_leapfrog_steps is not None
    assert state.phase6_retry_anchor_source == "phase6_failed_candidate_nearest_tau"
    assert state.payload()["phase6_retry_num_leapfrog_steps"] == (
        state.phase6_retry_num_leapfrog_steps
    )


def test_frozen_step_trajectory_soft_deadline_writes_partial_closeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, calls = _scripted_trajectory_runner({})
    ticks = iter([0.0, 0.0, 0.0, 100.0, 100.0, 740.0, 740.0, 740.0, 740.0])
    events: list[tuple[str, Mapping[str, Any]]] = []

    def fake_perf_counter() -> float:
        try:
            return next(ticks)
        except StopIteration:
            return 740.0

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(public_timeout_budget_s=810.0),
        run_full_chain=run,
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    assert len(calls) == 1
    assert result.final_status == "repair_or_retry"
    assert "acceptance_outside_pass_band" in result.repair_triggers
    assert result.diagnostics["expected_candidate_count"] == 1
    assert result.diagnostics["completed_candidate_count"] == 1
    assert result.diagnostics["skipped_candidate_count"] == 0
    assert result.diagnostics["public_timeout_closeout"] is None
    assert events[-1][0] == "trajectory_candidate_call_complete"
    event_extra = events[-1][1]["extra"]
    assert event_extra["completed_candidate_count"] == 1
    text = json.dumps(event_extra, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "num_leapfrog_steps",
        "step_size",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_phase6_soft_deadline_estimate_uses_recent_candidate_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", lambda: 100.0)

    no_veto = _phase6_next_candidate_soft_deadline_veto(
        stage_start_perf_counter_s=0.0,
        timeout_budget_s=200.0,
        completed_elapsed_s=(200.0, 12.0, 12.0, 12.0),
    )

    assert no_veto is None

    closeout = _phase6_next_candidate_soft_deadline_veto(
        stage_start_perf_counter_s=0.0,
        timeout_budget_s=200.0,
        completed_elapsed_s=(200.0, 80.0, 12.0, 12.0),
    )

    assert closeout is not None
    assert closeout["estimated_next_candidate_s"] == pytest.approx(100.0)
    assert closeout["completed_candidate_elapsed_count"] == 4
    assert closeout["completed_candidate_elapsed_recent_window"] == 3
    assert (
        closeout["completed_candidate_elapsed_estimator"]
        == "recent_max_times_safety_multiplier"
    )
    text = json.dumps(closeout, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "num_leapfrog_steps",
        "step_size",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_frozen_step_trajectory_global_soft_deadline_vetoes_before_first_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, calls = _scripted_trajectory_runner({})
    events: list[tuple[str, Mapping[str, Any]]] = []

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", lambda: 800.0)

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(
            public_timeout_budget_s=810.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=run,
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    assert calls == []
    assert result.final_status == "hard_veto"
    assert "phase6_public_timeout_soft_deadline" in result.hard_vetoes
    assert result.diagnostics["expected_candidate_count"] == 1
    assert result.diagnostics["completed_candidate_count"] == 0
    assert result.diagnostics["skipped_candidate_count"] == 1
    closeout = result.diagnostics["public_timeout_closeout"]
    assert closeout["deadline_clock_scope"] == "public_one_call_global"
    assert closeout["elapsed_s"] == pytest.approx(800.0)
    assert closeout["remaining_s"] == pytest.approx(10.0)
    assert closeout["stage_elapsed_s"] == pytest.approx(0.0)
    assert closeout["stage_remaining_s"] == pytest.approx(810.0)
    assert closeout["candidate_index"] == 0
    assert closeout["completed_candidate_count"] == 0
    assert closeout["closeout_required_before_next_candidate"] is True
    assert events[-1][0] == "trajectory_candidate_soft_deadline_closeout"
    event_extra = events[-1][1]["extra"]
    assert event_extra["soft_deadline"]["deadline_clock_scope"] == "public_one_call_global"
    assert event_extra["soft_deadline"]["stage_elapsed_s"] == pytest.approx(0.0)
    text = json.dumps(event_extra, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "num_leapfrog_steps",
        "step_size",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in text


def test_frozen_step_trajectory_stage_rejects_stale_phase5_lineage() -> None:
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    with pytest.raises(ValueError, match="requires passed Phase 5"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=replace(step_stage, final_status="hard_veto"),
            config=_trajectory_config(),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )

    with pytest.raises(ValueError, match="selected step hash mismatch"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=replace(step_stage, selected_step_hash="stale-step"),
            config=_trajectory_config(),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )

    missing_step_hash = replace(step_stage)
    object.__setattr__(missing_step_hash, "selected_step_hash", None)
    with pytest.raises(ValueError, match="requires selected step hash"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=missing_step_hash,
            config=_trajectory_config(),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )

    mismatched_l_payload = {
        **dict(step_stage.selected_step_payload),
        "num_leapfrog_steps": step_stage.fixed_num_leapfrog_steps + 1,
    }
    with pytest.raises(ValueError, match="selected step hash mismatch"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=replace(
                step_stage,
                selected_step_payload=mismatched_l_payload,
            ),
            config=_trajectory_config(),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )

    mismatched_l_payload_hash = hmc_kernel_tuning.stable_config_hash(
        mismatched_l_payload
    )
    with pytest.raises(ValueError, match="selected L payload mismatch"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=replace(
                step_stage,
                selected_step_payload=mismatched_l_payload,
                selected_step_hash=mismatched_l_payload_hash,
            ),
            config=_trajectory_config(),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )

    with pytest.raises(ValueError, match="selected bootstrap kernel hash mismatch"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=replace(
                step_stage,
                selected_bootstrap_kernel_hash="stale-bootstrap-hash",
            ),
            config=_trajectory_config(),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )


def test_frozen_step_trajectory_stage_rejects_target_scope_and_adapted_mass_mismatch() -> None:
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    with pytest.raises(ValueError, match="target_scope mismatch"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            fixed_mass_step_stage=step_stage,
            config=_trajectory_config(target_scope="wrong_scope"),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )

    windowed = _windowed_stage()
    bad_windowed_result = replace(windowed.windowed_mass_result)
    object.__setattr__(
        bad_windowed_result,
        "final_mass_artifact_signature",
        "stale-adapted-mass-signature",
    )
    bad_windowed = replace(windowed, windowed_mass_result=bad_windowed_result)
    step_stage = replace(
        _fixed_mass_step_stage(windowed_stage=windowed),
        windowed_stage_artifact_hash=bad_windowed.artifact_hash,
    )

    with pytest.raises(ValueError, match="Phase 4 adapted mass signature mismatch"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=bad_windowed,
            fixed_mass_step_stage=step_stage,
            config=_trajectory_config(),
            run_full_chain=_scripted_trajectory_runner({})[0],
        )


def test_frozen_step_trajectory_stage_rejects_adapter_and_artifact_lineage_mismatches() -> None:
    _expect_frozen_step_validation_error(
        "adapter signature mismatch",
        adapter=_MismatchedAdapter(),
    )

    _expect_frozen_step_validation_error(
        "Phase 4 adapter signature mismatch",
        windowed_stage=replace(
            _windowed_stage(),
            adapter_signature="stale-phase4-adapter-signature",
        ),
    )

    bad_geometry = _geometry()
    object.__setattr__(
        bad_geometry,
        "adapter_signature",
        "stale-geometry-adapter-signature",
    )
    _expect_frozen_step_validation_error(
        "geometry adapter signature mismatch",
        geometry=bad_geometry,
    )

    _expect_frozen_step_validation_error(
        "bootstrap adapter signature mismatch",
        bootstrap=replace(
            _bootstrap(),
            adapter_signature="stale-bootstrap-adapter-signature",
        ),
    )

    _expect_frozen_step_validation_error(
        "geometry artifact mismatch",
        windowed_stage=replace(
            _windowed_stage(),
            geometry_artifact_hash="stale-geometry-artifact-hash",
        ),
    )

    _expect_frozen_step_validation_error(
        "bootstrap artifact mismatch",
        windowed_stage=replace(
            _windowed_stage(),
            bootstrap_artifact_hash="stale-bootstrap-artifact-hash",
        ),
    )

    _expect_frozen_step_validation_error(
        "Phase 5 windowed artifact mismatch",
        fixed_mass_step_stage=replace(
            _fixed_mass_step_stage(),
            windowed_stage_artifact_hash="stale-windowed-stage-artifact-hash",
        ),
    )

    bad_windowed = replace(
        _windowed_stage(),
        selected_bootstrap_kernel_hash="stale-phase4-selected-bootstrap-hash",
    )
    step_stage = replace(
        _fixed_mass_step_stage(),
        windowed_stage_artifact_hash=bad_windowed.artifact_hash,
    )
    _expect_frozen_step_validation_error(
        "Phase 4 selected bootstrap hash mismatch",
        windowed_stage=bad_windowed,
        fixed_mass_step_stage=step_stage,
    )


def test_frozen_step_trajectory_stage_rejects_missing_selected_payloads_and_ladder_lineage() -> None:
    bootstrap = _bootstrap()
    object.__setattr__(bootstrap, "selected_round_index", None)
    windowed = _windowed_stage()
    object.__setattr__(windowed, "bootstrap_artifact_hash", bootstrap.artifact_hash)
    object.__setattr__(windowed, "selected_bootstrap_kernel_hash", None)
    step_stage = _fixed_mass_step_stage()
    object.__setattr__(step_stage, "selected_bootstrap_kernel_hash", None)
    object.__setattr__(step_stage, "windowed_stage_artifact_hash", windowed.artifact_hash)
    _expect_frozen_step_validation_error(
        "selected bootstrap kernel hash mismatch",
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
    )

    windowed, missing_step_payload = _windowed_and_step_stage()
    object.__setattr__(missing_step_payload, "selected_step_payload", None)
    _expect_frozen_step_validation_error(
        "requires selected step payload",
        windowed_stage=windowed,
        fixed_mass_step_stage=missing_step_payload,
    )

    windowed, missing_ladder = _windowed_and_step_stage()
    object.__setattr__(missing_ladder, "budget_ladder_result", None)
    _expect_frozen_step_validation_error(
        "requires Phase 5 budget ladder result",
        windowed_stage=windowed,
        fixed_mass_step_stage=missing_ladder,
    )

    windowed, failed_ladder = _windowed_and_step_stage()
    object.__setattr__(
        failed_ladder.budget_ladder_result,
        "selected_round_index",
        None,
    )
    _expect_frozen_step_validation_error(
        "requires passed Phase 5 ladder",
        windowed_stage=windowed,
        fixed_mass_step_stage=failed_ladder,
    )

    windowed, step_stage = _windowed_and_step_stage()
    bad_ladder = replace(
        step_stage.budget_ladder_result,
        mass_artifact_signature="stale-ladder-mass-signature",
    )
    _expect_frozen_step_validation_error(
        "ladder mass signature mismatch",
        windowed_stage=windowed,
        fixed_mass_step_stage=replace(step_stage, budget_ladder_result=bad_ladder),
    )

    windowed, step_stage = _windowed_and_step_stage()
    ladder = step_stage.budget_ladder_result
    rounds = list(ladder.rounds)
    selected_index = int(ladder.selected_round_index)
    rounds[selected_index] = replace(
        rounds[selected_index],
        tuned_step_size=float(rounds[selected_index].tuned_step_size) * 2.0,
    )
    bad_ladder = replace(ladder, rounds=tuple(rounds))
    _expect_frozen_step_validation_error(
        "ladder selected step hash mismatch",
        windowed_stage=windowed,
        fixed_mass_step_stage=replace(step_stage, budget_ladder_result=bad_ladder),
    )

    windowed, step_stage = _windowed_and_step_stage()
    bad_ladder = replace(
        step_stage.budget_ladder_result,
        adapter_signature="stale-ladder-adapter-signature",
    )
    object.__setattr__(step_stage, "budget_ladder_result", bad_ladder)
    object.__setattr__(step_stage, "selected_step_payload", bad_ladder.selected_config_payload)
    object.__setattr__(step_stage, "selected_step_hash", bad_ladder.selected_config_hash)
    _expect_frozen_step_validation_error(
        "ladder adapter signature mismatch",
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
    )

    windowed, step_stage = _windowed_and_step_stage()
    bad_ladder = replace(
        step_stage.budget_ladder_result,
        hmc_adapter_signature="stale-ladder-hmc-adapter-signature",
    )
    object.__setattr__(step_stage, "budget_ladder_result", bad_ladder)
    object.__setattr__(step_stage, "selected_step_payload", bad_ladder.selected_config_payload)
    object.__setattr__(step_stage, "selected_step_hash", bad_ladder.selected_config_hash)
    _expect_frozen_step_validation_error(
        "ladder HMC adapter signature mismatch",
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
    )


def test_frozen_step_trajectory_stage_rejects_dimension_mismatches() -> None:
    windowed, step_stage = _windowed_and_step_stage()
    _expect_frozen_step_validation_error(
        "target dimension mismatch",
        windowed_stage=windowed,
        fixed_mass_step_stage=replace(step_stage, target_dimension=3),
    )

    bad_geometry = _geometry()
    object.__setattr__(bad_geometry, "target_dimension", 3)
    windowed = _windowed_stage()
    object.__setattr__(windowed, "geometry_artifact_hash", bad_geometry.artifact_hash)
    step_stage = _fixed_mass_step_stage()
    object.__setattr__(step_stage, "windowed_stage_artifact_hash", windowed.artifact_hash)
    _expect_frozen_step_validation_error(
        "geometry dimension mismatch",
        geometry=bad_geometry,
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
    )


def test_frozen_step_trajectory_candidate_generation_clamps_to_internal_bounds() -> None:
    low = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=0.1),
        selected_step_size=0.2,
        fixed_bootstrap_l=3,
    )
    high = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=100.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=3,
    )

    assert low["candidate_l_values"] == (3,)
    assert high["candidate_l_values"] == (3, 25)
    assert low["internal_min_leapfrog"] == 3
    assert high["internal_max_leapfrog"] == 25
    assert high["local_max_leapfrog"] == 25


def test_frozen_step_trajectory_candidate_generation_preserves_distinct_clamped_edges() -> None:
    low = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=0.1),
        selected_step_size=0.2,
        fixed_bootstrap_l=5,
    )
    high = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=100.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=24,
    )
    mixed = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=25.6),
        selected_step_size=0.2,
        fixed_bootstrap_l=3,
    )

    assert low["raw_candidate_l_values"] == (-1, 0, 1, 2, 3, 5, 1)
    assert low["candidate_l_values"] == (3, 5)
    assert high["raw_candidate_l_values"] == (498, 499, 500, 501, 502, 24, 150)
    assert high["candidate_l_values"] == (24, 25)
    assert mixed["raw_candidate_l_values"] == (126, 127, 128, 129, 130, 3, 39)
    assert mixed["candidate_l_values"] == (3, 25)


def test_frozen_step_trajectory_candidate_generation_caps_bootstrap_l_at_center_plus_ten() -> None:
    generated = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=2.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=128,
    )

    assert generated["center_candidate_l"] == 10
    assert generated["local_max_leapfrog"] == 20
    assert generated["candidate_l_values"] == (3, 8, 9, 10, 11, 12, 20)


def test_frozen_step_trajectory_candidate_generation_expands_after_verification_repair() -> None:
    repaired_state = _HMCPhaseAttemptState(
        selected_step_size=0.2,
        selected_step_hash="previous-step-hash",
        selected_num_leapfrog_steps=11,
        selected_trajectory_hash="previous-trajectory-hash",
        verification_acceptance_rate=0.89,
        verification_acceptance_relation="above_acceptance_band",
        verification_repair_trigger="verification_acceptance_outside_pass_band",
        verification_repair_source="phase7_final_verification_acceptance",
        verification_repair_step_size=0.4,
        verification_repair_step_hash="repair-step-hash",
        verification_repair_applied=True,
        handoff_stage="phase6",
    )
    unrepaired_state = _HMCPhaseAttemptState(
        selected_step_size=0.2,
        selected_step_hash="previous-step-hash",
        selected_num_leapfrog_steps=11,
        selected_trajectory_hash="previous-trajectory-hash",
        handoff_stage="phase6",
    )

    repaired = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=1.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=23,
        attempt_state=repaired_state,
    )
    unrepaired = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=1.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=23,
        attempt_state=unrepaired_state,
    )

    assert unrepaired["verification_repair_neighborhood_applied"] is False
    assert unrepaired["candidate_l_values"] == (3, 4, 5, 6, 7, 11, 15)
    assert repaired["verification_repair_neighborhood_applied"] is True
    expected_candidate_set = (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    assert repaired["candidate_set_order_before_directional_repair"] == expected_candidate_set
    assert set(repaired["candidate_l_values"]) == set(expected_candidate_set)
    assert repaired["candidate_l_values"][:3] == (15, 14, 13)
    assert repaired["candidate_order"] == (
        "directional_high_acceptance_prioritize_longer_in_window"
    )


def test_frozen_step_trajectory_candidate_generation_expands_after_phase6_retry_anchor() -> None:
    repaired_state = _HMCPhaseAttemptState(
        selected_step_size=0.2,
        selected_step_hash="previous-step-hash",
        selected_num_leapfrog_steps=None,
        selected_trajectory_hash=None,
        phase6_retry_num_leapfrog_steps=11,
        phase6_retry_anchor_source="phase6_failed_candidate_nearest_tau",
        verification_acceptance_rate=0.89,
        verification_acceptance_relation="above_acceptance_band",
        verification_repair_trigger="phase6_trajectory_acceptance_outside_pass_band",
        verification_repair_source="phase6_frozen_step_trajectory_acceptance",
        verification_repair_step_size=0.4,
        verification_repair_step_hash="repair-step-hash",
        verification_repair_applied=True,
        handoff_stage="phase5_selected",
    )

    repaired = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=1.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=23,
        attempt_state=repaired_state,
    )

    assert repaired["previous_attempt_center_l"] == 11
    assert repaired["previous_selected_num_leapfrog_steps"] is None
    assert repaired["phase6_retry_num_leapfrog_steps"] == 11
    assert repaired["phase6_retry_anchor_source"] == "phase6_failed_candidate_nearest_tau"
    assert repaired["previous_l_anchor_role"] == "phase6_retry_anchor"
    assert repaired["verification_repair_neighborhood_applied"] is True
    expected_candidate_set = (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    assert repaired["candidate_set_order_before_directional_repair"] == expected_candidate_set
    assert set(repaired["candidate_l_values"]) == set(expected_candidate_set)
    assert repaired["candidate_l_values"][:3] == (15, 14, 13)
    assert repaired["candidate_order"] == (
        "directional_high_acceptance_prioritize_longer_in_window"
    )


def test_frozen_step_trajectory_high_acceptance_repair_prioritizes_longer_candidates() -> None:
    repaired_state = _HMCPhaseAttemptState(
        selected_step_size=0.2,
        selected_step_hash="previous-step-hash",
        selected_num_leapfrog_steps=11,
        selected_trajectory_hash="previous-trajectory-hash",
        verification_acceptance_rate=0.89,
        verification_acceptance_relation="above_acceptance_band",
        verification_repair_trigger="phase6_trajectory_acceptance_outside_pass_band",
        verification_repair_source="phase6_frozen_step_trajectory_inside_window_acceptance",
        verification_repair_step_size=0.4,
        verification_repair_step_hash="repair-step-hash",
        verification_repair_applied=True,
        handoff_stage="phase6",
    )

    repaired = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=1.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=23,
        attempt_state=repaired_state,
    )

    assert repaired["candidate_set_order_before_directional_repair"] == (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    assert repaired["candidate_l_values"][:2] == (15, 14)
    assert repaired["candidate_order"] == (
        "directional_high_acceptance_prioritize_longer_in_window"
    )


def test_frozen_step_trajectory_low_acceptance_repair_prioritizes_shorter_candidates() -> None:
    repaired_state = _HMCPhaseAttemptState(
        selected_step_size=0.2,
        selected_step_hash="previous-step-hash",
        selected_num_leapfrog_steps=11,
        selected_trajectory_hash="previous-trajectory-hash",
        verification_acceptance_rate=0.50,
        verification_acceptance_relation="below_acceptance_band",
        verification_repair_trigger="phase6_trajectory_acceptance_outside_pass_band",
        verification_repair_source="phase6_frozen_step_trajectory_inside_window_acceptance",
        verification_repair_step_size=0.1,
        verification_repair_step_hash="repair-step-hash",
        verification_repair_applied=True,
        handoff_stage="phase6",
    )

    repaired = _frozen_step_trajectory_candidate_generation(
        geometry=replace(_geometry(), target_trajectory_length=1.0),
        selected_step_size=0.2,
        fixed_bootstrap_l=23,
        attempt_state=repaired_state,
    )

    assert repaired["candidate_set_order_before_directional_repair"] == (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
    )
    assert repaired["candidate_l_values"][:2] == (3, 4)
    assert repaired["candidate_order"] == (
        "directional_low_acceptance_prioritize_shorter_in_window"
    )


def test_frozen_step_trajectory_directional_order_remains_private() -> None:
    repaired_state = _HMCPhaseAttemptState(
        selected_step_size=0.2,
        selected_step_hash="previous-step-hash",
        selected_num_leapfrog_steps=11,
        selected_trajectory_hash="previous-trajectory-hash",
        verification_acceptance_rate=0.89,
        verification_acceptance_relation="above_acceptance_band",
        verification_repair_trigger="phase6_trajectory_acceptance_outside_pass_band",
        verification_repair_source="phase6_frozen_step_trajectory_inside_window_acceptance",
        verification_repair_step_size=0.4,
        verification_repair_step_hash="repair-step-hash",
        verification_repair_applied=True,
        handoff_stage="phase6",
    )
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    run, calls = _scripted_trajectory_runner({})

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(max_leapfrog_steps=25),
        run_full_chain=run,
        _attempt_state=repaired_state,
    )
    public_summary = hmc_kernel_tuning._frozen_step_trajectory_public_summary(result)

    assert calls
    assert result.candidate_generation["candidate_order"] == "selected_pair_only"
    assert result.candidate_generation["no_second_frozen_epsilon_l_search"] is True
    public_text = json.dumps(public_summary, sort_keys=True)
    for forbidden in (
        "candidate_l_values",
        "candidate_set_order_before_directional_repair",
        "num_leapfrog_steps",
        "step_size",
    ):
        assert forbidden not in public_text


def test_frozen_step_trajectory_stage_hard_veto_on_nonfinite_candidate_diagnostic() -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
            finite_log_accept=False,
        )

    windowed = _windowed_stage()
    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=_fixed_mass_step_stage(windowed_stage=windowed),
        config=_trajectory_config(),
        run_full_chain=run,
    )

    assert result.final_status == "hard_veto"
    assert "trajectory_log_accept_nonfinite_or_missing" in result.hard_vetoes
    assert result.selected_trajectory_payload is None


def test_frozen_step_trajectory_stage_callback_continuation_veto_is_hard_veto() -> None:
    windowed = _windowed_stage()
    step_stage = _fixed_mass_step_stage(windowed_stage=windowed)
    center = int(
        np.ceil(
            _geometry().target_trajectory_length
            / step_stage.selected_step_size
        )
    )
    run, _calls = _scripted_trajectory_runner({center: 0.70})

    def callback(
        _round_payload: Mapping[str, Any],
        _samples: Any,
        _diagnostics: Mapping[str, Any],
    ) -> FixedMassHMCTuningBudgetCallbackResult:
        return FixedMassHMCTuningBudgetCallbackResult(
            continuation_vetoes=("artifact_cannot_answer_question",),
        )

    result = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=windowed,
        fixed_mass_step_stage=step_stage,
        config=_trajectory_config(),
        screen_callback=callback,
        run_full_chain=run,
    )

    assert result.final_status == "hard_veto"
    assert "frozen_step_trajectory_continuation_veto" in result.hard_vetoes
    assert "artifact_cannot_answer_question" in result.hard_vetoes
    assert result.selected_trajectory_payload is None


def test_frozen_step_trajectory_stage_public_exports_are_scoped_without_final_tuner() -> None:
    assert bayesfilter.HMCFrozenStepTrajectoryStageConfig is HMCFrozenStepTrajectoryStageConfig
    assert bayesfilter.HMCFrozenStepTrajectoryStageResult is HMCFrozenStepTrajectoryStageResult
    assert bayesfilter.run_hmc_frozen_step_trajectory_stage is run_hmc_frozen_step_trajectory_stage
    assert bayesfilter.FROZEN_STEP_TRAJECTORY_STAGE_NONCLAIMS is FROZEN_STEP_TRAJECTORY_STAGE_NONCLAIMS
    assert hasattr(bayesfilter, "tune_hmc_kernel")
    assert "tune_hmc_kernel" in bayesfilter.__all__
