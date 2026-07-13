from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning
from bayesfilter.inference import (
    FixedMassHMCTuningBudgetCallbackResult,
    HMCBootstrapScreenResult,
    HMCFixedMassStepStageConfig,
    HMCFixedMassStepStageResult,
    HMCGeometryInitializationConfig,
    HMCWindowedMassStageConfig,
    HMCWindowedMassStageResult,
    ValueScoreCapability,
    initialize_hmc_kernel_geometry,
    run_hmc_bootstrap_screen,
    run_hmc_fixed_mass_step_stage,
    run_hmc_frozen_step_trajectory_stage,
    run_hmc_windowed_mass_stage,
)


class _ToyGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "kernel-fixed-mass-step-toy-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_kernel_tuning_fixed_mass_step.py",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
            nonclaims=("tiny fixed-mass step fixture only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value


class _MismatchedAdapter(_ToyGaussianAdapter):
    def adapter_signature(self) -> str:
        return "kernel-fixed-mass-step-mismatched-v1"


@dataclass(frozen=True)
class _FakeRunResult:
    samples: Any
    trace: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _geometry(**overrides: Any):
    payload = {
        "adapter": _ToyGaussianAdapter(),
        "initial_position": np.zeros(2),
        "config": HMCGeometryInitializationConfig(
            geometry_scaling_c=0.5,
            stability_guard=0.8,
            covariance_jitter=0.0,
            seed=(123, 456),
        ),
    }
    payload.update(overrides)
    return initialize_hmc_kernel_geometry(**payload)


def _runtime_metadata() -> Mapping[str, Any]:
    return {
        "runtime": "tfp.mcmc.sample_chain",
        "sample_chain_invocation_count": 1,
        "sample_chain_call_s": 0.01,
        "trace_unavailability": {},
        "fixture_or_synthetic": False,
        "nonclaims": (
            "deterministic hmc contract plumbing result",
            "no sampler convergence claim",
            "no posterior validity claim",
        ),
    }


def _warmup_draws(warmup_steps: int = 12) -> np.ndarray:
    base = np.array(
        [
            [-0.20, 0.10],
            [-0.10, 0.00],
            [0.10, 0.20],
            [0.20, 0.10],
            [0.30, 0.30],
            [0.40, 0.25],
            [0.50, 0.40],
            [0.60, 0.35],
            [0.70, 0.50],
            [0.80, 0.45],
            [0.90, 0.55],
            [1.00, 0.60],
        ],
        dtype=float,
    )
    return base[:warmup_steps]


def _fake_result(
    *,
    num_results: int,
    acceptance: float = 0.70,
    step_size: float | None = None,
    num_adaptation_steps: int | None = None,
    samples: Any | None = None,
    finite_log_accept: bool = True,
    finite_samples: bool = True,
    finite_target_log_prob: bool = True,
    metadata_overrides: Mapping[str, Any] | None = None,
) -> _FakeRunResult:
    if samples is None:
        sample_array = _warmup_draws(max(num_results, 1))[:num_results]
    else:
        sample_array = np.asarray(samples, dtype=float)
    if not finite_samples:
        sample_array = sample_array.copy()
        sample_array.reshape(-1)[-1] = np.nan
    acceptance_count = int(round(float(acceptance) * int(num_results)))
    acceptance_trace = np.zeros(int(num_results), dtype=bool)
    acceptance_trace[:acceptance_count] = True
    log_accept = np.linspace(-0.2, 0.1, int(num_results))
    if not finite_log_accept:
        log_accept[-1] = np.nan
    target_log_prob = -0.5 * np.sum(np.square(sample_array), axis=-1)
    if not finite_target_log_prob:
        target_log_prob[-1] = np.nan
    trace: dict[str, Any] = {
        "is_accepted": tf.constant(acceptance_trace, dtype=tf.bool),
        "log_accept_ratio": tf.constant(log_accept, dtype=tf.float64),
        "target_log_prob": tf.constant(target_log_prob, dtype=tf.float64),
    }
    diagnostics: dict[str, Any] = {
        "acceptance_rate": tf.constant(float(acceptance), dtype=tf.float64),
        "finite_sample_count": tf.constant(
            int(np.sum(np.all(np.isfinite(sample_array), axis=-1))),
            dtype=tf.int32,
        ),
        "nonfinite_sample_count": tf.constant(
            int(np.sum(~np.all(np.isfinite(sample_array), axis=-1))),
            dtype=tf.int32,
        ),
        "trace_policy": "standard",
    }
    if step_size is not None:
        diagnostics["final_step_size"] = tf.constant(float(step_size), dtype=tf.float64)
        diagnostics["final_step_size_finite"] = tf.constant(
            bool(np.isfinite(step_size))
        )
        trace["step_size"] = tf.constant([float(step_size)], dtype=tf.float64)
    if num_adaptation_steps is not None:
        diagnostics["num_adaptation_steps"] = tf.constant(
            int(num_adaptation_steps),
            dtype=tf.int32,
        )
        diagnostics["target_accept_prob"] = tf.constant(0.70, dtype=tf.float64)
        trace["num_adaptation_steps"] = tf.constant(
            [int(num_adaptation_steps)],
            dtype=tf.int32,
        )
    metadata = dict(_runtime_metadata())
    if metadata_overrides is not None:
        metadata.update(dict(metadata_overrides))
    return _FakeRunResult(
        samples=tf.constant(sample_array, dtype=tf.float64),
        trace=trace,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def _bootstrap() -> HMCBootstrapScreenResult:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    return run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        run_full_chain=run,
    )


def _stage_config(**overrides: Any) -> HMCFixedMassStepStageConfig:
    payload = {
        "target_accept_prob": 0.70,
        "seed": (20260621, 50),
        "chain_execution_mode": "eager",
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
    }
    payload.update(overrides)
    return HMCFixedMassStepStageConfig(**payload)


def _windowed_config(**overrides: Any) -> HMCWindowedMassStageConfig:
    payload = {
        "target_accept_prob": 0.70,
        "seed": (20260621, 40),
        "chain_execution_mode": "eager",
        "target_scope": "kernel_fixed_mass_step_toy_gaussian",
    }
    payload.update(overrides)
    return HMCWindowedMassStageConfig(**payload)


def _windowed_stage() -> HMCWindowedMassStageResult:
    bootstrap = _bootstrap()

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=_warmup_draws(int(config.num_results)),
        )

    return run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=bootstrap,
        config=_windowed_config(),
        run_full_chain=run,
    )


def _scripted_step_runner(screen_acceptances: list[float] | Mapping[int, float]):
    calls: list[Mapping[str, Any]] = []
    acceptance_sequence = (
        None if isinstance(screen_acceptances, Mapping) else list(screen_acceptances)
    )
    acceptance_by_l = (
        {int(key): float(value) for key, value in screen_acceptances.items()}
        if isinstance(screen_acceptances, Mapping)
        else None
    )

    def run(adapter: Any, initial_state: Any, config: Any) -> _FakeRunResult:
        uses_tuning = bool(config.tuning_policy.uses_dual_averaging)
        leapfrog = int(config.num_leapfrog_steps)
        calls.append(
            {
                "role": "tune" if uses_tuning else "screen",
                "num_results": int(config.num_results),
                "num_burnin_steps": int(config.num_burnin_steps),
                "step_size": float(config.step_size),
                "num_leapfrog_steps": leapfrog,
                "seed": tuple(config.seed),
                "adapter_signature": adapter.adapter_signature(),
                "initial_state": np.asarray(initial_state, dtype=float),
                "use_xla": bool(config.use_xla),
            }
        )
        np.testing.assert_allclose(np.asarray(initial_state, dtype=float), np.zeros(2))
        if uses_tuning:
            target_tau = np.pi / 2.0
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                step_size=target_tau / float(leapfrog),
                num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
                samples=np.zeros((int(config.num_results), 2)),
            )
        if acceptance_by_l is not None:
            acceptance = acceptance_by_l.get(leapfrog, 0.82)
        else:
            assert acceptance_sequence is not None
            acceptance = acceptance_sequence.pop(0)
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=acceptance,
            samples=np.zeros((int(config.num_results), 2)),
        )

    return run, calls


def test_fixed_mass_step_config_does_not_expose_hmc_mechanics() -> None:
    parameters = set(inspect.signature(HMCFixedMassStepStageConfig).parameters)
    forbidden = {
        "budget_schedule",
        "tune_num_results",
        "screen_num_results",
        "screen_num_burnin_steps",
        "step_size",
        "initial_step_size",
        "num_leapfrog_steps",
        "max_leapfrog_steps",
        "min_leapfrog",
        "max_leapfrog",
        "trajectory_grid",
        "candidate_grid",
        "mass_window_schedule",
        "warmup_steps",
    }

    assert parameters.isdisjoint(forbidden)


def test_joint_l_epsilon_candidate_outside_tau_window_is_not_viable() -> None:
    class Round:
        tuned_step_size = 10.0
        budget = 3
        screen_diagnostics = {"acceptance_rate": 0.70}
        hard_vetoes = ()
        continuation_vetoes = ()
        repair_triggers = ()

    class Ladder:
        selected_round = Round()
        selected_round_index = 0
        final_status = "passed"
        passed = True
        rounds = (Round(),)
        artifact_hash = "fake-ladder-hash"

        def payload(self):
            return {"schema": "fake-ladder"}

    candidate = hmc_kernel_tuning._joint_l_epsilon_ladder_candidate_payload(
        round_index=0,
        grid_stage="test",
        candidate_index=0,
        num_leapfrog_steps=8,
        ladder=Ladder(),
        target_trajectory=1.5707963267948968,
        target_accept_prob=0.70,
        trajectory_window_lower_multiplier=0.8,
        trajectory_window_upper_multiplier=1.25,
        max_leapfrog_steps=8,
    )

    assert candidate["trajectory_length"] == pytest.approx(80.0)
    assert candidate["trajectory_window_relation"] == "above_trajectory_window"
    assert (
        candidate["trajectory_window_policy_role"]
        == "engineering_viability_gate_non_scientific"
    )
    assert candidate["trajectory_window_viability_gate_active"] is True
    assert candidate["viable"] is False
    assert "trajectory_length_outside_window" in candidate["repair_triggers"]
    assert "trajectory_length_above_window" in candidate["repair_triggers"]
    assert candidate["reports_posterior_convergence"] is False
    assert candidate["reports_sampler_superiority"] is False


def test_joint_l_epsilon_candidate_outside_tau_window_can_nominate_under_phase23() -> None:
    class Round:
        tuned_step_size = 10.0
        budget = 3
        screen_diagnostics = {"acceptance_rate": 0.70}
        hard_vetoes = ()
        continuation_vetoes = ()
        repair_triggers = ()

    class Ladder:
        selected_round = Round()
        selected_round_index = 0
        final_status = "passed"
        passed = True
        rounds = (Round(),)
        artifact_hash = "fake-ladder-hash"

        def payload(self):
            return {"schema": "fake-ladder"}

    candidate = hmc_kernel_tuning._joint_l_epsilon_ladder_candidate_payload(
        round_index=0,
        grid_stage="test",
        candidate_index=0,
        num_leapfrog_steps=8,
        ladder=Ladder(),
        target_trajectory=1.5707963267948968,
        target_accept_prob=0.70,
        trajectory_window_lower_multiplier=0.8,
        trajectory_window_upper_multiplier=1.25,
        max_leapfrog_steps=8,
        handoff_screen_policy="phase23_nomination_only",
    )

    assert candidate["handoff_screen_policy"] == "phase23_nomination_only"
    assert candidate["trajectory_window_relation"] == "above_trajectory_window"
    assert candidate["trajectory_window_viability_gate_active"] is False
    assert candidate["trajectory_window_nomination_only"] is True
    assert candidate["viable"] is False
    assert candidate["nomination_eligible"] is True
    assert (
        candidate["nomination_role"]
        == "phase23_candidate_for_phase7_verification"
    )
    assert "trajectory_length_outside_window" in candidate["repair_triggers"]
    assert candidate["reports_posterior_convergence"] is False
    assert candidate["reports_sampler_superiority"] is False


def test_joint_l_epsilon_phase23_selection_uses_nomination_key() -> None:
    candidates = (
        {
            "nomination_eligible": True,
            "screen_acceptance_rate": 0.70,
            "selected_step_size": 1.0,
            "trajectory_length": 10.0,
            "trajectory_window_relation": "above_trajectory_window",
            "selected_budget": 5,
            "num_leapfrog_steps": 10,
            "candidate_index": 0,
            "round_index": 0,
            "hard_vetoes": (),
            "continuation_vetoes": (),
            "viable": False,
        },
        {
            "nomination_eligible": True,
            "screen_acceptance_rate": 0.70,
            "selected_step_size": 1.0,
            "trajectory_length": 2.0,
            "trajectory_window_relation": "inside_trajectory_window",
            "selected_budget": 5,
            "num_leapfrog_steps": 2,
            "candidate_index": 1,
            "round_index": 0,
            "hard_vetoes": (),
            "continuation_vetoes": (),
            "viable": True,
        },
        {
            "nomination_eligible": False,
            "screen_acceptance_rate": 0.70,
            "selected_step_size": 1.0,
            "trajectory_length": 1.0,
            "trajectory_window_relation": "inside_trajectory_window",
            "selected_budget": 1,
            "num_leapfrog_steps": 1,
            "candidate_index": 2,
            "round_index": 0,
            "hard_vetoes": ("fixed_mass_step_ladder_error",),
            "continuation_vetoes": (),
            "viable": False,
        },
    )

    assert (
        hmc_kernel_tuning._select_joint_l_epsilon_candidate(
            candidates,
            target_accept_prob=0.70,
            target_trajectory=2.0,
            handoff_screen_policy="phase23_nomination_only",
        )
        == 1
    )
    assert (
        hmc_kernel_tuning._select_joint_l_epsilon_candidate(
            candidates,
            target_accept_prob=0.70,
            target_trajectory=2.0,
        )
        == 1
    )


def test_fixed_mass_step_stage_passes_with_frozen_mass_and_internal_budget() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    assert windowed.passed is True
    run, calls = _scripted_step_runner({3: 0.82, 4: 0.66, 5: 0.70, 7: 0.73})

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(),
        run_full_chain=run,
    )

    assert isinstance(result, HMCFixedMassStepStageResult)
    assert result.passed is True
    assert result.final_status == "passed"
    assert result.diagnostics["algorithm"] == "joint_l_epsilon_grid_fixed_mass_hmc"
    assert result.diagnostics["promoted_default"] is True
    assert result.diagnostics["final_local_grid_ran"] is True
    tune_l = [call["num_leapfrog_steps"] for call in calls if call["role"] == "tune"]
    screen_l = [call["num_leapfrog_steps"] for call in calls if call["role"] == "screen"]
    assert set(tune_l).issubset(set(screen_l))
    assert all(leapfrog in tune_l for leapfrog in {3, 4, 5, 6, 8})
    assert set(result.diagnostics["round_summaries"][0]["candidate_l_values"]) == {
        3,
        4,
        5,
        6,
        8,
    }
    assert set(result.diagnostics["round_summaries"][-1]["candidate_l_values"]) == {
        3,
        4,
        5,
        6,
        7,
    }
    assert result.fixed_num_leapfrog_steps == 5
    assert result.selected_step_payload["num_leapfrog_steps"] == 5
    assert result.selected_step_size == pytest.approx((np.pi / 2.0) / 5.0)
    assert all(call["step_size"] == pytest.approx(windowed.candidate_step_size) for call in calls if call["role"] == "tune")
    assert all(call["use_xla"] is False for call in calls)
    assert result.initial_step_size == pytest.approx(windowed.candidate_step_size)
    assert result.budget_ladder_config_payload["budget_schedule"] == (3, 6, 12)
    assert result.budget_ladder_config_payload["tune_num_results"] == 4
    assert result.budget_ladder_config_payload["screen_num_results"] == 4
    assert result.frozen_mass_invariant["passed"] is True
    assert result.frozen_mass_invariant["mass_update_allowed"] is False


def test_fixed_mass_step_private_callback_records_candidates_and_selected_pair() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.66, 5: 0.70, 7: 0.73})
    events: list[tuple[str, Mapping[str, Any]]] = []

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(),
        run_full_chain=run,
        _private_diagnostic_callback=lambda event_type, payload: events.append(
            (event_type, dict(payload))
        ),
    )

    assert result.passed is True
    candidate_events = [
        payload
        for event_type, payload in events
        if event_type == "joint_l_epsilon_candidate_complete"
    ]
    selected_events = [
        payload
        for event_type, payload in events
        if event_type == "joint_l_epsilon_round_selected"
    ]
    assert candidate_events
    assert selected_events
    assert candidate_events[-1]["candidate_completed_count"] == (
        candidate_events[-1]["candidate_count"]
    )
    assert any(payload["step_size"] is not None for payload in candidate_events)
    assert all(payload["num_leapfrog_steps"] > 0 for payload in candidate_events)
    assert all(payload["private_hmc_mechanics"] is True for payload in candidate_events)
    final_selection = selected_events[-1]
    assert final_selection["selected_pair_exists"] is True
    assert final_selection["num_leapfrog_steps"] == result.fixed_num_leapfrog_steps
    assert final_selection["step_size"] == pytest.approx(result.selected_step_size)
    assert final_selection["grid_stage"] == "final_local"
    assert final_selection["reports_posterior_convergence"] is False


def test_fixed_mass_step_config_use_xla_propagates_to_tune_and_screen_configs() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    run, calls = _scripted_step_runner({3: 0.70, 4: 0.70, 5: 0.70, 7: 0.70})

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(chain_execution_mode="tf_function", use_xla=True),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.config.payload()["use_xla"] is True
    assert calls
    assert all(call["use_xla"] is True for call in calls)
    assert result.selected_step_payload is not None
    assert result.selected_step_hash
    assert result.selected_step_size == pytest.approx(
        result.budget_ladder_result.selected_round.tuned_step_size
    )
    assert result.payload()["reports_trajectory_tuning"] is False
    assert result.payload()["reports_posterior_convergence"] is False
    assert "each candidate leapfrog count gets its own epsilon tuning ladder" in result.nonclaims


def test_fixed_mass_step_stage_repairs_without_selected_step_when_budget_exhausts() -> None:
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.83, 5: 0.84, 7: 0.84})
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "repair_or_retry"
    assert result.selected_step_payload is None
    assert result.selected_step_hash is None
    assert result.selected_step_size is None
    assert result.repair_step_payload is not None
    assert result.repair_step_hash
    assert result.budget_ladder_result.repair_config_payload is not None
    assert result.repair_step_size == pytest.approx(
        result.budget_ladder_result.repair_config_payload["step_size"]
    )
    assert result.repair_step_size > (
        result.budget_ladder_result.last_finite_tuned_round.tuned_step_size
    )
    assert result.budget_ladder_result.final_status == "budget_exhausted"
    assert "acceptance_outside_pass_band_inside_repair_band" in result.repair_triggers
    assert "joint_l_epsilon_no_viable_pair" in result.repair_triggers
    payload = result.payload()
    assert payload["repair_step_available"] is True
    assert payload["repair_step_payload_exposed"] is False
    assert "repair_step_payload" not in payload
    with pytest.raises(ValueError, match="requires passed Phase 5"):
        run_hmc_frozen_step_trajectory_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=geometry,
            bootstrap=bootstrap,
            windowed_stage=windowed,
            fixed_mass_step_stage=result,
            run_full_chain=_scripted_step_runner([0.70])[0],
        )


def test_fixed_mass_step_public_timeout_preflight_skips_ladder_without_mechanics() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    progress_events: list[tuple[str, Mapping[str, Any]]] = []

    def forbidden_run(*_args: Any, **_kwargs: Any) -> _FakeRunResult:
        raise AssertionError("fixed-mass ladder runner should be skipped")

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(
            public_timeout_budget_s=10.0,
            public_timeout_started_perf_counter_s=hmc_kernel_tuning.time.perf_counter()
            - 9.5,
        ),
        run_full_chain=forbidden_run,
        _progress_callback=lambda stage, payload: progress_events.append(
            (stage, dict(payload))
        ),
        _attempt_index=2,
    )

    assert result.final_status == "hard_veto"
    assert result.passed is False
    assert (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_HARD_VETO
        in result.hard_vetoes
    )
    assert (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_REPAIR_TRIGGER
        in result.repair_triggers
    )
    assert "joint_l_epsilon_no_viable_pair" not in result.repair_triggers
    assert result.selected_step_payload is None
    assert result.budget_ladder_result is None
    closeout = result.diagnostics["public_timeout_closeout"]
    assert closeout["schema"] == "bayesfilter.fixed_mass_step_public_timeout_closeout.v1"
    assert closeout["closeout_required_before_next_candidate"] is True
    assert closeout["candidate_index"] == 0
    assert closeout["completed_candidate_count"] == 0
    assert closeout["public_closeout_artifact_expected"] is True
    assert closeout["hmc_mechanics_exposed"] is False
    assert result.diagnostics["candidate_count"] == 0

    public_summary = hmc_kernel_tuning._stage_status_public_summary(result)
    public_closeout = public_summary["public_timeout_closeout"]
    assert public_closeout["hard_veto"] == (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_HARD_VETO
    )
    assert public_closeout["repair_trigger"] == (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_REPAIR_TRIGGER
    )
    assert public_closeout["hmc_mechanics_exposed"] is False
    assert progress_events
    progress_text = json.dumps(progress_events, sort_keys=True)
    public_text = json.dumps(public_summary, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "ladder_payload",
        "candidate_l_values",
    ):
        assert forbidden not in progress_text
        assert forbidden not in public_text


def test_fixed_mass_step_timeout_after_selected_pair_is_budget_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    clock = {"now": 0.0}
    run, calls = _scripted_step_runner({3: 0.82, 4: 0.82, 5: 0.82, 6: 0.82, 8: 0.70})
    progress_events: list[tuple[str, Mapping[str, Any]]] = []
    timed_screen_l: set[int] = set()

    def fake_perf_counter() -> float:
        return float(clock["now"])

    def timed_run(adapter: Any, initial_state: Any, config: Any) -> _FakeRunResult:
        result = run(adapter, initial_state, config)
        leapfrog = int(config.num_leapfrog_steps)
        if (
            not bool(config.tuning_policy.uses_dual_averaging)
            and leapfrog not in timed_screen_l
        ):
            timed_screen_l.add(leapfrog)
            clock["now"] += 12.0
        return result

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", fake_perf_counter)

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(
            public_timeout_budget_s=130.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        run_full_chain=timed_run,
        _max_leapfrog_steps=12,
        _progress_callback=lambda stage, payload: progress_events.append(
            (stage, dict(payload))
        ),
        _attempt_index=2,
    )

    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_BUDGET_INCOMPLETE_ROLE
    )
    assert result.passed is False
    assert result.hard_vetoes == ()
    assert (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_HARD_VETO
        not in result.hard_vetoes
    )
    assert (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_BUDGET_INCOMPLETE_REPAIR_TRIGGER
        in result.repair_triggers
    )
    assert result.selected_step_payload is None
    assert result.selected_step_hash is None
    assert result.repair_step_payload is None
    assert result.repair_step_hash is None
    assert result.payload()["repair_step_available"] is False
    assert result.diagnostics["viable_candidate_count"] >= 1
    assert result.diagnostics["public_timeout_budget_incomplete"] is True
    assert (
        result.diagnostics["selected_pair_progress_before_timeout_closeout"] is True
    )
    assert result.diagnostics["final_local_grid_ran"] is False

    closeout = result.diagnostics["public_timeout_closeout"]
    assert closeout["grid_stage"] == "edge_repair"
    assert closeout["candidate_index"] == 0
    assert closeout["completed_candidate_count"] == 0
    assert closeout["selected_pair_progress_before_closeout"] is True
    assert closeout["budget_incomplete"] is True
    assert closeout["budget_incomplete_scope"] == (
        "edge_or_final_local_after_selected_pair_progress"
    )
    assert closeout["diagnostic_role"] == (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_BUDGET_INCOMPLETE_ROLE
    )
    assert "hard_veto" not in closeout
    assert closeout["hmc_mechanics_exposed"] is False

    public_summary = hmc_kernel_tuning._stage_status_public_summary(result)
    public_closeout = public_summary["public_timeout_closeout"]
    assert public_closeout["budget_incomplete"] is True
    assert public_closeout["selected_pair_progress_before_closeout"] is True
    assert public_closeout["repair_trigger"] == (
        hmc_kernel_tuning._FIXED_MASS_STEP_PUBLIC_TIMEOUT_BUDGET_INCOMPLETE_REPAIR_TRIGGER
    )
    assert "hard_veto" not in public_closeout
    assert public_closeout["hmc_mechanics_exposed"] is False
    public_text = json.dumps(public_summary, sort_keys=True)
    progress_text = json.dumps(progress_events, sort_keys=True)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
        "ladder_payload",
        "candidate_l_values",
    ):
        assert forbidden not in public_text
        assert forbidden not in progress_text
    called_l = {int(call["num_leapfrog_steps"]) for call in calls}
    assert called_l.issubset({3, 4, 5, 6, 8})


def test_fixed_mass_step_stage_edge_selected_l_repairs_grid_and_final_local_retunes() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    run, calls = _scripted_step_runner(
        {
            3: 0.82,
            4: 0.82,
            5: 0.82,
            6: 0.82,
            8: 0.65,
            9: 0.70,
            10: 0.70,
            11: 0.82,
            12: 0.82,
        }
    )

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(),
        run_full_chain=run,
        _max_leapfrog_steps=12,
    )

    assert result.passed is True
    summaries = result.diagnostics["round_summaries"]
    assert [summary["grid_stage"] for summary in summaries] == [
        "initial",
        "edge_repair",
        "final_local",
    ]
    assert summaries[0]["edge_direction"] == "upper"
    assert summaries[1]["anchor_l"] == 12
    assert summaries[2]["candidate_l_values"] == (8, 9, 10, 11, 12)
    assert result.diagnostics["edge_repair_round_count"] == 1
    assert result.diagnostics["final_local_grid_ran"] is True
    assert result.fixed_num_leapfrog_steps == 9
    tune_l = [call["num_leapfrog_steps"] for call in calls if call["role"] == "tune"]
    assert 12 in tune_l
    assert tune_l.count(10) >= 2


def test_fixed_mass_step_stage_private_repair_handoff_preserves_bracket_state_without_public_leak() -> None:
    screen_acceptances = [0.9375, 0.125, 0.78125]
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        uses_tuning = bool(config.tuning_policy.uses_dual_averaging)
        if uses_tuning:
            return _fake_result(
                num_results=int(config.num_results),
                acceptance=0.70,
                step_size=0.2,
                num_adaptation_steps=config.tuning_policy.num_adaptation_steps,
                samples=np.zeros((int(config.num_results), 2)),
            )
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=screen_acceptances.pop(0),
            samples=np.zeros((int(config.num_results), 2)),
        )

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.final_status == "repair_or_retry"
    assert result.repair_step_payload is not None
    bracket_state = result.repair_step_payload["fixed_mass_bracket_state"]
    assert bracket_state["bracketed"] is True
    assert bracket_state["next_step_size"] == pytest.approx(result.repair_step_size)
    assert bracket_state["private_handoff_only"] is True

    attempt_state = hmc_kernel_tuning._phase7_attempt_state_from_stages(
        config=hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig(
            target_accept_prob=0.70,
            acceptance_band=(0.65, 0.75),
            repair_band=(0.55, 0.85),
            max_attempts=2,
            seed=(20260621, 70),
            chain_execution_mode="eager",
            target_scope="kernel_fixed_mass_step_toy_gaussian",
        ),
        windowed_stage=windowed,
        fixed_mass_step_stage=result,
        frozen_step_trajectory_stage=None,
    )

    assert attempt_state.handoff_stage == "phase5_repair"
    assert attempt_state.fixed_mass_bracket_state == bracket_state
    attempt_payload = attempt_state.payload()
    assert attempt_payload["fixed_mass_bracket_state_available"] is True
    assert attempt_payload["fixed_mass_bracket_state"] == bracket_state

    public_summary = hmc_kernel_tuning._stage_status_public_summary(result)
    public_text = json.dumps(public_summary, sort_keys=True)
    for forbidden in (
        "fixed_mass_bracket_state",
        "high_acceptance_step_lower_bound",
        "low_acceptance_step_upper_bound",
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "trace",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in public_text


def test_fixed_mass_step_stage_requires_passed_phase4() -> None:
    windowed = replace(_windowed_stage(), final_status="hard_veto")

    with pytest.raises(ValueError, match="requires passed Phase 4"):
        run_hmc_fixed_mass_step_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=windowed,
            config=_stage_config(),
            run_full_chain=_scripted_step_runner([0.70])[0],
        )


def test_fixed_mass_step_stage_rejects_signature_and_lineage_mismatch() -> None:
    with pytest.raises(ValueError, match="adapter signature"):
        run_hmc_fixed_mass_step_stage(
            adapter=_MismatchedAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=_windowed_stage(),
            config=_stage_config(),
            run_full_chain=_scripted_step_runner([0.70])[0],
        )

    bad_windowed = replace(
        _windowed_stage(),
        selected_bootstrap_kernel_hash="stale-selected-kernel",
    )
    with pytest.raises(ValueError, match="selected bootstrap kernel mismatch"):
        run_hmc_fixed_mass_step_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=bad_windowed,
            config=_stage_config(),
            run_full_chain=_scripted_step_runner([0.70])[0],
        )


def test_fixed_mass_step_stage_rejects_adapted_mass_signature_mismatch() -> None:
    windowed = _windowed_stage()
    bad_windowed_result = replace(windowed.windowed_mass_result)
    object.__setattr__(
        bad_windowed_result,
        "final_mass_artifact_signature",
        "stale-adapted-mass-signature",
    )
    bad_windowed = replace(windowed, windowed_mass_result=bad_windowed_result)

    with pytest.raises(ValueError, match="Phase 4 adapted mass signature mismatch"):
        run_hmc_fixed_mass_step_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            windowed_stage=bad_windowed,
            config=_stage_config(),
            run_full_chain=_scripted_step_runner([0.70])[0],
        )


def test_fixed_mass_step_stage_callback_roles_are_preserved() -> None:
    run, _calls = _scripted_step_runner([0.70])

    def callback(
        _round_payload: Mapping[str, Any],
        _samples: Any,
        _diagnostics: Mapping[str, Any],
    ) -> FixedMassHMCTuningBudgetCallbackResult:
        return FixedMassHMCTuningBudgetCallbackResult(
            continuation_vetoes=("artifact_cannot_answer_question",),
        )

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=_windowed_stage(),
        config=_stage_config(),
        screen_callback=callback,
        run_full_chain=run,
    )

    assert result.final_status == "hard_veto"
    assert "fixed_mass_step_continuation_veto" in result.hard_vetoes
    assert result.diagnostics["continuation_vetoes"] == (
        "artifact_cannot_answer_question",
    )
    assert result.selected_step_payload is None


def test_fixed_mass_step_stage_hard_veto_on_nonfinite_tune_diagnostic() -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        uses_tuning = bool(config.tuning_policy.uses_dual_averaging)
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            step_size=np.nan if uses_tuning else None,
            num_adaptation_steps=(
                config.tuning_policy.num_adaptation_steps if uses_tuning else None
            ),
            samples=np.zeros((int(config.num_results), 2)),
        )

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=_windowed_stage(),
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.final_status == "hard_veto"
    assert "tune_final_step_missing_or_nonfinite" in result.hard_vetoes
    assert result.selected_step_payload is None


def test_fixed_mass_step_stage_public_exports_are_scoped_without_final_tuner() -> None:
    assert bayesfilter.HMCFixedMassStepStageConfig is HMCFixedMassStepStageConfig
    assert bayesfilter.run_hmc_fixed_mass_step_stage is run_hmc_fixed_mass_step_stage
    assert hasattr(bayesfilter, "tune_hmc_kernel")
    assert "tune_hmc_kernel" in bayesfilter.__all__


def test_phase5_private_candidate_handoff_preserves_identity_selection_and_payload() -> None:
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.66, 5: 0.70, 7: 0.73})
    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=_windowed_stage(),
        config=_stage_config(),
        run_full_chain=run,
    )
    without_handoff = replace(result, _candidate_batch_handoff=None)

    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(result)
    assert handoff is not None
    assert handoff.candidate_count == result.diagnostics["candidate_count"]
    assert tuple(record.batch_ordinal for record in handoff.candidate_records) == tuple(
        range(handoff.candidate_count)
    )
    source_keys = tuple(record.source_key for record in handoff.candidate_records)
    assert len(set(source_keys)) == len(source_keys)
    round_local_indices = tuple(
        record.source_round_candidate_index for record in handoff.candidate_records
    )
    assert len(set(round_local_indices)) < len(round_local_indices)

    assert handoff.selected_batch_ordinal is not None
    selected = handoff.candidate_records[handoff.selected_batch_ordinal]
    assert selected.record_hash == handoff.selected_record_hash
    assert selected.handoff_eligible is True
    assert selected.source_grid_stage == "final_local"
    assert selected.num_leapfrog_steps == result.fixed_num_leapfrog_steps
    assert selected.selected_step_size == result.selected_step_size
    assert selected.selected_step_hash == result.selected_step_hash
    assert selected.ladder_artifact_hash == result.budget_ladder_result.artifact_hash
    assert handoff.repair_batch_ordinal is None
    assert handoff.verification_order_seed[0] == handoff.selected_batch_ordinal

    assert result.payload() == without_handoff.payload()
    assert result.artifact_hash == without_handoff.artifact_hash
    assert repr(result) == repr(without_handoff)
    assert result == without_handoff
    assert "_candidate_batch_handoff" not in result.payload()
    assert handoff.handoff_hash not in json.dumps(result.payload(), sort_keys=True)
    assert handoff.handoff_hash not in json.dumps(
        hmc_kernel_tuning._stage_status_public_summary(result),
        sort_keys=True,
    )
    assert not any(name.startswith("_HMCPhase5Candidate") for name in bayesfilter.__all__)
    assert "_phase5_candidate_batch_handoff" not in bayesfilter.__all__

    original_record_payload = selected.payload()
    original_record_hash = selected.record_hash
    original_handoff_hash = handoff.handoff_hash
    source_candidate = result.diagnostics["candidates"][selected.batch_ordinal]
    assert isinstance(source_candidate, dict)
    source_candidate["selected_step_size"] = 999.0
    source_candidate["hard_vetoes"] = ("mutated_after_handoff",)
    assert selected.payload() == original_record_payload
    assert selected.record_hash == original_record_hash
    assert handoff.handoff_hash == original_handoff_hash


def test_phase5_private_candidate_handoff_anchors_repair_to_exact_ladder() -> None:
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.83, 5: 0.84, 7: 0.84})
    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=_windowed_stage(),
        config=_stage_config(),
        run_full_chain=run,
    )

    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(result)
    assert handoff is not None
    assert result.final_status == "repair_or_retry"
    assert handoff.selected_batch_ordinal is None
    assert handoff.repair_batch_ordinal is not None
    repair = handoff.candidate_records[handoff.repair_batch_ordinal]
    assert repair.record_hash == handoff.repair_record_hash
    assert repair.repair_config_hash == result.repair_step_hash
    assert repair.num_leapfrog_steps == result.repair_step_payload["num_leapfrog_steps"]
    assert repair.ladder_artifact_hash == result.budget_ladder_result.artifact_hash
    repairable = tuple(
        record for record in handoff.candidate_records if record.repair_config_hash
    )
    assert len(repairable) > 1
    assert sum(
        record.ladder_artifact_hash == result.budget_ladder_result.artifact_hash
        for record in repairable
    ) == 1

    wrong_repair = next(
        record
        for record in repairable
        if record.batch_ordinal != repair.batch_ordinal
    )
    cross_wired = replace(
        handoff,
        repair_batch_ordinal=wrong_repair.batch_ordinal,
        repair_record_hash=wrong_repair.record_hash,
        handoff_hash="",
    )
    with pytest.raises(ValueError, match="repair candidate"):
        hmc_kernel_tuning._phase5_candidate_batch_handoff(
            replace(result, _candidate_batch_handoff=cross_wired)
        )

    anchorless = replace(
        handoff,
        repair_batch_ordinal=None,
        repair_record_hash=None,
        handoff_hash="",
    )
    with pytest.raises(ValueError, match="lost candidate provenance"):
        hmc_kernel_tuning._phase5_candidate_batch_handoff(
            replace(result, _candidate_batch_handoff=anchorless)
        )


def test_phase5_private_candidate_handoff_projects_phase23_nomination_policy() -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        uses_tuning = bool(config.tuning_policy.uses_dual_averaging)
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            step_size=10.0 if uses_tuning else None,
            num_adaptation_steps=(
                config.tuning_policy.num_adaptation_steps if uses_tuning else None
            ),
            samples=np.zeros((int(config.num_results), 2)),
        )

    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=_windowed_stage(),
        config=_stage_config(
            handoff_screen_policy="phase23_nomination_only",
            trajectory_window_lower_multiplier=0.8,
            trajectory_window_upper_multiplier=1.25,
        ),
        run_full_chain=run,
    )

    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(result)
    assert handoff is not None
    assert result.final_status == "passed"
    selected = handoff.candidate_records[handoff.selected_batch_ordinal]
    assert selected.trajectory_window_relation == "above_trajectory_window"
    assert selected.viable is False
    assert selected.nomination_eligible is True
    assert selected.handoff_eligible is True
    assert all(
        record.handoff_eligible == record.nomination_eligible
        for record in handoff.candidate_records
    )


def test_phase5_private_candidate_signals_retain_sources_and_hashes_fail_closed() -> None:
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.66, 5: 0.70, 7: 0.73})
    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=_windowed_stage(),
        config=_stage_config(),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(result)
    assert handoff is not None
    first = replace(
        handoff.candidate_records[0],
        hard_vetoes=("same_code", "same_code"),
        continuation_vetoes=(),
        repair_triggers=(),
        record_hash="",
    )
    second = replace(
        handoff.candidate_records[1],
        hard_vetoes=("same_code",),
        continuation_vetoes=(),
        repair_triggers=(),
        record_hash="",
    )
    signals = hmc_kernel_tuning._phase5_candidate_signal_records((first, second))
    assert tuple(signal.payload() for signal in signals) == (
        {
            "role": "hard_veto",
            "code": "same_code",
            "source_batch_ordinal": first.batch_ordinal,
        },
        {
            "role": "hard_veto",
            "code": "same_code",
            "source_batch_ordinal": second.batch_ordinal,
        },
    )

    changed_l = replace(
        handoff.candidate_records[0],
        num_leapfrog_steps=handoff.candidate_records[0].num_leapfrog_steps + 1,
        record_hash="",
    )
    assert changed_l.record_hash != handoff.candidate_records[0].record_hash

    selected = handoff.candidate_records[handoff.selected_batch_ordinal]
    changed_step = replace(
        selected,
        selected_step_size=selected.selected_step_size + 0.01,
        record_hash="",
    )
    changed_source = replace(
        selected,
        source_round_candidate_index=selected.source_round_candidate_index + 100,
        record_hash="",
    )
    changed_ladder = replace(
        selected,
        ladder_artifact_hash="changed-ladder-hash",
        record_hash="",
    )
    assert len(
        {
            selected.record_hash,
            changed_step.record_hash,
            changed_source.record_hash,
            changed_ladder.record_hash,
        }
    ) == 4

    first_signal = handoff.signal_records[0]
    changed_signal = replace(
        first_signal,
        source_batch_ordinal=(first_signal.source_batch_ordinal + 1)
        % handoff.candidate_count,
    )
    changed_signal_handoff = replace(
        handoff,
        signal_records=(changed_signal, *handoff.signal_records[1:]),
        handoff_hash="",
    )
    assert changed_signal_handoff.handoff_hash != handoff.handoff_hash
    with pytest.raises(ValueError, match="signal provenance mismatch"):
        hmc_kernel_tuning._phase5_candidate_batch_handoff(
            replace(result, _candidate_batch_handoff=changed_signal_handoff)
        )

    duplicate_ordinal = replace(
        handoff.candidate_records[1],
        batch_ordinal=handoff.candidate_records[0].batch_ordinal,
        record_hash="",
    )
    bad_identity = replace(
        handoff,
        candidate_records=(
            handoff.candidate_records[0],
            duplicate_ordinal,
            *handoff.candidate_records[2:],
        ),
        handoff_hash="",
    )
    with pytest.raises(ValueError, match="ordinals must be contiguous"):
        hmc_kernel_tuning._phase5_candidate_batch_handoff(
            replace(result, _candidate_batch_handoff=bad_identity)
        )

    count_mismatch = replace(
        result,
        diagnostics={
            **result.diagnostics,
            "candidate_count": result.diagnostics["candidate_count"] + 1,
        },
    )
    with pytest.raises(ValueError, match="count does not match"):
        hmc_kernel_tuning._phase5_candidate_batch_handoff(count_mismatch)

    corrupted = replace(handoff, handoff_hash="")
    object.__setattr__(corrupted, "handoff_hash", "corrupted")
    with pytest.raises(ValueError, match="handoff hash mismatch"):
        hmc_kernel_tuning._phase5_candidate_batch_handoff(
            replace(result, _candidate_batch_handoff=corrupted)
        )


def test_phase5_private_candidate_normalization_distinguishes_missing_and_nonfinite() -> None:
    missing = hmc_kernel_tuning._phase5_normalize_optional_number({}, "value")
    nonfinite = hmc_kernel_tuning._phase5_normalize_optional_number(
        {"value": np.nan},
        "value",
    )
    assert missing == (None, False, False)
    assert nonfinite == (None, True, False)

    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.66, 5: 0.70, 7: 0.73})
    result = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        windowed_stage=_windowed_stage(),
        config=_stage_config(),
        run_full_chain=run,
    )
    handoff = hmc_kernel_tuning._phase5_candidate_batch_handoff(result)
    assert handoff is not None
    record = handoff.candidate_records[0]
    missing_record = replace(
        record,
        selected_step_size=None,
        selected_step_size_observed=False,
        selected_step_size_finite=False,
        selected_step_hash=None,
        handoff_eligible=False,
        record_hash="",
    )
    nonfinite_record = replace(
        missing_record,
        selected_step_size_observed=True,
        record_hash="",
    )
    assert missing_record.record_hash != nonfinite_record.record_hash
    assert "NaN" not in json.dumps(nonfinite_record.payload(), sort_keys=True)
    assert "Infinity" not in json.dumps(nonfinite_record.payload(), sort_keys=True)


def test_phase5_private_candidate_handoff_preserves_historical_phase6_entry() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    windowed = _windowed_stage()
    run, _calls = _scripted_step_runner({3: 0.82, 4: 0.66, 5: 0.70, 7: 0.73})
    fixed = run_hmc_fixed_mass_step_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        config=_stage_config(),
        run_full_chain=run,
    )

    def phase6_run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _fake_result(
            num_results=int(config.num_results),
            acceptance=0.70,
            samples=np.zeros((int(config.num_results), 2)),
        )

    trajectory = run_hmc_frozen_step_trajectory_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        windowed_stage=windowed,
        fixed_mass_step_stage=fixed,
        run_full_chain=phase6_run,
    )

    assert hmc_kernel_tuning._phase5_candidate_batch_handoff(fixed) is not None
    assert trajectory.passed is True
    assert trajectory.fixed_mass_step_stage_artifact_hash == fixed.artifact_hash
