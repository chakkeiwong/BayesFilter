from __future__ import annotations

import inspect
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    BOOTSTRAP_SCREEN_NONCLAIMS,
    HMCBootstrapScreenConfig,
    HMCBootstrapScreenResult,
    HMCGeometryInitializationConfig,
    PrecomputedMassArtifact,
    ValueScoreCapability,
    initialize_hmc_kernel_geometry,
    run_hmc_bootstrap_screen,
)


class _ToyGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "kernel-bootstrap-toy-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_kernel_tuning_bootstrap.py",
            target_scope="kernel_bootstrap_toy_gaussian",
            nonclaims=("tiny bootstrap fixture only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value


class _MismatchedAdapter(_ToyGaussianAdapter):
    def adapter_signature(self) -> str:
        return "kernel-bootstrap-mismatched-v1"


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


def _config(**overrides: Any) -> HMCBootstrapScreenConfig:
    payload = {
        "target_accept_prob": 0.70,
        "acceptance_band": (0.65, 0.75),
        "repair_band": (0.55, 0.85),
        "max_repairs": 5,
        "screen_num_results": 4,
        "screen_num_burnin_steps": 2,
        "seed": (20260621, 30),
        "chain_execution_mode": "eager",
        "target_scope": "kernel_bootstrap_toy_gaussian",
    }
    payload.update(overrides)
    return HMCBootstrapScreenConfig(**payload)


def _fake_result(
    *,
    acceptance: float,
    finite_log_accept: bool = True,
    finite_samples: bool = True,
    target_log_prob_finite: bool = True,
    runtime_s: float = 0.01,
    use_xla: bool = False,
    timing_scope: str = "unit_test_sample_chain_scope",
) -> _FakeRunResult:
    sample_value = 0.0 if finite_samples else np.nan
    samples = tf.constant([[sample_value, sample_value]], dtype=tf.float64)
    trace = {
        "is_accepted": tf.constant([acceptance >= 0.5, acceptance >= 0.25]),
        "log_accept_ratio": tf.constant(
            [0.0, 0.1 if finite_log_accept else np.nan],
            dtype=tf.float64,
        ),
        "target_log_prob": tf.constant(
            [0.0, -0.5 if target_log_prob_finite else np.nan],
            dtype=tf.float64,
        ),
    }
    diagnostics = {
        "acceptance_rate": tf.constant(float(acceptance), dtype=tf.float64),
        "finite_sample_count": tf.constant(2 if finite_samples else 0, dtype=tf.int32),
        "nonfinite_sample_count": tf.constant(0 if finite_samples else 2, dtype=tf.int32),
        "trace_policy": "standard",
    }
    return _FakeRunResult(
        samples=samples,
        trace=trace,
        diagnostics=diagnostics,
        metadata={
            "sample_chain_call_s": runtime_s,
            "sample_chain_timing_scope": timing_scope,
            "jit_compile": bool(use_xla),
            "use_xla": bool(use_xla),
            "trace_unavailability": {},
            "nonclaims": ("fake runner only",),
        },
    )


def _scripted_runner(acceptances: list[float]):
    calls: list[tuple[float, int, tuple[int, int], str, bool]] = []

    def run(adapter: Any, initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append(
            (
                float(config.step_size),
                int(config.num_leapfrog_steps),
                tuple(config.seed),
                adapter.adapter_signature(),
                bool(config.use_xla),
            )
        )
        np.testing.assert_allclose(initial_state.numpy(), np.zeros(2))
        acceptance = acceptances.pop(0)
        return _fake_result(acceptance=acceptance, use_xla=bool(config.use_xla))

    return run, calls


def test_bootstrap_config_does_not_expose_hmc_mechanics() -> None:
    parameters = set(inspect.signature(HMCBootstrapScreenConfig).parameters)
    forbidden = {
        "step_size",
        "initial_step_size",
        "num_leapfrog_steps",
        "min_leapfrog",
        "step_size_candidates",
        "num_leapfrog_step_candidates",
        "trajectory_grid",
        "mass_window_schedule",
        "budget_schedule",
    }

    assert parameters.isdisjoint(forbidden)
    assert "max_leapfrog_steps" in parameters


def test_geometry_initialization_caps_initial_l_at_policy_max() -> None:
    geometry = _geometry(
        config=HMCGeometryInitializationConfig(
            geometry_scaling_c=1.0e-6,
            stability_guard=0.8,
            covariance_jitter=0.0,
            seed=(123, 456),
        )
    )

    assert geometry.unclamped_num_leapfrog_steps > 25
    assert geometry.initial_num_leapfrog_steps == 25
    assert geometry.formula_report["internal_max_leapfrog"] == 25
    assert geometry.formula_report["leapfrog_clamped"] is True


def test_geometry_initialization_uses_configured_max_leapfrog_steps() -> None:
    geometry = _geometry(
        config=HMCGeometryInitializationConfig(
            geometry_scaling_c=1.0e-6,
            max_leapfrog_steps=40,
            covariance_jitter=0.0,
            seed=(123, 456),
        )
    )

    assert geometry.unclamped_num_leapfrog_steps > 40
    assert geometry.initial_num_leapfrog_steps == 40
    assert geometry.formula_report["internal_max_leapfrog"] == 40


def test_bootstrap_acceptance_in_acceptance_band_selects_kernel() -> None:
    run, calls = _scripted_runner([0.70])
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(),
        run_full_chain=run,
    )

    assert isinstance(result, HMCBootstrapScreenResult)
    assert result.passed is True
    assert result.final_status == "passed"
    assert result.selected_kernel_payload is not None
    assert result.selected_kernel_hash
    assert result.selected_kernel_payload["sample_space"] == "latent_fixed_mass"
    assert len(calls) == 1
    assert calls[0][0] == pytest.approx(_geometry().initial_step_size)
    assert calls[0][1] == _geometry().initial_num_leapfrog_steps
    assert calls[0][3] != _ToyGaussianAdapter().adapter_signature()
    assert calls[0][4] is False
    assert "no posterior convergence claim" in result.nonclaims


def test_bootstrap_config_use_xla_propagates_to_full_chain_config() -> None:
    run, calls = _scripted_runner([0.70])
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(chain_execution_mode="tf_function", use_xla=True),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.config.payload()["use_xla"] is True
    assert calls[0][4] is True


def test_bootstrap_default_tf_function_route_uses_reusable_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import hmc_kernel_tuning as module

    acceptances = [0.80, 0.70]
    build_calls: list[Mapping[str, Any]] = []
    run_calls: list[tuple[float, int, tuple[int, int], str]] = []

    class _FakeReusableRunner:
        def __init__(self, config: Any) -> None:
            self.config = config

        def run(
            self,
            *,
            current_state: Any | None = None,
            seed: Any | None = None,
            step_size: Any | None = None,
        ) -> _FakeRunResult:
            del current_state
            run_calls.append(
                (
                    float(step_size),
                    int(self.config.num_leapfrog_steps),
                    tuple(seed),
                    self.config.target_status_trace_policy,
                )
            )
            return _fake_result(
                acceptance=acceptances.pop(0),
                use_xla=bool(self.config.use_xla),
                timing_scope=(
                    "reusable_tf_function_xla_first_call_compile_plus_execute_then_warm_execute"
                ),
            )

    def fake_builder(adapter: Any, initial_state: Any, config: Any) -> _FakeReusableRunner:
        del adapter
        np.testing.assert_allclose(initial_state.numpy(), np.zeros(2))
        build_calls.append(config.signature_payload())
        return _FakeReusableRunner(config)

    monkeypatch.setattr(
        module,
        "build_reusable_full_chain_tfp_hmc_runner",
        fake_builder,
    )

    geometry = _geometry(
        config=HMCGeometryInitializationConfig(
            geometry_scaling_c=1000.0,
            stability_guard=1000.0,
            covariance_jitter=0.0,
            seed=(123, 456),
        )
    )
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(
            max_repairs=2,
            chain_execution_mode="tf_function",
            use_xla=True,
            target_status_trace_policy="none",
        ),
    )

    assert result.passed is True
    assert [round_result.classification for round_result in result.rounds] == [
        "repair",
        "passed",
    ]
    assert len(build_calls) == 1
    assert len(run_calls) == 2
    assert run_calls[0][0] == pytest.approx(geometry.initial_step_size)
    assert run_calls[1][0] == pytest.approx(geometry.initial_step_size * 2.0)
    assert run_calls[0][1] == run_calls[1][1] == geometry.initial_num_leapfrog_steps
    assert geometry.initial_num_leapfrog_steps == 3
    assert [call[2] for call in run_calls] == [(20260621, 30), (20260621, 31)]
    route = result.bootstrap_runner_route
    assert route["active_route"] == "bootstrap_scoped_reusable_runner"
    assert route["semantic_source"] == "run_hmc_bootstrap_screen"
    assert route["single_use_build_count_for_bootstrap_rounds"] == 0
    assert route["reusable_runner_build_count"] == 1
    assert route["distinct_static_runner_contract_count"] == 1
    assert route["bootstrap_round_count"] == 2
    assert route["fallback_to_single_use_runner"] is False
    assert route["fallback_status"] == "none"
    assert route["selected_kernel_preservation"]["preserved"] is True
    assert route["selected_kernel_payload_hash"] == result.selected_kernel_hash
    assert route["round_route_events"][0]["runner_reused"] is False
    assert route["round_route_events"][1]["runner_reused"] is True
    payload_route = result.payload()["bootstrap_runner_route"]
    assert payload_route["active_route"] == "bootstrap_scoped_reusable_runner"


def test_bootstrap_uses_configured_step_repair_factor() -> None:
    run, calls = _scripted_runner([0.90, 0.70])
    geometry = _geometry()
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(max_repairs=1, step_repair_factor=1.25),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.config.payload()["step_repair_factor"] == pytest.approx(1.25)
    assert calls[0][0] == pytest.approx(geometry.initial_step_size)
    assert calls[1][0] == pytest.approx(geometry.initial_step_size * 1.25)


def test_bootstrap_injected_tf_function_runner_does_not_use_reusable_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import hmc_kernel_tuning as module

    def fail_builder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("injected run_full_chain must bypass reusable runner")

    monkeypatch.setattr(
        module,
        "build_reusable_full_chain_tfp_hmc_runner",
        fail_builder,
    )
    run, calls = _scripted_runner([0.70])

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(chain_execution_mode="tf_function", use_xla=True),
        run_full_chain=run,
    )

    assert result.passed is True
    assert len(calls) == 1
    route = result.bootstrap_runner_route
    assert route["active_route"] == "single_use_or_injected_runner"
    assert route["single_use_build_count_for_bootstrap_rounds"] == 0
    assert route["injected_runner_call_count"] == 1
    assert route["reusable_runner_build_count"] == 0
    assert route["fallback_status"] == "inactive_reusable_route"


def test_bootstrap_reusable_route_builds_distinct_runner_for_leapfrog_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import hmc_kernel_tuning as module

    acceptances = [0.95, 0.70]
    build_calls: list[Mapping[str, Any]] = []

    class _FakeReusableRunner:
        def __init__(self, config: Any) -> None:
            self.config = config

        def run(self, **_kwargs: Any) -> _FakeRunResult:
            return _fake_result(
                acceptance=acceptances.pop(0),
                use_xla=bool(self.config.use_xla),
            )

    def fake_builder(_adapter: Any, _initial_state: Any, config: Any) -> _FakeReusableRunner:
        build_calls.append(config.signature_payload())
        return _FakeReusableRunner(config)

    monkeypatch.setattr(
        module,
        "build_reusable_full_chain_tfp_hmc_runner",
        fake_builder,
    )

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(
            max_repairs=2,
            chain_execution_mode="tf_function",
            use_xla=True,
        ),
    )

    assert result.passed is True
    assert [call["num_leapfrog_steps"] for call in build_calls] == [4, 3]
    route = result.bootstrap_runner_route
    assert route["active_route"] == "bootstrap_scoped_reusable_runner"
    assert route["single_use_build_count_for_bootstrap_rounds"] == 0
    assert route["reusable_runner_build_count"] == 2
    assert route["distinct_static_runner_contract_count"] == 2
    assert [event["runner_reused"] for event in route["round_route_events"]] == [
        False,
        False,
    ]


def test_bootstrap_diagnostics_distinguish_xla_and_non_xla_screen_metadata() -> None:
    non_xla = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(),
        run_full_chain=lambda _adapter, _initial_state, config: _fake_result(
            acceptance=0.70,
            use_xla=bool(config.use_xla),
        ),
    )
    xla = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(chain_execution_mode="tf_function", use_xla=True),
        run_full_chain=lambda _adapter, _initial_state, config: _fake_result(
            acceptance=0.70,
            use_xla=bool(config.use_xla),
        ),
    )

    non_xla_diagnostics = non_xla.rounds[0].diagnostics
    non_xla_checks = non_xla_diagnostics["screen_diagnostic"]["checks"]
    assert non_xla_diagnostics["use_xla"] is False
    assert non_xla_diagnostics["xla_requested"] is False
    assert non_xla_diagnostics["compile_chain_with_xla"] is False
    assert non_xla_diagnostics["jit_compile_metadata"] == "false"
    assert non_xla_checks["use_xla_false"] is True
    assert non_xla_checks["compile_chain_with_xla_false"] is True

    xla_diagnostics = xla.rounds[0].diagnostics
    xla_checks = xla_diagnostics["screen_diagnostic"]["checks"]
    assert xla_diagnostics["use_xla"] is True
    assert xla_diagnostics["xla_requested"] is True
    assert xla_diagnostics["compile_chain_with_xla"] is True
    assert xla_diagnostics["jit_compile_metadata"] == "true"
    assert "use_xla_false" not in xla_checks
    assert "compile_chain_with_xla_false" not in xla_checks
    assert xla.rounds[0].classification == "passed"


def test_acceptance_inside_repair_band_outside_acceptance_band_repairs() -> None:
    run, calls = _scripted_runner([0.80, 0.70])
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(max_repairs=2),
        run_full_chain=run,
    )

    assert result.passed is True
    assert [round_result.classification for round_result in result.rounds] == [
        "repair",
        "passed",
    ]
    assert result.rounds[0].repair_triggers == (
        "acceptance_above_acceptance_band_inside_repair_band",
        "increase_epsilon_recompute_l",
    )
    assert calls[1][0] == pytest.approx(calls[0][0] * 2.0)


def test_low_acceptance_repairs_by_reducing_epsilon_and_recomputing_l() -> None:
    run, calls = _scripted_runner([0.20, 0.70])
    geometry = _geometry()
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(max_repairs=2),
        run_full_chain=run,
    )

    assert result.passed is True
    assert [round_result.classification for round_result in result.rounds] == [
        "repair",
        "passed",
    ]
    assert result.rounds[0].repair_action == "reduce_epsilon_recompute_l"
    assert calls[1][0] == pytest.approx(calls[0][0] * 0.5)
    assert calls[1][1] == int(np.ceil(geometry.target_trajectory_length / calls[1][0]))
    assert result.rounds[1].seed != geometry.seed_report["geometry_seed"]


def test_high_acceptance_repairs_by_increasing_epsilon_and_recomputing_l() -> None:
    run, calls = _scripted_runner([0.95, 0.70])
    geometry = _geometry()
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(max_repairs=2),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.rounds[0].repair_action == "increase_epsilon_recompute_l"
    assert calls[1][0] == pytest.approx(calls[0][0] * 2.0)
    assert result.rounds[1].unclamped_num_leapfrog_steps == int(
        np.ceil(geometry.target_trajectory_length / calls[1][0])
    )
    assert calls[1][1] == 3
    assert result.rounds[1].clamp_direction == "min"


def test_oscillatory_acceptance_uses_bracketed_midpoint_and_replaces_endpoints() -> None:
    run, calls = _scripted_runner([0.20, 0.95, 0.20, 0.95, 0.20])
    geometry = _geometry()
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(max_repairs=4),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "repair_budget_exhausted"
    assert [round_result.repair_action for round_result in result.rounds] == [
        "reduce_epsilon_recompute_l",
        "bracketed_log_step_midpoint_recompute_l",
        "bracketed_log_step_midpoint_recompute_l",
        "bracketed_log_step_midpoint_recompute_l",
        "bracketed_log_step_midpoint_recompute_l",
    ]
    step0 = calls[0][0]
    assert calls[1][0] == pytest.approx(step0 * 0.5)
    assert calls[2][0] == pytest.approx(np.sqrt(calls[0][0] * calls[1][0]))
    assert calls[3][0] == pytest.approx(np.sqrt(calls[1][0] * calls[2][0]))
    assert calls[4][0] == pytest.approx(np.sqrt(calls[2][0] * calls[3][0]))
    assert np.log(calls[1][0]) < np.log(calls[2][0])
    assert calls[2][0] < calls[0][0]
    assert calls[3][0] > calls[1][0]
    assert calls[3][0] < calls[2][0]
    assert calls[4][0] > calls[3][0]
    assert calls[4][0] < calls[2][0]
    assert [call[1] for call in calls[1:]] == [
        int(np.ceil(geometry.target_trajectory_length / call[0]))
        for call in calls[1:]
    ]


@pytest.mark.parametrize(
    "result_kwargs, expected_veto",
    [
        ({"finite_log_accept": False}, "screen_log_accept_nonfinite_or_missing"),
        ({"finite_samples": False}, "screen_samples_nonfinite_or_missing"),
        (
            {"target_log_prob_finite": False},
            "screen_target_log_prob_nonfinite_or_missing",
        ),
    ],
)
def test_nonfinite_screen_diagnostics_hard_veto(
    result_kwargs: Mapping[str, Any],
    expected_veto: str,
) -> None:
    def run(_adapter: Any, _initial_state: Any, _config: Any) -> _FakeRunResult:
        return _fake_result(acceptance=0.70, **result_kwargs)

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "hard_veto"
    assert result.rounds[0].hard_vetoes == (expected_veto,)


def test_nonfinite_runtime_metadata_hard_vetoes() -> None:
    def run(_adapter: Any, _initial_state: Any, _config: Any) -> _FakeRunResult:
        return _fake_result(acceptance=0.70, runtime_s=np.nan)

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "hard_veto"
    assert result.rounds[0].hard_vetoes == ("screen_runtime_missing_or_nonfinite",)


def test_adapter_signature_mismatch_hard_fails_before_screen() -> None:
    with pytest.raises(ValueError, match="adapter signature"):
        run_hmc_bootstrap_screen(
            adapter=_MismatchedAdapter(),
            geometry=_geometry(),
            config=_config(),
            run_full_chain=_scripted_runner([0.70])[0],
        )


def test_mass_artifact_signature_mismatch_fails_closed() -> None:
    geometry = _geometry()
    bad_mass = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=2.0 * np.eye(2),
        adapter_signature=geometry.adapter_signature,
        position_role="initial_position",
        covariance_source="unit_test_bad_mass",
        source="unit_test_bad_mass",
        jitter=0.0,
    )
    bad_geometry = replace(geometry, mass_artifact=bad_mass)

    with pytest.raises(ValueError, match="mass artifact signature"):
        run_hmc_bootstrap_screen(
            adapter=_ToyGaussianAdapter(),
            geometry=bad_geometry,
            config=_config(),
            run_full_chain=_scripted_runner([0.70])[0],
        )


def test_repair_loop_stops_after_max_repairs() -> None:
    run, _calls = _scripted_runner([0.20, 0.20, 0.20])
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(max_repairs=2),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "repair_budget_exhausted"
    assert len(result.rounds) == 3


def test_repeated_leapfrog_cap_saturation_no_longer_blocks_step_progress() -> None:
    run, calls = _scripted_runner([0.20, 0.20, 0.20, 0.20, 0.20, 0.20])
    geometry = _geometry(
        config=HMCGeometryInitializationConfig(
            geometry_scaling_c=1000.0,
            stability_guard=1000.0,
            covariance_jitter=0.0,
            seed=(123, 456),
        )
    )

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(max_repairs=5),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "repair_budget_exhausted"
    assert len(result.rounds) == 6
    assert [round_result.clamp_direction for round_result in result.rounds] == [
        "min",
        "min",
        "min",
        "min",
        "min",
        "min",
    ]
    assert calls[1][0] == pytest.approx(calls[0][0] * 0.5)
    assert all(call[1] == 3 for call in calls)


def test_repeated_max_cap_saturation_continues_when_high_acceptance_repair_can_escape_cap() -> None:
    run, calls = _scripted_runner([0.95, 0.95, 0.70])
    geometry = _geometry(
        config=HMCGeometryInitializationConfig(
            geometry_scaling_c=0.02,
            stability_guard=0.8,
            covariance_jitter=0.0,
            seed=(123, 456),
        )
    )

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(max_repairs=3),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.final_status == "passed"
    assert [round_result.clamp_direction for round_result in result.rounds[:2]] == [
        "max",
        "max",
    ]
    assert result.rounds[2].classification == "passed"
    assert calls[1][0] == pytest.approx(calls[0][0] * 2.0)
    assert calls[2][0] == pytest.approx(calls[1][0] * 2.0)
    assert calls[0][1] == geometry.config.max_leapfrog_steps
    assert calls[1][1] == geometry.config.max_leapfrog_steps
    assert calls[2][1] < geometry.config.max_leapfrog_steps


def test_seed_report_and_artifact_hash_are_deterministic() -> None:
    kwargs = {
        "adapter": _ToyGaussianAdapter(),
        "geometry": _geometry(),
        "config": _config(seed=(10, 20)),
    }
    result_a = run_hmc_bootstrap_screen(
        **kwargs,
        run_full_chain=_scripted_runner([0.70])[0],
    )
    result_b = run_hmc_bootstrap_screen(
        **kwargs,
        run_full_chain=_scripted_runner([0.70])[0],
    )

    assert result_a.seed_report == result_b.seed_report
    assert result_a.artifact_hash == result_b.artifact_hash
    assert result_a.seed_report["screen_round_seeds"] == ((10, 20),)
    assert result_a.seed_report["geometry_seed_distinct_from_bootstrap_seed"] is True


def test_bootstrap_progress_callback_emits_public_safe_call_boundaries() -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []

    def callback(stage: str, payload: Mapping[str, Any]) -> None:
        events.append((stage, dict(payload)))

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        assert events[-1][0] == "bootstrap_round_hmc_call_start"
        assert events[-1][1]["bootstrap_diagnostic_screen_num_results"] == int(
            config.num_results
        )
        assert events[-1][1]["bootstrap_diagnostic_screen_num_burnin_steps"] == int(
            config.num_burnin_steps
        )
        return _fake_result(acceptance=0.70)

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(max_repairs=0),
        run_full_chain=run,
        progress_callback=callback,
    )

    assert result.passed is True
    stages = [stage for stage, _payload in events]
    assert stages == [
        "bootstrap_round_start",
        "bootstrap_round_hmc_call_start",
        "bootstrap_round_hmc_call_complete",
        "bootstrap_round_classified",
    ]
    final_payload = events[-1][1]
    assert "screen_num_results" not in final_payload
    assert "screen_num_burnin_steps" not in final_payload
    assert final_payload["bootstrap_diagnostic_screen_num_results"] == 4
    assert final_payload["bootstrap_diagnostic_screen_num_burnin_steps"] == 2
    assert final_payload["classification"] == "passed"
    assert final_payload["diagnostic_role"] == "bootstrap_screen_promotion_only"
    assert final_payload["acceptance_relation_to_band"] == "inside"
    assert final_payload["hmc_mechanics_exposed"] is False
    event_text = str(events)
    for forbidden in (
        "step_size",
        "num_leapfrog_steps",
        "mass_artifact_payload",
        "samples",
        "target_log_prob",
        "final_state",
    ):
        assert forbidden not in event_text


def test_bootstrap_private_callback_records_step_and_l_repair() -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []
    geometry = _geometry(
        config=HMCGeometryInitializationConfig(
            geometry_scaling_c=1000.0,
            stability_guard=1000.0,
            covariance_jitter=0.0,
            seed=(123, 456),
        )
    )

    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=_config(max_repairs=1),
        run_full_chain=_scripted_runner([0.95, 0.70])[0],
        _private_diagnostic_callback=lambda event_type, payload: events.append(
            (event_type, dict(payload))
        ),
    )

    assert result.passed is True
    assert [event_type for event_type, _payload in events] == [
        "bootstrap_kernel_repair",
    ]
    payload = events[0][1]
    assert payload["previous_step_size"] == pytest.approx(geometry.initial_step_size)
    assert payload["step_size"] == pytest.approx(geometry.initial_step_size * 2.0)
    assert payload["previous_num_leapfrog_steps"] == geometry.initial_num_leapfrog_steps
    assert payload["num_leapfrog_steps"] > 0
    assert payload["private_hmc_mechanics"] is True
    assert payload["reports_posterior_convergence"] is False


def test_real_tiny_gaussian_bootstrap_returns_structured_result() -> None:
    result = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=_config(
            screen_num_results=8,
            screen_num_burnin_steps=2,
            chain_execution_mode="eager",
        ),
    )

    assert result.final_status in {
        "passed",
        "repair",
        "repair_budget_exhausted",
        "blocked_repeated_leapfrog_cap_saturation",
    }
    assert result.rounds
    assert result.rounds[0].diagnostic_role in {
        "bootstrap_screen_promotion_only",
        "bootstrap_acceptance_repair_trigger",
        "hard_veto",
    }
    assert result.nonclaims == BOOTSTRAP_SCREEN_NONCLAIMS
    assert result.payload()["reports_posterior_convergence"] is False
