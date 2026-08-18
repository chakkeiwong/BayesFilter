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

import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning
from bayesfilter.inference.hmc_tuning import build_windowed_warmup_schedule
from bayesfilter.hmc_route_contract import (
    LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
    OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
    UnsupportedHMCAlgorithmRoute,
)
from bayesfilter.inference import (
    HMCBootstrapScreenResult,
    HMCGeometryInitializationConfig,
    HMCWindowedMassStageConfig,
    HMCWindowedMassStageResult,
    PrecomputedMassArtifact,
    ValueScoreCapability,
    initialize_hmc_kernel_geometry,
    run_hmc_bootstrap_screen,
    run_hmc_windowed_mass_stage,
)
from bayesfilter.inference.hmc_coordinates import transform_from_precomputed_mass_artifact
from bayesfilter.inference.hmc_warmup import (
    MetricAdequacyDecision,
    compose_base_transform_with_nested_artifact,
)


class _ToyGaussianAdapter:
    parameter_dim = 2

    def adapter_signature(self) -> str:
        return "kernel-windowed-mass-toy-gaussian-v1"

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_kernel_tuning_windowed_mass.py",
            target_scope="kernel_windowed_mass_toy_gaussian",
            nonclaims=("tiny windowed mass fixture only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value


class _MismatchedAdapter(_ToyGaussianAdapter):
    def adapter_signature(self) -> str:
        return "kernel-windowed-mass-mismatched-v1"


class _RotatedGaussianAdapter(_ToyGaussianAdapter):
    def __init__(self) -> None:
        angle = np.pi / 5.0
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        self.covariance = rotation @ np.diag([1.0, 0.1]) @ rotation.T
        self.precision = np.linalg.inv(self.covariance)

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        precision = tf.convert_to_tensor(self.precision, dtype=value.dtype)
        score = -tf.linalg.matvec(precision, value)
        return -0.5 * tf.reduce_sum(value * -score, axis=-1), score


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


def _bootstrap() -> HMCBootstrapScreenResult:
    def run(_adapter: Any, _initial_state: Any, _config: Any) -> _FakeRunResult:
        return _fake_result(warmup_steps=4, acceptance_trace=[True, True, False, True])

    return run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        run_full_chain=run,
    )


def _stage_config(**overrides: Any) -> HMCWindowedMassStageConfig:
    payload = {
        "algorithm_id": LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
        "target_accept_prob": 0.70,
        "seed": (20260621, 40),
        "chain_execution_mode": "eager",
        "target_scope": "kernel_windowed_mass_toy_gaussian",
    }
    payload.update(overrides)
    return HMCWindowedMassStageConfig(**payload)


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
    warmup_steps: int = 12,
    acceptance_trace: list[bool] | None = None,
    finite_samples: bool = True,
    finite_log_accept: bool = True,
    finite_target_log_prob: bool = True,
    runtime_s: float = 0.01,
    metadata_overrides: Mapping[str, Any] | None = None,
) -> _FakeRunResult:
    samples = _warmup_draws(warmup_steps)
    if not finite_samples:
        samples = samples.copy()
        samples[-1, -1] = np.nan
    if acceptance_trace is None:
        acceptance_trace = [
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            True,
            True,
        ][:warmup_steps]
    trace = {
        "is_accepted": tf.constant(acceptance_trace, dtype=tf.bool),
        "log_accept_ratio": tf.constant(
            np.linspace(-0.2, 0.1, warmup_steps)
            if finite_log_accept
            else [0.0] * (warmup_steps - 1) + [np.nan],
            dtype=tf.float64,
        ),
        "target_log_prob": tf.constant(
            -0.5 * np.sum(np.square(samples), axis=-1)
            if finite_target_log_prob
            else [0.0] * (warmup_steps - 1) + [np.nan],
            dtype=tf.float64,
        ),
    }
    diagnostics = {
        "acceptance_rate": tf.constant(
            float(np.mean(np.asarray(acceptance_trace, dtype=float))),
            dtype=tf.float64,
        ),
        "finite_sample_count": tf.constant(
            int(np.sum(np.all(np.isfinite(samples), axis=-1))),
            dtype=tf.int32,
        ),
        "nonfinite_sample_count": tf.constant(
            int(np.sum(~np.all(np.isfinite(samples), axis=-1))),
            dtype=tf.int32,
        ),
        "trace_policy": "standard",
    }
    metadata = {
        "sample_chain_call_s": runtime_s,
        "trace_unavailability": {},
        "fixture_or_synthetic": True,
        "nonclaims": ("fake runner only",),
    }
    if metadata_overrides is not None:
        metadata.update(dict(metadata_overrides))
    return _FakeRunResult(
        samples=tf.constant(samples, dtype=tf.float64),
        trace=trace,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def _runtime_shaped_result(
    *,
    warmup_steps: int = 12,
    acceptance_trace: list[bool] | None = None,
    **kwargs: Any,
) -> _FakeRunResult:
    return _fake_result(
        warmup_steps=warmup_steps,
        acceptance_trace=acceptance_trace,
        metadata_overrides={
            "runtime": "tfp.mcmc.sample_chain",
            "sample_chain_invocation_count": 1,
            "fixture_or_synthetic": False,
            "nonclaims": (
                "deterministic hmc contract plumbing result",
                "no sampler convergence claim",
                "no posterior validity claim",
            ),
        },
        **kwargs,
    )


def _operational_budget(attempt_index: int = 0):
    return hmc_kernel_tuning._HMCAttemptBudgetPolicy(
        target_dimension=2,
        attempt_index=attempt_index,
        budget=256,
        phase4_warmup_steps=256,
        phase5_tune_budgets=(64, 128, 256),
        phase5_screen_num_results=64,
        phase5_screen_burnin_steps=16,
        phase6_screen_num_results=64,
        phase6_screen_burnin_steps=16,
        verification_num_results=128,
        verification_num_burnin_steps=32,
        serious_policy=False,
    )


def _operational_inputs():
    adapter = _RotatedGaussianAdapter()
    geometry = initialize_hmc_kernel_geometry(
        adapter=adapter,
        initial_position=np.array([0.4, -0.3]),
        config=HMCGeometryInitializationConfig(covariance_jitter=0.0),
    )
    bootstrap = run_hmc_bootstrap_screen(
        adapter=adapter,
        geometry=geometry,
        run_full_chain=lambda _adapter, _state, config: _runtime_shaped_result(
            warmup_steps=int(config.num_results),
            acceptance_trace=[True, True, False, True] * 4,
        ),
    )
    return adapter, geometry, bootstrap


def test_windowed_mass_config_does_not_expose_hmc_mechanics() -> None:
    parameters = set(inspect.signature(HMCWindowedMassStageConfig).parameters)
    forbidden = {
        "step_size",
        "initial_step_size",
        "num_leapfrog_steps",
        "min_leapfrog",
        "max_leapfrog",
        "num_results",
        "num_burnin_steps",
        "warmup_steps",
        "initial_buffer",
        "final_buffer",
        "first_window_size",
        "mass_window_schedule",
        "trajectory_grid",
        "candidate_grid",
        "budget_schedule",
    }

    assert parameters.isdisjoint(forbidden)


def test_metric_update_requirement_is_typed_and_compatible_by_default() -> None:
    default = HMCWindowedMassStageConfig()
    required = HMCWindowedMassStageConfig(
        metric_update_requirement="require_operational_update"
    )

    assert default.metric_update_requirement == "allow_valid_incumbent"
    assert default.payload()["metric_update_requirement"] == "allow_valid_incumbent"
    assert required.payload()["metric_update_requirement"] == (
        "require_operational_update"
    )
    with pytest.raises(ValueError, match="metric_update_requirement"):
        HMCWindowedMassStageConfig(metric_update_requirement="unknown")
    with pytest.raises(ValueError, match="incompatible with fixed_identity"):
        HMCWindowedMassStageConfig(
            mass_policy="fixed_identity",
            metric_update_requirement="require_operational_update",
        )


def test_required_metric_update_reports_nonpromoting_zero_update_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()

    def reject_metric(_states: Any, **_kwargs: Any) -> MetricAdequacyDecision:
        return MetricAdequacyDecision(
            outcome="no_update_insufficient_metric_evidence",
            covariance=None,
            estimator_family=None,
            report={
                "forced_test_fixture": True,
                "diagonal_fallback_used": False,
                "shrinkage_spd_not_treated_as_adequacy": True,
            },
        )

    import bayesfilter.inference.hmc_warmup as hmc_warmup

    monkeypatch.setattr(hmc_warmup, "assess_metric_covariance", reject_metric)

    result = run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            seed=(20260730, 810),
            chain_execution_mode="tf_function",
            target_scope="kernel_windowed_mass_toy_gaussian",
            metric_update_requirement="require_operational_update",
        ),
        _attempt_budget_policy=_operational_budget(),
    )

    assert result.operational_warmup_result is not None
    assert result.operational_warmup_result.status == "passed"
    assert result.operational_warmup_result.metric_adaptation_status == (
        "no_metric_update"
    )
    assert result.operational_warmup_result.operational_metric_update_count == 0
    assert result.final_status == "passed_no_metric_update"
    assert result.passed is False
    assert result.hard_vetoes == ()
    assert result.diagnostic_role == "metric_adaptation_not_observed"
    assert result.repair_triggers == (
        "windowed_mass_no_operational_metric_update",
    )
    assert result.diagnostics["required_operational_metric_update_missing"] is True
    assert result.diagnostics["metric_adaptation_status"] == "no_metric_update"


def test_fixed_identity_windowed_diagnostic_makes_no_mass_updates() -> None:
    result = hmc_kernel_tuning.run_windowed_mass_adaptation_diagnostic(
        hmc_kernel_tuning.HMCTuningPolicy.windowed_mass_adaptation(
            num_adaptation_steps=12,
            target_accept_prob=0.70,
            source="tests.fixed_identity",
        ),
        config=hmc_kernel_tuning.WindowedMassAdaptationConfig(
            warmup_steps=12,
            initial_buffer=2,
            final_buffer=2,
            first_window_size=3,
            min_window_samples=2,
            covariance_jitter=0.0,
            mass_policy="fixed_identity",
        ),
        initial_mass_artifact=hmc_kernel_tuning.PrecomputedMassArtifact.from_covariance(
            position=np.zeros(2),
            covariance=np.eye(2),
            adapter_signature="kernel-windowed-mass-toy-gaussian-v1",
            position_role="test",
            covariance_source="test",
            matrix_used_for_square_root="test",
            source="tests",
            jitter=0.0,
            regularization_report={},
            nonclaims=("test",),
        ),
        warmup_draws=_warmup_draws(12),
        initial_step_size=0.1,
    )
    assert result.mass_updates == ()
    assert result.final_mass_artifact_signature == result.initial_mass_artifact_signature
    assert result.semantic_checks()["fixed_identity_signature_unchanged"] is True


def test_operational_fixed_identity_mass_artifact_signature_is_preserved() -> None:
    """The operational compatibility projection must not rewrite fixed mass identity."""

    adapter, geometry, bootstrap = _operational_inputs()
    result = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=hmc_kernel_tuning.HMCWindowedMassStageConfig(
            target_accept_prob=0.70,
            seed=(20260730, 810),
            chain_execution_mode="eager",
            use_xla=False,
            target_scope="kernel_windowed_mass_toy_gaussian",
            mass_policy="fixed_identity",
        ),
        _attempt_budget_policy=_operational_budget(),
    )

    assert result.passed is True
    assert result.operational_mass_artifact is not None
    assert result.windowed_mass_result is not None
    assert result.adapted_mass_artifact_signature == result.initial_mass_artifact_signature


def test_public_windowed_stage_propagates_fixed_identity_to_internal_config() -> None:
    internal = hmc_kernel_tuning._windowed_mass_stage_internal_config(
        mass_policy="fixed_identity",
    )

    assert internal.mass_policy == "fixed_identity"
    assert all(
        not window.update_mass
        for window in build_windowed_warmup_schedule(internal)
    )


def test_windowed_mass_stage_runs_retained_draw_route_and_preserves_nonclaims() -> None:
    calls: list[tuple[int, int, float, int, bool]] = []

    def run(adapter: Any, initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append(
            (
                int(config.num_results),
                int(config.num_burnin_steps),
                float(config.step_size),
                int(config.num_leapfrog_steps),
                bool(config.use_xla),
            )
        )
        np.testing.assert_allclose(initial_state.numpy(), np.zeros(2))
        assert adapter.adapter_signature() == _bootstrap().hmc_adapter_signature
        return _runtime_shaped_result(warmup_steps=int(config.num_results))

    bootstrap = _bootstrap()
    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=bootstrap,
        config=_stage_config(),
        run_full_chain=run,
    )

    assert isinstance(result, HMCWindowedMassStageResult)
    assert result.passed is True
    assert result.final_status == "passed"
    assert calls == [(12, 1, bootstrap.selected_round.step_size, bootstrap.selected_round.num_leapfrog_steps, False)]
    assert result.draw_capture_policy["route"] == "retained_fixed_kernel_samples"
    assert result.draw_capture_policy["num_results"] == 12
    assert result.draw_capture_policy["api_discarded_burnin_counted_as_adaptation_input"] is False
    assert result.warmup_draw_provenance["adaptation_input_only"] is True
    assert result.warmup_draw_provenance["fixture_or_synthetic"] is False
    assert result.acceptance_telemetry_provenance["source"].endswith("trace.is_accepted")


def test_windowed_mass_stage_private_progress_callback_is_allowlisted() -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []
    forbidden_keys = {
        "step_size",
        "initial_step_size",
        "num_leapfrog_steps",
        "min_leapfrog",
        "max_leapfrog",
        "bracket",
        "bracket_low",
        "bracket_high",
        "acceptance_rate",
        "runtime_metadata",
        "raw_diagnostics",
        "trace",
        "samples",
        "mass",
        "mass_artifact",
        "mass_artifact_payload",
        "budget_policy",
        "phase4_warmup_steps",
        "phase5_tune_budgets",
        "config",
        "diagnostic_config",
    }

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _runtime_shaped_result(warmup_steps=int(config.num_results))

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(),
        run_full_chain=run,
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
        _attempt_index=3,
    )

    assert result.passed is True
    assert [stage for stage, _payload in events] == [
        "windowed_mass_runner_build_start",
        "windowed_mass_runner_build_complete",
        "windowed_mass_runner_execute_start",
        "windowed_mass_runner_execute_complete",
        "windowed_mass_capture_start",
        "windowed_mass_capture_complete",
        "windowed_mass_semantic_diagnostic_start",
        "windowed_mass_semantic_diagnostic_complete",
    ]
    for stage, payload in events:
        assert payload["stage"] == stage
        assert payload["attempt_index"] == 3
        assert payload["progress_only"] is True
        assert payload["hmc_mechanics_exposed"] is False
        assert payload["route_category"] == "injected_runner"
        assert payload["algorithm_id"] == LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID
        assert payload["route_contract_version"] == (
            "bayesfilter.hmc_algorithm_route.v1"
        )
        assert payload["algorithm_route"]["algorithm_id"] == payload["algorithm_id"]
        assert payload["reports_posterior_convergence"] is False
        assert payload["reports_sampler_superiority"] is False
        assert payload["reports_default_readiness"] is False
        assert payload["reports_external_client_scientific_claim"] is False
        assert payload["reports_gpu_or_xla_readiness"] is False
        assert "no posterior convergence claim" in payload["nonclaims"]
        assert set(payload).isdisjoint(forbidden_keys)
    for _stage, payload in events[1::2]:
        assert payload["completed"] is True
        assert payload["elapsed_s"] >= 0.0
    for _stage, payload in events[0::2]:
        assert payload["started"] is True
        assert payload["elapsed_s"] == pytest.approx(0.0)
        assert payload["started_perf_counter_s"] >= 0.0
        assert payload["timing_anchor_role"] == "process_local_monotonic_debug_only"


def test_windowed_mass_public_timeout_uses_segmented_chunk_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []
    built_configs: list[Any] = []
    calls: list[dict[str, Any]] = []

    class _ScriptedChunkRunner:
        def __init__(self, _adapter: Any, _initial_state: Any, config: Any) -> None:
            self.config = config
            self.call_count = 0

        def run(
            self,
            *,
            active_results: Any,
            current_state: Any = None,
            seed: Any = None,
            step_size: Any = None,
        ) -> Any:
            self.call_count += 1
            active = int(active_results)
            calls.append(
                {
                    "active_results": active,
                    "current_state": np.asarray(current_state, dtype=float).copy(),
                    "seed": tuple(int(item) for item in seed),
                    "step_size": float(step_size),
                    "burnin": int(self.config.num_burnin_steps),
                }
            )
            offset = 0.1 * self.call_count
            samples = _warmup_draws(active) + offset
            trace = {
                "is_accepted": tf.constant(
                    [True, False, True, True, False, True, True, False, True, True, False, True][
                        :active
                    ],
                    dtype=tf.bool,
                ),
                "log_accept_ratio": tf.constant(np.linspace(-0.2, 0.1, active), dtype=tf.float64),
                "target_log_prob": tf.constant(
                    -0.5 * np.sum(np.square(samples), axis=-1),
                    dtype=tf.float64,
                ),
            }
            return hmc_kernel_tuning.FixedSizeHMCChunkRunResult(
                samples=tf.constant(samples, dtype=tf.float64),
                valid_mask=tf.ones((active,), dtype=tf.bool),
                final_state=tf.constant(samples[-1], dtype=tf.float64),
                trace=trace,
                diagnostics={
                    "valid_sample_count": tf.constant(active, dtype=tf.int32),
                    "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                },
                metadata={
                    "step_size": 999.0,
                    "num_leapfrog_steps": 999,
                    "runtime": "private fake chunk metadata",
                },
            )

    def fake_builder(adapter: Any, initial_state: Any, config: Any) -> _ScriptedChunkRunner:
        built_configs.append(config)
        return _ScriptedChunkRunner(adapter, initial_state, config)

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_WINDOWED_MASS_SEGMENT_SIZE",
        5,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_fixed_size_hmc_chunk_runner",
        fake_builder,
    )

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(public_timeout_budget_s=1000.0),
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
        _attempt_index=2,
    )

    assert result.passed is True
    assert [config.trace_policy for config in built_configs] == ["standard", "standard"]
    assert [config.num_burnin_steps for config in built_configs] == [1, 0]
    assert [call["active_results"] for call in calls] == [5, 5, 2]
    np.testing.assert_allclose(calls[1]["current_state"], _warmup_draws(5)[-1] + 0.1)
    assert result.diagnostics["runtime_metadata"]["windowed_stage_segmented_chunk_runner"] is True
    assert result.diagnostics["runtime_metadata"]["completed_segment_count"] == 3
    assert result.diagnostics["samples_shape"] == (12, 2)
    assert result.acceptance_telemetry_provenance["finite_and_aligned"] is True
    segment_events = [
        (stage, payload)
        for stage, payload in events
        if stage.startswith("windowed_mass_segment_")
    ]
    assert [stage for stage, _payload in segment_events] == [
        "windowed_mass_segment_start",
        "windowed_mass_segment_complete",
        "windowed_mass_segment_start",
        "windowed_mass_segment_complete",
        "windowed_mass_segment_start",
        "windowed_mass_segment_complete",
    ]
    public_text = json.dumps([payload for _stage, payload in segment_events], sort_keys=True)
    for forbidden in (
        '"step_size"',
        '"num_leapfrog_steps"',
        '"samples"',
        '"trace"',
        '"target_log_prob"',
        '"final_state"',
        '"mass_artifact"',
    ):
        assert forbidden not in public_text
    for _stage, payload in segment_events:
        assert payload["hmc_mechanics_exposed"] is False
        assert payload["route_category"] == "segmented_windowed_mass_runner"
        assert payload["algorithm_id"] == LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID
        assert payload["algorithm_route"]["operational_authority"] is False
        assert payload["segment_count"] == 3


def test_windowed_mass_segmented_timeout_between_chunks_returns_closeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    events: list[tuple[str, Mapping[str, Any]]] = []
    calls = {"count": 0}

    class _SlowChunkRunner:
        def __init__(self, _adapter: Any, _initial_state: Any, _config: Any) -> None:
            pass

        def run(
            self,
            *,
            active_results: Any,
            current_state: Any = None,
            seed: Any = None,
            step_size: Any = None,
        ) -> Any:
            calls["count"] += 1
            clock["now"] = 80.0
            active = int(active_results)
            samples = _warmup_draws(active)
            return hmc_kernel_tuning.FixedSizeHMCChunkRunResult(
                samples=tf.constant(samples, dtype=tf.float64),
                valid_mask=tf.ones((active,), dtype=tf.bool),
                final_state=tf.constant(samples[-1], dtype=tf.float64),
                trace={
                    "is_accepted": tf.constant([True, False, True, True, False][:active]),
                    "log_accept_ratio": tf.constant(np.linspace(-0.1, 0.1, active), dtype=tf.float64),
                    "target_log_prob": tf.constant(
                        -0.5 * np.sum(np.square(samples), axis=-1),
                        dtype=tf.float64,
                    ),
                },
                diagnostics={},
                metadata={},
            )

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(hmc_kernel_tuning, "_WINDOWED_MASS_SEGMENT_SIZE", 5)
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_fixed_size_hmc_chunk_runner",
        lambda adapter, initial_state, config: _SlowChunkRunner(adapter, initial_state, config),
    )

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(
            public_timeout_budget_s=100.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
        _attempt_index=4,
    )

    assert calls["count"] == 1
    assert result.passed is False
    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "windowed_mass_resource_timeout_non_promoting"
    assert result.hard_vetoes == ()
    assert result.repair_triggers == (
        "windowed_mass_public_timeout_closeout_before_hmc_call",
    )
    closeout = result.diagnostics["public_timeout_closeout"]
    assert closeout["completed_segment_count"] == 1
    assert closeout["planned_segment_count"] == 3
    assert closeout["closeout_required_before_next_segment"] is True
    assert closeout["estimated_next_segment_s"] == pytest.approx(100.0)
    assert closeout["completed_segment_elapsed_estimator"] == (
        "recent_max_times_safety_multiplier"
    )
    assert closeout["hmc_mechanics_exposed"] is False
    assert [stage for stage, _payload in events if "segment" in stage] == [
        "windowed_mass_segment_start",
        "windowed_mass_segment_complete",
    ]
    assert events[-1][0] == "windowed_mass_public_timeout_closeout"
    assert events[-1][1]["public_timeout_closeout"]["completed_segment_count"] == 1
    public_text = json.dumps(events[-1][1], sort_keys=True)
    for forbidden in (
        '"step_size"',
        '"num_leapfrog_steps"',
        '"samples"',
        '"trace"',
        '"final_state"',
    ):
        assert forbidden not in public_text


def test_windowed_mass_segmented_staged_timeout_enlargement_allows_next_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 0.0}
    events: list[tuple[str, Mapping[str, Any]]] = []
    calls = {"count": 0}

    class _SlowChunkRunner:
        def __init__(self, _adapter: Any, _initial_state: Any, _config: Any) -> None:
            pass

        def run(
            self,
            *,
            active_results: Any,
            current_state: Any = None,
            seed: Any = None,
            step_size: Any = None,
        ) -> Any:
            del current_state, seed, step_size
            calls["count"] += 1
            clock["now"] += 80.0
            active = int(active_results)
            samples = _warmup_draws(active) + (0.1 * calls["count"])
            return hmc_kernel_tuning.FixedSizeHMCChunkRunResult(
                samples=tf.constant(samples, dtype=tf.float64),
                valid_mask=tf.ones((active,), dtype=tf.bool),
                final_state=tf.constant(samples[-1], dtype=tf.float64),
                trace={
                    "is_accepted": tf.constant(
                        [True, False, True, True, False][:active],
                        dtype=tf.bool,
                    ),
                    "log_accept_ratio": tf.constant(
                        np.linspace(-0.1, 0.1, active),
                        dtype=tf.float64,
                    ),
                    "target_log_prob": tf.constant(
                        -0.5 * np.sum(np.square(samples), axis=-1),
                        dtype=tf.float64,
                    ),
                },
                diagnostics={},
                metadata={
                    "fixed_size_chunk_runner": True,
                    "runtime": (
                        "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop"
                    ),
                },
            )

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(hmc_kernel_tuning, "_WINDOWED_MASS_SEGMENT_SIZE", 5)
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_fixed_size_hmc_chunk_runner",
        lambda adapter, initial_state, config: _SlowChunkRunner(
            adapter,
            initial_state,
            config,
        ),
    )

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(
            public_timeout_budget_s=100.0,
            staged_timeout_policy=hmc_kernel_tuning.HMCStagedTimeoutPolicy(
                stage_budgets_s={
                    "geometry_and_bootstrap": 100.0,
                    "phase7_pre_windowed": 100.0,
                    "windowed_mass": 100.0,
                    "fixed_mass_step": 100.0,
                    "frozen_step_trajectory": 100.0,
                    "fresh_fixed_kernel_verification": 100.0,
                },
                global_cap_s=1000.0,
                reserve_s=10.0,
                max_enlargement_rounds_per_stage=1,
                enlargement_multiplier=2.0,
            ),
            staged_timeout_global_started_perf_counter_s=0.0,
            staged_timeout_stage_started_perf_counter_s=0.0,
            staged_timeout_enlargement_rounds={"windowed_mass": 0},
        ),
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
        _attempt_index=6,
    )

    assert calls["count"] == 3
    assert result.passed is True
    assert "windowed_mass_public_timeout_closeout" not in [
        stage for stage, _payload in events
    ]
    assert [stage for stage, _payload in events if "segment" in stage] == [
        "windowed_mass_segment_start",
        "windowed_mass_segment_complete",
        "windowed_mass_segment_start",
        "windowed_mass_segment_complete",
        "windowed_mass_segment_start",
        "windowed_mass_segment_complete",
    ]


def test_windowed_mass_segmented_soft_deadline_skips_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 30.0}
    events: list[tuple[str, Mapping[str, Any]]] = []
    calls = {"count": 0}

    class _UnexpectedChunkRunner:
        def __init__(self, _adapter: Any, _initial_state: Any, _config: Any) -> None:
            pass

        def run(
            self,
            *,
            active_results: Any,
            current_state: Any = None,
            seed: Any = None,
            step_size: Any = None,
        ) -> Any:
            del active_results, current_state, seed, step_size
            calls["count"] += 1
            raise AssertionError("soft deadline should close out before segment 0")

    monkeypatch.setattr(hmc_kernel_tuning.time, "perf_counter", lambda: clock["now"])
    monkeypatch.setattr(hmc_kernel_tuning, "_WINDOWED_MASS_SEGMENT_SIZE", 5)
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_fixed_size_hmc_chunk_runner",
        lambda adapter, initial_state, config: _UnexpectedChunkRunner(
            adapter,
            initial_state,
            config,
        ),
    )

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(
            public_timeout_budget_s=100.0,
            public_timeout_started_perf_counter_s=0.0,
        ),
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
        _attempt_index=5,
    )

    assert calls["count"] == 0
    assert result.passed is False
    assert result.final_status == "budget_exhausted"
    assert result.diagnostic_role == "windowed_mass_resource_timeout_non_promoting"
    assert result.hard_vetoes == ()
    assert result.repair_triggers == (
        "windowed_mass_public_timeout_closeout_before_hmc_call",
    )
    closeout = result.diagnostics["public_timeout_closeout"]
    assert closeout["remaining_s"] == pytest.approx(70.0)
    assert closeout["reserve_s"] == pytest.approx(50.0)
    assert closeout["estimated_next_segment_s"] == pytest.approx(25.0)
    assert closeout["completed_segment_elapsed_count"] == 0
    assert closeout["completed_segment_elapsed_estimator"] == (
        "fallback_min_reserve_or_quarter_budget"
    )
    assert closeout["completed_segment_count"] == 0
    assert closeout["planned_segment_count"] == 3
    assert closeout["closeout_required_before_next_segment"] is True
    assert [stage for stage, _payload in events if "segment" in stage] == []
    assert events[-1][0] == "windowed_mass_public_timeout_closeout"
    public_text = json.dumps(events[-1][1], sort_keys=True)
    for forbidden in (
        '"step_size"',
        '"num_leapfrog_steps"',
        '"samples"',
        '"trace"',
        '"target_log_prob"',
        '"final_state"',
        '"mass_artifact"',
    ):
        assert forbidden not in public_text


def test_windowed_mass_config_use_xla_propagates_to_full_chain_config() -> None:
    calls: list[bool] = []

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append(bool(config.use_xla))
        return _runtime_shaped_result(warmup_steps=int(config.num_results))

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(chain_execution_mode="tf_function", use_xla=True),
        run_full_chain=run,
    )

    assert result.passed is True
    assert result.config.payload()["use_xla"] is True
    assert calls == [True]
    assert result.acceptance_telemetry_provenance["fixture_or_synthetic"] is False
    assert result.acceptance_telemetry_provenance["policy_filled_or_default"] is False
    assert result.windowed_mass_result is not None
    assert result.windowed_mass_result.final_mass_artifact_signature == result.adapted_mass_artifact_signature
    assert result.candidate_step_size == result.windowed_mass_result.final_step_size
    assert result.payload()["reports_posterior_convergence"] is False
    assert "no posterior convergence claim" in result.nonclaims


def test_windowed_mass_injected_tf_function_run_does_not_build_reusable_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fail_if_built(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("injected run_full_chain must bypass reusable runner")

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append(
            {
                "chain_execution_mode": config.chain_execution_mode,
                "use_xla": bool(config.use_xla),
            }
        )
        return _runtime_shaped_result(warmup_steps=int(config.num_results))

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_reusable_full_chain_tfp_hmc_runner",
        fail_if_built,
    )

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(chain_execution_mode="tf_function", use_xla=True),
        run_full_chain=run,
    )

    assert result.passed is True
    assert calls == [{"chain_execution_mode": "tf_function", "use_xla": True}]
    assert result.diagnostics["runtime_metadata"]["sample_chain_invocation_count"] == 1


def test_runner_identity_cannot_select_a_different_windowed_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_builder(*_args: Any, **_kwargs: Any) -> Any:
        calls.append("builder_called")
        raise AssertionError("blocked route must not construct a runner")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_reusable_full_chain_tfp_hmc_runner",
        fake_builder,
    )

    with pytest.raises(UnsupportedHMCAlgorithmRoute) as caught:
        run_hmc_windowed_mass_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            config=_stage_config(
                algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
                chain_execution_mode="tf_function",
                use_xla=False,
            ),
        )

    assert caught.value.decision.algorithm_id == (
        OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    )
    assert caught.value.decision.blocker_code == (
        "operational_windowed_warmup_requires_default_runner"
    )
    assert calls == []


def test_windowed_mass_stage_hard_vetoes_fixture_runtime_evidence() -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _fake_result(warmup_steps=int(config.num_results))

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "hard_veto"
    assert "windowed_stage_fixture_or_nonruntime_telemetry" in result.hard_vetoes
    assert result.warmup_draw_provenance["fixture_or_synthetic"] is True
    assert result.acceptance_telemetry_provenance["fixture_or_synthetic"] is True
    assert result.windowed_mass_result is None


def test_windowed_mass_stage_hard_vetoes_default_like_acceptance_trace() -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _runtime_shaped_result(
            warmup_steps=int(config.num_results),
            acceptance_trace=[True] * int(config.num_results),
        )

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "hard_veto"
    assert "windowed_stage_acceptance_telemetry_invalid_or_default" in result.hard_vetoes
    assert result.acceptance_telemetry_provenance["policy_filled_or_default"] is True
    assert result.windowed_mass_result is None


def test_windowed_mass_stage_accepts_constant_signed_sample_chain_telemetry() -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        result = _runtime_shaped_result(
            warmup_steps=int(config.num_results),
            acceptance_trace=[True] * int(config.num_results),
        )
        metadata = dict(result.metadata)
        metadata["program_signature"] = "signed-bayesfilter-sample-chain-runtime"
        return replace(result, metadata=metadata)

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.passed is True
    provenance = result.acceptance_telemetry_provenance
    assert provenance["constant_trace"] is True
    assert provenance["runtime_decision_count_supported"] is True
    assert provenance["policy_filled_or_default"] is False
    assert provenance["accepted_decision_count"] == 12
    assert provenance["acceptance_decision_count"] == 12


def test_windowed_mass_acceptance_runtime_support_rejects_count_mismatch() -> None:
    payload = {
        "expected_steps": 4,
        "acceptance_trace": np.ones((4,), dtype=float),
        "runtime_evidence": "tfp_hmc_runtime",
        "fixture_or_synthetic": False,
        "raw_diagnostics": {
            "acceptance_decision_source": "sample_chain_trace_counts",
            "accepted_decision_count": 5,
            "acceptance_decision_count": 5,
            "acceptance_trace_decision_count": 4,
            "raw_acceptance_shape": (4,),
        },
        "runtime_metadata": {
            "runtime": "tfp.mcmc.sample_chain",
            "sample_chain_invocation_count": 1,
            "program_signature": "signed-bayesfilter-sample-chain-runtime",
        },
    }

    assert (
        hmc_kernel_tuning._windowed_stage_acceptance_has_runtime_decision_support(
            payload
        )
        is False
    )
    assert (
        hmc_kernel_tuning._windowed_stage_acceptance_policy_filled_or_default(
            payload
        )
        is True
    )


def test_windowed_mass_segmented_constant_runtime_acceptance_uses_decision_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _AllAcceptedChunkRunner:
        def __init__(self, _adapter: Any, _initial_state: Any, config: Any) -> None:
            self.config = config
            self.call_count = 0

        def run(
            self,
            *,
            active_results: Any,
            current_state: Any = None,
            seed: Any = None,
            step_size: Any = None,
        ) -> Any:
            del seed, step_size
            self.call_count += 1
            active = int(active_results)
            base_state = np.asarray(current_state, dtype=float)
            samples = (
                np.tile(base_state, (int(self.config.max_results), 1))
                + 0.01 * self.call_count
            )
            trace = {
                "is_accepted": tf.ones((int(self.config.max_results),), dtype=tf.bool),
                "log_accept_ratio": tf.zeros(
                    (int(self.config.max_results),),
                    dtype=tf.float64,
                ),
                "target_log_prob": tf.zeros(
                    (int(self.config.max_results),),
                    dtype=tf.float64,
                ),
            }
            return hmc_kernel_tuning.FixedSizeHMCChunkRunResult(
                samples=tf.constant(samples, dtype=tf.float64),
                valid_mask=tf.range(int(self.config.max_results)) < active,
                final_state=tf.constant(samples[-1], dtype=tf.float64),
                trace=trace,
                diagnostics={
                    "valid_sample_count": tf.constant(active, dtype=tf.int32),
                    "nonfinite_valid_sample_count": tf.constant(0, dtype=tf.int32),
                    "accepted_decision_count": tf.constant(active, dtype=tf.int32),
                    "acceptance_decision_count": tf.constant(active, dtype=tf.int32),
                    "acceptance_rate": tf.constant(1.0, dtype=tf.float64),
                },
                metadata={
                    "runtime": (
                        "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop"
                    ),
                    "fixed_size_chunk_runner": True,
                    "fixture_or_synthetic": False,
                },
            )

    monkeypatch.setattr(hmc_kernel_tuning, "_WINDOWED_MASS_SEGMENT_SIZE", 5)
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "build_fixed_size_hmc_chunk_runner",
        lambda adapter, initial_state, config: _AllAcceptedChunkRunner(
            adapter,
            initial_state,
            config,
        ),
    )

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(public_timeout_budget_s=1000.0),
    )

    assert result.passed is True
    provenance = result.acceptance_telemetry_provenance
    assert provenance["constant_trace"] is True
    assert provenance["runtime_decision_count_supported"] is True
    assert provenance["policy_filled_or_default"] is False
    assert provenance["accepted_decision_count"] == 12
    assert provenance["acceptance_decision_count"] == 12


def test_windowed_mass_stage_requires_bootstrap_without_hard_veto() -> None:
    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        run_full_chain=lambda _adapter, _initial_state, _config: _fake_result(
            finite_log_accept=False
        ),
    )
    assert bootstrap.final_status == "hard_veto"

    with pytest.raises(ValueError, match="bootstrap preflight without hard veto"):
        run_hmc_windowed_mass_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=bootstrap,
            config=_stage_config(),
            run_full_chain=lambda *_args: _fake_result(),
        )


def test_windowed_mass_stage_accepts_non_promoting_bootstrap_preflight() -> None:
    geometry = _geometry()
    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        config=None,
        run_full_chain=lambda _adapter, _initial_state, _config: _fake_result(
            acceptance_trace=[True] * 12
        ),
    )
    assert bootstrap.passed is False
    assert bootstrap.final_status == "repair_budget_exhausted"

    calls: list[tuple[float, int]] = []

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append((float(config.step_size), int(config.num_leapfrog_steps)))
        return _runtime_shaped_result(warmup_steps=int(config.num_results))

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.passed is True
    assert calls == [
        (
            geometry.initial_step_size,
            geometry.initial_num_leapfrog_steps,
        )
    ]
    assert result.selected_bootstrap_kernel_hash != bootstrap.selected_kernel_hash
    assert result.diagnostic_run_config_payload["step_size"] == geometry.initial_step_size
    assert result.diagnostic_run_config_payload["num_leapfrog_steps"] == (
        geometry.initial_num_leapfrog_steps
    )


def test_windowed_mass_stage_retry_uses_private_selected_pair_seed() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    retry_state = hmc_kernel_tuning._HMCPhaseAttemptState(
        selected_step_size=0.125,
        selected_step_hash="previous-selected-step-hash",
        selected_num_leapfrog_steps=9,
        selected_trajectory_hash="previous-trajectory-hash",
        handoff_stage="phase6",
    )
    calls: list[tuple[float, int]] = []

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append((float(config.step_size), int(config.num_leapfrog_steps)))
        return _runtime_shaped_result(warmup_steps=int(config.num_results))

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(),
        run_full_chain=run,
        _attempt_state=retry_state,
    )

    assert result.passed is True
    assert calls == [(0.125, 9)]
    assert result.selected_bootstrap_kernel_hash == (
        hmc_kernel_tuning._active_bootstrap_handoff_kernel_hash(
            geometry=geometry,
            bootstrap=bootstrap,
        )
    )
    assert result.diagnostic_run_config_payload["step_size"] == pytest.approx(0.125)
    assert result.diagnostic_run_config_payload["num_leapfrog_steps"] == 9
    seed = result.diagnostics["mass_window_seed_kernel"]
    assert seed["uses_private_retry_pair"] is True
    assert seed["bootstrap_kernel_is_lineage_not_active_mass_window_seed"] is True
    assert seed["seed_kernel_source"] == "phase7_private_selected_step"


def test_windowed_mass_stage_retry_uses_private_repair_pair_seed() -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    retry_state = hmc_kernel_tuning._HMCPhaseAttemptState(
        selected_step_size=0.125,
        selected_step_hash="previous-selected-step-hash",
        phase6_retry_num_leapfrog_steps=11,
        phase6_retry_anchor_source="phase6_failed_candidate_nearest_tau",
        verification_acceptance_rate=0.90,
        verification_acceptance_relation="above_acceptance_band",
        verification_repair_trigger="phase6_trajectory_acceptance_outside_pass_band",
        verification_repair_source="phase6_frozen_step_trajectory_acceptance",
        verification_repair_step_size=0.25,
        verification_repair_step_hash="private-repair-step-hash",
        verification_repair_applied=True,
        handoff_stage="phase5_selected",
    )
    calls: list[tuple[float, int]] = []

    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        calls.append((float(config.step_size), int(config.num_leapfrog_steps)))
        return _runtime_shaped_result(warmup_steps=int(config.num_results))

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(),
        run_full_chain=run,
        _attempt_state=retry_state,
    )

    assert result.passed is True
    assert calls == [(0.25, 11)]
    assert result.diagnostic_run_config_payload["step_size"] == pytest.approx(0.25)
    assert result.diagnostic_run_config_payload["num_leapfrog_steps"] == 11
    seed = result.diagnostics["mass_window_seed_kernel"]
    assert seed["uses_private_retry_pair"] is True
    assert seed["seed_kernel_source"] == "phase7_private_repair_step"


def test_windowed_mass_stage_validates_adapter_and_mass_signatures() -> None:
    with pytest.raises(ValueError, match="adapter signature"):
        run_hmc_windowed_mass_stage(
            adapter=_MismatchedAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            config=_stage_config(),
            run_full_chain=lambda *_args: _fake_result(),
        )

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
        run_hmc_windowed_mass_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=bad_geometry,
            bootstrap=_bootstrap(),
            config=_stage_config(),
            run_full_chain=lambda *_args: _fake_result(),
        )


@pytest.mark.parametrize(
    "fake_kwargs, expected_veto",
    [
        ({"runtime_s": np.nan}, "windowed_stage_runtime_missing_or_nonfinite"),
        ({"finite_samples": False}, "windowed_stage_warmup_draws_invalid"),
        ({"finite_log_accept": False}, "windowed_stage_log_accept_invalid"),
        ({"finite_target_log_prob": False}, "windowed_stage_target_log_prob_invalid"),
    ],
)
def test_windowed_mass_stage_hard_vetoes_invalid_retained_diagnostics(
    fake_kwargs: Mapping[str, Any],
    expected_veto: str,
) -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        return _runtime_shaped_result(warmup_steps=int(config.num_results), **fake_kwargs)

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert result.final_status == "hard_veto"
    assert expected_veto in result.hard_vetoes
    assert result.windowed_mass_result is None


def test_windowed_mass_stage_hard_vetoes_missing_acceptance_trace() -> None:
    def run(_adapter: Any, _initial_state: Any, config: Any) -> _FakeRunResult:
        result = _fake_result(warmup_steps=int(config.num_results))
        trace = dict(result.trace)
        trace.pop("is_accepted")
        return replace(result, trace=trace)

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(),
        run_full_chain=run,
    )

    assert result.passed is False
    assert "windowed_stage_acceptance_telemetry_invalid_or_default" in result.hard_vetoes
    assert result.acceptance_telemetry_provenance["trace_key_present"] is False
    assert result.acceptance_telemetry_provenance["policy_filled_or_default"] is True


def test_real_tiny_gaussian_windowed_mass_stage_returns_structured_result() -> None:
    bootstrap = run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        config=None,
    )
    if not bootstrap.passed:
        pytest.skip(f"tiny bootstrap did not pass: {bootstrap.final_status}")

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=bootstrap,
        config=_stage_config(chain_execution_mode="eager"),
    )

    assert result.final_status in {"passed", "hard_veto"}
    assert result.draw_capture_policy["num_results"] == 12
    assert result.payload()["reports_fixed_mass_step_tuning"] is False
    assert result.payload()["reports_trajectory_tuning"] is False


def test_real_default_route_emits_operational_v2_and_exact_compatibility() -> None:
    adapter, geometry, bootstrap = _operational_inputs()

    result = run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
        ),
        _attempt_budget_policy=_operational_budget(),
    )

    assert result.passed is True
    assert result.operational_warmup_result is not None
    operational = result.operational_warmup_result
    assert operational.reasonable_epsilon.status == "externally_qualified"
    assert operational.reasonable_epsilon.qualification_source == (
        "bayesfilter_fixed_kernel_screen_handoff"
    )
    bootstrap_step = float(bootstrap.selected_kernel_payload["step_size"])
    assert all(
        np.max(window.consumed_step_size_trace)
        <= bootstrap_step * (1.0 + 1.0e-12)
        for window in operational.windows
        if window.dual_averaging_generation == 0
    )
    if operational.operational_metric_update_count == 0:
        rejected = [
            window.metric_decision
            for window in operational.windows
            if window.metric_decision is not None
            and window.metric_decision.outcome == "candidate_metric_rejected"
        ]
        assert rejected
        assert all(
            decision.report["candidate_rejection_stage"] == "reasonable_epsilon"
            and decision.report["incumbent_metric_retained"] is True
            for decision in rejected
        )
    assert operational.every_update_used_by_later_transition is True
    assert operational.public_payload()["schema"] == (
        "bayesfilter.hmc_operational_windowed_warmup.v2"
    )
    assert result.acceptance_telemetry_provenance["runtime_decision_count_supported"] is True
    assert result.acceptance_telemetry_provenance["source"].endswith("is_accepted")
    assert result.windowed_mass_result is not None
    assert all(
        update.reset_event["diagnostic_role"]
        == "legacy_v1_nonoperational_projection"
        for update in result.windowed_mass_result.mass_updates
    )
    _estimate, base_transform = transform_from_precomputed_mass_artifact(
        geometry.mass_artifact,
        source_coordinate_signature=geometry.mass_artifact_signature,
        estimator_family="geometry_position_covariance",
    )
    nested = result.windowed_mass_result.final_mass_artifact
    recomposed = compose_base_transform_with_nested_artifact(
        base_transform=base_transform,
        nested_artifact=nested,
        source_coordinate_signature=result.adapted_mass_artifact_signature,
    )
    probes = np.array([[0.0, 0.0], [0.2, -0.4]])
    np.testing.assert_allclose(
        recomposed.latent_to_theta(probes),
        operational.final_kernel_state.transform.latent_to_theta(probes),
        atol=1.0e-10,
    )
    public_text = json.dumps(operational.public_payload(), sort_keys=True)
    assert "private_start_bank_theta" not in public_text


def test_real_operational_route_with_generous_timeout_never_uses_legacy() -> None:
    adapter, geometry, bootstrap = _operational_inputs()
    events: list[tuple[str, Mapping[str, Any]]] = []

    result = run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
            public_timeout_budget_s=3600.0,
        ),
        _attempt_budget_policy=_operational_budget(),
        _progress_callback=lambda stage, payload: events.append((stage, payload)),
    )

    assert result.passed is True
    assert result.operational_warmup_result is not None
    assert result.operational_warmup_closeout is None
    assert result.config.algorithm_id == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    assert result.diagnostics["algorithm_id"] == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    assert result.diagnostics["algorithm_route"]["execution_control_capabilities"][
        "timeout"
    ] == "window_boundary_closeout"
    assert result.operational_warmup_result.algorithm_id == (
        OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    )
    stages = [stage for stage, _payload in events]
    assert stages[0] == "windowed_mass_operational_warmup_start"
    assert stages[-1] == "windowed_mass_operational_warmup_complete"
    completed = [
        payload["operational_progress"]["completed_transition_count"]
        for stage, payload in events
        if stage == "windowed_mass_operational_segment_complete"
    ]
    assert completed
    assert completed == sorted(set(completed))
    assert completed[-1] == result.operational_warmup_result.config.warmup_steps
    assert all(
        payload["algorithm_id"] == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
        and payload["route_contract_version"]
        == "bayesfilter.hmc_algorithm_route.v1"
        and payload["route_category"] == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
        for _stage, payload in events
    )


def test_legacy_projection_failure_cannot_change_operational_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()

    def fail_legacy_projection(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("forced non-operational compatibility failure")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_windowed_mass_adaptation_diagnostic",
        fail_legacy_projection,
    )
    result = run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
        ),
        _attempt_budget_policy=_operational_budget(),
    )

    assert result.passed is True
    assert result.operational_warmup_result is not None
    operational = result.operational_warmup_result
    assert result.windowed_mass_result is None
    assert result.diagnostics["runtime_metadata"][
        "legacy_v1_mass_updates_operational"
    ] is False
    compatibility = result.diagnostics["runtime_metadata"][
        "legacy_v1_compatibility_projection"
    ]
    assert compatibility["status"] == "unavailable_error"
    assert compatibility["authoritative"] is False
    assert compatibility["error_type"] == "RuntimeError"
    assert result.operational_mass_artifact is not None
    assert result.adapted_mass_artifact_signature == (
        hmc_kernel_tuning._mass_artifact_signature(result.operational_mass_artifact)
    )
    assert hmc_kernel_tuning._phase4_adapted_mass_artifact(result) is (
        result.operational_mass_artifact
    )
    assert result.candidate_step_size == pytest.approx(
        operational.final_kernel_state.epsilon
    )
    assert result.diagnostics["metric_adaptation_status"] == (
        operational.metric_adaptation_status
    )


def test_operational_retry_consumes_carried_transform_endpoint_step_and_l(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()
    first = run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
        ),
        _attempt_budget_policy=_operational_budget(),
    )
    assert first.passed and first.operational_warmup_result is not None
    first_operational = first.operational_warmup_result
    carried_mass = first.windowed_mass_result.final_mass_artifact
    retry_state = hmc_kernel_tuning._HMCPhaseAttemptState(
        mass_artifact_payload=carried_mass.to_payload(include_arrays=True),
        mass_artifact_signature=hmc_kernel_tuning._mass_artifact_signature(carried_mass),
        canonical_theta_state=first_operational.final_kernel_state.canonical_theta,
        private_start_bank_theta=first_operational.private_start_bank_theta,
        private_start_bank_signature=first_operational.private_start_bank_signature,
        selected_step_size=0.19,
        selected_step_hash="carried-selected-step",
        selected_num_leapfrog_steps=7,
        handoff_stage="phase5_selected",
    )
    observed: dict[str, Any] = {}
    real_runner = hmc_kernel_tuning.run_operational_windowed_warmup

    def capture_inputs(**kwargs: Any):
        observed.update(kwargs)
        return real_runner(**kwargs)

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_operational_windowed_warmup",
        capture_inputs,
    )
    second = run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
        ),
        _attempt_budget_policy=_operational_budget(attempt_index=1),
        _attempt_state=retry_state,
    )

    assert second.passed is True
    np.testing.assert_allclose(
        observed["initial_canonical_theta"],
        retry_state.canonical_theta_state,
    )
    assert observed["initial_step_size"] == pytest.approx(0.19)
    assert observed["initial_step_size_upper_bound"] is None
    assert observed["initial_step_qualification_source"] is None
    assert observed["trajectory_policy"].num_leapfrog_steps == 7
    _estimate, base_transform = transform_from_precomputed_mass_artifact(
        geometry.mass_artifact,
        source_coordinate_signature=geometry.mass_artifact_signature,
        estimator_family="geometry_position_covariance",
    )
    expected_transform = compose_base_transform_with_nested_artifact(
        base_transform=base_transform,
        nested_artifact=carried_mass,
        source_coordinate_signature=retry_state.mass_artifact_signature,
    )
    assert observed["initial_transform"].signature == expected_transform.signature
    assert second.diagnostics["mass_window_seed_kernel"]["uses_private_retry_pair"] is True
    retry_payload = retry_state.payload()
    assert retry_payload["private_start_bank_signature"] == (
        first_operational.private_start_bank_signature
    )
    assert "private_start_bank_theta" not in retry_payload


def test_operational_route_runtime_error_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_kwargs: Any) -> Any:
        raise RuntimeError("operational failure sentinel collision")

    monkeypatch.setattr(hmc_kernel_tuning, "run_operational_windowed_warmup", fail)

    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
        ),
    )

    assert result.passed is False
    assert "windowed_stage_hmc_error" in result.hard_vetoes
    assert result.diagnostics["hmc_error_type"] == "RuntimeError"
    assert "sentinel collision" in result.diagnostics["hmc_error_message"]
    assert result.diagnostics["runtime_metadata"] == {}
