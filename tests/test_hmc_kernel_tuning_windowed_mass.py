from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning
import bayesfilter.inference.hmc_warmup as hmc_warmup
from bayesfilter.inference.hmc_tuning import build_windowed_warmup_schedule
from bayesfilter.hmc_route_contract import (
    LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
    OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
    UnsupportedHMCAlgorithmRoute,
)
from bayesfilter.inference import (
    HMCBootstrapScreenResult,
    HMCGeometryInitializationConfig,
    HMCKernelTuningConfig,
    HMCStartBankDiagnosticResult,
    HMCWindowedMassStageConfig,
    HMCWindowedMassStageResult,
    PrecomputedMassArtifact,
    ValueScoreCapability,
    initialize_hmc_kernel_geometry,
    run_hmc_bootstrap_screen,
    run_hmc_start_bank_diagnostic,
    run_hmc_windowed_mass_stage,
)
from bayesfilter.inference.hmc_coordinates import (
    AffineCoordinateTransform,
    KernelState,
    MomentumMetric,
    PositionCovarianceEstimate,
    WarmupTrajectoryPolicy,
    transform_from_precomputed_mass_artifact,
)
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
    qualification = operational.start_bank_qualification.public_payload()
    assert qualification["shadow_decision_effect"] is False
    assert qualification["interpretation"] == "final_pass"
    assert result.diagnostics["raw_diagnostics"][
        "start_bank_qualification"
    ] == qualification
    assert qualification["scopes"]["authoritative_final_window"][
        "source_row_count"
    ] == operational.windows[-1].adaptation_canonical_states.shape[0]
    assert qualification["scopes"]["shadow_all_windows"][
        "source_row_count"
    ] == sum(
        window.adaptation_canonical_states.shape[0]
        for window in operational.windows
    )


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
        error = RuntimeError("operational failure sentinel collision")
        setattr(
            error,
            hmc_warmup._START_BANK_DIAGNOSTIC_ATTRIBUTE,
            {"schema": "forged.start_bank.v1"},
        )
        raise error

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
    assert "start_bank_qualification" not in result.diagnostics["raw_diagnostics"]


def _raise_validated_start_bank_failure() -> None:
    authoritative = hmc_warmup._assess_private_start_bank(
        np.zeros((4, 2)),
        scope="authoritative_final_window",
    )
    shadow = hmc_warmup._best_effort_shadow_start_bank_scope(
        np.arange(12.0).reshape((6, 2)),
        reference_transform=None,
        minimum_relative_separation=1.0e-4,
    )
    qualification = hmc_warmup._StartBankQualificationDiagnostic(
        authoritative=authoritative.diagnostic,
        shadow=shadow,
    )
    hmc_warmup._materialize_private_start_bank(
        authoritative,
        qualification=qualification,
    )


def test_operational_start_bank_failure_survives_real_windowed_stage_catch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []

    def fail(**_kwargs: Any) -> Any:
        _raise_validated_start_bank_failure()

    def callback(event: str, payload: Mapping[str, Any]) -> None:
        events.append((event, json.loads(json.dumps(payload))))
        payload["start_bank_qualification"]["interpretation"] = "mutated"

    monkeypatch.setattr(hmc_kernel_tuning, "run_operational_windowed_warmup", fail)
    result = run_hmc_windowed_mass_stage(
        adapter=_ToyGaussianAdapter(),
        geometry=_geometry(),
        bootstrap=_bootstrap(),
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
        ),
        _private_diagnostic_callback=callback,
    )

    assert result.passed is False
    assert result.diagnostics["hmc_error_type"] == "ValueError"
    assert result.diagnostics["hmc_error_message"] == (
        "operational warmup start bank is not sufficiently dispersed"
    )
    qualification = result.diagnostics["raw_diagnostics"][
        "start_bank_qualification"
    ]
    assert qualification["interpretation"] == "final_fail_shadow_pass"
    assert qualification["shadow_decision_effect"] is False
    assert [event for event, _payload in events] == ["start_bank_qualification"]
    assert events[0][1]["start_bank_qualification"] == qualification
    assert events[0][1]["private_hmc_mechanics"] is False
    serialized = json.dumps(events[0][1], sort_keys=True)
    assert "canonical_states" not in serialized
    assert "selected_row_indices" not in serialized


def test_start_bank_diagnostic_callback_failure_remains_fatal_after_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**_kwargs: Any) -> Any:
        _raise_validated_start_bank_failure()

    def callback(_event: str, _payload: Mapping[str, Any]) -> None:
        raise RuntimeError("diagnostic writer failed")

    monkeypatch.setattr(hmc_kernel_tuning, "run_operational_windowed_warmup", fail)
    with pytest.raises(RuntimeError, match="diagnostic writer failed"):
        run_hmc_windowed_mass_stage(
            adapter=_ToyGaussianAdapter(),
            geometry=_geometry(),
            bootstrap=_bootstrap(),
            config=_stage_config(
                algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
                chain_execution_mode="tf_function",
            ),
            _private_diagnostic_callback=callback,
        )


def _diagnostic_qualification(
    interpretation: str,
    *,
    shadow_rows: int = 16,
) -> Mapping[str, Any]:
    separated_authoritative = np.arange(8.0).reshape((4, 2))
    collapsed_authoritative = np.zeros((4, 2))
    separated_shadow = np.arange(float(shadow_rows * 2)).reshape((shadow_rows, 2))
    collapsed_shadow = np.zeros((shadow_rows, 2))

    if interpretation == "final_pass":
        authoritative = hmc_warmup._assess_private_start_bank(
            separated_authoritative,
            scope="authoritative_final_window",
        ).diagnostic
        shadow = hmc_warmup._assess_private_start_bank(
            separated_shadow,
            scope="shadow_all_windows",
        ).diagnostic
    elif interpretation == "final_fail_shadow_pass":
        authoritative = hmc_warmup._assess_private_start_bank(
            collapsed_authoritative,
            scope="authoritative_final_window",
        ).diagnostic
        shadow = hmc_warmup._assess_private_start_bank(
            separated_shadow,
            scope="shadow_all_windows",
        ).diagnostic
    elif interpretation == "both_fail":
        authoritative = hmc_warmup._assess_private_start_bank(
            collapsed_authoritative,
            scope="authoritative_final_window",
        ).diagnostic
        shadow = hmc_warmup._assess_private_start_bank(
            collapsed_shadow,
            scope="shadow_all_windows",
        ).diagnostic
    elif interpretation == "post_selection_invariant_failure":
        successful = hmc_warmup._assess_private_start_bank(
            separated_authoritative,
            scope="authoritative_final_window",
        ).diagnostic
        authoritative = replace(
            successful,
            selection_succeeded=False,
            failure_code="post_selection_pairwise_failure",
        )
        shadow = hmc_warmup._assess_private_start_bank(
            separated_shadow,
            scope="shadow_all_windows",
        ).diagnostic
    else:
        raise ValueError(f"unsupported test interpretation: {interpretation}")

    qualification = hmc_warmup._StartBankQualificationDiagnostic(
        authoritative=authoritative,
        shadow=shadow,
    ).public_payload()
    assert qualification["interpretation"] == interpretation
    return qualification


def _diagnostic_fixture_config() -> HMCKernelTuningConfig:
    return HMCKernelTuningConfig.diagnostic(
        target_scope="kernel_windowed_mass_toy_gaussian",
        seed=(20260818, 1401),
        chain_execution_mode="tf_function",
        use_xla=False,
        source="tests.start_bank_diagnostic",
    )


def _patch_diagnostic_prerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = _geometry()
    bootstrap = _bootstrap()
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "initialize_hmc_kernel_geometry",
        lambda **_kwargs: geometry,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_bootstrap_screen",
        lambda **_kwargs: bootstrap,
    )


@pytest.mark.parametrize(
    "interpretation",
    (
        "final_pass",
        "final_fail_shadow_pass",
        "both_fail",
        "post_selection_invariant_failure",
    ),
)
def test_diagnostic_entry_point_stops_at_callback_for_all_interpretations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interpretation: str,
) -> None:
    _patch_diagnostic_prerun(monkeypatch)
    qualification = _diagnostic_qualification(interpretation)
    after_callback_count = 0

    def windowed(**kwargs: Any) -> Any:
        nonlocal after_callback_count
        policy = kwargs["_attempt_budget_policy"]
        assert policy.attempt_index == 0
        assert policy.phase4_warmup_steps == 16
        kwargs["_private_diagnostic_callback"](
            "start_bank_qualification",
            {
                "stage": "windowed_mass_start_bank_boundary",
                "attempt_index": 0,
                "start_bank_qualification": qualification,
            },
        )
        after_callback_count += 1
        raise AssertionError("post-boundary sentinel was reached")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_windowed_mass_stage",
        windowed,
    )

    def forbidden_later_stage(**_kwargs: Any) -> Any:
        raise AssertionError("a post-boundary tuning stage was called")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_fixed_mass_step_stage",
        forbidden_later_stage,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_frozen_step_trajectory_stage",
        forbidden_later_stage,
    )

    output = tmp_path / f"diagnostic-{interpretation}"
    result = run_hmc_start_bank_diagnostic(
        adapter=_ToyGaussianAdapter(),
        initial_position=np.zeros(2),
        config=_diagnostic_fixture_config(),
        output_dir=output,
        parameter_scales=np.ones(2),
    )

    assert isinstance(result, HMCStartBankDiagnosticResult)
    assert result.diagnostic_valid is True
    assert result.start_bank_qualification == qualification
    assert result.start_bank_qualification_sha256 is not None
    assert after_callback_count == 0
    assert result.payload()["post_boundary_stage_counts"] == {
        "semantic_mass_diagnostic": 0,
        "fixed_mass_step": 0,
        "trajectory_selection": 0,
        "fresh_verification": 0,
        "final_kernel_construction": 0,
        "retained_sampling": 0,
    }

    result_path = output / "hmc_start_bank_diagnostic_result.json"
    event_path = (
        output
        / "private_diagnostics"
        / "hmc_start_bank_diagnostic_events.jsonl"
    )
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    event_rows = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(event_rows) == 1
    assert artifact["start_bank_qualification"] == event_rows[0][
        "start_bank_qualification"
    ]
    assert artifact["start_bank_qualification_sha256"] == event_rows[0][
        "start_bank_qualification_sha256"
    ]
    serialized = json.dumps({"artifact": artifact, "event": event_rows[0]})
    for forbidden in (
        "canonical_states",
        "reference_states",
        "selected_row_indices",
        "pairwise_distances",
        "traceback",
    ):
        assert forbidden not in serialized


def test_diagnostic_entry_point_marks_shadow_failure_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_diagnostic_prerun(monkeypatch)
    authoritative = hmc_warmup._assess_private_start_bank(
        np.arange(8.0).reshape((4, 2)),
        scope="authoritative_final_window",
    ).diagnostic
    shadow = hmc_warmup._shadow_start_bank_failure_diagnostic(
        np.zeros((16, 2)),
        minimum_relative_separation=1.0e-4,
        failure_code="shadow_assessment_failure",
    )
    qualification = hmc_warmup._StartBankQualificationDiagnostic(
        authoritative=authoritative,
        shadow=shadow,
    ).public_payload()

    def windowed(**kwargs: Any) -> Any:
        kwargs["_private_diagnostic_callback"](
            "start_bank_qualification",
            {
                "stage": "windowed_mass_start_bank_boundary",
                "start_bank_qualification": qualification,
            },
        )
        raise AssertionError("post-boundary sentinel was reached")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_windowed_mass_stage",
        windowed,
    )
    result = run_hmc_start_bank_diagnostic(
        adapter=_ToyGaussianAdapter(),
        initial_position=np.zeros(2),
        config=_diagnostic_fixture_config(),
        output_dir=tmp_path / "shadow-failure",
        parameter_scales=np.ones(2),
    )

    assert result.diagnostic_valid is False
    assert result.final_status == "diagnostic_incomplete_shadow_failure"
    assert result.hard_vetoes == ("shadow_start_bank_diagnostic_failure",)
    assert result.private_event_count == 1


def test_diagnostic_entry_point_preserves_bounded_preboundary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_diagnostic_prerun(monkeypatch)

    def fail_before_boundary(**_kwargs: Any) -> Any:
        raise RuntimeError("secret state at /private/location")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_windowed_mass_stage",
        fail_before_boundary,
    )
    output = tmp_path / "preboundary-failure"
    result = run_hmc_start_bank_diagnostic(
        adapter=_ToyGaussianAdapter(),
        initial_position=np.zeros(2),
        config=_diagnostic_fixture_config(),
        output_dir=output,
        parameter_scales=np.ones(2),
    )

    assert result.final_status == "invalid_before_start_bank_boundary"
    assert result.start_bank_qualification is None
    assert result.private_event_count == 0
    assert not (
        output
        / "private_diagnostics"
        / "hmc_start_bank_diagnostic_events.jsonl"
    ).exists()
    serialized = (
        output / "hmc_start_bank_diagnostic_result.json"
    ).read_text(encoding="utf-8")
    assert "secret state" not in serialized
    assert "/private/location" not in serialized
    assert "RuntimeError" in serialized


def test_diagnostic_entry_point_rejects_invalid_payload_before_event_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_diagnostic_prerun(monkeypatch)

    def invalid_payload(**kwargs: Any) -> Any:
        kwargs["_private_diagnostic_callback"](
            "start_bank_qualification",
            {
                "stage": "windowed_mass_start_bank_boundary",
                "start_bank_qualification": {
                    "schema": "forged.start_bank.v1",
                    "raw_states": [[1.0, 2.0]],
                },
            },
        )
        raise AssertionError("post-boundary sentinel was reached")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_hmc_windowed_mass_stage",
        invalid_payload,
    )
    output = tmp_path / "invalid-payload"
    result = run_hmc_start_bank_diagnostic(
        adapter=_ToyGaussianAdapter(),
        initial_position=np.zeros(2),
        config=_diagnostic_fixture_config(),
        output_dir=output,
        parameter_scales=np.ones(2),
    )

    assert result.final_status == "invalid_before_start_bank_boundary"
    assert result.start_bank_qualification is None
    assert result.private_event_count == 0
    assert not (
        output
        / "private_diagnostics"
        / "hmc_start_bank_diagnostic_events.jsonl"
    ).exists()
    serialized = (
        output / "hmc_start_bank_diagnostic_result.json"
    ).read_text(encoding="utf-8")
    assert "raw_states" not in serialized


def test_diagnostic_entry_point_refuses_existing_output_before_prerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_geometry(**_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError("geometry should not run after collision")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "initialize_hmc_kernel_geometry",
        forbidden_geometry,
    )
    output = tmp_path / "existing-output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="output already exists"):
        run_hmc_start_bank_diagnostic(
            adapter=_ToyGaussianAdapter(),
            initial_position=np.zeros(2),
            config=_diagnostic_fixture_config(),
            output_dir=output,
            parameter_scales=np.ones(2),
        )
    assert calls == 0


class _P4IdentityNestedTransform:
    def position_to_latent(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=float)

    def latent_to_position(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=float)


class _P4BaseAdapter:
    def __init__(self, signature: str) -> None:
        self._signature = signature

    def adapter_signature(self) -> str:
        return self._signature


class _P4NestedAdapter:
    def __init__(
        self,
        signature: str,
        *,
        target_signature: str = "p4-phase7-fixture-target",
        target_scope: str = "p4_phase7_fixture",
    ) -> None:
        self.transform = _P4IdentityNestedTransform()
        self._signature = signature
        self.base_adapter = _P4BaseAdapter(target_signature)
        self.target_scope = target_scope

    def adapter_signature(self) -> str:
        return self._signature

    def latent_to_position(self, value: Any) -> np.ndarray:
        return self.transform.latent_to_position(value)


# Source-site tuple fields are: ID, relative path, qualified owner, AST node
# type, required ``ast.unparse`` fragments, zero-based source occurrence,
# site kind, terminal consumer, key template, and upstream gate. Keeping this
# as literal-only data lets the retained manifest generator read it with
# ``ast.literal_eval`` without importing BayesFilter or TensorFlow.
_G1A_SOURCE_SITE_SPECS_RAW = (
    (
        "hmc_warmup.run_operational_windowed_warmup.initial_epsilon_seed_derivation.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("_seed(normalized_seed, -1, lane=1)",),
        0,
        "derivation",
        None,
        None,
        None,
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.initial_epsilon_seed_gate.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("consume_g2_seed(", "_G2_INITIAL_EPSILON_SEED_GATE_SITE_ID"),
        0,
        "terminal_consumption_gate",
        "hmc_runner_interface",
        "operational_warmup/reasonable_epsilon/initial",
        None,
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.initial_registry_dispatch.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup.consume_g2_seed",
        "Call",
        ("_g2_seed_use_registry.consume(",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.initial_epsilon_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.initial_reasonable_seed_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "keyword",
        ("seed=initial_reasonable_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.initial_epsilon_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.initial_seed_normalization.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("_strict_seed(seed, name='seed')",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.initial_epsilon_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.metric_seed_derivation.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("_seed(normalized_seed, window.index, lane=3)",),
        0,
        "derivation",
        None,
        None,
        None,
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.metric_seed_gate.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("consume_g2_seed(", "_G2_METRIC_BOUNDARY_SEED_GATE_SITE_ID"),
        0,
        "terminal_consumption_gate",
        "hmc_runner_interface",
        "operational_warmup/metric_boundary/<window>",
        None,
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.metric_registry_dispatch.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup.consume_g2_seed",
        "Call",
        ("_g2_seed_use_registry.consume(",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.metric_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.metric_reasonable_seed_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "keyword",
        ("seed=metric_boundary_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.metric_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.metric_seed_normalization.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("_strict_seed(seed, name='seed')",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.metric_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_derivation.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("_seed(normalized_seed, proposal_index)",),
        0,
        "derivation",
        None,
        None,
        None,
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("_g2_seed_use_registry.consume(", "_G2_REASONABLE_PROPOSAL_SEED_GATE_SITE_ID"),
        0,
        "terminal_consumption_gate",
        "tensorflow_stateless_rng",
        "operational_warmup/<context>/<index>/proposal/<proposal>",
        None,
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_list_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("proposal_seed_list.append(proposal_seed)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_tuple_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("tuple(proposal_seed_list)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_tensor_construction.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("tf.constant(proposal_seed, dtype=tf.int32)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_one_step_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon",
        "Call",
        ("one_step(tf.constant(proposal_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_kernel_seed_conversion.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon.one_step",
        "Call",
        ("tf.convert_to_tensor(proposal_seed, tf.int32)",),
        1,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_kernel_seed_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon.one_step",
        "keyword",
        ("seed=tf.convert_to_tensor(proposal_seed, tf.int32)",),
        1,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
    ),
    (
        "hmc_warmup.find_reasonable_epsilon.proposal_kernel_one_step_rng_call.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "find_reasonable_epsilon.one_step",
        "Call",
        ("kernel.one_step(", "seed=tf.convert_to_tensor(proposal_seed"),
        1,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.find_reasonable_epsilon.proposal_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_derivation.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("_seed(normalized_seed, segment_seed_index, lane=2)",),
        0,
        "derivation",
        None,
        None,
        None,
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_gate.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("consume_g2_seed(", "_G2_SEGMENT_SEED_GATE_SITE_ID"),
        0,
        "terminal_consumption_gate",
        "tfp_sample_chain",
        "operational_warmup/window/<window>/segment/<segment>",
        None,
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.segment_registry_dispatch.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup.consume_g2_seed",
        "Call",
        ("_g2_seed_use_registry.consume(",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_tensor_construction.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("tf.constant(segment_seed, dtype=tf.int32)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.segment_runner_seed_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("active_runner(", "tf.constant(segment_seed"),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.sample_chain_rng_call.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup.run_window",
        "Call",
        ("tfp.mcmc.sample_chain(", "seed=run_seed"),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.sample_chain_seed_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup.run_window",
        "keyword",
        ("seed=run_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.run_operational_windowed_warmup.segment_seed_gate.v1",
    ),
    (
        "hmc_warmup.build_phase7_engineering_probe_bank.p4_seed_derivation.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "Phase7EngineeringProbeBankConfig.derived_seed",
        "Call",
        ("tuple((int.from_bytes", "2147483647"),
        0,
        "derivation",
        None,
        None,
        None,
    ),
    (
        "hmc_warmup.build_phase7_engineering_probe_bank.p4_seed_gate.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "build_phase7_engineering_probe_bank",
        "Call",
        ("seed_use_registry.consume(", "_G2_P4_SEED_GATE_SITE_ID"),
        0,
        "terminal_consumption_gate",
        "tensorflow_stateless_rng",
        "p4/engineering_probe",
        None,
    ),
    (
        "hmc_warmup.build_phase7_engineering_probe_bank.offset_sampler_seed_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "build_phase7_engineering_probe_bank",
        "Call",
        ("sampler((config.chain_count, dimension), consumed_p4_seed)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.build_phase7_engineering_probe_bank.p4_seed_gate.v1",
    ),
    (
        "hmc_warmup.build_phase7_engineering_probe_bank.p4_seed_tensor_conversion.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "build_phase7_engineering_probe_bank",
        "Call",
        ("tf.convert_to_tensor(seed, dtype=tf.int32)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.build_phase7_engineering_probe_bank.p4_seed_gate.v1",
    ),
    (
        "hmc_warmup.build_phase7_engineering_probe_bank.stateless_normal_seed_pass_through.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "build_phase7_engineering_probe_bank",
        "keyword",
        ("seed=tf.convert_to_tensor(seed, dtype=tf.int32)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.build_phase7_engineering_probe_bank.p4_seed_gate.v1",
    ),
    (
        "hmc_warmup.build_phase7_engineering_probe_bank.stateless_normal_rng_call.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "build_phase7_engineering_probe_bank",
        "Call",
        ("tf.random.stateless_normal(", "seed=tf.convert_to_tensor(seed"),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_warmup.build_phase7_engineering_probe_bank.p4_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_derivation.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "run_hmc_bootstrap_screen",
        "Call",
        ("_round_seed(root_seed, round_index)",),
        0,
        "derivation",
        None,
        None,
        None,
    ),
    (
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "run_hmc_bootstrap_screen",
        "Call",
        ("_g2_seed_use_registry.consume(", "_G2_BOOTSTRAP_ROUND_SEED_GATE_SITE_ID"),
        0,
        "terminal_consumption_gate",
        "hmc_runner_interface",
        "bootstrap/round/<round>",
        None,
    ),
    (
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.screen_config_seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "run_hmc_bootstrap_screen",
        "keyword",
        ("seed=screen_seed",),
        1,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning._bootstrap_screen_config.seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_bootstrap_screen_config",
        "keyword",
        ("seed=seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc.bootstrap.FullChainHMCConfig.seed_normalization.v1",
        "bayesfilter/inference/hmc.py",
        "FullChainHMCConfig.__post_init__",
        "Call",
        ("tuple((int(item) for item in self.seed))",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.reusable_runner_config_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "run_hmc_bootstrap_screen",
        "Call",
        ("build_reusable_full_chain_tfp_hmc_runner(", "screen_config"),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.reusable_runner_seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "run_hmc_bootstrap_screen",
        "keyword",
        ("seed=screen_config.seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc.bootstrap.ReusableFullChainHMCRunner.run_seed_selection.v1",
        "bayesfilter/inference/hmc.py",
        "ReusableFullChainHMCRunner.run",
        "Assign",
        ("seed_value = self.config.seed if seed is None else seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc.bootstrap.ReusableFullChainHMCRunner.run_seed_conversion.v1",
        "bayesfilter/inference/hmc.py",
        "ReusableFullChainHMCRunner.run",
        "Call",
        ("tf.convert_to_tensor(seed_value, dtype=tf.int32)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc.bootstrap.ReusableFullChainHMCRunner.compiled_runner_seed_pass_through.v1",
        "bayesfilter/inference/hmc.py",
        "ReusableFullChainHMCRunner.run",
        "Call",
        ("self._runner(state_tensor, seed_tensor, step_tensor)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc.bootstrap.ReusableFullChainHMCRunner.sample_chain_rng_call.v1",
        "bayesfilter/inference/hmc.py",
        "ReusableFullChainHMCRunner._build_runner.run_chain",
        "Call",
        ("tfm.sample_chain(", "seed=seed"),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc.bootstrap.ReusableFullChainHMCRunner.sample_chain_seed_pass_through.v1",
        "bayesfilter/inference/hmc.py",
        "ReusableFullChainHMCRunner._build_runner.run_chain",
        "keyword",
        ("seed=seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning.run_hmc_bootstrap_screen.round_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_derivation.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_run_p4_windowed_boundary_attempt",
        "Call",
        ("_derive_seed(config.seed, stage_index=0)",),
        0,
        "derivation",
        None,
        None,
        None,
    ),
    (
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_run_p4_windowed_boundary_attempt",
        "Call",
        ("registry.consume(", "_G2_WINDOWED_STAGE_SEED_GATE_SITE_ID"),
        0,
        "terminal_consumption_gate",
        "hmc_runner_interface",
        "phase4/stage",
        None,
    ),
    (
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.diagnostic_config_seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_run_p4_windowed_boundary_attempt",
        "keyword",
        ("seed=stage_seed",),
        1,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning._windowed_stage_diagnostic_run_config.seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_windowed_stage_diagnostic_run_config",
        "keyword",
        ("seed=seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc.windowed_stage.FullChainHMCConfig.seed_normalization.v1",
        "bayesfilter/inference/hmc.py",
        "FullChainHMCConfig.__post_init__",
        "Call",
        ("tuple((int(item) for item in self.seed))",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning._operational_windowed_mass_capture.seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_run_p4_windowed_boundary_attempt",
        "keyword",
        ("stage_seed=stage_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning._operational_windowed_mass_capture.engineering_config_seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_operational_windowed_mass_capture",
        "keyword",
        ("root_seed=stage_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_warmup.Phase7EngineeringProbeBankConfig.root_seed_normalization.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "Phase7EngineeringProbeBankConfig.__post_init__",
        "Call",
        ("_strict_seed(self.root_seed, name='root_seed')",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_kernel_tuning._operational_windowed_mass_capture.warmup_seed_pass_through.v1",
        "bayesfilter/inference/hmc_kernel_tuning.py",
        "_operational_windowed_mass_capture",
        "keyword",
        ("seed=stage_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.root_seed_validation.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("_strict_builtin_seed(seed, name='seed')",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.initial_kernel_state_seed_lineage.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "keyword",
        ("seed_lineage=normalized_seed",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_coordinates.KernelState.windowed_stage_seed_lineage_tuple_normalization.v1",
        "bayesfilter/inference/hmc_coordinates.py",
        "KernelState.__post_init__",
        "Call",
        ("tuple(self.seed_lineage)",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_coordinates.KernelState.windowed_stage_seed_lineage_integer_normalization.v1",
        "bayesfilter/inference/hmc_coordinates.py",
        "KernelState.__post_init__",
        "Call",
        ("tuple((_positive_int(item, name='seed_lineage item'",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.final_kernel_state_seed_lineage.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "keyword",
        ("seed_lineage=normalized_seed",),
        1,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_warmup.run_operational_windowed_warmup.final_kernel_state_with_epsilon_call.v1",
        "bayesfilter/inference/hmc_warmup.py",
        "run_operational_windowed_warmup",
        "Call",
        ("kernel_state.with_epsilon(",),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
    (
        "hmc_coordinates.KernelState.with_epsilon_seed_lineage_forwarding.v1",
        "bayesfilter/inference/hmc_coordinates.py",
        "KernelState.with_epsilon",
        "Call",
        ("replace(self,", "epsilon=float(epsilon)"),
        0,
        "read_only_pass_through",
        None,
        None,
        "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1",
    ),
)


_G1A_BAYESFILTER_ROOT = Path("/home/ubuntu/python/BayesFilter")
_G1A_SOURCE_COVERAGE_MANIFEST = Path(
    "/home/ubuntu/python/MacroFinance/docs/reviews/"
    "daily_asset_midas_identifiable_multi_asset_expansion_phase_14_"
    "g1a_seed_source_coverage_manifest_2026_08_24.json"
)


def _g1a_owner_by_node(tree: ast.AST) -> dict[ast.AST, str]:
    owners: dict[ast.AST, str] = {}

    def visit(node: ast.AST, stack: tuple[str, ...]) -> None:
        nested = stack
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            nested = (*stack, node.name)
        owners[node] = ".".join(nested)
        for child in ast.iter_child_nodes(node):
            visit(child, nested)

    visit(tree, ())
    return owners


def _g1a_manifest_sites_from_final_sources() -> tuple[dict[str, object], ...]:
    trees: dict[str, ast.AST] = {}
    owners: dict[str, dict[ast.AST, str]] = {}
    rows: list[dict[str, object]] = []
    for (
        site_id,
        relative_path,
        owner_qualname,
        node_type_name,
        fragments,
        occurrence,
        site_kind,
        terminal_consumer,
        registry_key_template,
        upstream_gate_site_id,
    ) in _G1A_SOURCE_SITE_SPECS_RAW:
        tree = trees.setdefault(
            relative_path,
            ast.parse(
                (_G1A_BAYESFILTER_ROOT / relative_path).read_text(encoding="utf-8")
            ),
        )
        owner_map = owners.setdefault(relative_path, _g1a_owner_by_node(tree))
        node_type = getattr(ast, node_type_name)
        candidates = []
        for node in ast.walk(tree):
            if type(node) is not node_type or owner_map[node] != owner_qualname:
                continue
            expression = " ".join(ast.unparse(node).split())
            if all(fragment in expression for fragment in fragments):
                candidates.append(node)
        candidates.sort(
            key=lambda node: (
                int(getattr(node, "lineno", -1)),
                int(getattr(node, "col_offset", -1)),
            )
        )
        assert occurrence < len(candidates), (site_id, candidates)
        node = candidates[occurrence]
        node_digest = hashlib.sha256(
            ast.dump(
                node,
                annotate_fields=True,
                include_attributes=False,
            ).encode("ascii")
        ).hexdigest()
        rows.append(
            {
                "site_id": site_id,
                "source_path": str(_G1A_BAYESFILTER_ROOT / relative_path),
                "owner_qualname": owner_qualname,
                "node_ast_sha256": node_digest,
                "site_kind": site_kind,
                "terminal_consumer": terminal_consumer,
                "registry_key_template": registry_key_template,
                "upstream_gate_site_id": upstream_gate_site_id,
            }
        )
    return tuple(sorted(rows, key=lambda row: str(row["site_id"])))


def _g1a_canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def test_g1a_source_coverage_manifest_binds_every_preboundary_seed_site() -> None:
    retained = _G1A_SOURCE_COVERAGE_MANIFEST.read_bytes()
    assert retained == retained.rstrip(b"\n")
    payload = json.loads(retained.decode("ascii"))
    assert set(payload) == {
        "schema",
        "scope",
        "source_files",
        "sites",
        "excluded_post_boundary",
        "generator_contract",
        "manifest_content_sha256",
    }
    assert payload["schema"] == (
        "bayesfilter.hmc_g2_preboundary_seed_source_coverage.v1"
    )
    assert payload["scope"] == "g2_reachable_through_p4_boundary_only"
    assert payload["excluded_post_boundary"] == {
        "scope": "after_p4_boundary",
        "reason": "not_executed_or_claimed_by_g2",
        "future_review_required": True,
    }
    assert payload["generator_contract"] == {
        "python_ast_only": True,
        "no_project_import": True,
        "source_files_final_before_generation": True,
        "canonical_json": True,
    }

    source_paths = {
        str(_G1A_BAYESFILTER_ROOT / "bayesfilter/inference/hmc_warmup.py"):
            "writable_g1a",
        str(_G1A_BAYESFILTER_ROOT / "bayesfilter/inference/hmc_kernel_tuning.py"):
            "writable_g1a",
        str(_G1A_BAYESFILTER_ROOT / "bayesfilter/inference/hmc.py"):
            "read_only_interface",
        str(_G1A_BAYESFILTER_ROOT / "bayesfilter/inference/hmc_coordinates.py"):
            "read_only_interface",
    }
    expected_source_files = tuple(
        sorted(
            (
                {
                    "path": path,
                    "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "write_scope": write_scope,
                }
                for path, write_scope in source_paths.items()
            ),
            key=lambda row: row["path"],
        )
    )
    assert tuple(payload["source_files"]) == expected_source_files
    assert tuple(payload["sites"]) == _g1a_manifest_sites_from_final_sources()

    unsigned = dict(payload)
    content_digest = unsigned.pop("manifest_content_sha256")
    assert content_digest == hashlib.sha256(
        _g1a_canonical_json_bytes(unsigned)
    ).hexdigest()
    assert retained == _g1a_canonical_json_bytes(payload)
    retained_digest = hashlib.sha256(retained).hexdigest()
    assert retained_digest != content_digest
    for source_path in source_paths:
        source_text = Path(source_path).read_text(encoding="utf-8")
        assert content_digest not in source_text
        assert retained_digest not in source_text

    site_ids = tuple(row["site_id"] for row in payload["sites"])
    assert len(site_ids) == 60
    assert len(set(site_ids)) == 60
    declared_ids = {
        hmc_warmup._G2_P4_SEED_DERIVATION_SITE_ID,
        hmc_warmup._G2_P4_SEED_GATE_SITE_ID,
        hmc_warmup._G2_INITIAL_EPSILON_SEED_DERIVATION_SITE_ID,
        hmc_warmup._G2_INITIAL_EPSILON_SEED_GATE_SITE_ID,
        hmc_warmup._G2_SEGMENT_SEED_DERIVATION_SITE_ID,
        hmc_warmup._G2_SEGMENT_SEED_GATE_SITE_ID,
        hmc_warmup._G2_METRIC_BOUNDARY_SEED_DERIVATION_SITE_ID,
        hmc_warmup._G2_METRIC_BOUNDARY_SEED_GATE_SITE_ID,
        hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_DERIVATION_SITE_ID,
        hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_GATE_SITE_ID,
        hmc_kernel_tuning._G2_BOOTSTRAP_ROUND_SEED_DERIVATION_SITE_ID,
        hmc_kernel_tuning._G2_BOOTSTRAP_ROUND_SEED_GATE_SITE_ID,
        hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_DERIVATION_SITE_ID,
        hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_GATE_SITE_ID,
        *hmc_warmup._G2_P4_SEED_INTERFACE_HOPS,
        *hmc_warmup._G2_INITIAL_EPSILON_SEED_INTERFACE_HOPS,
        *hmc_warmup._G2_SEGMENT_SEED_INTERFACE_HOPS,
        *hmc_warmup._G2_METRIC_BOUNDARY_SEED_INTERFACE_HOPS,
        *hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_INTERFACE_HOPS,
        *hmc_kernel_tuning._G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS,
        *hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_INTERFACE_HOPS,
    }
    assert set(site_ids) == declared_ids

    gate_ids = {
        row["site_id"]
        for row in payload["sites"]
        if row["site_kind"] == "terminal_consumption_gate"
    }
    assert len(gate_ids) == 7
    for row in payload["sites"]:
        if row["site_kind"] == "read_only_pass_through":
            assert row["upstream_gate_site_id"] in gate_ids
        else:
            assert row["upstream_gate_site_id"] is None

    # The exact P4 route excludes the ordinary stage seed branch and never
    # invokes KernelState.remap. Geometry only reports reserved lineage seeds;
    # it has no RNG consumer, so none of these are registry entries.
    assert all(
        row["owner_qualname"] != "run_hmc_windowed_mass_stage"
        for row in payload["sites"]
    )
    for relative_path in (
        "bayesfilter/inference/hmc_warmup.py",
        "bayesfilter/inference/hmc_kernel_tuning.py",
    ):
        tree = ast.parse(
            (_G1A_BAYESFILTER_ROOT / relative_path).read_text(encoding="utf-8")
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
                and node.func.attr == "remap"
                for node in ast.walk(tree)
            )

    # Discover seed-lineage-preserving KernelState helper calls independently
    # of the literal site list. Any new reachable replace/forwarding helper must
    # acquire both a caller site and a method-body site in the retained map.
    coordinates_tree = ast.parse(
        (_G1A_BAYESFILTER_ROOT / "bayesfilter/inference/hmc_coordinates.py").read_text(
            encoding="utf-8"
        )
    )
    kernel_state = next(
        node
        for node in coordinates_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KernelState"
    )
    forwarding_nodes: dict[str, ast.Call] = {}
    for method in kernel_state.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "replace"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            ):
                continue
            if not any(keyword.arg == "seed_lineage" for keyword in node.keywords):
                forwarding_nodes[method.name] = node

    warmup_tree = ast.parse(
        (_G1A_BAYESFILTER_ROOT / "bayesfilter/inference/hmc_warmup.py").read_text(
            encoding="utf-8"
        )
    )
    warmup_owners = _g1a_owner_by_node(warmup_tree)
    reachable_forwarding_calls = tuple(
        node
        for node in ast.walk(warmup_tree)
        if isinstance(node, ast.Call)
        and warmup_owners[node] == "run_operational_windowed_warmup"
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in forwarding_nodes
    )
    assert tuple(node.func.attr for node in reachable_forwarding_calls) == (
        "with_epsilon",
    )
    windowed_gate = hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_GATE_SITE_ID
    windowed_site_digests = {
        row["node_ast_sha256"]
        for row in payload["sites"]
        if row["upstream_gate_site_id"] == windowed_gate
    }
    for call in reachable_forwarding_calls:
        call_digest = hashlib.sha256(
            ast.dump(
                call,
                annotate_fields=True,
                include_attributes=False,
            ).encode("ascii")
        ).hexdigest()
        method_digest = hashlib.sha256(
            ast.dump(
                forwarding_nodes[call.func.attr],
                annotate_fields=True,
                include_attributes=False,
            ).encode("ascii")
        ).hexdigest()
        assert call_digest in windowed_site_digests
        assert method_digest in windowed_site_digests


def test_g1a_registry_counts_logical_leaves_not_interface_hops() -> None:
    manifest_bytes = _G1A_SOURCE_COVERAGE_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes.decode("ascii"))
    contracts = {
        row["site_id"]: {
            name: row[name]
            for name in (
                "site_id",
                "source_path",
                "owner_qualname",
                "site_kind",
                "terminal_consumer",
                "registry_key_template",
                "upstream_gate_site_id",
            )
        }
        for row in manifest["sites"]
    }
    registry = hmc_warmup.G2PreboundarySeedUseRegistry(
        source_coverage_artifact_sha256=hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        source_site_contracts=contracts,
    )
    logical_leaf_count = 0

    bootstrap_root = (101, 1000)
    for round_index in range(2):
        seed = hmc_kernel_tuning._round_seed(bootstrap_root, round_index)
        registry.consume(
            derivation_site_id=(
                hmc_kernel_tuning._G2_BOOTSTRAP_ROUND_SEED_DERIVATION_SITE_ID
            ),
            terminal_gate_site_id=(
                hmc_kernel_tuning._G2_BOOTSTRAP_ROUND_SEED_GATE_SITE_ID
            ),
            key=f"bootstrap/round/{round_index:02d}",
            owner_file="hmc_kernel_tuning.py",
            owner_qualname="run_hmc_bootstrap_screen",
            terminal_consumer="hmc_runner_interface",
            derivation={
                "kind": "round_offset",
                "base_key": "bootstrap/root",
                "round_index": round_index,
            },
            indices=({"name": "round_index", "value": round_index},),
            seed=seed,
            interface_hop_site_ids=(
                hmc_kernel_tuning._G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS
            ),
        )
        logical_leaf_count += 1

    stage_seed = hmc_kernel_tuning._derive_seed((202, 2000), stage_index=0)
    registry.consume(
        derivation_site_id=(
            hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_DERIVATION_SITE_ID
        ),
        terminal_gate_site_id=(
            hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_GATE_SITE_ID
        ),
        key="phase4/stage",
        owner_file="hmc_kernel_tuning.py",
        owner_qualname="_run_p4_windowed_boundary_attempt",
        terminal_consumer="hmc_runner_interface",
        derivation={
            "kind": "derive_stage",
            "base_key": "windowed_stage_config.seed",
            "stage_index": 0,
        },
        indices=(),
        seed=stage_seed,
        interface_hop_site_ids=(
            hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_INTERFACE_HOPS
        ),
    )
    logical_leaf_count += 1

    initial_seed = hmc_warmup._seed(stage_seed, -1, lane=1)
    registry.consume(
        derivation_site_id=hmc_warmup._G2_INITIAL_EPSILON_SEED_DERIVATION_SITE_ID,
        terminal_gate_site_id=hmc_warmup._G2_INITIAL_EPSILON_SEED_GATE_SITE_ID,
        key="operational_warmup/reasonable_epsilon/initial",
        owner_file="hmc_warmup.py",
        owner_qualname="run_operational_windowed_warmup",
        terminal_consumer="hmc_runner_interface",
        derivation={
            "kind": "warmup_index_lane",
            "base_key": "operational_warmup/root",
            "index": -1,
            "lane": 1,
        },
        indices=(),
        seed=initial_seed,
        interface_hop_site_ids=hmc_warmup._G2_INITIAL_EPSILON_SEED_INTERFACE_HOPS,
    )
    logical_leaf_count += 1

    def consume_proposals(base_seed: tuple[int, int], base_key: str) -> int:
        consumed = 0
        for proposal_index in range(4):
            proposal_seed = hmc_warmup._seed(base_seed, proposal_index)
            registry.consume(
                derivation_site_id=(
                    hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_DERIVATION_SITE_ID
                ),
                terminal_gate_site_id=(
                    hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_GATE_SITE_ID
                ),
                key=f"{base_key}/proposal/{proposal_index:02d}",
                owner_file="hmc_warmup.py",
                owner_qualname="find_reasonable_epsilon",
                terminal_consumer="tensorflow_stateless_rng",
                derivation={
                    "kind": "warmup_index_lane",
                    "base_key": base_key,
                    "index": proposal_index,
                    "lane": 0,
                },
                indices=({"name": "proposal_index", "value": proposal_index},),
                seed=proposal_seed,
                interface_hop_site_ids=(
                    hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_INTERFACE_HOPS
                ),
            )
            consumed += 1
        return consumed

    logical_leaf_count += consume_proposals(
        initial_seed,
        "operational_warmup/reasonable_epsilon/initial",
    )
    schedule = build_windowed_warmup_schedule(
        hmc_warmup.WindowedMassAdaptationConfig(
            # One update window keeps this count-only fixture collision-free.
            # The live two-window arithmetic collision has its own fail-closed
            # regression below and must not be hidden by this schema test.
            warmup_steps=8,
            initial_buffer=2,
            final_buffer=2,
            first_window_size=4,
            min_window_samples=2,
        )
    )
    segment_size = hmc_kernel_tuning._OPERATIONAL_WARMUP_SEGMENT_SIZE
    for window in schedule:
        segment_count = (window.length + segment_size - 1) // segment_size
        for segment_index in range(segment_count):
            seed_index = window.index * 100_000 + segment_index
            segment_seed = hmc_warmup._seed(stage_seed, seed_index, lane=2)
            registry.consume(
                derivation_site_id=hmc_warmup._G2_SEGMENT_SEED_DERIVATION_SITE_ID,
                terminal_gate_site_id=hmc_warmup._G2_SEGMENT_SEED_GATE_SITE_ID,
                key=(
                    f"operational_warmup/window/{window.index:02d}/"
                    f"segment/{segment_index:02d}"
                ),
                owner_file="hmc_warmup.py",
                owner_qualname="run_operational_windowed_warmup",
                terminal_consumer="tfp_sample_chain",
                derivation={
                    "kind": "warmup_index_lane",
                    "base_key": "operational_warmup/root",
                    "index": seed_index,
                    "lane": 2,
                },
                indices=(
                    {"name": "window_index", "value": window.index},
                    {"name": "segment_index", "value": segment_index},
                ),
                seed=segment_seed,
                interface_hop_site_ids=hmc_warmup._G2_SEGMENT_SEED_INTERFACE_HOPS,
            )
            logical_leaf_count += 1
        if window.update_mass:
            metric_seed = hmc_warmup._seed(stage_seed, window.index, lane=3)
            registry.consume(
                derivation_site_id=(
                    hmc_warmup._G2_METRIC_BOUNDARY_SEED_DERIVATION_SITE_ID
                ),
                terminal_gate_site_id=(
                    hmc_warmup._G2_METRIC_BOUNDARY_SEED_GATE_SITE_ID
                ),
                key=f"operational_warmup/metric_boundary/{window.index:02d}",
                owner_file="hmc_warmup.py",
                owner_qualname="run_operational_windowed_warmup",
                terminal_consumer="hmc_runner_interface",
                derivation={
                    "kind": "warmup_index_lane",
                    "base_key": "operational_warmup/root",
                    "index": window.index,
                    "lane": 3,
                },
                indices=({"name": "window_index", "value": window.index},),
                seed=metric_seed,
                interface_hop_site_ids=(
                    hmc_warmup._G2_METRIC_BOUNDARY_SEED_INTERFACE_HOPS
                ),
            )
            logical_leaf_count += 1
            logical_leaf_count += consume_proposals(
                metric_seed,
                f"operational_warmup/metric_boundary/{window.index:02d}",
            )

    p4_config = hmc_warmup.Phase7EngineeringProbeBankConfig(
        chain_count=4,
        covariance_multiplier=2.0,
        root_seed=stage_seed,
    )
    registry.consume(
        derivation_site_id=hmc_warmup._G2_P4_SEED_DERIVATION_SITE_ID,
        terminal_gate_site_id=hmc_warmup._G2_P4_SEED_GATE_SITE_ID,
        key="p4/engineering_probe",
        owner_file="hmc_warmup.py",
        owner_qualname="build_phase7_engineering_probe_bank",
        terminal_consumer="tensorflow_stateless_rng",
        derivation={
            "kind": "p4_domain_hash",
            "base_key": "engineering_probe_config.root_seed",
            "domain_label": hmc_warmup._PHASE7_ENGINEERING_PROBE_SEED_DOMAIN,
        },
        indices=(),
        seed=p4_config.derived_seed,
        interface_hop_site_ids=hmc_warmup._G2_P4_SEED_INTERFACE_HOPS,
        is_p4=True,
    )
    logical_leaf_count += 1
    payload = registry.complete_payload()
    assert len(payload["entries"]) == logical_leaf_count
    assert sum(row["consumption_count"] for row in payload["entries"]) == (
        logical_leaf_count
    )
    assert payload["preboundary_consumed_seed_count"] == logical_leaf_count - 1
    assert payload["entries"][-1]["key"] == "p4/engineering_probe"
    assert payload["entries"][-1]["is_p4"] is True
    unsigned = dict(payload)
    signature = unsigned.pop("seed_use_registry_signature")
    assert signature == hashlib.sha256(
        _g1a_canonical_json_bytes(unsigned)
    ).hexdigest()


def test_g1a_metric_boundary_seed_collision_fails_before_second_use() -> None:
    """Lock the current two-window collision as a pre-RNG shared invalidity."""

    manifest_bytes = _G1A_SOURCE_COVERAGE_MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes.decode("ascii"))
    contracts = {
        row["site_id"]: {
            name: row[name]
            for name in (
                "site_id",
                "source_path",
                "owner_qualname",
                "site_kind",
                "terminal_consumer",
                "registry_key_template",
                "upstream_gate_site_id",
            )
        }
        for row in manifest["sites"]
    }
    registry = hmc_warmup.G2PreboundarySeedUseRegistry(
        source_coverage_artifact_sha256=hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        source_site_contracts=contracts,
    )
    root_seed = (202, 11176)
    first_metric_seed = hmc_warmup._seed(root_seed, 1, lane=3)
    first_proposal_seed = hmc_warmup._seed(first_metric_seed, 0)
    second_metric_seed = hmc_warmup._seed(root_seed, 2, lane=3)
    assert first_proposal_seed == second_metric_seed

    registry.consume(
        derivation_site_id=hmc_warmup._G2_METRIC_BOUNDARY_SEED_DERIVATION_SITE_ID,
        terminal_gate_site_id=hmc_warmup._G2_METRIC_BOUNDARY_SEED_GATE_SITE_ID,
        key="operational_warmup/metric_boundary/01",
        owner_file="hmc_warmup.py",
        owner_qualname="run_operational_windowed_warmup",
        terminal_consumer="hmc_runner_interface",
        derivation={
            "kind": "warmup_index_lane",
            "base_key": "operational_warmup/root",
            "index": 1,
            "lane": 3,
        },
        indices=({"name": "window_index", "value": 1},),
        seed=first_metric_seed,
        interface_hop_site_ids=hmc_warmup._G2_METRIC_BOUNDARY_SEED_INTERFACE_HOPS,
    )
    registry.consume(
        derivation_site_id=(
            hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_DERIVATION_SITE_ID
        ),
        terminal_gate_site_id=(
            hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_GATE_SITE_ID
        ),
        key="operational_warmup/metric_boundary/01/proposal/00",
        owner_file="hmc_warmup.py",
        owner_qualname="find_reasonable_epsilon",
        terminal_consumer="tensorflow_stateless_rng",
        derivation={
            "kind": "warmup_index_lane",
            "base_key": "operational_warmup/metric_boundary/01",
            "index": 0,
            "lane": 0,
        },
        indices=({"name": "proposal_index", "value": 0},),
        seed=first_proposal_seed,
        interface_hop_site_ids=(
            hmc_warmup._G2_REASONABLE_PROPOSAL_SEED_INTERFACE_HOPS
        ),
    )
    with pytest.raises(hmc_warmup._G2SeedRegistryError) as raised:
        registry.consume(
            derivation_site_id=(
                hmc_warmup._G2_METRIC_BOUNDARY_SEED_DERIVATION_SITE_ID
            ),
            terminal_gate_site_id=hmc_warmup._G2_METRIC_BOUNDARY_SEED_GATE_SITE_ID,
            key="operational_warmup/metric_boundary/02",
            owner_file="hmc_warmup.py",
            owner_qualname="run_operational_windowed_warmup",
            terminal_consumer="hmc_runner_interface",
            derivation={
                "kind": "warmup_index_lane",
                "base_key": "operational_warmup/root",
                "index": 2,
                "lane": 3,
            },
            indices=({"name": "window_index", "value": 2},),
            seed=second_metric_seed,
            interface_hop_site_ids=(
                hmc_warmup._G2_METRIC_BOUNDARY_SEED_INTERFACE_HOPS
            ),
        )
    carrier = hmc_warmup.g2_preboundary_shared_invalidity_exception(
        registry,
        stage=hmc_warmup._G2_METRIC_BOUNDARY_SEED_GATE_SITE_ID,
        cause=raised.value,
    )
    rollback_candidate = hmc_warmup.MetricAdequacyDecision(
        outcome="dense_update",
        covariance=np.eye(2),
        estimator_family="g1a_injected_fixture",
        report={"metric_evidence": "deterministic_injected"},
    )
    with pytest.raises(ValueError) as propagated:
        hmc_warmup._rejected_metric_candidate(
            rollback_candidate,
            stage="reasonable_epsilon",
            error=carrier,
        )
    assert propagated.value is carrier
    public = hmc_warmup.g2_preboundary_shared_invalidity_payload_from_exception(
        carrier
    )
    snapshot = hmc_warmup.g2_seed_private_evidence_from_exception(carrier)
    assert public is not None
    assert public["failure_code"] == "seed_registry_preboundary_duplicate"
    assert public["p4_boundary_stage"] == "not_entered"
    assert snapshot is not None
    assert snapshot["failure_code"] == "seed_registry_preboundary_duplicate"
    assert snapshot["failure_stage"] == "preboundary"
    assert snapshot["preboundary_consumed_seed_count"] == 2
    assert snapshot["p4_seed_consumed"] is False


def _p4_seed_registry() -> hmc_warmup.G2PreboundarySeedUseRegistry:
    source_path = "bayesfilter/inference/hmc_warmup.py"
    owner = "build_phase7_engineering_probe_bank"
    contracts: dict[str, dict[str, object]] = {
        hmc_warmup._G2_P4_SEED_DERIVATION_SITE_ID: {
            "site_id": hmc_warmup._G2_P4_SEED_DERIVATION_SITE_ID,
            "source_path": source_path,
            "owner_qualname": "Phase7EngineeringProbeBankConfig.derived_seed",
            "site_kind": "derivation",
            "terminal_consumer": None,
            "registry_key_template": None,
            "upstream_gate_site_id": None,
        },
        hmc_warmup._G2_P4_SEED_GATE_SITE_ID: {
            "site_id": hmc_warmup._G2_P4_SEED_GATE_SITE_ID,
            "source_path": source_path,
            "owner_qualname": owner,
            "site_kind": "terminal_consumption_gate",
            "terminal_consumer": "tensorflow_stateless_rng",
            "registry_key_template": "p4/engineering_probe",
            "upstream_gate_site_id": None,
        },
    }
    for hop_id in hmc_warmup._G2_P4_SEED_INTERFACE_HOPS:
        contracts[hop_id] = {
            "site_id": hop_id,
            "source_path": source_path,
            "owner_qualname": owner,
            "site_kind": "read_only_pass_through",
            "terminal_consumer": None,
            "registry_key_template": None,
            "upstream_gate_site_id": hmc_warmup._G2_P4_SEED_GATE_SITE_ID,
        }
    return hmc_warmup.G2PreboundarySeedUseRegistry(
        source_coverage_artifact_sha256="b" * 64,
        source_site_contracts=contracts,
    )


def _p4_fixed_offsets(
    shape: tuple[int, int],
    _seed: tuple[int, int],
) -> np.ndarray:
    assert shape == (4, 2)
    return np.array(
        [[1.0, 0.2], [-1.0, 0.2], [0.3, 1.0], [0.3, -1.0]],
        dtype=float,
    )


def _g1a_stage_seed_registry(
    *,
    include_stage_gate: bool = True,
) -> hmc_warmup.G2PreboundarySeedUseRegistry:
    derivation_id = hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_DERIVATION_SITE_ID
    gate_id = hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_GATE_SITE_ID
    contracts: dict[str, dict[str, object]] = {
        derivation_id: {
            "site_id": derivation_id,
            "source_path": "bayesfilter/inference/hmc_kernel_tuning.py",
            "owner_qualname": "_run_p4_windowed_boundary_attempt",
            "site_kind": "derivation",
            "terminal_consumer": None,
            "registry_key_template": None,
            "upstream_gate_site_id": None,
        },
    }
    if include_stage_gate:
        contracts[gate_id] = {
            "site_id": gate_id,
            "source_path": "bayesfilter/inference/hmc_kernel_tuning.py",
            "owner_qualname": "_run_p4_windowed_boundary_attempt",
            "site_kind": "terminal_consumption_gate",
            "terminal_consumer": "hmc_runner_interface",
            "registry_key_template": "phase4/stage",
            "upstream_gate_site_id": None,
        }
        for hop_id in hmc_kernel_tuning._G2_WINDOWED_STAGE_SEED_INTERFACE_HOPS:
            contracts[hop_id] = {
                "site_id": hop_id,
                "source_path": (
                    "bayesfilter/inference/hmc.py"
                    if hop_id.startswith("hmc.")
                    else "bayesfilter/inference/hmc_coordinates.py"
                    if hop_id.startswith("hmc_coordinates.")
                    else "bayesfilter/inference/hmc_warmup.py"
                    if hop_id.startswith("hmc_warmup.")
                    else "bayesfilter/inference/hmc_kernel_tuning.py"
                ),
                "owner_qualname": "read_only_seed_pass_through",
                "site_kind": "read_only_pass_through",
                "terminal_consumer": None,
                "registry_key_template": None,
                "upstream_gate_site_id": gate_id,
            }
    return hmc_warmup.G2PreboundarySeedUseRegistry(
        source_coverage_artifact_sha256="e" * 64,
        source_site_contracts=contracts,
    )


def _p4_operational_fixture(*, policy_id: str | None = None) -> Any:
    estimate = PositionCovarianceEstimate(
        center=np.zeros(2),
        covariance=np.eye(2),
        source_coordinate_signature="p4-phase7-fixture-source",
        estimator_family="deterministic_test_fixture",
        state_count=4,
        effective_rank=2,
        regularization_report={"method": "none"},
        adequacy_report={"passed": True},
    )
    transform = AffineCoordinateTransform.from_covariance_estimate(estimate)
    state = KernelState(
        canonical_theta=np.zeros(2),
        active_latent=np.zeros(2),
        transform=transform,
        momentum_metric=MomentumMetric.identity_for(transform),
        epsilon=None,
        trajectory_policy=WarmupTrajectoryPolicy(3, 16),
        adaptation_generation=1,
        seed_lineage=(20260821, 30),
        evidence_status="p4_phase7_fixture",
    ).with_epsilon(0.1, evidence_status="p4_phase7_fixture_frozen")
    config = hmc_warmup.Phase7EngineeringProbeBankConfig(
        chain_count=4,
        covariance_multiplier=2.0,
        root_seed=(20260821, 9207),
    )
    seed_registry = _p4_seed_registry()
    build = hmc_warmup.build_phase7_engineering_probe_bank(
        final_kernel_state=state,
        config=config,
        position_covariance_estimate_signature=estimate.signature,
        p4_transform_signature=transform.signature,
        applied_metric_update_count=state.adaptation_generation,
        seed_use_registry=seed_registry,
        target_signature=hmc_warmup._phase7_engineering_probe_target_signature(
            _P4BaseAdapter("p4-phase7-fixture-target")
        ),
        target_health_fn=lambda candidates: {
            "shared_invalidity_reasons": (),
            "candidate_data_invalidity_reasons": (),
            "target_value_finite": True,
            "target_score_finite": True,
            "target_status_failure_count": 0,
            "evaluated_draw_count": int(tf.convert_to_tensor(candidates).shape[0]),
        },
        _offset_sampler=_p4_fixed_offsets,
    )
    return SimpleNamespace(
        private_start_bank_policy_id=(
            hmc_warmup.PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID
            if policy_id is None
            else policy_id
        ),
        engineering_probe_bank_qualification=build.qualification,
        private_start_bank_theta=build.canonical_theta,
        private_start_bank_signature=build.qualification.content_signature,
        seed_root=(20260821, 9207),
        final_kernel_state=state,
        target_scope="p4_phase7_fixture",
        g2_seed_use_registry=seed_registry,
        g2_seed_use_registry_payload=build.seed_use_registry_payload,
    )


def test_p4_multiplier_is_explicit_private_and_propagates_without_hmc() -> None:
    stage = HMCWindowedMassStageConfig(
        seed=(987654321, 123456789),
        engineering_probe_covariance_multiplier=2.125
    )
    stage_payload = stage.payload()
    public = stage_payload["engineering_probe_bank"]
    serialized = json.dumps(stage_payload, sort_keys=True)
    assert public["configured"] is True
    assert public["policy_id"] == hmc_warmup.PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID
    assert public["private_config_signature"]
    assert public["covariance_multiplier_exposed"] is False
    assert stage_payload["seed"] is None
    assert stage_payload["seed_signature"]
    assert stage_payload["seed_values_exposed"] is False
    assert "987654321" not in serialized
    assert "123456789" not in serialized
    assert "2.125" not in serialized
    diagnostic_public = hmc_kernel_tuning._engineering_probe_diagnostic_config_public_payload(
        {
            "seed": (987654321, 123456789),
            "step_size": 0.1,
        },
        configured=True,
    )
    assert diagnostic_public is not None
    assert diagnostic_public["seed"] is None
    assert diagnostic_public["seed_signature"]
    assert "987654321" not in json.dumps(diagnostic_public, sort_keys=True)
    assert "123456789" not in json.dumps(diagnostic_public, sort_keys=True)
    assert HMCWindowedMassStageConfig().engineering_probe_covariance_multiplier is None

    for config_type in (
        HMCWindowedMassStageConfig,
        hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig,
        HMCKernelTuningConfig,
    ):
        for invalid in (0.0, -1.0, np.nan, np.inf, True):
            with pytest.raises(ValueError, match="positive and finite"):
                config_type(engineering_probe_covariance_multiplier=invalid)

    top = HMCKernelTuningConfig.diagnostic(
        engineering_probe_covariance_multiplier=2.125
    )
    loop = hmc_kernel_tuning._public_loop_config(top)
    propagated_stage = hmc_kernel_tuning._phase7_windowed_stage_config(
        loop,
        attempt_index=0,
    )
    assert loop.engineering_probe_covariance_multiplier == pytest.approx(2.125)
    assert propagated_stage.engineering_probe_covariance_multiplier == pytest.approx(
        2.125
    )

    for config_type in (
        hmc_kernel_tuning.HMCTuneVerifyRepairLoopConfig,
        HMCKernelTuningConfig,
    ):
        configured_payload = config_type(
            seed=(987654321, 123456789),
            engineering_probe_covariance_multiplier=2.125,
        ).payload()
        assert configured_payload["seed"] is None
        assert configured_payload["seed_signature"]
        assert configured_payload["seed_values_exposed"] is False
        serialized_config = json.dumps(configured_payload, sort_keys=True)
        assert "987654321" not in serialized_config
        assert "123456789" not in serialized_config


def test_g1a_carrier_first_candidate_and_shared_results_are_closed() -> None:
    operational = _p4_operational_fixture()
    qualification = operational.engineering_probe_bank_qualification
    assert qualification is not None
    private = operational.g2_seed_use_registry_payload
    registry = operational.g2_seed_use_registry
    config = HMCWindowedMassStageConfig(
        algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        target_scope="p4_phase7_fixture",
        chain_execution_mode="eager",
        engineering_probe_covariance_multiplier=2.0,
    )

    def carrier_exception(candidate: object) -> ValueError:
        error = ValueError("SENTINEL_OUTER_SECRET /private/outer")
        setattr(
            error,
            hmc_warmup._PHASE7_ENGINEERING_PROBE_DIAGNOSTIC_ATTRIBUTE,
            candidate,
        )
        setattr(
            error,
            hmc_warmup._G2_PREBOUNDARY_SEED_PRIVATE_EVIDENCE_ATTRIBUTE,
            private,
        )
        return error

    def result(
        *,
        boundary: Mapping[str, Any],
        final_status: str,
        diagnostic_role: str,
        hard_vetoes: tuple[str, ...],
    ) -> HMCWindowedMassStageResult:
        return HMCWindowedMassStageResult(
            config=config,
            geometry_artifact_hash="geometry",
            bootstrap_artifact_hash="bootstrap",
            selected_bootstrap_kernel_hash="selected",
            adapter_signature="adapter",
            hmc_adapter_signature="hmc-adapter",
            initial_mass_artifact_signature="mass",
            target_dimension=2,
            final_status=final_status,
            diagnostic_role=diagnostic_role,
            hard_vetoes=hard_vetoes,
            diagnostics={
                "engineering_probe_boundary": dict(boundary),
                "hmc_error_type": None,
                "hmc_error_message": None,
                "passed": False,
            },
            draw_capture_policy={},
            warmup_draw_provenance={"source": "none_p4_boundary"},
            acceptance_telemetry_provenance={"source": "none_p4_boundary"},
            diagnostic_run_config_payload=None,
            windowed_config_payload={},
            windowed_mass_result=None,
            seed_report={},
            diagnostic_roles={},
        )

    candidate = replace(
        qualification,
        outcome="candidate_policy_instance_invalid",
        failure_code="target_score_nonfinite",
        target_score_finite_count=0,
    )
    candidate_tracker = hmc_warmup._G2P4BoundaryActionTracker()
    candidate_tracker.mark_builder_entered()
    candidate_tracker.mark_seed_consumed()
    candidate_tracker.mark_rng_invoked()
    candidate_tracker.record_target_callback_entry(
        invocation_count=1,
        row_count=4,
        dimension=2,
    )
    candidate_tracker.mark_candidate_terminal()
    candidate_capture, classification, failure_code, _private = (
        hmc_kernel_tuning._p4_boundary_capture_from_exception(
            carrier_exception(candidate),
            registry=registry,
            action_tracker=candidate_tracker,
        )
    )
    candidate_boundary = candidate_capture["raw_diagnostics"][
        "engineering_probe_boundary"
    ]
    assert classification == "candidate_rejected"
    assert failure_code == "target_score_nonfinite"
    candidate_result = result(
        boundary=candidate_boundary,
        final_status="candidate_rejected",
        diagnostic_role="p4e_candidate_boundary_rejection",
        hard_vetoes=(),
    )
    candidate_payload = candidate_result.payload()
    assert candidate_payload["final_status"] == "candidate_rejected"
    assert candidate_payload["hard_vetoes"] == ()
    assert candidate_payload["operational_warmup_result"] is None
    assert candidate_payload["operational_mass_artifact_available"] is False

    shared = replace(
        qualification,
        outcome="shared_implementation_invalid",
        failure_code="target_callback_exception",
        p4_boundary_stage="rng_invoked",
    )
    shared_tracker = hmc_warmup._G2P4BoundaryActionTracker()
    shared_tracker.mark_builder_entered()
    shared_tracker.mark_seed_consumed()
    shared_tracker.mark_rng_invoked()
    shared_tracker.record_target_callback_entry(
        invocation_count=1,
        row_count=4,
        dimension=2,
    )
    shared_capture, classification, failure_code, _private = (
        hmc_kernel_tuning._p4_boundary_capture_from_exception(
            carrier_exception(shared),
            registry=registry,
            action_tracker=shared_tracker,
        )
    )
    shared_boundary = shared_capture["raw_diagnostics"][
        "engineering_probe_boundary"
    ]
    assert classification == "shared_implementation_invalid"
    shared_result = result(
        boundary=shared_boundary,
        final_status="hard_veto",
        diagnostic_role="shared_implementation_invalid",
        hard_vetoes=(failure_code,),
    )
    shared_payload = shared_result.payload()
    assert shared_payload["hard_vetoes"] == ("target_callback_exception",)
    serialized = json.dumps(
        {"candidate": candidate_payload, "shared": shared_payload},
        sort_keys=True,
    )
    assert "SENTINEL_OUTER_SECRET" not in serialized
    assert "/private/outer" not in serialized
    assert "windowed_stage_hmc_error" not in serialized


def test_g1a_resigned_malformed_private_evidence_invalidates_carrier() -> None:
    operational = _p4_operational_fixture()
    qualification = operational.engineering_probe_bank_qualification
    assert qualification is not None
    malformed = dict(operational.g2_seed_use_registry_payload)
    malformed["unexpected_private_field"] = "SENTINEL_PRIVATE_VALUE"
    unsigned = dict(malformed)
    unsigned.pop("seed_use_registry_signature")
    malformed["seed_use_registry_signature"] = (
        hmc_warmup._canonical_ascii_sha256(unsigned)
    )
    error = ValueError("SENTINEL_OUTER_SECRET /private/malformed")
    setattr(
        error,
        hmc_warmup._PHASE7_ENGINEERING_PROBE_DIAGNOSTIC_ATTRIBUTE,
        qualification,
    )
    setattr(
        error,
        hmc_warmup._G2_PREBOUNDARY_SEED_PRIVATE_EVIDENCE_ATTRIBUTE,
        malformed,
    )
    tracker = hmc_warmup._G2P4BoundaryActionTracker()
    tracker.mark_builder_entered()
    tracker.mark_seed_consumed()
    tracker.mark_rng_invoked()
    tracker.record_target_callback_entry(
        invocation_count=1,
        row_count=4,
        dimension=2,
    )
    tracker.mark_candidate_terminal()

    capture, classification, failure_code, private = (
        hmc_kernel_tuning._p4_boundary_capture_from_exception(
            error,
            registry=operational.g2_seed_use_registry,
            action_tracker=tracker,
        )
    )
    boundary = capture["raw_diagnostics"]["engineering_probe_boundary"]
    assert classification == "shared_implementation_invalid"
    assert failure_code == "qualification_carrier_invalid"
    assert private is None
    assert boundary["seed_registry_evidence_kind"] == (
        "unavailable_invalid_carrier"
    )
    serialized = json.dumps(capture, sort_keys=True)
    assert "SENTINEL_PRIVATE_VALUE" not in serialized
    assert "SENTINEL_OUTER_SECRET" not in serialized
    assert "/private/malformed" not in serialized


def test_g1a_unknown_exception_preserves_independent_p4_action_stage() -> None:
    operational = _p4_operational_fixture()
    tracker = hmc_warmup._G2P4BoundaryActionTracker()
    tracker.mark_builder_entered()
    tracker.mark_seed_consumed()
    tracker.mark_rng_invoked()
    error = RuntimeError("SENTINEL_UNKNOWN_SECRET /private/unknown")
    capture, classification, failure_code, private = (
        hmc_kernel_tuning._p4_boundary_capture_from_exception(
            error,
            registry=operational.g2_seed_use_registry,
            action_tracker=tracker,
        )
    )
    boundary = capture["raw_diagnostics"]["engineering_probe_boundary"]
    assert classification == "shared_implementation_invalid"
    assert failure_code == "unexpected_builder_exception"
    assert private is None
    assert boundary["p4_boundary_stage"] == "rng_invoked"
    assert boundary["p4_builder_entered"] is True
    assert boundary["p4_seed_consumed"] is True
    assert boundary["p4_rng_batch_invoked"] is True
    serialized = json.dumps(capture, sort_keys=True)
    assert "SENTINEL_UNKNOWN_SECRET" not in serialized
    assert "/private/unknown" not in serialized


@pytest.mark.parametrize(
    ("failure_site", "expected_code"),
    (
        ("stage_gate", "seed_registry_source_coverage_invalid"),
        ("diagnostic_config", "unexpected_builder_exception"),
        ("signature_payload", "unexpected_builder_exception"),
        ("capture_entry", "unexpected_builder_exception"),
    ),
)
def test_g1a_p4_attempt_try_redacts_each_preboundary_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_site: str,
    expected_code: str,
) -> None:
    registry = _g1a_stage_seed_registry(
        include_stage_gate=failure_site != "stage_gate"
    )
    config = HMCWindowedMassStageConfig(
        algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
        target_scope="p4_phase7_fixture",
        chain_execution_mode="eager",
        engineering_probe_covariance_multiplier=2.0,
        seed=(20260824, 400),
    )
    windowed_config = hmc_kernel_tuning._windowed_mass_stage_internal_config(
        None,
        mass_policy=config.mass_policy,
    )
    capture_calls = 0

    class BrokenSignatureConfig:
        def signature_payload(self) -> Mapping[str, Any]:
            raise RuntimeError("SENTINEL_SIGNATURE_SECRET /private/signature")

    if failure_site == "diagnostic_config":
        monkeypatch.setattr(
            hmc_kernel_tuning,
            "_windowed_stage_diagnostic_run_config",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("SENTINEL_CONFIG_SECRET /private/config")
            ),
        )
    elif failure_site == "signature_payload":
        monkeypatch.setattr(
            hmc_kernel_tuning,
            "_windowed_stage_diagnostic_run_config",
            lambda *_args, **_kwargs: BrokenSignatureConfig(),
        )

    def capture_entry(**_kwargs: Any) -> Any:
        nonlocal capture_calls
        capture_calls += 1
        raise RuntimeError("SENTINEL_CAPTURE_SECRET /private/capture")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_operational_windowed_mass_capture",
        capture_entry,
    )
    result = hmc_kernel_tuning._run_p4_windowed_boundary_attempt(
        adapter=object(),
        geometry=SimpleNamespace(),
        hmc_adapter_signature="hmc-adapter",
        stage_mass_artifact=SimpleNamespace(),
        mass_window_seed_kernel={"step_size": 0.1, "num_leapfrog_steps": 3},
        windowed_config=windowed_config,
        config=config,
        target_scope="p4_phase7_fixture",
        attempt_state=None,
        route_decision=SimpleNamespace(),
        progress_callback=None,
        attempt_index=0,
        registry=registry,
    )
    capture = result[4]
    assert result[7] == "shared_implementation_invalid"
    assert result[8] == expected_code
    boundary = capture["raw_diagnostics"]["engineering_probe_boundary"]
    assert boundary["schema"] == (
        "bayesfilter.hmc_g2_preboundary_shared_invalidity.v2"
    )
    assert boundary["final_lineage_available"] is False
    assert boundary["p4_boundary_stage"] == "not_entered"
    assert boundary["p4_builder_entered"] is False
    assert boundary["p4_seed_consumed"] is False
    assert boundary["p4_rng_batch_invoked"] is False
    assert capture_calls == (1 if failure_site == "capture_entry" else 0)
    serialized = json.dumps(capture, sort_keys=True)
    for secret in (
        "SENTINEL_CONFIG_SECRET",
        "SENTINEL_SIGNATURE_SECRET",
        "SENTINEL_CAPTURE_SECRET",
        "/private/",
        "RuntimeError",
    ):
        assert secret not in serialized


def test_g1a_p4_attempt_single_try_dominates_all_four_operations() -> None:
    source = textwrap.dedent(
        inspect.getsource(hmc_kernel_tuning._run_p4_windowed_boundary_attempt)
    )
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    tries = [node for node in ast.walk(function) if isinstance(node, ast.Try)]
    assert len(tries) == 1
    protected = tries[0]
    protected_calls = {
        node.func.id
        for statement in protected.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "_derive_seed",
        "_windowed_stage_diagnostic_run_config",
        "_operational_windowed_mass_capture",
    } <= protected_calls
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "signature_payload"
        for statement in protected.body
        for node in ast.walk(statement)
    )


@pytest.mark.parametrize(
    ("classification", "expected_status", "expected_role"),
    (
        (
            "candidate_rejected",
            "candidate_rejected",
            "p4e_candidate_boundary_rejection",
        ),
        (
            "shared_implementation_invalid",
            "hard_veto",
            "shared_implementation_invalid",
        ),
    ),
)
def test_g1a_complete_windowed_stage_bypasses_generic_classifier(
    monkeypatch: pytest.MonkeyPatch,
    classification: str,
    expected_status: str,
    expected_role: str,
) -> None:
    operational = _p4_operational_fixture()
    qualification = operational.engineering_probe_bank_qualification
    assert qualification is not None
    if classification == "candidate_rejected":
        carrier = replace(
            qualification,
            outcome="candidate_policy_instance_invalid",
            failure_code="target_score_nonfinite",
            target_score_finite_count=0,
        )
    else:
        carrier = replace(
            qualification,
            outcome="shared_implementation_invalid",
            failure_code="target_callback_exception",
            p4_boundary_stage="rng_invoked",
        )
    boundary = carrier.public_payload()
    failure_code = str(boundary["failure_code"])
    capture = hmc_kernel_tuning._p4_boundary_scalar_capture(boundary)

    class FakeGeometry:
        mass_artifact = object()
        artifact_hash = "geometry-hash"
        adapter_signature = "adapter-signature"
        target_dimension = 2
        seed_report: Mapping[str, Any] = {}

    class FakeBootstrap:
        hmc_adapter_signature = "hmc-adapter-signature"
        artifact_hash = "bootstrap-hash"
        seed_report: Mapping[str, Any] = {}

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "HMCGeometryInitializationResult",
        FakeGeometry,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "HMCBootstrapScreenResult",
        FakeBootstrap,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_validate_windowed_stage_inputs",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_resolve_windowed_stage_target_scope",
        lambda *_args, **_kwargs: "p4_phase7_fixture",
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_mass_artifact_signature",
        lambda _artifact: "mass-signature",
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_build_bootstrap_fixed_mass_adapter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "stable_adapter_signature",
        lambda _adapter: "hmc-adapter-signature",
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_windowed_stage_initial_mass_artifact",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_active_bootstrap_handoff_kernel_payload",
        lambda **_kwargs: {"step_size": 0.1, "num_leapfrog_steps": 3},
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_active_bootstrap_handoff_kernel_hash",
        lambda **_kwargs: "selected-hash",
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_phase7_windowed_mass_seed_kernel_payload",
        lambda **_kwargs: {"step_size": 0.1, "num_leapfrog_steps": 3},
    )
    generic_classifier_calls = 0

    def forbidden_classifier(*_args: Any, **_kwargs: Any) -> tuple[str, ...]:
        nonlocal generic_classifier_calls
        generic_classifier_calls += 1
        raise AssertionError("P4 carrier must bypass the generic classifier")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_classify_windowed_stage_capture",
        forbidden_classifier,
    )
    monkeypatch.setattr(
        hmc_kernel_tuning,
        "_run_p4_windowed_boundary_attempt",
        lambda **_kwargs: (
            None,
            None,
            None,
            None,
            capture,
            None,
            None,
            classification,
            failure_code,
            operational.g2_seed_use_registry_payload,
        ),
    )
    result = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=object(),
        geometry=FakeGeometry(),
        bootstrap=FakeBootstrap(),
        config=HMCWindowedMassStageConfig(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            target_scope="p4_phase7_fixture",
            chain_execution_mode="eager",
            engineering_probe_covariance_multiplier=2.0,
        ),
        _g2_seed_use_registry=_g1a_stage_seed_registry(),
    )
    payload = result.payload()
    assert result.final_status == expected_status
    assert result.diagnostic_role == expected_role
    assert generic_classifier_calls == 0
    assert payload["operational_warmup_result"] is None
    assert payload["windowed_mass_result"] is None
    assert payload["diagnostics"]["hmc_error_type"] is None
    assert payload["diagnostics"]["hmc_error_message"] is None
    assert payload["hard_vetoes"] == (
        () if classification == "candidate_rejected" else (failure_code,)
    )
    assert payload["warmup_draw_provenance"] == {
        **payload["warmup_draw_provenance"],
        "source": "none_p4_boundary",
        "sample_space": None,
        "samples_shape": None,
        "adaptation_input_only": False,
        "posterior_samples": False,
    }
    assert payload["acceptance_telemetry_provenance"]["source"] == (
        "none_p4_boundary"
    )
    assert payload["acceptance_telemetry_provenance"]["shape"] is None
    assert payload["acceptance_telemetry_provenance"][
        "runtime_decision_count_supported"
    ] is False


def test_phase7_initial_state_consumes_only_p4_bank_with_rowwise_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_hmc(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("deterministic P4-E lineage test must not run HMC")

    monkeypatch.setattr(hmc_kernel_tuning, "run_full_chain_tfp_hmc", forbidden_hmc)
    operational = _p4_operational_fixture()
    stage = SimpleNamespace(
        config=HMCWindowedMassStageConfig(
            seed=(20260821, 31),
            target_scope="p4_phase7_fixture",
            engineering_probe_covariance_multiplier=2.0,
        ),
        operational_warmup_result=operational,
        target_dimension=2,
    )
    phase4_adapter = _P4NestedAdapter("p4-phase4-adapter")
    verification_adapter = _P4NestedAdapter("p4-verification-adapter")

    initial_state, lineage = hmc_kernel_tuning._phase7_verification_initial_state(
        windowed_stage=stage,
        phase4_adapter=phase4_adapter,
        verification_adapter=verification_adapter,
        verification_hmc_signature="p4-verification-adapter",
    )

    np.testing.assert_allclose(
        initial_state,
        operational.final_kernel_state.transform.theta_to_latent(
            operational.private_start_bank_theta
        ),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert lineage["source"] == "phase7_engineering_probe_bank_v1"
    assert lineage["policy_id"] == hmc_warmup.PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID
    assert lineage["qualification_content_signature"] == (
        operational.engineering_probe_bank_qualification.content_signature
    )
    assert lineage["count"] == 4
    assert lineage["canonical_round_trip_passed"] is True
    assert lineage["final_coordinate_match_passed"] is True
    assert lineage["raw_values_exposed"] is False
    assert lineage["evidence_role"] == "engineering_only"
    assert lineage["promotion_role"] == "non_promoting"
    assert lineage["reports_posterior_convergence"] is False


def test_phase7_initial_state_rejects_legacy_operational_bank_without_fallback() -> None:
    operational = _p4_operational_fixture(policy_id="bayesfilter.greedy_four_start_bank.v1")
    stage = SimpleNamespace(
        operational_warmup_result=operational,
        target_dimension=2,
    )

    with pytest.raises(ValueError, match="requires the explicit P4-E"):
        hmc_kernel_tuning._phase7_verification_initial_state(
            windowed_stage=stage,
            phase4_adapter=_P4NestedAdapter("p4-phase4-adapter"),
            verification_adapter=_P4NestedAdapter("p4-verification-adapter"),
            verification_hmc_signature="p4-verification-adapter",
        )


def _p4_stage_fixture() -> tuple[Any, Any]:
    operational = _p4_operational_fixture()
    stage = SimpleNamespace(
        config=HMCWindowedMassStageConfig(
            seed=(20260821, 31),
            target_scope="p4_phase7_fixture",
            engineering_probe_covariance_multiplier=2.0,
        ),
        operational_warmup_result=operational,
        target_dimension=2,
    )
    return stage, operational


def _p4_operational_with_qualification(
    operational: Any,
    qualification: Any,
    *,
    content_signature: str,
) -> Any:
    values = vars(operational).copy()
    values["engineering_probe_bank_qualification"] = qualification
    values["private_start_bank_signature"] = content_signature
    return SimpleNamespace(**values)


def test_phase7_initial_state_rejects_stale_target_identity_even_when_content_matches() -> None:
    stage, operational = _p4_stage_fixture()
    wrong_target = hmc_warmup._phase7_engineering_probe_target_signature(
        _P4BaseAdapter("p4-stale-target")
    )
    qualification = operational.engineering_probe_bank_qualification
    assert qualification is not None
    content_signature = hmc_warmup._phase7_engineering_probe_bank_content_signature(
        operational.private_start_bank_theta,
        transform_signature=qualification.transform_signature,
        target_signature=wrong_target,
        config_signature=qualification.config_signature,
    )
    stale_qualification = replace(
        qualification,
        target_signature=wrong_target,
        content_signature=content_signature,
    )
    stale_operational = _p4_operational_with_qualification(
        operational,
        stale_qualification,
        content_signature=content_signature,
    )
    stage.operational_warmup_result = stale_operational

    with pytest.raises(ValueError, match="target identity mismatch"):
        hmc_kernel_tuning._phase7_verification_initial_state(
            windowed_stage=stage,
            phase4_adapter=_P4NestedAdapter("p4-phase4-adapter"),
            verification_adapter=_P4NestedAdapter("p4-verification-adapter"),
            verification_hmc_signature="p4-verification-adapter",
        )


def test_phase7_initial_state_uses_builder_target_signature_fallback() -> None:
    stage, operational = _p4_stage_fixture()

    class _NoExplicitSignature:
        pass

    base_adapter = _NoExplicitSignature()
    qualification = operational.engineering_probe_bank_qualification
    assert qualification is not None
    target_signature = hmc_warmup._phase7_engineering_probe_target_signature(
        base_adapter
    )
    content_signature = hmc_warmup._phase7_engineering_probe_bank_content_signature(
        operational.private_start_bank_theta,
        transform_signature=qualification.transform_signature,
        target_signature=target_signature,
        config_signature=qualification.config_signature,
    )
    stage.operational_warmup_result = _p4_operational_with_qualification(
        operational,
        replace(
            qualification,
            target_signature=target_signature,
            content_signature=content_signature,
        ),
        content_signature=content_signature,
    )
    phase4_adapter = _P4NestedAdapter("p4-phase4-adapter")
    phase4_adapter.base_adapter = base_adapter

    initial_state, _lineage = hmc_kernel_tuning._phase7_verification_initial_state(
        windowed_stage=stage,
        phase4_adapter=phase4_adapter,
        verification_adapter=_P4NestedAdapter("p4-verification-adapter"),
        verification_hmc_signature="p4-verification-adapter",
    )
    assert initial_state.shape == (4, 2)


def test_phase7_initial_state_rejects_stale_config_and_seed_lineage() -> None:
    stage, operational = _p4_stage_fixture()
    qualification = operational.engineering_probe_bank_qualification
    assert qualification is not None
    wrong_config = hmc_warmup.Phase7EngineeringProbeBankConfig(
        chain_count=4,
        covariance_multiplier=3.0,
        root_seed=(20260821, 12345),
    )
    content_signature = hmc_warmup._phase7_engineering_probe_bank_content_signature(
        operational.private_start_bank_theta,
        transform_signature=qualification.transform_signature,
        target_signature=qualification.target_signature,
        config_signature=wrong_config.config_signature,
    )
    stale_qualification = replace(
        qualification,
        config_signature=wrong_config.config_signature,
        derived_seed_signature=wrong_config.derived_seed_signature,
        content_signature=content_signature,
    )
    stage.operational_warmup_result = _p4_operational_with_qualification(
        operational,
        stale_qualification,
        content_signature=content_signature,
    )

    with pytest.raises(ValueError, match="configuration lineage mismatch"):
        hmc_kernel_tuning._phase7_verification_initial_state(
            windowed_stage=stage,
            phase4_adapter=_P4NestedAdapter("p4-phase4-adapter"),
            verification_adapter=_P4NestedAdapter("p4-verification-adapter"),
            verification_hmc_signature="p4-verification-adapter",
        )
