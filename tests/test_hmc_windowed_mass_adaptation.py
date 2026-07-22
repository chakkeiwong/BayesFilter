from __future__ import annotations

import json

import numpy as np
import pytest

from bayesfilter.inference import (
    FullChainHMCConfig,
    HMCTuningPolicy,
    PrecomputedMassArtifact,
    WindowedMassAdaptationConfig,
    build_windowed_warmup_schedule,
    run_windowed_mass_adaptation_diagnostic,
    validate_windowed_shrinkage_target,
    welford_covariance,
)


def _mass_artifact(
    *,
    dim: int = 2,
    adapter_signature: str = "phase4_windowed_adapter_v1",
) -> PrecomputedMassArtifact:
    precision = np.eye(dim)
    precision[0, 0] = 4.0
    if dim > 1:
        precision[0, 1] = 0.25
        precision[1, 0] = 0.25
        precision[1, 1] = 3.0
    return PrecomputedMassArtifact.from_negative_hessian(
        position=np.zeros(dim),
        negative_hessian=precision,
        adapter_signature=adapter_signature,
        covariance_source="phase2_regularized_hessian",
        jitter=0.0,
        eigenvalue_floor=1.0e-6,
    )


def _policy() -> HMCTuningPolicy:
    return HMCTuningPolicy.windowed_mass_adaptation(
        num_adaptation_steps=12,
        target_accept_prob=0.72,
        source="tests/test_hmc_windowed_mass_adaptation.py",
    )


def _config() -> WindowedMassAdaptationConfig:
    return WindowedMassAdaptationConfig(
        warmup_steps=12,
        initial_buffer=2,
        final_buffer=2,
        first_window_size=3,
        min_window_samples=2,
        mass_shrinkage=0.25,
        covariance_jitter=1.0e-6,
        step_adaptation_rate=0.03,
    )


def _warmup_draws() -> np.ndarray:
    return np.array(
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


def test_windowed_schedule_is_fast_slow_final_and_contiguous():
    schedule = build_windowed_warmup_schedule(_config())

    assert [window.kind for window in schedule] == [
        "initial_fast",
        "slow",
        "slow",
        "final_fast",
    ]
    assert [(window.start, window.end) for window in schedule] == [
        (0, 2),
        (2, 5),
        (5, 10),
        (10, 12),
    ]
    assert [window.update_mass for window in schedule] == [False, True, True, False]


def test_fixed_identity_windowed_schedule_disables_slow_mass_updates():
    config = WindowedMassAdaptationConfig(
        **{
            **_config().payload(),
            "mass_policy": "fixed_identity",
        }
    )

    schedule = build_windowed_warmup_schedule(config)

    assert [window.update_mass for window in schedule] == [False, False, False, False]


def test_welford_covariance_matches_numpy_sample_covariance():
    samples = _warmup_draws()[2:7]

    result = welford_covariance(samples)

    np.testing.assert_allclose(result.mean, np.mean(samples, axis=0))
    np.testing.assert_allclose(result.covariance, np.cov(samples, rowvar=False))
    assert result.count == samples.shape[0]
    assert result.finite is True


def test_windowed_shrinkage_target_rejects_stale_or_misordered_artifact():
    initial = _mass_artifact(adapter_signature="phase4_windowed_adapter_v1")
    stale = _mass_artifact(adapter_signature="other_coordinate_order_v1")
    wrong_dim = _mass_artifact(dim=3, adapter_signature="phase4_windowed_adapter_v1")

    with pytest.raises(ValueError, match="coordinate-incompatible"):
        validate_windowed_shrinkage_target(
            initial_mass_artifact=initial,
            shrinkage_target_mass_artifact=stale,
        )
    with pytest.raises(ValueError, match="coordinate-incompatible"):
        validate_windowed_shrinkage_target(
            initial_mass_artifact=initial,
            shrinkage_target_mass_artifact=wrong_dim,
        )
    with pytest.raises(ValueError, match="coordinate-incompatible"):
        validate_windowed_shrinkage_target(
            initial_mass_artifact=initial,
            shrinkage_target_mass_artifact=initial,
            expected_adapter_signature="not_the_phase2_adapter",
        )


def test_windowed_mass_runner_rejects_stale_shrinkage_target_integration():
    with pytest.raises(ValueError, match="coordinate-incompatible"):
        run_windowed_mass_adaptation_diagnostic(
            _policy(),
            config=_config(),
            initial_mass_artifact=_mass_artifact(
                adapter_signature="phase4_windowed_adapter_v1"
            ),
            shrinkage_target_mass_artifact=_mass_artifact(
                adapter_signature="stale_phase2_adapter_v1"
            ),
            warmup_draws=_warmup_draws(),
            initial_step_size=0.08,
        )


def test_windowed_mass_adaptation_smoke_updates_mass_and_reset_telemetry():
    artifact = _mass_artifact()
    acceptance = np.array(
        [0.65, 0.70, 0.72, 0.75, 0.78, 0.70, 0.68, 0.74, 0.73, 0.71, 0.72, 0.76]
    )

    result = run_windowed_mass_adaptation_diagnostic(
        _policy(),
        config=_config(),
        initial_mass_artifact=artifact,
        warmup_draws=_warmup_draws(),
        initial_step_size=0.08,
        acceptance_trace=acceptance,
        expected_adapter_signature="phase4_windowed_adapter_v1",
        target_failure_classification={
            "classification": "tuning_diagnostic_passed_not_convergence",
            "diagnostic_role": "diagnostic_only",
            "nonclaims": ("no posterior convergence claim",),
        },
    )
    payload = result.payload()

    assert result.passed is True
    assert result.final_step_size > 0.0
    assert len(result.step_size_trace) == _config().warmup_steps
    assert len(result.mass_updates) == 2
    assert result.semantic_checks() == {
        "window_schedule_contiguous": True,
        "mass_update_count_matches_slow_windows": True,
        "every_update_has_dual_averaging_reset": True,
        "final_mass_artifact_frozen_payload": True,
        "fixed_identity_signature_unchanged": True,
        "shrinkage_target_compatible": True,
        "does_not_report_posterior_convergence": True,
    }
    assert result.initial_mass_artifact_signature != result.final_mass_artifact_signature
    assert artifact.covariance.flags.writeable is False
    assert all(
        update.reset_event["event"] == "dual_averaging_reset"
        for update in result.mass_updates
    )
    assert all(
        update.mass_artifact_payload["regularization_report"][
            "dual_averaging_reset_required"
        ]
        for update in result.mass_updates
    )
    assert payload["diagnostics"]["reports_posterior_convergence"] is False
    assert "windowed mass adaptation diagnostic only" in payload["diagnostics"][
        "nonclaims"
    ]
    json.dumps(payload)


def test_windowed_mass_rebuild_uses_validated_shrinkage_target_provenance():
    initial = _mass_artifact()
    target = PrecomputedMassArtifact.from_negative_hessian(
        position=np.array([1.5, -0.25]),
        negative_hessian=np.array([[4.0, 0.25], [0.25, 3.0]]),
        adapter_signature="phase4_windowed_adapter_v1",
        position_role="map",
        covariance_source="alternate_phase2_regularized_hessian",
        jitter=0.0,
        eigenvalue_floor=1.0e-6,
    )

    result = run_windowed_mass_adaptation_diagnostic(
        _policy(),
        config=_config(),
        initial_mass_artifact=initial,
        shrinkage_target_mass_artifact=target,
        warmup_draws=_warmup_draws(),
        initial_step_size=0.08,
    )

    assert result.compatibility["compatible"] is True
    assert result.shrinkage_target_signature != result.initial_mass_artifact_signature
    final_payload = result.final_mass_artifact_payload
    assert final_payload["adapter_signature"] == "phase4_windowed_adapter_v1"
    assert final_payload["position_role"] == target.position_role


def test_windowed_mass_adaptation_requires_reviewed_policy_object():
    with pytest.raises(ValueError, match="reviewed HMCTuningPolicy"):
        run_windowed_mass_adaptation_diagnostic(
            "windowed_mass_adaptation",
            config=_config(),
            initial_mass_artifact=_mass_artifact(),
            warmup_draws=_warmup_draws(),
            initial_step_size=0.08,
        )

    with pytest.raises(ValueError, match="not implemented"):
        run_windowed_mass_adaptation_diagnostic(
            HMCTuningPolicy.windowed_mass_adaptation_future(),
            config=_config(),
            initial_mass_artifact=_mass_artifact(),
            warmup_draws=_warmup_draws(),
            initial_step_size=0.08,
        )

    mismatched_policy = HMCTuningPolicy.windowed_mass_adaptation(
        num_adaptation_steps=11,
        target_accept_prob=0.72,
        source="tests/test_hmc_windowed_mass_adaptation.py",
    )
    with pytest.raises(ValueError, match="steps must equal"):
        run_windowed_mass_adaptation_diagnostic(
            mismatched_policy,
            config=_config(),
            initial_mass_artifact=_mass_artifact(),
            warmup_draws=_warmup_draws(),
            initial_step_size=0.08,
        )


def test_windowed_mass_adaptation_preserves_target_invalidity_hard_veto():
    result = run_windowed_mass_adaptation_diagnostic(
        _policy(),
        config=_config(),
        initial_mass_artifact=_mass_artifact(),
        warmup_draws=_warmup_draws(),
        initial_step_size=0.08,
        target_failure_classification={
            "classification": "target_invalidity_not_tuning_success",
            "diagnostic_role": "hard_veto",
            "nonclaims": ("no posterior convergence claim",),
        },
    )

    assert result.passed is False
    assert result.payload()["diagnostics"]["target_failure_classification"][
        "classification"
    ] == "target_invalidity_not_tuning_success"


def test_windowed_mass_rebuild_enforces_covariance_floor_and_condition_cap():
    config = WindowedMassAdaptationConfig(
        warmup_steps=12,
        initial_buffer=2,
        final_buffer=2,
        first_window_size=3,
        min_window_samples=2,
        mass_shrinkage=0.0,
        covariance_jitter=0.0,
        eigenvalue_floor=0.05,
        max_condition_number=10.0,
        step_adaptation_rate=0.0,
    )
    draws = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    result = run_windowed_mass_adaptation_diagnostic(
        HMCTuningPolicy.windowed_mass_adaptation(
            num_adaptation_steps=12,
            target_accept_prob=0.72,
            source="tests/test_hmc_windowed_mass_adaptation.py",
        ),
        config=config,
        initial_mass_artifact=_mass_artifact(),
        warmup_draws=draws,
        initial_step_size=0.08,
    )

    for update in result.mass_updates:
        report = update.mass_artifact_payload["regularization_report"]
        eigen = update.mass_artifact_payload["eigen_summary"]
        assert update.mass_artifact_payload["covariance_source"] == (
            f"windowed_welford_shrinkage_window_{update.window.index}"
        )
        assert report["method"] == "welford_covariance_shrinkage"
        assert report["window_index"] == update.window.index
        assert report["window_start"] == update.window.start
        assert report["window_end"] == update.window.end
        assert report["dual_averaging_reset_required"] is True
        assert report["covariance_regularization_method"] == (
            "symmetric_eigendecomposition_floor"
        )
        assert report["covariance_jitter"] == pytest.approx(0.0)
        assert report["requested_covariance_eigenvalue_floor"] == pytest.approx(0.05)
        assert report["effective_covariance_eigenvalue_floor"] == pytest.approx(
            max(0.05, report["raw_covariance_max_eigenvalue"] / 10.0)
        )
        assert report["covariance_max_condition_number"] == pytest.approx(10.0)
        assert report["regularized_covariance_min_eigenvalue"] >= 0.05 - 1.0e-12
        assert report["covariance_clipped_eigenvalue_count"] >= 1
        assert eigen["condition_number"] <= 10.0 + 1.0e-12


def test_fixed_mass_full_chain_rejects_windowed_policy_hidden_updates():
    with pytest.raises(ValueError, match="Phase 4 windowed diagnostic runner"):
        FullChainHMCConfig(
            num_results=4,
            num_burnin_steps=4,
            step_size=0.05,
            num_leapfrog_steps=2,
            seed=(20260613, 21),
            tuning_policy=_policy(),
        )
