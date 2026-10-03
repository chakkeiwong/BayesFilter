"""Metric preparation mechanics and diagnostic-role regression tests."""
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_warmup
from bayesfilter.inference.hmc_preparation import HMCPreparationProgress


def test_interleaving_constant_chains_fabricates_temporal_information():
    states = np.tile(np.array([-3., -1., 1., 3.])[None, :, None], (64, 1, 1))
    proper = hmc_warmup.assess_metric_covariance(states)
    flattened = hmc_warmup.assess_metric_covariance(states.reshape(-1, 1))
    assert proper.outcome == "no_update_insufficient_metric_evidence"
    assert proper.report["minimum_effective_sample_size"] == 0.
    assert flattened.outcome == "dense_update"
    assert flattened.report["minimum_effective_sample_size"] >= 8.
    assert proper.report["chain_count"] == 4
    assert proper.report["draw_count_per_chain"] == 64
    assert set(proper.report["dense_failed_checks"]) == {
        "effective_information_sufficient", "within_chain_variances_finite_positive"}


@pytest.mark.parametrize("rhat", [1., 50., float("nan"), None, "error"])
def test_metric_outcome_does_not_depend_on_rhat(monkeypatch, rhat):
    states = np.random.default_rng(4021).normal(size=(80, 4, 3))
    expected = hmc_warmup.assess_metric_covariance(states)

    def report(_states):
        if rhat == "error":
            raise RuntimeError("reporting failure")
        return None if rhat is None else tf.fill((3,), tf.constant(rhat, tf.float64))

    monkeypatch.setattr(hmc_warmup, "_split_rhat_by_coordinate", report)
    actual = hmc_warmup.assess_metric_covariance(states)
    assert actual.outcome == expected.outcome == "dense_update"
    assert actual.report["dense_checks"] == expected.report["dense_checks"]
    np.testing.assert_array_equal(actual.covariance, expected.covariance)
    assert actual.report["rhat_used_for_metric_decision"] is False


def test_singleton_chain_has_exact_rank_two_estimator_parity():
    states = np.random.default_rng(901).normal(size=(96, 3))
    one = hmc_warmup.assess_metric_covariance(states)
    explicit = hmc_warmup.assess_metric_covariance(states[:, None, :])
    assert one.outcome == explicit.outcome == "dense_update"
    assert one.report["dense_checks"] == explicit.report["dense_checks"]
    assert one.report["effective_sample_size_by_coordinate"] == explicit.report["effective_sample_size_by_coordinate"]
    np.testing.assert_array_equal(one.covariance, explicit.covariance)


@pytest.mark.parametrize("metric_policy", ["temporal_information", "finite_window"])
def test_operational_call_chain_preserves_single_chain_time_and_reports_decisions(monkeypatch, tmp_path, metric_policy):
    from bayesfilter.hmc_route_contract import OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    from bayesfilter.inference.hmc_mass_adaptation import run_hmc_windowed_mass_stage
    from tests.test_hmc_kernel_tuning_windowed_mass import _operational_inputs, _operational_budget, _stage_config

    adapter, geometry, bootstrap = _operational_inputs()
    original = hmc_warmup.assess_metric_covariance
    shapes = []

    def assess(states, **kwargs):
        shapes.append(tuple(states.shape))
        assert kwargs["metric_evidence_policy"] == metric_policy
        return original(states, **kwargs)

    monkeypatch.setattr(hmc_warmup, "assess_metric_covariance", assess)
    with HMCPreparationProgress(tmp_path) as progress:
        result = run_hmc_windowed_mass_stage(
            adapter=adapter, geometry=geometry, bootstrap=bootstrap,
            config=_stage_config(algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
                                 chain_execution_mode="tf_function", metric_evidence_policy=metric_policy),
            _attempt_budget_policy=_operational_budget(), _progress_callback=progress.phase,
        )
    operational = result.operational_warmup_result
    assert operational is not None
    assert shapes == [tuple(w.adaptation_latent_states.shape) for w in operational.windows if w.window.update_mass]
    assert all(len(shape) == 2 for shape in shapes)
    record = json.loads((tmp_path / "preparation_progress.json").read_text())
    windows = [event["details"]["window"] for event in record["events"]
               if event["phase"] == "windowed_mass_metric_decision"]
    assert len(windows) == len(operational.windows)
    for reported, actual in zip(windows, operational.windows):
        assert reported == json.loads(json.dumps(actual.public_payload()))
        if reported["metric_decision"]:
            checks = reported["metric_decision"]["report"]
            assert checks["draw_count_per_chain"] == actual.window.length
            assert checks["chain_count"] == 1


def test_rejected_metric_report_survives_strict_progress_json(tmp_path):
    decision = hmc_warmup.assess_metric_covariance(np.random.default_rng(8).normal(size=(80, 2)))
    decision = hmc_warmup._rejected_metric_candidate(decision, stage="reasonable_epsilon", error=ValueError("probe failure"))
    with HMCPreparationProgress(tmp_path) as progress:
        progress.phase("metric", {"decision": decision.payload(), "undefined": float("inf")})
    payload = json.loads((tmp_path / "preparation_progress.json").read_text())["events"][1]["details"]
    assert payload["decision"]["report"]["candidate_rejection_stage"] == "reasonable_epsilon"
    assert payload["decision"]["report"]["incumbent_metric_retained"] is True
    assert payload["undefined"] == {"nonfinite": "inf"}


def test_schedule_capacity_exposes_standard_preset_count_impossibility():
    from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig, _public_budget_policy_factory
    from bayesfilter.inference.hmc_mass_adaptation import _windowed_mass_stage_internal_config

    reports = {}
    for preset in ("standard", "serious"):
        cfg = HMCKernelTuningConfig(preset=preset)
        budget = _public_budget_policy_factory(cfg)(6, 0)
        windows = _windowed_mass_stage_internal_config(budget)
        reports[preset] = hmc_warmup.metric_schedule_capacity(windows, 6)
    assert reports["standard"]["slow_window_lengths"] == (30, 60, 30)
    assert reports["standard"]["dense_state_count_reachable"] is False
    assert reports["serious"]["dense_state_count_reachable"] is True
    assert reports["serious"]["adequacy_guaranteed"] is False


@pytest.mark.parametrize("dimension", [1, 2, 6, 10, 100])
def test_automatic_finite_window_schedule_uses_funded_count_floor(dimension):
    from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig, _public_budget_policy_factory
    from bayesfilter.inference.hmc_mass_adaptation import _windowed_mass_stage_internal_config
    from bayesfilter.inference.hmc_tuning import build_windowed_warmup_schedule

    budget = _public_budget_policy_factory(HMCKernelTuningConfig(preset="standard"))(dimension, 0)
    original = _windowed_mass_stage_internal_config(budget)
    repaired = _windowed_mass_stage_internal_config(budget, metric_evidence_policy="finite_window")
    windows = build_windowed_warmup_schedule(repaired)
    floor, _ = hmc_warmup.metric_covariance_state_requirements(dimension)
    assert repaired.warmup_steps == original.warmup_steps
    assert (repaired.initial_buffer, repaired.final_buffer) == (original.initial_buffer, original.final_buffer)
    assert windows[0].start == 0 and windows[-1].end == original.warmup_steps
    assert all(a.end == b.start for a, b in zip(windows, windows[1:]))
    assert all(w.length >= floor for w in windows if w.update_mass)
    assert hmc_warmup.metric_schedule_capacity(repaired, dimension)["dense_state_count_reachable"]


@pytest.mark.parametrize("warmup, expected_min, dense, diagonal", [
    (60, 32, False, True), (20, 2, False, False)])
def test_finite_window_schedule_does_not_invent_unfunded_states(warmup, expected_min, dense, diagonal):
    from dataclasses import replace
    from bayesfilter.inference.hmc_configuration import HMCKernelTuningConfig, _public_budget_policy_factory
    from bayesfilter.inference.hmc_mass_adaptation import _windowed_mass_stage_internal_config

    budget = _public_budget_policy_factory(HMCKernelTuningConfig(preset="standard"))(6, 0)
    budget = replace(budget, phase4_warmup_steps=warmup)
    config = _windowed_mass_stage_internal_config(budget, metric_evidence_policy="finite_window")
    report = hmc_warmup.metric_schedule_capacity(config, 6)
    assert config.warmup_steps == warmup
    assert config.min_window_samples == expected_min
    assert report["dense_state_count_reachable"] is dense
    assert report["diagonal_state_count_reachable"] is diagonal


def test_finite_window_option_proposes_low_information_covariance_without_changing_estimator():
    time = np.arange(400.)
    states = np.stack([np.sin(2. * np.pi * k * time / 400.) for k in range(1, 5)], axis=1)
    default = hmc_warmup.assess_metric_covariance(states)
    optional = hmc_warmup.assess_metric_covariance(states, metric_evidence_policy="finite_window")
    assert default.outcome != "dense_update"
    assert optional.outcome == "dense_update"
    assert optional.report["dense_temporal_information_sufficient"] is False
    assert optional.report["temporal_information_used_for_metric_decision"] is False
    empirical = np.cov(states, rowvar=False)
    np.testing.assert_allclose(optional.covariance, .75 * empirical + .25 * np.diag(np.diag(empirical)), atol=1e-14)
    assert np.min(np.linalg.eigvalsh(optional.covariance)) > 0.
    scaled = states * np.array([.001, -2., 30., .2])
    scaled_result = hmc_warmup.assess_metric_covariance(scaled, metric_evidence_policy="finite_window")
    assert scaled_result.report["dense_checks"] == optional.report["dense_checks"]


def test_metric_report_distinguishes_temporal_information_from_rank_failure():
    values = np.random.default_rng(42).normal(size=100)
    states = np.stack([values, values], axis=1)
    decision = hmc_warmup.assess_metric_covariance(states)
    assert decision.outcome == "diagonal_fallback"
    assert decision.report["dense_information_gate_passed"] is True
    assert decision.report["dense_proposal_checks_passed"] is False
    assert "full_raw_rank" in decision.report["dense_failed_checks"]


@pytest.mark.parametrize("kind", ["constant", "small", "separated_constant_chains", "nan"])
def test_finite_window_option_keeps_numerical_and_count_vetoes(kind):
    states = np.random.default_rng(4).normal(size=(100, 2))
    if kind == "constant":
        states[:, 0] = 1.
    elif kind == "small":
        states = states[:3]
    elif kind == "separated_constant_chains":
        states = np.tile(np.array([-3., -1., 1., 3.])[None, :, None], (64, 1, 1))
    else:
        states[5, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            hmc_warmup.assess_metric_covariance(states, metric_evidence_policy="finite_window")
        return
    decision = hmc_warmup.assess_metric_covariance(states, metric_evidence_policy="finite_window")
    assert decision.update_applied is False


def test_metric_policy_reaches_every_public_preparation_configuration_hop():
    from bayesfilter.inference.hmc_configuration import (
        HMCKernelTuningConfig, _public_loop_config, _phase7_windowed_stage_config,
    )
    from bayesfilter.inference.hmc_mass_adaptation import _windowed_mass_stage_internal_config
    from bayesfilter.inference.hmc_tuning import WindowedMassAdaptationConfig
    cfg = HMCKernelTuningConfig(metric_evidence_policy="finite_window")
    loop = _public_loop_config(cfg)
    stage = _phase7_windowed_stage_config(loop, attempt_index=0)
    internal = _windowed_mass_stage_internal_config(metric_evidence_policy=stage.metric_evidence_policy)
    for item in (cfg, loop, stage, internal):
        assert item.metric_evidence_policy == "finite_window"
        assert item.payload()["metric_evidence_policy"] == "finite_window"
    assert WindowedMassAdaptationConfig(**internal.payload()) == internal
    assert HMCKernelTuningConfig().metric_evidence_policy == "temporal_information"
    with pytest.raises(ValueError, match="metric_evidence_policy"):
        HMCKernelTuningConfig(metric_evidence_policy="unknown")
    with pytest.raises(ValueError, match="windowed_adaptive"):
        HMCKernelTuningConfig(mass_policy="fixed_identity", metric_evidence_policy="finite_window")
