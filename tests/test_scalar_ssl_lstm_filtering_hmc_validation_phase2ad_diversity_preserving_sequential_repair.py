from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2ad_diversity_preserving_sequential_repair",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StandardNormalAdapter:
    parameter_dim = 4
    target_scope = "test:standard_normal"

    def log_prob_and_grad(self, values):
        tensor = tf.convert_to_tensor(values, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(tensor), axis=-1), -tensor


def test_settings_lock_phase2ad_runtime_contract() -> None:
    harness = _load_harness()

    payload = harness.Phase2ADSequentialSettings().payload()

    assert payload["particle_count"] == 128
    assert payload["seed"] == (20260709, 6801)
    assert payload["max_stages"] == 48
    assert payload["bisection_iterations"] == 32
    assert payload["minimum_beta_increment"] == 1.0e-4
    assert payload["resample_boundary_tolerance"] == 1.0e-4
    assert payload["repair_policy"] == (
        "accept_nonterminal_systematic_resampling_only_when_realized_projected_"
        "root_ancestor_fraction_meets_threshold"
    )
    assert payload["final_ancestor_measurement"] == (
        "after_last_completed_nonterminal_resampling_and_rejuvenation_"
        "with_no_terminal_resampling"
    )
    assert payload["terminal_weight_measurement"] == "beta_1_pre_final_resampling"
    assert payload["hmc_usage"] is False


def test_normalized_weights_summary_matches_manual_ess() -> None:
    harness = _load_harness()

    weights, summary = harness.normalized_weights_from_log(np.log([1.0, 2.0, 1.0]))

    np.testing.assert_allclose(weights, [0.25, 0.5, 0.25])
    np.testing.assert_allclose(summary["ess"], 1.0 / (0.25**2 + 0.5**2 + 0.25**2))
    np.testing.assert_allclose(summary["ess_ratio"], summary["ess"] / 3.0)
    assert summary["max"] == 0.5


def test_systematic_resample_is_reproducible_and_in_range() -> None:
    harness = _load_harness()
    rng1 = np.random.default_rng((20260709, 6801))
    rng2 = np.random.default_rng((20260709, 6801))

    first = harness.systematic_resample([0.1, 0.2, 0.7], rng1)
    second = harness.systematic_resample([0.1, 0.2, 0.7], rng2)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3,)
    assert np.all((0 <= first) & (first < 3))


def test_select_next_beta_uses_terminal_when_admissible() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings()

    selection = harness.select_next_beta(0.0, np.zeros(8), settings)

    assert selection["next_beta"] == 1.0
    assert selection["target_reached"] is True
    assert selection["vetoes"] == ()
    assert selection["ess_ratio"] == 1.0


def test_select_next_beta_vetoes_stalled_increment() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings(minimum_beta_increment=0.9)
    log_ratio = np.array([0.0, 100.0, -100.0, 50.0, -50.0])

    selection = harness.select_next_beta(0.0, log_ratio, settings)

    assert "temperature_increment_stalled" in selection["vetoes"]


def test_select_next_beta_uses_cumulative_log_weights_for_ess() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings(
        target_ess_ratio=0.8,
        minimum_ess_ratio=0.5,
    )
    log_ratio = np.array([0.0, 3.0, -3.0, 1.0, -1.0])
    current_log_weights = np.array([0.0, 0.5, -0.5, 0.0, 0.0])

    selection = harness.select_next_beta(
        0.0,
        log_ratio,
        settings,
        current_log_weights,
    )
    _weights, cumulative_summary = harness.normalized_weights_from_log(
        current_log_weights + selection["delta_beta"] * log_ratio
    )

    np.testing.assert_allclose(selection["ess_ratio"], cumulative_summary["ess_ratio"])
    assert selection["vetoes"] == ()


def test_select_next_beta_falls_back_to_minimum_threshold_when_target_stalls() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings(
        target_ess_ratio=0.8,
        minimum_ess_ratio=0.5,
    )
    log_ratio = np.array([0.0, 3.0, -3.0, 1.0, -1.0])
    current_log_weights = np.array([0.0, 1.0, -1.0, 0.0, 0.0])

    selection = harness.select_next_beta(
        0.0,
        log_ratio,
        settings,
        current_log_weights,
    )

    assert selection["delta_beta"] >= settings.minimum_beta_increment
    assert selection["ess_ratio"] >= settings.minimum_ess_ratio
    assert selection["rule"] == "bisection_largest_minimum_admissible_increment"
    assert selection["vetoes"] == ()


def test_phase2ad_resampling_consideration_accepts_diverse_fallback_draw() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings(particle_count=4)

    consideration = harness.phase2ad_resampling_consideration(
        {
            "rule": "bisection_largest_minimum_admissible_increment",
            "target_reached": False,
        },
        {"ess_ratio": 0.90},
        np.ones(4) / 4.0,
        np.arange(4),
        np.random.default_rng(12),
        terminal_before_resampling=False,
        settings=settings,
    )

    assert consideration["considered"] is True
    assert consideration["accepted"] is True
    assert consideration["reason"] == "minimum_threshold_fallback"
    assert consideration["projected_unique_root_ancestor_fraction"] >= 0.25
    assert len(consideration["resample_indices"]) == 4


def test_phase2ad_resampling_consideration_skips_low_diversity_draw() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings(
        particle_count=4,
        unique_ancestor_fraction_min=0.5,
    )

    consideration = harness.phase2ad_resampling_consideration(
        {
            "rule": "bisection_largest_target_admissible_increment",
            "target_reached": False,
        },
        {"ess_ratio": settings.resample_ess_ratio + 0.5 * settings.resample_boundary_tolerance},
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0, 1, 2, 3]),
        np.random.default_rng(12),
        terminal_before_resampling=False,
        settings=settings,
    )

    assert consideration["considered"] is True
    assert consideration["accepted"] is False
    assert consideration["reason"] == "ess_within_resample_boundary_tolerance"
    assert consideration["skip_reason"] == "projected_unique_root_ancestor_fraction_below_threshold"
    assert consideration["projected_unique_root_ancestor_fraction"] == 0.25


def test_phase2ad_resampling_consideration_disallows_terminal_resampling() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings()

    consideration = harness.phase2ad_resampling_consideration(
        {
            "rule": "bisection_largest_minimum_admissible_increment",
            "target_reached": True,
        },
        {"ess_ratio": settings.resample_ess_ratio},
        np.ones(settings.particle_count) / settings.particle_count,
        np.arange(settings.particle_count),
        np.random.default_rng(12),
        terminal_before_resampling=True,
        settings=settings,
    )

    assert consideration["considered"] is False
    assert consideration["accepted"] is False
    assert consideration["skip_reason"] == "terminal_resampling_disallowed"


def test_rejuvenation_updates_target_values_and_records_acceptance() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings(particle_count=8)
    particles = np.zeros((8, 4))
    target = np.zeros((8,))

    new_particles, new_target, summary = harness.rejuvenate_particles(
        _StandardNormalAdapter(),
        particles,
        target,
        beta=0.5,
        rng=np.random.default_rng(123),
        settings=settings,
    )

    assert new_particles.shape == particles.shape
    assert new_target.shape == target.shape
    assert summary["proposal_count"] == 8
    assert 0 <= summary["accepted_count"] <= 8
    assert 0.0 <= summary["acceptance_rate"] <= 1.0
    assert summary["vetoes"] == ()


def test_sequential_tempering_standard_normal_reaches_beta_one() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings(
        particle_count=16,
        max_stages=4,
        unique_ancestor_fraction_min=0.1,
    )

    result = harness.run_sequential_tempering(_StandardNormalAdapter(), settings)
    gate = harness.evaluate_phase2ad_gate(
        result,
        settings,
        {"vetoes": []},
        {"vetoes": []},
    )

    assert result["computed"] is True
    assert result["terminal_beta"] == 1.0
    assert result["stage_count"] == 1
    assert result["terminal_pre_final_resampling_summary"]["ess_ratio"] == 1.0
    assert gate["phase2ad_candidate_nominated"] is True


def test_gate_vetoes_terminal_weight_failure_before_final_resampling() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings()
    sequential = {
        "computed": True,
        "vetoes": [],
        "terminal_beta": 1.0,
        "stage_count": 1,
        "terminal_pre_final_resampling_summary": {
            "ess_ratio": 0.6,
            "max": 0.5,
        },
        "stage_rows": [
            {"pre_resample_weight_summary": {"ess_ratio": 0.6}},
        ],
        "unique_ancestor_fraction": 1.0,
        "aggregate_rejuvenation_acceptance": 0.5,
    }

    gate = harness.evaluate_phase2ad_gate(
        sequential,
        settings,
        {"vetoes": []},
        {"vetoes": []},
    )

    assert gate["phase2ad_candidate_nominated"] is False
    assert "terminal_pre_final_resampling_max_weight_above_threshold" in gate["vetoes"]


def test_gate_vetoes_are_reportable_for_top_level_decision() -> None:
    harness = _load_harness()
    settings = harness.Phase2ADSequentialSettings()
    sequential = {
        "computed": True,
        "vetoes": [],
        "terminal_beta": 1.0,
        "stage_count": 1,
        "terminal_pre_final_resampling_summary": {
            "ess_ratio": 0.6,
            "max": 0.5,
        },
        "stage_rows": [
            {"pre_resample_weight_summary": {"ess_ratio": 0.6}},
        ],
        "unique_ancestor_fraction": 1.0,
        "aggregate_rejuvenation_acceptance": 0.5,
    }

    gate = harness.evaluate_phase2ad_gate(
        sequential,
        settings,
        {"vetoes": []},
        {"vetoes": []},
    )
    all_vetoes = tuple(dict.fromkeys((*(), *gate["vetoes"])))

    assert "terminal_pre_final_resampling_max_weight_above_threshold" in all_vetoes
