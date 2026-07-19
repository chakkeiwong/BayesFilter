from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import ModuleType

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_phase8_sample_size_margin_preflight_2026_07_17.py"
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location("phase8_sample_size_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_historical_bindings_fail_closed_and_confirmation_stays_blinded(
    harness: ModuleType,
) -> None:
    bindings = harness._validate_bindings()
    assert bindings["failed_448"]["decision"] == (
        "PHASE8_CONTROLLED_NOMINATION_UNDERPOWERED_REPAIR_REQUIRED"
    )
    assert bindings["failed_1984"]["decision"] == (
        "PHASE8_POWER_REPAIR_NOMINATION_UNDERPOWERED_STOP"
    )
    split = bindings["pilot"]["split_contract"]
    assert split["confirmation_forecast_bank_opened"] is False
    assert split["g_h_predictive_difference_computed"] is False


def test_material_mode_binds_passing_smoke_without_selection(
    harness: ModuleType,
) -> None:
    bindings = harness._validate_bindings(require_smoke=True)
    smoke = bindings["smoke"]
    assert smoke["decision"] == (
        "PHASE8_SAMPLE_SIZE_PREFLIGHT_SMOKE_PASSED_MATERIAL_REQUIRED"
    )
    assert smoke["configuration"]["margin_selection"] is None
    assert smoke["configuration"]["mmd_tolerance_selection"] is None
    assert harness._sha256(harness.SMOKE_RECEIPT_PATH) == harness.SMOKE_RECEIPT_SHA256


def test_draw_grid_and_margin_scenarios_are_prospective_and_labeled(
    harness: ModuleType,
) -> None:
    assert harness.DRAW_GRID == (1984, 3072, 4096, 6144, 8192)
    assert all(draws % harness.BLOCK_LENGTH == 0 for draws in harness.DRAW_GRID)
    scenarios = {row.name: row for row in harness.SCENARIOS}
    assert scenarios["historical_original_symmetric"].mean_margin == pytest.approx(0.15)
    assert scenarios["historical_repair_tost"].mean_margin == pytest.approx(0.10)
    midpoint = scenarios["anchor_midpoint_tost"]
    assert midpoint.mean_margin == pytest.approx(0.125)
    assert midpoint.log_variance_margin == pytest.approx(
        0.5 * (math.log(1.05) + math.log(1.25))
    )
    assert midpoint.role == "arithmetic_sensitivity"


def test_analytical_requirement_increases_with_power_and_joint_guard(
    harness: ModuleType,
) -> None:
    bindings = harness._validate_bindings()
    result = harness._analytical_preflight(bindings["failed_1984"])
    variance = next(
        row
        for row in result["requirements"]
        if row["name"] == "trueeq_variance_1p05_historical_repair"
    )
    assert variance["required_draws_80pct_single_limiting_coordinate"] > 1984
    assert (
        variance["required_draws_90pct_single_limiting_coordinate"]
        > variance["required_draws_80pct_single_limiting_coordinate"]
    )
    assert (
        variance["required_draws_80pct_conservative_20_coordinate_lower_bound"]
        > variance["required_draws_80pct_single_limiting_coordinate"]
    )


def test_required_draws_respects_clearance_and_block_rounding(
    harness: ModuleType,
) -> None:
    small_clearance = harness._required_draws(
        standard_error_1984=0.025,
        clearance=0.05,
        critical=1.88,
        power=0.80,
    )
    large_clearance = harness._required_draws(
        standard_error_1984=0.025,
        clearance=0.10,
        critical=1.88,
        power=0.80,
    )
    assert small_clearance is not None and large_clearance is not None
    assert small_clearance > large_clearance
    assert small_clearance % harness.BLOCK_LENGTH == 0
    assert harness._required_draws(
        standard_error_1984=0.025,
        clearance=0.0,
        critical=1.88,
        power=0.80,
    ) is None


def test_joint_feature_decision_requires_all_coordinates(harness: ModuleType) -> None:
    scenario = next(row for row in harness.SCENARIOS if row.name == "historical_repair_tost")
    standard_error = tf.fill([harness.FEATURE_COUNT], tf.constant(0.01, tf.float64))
    estimates = tf.zeros([2, harness.FEATURE_COUNT], tf.float64)
    estimates = tf.tensor_scatter_nd_update(estimates, [[1, 7]], [0.095])
    masks = harness._feature_status_masks(estimates, standard_error, scenario)
    assert bool(masks["pass"][0])
    assert not bool(masks["pass"][1])


def test_frechet_bounds_do_not_assume_feature_mmd_independence(
    harness: ModuleType,
) -> None:
    lower, upper = harness._frechet_bounds(0.90, 0.85)
    assert lower == pytest.approx(0.75)
    assert upper == pytest.approx(0.85)
    with pytest.raises(harness.PreflightError, match=r"\[0,1\]"):
        harness._frechet_bounds(1.1, 0.5)


def test_wilson_interval_contains_observed_probability(harness: ModuleType) -> None:
    lower, upper = harness._wilson_interval(800, 1000)
    assert lower < 0.8 < upper
    assert upper - lower < 0.06


def test_runner_has_no_confirmation_archive_or_hmc_acquisition_input(
    harness: ModuleType,
) -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retained-private" not in source
    assert "retained_samples.tftensor" not in source
    assert "segment-001_retained_samples" not in source
    assert 'mode not in {"smoke", "material"}' in source
    assert "HMC acquisition and confirmation remain closed" in source
    with pytest.raises(harness.PreflightError, match="remain closed"):
        harness.run(
            mode="hmc",
            output=Path("/tmp/must-not-exist.json"),
            wall_cap_seconds=1.0,
        )


def test_seeds_are_fresh_and_margin_selection_is_forbidden(harness: ModuleType) -> None:
    assert harness.PILOT_SEED not in {(14001, 14002), (16001, 16002)}
    assert harness.MONTE_CARLO_SEED not in {
        (14001, 14002),
        (16001, 16002),
        harness.PILOT_SEED,
    }
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"margin_selection": None' in source
    assert '"mmd_tolerance_selection": None' in source


def test_resource_projection_uses_public_warm_segments(harness: ModuleType) -> None:
    phase7 = harness._validate_bindings()["phase7"]
    projection = harness._resource_projection(phase7)
    assert projection["warm_segment_seconds_per_256_draws"] == pytest.approx(
        140.63920265401248
    )
    assert projection["warm_forecast_seconds_per_64_draws"] == pytest.approx(
        5.000171198975295
    )
    assert projection["projections"]["3072"][
        "segment_rounded_acquired_draws_per_chain"
    ] == 3328
    assert projection["projections"]["3072"][
        "unused_segment_surplus_draws_per_chain"
    ] == 192
    assert (
        projection["projections"]["8192"][
            "estimated_total_hmc_plus_forecast_gpu_hours"
        ]
        > projection["projections"]["1984"][
            "estimated_total_hmc_plus_forecast_gpu_hours"
        ]
    )


def test_smoke_operating_loop_uses_only_generated_family(
    harness: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    family = harness.FAMILIES[3]
    pilot = {
        "covariance_at_8192": tf.eye(harness.FEATURE_COUNT, dtype=tf.float64).numpy().tolist(),
        "truth": [0.0] * harness.FEATURE_COUNT,
        "mmd_mean_estimate": 0.0,
        "mmd_root_mean_square_standard_error_at_8192": 0.01,
        "mmd_harmonic_degrees_of_freedom_at_8192": 100.0,
    }
    monkeypatch.setattr(harness, "DRAW_GRID", (1984,))
    monkeypatch.setattr(harness, "MMD_TOLERANCES", (0.04,))
    rows = harness._operating_rows({family.name: pilot}, monte_carlo_count=32)
    assert set(rows["historical_repair_tost"]["1984"]) == {family.name}
