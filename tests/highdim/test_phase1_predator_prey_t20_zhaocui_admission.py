from __future__ import annotations

import importlib.util
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py"
DTYPE = tf.float64


def _load_module():
    spec = importlib.util.spec_from_file_location("phase1_predator_prey_leaderboard", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to load highdim leaderboard module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predator_prey_t20_zhaocui_row_is_blocked_until_fixed_variant_exists() -> None:
    module = _load_module()
    observations = module._predator_prey_observations()
    theta = module._predator_prey_theta()

    assert observations.shape == (20, 2)
    assert theta.shape == (6,)
    assert [float(value) for value in theta.numpy().tolist()] == [
        0.6,
        114.0,
        25.0,
        0.3,
        0.5,
        0.5,
    ]

    row = module._zhao_cui_predator_prey_tt_cell()

    assert row["row_id"] == "zhao_cui_predator_prey_T20"
    assert row["algorithm_id"] == "zhao_cui_scalar_or_multistate"
    assert row["comparison_status"] == "blocked_or_status_only"
    assert row["numeric_execution_status"] == (
        "blocked_zhao_cui_predator_prey_t20_multistate_tt_value_score"
    )
    assert row["score_status"] == "blocked_manual_tt_score_adapter"
    assert row["score_coordinate_system"] == "theta=(r,K,a,s,u,v)"
    assert row["log_likelihood"] is None
    assert row["average_log_likelihood"] is None
    assert row["score"] is None
    assert row["route_role"] == "historical_diagnostic_only"
    assert "fixed-variant" in row["reason"].lower()
    assert any("historical" in item.lower() for item in row["nonclaims"])


def test_predator_prey_t20_zhaocui_derivative_config_disables_fd() -> None:
    module = _load_module()
    model = module.highdim.p30_predator_prey_fixture_model()
    derivative_config = module.highdim.FixedBranchDerivativeConfig(
        parameter_indices=tuple(range(int(model.parameter_dim()))),
        finite_difference_h=(),
        solve_condition_number_veto=1e30,
    )

    assert derivative_config.parameter_indices == (0, 1, 2, 3, 4, 5)
    assert derivative_config.finite_difference_h == ()


def test_predator_prey_t20_zhaocui_row_fails_closed_for_local_leaderboard_contracts() -> None:
    module = _load_module()
    row = module._apply_phase7_status(module._zhao_cui_predator_prey_tt_cell())

    module._validate_analytical_score_contract([row])
    assert row["comparison_status"] == "blocked_or_status_only"
    assert row["phase7_batch_gpu_xla_status"]["timing_rank_status"] == (
        "not_rankable_correctness_gate_open"
    )
    assert row["phase7_batch_gpu_xla_status"]["gpu_xla_status"] == (
        "not_applicable_until_value_score_row_exists"
    )


def test_predator_prey_manual_local_density_scores_match_diagnostic_tape() -> None:
    module = _load_module()
    model = module.highdim.p30_predator_prey_fixture_model()
    theta = module._predator_prey_theta()
    previous = tf.constant([[50.0, 5.0], [80.0, 3.0]], dtype=DTYPE)
    current = model.transition_mean(theta, previous) + tf.constant(
        [[0.1, -0.2], [-0.3, 0.4]],
        dtype=DTYPE,
    )
    observation = current[0]

    with tf.GradientTape() as tape:
        tape.watch(theta)
        transition_values = model.transition_log_density(theta, previous, current, t=1)
    expected_transition = tape.jacobian(transition_values, theta)

    actual_transition = model.transition_log_density_parameter_score(
        theta,
        previous,
        current,
        t=1,
    )
    actual_initial = model.initial_log_density_parameter_score(theta, previous)
    actual_observation = model.observation_log_density_parameter_score(
        theta,
        current,
        observation,
        t=1,
    )

    tf.debugging.assert_near(actual_transition, expected_transition, atol=2e-7, rtol=2e-7)
    tf.debugging.assert_near(
        actual_initial,
        tf.zeros([2, 6], dtype=DTYPE),
        atol=1e-15,
    )
    tf.debugging.assert_near(
        actual_observation,
        tf.zeros([2, 6], dtype=DTYPE),
        atol=1e-15,
    )
