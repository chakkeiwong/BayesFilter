from __future__ import annotations

import argparse

import tensorflow as tf

from docs.benchmarks import run_contract_e_tp_phase6_zhao_cui_comparator as comparator
from bayesfilter.highdim import ukf_initializer as p76


def _args(rank: int = 2) -> argparse.Namespace:
    return argparse.Namespace(
        row="actual_sv",
        horizon=2,
        degree=4,
        order=13,
        rank=rank,
        coordinate_half_width=8.0,
        density_tau=1e-8,
        seed="svx-zc-ukf-wiring-test",
        target_preparation=None,
    )


def test_actual_sv_default_uses_repository_ukf_initializer_and_cores() -> None:
    result = comparator._run(_args())
    config = result["config"]

    assert config["initializer_id"] == p76.P76_UKF_INITIALIZER_RULE
    assert config["status"] == "ukf_initializer_default"
    assert config["ukf_initializer"]["initializer_rule"] == p76.P76_UKF_INITIALIZER_RULE
    assert config["ukf_initializer"]["initializer_default"] is True
    assert config["ukf_initializer"]["ukf_target_nonclaim"] == (
        "not exact transformed same-target admission"
    )
    assert config["ukf_initializer"]["initial_core_hash"]
    assert config["ukf_initializer"]["adjacent_core_hash"]
    assert config["density_tau"] == 0.0
    assert result["route_id"] == "zhao_cui_fixed_adjacent_state_squared_tt_v1"

    steps = result["finite_program"]["steps"]
    assert len(steps) == 2
    assert steps[0]["fit_dimension"] == 1
    assert steps[1]["fit_dimension"] == 2


def test_ukf_projection_changes_with_center_and_scale() -> None:
    basis = comparator._basis(1, 4)
    kwargs = {
        "gamma": 4.0,
        "quadrature_order": 13,
        "reference_offset": tf.zeros([1], dtype=tf.float64),
        "reference_matrix": tf.constant([[8.0]], dtype=tf.float64),
    }
    centered = p76.p76_gaussian_sqrt_projection_coefficients(
        basis,
        center=tf.zeros([1], dtype=tf.float64),
        linear_map=tf.eye(1, dtype=tf.float64),
        **kwargs,
    )
    shifted = p76.p76_gaussian_sqrt_projection_coefficients(
        basis,
        center=tf.constant([1.5], dtype=tf.float64),
        linear_map=tf.constant([[2.0]], dtype=tf.float64),
        **kwargs,
    )
    assert not bool(tf.reduce_all(tf.equal(centered[0], shifted[0])).numpy())
