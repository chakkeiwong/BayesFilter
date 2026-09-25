"""Pre-refactor configuration baselines for LEDH engine unification.

The matrix is hierarchical: Contract-E owns the correction controls, and
several controls must have no effect when their parent mechanism is disabled.
Off configurations remain supported diagnostics, not production programs.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_production_program_v1 import (
    validate_ledh_production_configuration,
)

DTYPE = tf.float64
BASE = {
    "flow_substeps": 6,
    "reset_epsilon": 2.0,
    "reset_sinkhorn_steps": 4,
    "reset_balance_steps": 2,
    "reset_ridge": 1.0e-5,
}
CONFIGS = {
    "none_1": ({"reset_policy": "none", "annealed_stages": 1}, -6.421888945684312, -2.0767004571287364),
    "none_3": ({"reset_policy": "none", "annealed_stages": 3, "annealed_seed": 17}, -6.177415774916513, -3.1655511376126007),
    "contract_e_off": ({"reset_policy": "contract_e", "correction_steps": 0, "pairwise_steps": 0, "coordinate_cap": 0.0}, -6.580126823622971, -2.830358137563799),
    "diagonal_no_trust": ({"reset_policy": "contract_e", "correction_steps": 2, "correction_strength": 0.2, "correction_lm_damping": 0.001, "correction_lm_scale_floor": 1.0e-6, "correction_trust_radius": 0.0}, -6.6886307854813065, -3.0800549391045435),
    "diagonal_trust": ({"reset_policy": "contract_e", "correction_steps": 2, "correction_strength": 0.2, "correction_lm_damping": 0.001, "correction_lm_scale_floor": 1.0e-6, "correction_trust_radius": 0.1}, -6.637338471748986, -2.9058777822342563),
    "pairwise": ({"reset_policy": "contract_e", "correction_steps": 2, "correction_strength": 0.2, "correction_lm_damping": 0.001, "correction_lm_scale_floor": 1.0e-6, "correction_trust_radius": 0.1, "pairwise_steps": 2, "pairwise_strength": 0.02, "pairwise_rms_cap": 2.0}, -6.641406263342637, -2.921559653136345),
    "full_reference": ({"reset_policy": "contract_e", "correction_steps": 2, "correction_strength": 0.2, "correction_lm_damping": 0.001, "correction_lm_scale_floor": 1.0e-6, "correction_trust_radius": 0.1, "pairwise_steps": 2, "pairwise_strength": 0.02, "pairwise_rms_cap": 2.0, "coordinate_cap": 0.98, "coordinate_cap_power": 8}, -6.560209285241854, -2.8428002574902846),
    "annealed_full": ({"reset_policy": "contract_e", "annealed_stages": 3, "annealed_seed": 19, "correction_steps": 2, "correction_strength": 0.2, "correction_lm_damping": 0.001, "correction_lm_scale_floor": 1.0e-6, "correction_trust_radius": 0.1, "pairwise_steps": 2, "pairwise_strength": 0.02, "pairwise_rms_cap": 2.0, "coordinate_cap": 0.98, "coordinate_cap_power": 8}, -5.9514923837075, -2.8180710030763176),
}


def _fixture():
    rng = np.random.default_rng(20260910)
    n, dim, horizon = 16, 2, 2

    def transition(theta, points):
        return points + theta[0] * tf.sin(points)

    def transition_tangent(theta, points, d_points):
        return tf.sin(points) + d_points + theta[0] * tf.cos(points) * d_points

    def observation(points):
        return points

    model = NonlinearScoreModel(
        transition_mean_fn=transition,
        transition_mean_tangent_fn=transition_tangent,
        observation_fn=observation,
        observation_jacobian_fn=lambda points: tf.broadcast_to(tf.eye(dim, dtype=DTYPE), [tf.shape(points)[0], dim, dim]),
        observation_tangent_fn=lambda points, d_points: d_points,
        process_covariance=tf.constant(0.4 * np.eye(dim), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(dim), DTYPE),
    )
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covariances = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    design = tf.constant(np.tile(np.concatenate([np.eye(dim), -np.eye(dim)]), (n // (2 * dim), 1)), DTYPE)
    return model, tf.constant([0.6], DTYPE), initial, covariances, noises, observations, design


def _run(config: dict, *, with_score: bool = True):
    model, theta, initial, covariances, noises, observations, design = _fixture()
    return canonical_value_and_analytical_score(
        model, theta, initial, covariances, noises, observations,
        with_score=with_score, reset_design=design, **BASE, **config,
    )


@pytest.mark.parametrize("name", tuple(CONFIGS))
def test_pre_refactor_configuration_baseline(name: str):
    config, expected_value, expected_score = CONFIGS[name]
    value, score = _run(config)
    np.testing.assert_allclose(value.numpy(), expected_value, rtol=1.0e-12, atol=0.0)
    np.testing.assert_allclose(score[0].numpy(), expected_score, rtol=1.0e-12, atol=0.0)


def test_with_score_switch_does_not_change_value():
    config = CONFIGS["full_reference"][0]
    value_with, _ = _run(config, with_score=True)
    value_without, score = _run(config, with_score=False)
    np.testing.assert_array_equal(value_with.numpy(), value_without.numpy())
    assert score is None


@pytest.mark.parametrize(
    ("base_name", "inactive_overrides"),
    (
        (
            "none_1",
            {
                "correction_steps": 4,
                "correction_trust_radius": 0.1,
                "pairwise_steps": 4,
                "pairwise_rms_cap": 1.5,
                "coordinate_cap": 0.9,
            },
        ),
        (
            "contract_e_off",
            {
                "correction_trust_radius": 0.1,
                "pairwise_rms_cap": 1.5,
                "coordinate_cap": 0.9,
            },
        ),
        (
            "diagonal_trust",
            {"pairwise_rms_cap": 1.5},
        ),
    ),
)
def test_inactive_controls_have_no_effect(
    base_name: str, inactive_overrides: dict
):
    base_config = CONFIGS[base_name][0]
    base_value, base_score = _run(base_config)
    changed_value, changed_score = _run(base_config | inactive_overrides)
    np.testing.assert_array_equal(base_value.numpy(), changed_value.numpy())
    np.testing.assert_array_equal(base_score.numpy(), changed_score.numpy())


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ("none_1", "contract_e_off"),
        ("contract_e_off", "diagonal_no_trust"),
        ("diagonal_no_trust", "diagonal_trust"),
        ("diagonal_trust", "pairwise"),
        ("pairwise", "full_reference"),
        ("full_reference", "annealed_full"),
    ),
)
def test_active_configuration_changes_executed_program(left: str, right: str):
    left_value, left_score = _run(CONFIGS[left][0])
    right_value, right_score = _run(CONFIGS[right][0])
    assert not np.array_equal(left_value.numpy(), right_value.numpy())
    assert not np.array_equal(left_score.numpy(), right_score.numpy())


def test_disabled_mechanisms_cannot_be_labeled_production():
    with pytest.raises(ValueError, match="fails LEDH_PRODUCTION_PROGRAM_V1"):
        validate_ledh_production_configuration(
            reset_policy="contract_e",
            dual_cap_enabled=False,
            trust_region_enabled=False,
            program_label="production",
        )

    result = validate_ledh_production_configuration(
        reset_policy="contract_e",
        dual_cap_enabled=False,
        trust_region_enabled=False,
        program_label="ablation",
    )
    assert result["configuration_status"] == "labeled_deviation"
    assert not result["valid"]
