from __future__ import annotations

import math
from pathlib import Path

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.predictive_equivalence import (
    PredictiveContractError,
    batched_chain_bartlett_long_run_covariance,
    batched_quadratic_loss_confidence_bounds,
    chain_bartlett_long_run_covariance,
    chain_batch_long_run_covariance,
    classify_dual_proper_score_equivalence,
    classify_proper_score_equivalence,
    growing_hac_bandwidth,
    horizon_proper_score_loss,
    proper_score_loss,
    quadratic_loss_confidence_bounds,
)


F64 = tf.float64
ROOT = Path(__file__).resolve().parents[1]


def test_growing_hac_bandwidth_has_required_asymptotic_shape() -> None:
    draw_grid = (64, 512, 4096, 32768)
    bandwidths = tuple(growing_hac_bandwidth(draws) for draws in draw_grid)
    assert bandwidths == (4, 8, 16, 32)
    assert all(left < right for left, right in zip(bandwidths, bandwidths[1:]))
    fractions = tuple(bandwidth / draws for bandwidth, draws in zip(bandwidths, draw_grid))
    assert all(left > right for left, right in zip(fractions, fractions[1:]))


@pytest.mark.parametrize(
    "draw_count,multiplier,match",
    [
        (1, 1.0, "at least two"),
        (16, 0.0, "positive"),
        (2, 10.0, "smaller"),
    ],
)
def test_growing_hac_bandwidth_rejects_invalid_contract(
    draw_count: int, multiplier: float, match: str
) -> None:
    with pytest.raises(PredictiveContractError, match=match):
        growing_hac_bandwidth(draw_count, multiplier=multiplier)


def test_bartlett_hac_matches_hand_calculation_and_pooled_scaling() -> None:
    values = tf.constant(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[1.0], [0.0], [-1.0], [0.0]],
        ],
        F64,
    )
    result = chain_bartlett_long_run_covariance(values, jit_compile=False)
    centered = values - tf.reduce_mean(values, axis=1, keepdims=True)
    gamma_zero = tf.reduce_mean(
        tf.reduce_sum(centered * centered, axis=1) / tf.constant(4.0, F64),
        axis=0,
    )
    gamma_one = tf.reduce_mean(
        tf.reduce_sum(centered[:, 1:] * centered[:, :-1], axis=1)
        / tf.constant(4.0, F64),
        axis=0,
    )
    expected_spectral = gamma_zero + gamma_one
    tf.debugging.assert_near(
        result.spectral_covariance[0, 0], expected_spectral[0], atol=1e-14
    )
    tf.debugging.assert_near(
        result.pooled_mean_covariance[0, 0],
        expected_spectral[0] / tf.constant(8.0, F64),
        atol=1e-14,
    )
    assert result.bandwidth == 1
    assert result.chain_count == 2
    assert result.draw_count == 4
    assert result.numerically_admissible
    assert result.inference_admissible


def test_bartlett_hac_fail_closed_ridge_policy() -> None:
    base = tf.reshape(tf.cast(tf.range(4 * 32), F64), [4, 32, 1])
    singular = tf.concat((base, 2.0 * base), axis=-1)
    invalid = chain_bartlett_long_run_covariance(
        singular,
        ridge_ladder=(0.0,),
        condition_number_max=1.0e12,
        jit_compile=False,
    )
    assert not invalid.numerically_admissible
    assert not invalid.inference_admissible
    assert int(invalid.selected_ridge_index) == -1
    assert bool(tf.reduce_all(tf.math.is_nan(invalid.precision)))
    repaired = chain_bartlett_long_run_covariance(
        singular,
        ridge_ladder=(0.0, 1.0e-8),
        condition_number_max=1.0e12,
        jit_compile=False,
    )
    assert repaired.numerically_admissible
    assert not repaired.inference_admissible
    assert int(repaired.selected_ridge_index) == 1

    with pytest.raises(PredictiveContractError, match="begin at zero"):
        chain_bartlett_long_run_covariance(
            singular,
            ridge_ladder=(1.0e-8,),
            condition_number_max=1.0e12,
            jit_compile=False,
        )


def test_bartlett_hac_default_xla_matches_eager() -> None:
    draw = tf.reshape(tf.cast(tf.range(64), F64), (1, 64, 1))
    chain = tf.reshape(tf.cast(tf.range(4), F64), (4, 1, 1))
    feature = tf.reshape(tf.constant([1.0, 1.7, -0.4], F64), (1, 1, 3))
    values = tf.sin((draw + 0.3 * chain + 1.0) * feature)
    eager = chain_bartlett_long_run_covariance(values, jit_compile=False)
    compiled = chain_bartlett_long_run_covariance(values)
    for name in (
        "spectral_covariance",
        "pooled_mean_covariance",
        "regularized_covariance",
        "precision",
        "eigenvalues",
        "condition_number",
    ):
        tf.debugging.assert_near(getattr(compiled, name), getattr(eager, name), atol=1e-10)
    assert compiled.bandwidth == eager.bandwidth == 4
    assert compiled.numerically_admissible == eager.numerically_admissible
    assert compiled.inference_admissible == eager.inference_admissible


def _ar1_chains(*, chain_count: int, draw_count: int, phi: float) -> tf.Tensor:
    burnin = 1024
    innovations = tf.random.stateless_normal(
        [draw_count + burnin, chain_count],
        seed=tf.constant([20260717, 1901], tf.int32),
        dtype=F64,
    ) * math.sqrt(1.0 - phi**2)

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return tf.constant(phi, F64) * previous + innovation

    paths = tf.scan(transition, innovations, initializer=tf.zeros([chain_count], F64))
    return tf.transpose(paths[burnin:])[:, :, tf.newaxis]


def test_growing_hac_repairs_fixed_sixteen_ar1_limit() -> None:
    phi = 0.6
    truth = (1.0 + phi) / (1.0 - phi)
    fixed_limit = 1.0 + 2.0 * sum(
        (1.0 - lag / 16.0) * phi**lag for lag in range(1, 16)
    )
    assert truth == pytest.approx(4.0)
    assert fixed_limit == pytest.approx(3.531382239526912)
    assert 1.0 - fixed_limit / truth == pytest.approx(0.11715444011827203)

    paths = _ar1_chains(chain_count=32, draw_count=32768, phi=phi)
    short = chain_bartlett_long_run_covariance(
        paths[:, :4096], jit_compile=False
    ).spectral_covariance[0, 0]
    long = chain_bartlett_long_run_covariance(
        paths, jit_compile=False
    ).spectral_covariance[0, 0]
    fixed = chain_batch_long_run_covariance(
        paths, block_length=16, jit_compile=False
    ).spectral_covariance[0, 0]
    assert abs(float(long) - truth) < abs(float(short) - truth)
    assert abs(float(long) - truth) < 0.75 * abs(float(fixed) - truth)


def test_proper_score_loss_constants_and_order_match_chapter() -> None:
    weights = tf.constant([0.25, 0.75], F64)
    loss = proper_score_loss(weights)
    tf.debugging.assert_near(
        tf.linalg.diag_part(loss.loss_matrix),
        tf.constant([0.125, 0.375, 0.0625, 0.1875], F64),
    )
    text = (
        ROOT / "docs/chapters/ch28a_neural_network_state_space_model_applications.tex"
    ).read_text(encoding="utf-8")
    assert "\\frac12\\delta_{\\mu,h}^2" in text
    assert "\\frac14\\delta_{\\log v,h}^2" in text
    assert "\\ell_N=\\max\\{1,\\lfloor\\kappa_{\\rm HAC}N^{1/3}\\rfloor\\}" in text
    assert "3.53138" in text
    assert "\\label{eq:bf-ssl-lstm-horizon-loss}" in text
    assert "\\label{eq:bf-ssl-lstm-dual-loss-threshold}" in text
    assert "\\label{eq:bf-ssl-lstm-dual-equivalence}" in text
    assert "0.0012448" in text
    assert "0.0068491" in text
    assert "5/256=1.95\\%" in text
    assert "76/256=29.69\\%" in text


@pytest.mark.parametrize(
    "weights",
    [
        tf.constant([0.0, 1.0], F64),
        tf.constant([0.4, 0.4], F64),
    ],
)
def test_proper_score_loss_rejects_invalid_weights(weights: tf.Tensor) -> None:
    with pytest.raises(PredictiveContractError, match="positive and sum to one"):
        proper_score_loss(weights)


def test_horizon_proper_score_loss_embeds_rank_two_loss() -> None:
    loss = horizon_proper_score_loss(3, 1)
    tf.debugging.assert_near(
        tf.linalg.diag_part(loss.loss_matrix),
        tf.constant([0.0, 0.5, 0.0, 0.0, 0.25, 0.0], F64),
    )
    assert int(tf.math.count_nonzero(tf.linalg.diag_part(loss.loss_matrix))) == 2
    with pytest.raises(PredictiveContractError, match="outside"):
        horizon_proper_score_loss(3, 3)


def _brute_force_bounds(
    estimate: tf.Tensor,
    covariance: tf.Tensor,
    loss_matrix: tf.Tensor,
    radius_squared: tf.Tensor,
) -> tuple[float, float]:
    factor = tf.linalg.cholesky(covariance)
    angles = tf.linspace(tf.constant(0.0, F64), tf.constant(2.0 * math.pi, F64), 400001)
    circle = tf.stack((tf.cos(angles), tf.sin(angles)), axis=1)
    boundary = estimate[tf.newaxis, :] + tf.sqrt(radius_squared) * tf.matmul(
        circle, factor, transpose_b=True
    )
    candidates = tf.concat((estimate[tf.newaxis, :], boundary), axis=0)
    values = tf.einsum("ni,ij,nj->n", candidates, loss_matrix, candidates)
    return float(tf.reduce_min(values)), float(tf.reduce_max(values))


def test_quadratic_loss_bounds_match_dense_boundary_reference() -> None:
    loss = proper_score_loss(tf.constant([1.0], F64))
    estimate = tf.constant([0.25, -0.35], F64)
    covariance = tf.constant([[0.04, 0.012], [0.012, 0.03]], F64)
    bounds = quadratic_loss_confidence_bounds(
        estimate, covariance, loss, jit_compile=False
    )
    reference_lower, reference_upper = _brute_force_bounds(
        estimate,
        covariance,
        loss.loss_matrix,
        bounds.confidence_radius_squared,
    )
    assert bounds.inference_admissible
    assert float(bounds.lower_bound) == pytest.approx(reference_lower, abs=2e-6)
    assert float(bounds.upper_bound) == pytest.approx(reference_upper, abs=2e-6)
    assert float(bounds.lower_kkt_residual) < 1e-9
    assert float(bounds.upper_kkt_residual) < 1e-9


def test_quadratic_loss_bounds_handle_centered_hard_case_and_interior_minimum() -> None:
    loss = proper_score_loss(tf.constant([1.0], F64))
    covariance = tf.eye(2, dtype=F64) * tf.constant(0.01, F64)
    bounds = quadratic_loss_confidence_bounds(
        tf.zeros([2], F64), covariance, loss, jit_compile=False
    )
    radius_squared = tfp.distributions.Chi2(tf.constant(2.0, F64)).quantile(
        tf.constant(0.95, F64)
    )
    assert float(bounds.lower_bound) == pytest.approx(0.0, abs=1e-14)
    assert float(bounds.upper_bound) == pytest.approx(
        float(radius_squared) * 0.01 * 0.5, abs=1e-12
    )
    tf.debugging.assert_near(bounds.lower_optimizer, tf.zeros([2], F64), atol=1e-14)
    assert bounds.inference_admissible


def test_quadratic_loss_bounds_support_rank_deficient_horizon_loss() -> None:
    loss = horizon_proper_score_loss(2, 0)
    estimate = tf.constant([0.25, -3.0, -0.35, 2.0], F64)
    covariance = tf.constant(
        [
            [0.04, 0.0, 0.012, 0.0],
            [0.0, 0.03, 0.0, 0.0],
            [0.012, 0.0, 0.03, 0.0],
            [0.0, 0.0, 0.0, 0.02],
        ],
        F64,
    )
    bounds = quadratic_loss_confidence_bounds(
        estimate, covariance, loss, jit_compile=False
    )
    assert bounds.inference_admissible
    assert float(bounds.lower_bound) >= -1.0e-14
    assert float(bounds.lower_kkt_residual) < 1.0e-8
    assert float(bounds.upper_kkt_residual) < 1.0e-8


def test_batched_quadratic_bounds_match_scalar_for_average_and_horizons() -> None:
    estimates = tf.constant(
        [[0.1, -0.2, 0.05, 0.15], [0.25, 0.1, -0.15, -0.05]], F64
    )
    covariances = tf.stack(
        (
            tf.linalg.diag(tf.constant([0.02, 0.03, 0.01, 0.015], F64)),
            tf.constant(
                [
                    [0.03, 0.004, 0.0, 0.0],
                    [0.004, 0.025, 0.0, 0.0],
                    [0.0, 0.0, 0.012, 0.001],
                    [0.0, 0.0, 0.001, 0.018],
                ],
                F64,
            ),
        )
    )
    losses = (
        proper_score_loss(tf.constant([0.5, 0.5], F64)),
        horizon_proper_score_loss(2, 0),
        horizon_proper_score_loss(2, 1),
    )
    batched = batched_quadratic_loss_confidence_bounds(
        estimates,
        covariances,
        tf.stack([loss.loss_matrix for loss in losses]),
        jit_compile=False,
    )
    assert bool(tf.reduce_all(batched.inference_admissible))
    for batch_index in range(2):
        for loss_index, loss in enumerate(losses):
            scalar = quadratic_loss_confidence_bounds(
                estimates[batch_index],
                covariances[batch_index],
                loss,
                jit_compile=False,
            )
            tf.debugging.assert_near(
                batched.point_loss[batch_index, loss_index], scalar.point_loss, atol=1e-11
            )
            tf.debugging.assert_near(
                batched.lower_bound[batch_index, loss_index], scalar.lower_bound, atol=1e-10
            )
            tf.debugging.assert_near(
                batched.upper_bound[batch_index, loss_index], scalar.upper_bound, atol=1e-10
            )


def test_batched_quadratic_bounds_reject_asymmetric_covariance() -> None:
    estimate = tf.zeros([1, 2], F64)
    covariance = tf.constant([[[1.0, 0.1], [0.0, 1.0]]], F64)
    loss = proper_score_loss(tf.constant([1.0], F64))
    with pytest.raises(PredictiveContractError, match="symmetric"):
        batched_quadratic_loss_confidence_bounds(
            estimate,
            covariance,
            loss.loss_matrix[tf.newaxis, :, :],
            jit_compile=False,
        )


def test_batched_bartlett_matches_scalar_and_preserves_zero_ridge_veto() -> None:
    draw = tf.reshape(tf.cast(tf.range(64), F64), (1, 64, 1))
    chain = tf.reshape(tf.cast(tf.range(4), F64), (4, 1, 1))
    feature = tf.reshape(tf.constant([1.0, 1.7, -0.4], F64), (1, 1, 3))
    first = tf.sin((draw + 0.3 * chain + 1.0) * feature)
    second = tf.cos((draw + 0.2 * chain + 0.5) * feature)
    values = tf.stack((first, second))
    batched = batched_chain_bartlett_long_run_covariance(values, jit_compile=False)
    assert bool(tf.reduce_all(batched.inference_admissible))
    for index in range(2):
        scalar = chain_bartlett_long_run_covariance(
            values[index], ridge_ladder=(0.0,), jit_compile=False
        )
        tf.debugging.assert_near(
            batched.pooled_mean_covariance[index], scalar.pooled_mean_covariance, atol=1e-12
        )
    singular = tf.concat((values[:, :, :, :1], values[:, :, :, :1]), axis=-1)
    invalid = batched_chain_bartlett_long_run_covariance(
        singular, ridge_ladder=(0.0,), jit_compile=False
    )
    assert not bool(tf.reduce_any(invalid.inference_admissible))


def test_quadratic_loss_bounds_fail_closed_for_singular_covariance() -> None:
    loss = proper_score_loss(tf.constant([1.0], F64))
    result = quadratic_loss_confidence_bounds(
        tf.constant([0.1, 0.2], F64),
        tf.constant([[1.0, 1.0], [1.0, 1.0]], F64),
        loss,
        jit_compile=False,
    )
    assert not result.inference_admissible
    assert bool(tf.math.is_nan(result.lower_bound))


def test_quadratic_loss_bounds_default_xla_matches_eager() -> None:
    loss = proper_score_loss(tf.constant([0.4, 0.6], F64))
    estimate = tf.constant([0.1, -0.2, 0.05, 0.15], F64)
    covariance = tf.linalg.diag(tf.constant([0.02, 0.03, 0.01, 0.015], F64))
    eager = quadratic_loss_confidence_bounds(
        estimate, covariance, loss, jit_compile=False
    )
    compiled = quadratic_loss_confidence_bounds(estimate, covariance, loss)
    for name in (
        "point_loss",
        "lower_bound",
        "upper_bound",
        "lower_optimizer",
        "upper_optimizer",
        "lower_kkt_residual",
        "upper_kkt_residual",
    ):
        tf.debugging.assert_near(getattr(compiled, name), getattr(eager, name), atol=1e-10)
    assert compiled.inference_admissible == eager.inference_admissible


def test_proper_score_three_way_decision_and_mechanics_veto() -> None:
    loss = proper_score_loss(tf.constant([1.0], F64))
    covariance = tf.eye(2, dtype=F64) * tf.constant(0.01, F64)
    passing = quadratic_loss_confidence_bounds(
        tf.zeros([2], F64), covariance, loss, jit_compile=False
    )
    material = quadratic_loss_confidence_bounds(
        tf.constant([1.0, 0.0], F64), covariance, loss, jit_compile=False
    )
    inconclusive = quadratic_loss_confidence_bounds(
        tf.constant([0.3, 0.0], F64), covariance, loss, jit_compile=False
    )
    tolerance = tf.constant(0.1, F64)
    assert classify_proper_score_equivalence(
        passing, acceptable_loss=tolerance
    ).status == "PASS"
    assert classify_proper_score_equivalence(
        material, acceptable_loss=tolerance
    ).status == "MATERIAL_DIFFERENCE"
    assert classify_proper_score_equivalence(
        inconclusive, acceptable_loss=tolerance
    ).status == "INCONCLUSIVE_UNDERPOWERED"
    veto = classify_proper_score_equivalence(
        passing, acceptable_loss=tolerance, mechanics_only=True
    )
    assert veto.status == "INVALID_HARD_VETO"
    assert veto.hard_veto_codes == ("MECHANICS_ONLY_CANNOT_PASS",)


def test_dual_proper_score_decision_uses_average_and_horizonwise_rules() -> None:
    average_loss = proper_score_loss(tf.fill([10], tf.constant(0.1, F64)))
    horizon_losses = tuple(horizon_proper_score_loss(10, index) for index in range(10))
    covariance = tf.eye(20, dtype=F64) * tf.constant(1.0e-7, F64)
    estimate = tf.tensor_scatter_nd_update(
        tf.zeros([20], F64), [[1]], [tf.constant(0.2, F64)]
    )
    average_bounds = quadratic_loss_confidence_bounds(
        estimate, covariance, average_loss, jit_compile=False
    )
    horizon_bounds = tuple(
        quadratic_loss_confidence_bounds(
            estimate, covariance, loss, jit_compile=False
        )
        for loss in horizon_losses
    )
    decision = classify_dual_proper_score_equivalence(
        average_bounds,
        horizon_bounds,
        acceptable_average_loss=tf.constant(0.006849, F64),
        acceptable_horizon_loss=tf.constant(0.006849, F64),
    )
    assert float(average_bounds.upper_bound) < 0.006849
    assert decision.status == "MATERIAL_DIFFERENCE"
    assert bool(tf.reduce_any(decision.horizon_loss_lower_bounds > 0.006849))


def test_dual_proper_score_decision_rejects_mixed_joint_regions() -> None:
    average_loss = proper_score_loss(tf.constant([0.5, 0.5], F64))
    horizon_losses = tuple(horizon_proper_score_loss(2, index) for index in range(2))
    covariance = tf.eye(4, dtype=F64) * tf.constant(0.001, F64)
    average = quadratic_loss_confidence_bounds(
        tf.zeros([4], F64), covariance, average_loss, jit_compile=False
    )
    horizons = tuple(
        quadratic_loss_confidence_bounds(
            tf.ones([4], F64) * (0.01 if index == 0 else 0.0),
            covariance,
            loss,
            jit_compile=False,
        )
        for index, loss in enumerate(horizon_losses)
    )
    decision = classify_dual_proper_score_equivalence(
        average,
        horizons,
        acceptable_average_loss=tf.constant(0.01, F64),
        acceptable_horizon_loss=tf.constant(0.01, F64),
    )
    assert decision.status == "INVALID_HARD_VETO"
    assert decision.hard_veto_codes == ("LOSS_BOUNDS_NOT_ONE_JOINT_REGION",)


def test_proper_score_decision_rejects_tampered_bounds() -> None:
    loss = proper_score_loss(tf.constant([1.0], F64))
    bounds = quadratic_loss_confidence_bounds(
        tf.zeros([2], F64),
        tf.eye(2, dtype=F64) * tf.constant(0.01, F64),
        loss,
        jit_compile=False,
    )
    object.__setattr__(bounds, "upper_bound", tf.constant(0.0, F64))
    decision = classify_proper_score_equivalence(
        bounds, acceptable_loss=tf.constant(0.1, F64)
    )
    assert decision.status == "INVALID_HARD_VETO"
    assert decision.hard_veto_codes == ("LOSS_BOUNDS_UNAUTHENTICATED",)
