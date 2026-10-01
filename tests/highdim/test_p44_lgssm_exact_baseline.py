from __future__ import annotations

import math

import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter import StatePartition
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_filter
from bayesfilter.structural_tf import make_affine_structural_tf


DTYPE = tf.float64


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _observations(dim: int) -> tf.Tensor:
    values = tf.constant(
        [
            [0.10, -0.04, 0.06],
            [-0.03, 0.08, -0.02],
            [0.05, -0.01, 0.07],
        ],
        dtype=DTYPE,
    )
    return values[:, : int(dim)]


def _theta0() -> tf.Tensor:
    return tf.constant([0.25, math.log(0.18), math.log(0.12), 0.04], dtype=DTYPE)


def _physical_parts(theta: tf.Tensor, dim: int) -> dict[str, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    # Pad scales for dim > 3 with the last value
    scale = tf.constant([1.0, 0.85, 0.70] + [0.55] * max(0, dim - 3), dtype=DTYPE)[:dim]
    q_scale = tf.constant([0.90, 1.10, 1.30] + [1.0] * max(0, dim - 3), dtype=DTYPE)[:dim]
    r_scale = tf.constant([1.00, 1.20, 0.80] + [1.0] * max(0, dim - 3), dtype=DTYPE)[:dim]
    mean_scale = tf.constant([1.00, -0.50, 0.25] + [0.0] * max(0, dim - 3), dtype=DTYPE)[:dim]
    initial_cov_scale = tf.constant([0.60, 0.80, 1.00] + [1.0] * max(0, dim - 3), dtype=DTYPE)[:dim]
    rho = 0.55 * tf.tanh(theta[0]) * scale
    q_diag = tf.exp(theta[1]) * q_scale
    r_diag = tf.exp(theta[2]) * r_scale
    raw_initial_mean = theta[3] * mean_scale
    raw_initial_covariance = tf.linalg.diag(initial_cov_scale)
    transition_matrix = tf.linalg.diag(rho)
    transition_covariance = tf.linalg.diag(q_diag)
    observation_covariance = tf.linalg.diag(r_diag)
    return {
        "raw_initial_mean": raw_initial_mean,
        "raw_initial_covariance": raw_initial_covariance,
        "transition_matrix": transition_matrix,
        "transition_covariance": transition_covariance,
        "observation_matrix": tf.eye(dim, dtype=DTYPE),
        "observation_covariance": observation_covariance,
    }


def _lgssm_model(theta: tf.Tensor, dim: int) -> highdim.LinearGaussianSSM:
    parts = _physical_parts(theta, dim)
    initial_mean = tf.linalg.matvec(parts["transition_matrix"], parts["raw_initial_mean"])
    initial_covariance = (
        parts["transition_matrix"]
        @ parts["raw_initial_covariance"]
        @ tf.transpose(parts["transition_matrix"])
        + parts["transition_covariance"]
    )
    return highdim.LinearGaussianSSM(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        transition_matrix=parts["transition_matrix"],
        transition_covariance=parts["transition_covariance"],
        observation_matrix=parts["observation_matrix"],
        observation_covariance=parts["observation_covariance"],
    )


def _structural_model(theta: tf.Tensor, dim: int):
    parts = _physical_parts(theta, dim)
    padding_dim = max(0, 3 - (2 * dim))
    innovation_dim = dim + padding_dim
    partition = StatePartition(
        state_names=tuple(f"x{axis}" for axis in range(dim)),
        stochastic_indices=tuple(range(dim)),
        deterministic_indices=(),
        innovation_dim=innovation_dim,
    )
    process_chol = tf.linalg.cholesky(parts["transition_covariance"])
    innovation_matrix = tf.concat(
        [
            process_chol,
            tf.zeros([dim, padding_dim], dtype=DTYPE),
        ],
        axis=1,
    )
    return make_affine_structural_tf(
        partition=partition,
        initial_mean=parts["raw_initial_mean"],
        initial_covariance=parts["raw_initial_covariance"],
        transition_offset=tf.zeros([dim], dtype=DTYPE),
        transition_matrix=parts["transition_matrix"],
        innovation_matrix=innovation_matrix,
        innovation_covariance=tf.eye(innovation_dim, dtype=DTYPE),
        observation_offset=tf.zeros([dim], dtype=DTYPE),
        observation_matrix=parts["observation_matrix"],
        observation_covariance=parts["observation_covariance"],
        name=f"p44_m1_lgssm_dim_{dim}",
    )


def _exact_filter_config(dim: int) -> highdim.FixedBranchFilterConfig:
    return highdim.FixedBranchFilterConfig(
        fit_config=None,
        density_tau=0.0,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(highdim.IdentityCoordinateMap(dim),),
        measure_convention=_convention(),
        deterministic_seed=f"p44-m1-lgssm-exact-dim-{dim}",
    )


def _tt_artifact_filter_config(dim: int) -> highdim.FixedBranchFilterConfig:
    convention = _convention()
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 4) for _ in range(dim)],
        convention,
    )
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=tuple([1] * (dim + 1)),
            ridge=1e-10,
            max_sweeps=1,
            sweep_order=tuple(range(dim)),
            row_budget=10_000,
            column_budget=256,
            dense_matrix_byte_budget=10_000_000,
            normal_matrix_byte_budget=10_000_000,
            condition_number_warning=1e12,
            condition_number_veto=1e16,
            holdout_tolerance=1e6,
        ),
        density_tau=1e-12,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(highdim.IdentityCoordinateMap(dim),),
        measure_convention=convention,
        deterministic_seed=f"p44-m1-lgssm-tt-artifact-dim-{dim}",
        product_basis=product_basis,
        fit_quadrature_order=6,
    )


def _kalman_value(theta: tf.Tensor, dim: int) -> tf.Tensor:
    result = highdim.FixedBranchSquaredTTFilter(_exact_filter_config(dim)).log_likelihood(
        _lgssm_model(theta, dim),
        tf.zeros([0], dtype=DTYPE),
        _observations(dim),
    )
    return result.log_likelihood


def _cut4_value(theta: tf.Tensor, dim: int) -> tf.Tensor:
    result = tf_svd_cut4_filter(
        _observations(dim),
        _structural_model(theta, dim),
        innovation_floor=tf.constant(1e-12, dtype=DTYPE),
    )
    return result.log_likelihood


def _zhaocui_artifact_value(theta: tf.Tensor, dim: int) -> tuple[tf.Tensor, object]:
    result = highdim.FixedBranchSquaredTTFilter(_tt_artifact_filter_config(dim)).log_likelihood(
        _lgssm_model(theta, dim),
        tf.zeros([0], dtype=DTYPE),
        _observations(dim),
    )
    return result.log_likelihood, result


def _value_and_score(value_fn, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = value_fn(theta)
    score = tape.gradient(value, theta)
    if score is None:
        raise AssertionError("GradientTape returned None")
    return value, score


def _directions(size: int) -> tf.Tensor:
    eye = tf.eye(size, dtype=DTYPE)
    mixed_a = tf.cast(tf.range(1, size + 1), DTYPE)
    mixed_a = mixed_a / tf.linalg.norm(mixed_a)
    mixed_b = tf.where(
        tf.math.floormod(tf.range(size), 2) == 0,
        tf.ones([size], dtype=DTYPE),
        -tf.ones([size], dtype=DTYPE),
    )
    mixed_b = mixed_b / tf.linalg.norm(mixed_b)
    mixed_c = tf.reverse(mixed_a, axis=[0])
    directions = tf.concat(
        [eye, mixed_a[tf.newaxis, :], mixed_b[tf.newaxis, :], mixed_c[tf.newaxis, :]],
        axis=0,
    )
    assert int(directions.shape[0]) >= 5
    return directions


def _assert_directional_score_match(candidate: tf.Tensor, reference: tf.Tensor, atol: float) -> None:
    directional = tf.linalg.matvec(_directions(int(reference.shape[0])), candidate - reference)
    tf.debugging.assert_near(directional, tf.zeros_like(directional), atol=atol, rtol=atol)


def test_p44_m1_declares_score_parameterization_and_zhaocui_nonclaim() -> None:
    theta = _theta0()
    parts = _physical_parts(theta, dim=3)

    assert int(theta.shape[0]) == 4
    assert parts["transition_matrix"].shape == (3, 3)
    assert parts["observation_covariance"].shape == (3, 3)
    assert (
        "independent TT-propagated LGSSM likelihood is not claimed in M1"
        in "independent TT-propagated LGSSM likelihood is not claimed in M1"
    )


def test_p44_m1_scalar_tt_propagation_gap_is_not_promoted() -> None:
    theta = _theta0()
    observations = _observations(1)[:2]
    kalman_value = highdim.FixedBranchSquaredTTFilter(_exact_filter_config(1)).log_likelihood(
        _lgssm_model(theta, dim=1),
        tf.zeros([0], dtype=DTYPE),
        observations,
    ).log_likelihood
    tt_value = highdim.scalar_nonlinear_fixed_design_tt_value_path(
        _lgssm_model(theta, dim=1),
        tf.zeros([0], dtype=DTYPE),
        observations,
        _tt_artifact_filter_config(1),
        fixture_id="p44.m1.scalar-tt-gap.nonclaim",
        branch_seed_prefix="p44-m1-scalar-tt-gap",
    ).log_likelihood

    assert bool(tf.math.is_finite(tt_value).numpy())
    assert float(tf.abs(tt_value - kalman_value).numpy()) > 1e-3


def test_p44_m1_cut4_and_zhaocui_artifact_lane_match_kalman_value_and_score_dims_1_2_3() -> None:
    for dim in (1, 2, 3):
        theta = _theta0()
        kalman_value, kalman_score = _value_and_score(
            lambda current_theta: _kalman_value(current_theta, dim),
            theta,
        )
        cut4_value, cut4_score = _value_and_score(
            lambda current_theta: _cut4_value(current_theta, dim),
            theta,
        )
        artifact_value, artifact_score = _value_and_score(
            lambda current_theta: _zhaocui_artifact_value(current_theta, dim)[0],
            theta,
        )
        _artifact_value, artifact_result = _zhaocui_artifact_value(theta, dim)

        tf.debugging.assert_near(cut4_value, kalman_value, atol=5e-10, rtol=5e-10)
        tf.debugging.assert_near(cut4_score, kalman_score, atol=5e-8, rtol=5e-8)
        _assert_directional_score_match(cut4_score, kalman_score, atol=5e-8)

        tf.debugging.assert_near(artifact_value, kalman_value, atol=2e-12, rtol=2e-12)
        tf.debugging.assert_near(artifact_score, kalman_score, atol=2e-12, rtol=2e-12)
        _assert_directional_score_match(artifact_score, kalman_score, atol=2e-12)

        assert artifact_result.diagnostics["tt_artifacts_present"] is True
        assert all(step.diagnostics["tt_artifact_status"] == "OK" for step in artifact_result.steps)
        assert all(
            step.diagnostics["tt_artifact_target"]
            == "current_filtering_density_under_reference_measure"
            for step in artifact_result.steps
        )
