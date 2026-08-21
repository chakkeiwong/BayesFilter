from __future__ import annotations

import math

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import bayesfilter.highdim as highdim
import bayesfilter.highdim.sv_mixture_cut4 as sv_mixture_cut4_module
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_filter
from bayesfilter.structural import StatePartition, StructuralFilterConfig
from bayesfilter.structural_tf import TFStructuralStateSpace


_STD_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, dtype=tf.float64),
    scale=tf.constant(1.0, dtype=tf.float64),
)


def _observations(dim: int) -> tf.Tensor:
    values = tf.constant(
        [
            [0.12, -0.08, 0.05],
            [-0.07, 0.11, -0.04],
        ],
        dtype=tf.float64,
    )
    return values[:, : int(dim)]


def _physical_parameters(dim: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gamma = tf.constant([0.60, 0.52, 0.47], dtype=tf.float64)[: int(dim)]
    beta = tf.constant([0.40, 0.35, 0.45], dtype=tf.float64)[: int(dim)]
    sigma = tf.constant([1.00, 0.85, 0.75], dtype=tf.float64)[: int(dim)]
    return gamma, beta, sigma


def _theta_from_physical(gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tf.stack([_STD_NORMAL.quantile(gamma), tf.math.log(beta)], axis=1), [-1])


def _physical_from_theta(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    theta_matrix = tf.reshape(tf.convert_to_tensor(theta, dtype=tf.float64), [-1, 2])
    return _STD_NORMAL.cdf(theta_matrix[:, 0]), tf.exp(theta_matrix[:, 1])


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _tt_config(seed: str = "p43-sv-gradient") -> highdim.FixedBranchFilterConfig:
    convention = _convention()
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 48)],
        convention,
    )
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1),
            ridge=1e-12,
            max_sweeps=2,
            sweep_order=(0,),
            row_budget=512,
            column_budget=128,
            dense_matrix_byte_budget=200_000,
            normal_matrix_byte_budget=100_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=5e-4,
        ),
        density_tau=0.0,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=tf.float64),
                matrix=tf.constant([[8.0]], dtype=tf.float64),
            ),
        ),
        measure_convention=convention,
        deterministic_seed=seed,
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(tf.ones([1, product_basis.bases[0].basis_dim, 1], dtype=tf.float64)),
        ),
        fit_quadrature_order=141,
    )


def _ksc_kalman_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_kalman_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _ksc_cut4_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_cut4_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _ksc_fixed_sgqf_value(
    theta: tf.Tensor,
    observations: tf.Tensor,
    sigma: tf.Tensor,
    *,
    sparse_level: int = 2,
) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_fixed_sgqf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=sparse_level,
    ).log_likelihood


def _ksc_time0_fixed_sgqf_wrapper_value(
    theta: tf.Tensor,
    observations: tf.Tensor,
    sigma: tf.Tensor,
    *,
    sparse_level: int = 2,
) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_fixed_sgqf_filter(
        observations[:1],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=sparse_level,
    ).log_likelihood


def _ksc_two_observation_fixed_sgqf_wrapper_value(
    theta: tf.Tensor,
    observations: tf.Tensor,
    sigma: tf.Tensor,
    *,
    sparse_level: int = 2,
) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_fixed_sgqf_filter(
        observations[:2],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=sparse_level,
    ).log_likelihood


def _ksc_single_component_time0_ukf_value(
    theta: tf.Tensor,
    observations: tf.Tensor,
    sigma: tf.Tensor,
    *,
    component: int,
) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    z = highdim.transformed_sv_panel_observations(observations, offset=tf.constant(1e-8, dtype=tf.float64))
    mixture = highdim.ksc_1998_log_chi_square_mixture()
    current_mean = tf.zeros([1], dtype=tf.float64)
    current_covariance = tf.linalg.diag(tf.square(sigma) / (1.0 - tf.square(gamma)))
    structural = sv_mixture_cut4_module._panel_transformed_sv_component_ukf_structural_model(
        current_mean=current_mean,
        current_covariance=current_covariance,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        mixture_means=tf.reshape(mixture.means[component], [1]),
        mixture_variances=tf.reshape(mixture.variances[component], [1]),
        time_index=0,
    )
    result = tf_svd_sigma_point_filter(
        z[0:1],
        structural,
        backend="tf_svd_ukf",
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        return_filtered=True,
    )
    return result.log_likelihood


def _ksc_time0_ukf_wrapper_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_ukf_filter(
        observations[:1],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _ksc_two_observation_ukf_wrapper_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.independent_panel_sv_mixture_ukf_filter(
        observations[:2],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _lane_a_fixed_sgqf_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor, *, sparse_level: int = 2) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.exact_transformed_sv_independent_panel_fixed_sgqf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=sparse_level,
    ).log_likelihood



def _lane_b_dense_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.actual_transformed_sv_independent_panel_augmented_noise_dense_gaussian_closure_reference(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood



def _lane_b_sgqf_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor, *, sparse_level: int = 2) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=sparse_level,
    ).log_likelihood



def _lane_b_ukf_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.actual_transformed_sv_independent_panel_augmented_noise_ukf_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    ).log_likelihood


def _exact_dense_value(theta: tf.Tensor, observations: tf.Tensor, sigma: tf.Tensor) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.exact_transformed_sv_independent_panel_dense_reference(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        order=401,
        radius=8.0,
    ).log_likelihood


def _exact_tt_value(
    theta: tf.Tensor,
    observations: tf.Tensor,
    sigma: tf.Tensor,
    *,
    seed: str,
) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    return highdim.exact_transformed_sv_independent_panel_zhaocui_tt_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        config=_tt_config(seed),
        fixture_id=f"p43.exact-transformed-gradient.{seed}",
        branch_seed_prefix=f"p43-exact-transformed-gradient-{seed}",
    ).log_likelihood


def _exact_tt_score(
    theta: tf.Tensor,
    observations: tf.Tensor,
    sigma: tf.Tensor,
    *,
    seed: str,
) -> tf.Tensor:
    gamma, beta = _physical_from_theta(theta)
    derivative_config = highdim.FixedBranchDerivativeConfig(parameter_indices=tuple(range(int(theta.shape[0]))))
    return highdim.exact_transformed_sv_independent_panel_zhaocui_tt_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        config=_tt_config(seed),
        derivative_config=derivative_config,
        fixture_id=f"p43.exact-transformed-score.{seed}",
        branch_seed_prefix=f"p43-exact-transformed-score-{seed}",
    ).score


def _value_and_score(value_fn, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = value_fn(theta)
    score = tape.gradient(value, theta)
    if score is None:
        raise AssertionError("GradientTape returned None")
    return value, score


def _centered_finite_difference_score(value_fn, theta: tf.Tensor, step: float = 1e-5) -> tf.Tensor:
    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    score = []
    for axis in range(int(theta_tensor.shape[0])):
        direction = tf.one_hot(axis, int(theta_tensor.shape[0]), dtype=tf.float64) * tf.constant(step, dtype=tf.float64)
        plus = value_fn(theta_tensor + direction)
        minus = value_fn(theta_tensor - direction)
        score.append((plus - minus) / (2.0 * tf.constant(step, dtype=tf.float64)))
    return tf.stack(score)


def _relative_error(candidate: tf.Tensor, reference: tf.Tensor) -> tf.Tensor:
    return tf.linalg.norm(candidate - reference) / tf.maximum(
        tf.constant(1.0, dtype=tf.float64),
        tf.linalg.norm(reference),
    )


def _directions(size: int) -> tf.Tensor:
    eye = tf.eye(size, dtype=tf.float64)
    mixed_a = tf.cast(tf.range(1, size + 1), tf.float64)
    mixed_a = mixed_a / tf.linalg.norm(mixed_a)
    mixed_b = tf.where(
        tf.math.floormod(tf.range(size), 2) == 0,
        tf.ones([size], dtype=tf.float64),
        -tf.ones([size], dtype=tf.float64),
    )
    mixed_b = mixed_b / tf.linalg.norm(mixed_b)
    mixed_c = tf.reverse(mixed_a, axis=[0])
    directions = tf.concat(
        [eye, mixed_a[tf.newaxis, :], mixed_b[tf.newaxis, :], mixed_c[tf.newaxis, :]],
        axis=0,
    )
    if int(directions.shape[0]) < 5:
        extra = tf.ones([1, size], dtype=tf.float64) / tf.sqrt(tf.cast(size, tf.float64))
        directions = tf.concat([directions, extra], axis=0)
    return directions


def _assert_directional_residuals(
    candidate_score: tf.Tensor,
    reference_score: tf.Tensor,
    *,
    atol: float,
) -> None:
    diff = candidate_score - reference_score
    directional = tf.linalg.matvec(_directions(int(diff.shape[0])), diff)
    tf.debugging.assert_near(directional, tf.zeros_like(directional), atol=atol, rtol=atol)
    assert int(directional.shape[0]) >= 5


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_ksc_cut4_matches_kalman_value_and_diagnostic_score(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    kalman_value, kalman_score = _value_and_score(
        lambda current_theta: _ksc_kalman_value(current_theta, observations, sigma),
        theta,
    )
    cut4_value, cut4_score = _value_and_score(
        lambda current_theta: _ksc_cut4_value(current_theta, observations, sigma),
        theta,
    )

    tf.debugging.assert_near(cut4_value, kalman_value, atol=2e-6, rtol=2e-6)
    tf.debugging.assert_near(cut4_score, kalman_score, atol=2e-8, rtol=2e-8)
    tf.debugging.assert_less(_relative_error(cut4_score, kalman_score), tf.constant(1e-8, dtype=tf.float64))
    _assert_directional_residuals(cut4_score, kalman_score, atol=2e-8)


def test_p43_ksc_ukf_single_component_time0_score_matches_centered_finite_difference() -> None:
    observations = _observations(1)
    gamma, beta, sigma = _physical_parameters(1)
    theta = _theta_from_physical(gamma, beta)
    z = highdim.transformed_sv_panel_observations(observations, offset=tf.constant(1e-8, dtype=tf.float64))
    mixture = highdim.ksc_1998_log_chi_square_mixture()
    current_mean = tf.zeros([1], dtype=tf.float64)
    current_covariance = tf.linalg.diag(tf.square(sigma) / (1.0 - tf.square(gamma)))
    d_current_mean = tf.zeros([2, 1], dtype=tf.float64)
    d_current_covariance = sv_mixture_cut4_module._gamma_seed_covariance_derivatives(gamma, sigma)
    component = 0

    analytic = sv_mixture_cut4_module._ukf_component_score_update(
        observation=z[0],
        current_mean=current_mean,
        current_covariance=current_covariance,
        d_current_mean=d_current_mean,
        d_current_covariance=d_current_covariance,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        mixture_means=tf.reshape(mixture.means[component], [1]),
        mixture_variances=tf.reshape(mixture.variances[component], [1]),
        time_index=0,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _ksc_single_component_time0_ukf_value(
            current_theta,
            observations,
            sigma,
            component=component,
        ),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=2e-6, rtol=2e-6)
    assert analytic.diagnostics.backend == "tf_principal_sqrt_ukf_score"
    assert analytic.diagnostics.as_dict()["derivative_method"] == "analytic_first_order_principal_sqrt_sylvester"


def test_p43_ksc_ukf_time0_wrapper_score_matches_centered_finite_difference() -> None:
    observations = _observations(1)
    gamma, beta, sigma = _physical_parameters(1)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.independent_panel_sv_mixture_ukf_score(
        observations[:1],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _ksc_time0_ukf_wrapper_value(current_theta, observations, sigma),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=2e-6, rtol=2e-6)
    assert analytic.diagnostics["wrapper_score_contract"] == "principal_sqrt_analytic_component_score_logsumexp_aggregation"


def test_p43_ksc_ukf_two_observation_wrapper_score_matches_centered_finite_difference() -> None:
    observations = _observations(1)
    gamma, beta, sigma = _physical_parameters(1)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.independent_panel_sv_mixture_ukf_score(
        observations[:2],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _ksc_two_observation_ukf_wrapper_value(current_theta, observations, sigma),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=2e-6, rtol=2e-6)


def test_p43_ksc_fixed_sgqf_time0_wrapper_score_matches_centered_finite_difference() -> None:
    observations = _observations(1)
    gamma, beta, sigma = _physical_parameters(1)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.independent_panel_sv_mixture_fixed_sgqf_score(
        observations[:1],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=4,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _ksc_time0_fixed_sgqf_wrapper_value(
            current_theta,
            observations,
            sigma,
            sparse_level=4,
        ),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=2e-6, rtol=2e-6)
    assert analytic.diagnostics["wrapper_score_contract"] == "analytic_component_score_logsumexp_aggregation"


def test_p43_ksc_fixed_sgqf_two_observation_wrapper_score_matches_centered_finite_difference() -> None:
    observations = _observations(1)
    gamma, beta, sigma = _physical_parameters(1)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.independent_panel_sv_mixture_fixed_sgqf_score(
        observations[:2],
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=4,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _ksc_two_observation_fixed_sgqf_wrapper_value(
            current_theta,
            observations,
            sigma,
            sparse_level=4,
        ),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=2e-6, rtol=2e-6)
    assert analytic.diagnostics["wrapper_score_contract"] == "analytic_component_score_logsumexp_aggregation"


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_ksc_fixed_sgqf_matches_kalman_value_only(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    kalman_value = _ksc_kalman_value(theta, observations, sigma)
    sgqf_value = _ksc_fixed_sgqf_value(theta, observations, sigma)

    tf.debugging.assert_near(sgqf_value, kalman_value, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_exact_transformed_zhaocui_matches_dense_value_and_diagnostic_score(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    dense_value, dense_score = _value_and_score(
        lambda current_theta: _exact_dense_value(current_theta, observations, sigma),
        theta,
    )
    tt_value = _exact_tt_value(
        theta,
        observations,
        sigma,
        seed=f"dim-{dim}",
    )
    tt_score = _exact_tt_score(
        theta,
        observations,
        sigma,
        seed=f"dim-{dim}",
    )

    tf.debugging.assert_near(tt_value, dense_value, atol=2e-6, rtol=2e-6)
    tf.debugging.assert_near(tt_score, dense_score, atol=2e-5, rtol=2e-5)
    tf.debugging.assert_less(_relative_error(tt_score, dense_score), tf.constant(1e-5, dtype=tf.float64))
    _assert_directional_residuals(tt_score, dense_score, atol=2e-5)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_ksc_mixture_gradient_gap_to_exact_transformed_is_approximation_only(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    ksc_value, ksc_score = _value_and_score(
        lambda current_theta: _ksc_kalman_value(current_theta, observations, sigma),
        theta,
    )
    exact_value, exact_score = _value_and_score(
        lambda current_theta: _exact_dense_value(current_theta, observations, sigma),
        theta,
    )
    value_gap = ksc_value - exact_value
    score_gap = ksc_score - exact_score

    assert bool(tf.math.is_finite(value_gap).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score_gap)).numpy())
    assert abs(float(value_gap.numpy())) > 0.0
    assert float(tf.linalg.norm(score_gap).numpy()) > 0.0


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_ksc_fixed_sgqf_wrapper_score_matches_centered_finite_difference(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.independent_panel_sv_mixture_fixed_sgqf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=4,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _ksc_fixed_sgqf_value(
            current_theta,
            observations,
            sigma,
            sparse_level=4,
        ),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=2e-6, rtol=2e-6)
    assert analytic.diagnostics["wrapper_score_contract"] == "analytic_component_score_logsumexp_aggregation"


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_ksc_fixed_sgqf_is_value_and_analytical_score_same_target_surrogate_evidence(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    sgqf_value = _ksc_fixed_sgqf_value(theta, observations, sigma)
    cut4_value = _ksc_cut4_value(theta, observations, sigma)
    exact_value = _exact_dense_value(theta, observations, sigma)

    assert bool(tf.math.is_finite(sgqf_value).numpy())
    tf.debugging.assert_near(sgqf_value, cut4_value, atol=2e-5, rtol=2e-5)
    assert abs(float((sgqf_value - exact_value).numpy())) > 0.0


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_lane_a_fixed_sgqf_wrapper_score_matches_centered_finite_difference(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=4,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _lane_a_fixed_sgqf_value(
            current_theta,
            observations,
            sigma,
            sparse_level=4,
        ),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=2e-6, rtol=2e-6)
    assert analytic.diagnostics["wrapper_score_contract"] == "manual_forward_sensitivity_direct_likelihood_reweighting"
    assert analytic.diagnostics["derivative_method"] == "manual_forward_sensitivity_closed_form_recurrence"
    assert analytic.diagnostics["lane_id"] == "lane_a_direct_likelihood_quadrature"


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_lane_b_fixed_sgqf_wrapper_score_matches_centered_finite_difference(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=4,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _lane_b_sgqf_value(
            current_theta,
            observations,
            sigma,
            sparse_level=4,
        ),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=3e-5, rtol=3e-5)
    assert analytic.diagnostics["wrapper_score_contract"] == "gradient_tape_lane_b_sgqf_gaussian_closure"
    assert analytic.diagnostics["lane_id"] == "lane_b_augmented_noise_gaussian_closure"


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_lane_b_principal_sqrt_ukf_wrapper_score_matches_centered_finite_difference(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    analytic = highdim.actual_transformed_sv_independent_panel_augmented_noise_ukf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    )
    finite_difference = _centered_finite_difference_score(
        lambda current_theta: _lane_b_ukf_value(current_theta, observations, sigma),
        theta,
    )

    tf.debugging.assert_near(analytic.score, finite_difference, atol=3e-5, rtol=3e-5)
    assert analytic.diagnostics["wrapper_score_contract"] == "gradient_tape_lane_b_ukf_gaussian_closure"
    assert analytic.diagnostics["backend_role"] == "diagnostic_autodiff_score_only"
    assert analytic.diagnostics["score_admission_status"] == (
        "not_admitted_requires_reviewed_analytical_augmented_noise_score"
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_lane_b_value_gap_to_lane_a_is_approximation_only(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    lane_a_value, lane_a_score = _value_and_score(
        lambda current_theta: _exact_dense_value(current_theta, observations, sigma),
        theta,
    )
    lane_b_value, lane_b_score = _value_and_score(
        lambda current_theta: _lane_b_dense_value(current_theta, observations, sigma),
        theta,
    )
    value_gap = lane_b_value - lane_a_value
    score_gap = lane_b_score - lane_a_score

    assert bool(tf.math.is_finite(value_gap).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score_gap)).numpy())
    assert abs(float(value_gap.numpy())) > 0.0
    assert float(tf.linalg.norm(score_gap).numpy()) > 0.0




@pytest.mark.parametrize("dim", [1, 2, 3])
def test_p43_lane_b_sgqf_gradient_stays_closer_than_ukf_to_dense_reference(dim: int) -> None:
    observations = _observations(dim)
    gamma, beta, sigma = _physical_parameters(dim)
    theta = _theta_from_physical(gamma, beta)

    _dense_value, dense_score = _value_and_score(
        lambda current_theta: _lane_b_dense_value(current_theta, observations, sigma),
        theta,
    )
    sgqf = highdim.actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        sparse_level=4,
    )
    ukf = highdim.actual_transformed_sv_independent_panel_augmented_noise_ukf_score(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
    )

    sgqf_rel = _relative_error(sgqf.score, dense_score)
    ukf_rel = _relative_error(ukf.score, dense_score)
    sgqf_abs = tf.linalg.norm(sgqf.score - dense_score)
    ukf_abs = tf.linalg.norm(ukf.score - dense_score)

    tf.debugging.assert_less(tf.constant(sgqf_rel, dtype=tf.float64), tf.constant(ukf_rel, dtype=tf.float64))
    tf.debugging.assert_less(sgqf_abs, ukf_abs)


def _generalized_sv_diagnostic_model(
    *,
    observation: tf.Tensor,
    beta: tf.Tensor,
) -> TFStructuralStateSpace:
    gamma_s = tf.constant(0.55, dtype=tf.float64)
    gamma_h = tf.constant(0.65, dtype=tf.float64)
    sigma_s = tf.constant(0.30, dtype=tf.float64)
    sigma_h = tf.constant(0.45, dtype=tf.float64)
    initial_covariance = tf.linalg.diag(
        tf.stack(
            [
                tf.square(sigma_s) / (1.0 - tf.square(gamma_s)),
                tf.square(sigma_h) / (1.0 - tf.square(gamma_h)),
            ]
        )
    )
    observed_scalar = tf.reshape(tf.convert_to_tensor(observation, dtype=tf.float64), [])

    def transition_fn(previous_state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        return tf.convert_to_tensor(previous_state, dtype=tf.float64)

    def observation_fn(state_points: tf.Tensor) -> tf.Tensor:
        points = tf.convert_to_tensor(state_points, dtype=tf.float64)
        if points.shape.rank == 1:
            points = points[tf.newaxis, :]
        residual = observed_scalar - beta * points[:, 0]
        transformed_residual = tf.math.log(tf.square(residual) + tf.constant(1e-8, dtype=tf.float64)) - points[:, 1]
        return transformed_residual[:, tf.newaxis]

    return TFStructuralStateSpace(
        partition=StatePartition(
            state_names=("s", "h"),
            stochastic_indices=(0, 1),
            deterministic_indices=(),
            innovation_dim=2,
        ),
        config=StructuralFilterConfig(
            integration_space="innovation",
            deterministic_completion="none",
            approximation_label="p43_generalized_sv_cut4_gradient_diagnostic_not_exact",
        ),
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=initial_covariance,
        innovation_covariance=tf.eye(2, dtype=tf.float64),
        observation_covariance=tf.reshape(tf.constant(1.0, dtype=tf.float64), [1, 1]),
        transition_fn=transition_fn,
        observation_fn=observation_fn,
        name="p43_generalized_sv_cut4_gradient_diagnostic_not_exact",
    )


def test_p43_generalized_sv_cut4_diagnostic_value_and_gradient_are_finite_with_nonclaim() -> None:
    theta = tf.constant(math.log(0.35), dtype=tf.float64)

    with tf.GradientTape() as tape:
        tape.watch(theta)
        beta = tf.exp(theta)
        model = _generalized_sv_diagnostic_model(
            observation=tf.constant(0.18, dtype=tf.float64),
            beta=beta,
        )
        result = tf_svd_cut4_filter(
            tf.constant([[0.0]], dtype=tf.float64),
            model,
            return_filtered=True,
        )
        value = result.log_likelihood
    score = tape.gradient(value, theta)

    assert bool(tf.math.is_finite(value).numpy())
    assert score is not None
    assert bool(tf.math.is_finite(score).numpy())
    assert result.metadata.approximation_label == "p43_generalized_sv_cut4_gradient_diagnostic_not_exact"
    assert int(result.diagnostics.extra["augmented_dim"].numpy()) == 4
