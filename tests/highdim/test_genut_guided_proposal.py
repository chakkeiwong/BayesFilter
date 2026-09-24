from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_candidate import (
    gaussian_genut_design,
    replicate_positive_genut,
)
from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as standard_score
from bayesfilter.highdim.cubature_genut_filter import _restore_cloud_jvp_core
from bayesfilter.highdim.genut_guided_proposal_tf import (
    LGSSMGuidedProposalSpec,
    _restore_cloud_primal,
    defensive_mixture_log_density,
    exact_lgssm_conditional_moments,
    finite_value_standard_score_guided_proposal,
    guided_lgssm_step,
)


def _matrix(dtype: tf.dtypes.DType = tf.float64) -> tf.Tensor:
    return tf.constant(
        [[1.0, 0.25, -0.15], [0.2, 1.1, 0.3], [-0.1, 0.35, 0.9]],
        dtype,
    )


def _theta(dtype: tf.dtypes.DType = tf.float64) -> tf.Tensor:
    return tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], dtype)


def _normal_log_prob(values: tf.Tensor, mean: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    residual = values - mean
    solve = tf.linalg.triangular_solve(chol, tf.transpose(residual))
    return -0.5 * (
        tf.cast(tf.shape(values)[1], values.dtype) * tf.cast(math.log(2.0 * math.pi), values.dtype)
        + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        + tf.reduce_sum(tf.square(solve), axis=0)
    )


def test_exact_conditional_matches_independent_covariance_formula() -> None:
    theta = _theta()
    parents = tf.constant([[0.3, -0.2, 0.1], [-0.4, 0.5, 0.2]], tf.float64)
    observation = tf.constant([0.8, -0.1, 0.4], tf.float64)
    mean, factor = exact_lgssm_conditional_moments(theta, parents, observation, _matrix())
    q = tf.square(theta[3]) * tf.eye(3, dtype=tf.float64)
    r = tf.square(theta[4]) * tf.eye(3, dtype=tf.float64)
    h = _matrix()
    innovation_covariance = h @ q @ tf.transpose(h) + r
    gain = tf.linalg.matmul(q, h, transpose_b=True) @ tf.linalg.inv(innovation_covariance)
    prior_mean = parents * theta[:3]
    expected_mean = prior_mean + tf.linalg.matmul(
        observation[None, :] - tf.linalg.matmul(prior_mean, h, transpose_b=True),
        gain,
        transpose_b=True,
    )
    expected_covariance = q - gain @ h @ q
    tf.debugging.assert_near(mean, expected_mean, atol=2e-12, rtol=2e-12)
    tf.debugging.assert_near(factor @ tf.transpose(factor), expected_covariance, atol=2e-12, rtol=2e-12)


def test_identity_observation_reduces_to_hand_computable_scalar_conditionals() -> None:
    theta = _theta()
    parents = tf.constant([[0.3, -0.2, 0.1]], tf.float64)
    observation = tf.constant([0.8, -0.1, 0.4], tf.float64)
    mean, factor = exact_lgssm_conditional_moments(
        theta, parents, observation, tf.eye(3, dtype=tf.float64)
    )
    q_variance = tf.square(theta[3])
    r_variance = tf.square(theta[4])
    expected_variance = q_variance * r_variance / (q_variance + r_variance)
    prior_mean = parents[0] * theta[:3]
    expected_mean = (
        r_variance * prior_mean + q_variance * observation
    ) / (q_variance + r_variance)
    tf.debugging.assert_near(mean[0], expected_mean, atol=2e-15, rtol=2e-15)
    tf.debugging.assert_near(
        tf.square(tf.linalg.diag_part(factor)),
        tf.fill([3], expected_variance),
        atol=2e-15,
        rtol=2e-15,
    )


def test_exact_conditional_corrected_weight_is_predictive_and_child_independent() -> None:
    theta = _theta()
    parent = tf.constant([[0.3, -0.2, 0.1]], tf.float64)
    parents = tf.repeat(parent, 4, axis=0)
    observation = tf.constant([0.8, -0.1, 0.4], tf.float64)
    noise = tf.constant(
        [[-1.0, 0.2, 0.7], [0.0, 0.0, 0.0], [0.4, -1.2, 0.3], [1.3, 0.8, -0.9]],
        tf.float64,
    )
    step = guided_lgssm_step(
        LGSSMGuidedProposalSpec(_matrix(), 0.0),
        theta,
        parents,
        tf.fill([4], tf.constant(0.25, tf.float64)),
        observation,
        noise,
        tf.constant([0.1, 0.3, 0.6, 0.9], tf.float64),
    )
    q = tf.square(theta[3]) * tf.eye(3, dtype=tf.float64)
    r = tf.square(theta[4]) * tf.eye(3, dtype=tf.float64)
    predicted_mean = tf.linalg.matvec(_matrix(), parent[0] * theta[:3])
    predicted_covariance = _matrix() @ q @ tf.transpose(_matrix()) + r
    predictive = _normal_log_prob(
        observation[None, :], predicted_mean[None, :], predicted_covariance
    )[0]
    corrected_without_previous = step["log_unnormalized_weights"] - tf.math.log(
        tf.constant(0.25, tf.float64)
    )
    tf.debugging.assert_near(
        corrected_without_previous,
        tf.fill([4], predictive),
        atol=3e-12,
        rtol=3e-12,
    )


def test_defensive_mixture_density_matches_direct_formula() -> None:
    base = tf.constant([-3.0, -1.2, -0.7], tf.float64)
    guided = tf.constant([-0.8, -1.5, -2.3], tf.float64)
    rho = tf.constant(0.25, tf.float64)
    actual = defensive_mixture_log_density(rho, base, guided)
    direct_density = rho * tf.exp(base) + (1.0 - rho) * tf.exp(guided)
    expected = tf.math.log(direct_density)
    tf.debugging.assert_near(actual, expected, atol=2e-15, rtol=2e-15)


def test_fixed_component_choices_replay_and_full_mixture_density_is_used() -> None:
    theta = _theta()
    parents = tf.random.stateless_normal([8, 3], [101, 102], dtype=tf.float64)
    noise = tf.random.stateless_normal([8, 3], [103, 104], dtype=tf.float64)
    uniforms = tf.constant([0.05, 0.15, 0.24, 0.3, 0.6, 0.8, 0.95, 0.99], tf.float64)
    observation = tf.constant([0.2, -0.4, 0.1], tf.float64)
    spec = LGSSMGuidedProposalSpec(_matrix(), 0.25)
    args = (
        spec,
        theta,
        parents,
        tf.fill([8], tf.constant(0.125, tf.float64)),
        observation,
        noise,
        uniforms,
    )
    first = guided_lgssm_step(*args)
    replay = guided_lgssm_step(*args)
    tf.debugging.assert_equal(first["particles"], replay["particles"])
    tf.debugging.assert_equal(first["proposal_log_density"], replay["proposal_log_density"])
    direct = tf.math.log(
        0.25 * tf.exp(first["transition_log_density"])
        + 0.75 * tf.exp(first["conditional_log_density"])
    )
    tf.debugging.assert_near(first["proposal_log_density"], direct, atol=2e-14, rtol=2e-14)


def _finite_inputs(dtype: tf.dtypes.DType, horizon: int = 2, count: int = 12):
    observations = tf.random.stateless_normal([horizon, 3], [201, 202], dtype=dtype)
    initial = tf.random.stateless_normal([count, 3], [203, 204], dtype=dtype)
    process = tf.random.stateless_normal([horizon, count, 3], [205, 206], dtype=dtype)
    uniforms = tf.random.stateless_uniform([horizon, count], [207, 208], dtype=dtype)
    design = tf.cast(
        replicate_positive_genut(gaussian_genut_design(dim=3), num_particles=count),
        dtype,
    )
    return observations, initial, process, uniforms, design


def test_primal_reset_matches_existing_zero_tangent_value_path() -> None:
    dtype = tf.float64
    count = 12
    particles = tf.random.stateless_normal([count, 3], [181, 182], dtype=dtype)
    logits = tf.random.stateless_normal([count], [183, 184], dtype=dtype)
    weights = tf.nn.softmax(logits)
    design = tf.cast(
        replicate_positive_genut(
            gaussian_genut_design(dim=3), num_particles=count
        ),
        dtype,
    )
    actual = _restore_cloud_primal(
        particles,
        weights,
        design,
        epsilon=2.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
    )
    reference = _restore_cloud_jvp_core(
        particles,
        weights,
        tf.zeros([count, 3, 1], dtype),
        tf.zeros([count, 1], dtype),
        design,
        epsilon=2.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
        parameter_count=1,
    )
    assert bool(actual["reset_valid"].numpy())
    tf.debugging.assert_near(actual["particles"], reference["particles"])
    for name in (
        "mean_residual",
        "minimum_gap_eigenvalue",
        "minimum_row_mass",
        "maximum_raw_row_residual",
        "maximum_raw_column_residual",
        "maximum_post_quotient_column_residual",
        "post_quotient_column_tv_error",
    ):
        tf.debugging.assert_near(actual[name], reference[name])


def test_standard_score_matches_existing_weighted_backward_recursion_at_t1() -> None:
    dtype = tf.float64
    theta = _theta(dtype)
    observations, initial, process, uniforms, design = _finite_inputs(
        dtype, horizon=1
    )
    value, score, diagnostics = finite_value_standard_score_guided_proposal(
        LGSSMGuidedProposalSpec(_matrix(dtype), 0.25),
        theta,
        observations,
        initial,
        process,
        uniforms,
        design,
        sinkhorn_steps=4,
    )
    parents = initial * (
        theta[3] / tf.sqrt(1.0 - tf.square(theta[:3]))
    )[None, :]
    initial_marks = standard_score._initial_target_model_score_marks(theta, parents)
    step = guided_lgssm_step(
        LGSSMGuidedProposalSpec(_matrix(dtype), 0.25),
        theta,
        parents,
        tf.fill([12], tf.constant(1.0 / 12.0, dtype)),
        observations[0],
        process[0],
        uniforms[0],
    )
    expected_marks = standard_score._target_model_progressive_score_marks(
        theta,
        parents,
        tf.fill([12], -tf.math.log(tf.constant(12.0, dtype))),
        initial_marks,
        step["particles"],
        observations[0],
    )
    assert bool(diagnostics["program_valid"].numpy())
    assert bool(tf.math.is_finite(value).numpy())
    tf.debugging.assert_near(
        score,
        tf.einsum("n,np->p", step["normalized_weights"], expected_marks),
    )
    expected_carried_marks = standard_score._target_model_progressive_score_marks(
        theta,
        parents,
        tf.fill([12], -tf.math.log(tf.constant(12.0, dtype))),
        initial_marks,
        diagnostics["final_particles"],
        observations[0],
    )
    tf.debugging.assert_near(
        diagnostics["final_score_marks"], expected_carried_marks
    )


def test_standard_score_route_compiles_with_cpu_xla() -> None:
    dtype = tf.float32
    theta = _theta(dtype)
    inputs = _finite_inputs(dtype, horizon=1)
    spec = LGSSMGuidedProposalSpec(_matrix(dtype), 0.25)

    @tf.function(jit_compile=True)
    def compiled(theta_value, observations, initial, process, uniforms, design):
        with tf.device("/CPU:0"):
            return finite_value_standard_score_guided_proposal(
                spec,
                theta_value,
                observations,
                initial,
                process,
                uniforms,
                design,
                sinkhorn_steps=4,
            )

    value, score, diagnostics = compiled(theta, *inputs)
    assert bool(diagnostics["program_valid"].numpy())
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_rho_one_replays_corrected_bootstrap_value_and_standard_score() -> None:
    dtype = tf.float32
    theta = _theta(dtype)
    observations, initial, process, uniforms, design = _finite_inputs(dtype)
    first = finite_value_standard_score_guided_proposal(
        LGSSMGuidedProposalSpec(_matrix(dtype), 1.0),
        theta,
        observations,
        initial,
        process,
        uniforms,
        design,
        sinkhorn_steps=4,
    )
    replay = finite_value_standard_score_guided_proposal(
        LGSSMGuidedProposalSpec(_matrix(dtype), 1.0),
        theta,
        observations,
        initial,
        process,
        uniforms,
        design,
        sinkhorn_steps=4,
    )
    assert bool(first[2]["program_valid"].numpy())
    assert bool(replay[2]["program_valid"].numpy())
    tf.debugging.assert_equal(first[0], replay[0])
    tf.debugging.assert_equal(first[1], replay[1])


def test_noncanonical_observation_matrix_is_rejected_by_standard_score_provider() -> None:
    with pytest.raises(tf.errors.InvalidArgumentError, match="canonical LGSSM"):
        LGSSMGuidedProposalSpec(
            tf.eye(3, dtype=tf.float64),
            0.25,
        )


def test_invalid_proposal_settings_and_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="rho"):
        LGSSMGuidedProposalSpec(_matrix(), -0.1)
    with pytest.raises(ValueError, match="transition before"):
        LGSSMGuidedProposalSpec(
            _matrix(), 0.25, transition_before_first_observation=False
        )
    theta = tf.tensor_scatter_nd_update(_theta(), [[3]], [-0.2])
    with pytest.raises(tf.errors.InvalidArgumentError):
        exact_lgssm_conditional_moments(
            theta,
            tf.zeros([2, 3], tf.float64),
            tf.zeros([3], tf.float64),
            _matrix(),
        )
