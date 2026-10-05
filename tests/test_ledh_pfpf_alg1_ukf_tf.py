from __future__ import annotations

import numpy as np
import tensorflow as tf

from pathlib import Path

from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_alg1_ukf_tf import (
    METHOD_GENERATION,
    OT_CANONICAL_TRANSPORT_CONVENTION,
    OT_COVARIANCE_CARRY_ROUTE,
    OT_PFPF_CORRECTION_ROUTE,
    OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
    apply_classical_resampling_state_tf,
    apply_ot_resampling_state_tf,
    algorithm1_route_identifiers,
    canonical_transport_from_sinkhorn_coupling_tf,
    carry_covariances_with_canonical_transport_tf,
    ledh_alg1_coefficients_tf,
    li_coates_ledh_alg1_time_step_tf,
    li_coates_ledh_alg1_time_step_vectorized_particles_tf,
    run_ledh_pfpf_alg1_scalar_sv_graph_tf,
    run_ledh_pfpf_alg1_ukf_tf,
    ukf_predict_additive_tf,
    ukf_update_additive_tf,
    validate_algorithm1_route_identifiers,
    validate_pseudo_time_steps_tf,
)


DTYPE = tf.float64
ROOT = Path(__file__).resolve().parents[1]


def test_ukf_prediction_matches_linear_kalman_and_deterministic_edge() -> None:
    transition_matrix = tf.constant([[0.9, 0.2], [-0.1, 0.7]], dtype=DTYPE)
    x_prev = tf.constant([0.3, -0.4], dtype=DTYPE)
    p_prev = tf.constant([[0.6, 0.1], [0.1, 0.4]], dtype=DTYPE)
    q = tf.zeros([2, 2], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.linalg.matmul(points, transition_matrix, transpose_b=True)

    result = ukf_predict_additive_tf(
        previous_state=x_prev,
        previous_covariance=p_prev,
        transition_mean_fn=transition_mean,
        process_noise_covariance=q,
        time_index=0,
        covariance_floor=0.0,
    )

    expected_mean = tf.linalg.matvec(transition_matrix, x_prev)
    expected_covariance = transition_matrix @ p_prev @ tf.transpose(transition_matrix)
    np.testing.assert_allclose(result.mean.numpy(), expected_mean.numpy(), atol=1e-12)
    np.testing.assert_allclose(result.covariance.numpy(), expected_covariance.numpy(), atol=1e-12)


def test_ukf_update_matches_identity_observation_kalman() -> None:
    mean = tf.constant([0.2, -0.1], dtype=DTYPE)
    covariance = tf.constant([[0.5, 0.05], [0.05, 0.3]], dtype=DTYPE)
    observation = tf.constant([0.4, -0.2], dtype=DTYPE)
    r = tf.linalg.diag(tf.constant([0.2, 0.4], dtype=DTYPE))

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.identity(points)

    result = ukf_update_additive_tf(
        predicted_mean=mean,
        predicted_covariance=covariance,
        observation=observation,
        observation_mean_fn=observation_mean,
        observation_covariance=r,
        time_index=0,
        covariance_floor=0.0,
    )

    innovation_covariance = covariance + r
    gain = tf.transpose(tf.linalg.solve(innovation_covariance, tf.transpose(covariance)))
    expected_mean = mean + tf.linalg.matvec(gain, observation - mean)
    expected_covariance = covariance - gain @ innovation_covariance @ tf.transpose(gain)
    np.testing.assert_allclose(result.mean.numpy(), expected_mean.numpy(), atol=1e-12)
    np.testing.assert_allclose(result.covariance.numpy(), expected_covariance.numpy(), atol=1e-12)


def test_ledh_coefficients_use_zero_noise_anchor_and_predicted_covariance() -> None:
    predicted_covariance = tf.constant([[0.7]], dtype=DTYPE)
    observation_covariance = tf.constant([[0.2]], dtype=DTYPE)
    auxiliary_state = tf.constant([0.5], dtype=DTYPE)
    zero_noise_anchor = tf.constant([0.1], dtype=DTYPE)
    observation = tf.constant([0.4], dtype=DTYPE)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(tf.cast(points, DTYPE), [-1, 1])
        return 2.0 * values + 0.3

    def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
        del point
        return tf.constant([[2.0]], dtype=DTYPE)

    coefficients = ledh_alg1_coefficients_tf(
        auxiliary_state=auxiliary_state,
        zero_noise_anchor=zero_noise_anchor,
        predicted_covariance=predicted_covariance,
        observation=observation,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        observation_covariance=observation_covariance,
        lambda_value=tf.constant(0.25, dtype=DTYPE),
        time_index=0,
        covariance_floor=0.0,
    )

    h = 2.0
    p = 0.7
    r = 0.2
    lam = 0.25
    e = 0.3
    a = -0.5 * p * h * (lam * h * p * h + r) ** -1 * h
    b = (1.0 + 2.0 * lam * a) * (
        (1.0 + lam * a) * p * h * (1.0 / r) * (0.4 - e)
        + a * 0.1
    )
    np.testing.assert_allclose(coefficients["A"].numpy(), [[a]], atol=1e-12)
    np.testing.assert_allclose(coefficients["b"].numpy(), [b], atol=1e-12)


def test_time_step_uses_per_particle_covariances_and_det_product() -> None:
    ancestors = tf.constant([[0.0], [1.0]], dtype=DTYPE)
    previous_covariances = tf.constant([[[0.1]], [[0.4]]], dtype=DTYPE)
    pre_flow = tf.constant([[0.1], [1.2]], dtype=DTYPE)
    observation = tf.constant([0.25], dtype=DTYPE)
    eps = tf.constant([0.5, 0.5], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.identity(tf.reshape(points, [-1, 1]))

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - x_prev, [-1])
        return -0.5 * residual * residual

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.05]], dtype=DTYPE)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return values

    def observation_jacobian(_point: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[1.0]], dtype=DTYPE)

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.2]], dtype=DTYPE)

    result = li_coates_ledh_alg1_time_step_tf(
        ancestors=ancestors,
        previous_covariances=previous_covariances,
        pre_flow_particles=pre_flow,
        observation=observation,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        time_index=0,
        pseudo_time_steps=eps,
        covariance_floor=0.0,
    )

    assert result.predicted_covariances.shape == (2, 1, 1)
    np.testing.assert_allclose(
        result.predicted_covariances.numpy()[:, 0, 0],
        np.array([0.15, 0.45]),
        atol=1e-12,
    )
    assert not np.allclose(result.forward_log_det.numpy()[0], result.forward_log_det.numpy()[1])
    assert result.diagnostics["flow_anchor_route"] == "zero_noise_transition"
    assert result.diagnostics["finite_forward_log_det"]


def test_pseudo_time_steps_must_be_positive_and_sum_to_one() -> None:
    np.testing.assert_allclose(
        validate_pseudo_time_steps_tf(tf.constant([0.25, 0.75], dtype=DTYPE)).numpy(),
        [0.25, 0.75],
    )

    with np.testing.assert_raises(ValueError):
        validate_pseudo_time_steps_tf(tf.constant([0.25, 0.5], dtype=DTYPE))

    with np.testing.assert_raises(ValueError):
        validate_pseudo_time_steps_tf(tf.constant([1.0, 0.0], dtype=DTYPE))


def test_auxiliary_and_actual_paths_replay_same_affine_trace() -> None:
    ancestors = tf.constant([[-0.3], [0.4]], dtype=DTYPE)
    previous_covariances = tf.constant([[[0.2]], [[0.5]]], dtype=DTYPE)
    pre_flow = tf.constant([[-0.24], [0.31]], dtype=DTYPE)
    observation = tf.constant([0.2], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return 0.9 * tf.reshape(points, [-1, 1]) + 0.1

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
        return -0.5 * residual * residual / 0.08

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.04]], dtype=DTYPE)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return values + 0.05 * tf.square(values)

    def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
        value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
        return tf.reshape(1.0 + 0.1 * value, [1, 1])

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.15]], dtype=DTYPE)

    result = li_coates_ledh_alg1_time_step_tf(
        ancestors=ancestors,
        previous_covariances=previous_covariances,
        pre_flow_particles=pre_flow,
        observation=observation,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        time_index=0,
        pseudo_time_steps=tf.constant([0.4, 0.6], dtype=DTYPE),
    )

    for i in range(2):
        auxiliary = result.auxiliary_anchors[i]
        actual = pre_flow[i]
        for j, eps in enumerate(tf.unstack(result.pseudo_time_steps)):
            a_matrix = result.flow_matrices_by_particle_step[i, j]
            b_vector = result.flow_offsets_by_particle_step[i, j]
            auxiliary = auxiliary + eps * (tf.linalg.matvec(a_matrix, auxiliary) + b_vector)
            actual = actual + eps * (tf.linalg.matvec(a_matrix, actual) + b_vector)

        np.testing.assert_allclose(
            auxiliary.numpy(),
            result.auxiliary_terminal_states.numpy()[i],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            actual.numpy(),
            result.post_flow_particles.numpy()[i],
            atol=1e-12,
        )


def test_scalar_determinant_product_matches_manual_ledh_steps() -> None:
    ancestors = tf.constant([[0.2]], dtype=DTYPE)
    previous_covariances = tf.constant([[[0.3]]], dtype=DTYPE)
    pre_flow = tf.constant([[0.25]], dtype=DTYPE)
    observation = tf.constant([0.15], dtype=DTYPE)
    eps = tf.constant([0.25, 0.75], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.identity(tf.reshape(points, [-1, 1]))

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - x_prev, [-1])
        return -0.5 * residual * residual

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.05]], dtype=DTYPE)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return 1.4 * values + 0.1

    def observation_jacobian(_point: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[1.4]], dtype=DTYPE)

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.3]], dtype=DTYPE)

    result = li_coates_ledh_alg1_time_step_tf(
        ancestors=ancestors,
        previous_covariances=previous_covariances,
        pre_flow_particles=pre_flow,
        observation=observation,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        time_index=0,
        pseudo_time_steps=eps,
        covariance_floor=0.0,
    )

    p = float(result.predicted_covariances.numpy()[0, 0, 0])
    h = 1.4
    r = 0.3
    manual = 0.0
    lam = 0.0
    for step in (0.25, 0.75):
        lam += step
        a = -0.5 * p * h * (lam * h * p * h + r) ** -1 * h
        manual += np.log(abs(1.0 + step * a))
    np.testing.assert_allclose(result.forward_log_det.numpy()[0], manual, atol=1e-12)


def test_vectorized_particle_time_step_matches_looped_reference() -> None:
    ancestors = tf.constant([[-0.2], [0.1], [0.4]], dtype=DTYPE)
    previous_covariances = tf.constant([[[0.2]], [[0.25]], [[0.3]]], dtype=DTYPE)
    pre_flow = tf.constant([[-0.16], [0.09], [0.42]], dtype=DTYPE)
    observation = tf.constant([0.15], dtype=DTYPE)
    eps = tf.constant([0.4, 0.6], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return 0.8 * values + 0.05 * tf.square(values)

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
        variance = tf.constant(0.07, dtype=DTYPE)
        return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.07]], dtype=DTYPE)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return values + 0.1 * tf.square(values)

    def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
        value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
        return tf.reshape(1.0 + 0.2 * value, [1, 1])

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.11]], dtype=DTYPE)

    looped = li_coates_ledh_alg1_time_step_tf(
        ancestors=ancestors,
        previous_covariances=previous_covariances,
        pre_flow_particles=pre_flow,
        observation=observation,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        time_index=0,
        pseudo_time_steps=eps,
    )
    vectorized = li_coates_ledh_alg1_time_step_vectorized_particles_tf(
        ancestors=ancestors,
        previous_covariances=previous_covariances,
        pre_flow_particles=pre_flow,
        observation=observation,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        time_index=0,
        pseudo_time_steps=eps,
    )

    np.testing.assert_allclose(
        vectorized.post_flow_particles.numpy(),
        looped.post_flow_particles.numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        vectorized.predicted_covariances.numpy(),
        looped.predicted_covariances.numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        vectorized.updated_covariances.numpy(),
        looped.updated_covariances.numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        vectorized.forward_log_det.numpy(),
        looped.forward_log_det.numpy(),
        atol=1e-12,
    )
    assert vectorized.diagnostics["particle_batch_route"] == "tf_while_loop_fixed_rows"


def test_sv_scalar_graph_matches_looped_reference_for_scalar_linear_fixture() -> None:
    gamma = tf.constant(0.72, dtype=DTYPE)
    beta = tf.constant(1.35, dtype=DTYPE)
    sigma = tf.constant(1.0, dtype=DTYPE)
    observation_variance = tf.constant(2.0, dtype=DTYPE)
    raw_observations = tf.constant([0.8, -1.2, 0.45], dtype=DTYPE)
    flow_observations = (
        tf.math.log(tf.square(raw_observations) + tf.constant(1e-6, dtype=DTYPE))
        - 2.0 * tf.math.log(beta)
    )
    initial_covariance = tf.reshape(tf.square(sigma) / (1.0 - tf.square(gamma)), [1, 1])

    def seed_pair(seed: int, salt: int) -> tf.Tensor:
        return tf.stack((tf.cast(seed % 2147483647, tf.int32), tf.cast(salt % 2147483647, tf.int32)))

    def initial_sample(num_particles: int, seed: int) -> tf.Tensor:
        return tf.sqrt(initial_covariance[0, 0]) * tf.random.stateless_normal(
            [int(num_particles), 1],
            seed=seed_pair(seed, 110),
            dtype=DTYPE,
        )

    def transition_sample(particles: tf.Tensor, seed: int, time_index: int) -> tf.Tensor:
        noise = sigma * tf.random.stateless_normal(
            [int(particles.shape[0]), 1],
            seed=seed_pair(seed, 1110 + time_index),
            dtype=DTYPE,
        )
        return gamma * tf.cast(particles, DTYPE) + noise

    def transition_mean(points: tf.Tensor, _time_index: int) -> tf.Tensor:
        return gamma * tf.cast(points, DTYPE)

    def transition_log_density(next_particles: tf.Tensor, previous_particles: tf.Tensor, _time_index: int) -> tf.Tensor:
        residual = tf.cast(next_particles, DTYPE)[:, 0] - gamma * tf.cast(previous_particles, DTYPE)[:, 0]
        return -0.5 * tf.math.log(2.0 * np.pi * tf.square(sigma)) - 0.5 * tf.square(residual / sigma)

    def observation_mean(points: tf.Tensor, _time_index: int) -> tf.Tensor:
        return tf.cast(points, DTYPE)

    def observation_jacobian(_point: tf.Tensor, _time_index: int) -> tf.Tensor:
        return tf.eye(1, dtype=DTYPE)

    def observation_log_density(points: tf.Tensor, observation: tf.Tensor, _time_index: int) -> tf.Tensor:
        y = tf.reshape(tf.cast(observation, DTYPE), [1])[0]
        log_scale = tf.math.log(beta) + 0.5 * tf.cast(points, DTYPE)[:, 0]
        standardized = y * tf.exp(-log_scale)
        return (
            -0.5 * tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
            - log_scale
            - 0.5 * tf.square(standardized)
        )

    def target_observation_log_density(points: tf.Tensor, _observation: tf.Tensor, time_index: int) -> tf.Tensor:
        return observation_log_density(points, raw_observations[time_index], time_index)

    def process_covariance(_point: tf.Tensor, _time_index: int) -> tf.Tensor:
        return tf.reshape(tf.square(sigma), [1, 1])

    def observation_covariance(_time_index: int) -> tf.Tensor:
        return tf.reshape(observation_variance, [1, 1])

    looped = run_ledh_pfpf_alg1_ukf_tf(
        observations=flow_observations,
        initial_sample=initial_sample,
        initial_covariance=initial_covariance,
        transition_sample=transition_sample,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        observation_log_density_fn=target_observation_log_density,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        seed=81120,
        num_particles=4,
        pseudo_time_steps=tf.constant([1.0], dtype=DTYPE),
        resampling_route="none",
        jit_compile=False,
    )
    graph = run_ledh_pfpf_alg1_scalar_sv_graph_tf(
        flow_observations=flow_observations,
        raw_observations=raw_observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        observation_variance=observation_variance,
        seed=81120,
        num_particles=4,
        pseudo_time_steps=tf.constant([1.0], dtype=DTYPE),
    )

    assert graph.finite
    assert graph.route_identifiers["graph_specialization_route"] == "p8g_scalar_sv_graph"
    assert graph.route_identifiers["time_loop_route"] == "tf_while_loop"
    np.testing.assert_allclose(
        graph.log_likelihood_estimate.numpy(),
        looped.log_likelihood_estimate.numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        graph.filtered_means.numpy(),
        looped.filtered_means.numpy(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        graph.ess_by_time.numpy(),
        looped.ess_by_time.numpy(),
        atol=1e-10,
    )


def test_forward_log_det_matches_autodiff_jacobian_of_actual_map() -> None:
    ancestor = tf.constant([[0.2]], dtype=DTYPE)
    previous_covariances = tf.constant([[[0.3]]], dtype=DTYPE)
    observation = tf.constant([0.15], dtype=DTYPE)
    eps = tf.constant([0.25, 0.75], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.identity(tf.reshape(points, [-1, 1]))

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - x_prev, [-1])
        return -0.5 * residual * residual

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.05]], dtype=DTYPE)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return 1.4 * values + 0.1

    def observation_jacobian(_point: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[1.4]], dtype=DTYPE)

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.3]], dtype=DTYPE)

    pre_flow_scalar = tf.Variable(0.25, dtype=DTYPE)
    with tf.GradientTape() as tape:
        result = li_coates_ledh_alg1_time_step_tf(
            ancestors=ancestor,
            previous_covariances=previous_covariances,
            pre_flow_particles=tf.reshape(pre_flow_scalar, [1, 1]),
            observation=observation,
            transition_mean_fn=transition_mean,
            transition_log_density_fn=transition_log_density,
            observation_mean_fn=observation_mean,
            observation_jacobian_fn=observation_jacobian,
            process_noise_covariance_fn=process_covariance,
            observation_covariance_fn=observation_covariance,
            time_index=0,
            pseudo_time_steps=eps,
            covariance_floor=0.0,
        )
        post_flow_scalar = result.post_flow_particles[0, 0]

    jacobian = tape.gradient(post_flow_scalar, pre_flow_scalar)
    assert jacobian is not None
    np.testing.assert_allclose(
        tf.math.log(tf.abs(jacobian)).numpy(),
        result.forward_log_det.numpy()[0],
        atol=1e-12,
    )


def test_nonlinear_fixture_produces_particle_indexed_covariance_variation() -> None:
    ancestors = tf.constant([[-0.4], [0.2], [0.8]], dtype=DTYPE)
    previous_covariances = tf.fill([3, 1, 1], tf.constant(0.2, dtype=DTYPE))
    pre_flow = ancestors + tf.constant([[0.02], [-0.01], [0.03]], dtype=DTYPE)
    observation = tf.constant([0.3], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return values + 0.15 * tf.square(values)

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
        return -0.5 * residual * residual / 0.05

    def process_covariance(x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        value = tf.reshape(tf.cast(x_prev, DTYPE), [-1])[0]
        return tf.reshape(0.04 + 0.02 * tf.square(value), [1, 1])

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return values + 0.1 * tf.square(values)

    def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
        value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
        return tf.reshape(1.0 + 0.2 * value, [1, 1])

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.12]], dtype=DTYPE)

    result = li_coates_ledh_alg1_time_step_tf(
        ancestors=ancestors,
        previous_covariances=previous_covariances,
        pre_flow_particles=pre_flow,
        observation=observation,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        time_index=0,
        pseudo_time_steps=tf.constant([0.5, 0.5], dtype=DTYPE),
    )

    diagonal = result.predicted_covariances.numpy()[:, 0, 0]
    assert bool(tf.reduce_all(tf.math.is_finite(result.predicted_covariances)).numpy())
    assert float(np.max(diagonal) - np.min(diagonal)) > 1e-4


def test_resampling_gathers_covariances_with_particles() -> None:
    particles = tf.constant([[0.0], [1.0], [2.0]], dtype=DTYPE)
    covariances = tf.reshape(tf.constant([0.1, 0.2, 0.3], dtype=DTYPE), [3, 1, 1])
    indices = tf.constant([2, 0, 2], dtype=tf.int32)

    resampled_particles, resampled_covariances = apply_classical_resampling_state_tf(
        particles=particles,
        covariances=covariances,
        ancestor_indices=indices,
    )

    np.testing.assert_allclose(resampled_particles.numpy()[:, 0], [2.0, 0.0, 2.0])
    np.testing.assert_allclose(resampled_covariances.numpy()[:, 0, 0], [0.3, 0.1, 0.3])


def test_sinkhorn_canonical_transport_and_covariance_carry_share_same_matrix() -> None:
    coupling = tf.constant(
        [
            [0.30, 0.10],
            [0.20, 0.40],
        ],
        dtype=DTYPE,
    )
    particles = tf.constant([[-1.0], [2.0]], dtype=DTYPE)
    covariances = tf.constant([[[0.2]], [[0.8]]], dtype=DTYPE)

    canonical, transport_diag = canonical_transport_from_sinkhorn_coupling_tf(coupling)
    transported = tf.linalg.matmul(canonical, particles)
    carried, covariance_diag = carry_covariances_with_canonical_transport_tf(
        canonical_transport=canonical,
        covariances=covariances,
        covariance_floor=0.0,
    )

    expected = tf.constant(
        [
            [0.30 / 0.50, 0.20 / 0.50],
            [0.10 / 0.50, 0.40 / 0.50],
        ],
        dtype=DTYPE,
    )
    np.testing.assert_allclose(canonical.numpy(), expected.numpy(), atol=1e-12)
    np.testing.assert_allclose(tf.reduce_sum(canonical, axis=1).numpy(), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(transported.numpy(), (expected @ particles).numpy(), atol=1e-12)
    np.testing.assert_allclose(
        carried[:, 0, 0].numpy(),
        tf.linalg.matvec(expected, covariances[:, 0, 0]).numpy(),
        atol=1e-12,
    )
    assert transport_diag["canonical_transport_matrix_convention"].numpy().decode("utf-8") == (
        OT_CANONICAL_TRANSPORT_CONVENTION
    )
    assert bool(covariance_diag["finite_carried_covariances"].numpy())
    assert covariance_diag["covariance_carry_route"].numpy().decode("utf-8") == OT_COVARIANCE_CARRY_ROUTE


def test_apply_ot_resampling_state_tf_emits_sinkhorn_covariance_carry_diagnostics() -> None:
    particles = tf.constant([[-1.0], [0.25], [1.5]], dtype=DTYPE)
    covariances = tf.constant([[[0.2]], [[0.5]], [[0.9]]], dtype=DTYPE)
    weights = tf.constant([0.80, 0.15, 0.05], dtype=DTYPE)
    log_weights = tf.math.log(weights)

    transported, carried, diagnostics = apply_ot_resampling_state_tf(
        particles=particles,
        covariances=covariances,
        weights=weights,
        log_weights=log_weights,
        resampling_route=OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
        covariance_floor=0.0,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=80,
        sinkhorn_tolerance=1e-7,
    )

    assert transported.shape == particles.shape
    assert carried.shape == covariances.shape
    assert diagnostics["resampling_method"] == "fixed_target_sinkhorn"
    assert diagnostics["relaxed_resampling_not_categorical"] is True
    assert diagnostics["canonical_transport_matrix_convention"] == OT_CANONICAL_TRANSPORT_CONVENTION
    assert diagnostics["covariance_carry_route"] == OT_COVARIANCE_CARRY_ROUTE
    assert diagnostics["pfpf_correction_route"] == OT_PFPF_CORRECTION_ROUTE
    assert diagnostics["finite_transport"] is True
    assert diagnostics["finite_particles"] is True
    assert diagnostics["canonical_transport_row_sum_residual"] < 1e-10
    assert diagnostics["canonical_transport_row_sum_tolerance"] == 0.005
    assert diagnostics["canonical_transport_shape"] == [3, 3]


def test_ot_resampling_blocks_malformed_canonical_transport_before_state_carry(
    monkeypatch,
) -> None:
    particles = tf.constant([[-1.0], [0.25], [1.5]], dtype=DTYPE)
    covariances = tf.constant([[[0.2]], [[0.5]], [[0.9]]], dtype=DTYPE)
    weights = tf.constant([0.80, 0.15, 0.05], dtype=DTYPE)

    def malformed_transport(_coupling):
        return (
            tf.constant(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=DTYPE,
            ),
            {},
        )

    monkeypatch.setattr(
        "experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_alg1_ukf_tf.canonical_transport_from_sinkhorn_coupling_tf",
        malformed_transport,
    )
    with np.testing.assert_raises(FloatingPointError):
        apply_ot_resampling_state_tf(
            particles=particles,
            covariances=covariances,
            weights=weights,
            log_weights=tf.math.log(weights),
            resampling_route=OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
            covariance_floor=0.0,
            sinkhorn_epsilon=1.0,
            sinkhorn_iterations=80,
            sinkhorn_tolerance=1e-7,
        )


def test_route_identifier_rejects_old_ledh_pfpf_ot_claim() -> None:
    route = algorithm1_route_identifiers(resampling_route="none")
    assert route["method_generation"] == METHOD_GENERATION
    validate_algorithm1_route_identifiers(route)
    ot_route = algorithm1_route_identifiers(
        resampling_route=OT_SINKHORN_COVARIANCE_CARRY_ROUTE
    )
    assert ot_route["route_variant"] == "p8h_sv_scalar_graph_ot_resampled_alg1"
    assert ot_route["covariance_carry_route"] == OT_COVARIANCE_CARRY_ROUTE
    assert ot_route["pfpf_correction_route"] == OT_PFPF_CORRECTION_ROUTE
    assert ot_route["canonical_transport_matrix_convention"] == OT_CANONICAL_TRANSPORT_CONVENTION
    assert ot_route["p8g_no_resampling_evidence_status"] == (
        "quarantined_historical_diagnostic_only"
    )
    validate_algorithm1_route_identifiers(ot_route)

    bad = dict(route)
    bad["flow_source_route"] = "dpf_ledh_pfpf_ot"
    with np.testing.assert_raises(ValueError):
        validate_algorithm1_route_identifiers(bad)


def test_algorithm1_module_does_not_import_numpy_or_old_ledh_pfpf_ot_route() -> None:
    text = (
        ROOT
        / "experiments"
        / "dpf_implementation"
        / "tf_tfp"
        / "filters"
        / "ledh_pfpf_alg1_ukf_tf.py"
    ).read_text(encoding="utf-8")

    assert "import numpy" not in text
    assert "from numpy" not in text
    assert "ledh_pfpf_ot_tf" not in text
    assert "run_ledh_pfpf_ot_tf" not in text


def test_small_filter_run_is_finite_and_records_route_identifiers() -> None:
    observations = tf.constant([[0.1], [0.2]], dtype=DTYPE)
    initial_covariance = tf.constant([[0.3]], dtype=DTYPE)

    def initial_sample(num_particles: int, _seed: int) -> tf.Tensor:
        return tf.reshape(tf.linspace(tf.constant(-0.2, DTYPE), tf.constant(0.2, DTYPE), num_particles), [-1, 1])

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return 0.8 * tf.reshape(points, [-1, 1])

    def transition_sample(ancestors: tf.Tensor, _seed: int, t: int) -> tf.Tensor:
        offsets = tf.reshape(tf.cast(tf.range(tf.shape(ancestors)[0]), DTYPE), [-1, 1])
        centered = offsets - tf.reduce_mean(offsets)
        return transition_mean(ancestors, t) + 0.02 * centered

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
        variance = tf.constant(0.04, DTYPE)
        return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return tf.square(values) + values

    def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
        value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
        return tf.reshape(2.0 * value + 1.0, [1, 1])

    def observation_log_density(x: tf.Tensor, y: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(y, [1])[0] - tf.reshape(observation_mean(x, t), [-1])
        variance = tf.constant(0.05, DTYPE)
        return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.04]], dtype=DTYPE)

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.05]], dtype=DTYPE)

    result = run_ledh_pfpf_alg1_ukf_tf(
        observations=observations,
        initial_sample=initial_sample,
        initial_covariance=initial_covariance,
        transition_sample=transition_sample,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        observation_log_density_fn=observation_log_density,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        seed=11,
        num_particles=5,
        pseudo_time_steps=tf.constant([0.5, 0.5], dtype=DTYPE),
    )

    assert result.finite
    assert result.route_identifiers["method_generation"] == METHOD_GENERATION
    assert result.particle_covariances_by_time.shape == (2, 5, 1, 1)
    assert result.resampling_diagnostics[0]["flow_source_route"] == "li_coates_2017_algorithm1_ledh_pfpf"


def test_small_filter_run_triggers_p8h_sinkhorn_ot_covariance_carry() -> None:
    observations = tf.constant([[0.1], [0.2]], dtype=DTYPE)
    initial_covariance = tf.constant([[0.3]], dtype=DTYPE)

    def initial_sample(num_particles: int, _seed: int) -> tf.Tensor:
        return tf.reshape(tf.linspace(tf.constant(-0.2, DTYPE), tf.constant(0.2, DTYPE), num_particles), [-1, 1])

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return 0.8 * tf.reshape(points, [-1, 1])

    def transition_sample(ancestors: tf.Tensor, _seed: int, t: int) -> tf.Tensor:
        offsets = tf.reshape(tf.cast(tf.range(tf.shape(ancestors)[0]), DTYPE), [-1, 1])
        centered = offsets - tf.reduce_mean(offsets)
        return transition_mean(ancestors, t) + 0.02 * centered

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
        variance = tf.constant(0.04, DTYPE)
        return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return tf.square(values) + values

    def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
        value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
        return tf.reshape(2.0 * value + 1.0, [1, 1])

    def observation_log_density(x: tf.Tensor, y: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(y, [1])[0] - tf.reshape(observation_mean(x, t), [-1])
        variance = tf.constant(0.05, DTYPE)
        return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.04]], dtype=DTYPE)

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.05]], dtype=DTYPE)

    result = run_ledh_pfpf_alg1_ukf_tf(
        observations=observations,
        initial_sample=initial_sample,
        initial_covariance=initial_covariance,
        transition_sample=transition_sample,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        observation_log_density_fn=observation_log_density,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        seed=11,
        num_particles=5,
        pseudo_time_steps=tf.constant([0.5, 0.5], dtype=DTYPE),
        resampling_route=OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
        ess_threshold_ratio=1.01,
        sinkhorn_epsilon=1.0,
        sinkhorn_iterations=80,
        sinkhorn_tolerance=1e-7,
    )

    assert result.finite
    assert result.resampling_count == 2
    assert result.route_identifiers["resampling_route"] == OT_SINKHORN_COVARIANCE_CARRY_ROUTE
    assert result.route_identifiers["route_variant"] == "p8h_sv_scalar_graph_ot_resampled_alg1"
    first_diag = result.resampling_diagnostics[0]
    assert first_diag["resampled"] is True
    assert first_diag["resampling_method"] == "fixed_target_sinkhorn"
    assert first_diag["covariance_carry_route"] == OT_COVARIANCE_CARRY_ROUTE
    assert first_diag["pfpf_correction_route"] == OT_PFPF_CORRECTION_ROUTE
    assert first_diag["canonical_transport_matrix_convention"] == OT_CANONICAL_TRANSPORT_CONVENTION
    assert first_diag["relaxed_resampling_not_categorical"] is True
    assert first_diag["finite_carried_covariances"] is True
    assert first_diag["canonical_transport_shape"] == [5, 5]
    assert first_diag["ess_triggered"] is True
    assert result.particle_covariances_by_time.shape == (2, 5, 1, 1)


def test_corrected_log_weight_matches_manual_pfpf_formula() -> None:
    observations = tf.constant([[0.18]], dtype=DTYPE)
    initial_covariance = tf.constant([[0.25]], dtype=DTYPE)
    pseudo_time_steps = tf.constant([0.4, 0.6], dtype=DTYPE)

    def initial_sample(num_particles: int, _seed: int) -> tf.Tensor:
        assert num_particles == 2
        return tf.constant([[-0.2], [0.25]], dtype=DTYPE)

    def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        return 0.85 * tf.reshape(points, [-1, 1]) + 0.05

    def transition_sample(ancestors: tf.Tensor, _seed: int, t: int) -> tf.Tensor:
        del t
        offsets = tf.constant([[0.03], [-0.02]], dtype=DTYPE)
        return transition_mean(ancestors, 0) + offsets

    def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
        variance = tf.constant(0.07, dtype=DTYPE)
        return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

    def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
        values = tf.reshape(points, [-1, 1])
        return values + 0.1 * tf.square(values)

    def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
        value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
        return tf.reshape(1.0 + 0.2 * value, [1, 1])

    def observation_log_density(x: tf.Tensor, y: tf.Tensor, t: int) -> tf.Tensor:
        residual = tf.reshape(y, [1])[0] - tf.reshape(observation_mean(x, t), [-1])
        variance = tf.constant(0.11, dtype=DTYPE)
        return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

    def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
        return tf.constant([[0.07]], dtype=DTYPE)

    def observation_covariance(_t: int) -> tf.Tensor:
        return tf.constant([[0.11]], dtype=DTYPE)

    run_result = run_ledh_pfpf_alg1_ukf_tf(
        observations=observations,
        initial_sample=initial_sample,
        initial_covariance=initial_covariance,
        transition_sample=transition_sample,
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        observation_log_density_fn=observation_log_density,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        seed=17,
        num_particles=2,
        pseudo_time_steps=pseudo_time_steps,
        resampling_route="none",
    )

    ancestors = initial_sample(2, 17)
    pre_flow = transition_sample(ancestors, 17, 0)
    step = li_coates_ledh_alg1_time_step_tf(
        ancestors=ancestors,
        previous_covariances=tf.tile(initial_covariance[tf.newaxis, :, :], [2, 1, 1]),
        pre_flow_particles=pre_flow,
        observation=observations[0],
        transition_mean_fn=transition_mean,
        transition_log_density_fn=transition_log_density,
        observation_mean_fn=observation_mean,
        observation_jacobian_fn=observation_jacobian,
        process_noise_covariance_fn=process_covariance,
        observation_covariance_fn=observation_covariance,
        time_index=0,
        pseudo_time_steps=pseudo_time_steps,
    )
    previous_log_weights = tf.fill([2], -tf.math.log(tf.constant(2.0, dtype=DTYPE)))
    manual = (
        previous_log_weights
        + transition_log_density(step.post_flow_particles, ancestors, 0)
        + observation_log_density(step.post_flow_particles, observations[0], 0)
        - step.pre_flow_log_density
        + step.forward_log_det
    )

    np.testing.assert_allclose(
        run_result.corrected_log_weights_by_time.numpy()[0],
        manual.numpy(),
        atol=1e-12,
    )


def test_fixed_branch_gradient_smoke_for_no_resampling_path() -> None:
    observations = tf.constant([[0.1], [0.2]], dtype=DTYPE)
    initial_covariance = tf.constant([[0.3]], dtype=DTYPE)

    def value_for_scale(scale: tf.Tensor) -> tf.Tensor:
        def initial_sample(num_particles: int, _seed: int) -> tf.Tensor:
            base = tf.linspace(tf.constant(-0.2, DTYPE), tf.constant(0.2, DTYPE), num_particles)
            return tf.reshape(base, [-1, 1])

        def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
            return scale * tf.reshape(points, [-1, 1])

        def transition_sample(ancestors: tf.Tensor, _seed: int, t: int) -> tf.Tensor:
            offsets = tf.reshape(tf.cast(tf.range(tf.shape(ancestors)[0]), DTYPE), [-1, 1])
            centered = offsets - tf.reduce_mean(offsets)
            return transition_mean(ancestors, t) + 0.02 * centered

        def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
            residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
            variance = tf.constant(0.04, DTYPE)
            return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

        def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
            values = tf.reshape(points, [-1, 1])
            return tf.square(values) + values

        def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
            value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
            return tf.reshape(2.0 * value + 1.0, [1, 1])

        def observation_log_density(x: tf.Tensor, y: tf.Tensor, t: int) -> tf.Tensor:
            residual = tf.reshape(y, [1])[0] - tf.reshape(observation_mean(x, t), [-1])
            variance = tf.constant(0.05, DTYPE)
            return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

        def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
            return tf.constant([[0.04]], dtype=DTYPE)

        def observation_covariance(_t: int) -> tf.Tensor:
            return tf.constant([[0.05]], dtype=DTYPE)

        result = run_ledh_pfpf_alg1_ukf_tf(
            observations=observations,
            initial_sample=initial_sample,
            initial_covariance=initial_covariance,
            transition_sample=transition_sample,
            transition_mean_fn=transition_mean,
            transition_log_density_fn=transition_log_density,
            observation_mean_fn=observation_mean,
            observation_jacobian_fn=observation_jacobian,
            observation_log_density_fn=observation_log_density,
            process_noise_covariance_fn=process_covariance,
            observation_covariance_fn=observation_covariance,
            seed=13,
            num_particles=5,
            pseudo_time_steps=tf.constant([0.5, 0.5], dtype=DTYPE),
            resampling_route="none",
        )
        return result.log_likelihood_estimate

    scale = tf.Variable(0.8, dtype=DTYPE)
    with tf.GradientTape() as tape:
        value = value_for_scale(scale)
    gradient = tape.gradient(value, scale)

    assert gradient is not None
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.math.is_finite(gradient).numpy())


def test_p8h_sinkhorn_ot_path_has_connected_gradient_smoke() -> None:
    observations = tf.constant([[0.1], [0.2]], dtype=DTYPE)
    initial_covariance = tf.constant([[0.3]], dtype=DTYPE)

    def value_for_scale(scale: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
        def initial_sample(num_particles: int, _seed: int) -> tf.Tensor:
            base = tf.linspace(tf.constant(-0.2, DTYPE), tf.constant(0.2, DTYPE), num_particles)
            return tf.reshape(base, [-1, 1])

        def transition_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
            return scale * tf.reshape(points, [-1, 1])

        def transition_sample(ancestors: tf.Tensor, _seed: int, t: int) -> tf.Tensor:
            offsets = tf.reshape(tf.cast(tf.range(tf.shape(ancestors)[0]), DTYPE), [-1, 1])
            centered = offsets - tf.reduce_mean(offsets)
            return transition_mean(ancestors, t) + 0.02 * centered

        def transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, t: int) -> tf.Tensor:
            residual = tf.reshape(x_next - transition_mean(x_prev, t), [-1])
            variance = tf.constant(0.04, DTYPE)
            return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

        def observation_mean(points: tf.Tensor, _t: int) -> tf.Tensor:
            values = tf.reshape(points, [-1, 1])
            return tf.square(values) + values

        def observation_jacobian(point: tf.Tensor, _t: int) -> tf.Tensor:
            value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
            return tf.reshape(2.0 * value + 1.0, [1, 1])

        def observation_log_density(x: tf.Tensor, y: tf.Tensor, t: int) -> tf.Tensor:
            residual = tf.reshape(y, [1])[0] - tf.reshape(observation_mean(x, t), [-1])
            variance = tf.constant(0.05, DTYPE)
            return -0.5 * (tf.math.log(2.0 * np.pi * variance) + residual * residual / variance)

        def process_covariance(_x_prev: tf.Tensor, _t: int) -> tf.Tensor:
            return tf.constant([[0.04]], dtype=DTYPE)

        def observation_covariance(_t: int) -> tf.Tensor:
            return tf.constant([[0.05]], dtype=DTYPE)

        result = run_ledh_pfpf_alg1_ukf_tf(
            observations=observations,
            initial_sample=initial_sample,
            initial_covariance=initial_covariance,
            transition_sample=transition_sample,
            transition_mean_fn=transition_mean,
            transition_log_density_fn=transition_log_density,
            observation_mean_fn=observation_mean,
            observation_jacobian_fn=observation_jacobian,
            observation_log_density_fn=observation_log_density,
            process_noise_covariance_fn=process_covariance,
            observation_covariance_fn=observation_covariance,
            seed=13,
            num_particles=5,
            pseudo_time_steps=tf.constant([0.5, 0.5], dtype=DTYPE),
            resampling_route=OT_SINKHORN_COVARIANCE_CARRY_ROUTE,
            ess_threshold_ratio=1.01,
            sinkhorn_epsilon=1.0,
            sinkhorn_iterations=80,
            sinkhorn_tolerance=1e-7,
        )
        assert result.resampling_count == 2
        assert result.route_identifiers["resampling_route"] == OT_SINKHORN_COVARIANCE_CARRY_ROUTE
        return result.log_likelihood_estimate

    scale = tf.Variable(0.8, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(observations)
        value = value_for_scale(scale, observations)
    gradient, observation_gradient = tape.gradient(value, (scale, observations))

    assert gradient is not None
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.math.is_finite(gradient).numpy())
    assert abs(float(gradient.numpy())) > 0.0
    assert observation_gradient is not None
    direction = tf.constant([[.4], [-.6]], DTYPE)
    step = tf.constant(1e-5, DTYPE)
    upper = value_for_scale(scale, observations + step * direction)
    lower = value_for_scale(scale, observations - step * direction)
    np.testing.assert_allclose(tf.reduce_sum(observation_gradient * direction),
                               (upper - lower) / (2. * step), rtol=1e-6, atol=1e-7)
