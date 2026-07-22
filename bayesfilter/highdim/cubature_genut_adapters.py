"""Candidate-only nonlinear adapters with explicit target equations."""

from __future__ import annotations

import math

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_filter import CandidateModelAdapter


def exact_transformed_sv_candidate_adapter(*, sigma: float = 1.0) -> CandidateModelAdapter:
    """Build the exact transformed-SV adapter used by the pilot scope."""

    sigma_tensor = tf.constant(float(sigma), tf.float32)
    normalizer = tf.constant(1.0 / math.sqrt(2.0 * math.pi), tf.float32)
    log_two_pi = tf.constant(math.log(2.0 * math.pi), tf.float32)

    def physical(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        gamma = 0.5 * (1.0 + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2.0, tf.float32))))
        dgamma = normalizer * tf.exp(-0.5 * tf.square(theta[0]))
        beta = tf.exp(theta[1])
        return gamma, dgamma, beta

    def initial_value(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        gamma, _, _ = physical(theta)
        std = sigma_tensor / tf.sqrt(1.0 - tf.square(gamma))
        return noise * std

    def initial_tangent(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        gamma, dgamma, _ = physical(theta)
        denominator = 1.0 - tf.square(gamma)
        dstd = sigma_tensor * gamma * dgamma / tf.pow(denominator, 1.5)
        first = noise[:, 0] * dstd
        return tf.stack([first, tf.zeros_like(first)], axis=-1)[:, None, :]

    def transition_value(theta: tf.Tensor, particles: tf.Tensor, noise: tf.Tensor, _time: tf.Tensor) -> tf.Tensor:
        gamma, _, _ = physical(theta)
        return gamma * particles + sigma_tensor * noise

    def transition_tangent(
        theta: tf.Tensor,
        particles: tf.Tensor,
        noise: tf.Tensor,
        particle_tangent: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        gamma, dgamma, _ = physical(theta)
        first = dgamma * particles[:, 0] + gamma * particle_tangent[:, 0, 0]
        second = gamma * particle_tangent[:, 0, 1]
        return tf.stack([first, second], axis=-1)[:, None, :]

    def observation_value(
        theta: tf.Tensor,
        particles: tf.Tensor,
        observation: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        _, _, beta = physical(theta)
        residual = observation[0] - 2.0 * tf.math.log(beta) - particles[:, 0]
        return 0.5 * residual - 0.5 * tf.exp(residual) - 0.5 * log_two_pi

    def observation_tangent(
        theta: tf.Tensor,
        particles: tf.Tensor,
        particle_tangent: tf.Tensor,
        observation: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        _, _, _ = physical(theta)
        residual = observation[0] - 2.0 * theta[1] - particles[:, 0]
        residual_tangent = tf.stack(
            [-particle_tangent[:, 0, 0], -2.0 - particle_tangent[:, 0, 1]], axis=-1
        )
        return (0.5 - 0.5 * tf.exp(residual))[:, None] * residual_tangent

    return CandidateModelAdapter(
        state_dimension=1,
        parameter_count=2,
        initial_value=initial_value,
        initial_tangent=initial_tangent,
        transition_value=transition_value,
        transition_tangent=transition_tangent,
        observation_value=observation_value,
        observation_tangent=observation_tangent,
    )


def diagonal_lgssm_candidate_adapter(
    *, observation_matrix: tf.Tensor
) -> CandidateModelAdapter:
    """Build the five-parameter stationary diagonal LGSSM adapter.

    The parameter order is ``(phi_1, phi_2, phi_3, q_scale, r_scale)``.
    This adapter exists so LGSSM diagnostics use the same batch-native reset
    and recursive forward-sensitivity implementation as nonlinear candidates.
    """

    matrix = tf.convert_to_tensor(observation_matrix, tf.float32)
    if matrix.shape != (3, 3):
        raise ValueError("observation_matrix must have shape (3, 3)")
    parameter_eye = tf.eye(5, dtype=tf.float32)
    phi_tangent = parameter_eye[:3]
    q_tangent = parameter_eye[3]
    r_tangent = parameter_eye[4]
    log_two_pi = tf.constant(math.log(2.0 * math.pi), tf.float32)

    def physical(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return theta[:3], theta[3], theta[4]

    def initial_value(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        phi, q_scale, _ = physical(theta)
        standard_deviation = q_scale / tf.sqrt(1.0 - tf.square(phi))
        return noise * standard_deviation[None, :]

    def initial_tangent(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        phi, q_scale, _ = physical(theta)
        denominator = 1.0 - tf.square(phi)
        standard_deviation_tangent = (
            q_scale
            * phi[:, None]
            * phi_tangent
            / tf.pow(denominator, 1.5)[:, None]
            + q_tangent[None, :] / tf.sqrt(denominator)[:, None]
        )
        return noise[:, :, None] * standard_deviation_tangent[None, :, :]

    def transition_value(
        theta: tf.Tensor,
        particles: tf.Tensor,
        noise: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        phi, q_scale, _ = physical(theta)
        return particles * phi[None, :] + q_scale * noise

    def transition_tangent(
        theta: tf.Tensor,
        particles: tf.Tensor,
        noise: tf.Tensor,
        tangent: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        phi, _, _ = physical(theta)
        return (
            tangent * phi[None, :, None]
            + particles[:, :, None] * phi_tangent[None, :, :]
            + noise[:, :, None] * q_tangent[None, None, :]
        )

    def observation_value(
        theta: tf.Tensor,
        particles: tf.Tensor,
        observation: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        _, _, r_scale = physical(theta)
        residual = observation[None, :] - particles @ tf.transpose(matrix)
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=1) / tf.square(r_scale)
            + tf.cast(3, tf.float32) * 2.0 * tf.math.log(r_scale)
            + tf.cast(3, tf.float32) * log_two_pi
        )

    def observation_tangent(
        theta: tf.Tensor,
        particles: tf.Tensor,
        tangent: tf.Tensor,
        observation: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        _, _, r_scale = physical(theta)
        residual = observation[None, :] - particles @ tf.transpose(matrix)
        prediction_tangent = tf.einsum("od,ndp->nop", matrix, tangent)
        residual_tangent = -prediction_tangent
        squared_norm = tf.reduce_sum(tf.square(residual), axis=1)
        result = -tf.reduce_sum(
            residual[:, :, None] * residual_tangent, axis=1
        ) / tf.square(r_scale)
        return result + (
            squared_norm / tf.pow(r_scale, 3)
            - tf.cast(3, tf.float32) / r_scale
        )[:, None] * r_tangent[None, :]

    return CandidateModelAdapter(
        state_dimension=3,
        parameter_count=5,
        initial_value=initial_value,
        initial_tangent=initial_tangent,
        transition_value=transition_value,
        transition_tangent=transition_tangent,
        observation_value=observation_value,
        observation_tangent=observation_tangent,
    )


def reduced_sir_candidate_adapter(
    *, transition_before_first_observation: bool = True
) -> CandidateModelAdapter:
    """Build the two-state continuous preclip SIR feasibility adapter.

    The state is ``(S, I)`` and the parameter order is
    ``(log_kappa_scale, log_nu_scale, log_obs_noise_scale)``.  This is an
    explicit reduced diagnostic target, not the clipped Austria leaderboard
    measure.  The RK4 state and parameter tangents are written directly so the
    XLA candidate path does not call autodiff or Python sample loops.
    """

    base_kappa = tf.constant(0.1, tf.float32)
    base_nu = tf.constant(1.0, tf.float32)
    base_obs_variance = tf.constant(0.16, tf.float32)
    initial_mean = tf.constant([0.3, 0.2], tf.float32)
    initial_chol = tf.constant([[0.5, 0.0], [0.0, 0.4]], tf.float32)
    process_chol = tf.constant([[0.5, 0.0], [0.0, 0.4]], tf.float32)
    log_two_pi = tf.constant(math.log(2.0 * math.pi), tf.float32)
    parameter_eye = tf.eye(3, dtype=tf.float32)

    def physical(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            base_kappa * tf.exp(theta[0]),
            base_nu * tf.exp(theta[1]),
            base_obs_variance * tf.exp(2.0 * theta[2]),
        )

    def rhs_with_tangent(
        theta: tf.Tensor, state: tf.Tensor, tangent: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        kappa, nu, _ = physical(theta)
        susceptible = state[:, 0]
        infectious = state[:, 1]
        force = kappa * susceptible * infectious
        rhs = tf.stack([-force, force - nu * infectious], axis=1)
        jacobian = tf.stack(
            [
                tf.stack([-kappa * infectious, -kappa * susceptible], axis=1),
                tf.stack([kappa * infectious, kappa * susceptible - nu], axis=1),
            ],
            axis=1,
        )
        direct = tf.stack(
            [
                tf.stack([-force, tf.zeros_like(force), tf.zeros_like(force)], axis=1),
                tf.stack([force, -nu * infectious, tf.zeros_like(force)], axis=1),
            ],
            axis=1,
        )
        tangent_rhs = tf.einsum("nij,njp->nip", jacobian, tangent) + direct
        return rhs, tangent_rhs

    def rk4_with_tangent(
        theta: tf.Tensor, state: tf.Tensor, tangent: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        step = tf.constant(0.005, tf.float32)
        k1, d1 = rhs_with_tangent(theta, state, tangent)
        state2 = state + 0.5 * step * k1
        tangent2 = tangent + 0.5 * step * d1
        k2, d2 = rhs_with_tangent(theta, state2, tangent2)
        state3 = state + 0.5 * step * k2
        tangent3 = tangent + 0.5 * step * d2
        k3, d3 = rhs_with_tangent(theta, state3, tangent3)
        state4 = state + step * k3
        tangent4 = tangent + step * d3
        k4, d4 = rhs_with_tangent(theta, state4, tangent4)
        return (
            state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            tangent + (step / 6.0) * (d1 + 2.0 * d2 + 2.0 * d3 + d4),
        )

    def initial_value(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        del theta
        return initial_mean[None, :] + tf.linalg.matmul(noise, initial_chol, transpose_b=True)

    def initial_tangent(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        del theta
        return tf.zeros([tf.shape(noise)[0], 2, 3], tf.float32)

    def transition_value(
        theta: tf.Tensor, particles: tf.Tensor, noise: tf.Tensor, _time: tf.Tensor
    ) -> tf.Tensor:
        if not transition_before_first_observation:
            return tf.cond(
                tf.equal(_time, 0),
                lambda: particles,
                lambda: transition_value_after_initial(theta, particles, noise, _time),
            )
        return transition_value_after_initial(theta, particles, noise, _time)

    def transition_value_after_initial(
        theta: tf.Tensor, particles: tf.Tensor, noise: tf.Tensor, _time: tf.Tensor
    ) -> tf.Tensor:
        susceptible = tf.maximum(particles[:, 0], tf.constant(0.0, tf.float32))
        physical = tf.stack([susceptible, particles[:, 1]], axis=1)
        clip_threshold = 0 if transition_before_first_observation else 1
        previous = tf.where(_time > clip_threshold, physical, particles)
        tangent = tf.zeros([tf.shape(particles)[0], 2, 3], tf.float32)
        mean, _ = rk4_with_tangent(theta, previous, tangent)
        return mean + tf.linalg.matmul(noise, process_chol, transpose_b=True)

    def transition_tangent(
        theta: tf.Tensor,
        particles: tf.Tensor,
        noise: tf.Tensor,
        particle_tangent: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        if not transition_before_first_observation:
            return tf.cond(
                tf.equal(_time, 0),
                lambda: particle_tangent,
                lambda: transition_tangent_after_initial(
                    theta, particles, noise, particle_tangent, _time
                ),
            )
        return transition_tangent_after_initial(
            theta, particles, noise, particle_tangent, _time
        )

    def transition_tangent_after_initial(
        theta: tf.Tensor,
        particles: tf.Tensor,
        noise: tf.Tensor,
        particle_tangent: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        susceptible = tf.maximum(particles[:, 0], tf.constant(0.0, tf.float32))
        physical = tf.stack([susceptible, particles[:, 1]], axis=1)
        active = tf.cast(particles[:, 0] > 0.0, tf.float32)
        physical_tangent = tf.concat(
            [particle_tangent[:, 0:1, :] * active[:, None, None], particle_tangent[:, 1:2, :]],
            axis=1,
        )
        clip_threshold = 0 if transition_before_first_observation else 1
        previous = tf.where(_time > clip_threshold, physical, particles)
        previous_tangent = tf.where(
            _time > clip_threshold, physical_tangent, particle_tangent
        )
        del noise
        _, tangent = rk4_with_tangent(theta, previous, previous_tangent)
        return tangent

    def observation_value(
        theta: tf.Tensor,
        particles: tf.Tensor,
        observation: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        del _time
        _, _, variance = physical(theta)
        residual = observation[1] - particles[:, 1]
        return -0.5 * (
            tf.square(residual) / variance + tf.math.log(variance) + log_two_pi
        )

    def observation_tangent(
        theta: tf.Tensor,
        particles: tf.Tensor,
        particle_tangent: tf.Tensor,
        observation: tf.Tensor,
        _time: tf.Tensor,
    ) -> tf.Tensor:
        del _time
        _, _, variance = physical(theta)
        residual = observation[1] - particles[:, 1]
        state_part = residual[:, None] * particle_tangent[:, 1, :] / variance
        variance_part = (tf.square(residual) / variance - 1.0)[:, None] * parameter_eye[2][None, :]
        return state_part + variance_part

    return CandidateModelAdapter(
        state_dimension=2,
        parameter_count=3,
        initial_value=initial_value,
        initial_tangent=initial_tangent,
        transition_value=transition_value,
        transition_tangent=transition_tangent,
        observation_value=observation_value,
        observation_tangent=observation_tangent,
    )


def predator_prey_candidate_adapter() -> CandidateModelAdapter:
    """Build the six-parameter additive-Gaussian predator-prey adapter."""

    initial_mean = tf.constant([50.0, 5.0], tf.float32)
    initial_chol = tf.eye(2, dtype=tf.float32)
    process_chol = tf.constant([[2.0, 0.0], [0.0, 2.0]], tf.float32)
    observation_variance = tf.constant(4.0, tf.float32)
    parameter_eye = tf.eye(6, dtype=tf.float32)
    initial_parameter_values = tf.constant(
        [0.6, 114.0, 25.0, 0.3, 0.5, 0.5], tf.float32
    )

    def rhs_tangent(
        theta: tf.Tensor, state: tf.Tensor, tangent: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        r, capacity, half_sat, s_rate, u_rate, v_rate = tf.unstack(theta)
        prey = state[:, 0]
        predator = state[:, 1]
        denominator = half_sat + prey
        interaction = prey * predator / denominator
        d_prey = r * prey * (1.0 - prey / capacity) - s_rate * interaction
        d_predator = u_rate * interaction - v_rate * predator
        d_state = tf.stack([tangent[:, 0, :], tangent[:, 1, :]], axis=1)
        d_pre, d_pred = d_state[:, 0, :], d_state[:, 1, :]
        d_interaction_state = (
            predator[:, None] * half_sat / tf.square(denominator)[:, None] * d_pre
            + prey[:, None] / denominator[:, None] * d_pred
        )
        d_interaction_direct = (
            -prey * predator / tf.square(denominator)
        )[:, None] * parameter_eye[2][None, :]
        d_interaction = d_interaction_state + d_interaction_direct
        logistic = prey * (1.0 - prey / capacity)
        d_logistic = (
            (1.0 - 2.0 * prey / capacity)[:, None] * d_pre
            + (tf.square(prey) / tf.square(capacity))[:, None] * parameter_eye[1][None, :]
        )
        direct_prey = logistic[:, None] * parameter_eye[0][None, :]
        direct_pred = interaction[:, None] * parameter_eye[4][None, :]
        d_rhs_prey = direct_prey + r * d_logistic - interaction[:, None] * parameter_eye[3][None, :] - s_rate * d_interaction
        d_rhs_predator = direct_pred + u_rate * d_interaction - predator[:, None] * parameter_eye[5][None, :] - v_rate * d_pred
        return (
            tf.stack([d_prey, d_predator], axis=1),
            tf.stack([d_rhs_prey, d_rhs_predator], axis=1),
        )

    def rk4(theta: tf.Tensor, state: tf.Tensor, tangent: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        step = tf.constant(0.1, tf.float32)
        current, current_tangent = state, tangent
        def body(index, current, current_tangent):
            k1, d1 = rhs_tangent(theta, current, current_tangent)
            k2, d2 = rhs_tangent(theta, current + 0.5 * step * k1, current_tangent + 0.5 * step * d1)
            k3, d3 = rhs_tangent(theta, current + 0.5 * step * k2, current_tangent + 0.5 * step * d2)
            k4, d4 = rhs_tangent(theta, current + step * k3, current_tangent + step * d3)
            return (
                index + 1,
                current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
                current_tangent + step / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4),
            )
        _, current, current_tangent = tf.while_loop(
            lambda index, *_: index < 20,
            body,
            (tf.zeros([], tf.int32), current, current_tangent),
            maximum_iterations=20,
        )
        return current, current_tangent

    def initial_value(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        del theta
        return initial_mean[None, :] + tf.linalg.matmul(noise, initial_chol, transpose_b=True)

    def initial_tangent(theta: tf.Tensor, noise: tf.Tensor) -> tf.Tensor:
        del theta
        return tf.zeros([tf.shape(noise)[0], 2, 6], tf.float32)

    def transition_value(theta, particles, noise, _time):
        tangent = tf.zeros([tf.shape(particles)[0], 2, 6], tf.float32)
        mean, _ = rk4(theta, particles, tangent)
        return mean + tf.linalg.matmul(noise, process_chol, transpose_b=True)

    def transition_tangent(theta, particles, noise, particle_tangent, _time):
        del noise
        _, tangent = rk4(theta, particles, particle_tangent)
        return tangent

    def observation_value(theta, particles, observation, _time):
        del theta, _time
        residual = observation[None, :] - particles
        return -0.5 * (tf.reduce_sum(tf.square(residual), axis=1) / observation_variance + 2.0 * tf.math.log(observation_variance) + 2.0 * tf.math.log(tf.constant(2.0 * math.pi, tf.float32)))

    def observation_tangent(theta, particles, tangent, observation, _time):
        del theta, _time
        residual = observation[None, :] - particles
        return tf.reduce_sum(residual[:, :, None] * tangent, axis=1) / observation_variance

    return CandidateModelAdapter(
        state_dimension=2,
        parameter_count=6,
        initial_value=initial_value,
        initial_tangent=initial_tangent,
        transition_value=transition_value,
        transition_tangent=transition_tangent,
        observation_value=observation_value,
        observation_tangent=observation_tangent,
    )


__all__ = [
    "diagonal_lgssm_candidate_adapter",
    "exact_transformed_sv_candidate_adapter",
    "reduced_sir_candidate_adapter",
    "predator_prey_candidate_adapter",
]
