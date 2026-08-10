"""Batch-native model equations for the finite GenUT NeuTra route."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_batch_tf import BatchCandidateModelAdapter


_NORMALIZER = 0.398942280401432677939946059934
_LOG_TWO_PI = 1.837877066409345483560659472811


def diagonal_lgssm_batch_adapter(
    *, observation_matrix: tf.Tensor
) -> BatchCandidateModelAdapter:
    """Five-parameter stationary diagonal LGSSM in physical coordinates."""

    matrix = tf.convert_to_tensor(observation_matrix, tf.float32)
    if matrix.shape != (3, 3):
        raise ValueError("observation_matrix must have shape (3, 3)")
    eye = tf.eye(5, dtype=tf.float32)

    def initial_value(theta, noise):
        phi, q_scale = theta[:, :3], theta[:, 3]
        scale = q_scale[:, None] / tf.sqrt(1.0 - tf.square(phi))
        return noise[None, :, :] * scale[:, None, :]

    def initial_tangent(theta, noise):
        phi, q_scale = theta[:, :3], theta[:, 3]
        denominator = 1.0 - tf.square(phi)
        scale_tangent = (
            q_scale[:, None, None]
            * phi[:, :, None]
            * eye[None, :3, :]
            / tf.pow(denominator, 1.5)[:, :, None]
            + eye[None, 3, None, :] / tf.sqrt(denominator)[:, :, None]
        )
        return noise[None, :, :, None] * scale_tangent[:, None, :, :]

    def transition_value(theta, particles, noise, _time):
        return (
            particles * theta[:, None, :3]
            + theta[:, None, 3:4] * noise[None, :, :]
        )

    def transition_tangent(theta, particles, noise, tangent, _time):
        return (
            tangent * theta[:, None, :3, None]
            + particles[:, :, :, None] * eye[None, None, :3, :]
            + noise[None, :, :, None] * eye[None, None, 3, None, :]
        )

    def observation_value(theta, particles, observation, _time):
        prediction = tf.einsum("od,bnd->bno", matrix, particles)
        residual = observation[None, None, :] - prediction
        scale = theta[:, 4]
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=2) / tf.square(scale)[:, None]
            + 6.0 * tf.math.log(scale)[:, None]
            + 3.0 * tf.cast(_LOG_TWO_PI, theta.dtype)
        )

    def observation_tangent(theta, particles, tangent, observation, _time):
        prediction = tf.einsum("od,bnd->bno", matrix, particles)
        prediction_tangent = tf.einsum("od,bndp->bnop", matrix, tangent)
        residual = observation[None, None, :] - prediction
        scale = theta[:, 4]
        result = tf.reduce_sum(
            residual[:, :, :, None] * prediction_tangent, axis=2
        ) / tf.square(scale)[:, None, None]
        direct = (
            tf.reduce_sum(tf.square(residual), axis=2) / tf.pow(scale, 3)[:, None]
            - 3.0 / scale[:, None]
        )
        return result + direct[:, :, None] * eye[None, None, 4, :]

    return BatchCandidateModelAdapter(
        3,
        5,
        initial_value,
        initial_tangent,
        transition_value,
        transition_tangent,
        observation_value,
        observation_tangent,
    )


def ksc_mixture_sv_batch_adapter() -> BatchCandidateModelAdapter:
    """KSC mixture model in `(z_gamma, log_beta)` filter coordinates."""

    mixture_weights = tf.constant(
        [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750],
        tf.float32,
    )
    means = tf.constant(
        [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819],
        tf.float32,
    ) - tf.constant(1.2704, tf.float32)
    variances = tf.constant(
        [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261],
        tf.float32,
    )
    eye = tf.eye(2, dtype=tf.float32)

    def physical(theta):
        gamma = 0.5 * (
            1.0
            + tf.math.erf(theta[:, 0] / tf.sqrt(tf.constant(2.0, tf.float32)))
        )
        dgamma = tf.cast(_NORMALIZER, theta.dtype) * tf.exp(
            -0.5 * tf.square(theta[:, 0])
        )
        return gamma, dgamma

    def initial_value(theta, noise):
        gamma, _ = physical(theta)
        scale = tf.math.rsqrt(1.0 - tf.square(gamma))
        return noise[None, :, :] * scale[:, None, None]

    def initial_tangent(theta, noise):
        gamma, dgamma = physical(theta)
        first = (
            noise[None, :, 0]
            * gamma[:, None]
            * dgamma[:, None]
            / tf.pow(1.0 - tf.square(gamma), 1.5)[:, None]
        )
        zeros = tf.zeros_like(first)
        return tf.stack([first, zeros], axis=-1)[:, :, None, :]

    def transition_value(theta, particles, noise, _time):
        gamma, _ = physical(theta)
        return gamma[:, None, None] * particles + noise[None, :, :]

    def transition_tangent(theta, particles, noise, tangent, _time):
        del noise
        gamma, dgamma = physical(theta)
        direct = (
            dgamma[:, None] * particles[:, :, 0] + gamma[:, None] * tangent[:, :, 0, 0]
        )
        second = gamma[:, None] * tangent[:, :, 0, 1]
        return tf.stack([direct, second], axis=-1)[:, :, None, :]

    def component_terms(theta, particles, observation):
        residual = (
            observation[0]
            - 2.0 * theta[:, None, None, 1]
            - particles[:, :, 0, None]
            - means[None, None, :]
        )
        terms = (
            tf.math.log(mixture_weights)[None, None, :]
            - 0.5
            * (
                tf.square(residual) / variances[None, None, :]
                + tf.math.log(variances)[None, None, :]
                + tf.cast(_LOG_TWO_PI, theta.dtype)
            )
        )
        return residual, terms

    def observation_value(theta, particles, observation, _time):
        _, terms = component_terms(theta, particles, observation)
        return tf.reduce_logsumexp(terms, axis=2)

    def observation_tangent(theta, particles, tangent, observation, _time):
        residual, terms = component_terms(theta, particles, observation)
        responsibilities = tf.nn.softmax(terms, axis=2)
        location_score = tf.reduce_sum(
            responsibilities * residual / variances[None, None, :], axis=2
        )
        location_tangent = tangent[:, :, 0, :] + 2.0 * eye[None, None, 1, :]
        return location_score[:, :, None] * location_tangent

    return BatchCandidateModelAdapter(
        1,
        2,
        initial_value,
        initial_tangent,
        transition_value,
        transition_tangent,
        observation_value,
        observation_tangent,
    )


def parameterized_austria_sir_batch_adapter() -> BatchCandidateModelAdapter:
    """Three-log-scale Austria SIR with the source half-step RK4 stage."""

    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    base = zhao_cui_sir_austria_model()
    initial_mean = tf.cast(base.initial_mean, tf.float32)
    adjacency = tf.cast(base._adjacency_matrix, tf.float32)  # noqa: SLF001
    degree = tf.reduce_sum(adjacency, axis=1)
    eye = tf.eye(3, dtype=tf.float32)
    step = tf.constant(0.005, tf.float32)

    def physical(theta):
        return (
            0.1 * tf.exp(theta[:, 0]),
            18.0 * tf.exp(theta[:, 1]),
            100.0 * tf.exp(2.0 * theta[:, 2]),
        )

    def rhs(theta, state, tangent):
        kappa, nu, _ = physical(theta)
        susceptible, infectious = state[:, :, 0::2], state[:, :, 1::2]
        d_s, d_i = tangent[:, :, 0::2, :], tangent[:, :, 1::2, :]
        neighbor_s = tf.einsum("bnj,kj->bnk", susceptible, adjacency) - susceptible * degree
        neighbor_i = tf.einsum("bnj,kj->bnk", infectious, adjacency) - infectious * degree
        d_neighbor_s = tf.einsum("bnjp,kj->bnkp", d_s, adjacency) - d_s * degree[None, None, :, None]
        d_neighbor_i = tf.einsum("bnjp,kj->bnkp", d_i, adjacency) - d_i * degree[None, None, :, None]
        infection = kappa[:, None, None] * susceptible * infectious
        d_infection = kappa[:, None, None, None] * (
            infectious[:, :, :, None] * d_s + susceptible[:, :, :, None] * d_i
        ) + infection[:, :, :, None] * eye[None, None, None, 0, :]
        rhs_s = -infection + 0.5 * neighbor_s
        rhs_i = infection - nu[:, None, None] * infectious + 0.5 * neighbor_i
        d_rhs_s = -d_infection + 0.5 * d_neighbor_s
        d_rhs_i = (
            d_infection
            - nu[:, None, None, None] * d_i
            - (nu[:, None, None] * infectious)[:, :, :, None]
            * eye[None, None, None, 1, :]
            + 0.5 * d_neighbor_i
        )
        return (
            tf.reshape(tf.stack([rhs_s, rhs_i], axis=3), [tf.shape(state)[0], tf.shape(state)[1], 18]),
            tf.reshape(tf.stack([d_rhs_s, d_rhs_i], axis=3), [tf.shape(state)[0], tf.shape(state)[1], 18, 3]),
        )

    def rk4(theta, state, tangent):
        current, current_tangent = state, tangent
        for _ in range(4):
            k1, d1 = rhs(theta, current, current_tangent)
            k2, d2 = rhs(theta, current + 0.5 * step * k1, current_tangent + 0.5 * step * d1)
            k3, d3 = rhs(theta, current + 0.5 * step * k2, current_tangent + 0.5 * step * d2)
            k4, d4 = rhs(theta, current + 0.5 * step * k3, current_tangent + 0.5 * step * d3)
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_tangent = current_tangent + step / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        return current, current_tangent

    def initial_value(theta, noise):
        return tf.broadcast_to(initial_mean[None, None, :] + noise[None, :, :], [tf.shape(theta)[0], tf.shape(noise)[0], 18])

    def initial_tangent(theta, noise):
        return tf.zeros([tf.shape(theta)[0], tf.shape(noise)[0], 18, 3], tf.float32)

    def transition_value(theta, particles, noise, _time):
        zeros = tf.zeros([tf.shape(theta)[0], tf.shape(particles)[1], 18, 3], tf.float32)
        return rk4(theta, particles, zeros)[0] + noise[None, :, :]

    def transition_tangent(theta, particles, noise, tangent, _time):
        del noise
        return rk4(theta, particles, tangent)[1]

    def observation_value(theta, particles, observation, _time):
        _, _, variance = physical(theta)
        residual = observation[None, None, :] - particles[:, :, 1::2]
        return -0.5 * tf.reduce_sum(
            tf.cast(_LOG_TWO_PI, theta.dtype)
            + tf.math.log(variance)[:, None, None]
            + tf.square(residual) / variance[:, None, None],
            axis=2,
        )

    def observation_tangent(theta, particles, tangent, observation, _time):
        _, _, variance = physical(theta)
        residual = observation[None, None, :] - particles[:, :, 1::2]
        state_term = tf.reduce_sum(
            residual[:, :, :, None]
            * tangent[:, :, 1::2, :]
            / variance[:, None, None, None],
            axis=2,
        )
        direct = tf.reduce_sum(
            tf.square(residual) / variance[:, None, None] - 1.0, axis=2
        )
        return state_term + direct[:, :, None] * eye[None, None, 2, :]

    return BatchCandidateModelAdapter(
        18,
        3,
        initial_value,
        initial_tangent,
        transition_value,
        transition_tangent,
        observation_value,
        observation_tangent,
    )


def predator_prey_batch_adapter() -> BatchCandidateModelAdapter:
    """Six-parameter additive-Gaussian predator-prey physical model."""

    initial_mean = tf.constant([50.0, 5.0], tf.float32)
    eye = tf.eye(6, dtype=tf.float32)
    step = tf.constant(0.1, tf.float32)

    def rhs(theta, state, tangent):
        r, capacity, half_sat, s_rate, u_rate, v_rate = tf.unstack(theta, axis=1)
        prey, predator = state[:, :, 0], state[:, :, 1]
        d_prey, d_predator = tangent[:, :, 0, :], tangent[:, :, 1, :]
        denominator = half_sat[:, None] + prey
        interaction = prey * predator / denominator
        interaction_tangent = (
            predator[:, :, None] * half_sat[:, None, None] / tf.square(denominator)[:, :, None] * d_prey
            + prey[:, :, None] / denominator[:, :, None] * d_predator
            - (prey * predator / tf.square(denominator))[:, :, None] * eye[None, None, 2, :]
        )
        logistic = prey * (1.0 - prey / capacity[:, None])
        logistic_tangent = (
            (1.0 - 2.0 * prey / capacity[:, None])[:, :, None] * d_prey
            + (tf.square(prey) / tf.square(capacity)[:, None])[:, :, None] * eye[None, None, 1, :]
        )
        value = tf.stack(
            [
                r[:, None] * logistic - s_rate[:, None] * interaction,
                u_rate[:, None] * interaction - v_rate[:, None] * predator,
            ],
            axis=2,
        )
        d_value = tf.stack(
            [
                logistic[:, :, None] * eye[None, None, 0, :]
                + r[:, None, None] * logistic_tangent
                - interaction[:, :, None] * eye[None, None, 3, :]
                - s_rate[:, None, None] * interaction_tangent,
                interaction[:, :, None] * eye[None, None, 4, :]
                + u_rate[:, None, None] * interaction_tangent
                - predator[:, :, None] * eye[None, None, 5, :]
                - v_rate[:, None, None] * d_predator,
            ],
            axis=2,
        )
        return value, d_value

    def rk4(theta, state, tangent):
        current, current_tangent = state, tangent
        for _ in range(20):
            k1, d1 = rhs(theta, current, current_tangent)
            k2, d2 = rhs(theta, current + 0.5 * step * k1, current_tangent + 0.5 * step * d1)
            k3, d3 = rhs(theta, current + 0.5 * step * k2, current_tangent + 0.5 * step * d2)
            k4, d4 = rhs(theta, current + step * k3, current_tangent + step * d3)
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            current_tangent = current_tangent + step / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        return current, current_tangent

    def initial_value(theta, noise):
        return tf.broadcast_to(initial_mean[None, None, :] + noise[None, :, :], [tf.shape(theta)[0], tf.shape(noise)[0], 2])

    def initial_tangent(theta, noise):
        return tf.zeros([tf.shape(theta)[0], tf.shape(noise)[0], 2, 6], tf.float32)

    def transition_value(theta, particles, noise, _time):
        zeros = tf.zeros([tf.shape(theta)[0], tf.shape(particles)[1], 2, 6], tf.float32)
        return rk4(theta, particles, zeros)[0] + 2.0 * noise[None, :, :]

    def transition_tangent(theta, particles, noise, tangent, _time):
        del noise
        return rk4(theta, particles, tangent)[1]

    def observation_value(theta, particles, observation, _time):
        del theta
        residual = observation[None, None, :] - particles
        return -0.5 * (
            tf.reduce_sum(tf.square(residual), axis=2) / 4.0
            + 2.0 * tf.math.log(tf.constant(4.0, tf.float32))
            + 2.0 * tf.cast(_LOG_TWO_PI, particles.dtype)
        )

    def observation_tangent(theta, particles, tangent, observation, _time):
        del theta
        residual = observation[None, None, :] - particles
        return tf.reduce_sum(residual[:, :, :, None] * tangent, axis=2) / 4.0

    return BatchCandidateModelAdapter(
        2,
        6,
        initial_value,
        initial_tangent,
        transition_value,
        transition_tangent,
        observation_value,
        observation_tangent,
    )


__all__ = [
    "diagonal_lgssm_batch_adapter",
    "ksc_mixture_sv_batch_adapter",
    "parameterized_austria_sir_batch_adapter",
    "predator_prey_batch_adapter",
]
