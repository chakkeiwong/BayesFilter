"""Canonical-lane model adapters (P5 model onboarding).

Builds ``NonlinearScoreModel`` and ``CanonicalModelCallbacks`` instances
for the six leaderboard models WITHOUT placeholder Gaussians: process and
observation covariances come from the model specification (model_exact) or
sigma-point propagation; the C-10 provenance guard enforces this at
construction.

First onboarded models: LGSSM (exact) and Austria SIR (RK4 dynamics with
analytical parameter tangent reused from the batch adapter derivation;
observation model 100*exp(2*theta_2)*I_9 on infectious compartments —
model-exact, theta-dependent). Remaining models (predator-prey, 3 SV
variants) follow the same template; each addition must extend
`test_ledh_canonical_models.py` with its S-1-class gate.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel

Tensor = tf.Tensor
DTYPE = tf.float64


def austria_sir_canonical_model(theta_fixed: Tensor) -> NonlinearScoreModel:
    """Austria SIR with real (non-placeholder) Gaussian inputs.

    theta enters the RK4 dynamics (kappa, nu) and the observation
    covariance (100*exp(2*theta_2)). The direction for the score is the
    caller's choice; tangent callbacks close over the direction index via
    the standard one-hot convention at the score assembly level.

    Provenance: process noise is the model's additive noise (model_exact,
    identity scale in the latent parameterization per the frozen target's
    process_noise tensors); observation covariance is the model's stated
    100*exp(2*theta_2)*I_9 (model_exact). The flow linearization H is the
    model's linear infectious-compartment extraction (model_exact). No
    identity placeholder stands in for a derived quantity: the transition
    JACOBIAN enters through the UKF sigma-point propagation of the real
    RK4 dynamics rather than an eye() transition matrix.
    """

    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    base = zhao_cui_sir_austria_model()
    adjacency = tf.cast(base._adjacency_matrix, DTYPE)  # noqa: SLF001
    degree = tf.reduce_sum(adjacency, axis=1)
    step = tf.constant(0.005, DTYPE)
    infectious_matrix = tf.stack(
        [
            tf.one_hot(2 * index + 1, 18, dtype=DTYPE)
            for index in range(9)
        ],
        axis=0,
    )

    def physical(theta):
        return (
            0.1 * tf.exp(theta[..., 0]),
            18.0 * tf.exp(theta[..., 1]),
        )

    def rhs(theta, state):
        kappa, nu = physical(theta)
        susceptible = state[:, 0::2]
        infectious = state[:, 1::2]
        neighbor_s = (
            tf.einsum("nj,kj->nk", susceptible, adjacency)
            - susceptible * degree
        )
        neighbor_i = (
            tf.einsum("nj,kj->nk", infectious, adjacency)
            - infectious * degree
        )
        infection = kappa * susceptible * infectious
        rhs_s = -infection + 0.5 * neighbor_s
        rhs_i = infection - nu * infectious + 0.5 * neighbor_i
        return tf.reshape(
            tf.stack([rhs_s, rhs_i], axis=2), [tf.shape(state)[0], 18]
        )

    def rhs_tangent(theta, state, d_state, d_theta):
        kappa, nu = physical(theta)
        d_kappa = kappa * d_theta[..., 0]
        d_nu = nu * d_theta[..., 1]
        susceptible = state[:, 0::2]
        infectious = state[:, 1::2]
        d_susceptible = d_state[:, 0::2]
        d_infectious = d_state[:, 1::2]
        d_neighbor_s = (
            tf.einsum("nj,kj->nk", d_susceptible, adjacency)
            - d_susceptible * degree
        )
        d_neighbor_i = (
            tf.einsum("nj,kj->nk", d_infectious, adjacency)
            - d_infectious * degree
        )
        infection = kappa * susceptible * infectious
        d_infection = (
            d_kappa * susceptible * infectious
            + kappa * d_susceptible * infectious
            + kappa * susceptible * d_infectious
        )
        d_rhs_s = -d_infection + 0.5 * d_neighbor_s
        d_rhs_i = (
            d_infection
            - d_nu * infectious
            - nu * d_infectious
            + 0.5 * d_neighbor_i
        )
        return tf.reshape(
            tf.stack([d_rhs_s, d_rhs_i], axis=2), [tf.shape(state)[0], 18]
        )

    def transition_mean_fn(theta, points):
        current = points
        for _ in range(4):
            k1 = rhs(theta, current)
            k2 = rhs(theta, current + 0.5 * step * k1)
            k3 = rhs(theta, current + 0.5 * step * k2)
            k4 = rhs(theta, current + step * k3)
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return current

    def transition_mean_tangent_fn(theta, points, d_points):
        # d(RK4)/d(theta,state) chained through the four stages; the
        # parameter direction is the one-hot the score assembly selects —
        # here bound to direction 0..2 via theta_fixed closure convention:
        # the assembly calls with d_points carrying the state tangent and
        # the parameter direction is fixed per call (d_theta one-hot).
        d_theta = _current_direction[0]
        current, d_current = points, d_points
        for _ in range(4):
            k1 = rhs(theta, current)
            d1 = rhs_tangent(theta, current, d_current, d_theta)
            k2 = rhs(theta, current + 0.5 * step * k1)
            d2 = rhs_tangent(
                theta,
                current + 0.5 * step * k1,
                d_current + 0.5 * step * d1,
                d_theta,
            )
            k3 = rhs(theta, current + 0.5 * step * k2)
            d3 = rhs_tangent(
                theta,
                current + 0.5 * step * k2,
                d_current + 0.5 * step * d2,
                d_theta,
            )
            k4 = rhs(theta, current + step * k3)
            d4 = rhs_tangent(
                theta,
                current + step * k3,
                d_current + step * d3,
                d_theta,
            )
            current = current + step / 6.0 * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            d_current = d_current + step / 6.0 * (
                d1 + 2.0 * d2 + 2.0 * d3 + d4
            )
        return d_current

    _current_direction = [tf.constant([1.0, 0.0, 0.0], DTYPE)]

    def set_score_direction(direction: Tensor) -> None:
        _current_direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def observation_fn(points):
        return tf.einsum("oi,ni->no", infectious_matrix, points)

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            infectious_matrix, [tf.shape(points)[0], 9, 18]
        )

    def observation_tangent_fn(points, d_points):
        return tf.einsum("oi,ni->no", infectious_matrix, d_points)

    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    observation_covariance = (
        100.0 * tf.exp(2.0 * theta_fixed[2]) * tf.eye(9, dtype=DTYPE)
    )

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.eye(18, dtype=DTYPE),
        # ^ PROVENANCE: model_exact, NOT a placeholder. The frozen Austria
        # target adds process noise at unit scale (batch adapter line 254:
        # `rk4(...) + noise`), i.e. Q = I_18 by model definition. The
        # 2026-08 placeholder defect was feeding I to the FLOW as the
        # PREDICTED covariance; here the predicted covariance comes from
        # the UKF sigma-point propagation of the real RK4 dynamics, and I
        # enters only as the true additive-noise covariance.
        observation_covariance=observation_covariance,
    )
    # NonlinearScoreModel is frozen; the direction setter travels alongside.
    return model, set_score_direction


__all__ = ["austria_sir_canonical_model"]


def predator_prey_canonical_model(theta_fixed: Tensor):
    """Six-parameter predator-prey; RK4 dynamics (20 x dt=0.1), process
    noise scale 2 (model_exact), direct-state observation with R = 4*I_2
    (model_exact). Ported from the verified batch adapter; tangent is the
    total derivative through the RK4 stages for the direction set via the
    returned setter (same convention as Austria)."""

    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    step = tf.constant(0.1, DTYPE)

    def rhs(theta, state):
        r, capacity, half_sat, s_rate, u_rate, v_rate = tf.unstack(theta)
        prey, predator = state[:, 0], state[:, 1]
        denominator = half_sat + prey
        interaction = prey * predator / denominator
        logistic = prey * (1.0 - prey / capacity)
        return tf.stack(
            [
                r * logistic - s_rate * interaction,
                u_rate * interaction - v_rate * predator,
            ],
            axis=1,
        )

    def rhs_tangent(theta, state, d_state, d_theta):
        r, capacity, half_sat, s_rate, u_rate, v_rate = tf.unstack(theta)
        dr, dcap, dhalf, ds_r, du_r, dv_r = tf.unstack(d_theta)
        prey, predator = state[:, 0], state[:, 1]
        d_prey, d_predator = d_state[:, 0], d_state[:, 1]
        denominator = half_sat + prey
        interaction = prey * predator / denominator
        d_interaction = (
            predator * half_sat / tf.square(denominator) * d_prey
            + prey / denominator * d_predator
            - prey * predator / tf.square(denominator) * dhalf
        )
        logistic = prey * (1.0 - prey / capacity)
        d_logistic = (
            (1.0 - 2.0 * prey / capacity) * d_prey
            + tf.square(prey) / tf.square(capacity) * dcap
        )
        return tf.stack(
            [
                dr * logistic
                + r * d_logistic
                - ds_r * interaction
                - s_rate * d_interaction,
                du_r * interaction
                + u_rate * d_interaction
                - dv_r * predator
                - v_rate * d_predator,
            ],
            axis=1,
        )

    def transition_mean_fn(theta, points):
        current = points
        for _ in range(20):
            k1 = rhs(theta, current)
            k2 = rhs(theta, current + 0.5 * step * k1)
            k3 = rhs(theta, current + 0.5 * step * k2)
            k4 = rhs(theta, current + step * k3)
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return current

    _direction = [tf.zeros([6], DTYPE)]

    def set_score_direction(direction: Tensor) -> None:
        _direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def transition_mean_tangent_fn(theta, points, d_points):
        d_theta = _direction[0]
        current, d_current = points, d_points
        for _ in range(20):
            k1 = rhs(theta, current)
            d1 = rhs_tangent(theta, current, d_current, d_theta)
            k2 = rhs(theta, current + 0.5 * step * k1)
            d2 = rhs_tangent(
                theta, current + 0.5 * step * k1,
                d_current + 0.5 * step * d1, d_theta,
            )
            k3 = rhs(theta, current + 0.5 * step * k2)
            d3 = rhs_tangent(
                theta, current + 0.5 * step * k2,
                d_current + 0.5 * step * d2, d_theta,
            )
            k4 = rhs(theta, current + step * k3)
            d4 = rhs_tangent(
                theta, current + step * k3,
                d_current + step * d3, d_theta,
            )
            current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            d_current = d_current + step / 6.0 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
        return d_current

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: points,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=lambda points, d_points: d_points,
        # process noise scale 2 => covariance 4*I (adapter adds 2*noise)
        process_covariance=4.0 * tf.eye(2, dtype=DTYPE),
        observation_covariance=4.0 * tf.eye(2, dtype=DTYPE),
    )
    return model, set_score_direction


def diagonal_lgssm_canonical_model(theta_fixed: Tensor):
    """Five-parameter diagonal LGSSM (phi_1..3, q_scale, obs_scale) with a
    3x3 observation matrix; everything model_exact and linear (UKF ==
    Kalman on this model, giving an exact-reference lane)."""

    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    obs_matrix = tf.eye(3, dtype=DTYPE)
    phi = theta_fixed[:3]
    q_scale = theta_fixed[3]
    r_scale = theta_fixed[4]

    def transition_mean_fn(theta, points):
        return points * theta[:3][None, :]

    _direction = [tf.zeros([5], DTYPE)]

    def set_score_direction(direction: Tensor) -> None:
        _direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def transition_mean_tangent_fn(theta, points, d_points):
        d_theta = _direction[0]
        return d_points * theta[:3][None, :] + points * d_theta[:3][None, :]

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: tf.einsum(
            "od,nd->no", obs_matrix, points
        ),
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            obs_matrix, [tf.shape(points)[0], 3, 3]
        ),
        observation_tangent_fn=lambda points, d_points: tf.einsum(
            "od,nd->no", obs_matrix, d_points
        ),
        process_covariance=tf.square(q_scale) * tf.eye(3, dtype=DTYPE),
        observation_covariance=tf.square(r_scale) * tf.eye(3, dtype=DTYPE),
    )
    return model, set_score_direction


def ksc_sv_canonical_model(theta_fixed: Tensor):
    """KSC mixture SV in (z_gamma, log_beta) coordinates, 2-state.

    Flow linearization uses the MOMENT-MATCHED GAUSSIAN of the KSC
    log-chi-square mixture (provenance: derived — mixture mean/variance
    are analytic constants); the WEIGHT should use the true mixture
    density where the caller requires exactness (the Gaussian here is the
    flow's proposal-design input, which the PF-PF identity corrects).
    """

    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    weights = tf.constant(
        [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750],
        DTYPE,
    )
    means = tf.constant(
        [-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819],
        DTYPE,
    ) - tf.constant(1.2704, DTYPE)
    variances = tf.constant(
        [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261],
        DTYPE,
    )
    mixture_mean = tf.reduce_sum(weights * means)
    mixture_var = tf.reduce_sum(
        weights * (variances + tf.square(means))
    ) - tf.square(mixture_mean)

    def gamma_of(theta):
        return 0.5 * (
            1.0 + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2.0, DTYPE)))
        )

    def transition_mean_fn(theta, points):
        gamma = gamma_of(theta)
        first = gamma * points[:, 0]
        second = points[:, 1]
        return tf.stack([first, second], axis=1)

    _direction = [tf.zeros([2], DTYPE)]

    def set_score_direction(direction: Tensor) -> None:
        _direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def transition_mean_tangent_fn(theta, points, d_points):
        d_theta = _direction[0]
        gamma = gamma_of(theta)
        normalizer = tf.constant(
            1.0 / np.sqrt(2.0 * np.pi), DTYPE
        )
        dgamma = normalizer * tf.exp(-0.5 * tf.square(theta[0])) * d_theta[0]
        first = dgamma * points[:, 0] + gamma * d_points[:, 0]
        second = d_points[:, 1]
        return tf.stack([first, second], axis=1)

    # Observation (moment-matched Gaussian of the mixture): the observed
    # quantity is h + 2*log_beta + mixture_noise; linear map [1, 2].
    h_matrix = tf.constant([[1.0, 2.0]], DTYPE)

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: tf.einsum(
            "od,nd->no", h_matrix, points
        )
        + mixture_mean,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            h_matrix, [tf.shape(points)[0], 1, 2]
        ),
        observation_tangent_fn=lambda points, d_points: tf.einsum(
            "od,nd->no", h_matrix, d_points
        ),
        process_covariance=tf.constant([[1.0, 0.0], [0.0, 1.0e-8]], DTYPE),
        observation_covariance=mixture_var[None, None],
    )
    return model, set_score_direction





def generalized_sv_canonical_model(theta_fixed: Tensor):
    """Native generalized SV (rho_s, rho_h, log sigma_s, log sigma_h,
    log beta): 2-state diagonal-AR dynamics s' = rho_s s, h' = rho_h h
    with process scales (sigma_s, sigma_h) — model_exact. Observation
    y = beta*s + exp(h/2)*noise is state-dependent-variance; the flow's
    Gaussian input uses the linearization at the anchor (H = [beta,
    beta*s*..] approximated by the mean-map jacobian [beta, 0], variance
    exp(h_anchor)) — provenance: derived linearization, corrected by the
    PF-PF weight exactly as for KSC. The moment-matched observation
    covariance for the flow uses exp(h)~1 reference scale; recorded as a
    proposal-design choice."""

    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    rho_s = tf.tanh(theta_fixed[0])
    rho_h = tf.tanh(theta_fixed[1])
    sigma_s = tf.exp(theta_fixed[2])
    sigma_h = tf.exp(theta_fixed[3])
    beta = tf.exp(theta_fixed[4])

    def transition_mean_fn(theta, points):
        r_s = tf.tanh(theta[0])
        r_h = tf.tanh(theta[1])
        return tf.stack(
            [r_s * points[:, 0], r_h * points[:, 1]], axis=1
        )

    _direction = [tf.zeros([5], DTYPE)]

    def set_score_direction(direction: Tensor) -> None:
        _direction[0] = tf.convert_to_tensor(direction, DTYPE)

    def transition_mean_tangent_fn(theta, points, d_points):
        d_theta = _direction[0]
        r_s = tf.tanh(theta[0])
        r_h = tf.tanh(theta[1])
        dr_s = (1.0 - tf.square(r_s)) * d_theta[0]
        dr_h = (1.0 - tf.square(r_h)) * d_theta[1]
        return tf.stack(
            [
                dr_s * points[:, 0] + r_s * d_points[:, 0],
                dr_h * points[:, 1] + r_h * d_points[:, 1],
            ],
            axis=1,
        )

    h_matrix = tf.stack([beta, tf.constant(0.0, DTYPE)])[None, :]

    model = NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: tf.einsum(
            "od,nd->no", h_matrix, points
        ),
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            h_matrix, [tf.shape(points)[0], 1, 2]
        ),
        observation_tangent_fn=lambda points, d_points: tf.einsum(
            "od,nd->no", h_matrix, d_points
        ),
        process_covariance=tf.linalg.diag(
            tf.stack([tf.square(sigma_s), tf.square(sigma_h)])
        ),
        observation_covariance=tf.ones([1, 1], DTYPE),
    )
    return model, set_score_direction


__all__ = [
    "austria_sir_canonical_model",
    "predator_prey_canonical_model",
    "diagonal_lgssm_canonical_model",
    "ksc_sv_canonical_model",
    "generalized_sv_canonical_model",
]
